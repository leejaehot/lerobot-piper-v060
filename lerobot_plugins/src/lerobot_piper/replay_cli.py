from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
from lerobot_piper.cli_utils import (
    available_profiles,
    check_can_interfaces,
    deep_merge as _deep_merge,
    load_yaml as _load,
    override as _override,
    profile_directory,
    resolve_profile_path,
    section as _section,
    wrapped_lines as _line,
)
from lerobot_piper.console_ui import paint, phase, supports_color
from lerobot_piper.project_paths import PIPER_ROOT

DEFAULT_CONFIG = PIPER_ROOT / "configs/replay.yaml"
PROFILE_DIRECTORY = "replays"
CAN_INIT = PIPER_ROOT / "scripts/can_init.sh"
MOTOR_NAMES = tuple([f"joint{index}.pos" for index in range(1, 7)] + ["gripper.pos"])


@dataclass(frozen=True)
class ReplayData:
    dataset: Any
    actions: np.ndarray
    states: np.ndarray
    action_names: tuple[str, ...]
    state_names: tuple[str, ...]
    image_keys: tuple[str, ...]
    first_frame: int
    last_frame: int

    @property
    def num_frames(self) -> int:
        return self.last_frame - self.first_frame

    @property
    def start_pose(self) -> dict[str, float]:
        return {
            name.removesuffix(".pos"): float(value)
            for name, value in zip(
                self.state_names,
                self.states[self.first_frame],
                strict=True,
            )
        }


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="piper_replay",
        description=(
            "Inspect a recorded Piper episode in Rerun, or explicitly replay its actions "
            "on the follower."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("profile_name", nargs="?", help="profile from configs/replays")
    parser.add_argument("--profile", dest="profile_option", help="profile name or YAML path")
    parser.add_argument("--list-profiles", action="store_true", help="list replay profiles and exit")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML defaults")
    parser.add_argument("--repo-id", help="LeRobot dataset repository ID")
    parser.add_argument("--root", type=Path, help="local LeRobot dataset directory")
    parser.add_argument("--episode", type=int, help="zero-based episode index")
    parser.add_argument("--start-frame", type=int, help="zero-based frame within the episode")
    length = parser.add_mutually_exclusive_group()
    length.add_argument("--frames", type=int, help="number of frames to replay")
    length.add_argument("--seconds", type=float, help="maximum replay duration in dataset seconds")
    parser.add_argument("--rate", type=float, help="playback rate; 1 is recorded real time")
    parser.add_argument(
        "--hold",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="keep the Rerun viewer open for timeline inspection after preview",
    )
    parser.add_argument(
        "--compress-images",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="JPEG-compress frames sent to Rerun",
    )

    parser.add_argument(
        "--hardware",
        action="store_true",
        help="send recorded actions to the follower (otherwise only open Rerun)",
    )
    parser.add_argument("--speed", type=int, help="follower controller speed percent")
    parser.add_argument(
        "--max-relative-target",
        type=float,
        help="maximum normalized joint change accepted per control tick",
    )
    parser.add_argument(
        "--gripper-speed-mm-s",
        type=float,
        help="maximum commanded gripper travel speed",
    )
    parser.add_argument(
        "--align-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="align to the selected episode's first state before hardware replay",
    )
    parser.add_argument("--align-speed", type=int, help="startup alignment speed percent")
    parser.add_argument("--align-timeout", type=float, help="startup alignment timeout")
    parser.add_argument(
        "--return-to-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="return to the selected episode's first state before disconnecting",
    )
    parser.add_argument(
        "--wait-for-support",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="wait for Enter before releasing follower torque",
    )
    parser.add_argument("--sounds", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--init-can", action="store_true", help="initialize Piper CAN before hardware replay")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan only")
    parser.add_argument("--yes", "-y", action="store_true", help="skip hardware confirmation")
    parser.add_argument("--debug", action="store_true", help="show a traceback on failure")
    return parser.parse_args(argv)


def _print_profiles(config_path: Path) -> None:
    profiles = available_profiles(config_path, PROFILE_DIRECTORY)
    if not profiles:
        print(f"No replay profiles found in {profile_directory(config_path, PROFILE_DIRECTORY)}")
        return
    print("Available Piper replay profiles:")
    for path in profiles:
        data = _load(path)
        profile = _section(data, "profile")
        dataset = _section(data, "dataset")
        detail = str(profile.get("description", "")).strip() or str(dataset.get("repo_id", ""))
        print(f"  {path.stem:<20} {detail}")


def _load_configuration(args: argparse.Namespace) -> dict[str, Any]:
    if args.profile_name and args.profile_option:
        raise ValueError("Specify a profile once: positional profile or --profile")
    base = _load(args.config)
    selector = args.profile_option or args.profile_name
    if selector is None:
        selector = _section(base, "replay").get("default_profile")
    if selector is None:
        return base

    profile_path = resolve_profile_path(str(selector), args.config, PROFILE_DIRECTORY)
    if profile_path is None:
        names = ", ".join(
            path.stem for path in available_profiles(args.config, PROFILE_DIRECTORY)
        ) or "none"
        raise ValueError(f"Unknown replay profile {selector!r}; available: {names}")
    merged = _deep_merge(base, _load(profile_path))
    profile = dict(_section(merged, "profile"))
    profile.setdefault("name", profile_path.stem)
    merged["profile"] = profile
    return merged


def _effective(data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    dataset = _section(data, "dataset")
    replay = _section(data, "replay")
    arm = _section(data, "arm")
    audio = _section(data, "audio")
    profile = _section(data, "profile")

    lerobot_root = Path(
        os.path.expandvars(
            os.path.expanduser(
                os.getenv("PIPER_LEROBOT_ROOT", str(dataset.get("lerobot_root", "~/lerobot_v060")))
            )
        )
    ).resolve()
    root_value = _override(args.root, dataset.get("root"))
    if root_value is None:
        raise ValueError("dataset.root or --root is required")
    root = Path(os.path.expandvars(os.path.expanduser(str(root_value))))
    if not root.is_absolute():
        root = lerobot_root / root

    return {
        "profile_name": str(profile.get("name", "")).strip() or None,
        "repo_id": str(_override(args.repo_id, dataset.get("repo_id", ""))).strip(),
        "root": root.resolve(),
        "episode": int(_override(args.episode, replay.get("episode", 0))),
        "start_frame": int(_override(args.start_frame, replay.get("start_frame", 0))),
        "frames": _override(args.frames, replay.get("frames")),
        "seconds": _override(args.seconds, replay.get("seconds")),
        "rate": float(_override(args.rate, replay.get("rate", 1.0))),
        "hold": bool(_override(args.hold, replay.get("hold", True))),
        "compress_images": bool(
            _override(args.compress_images, replay.get("compress_images", True))
        ),
        "hardware": bool(args.hardware),
        "follower_can": os.getenv(
            "PIPER_FOLLOWER_CAN", str(arm.get("follower_can", "can_follower"))
        ),
        "speed_percent": int(_override(args.speed, arm.get("speed_percent", 20))),
        "max_relative_target": float(
            _override(args.max_relative_target, arm.get("max_relative_target", 3))
        ),
        "gripper_speed_mm_s": float(
            _override(args.gripper_speed_mm_s, arm.get("gripper_speed_mm_s", 40))
        ),
        "align_start": bool(_override(args.align_start, arm.get("align_start", True))),
        "align_speed": int(_override(args.align_speed, arm.get("align_speed_percent", 20))),
        "align_timeout": float(
            _override(args.align_timeout, arm.get("align_timeout_seconds", 20))
        ),
        "joint_tolerance": float(arm.get("joint_tolerance_degrees", 2)),
        "gripper_tolerance": float(arm.get("gripper_tolerance_mm", 2)),
        "return_to_start": bool(
            _override(args.return_to_start, arm.get("return_to_start", True))
        ),
        "wait_for_support": bool(
            _override(args.wait_for_support, arm.get("wait_for_support", True))
        ),
        "play_sounds": bool(_override(args.sounds, audio.get("enabled", True))),
    }


def _validate_settings(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if not cfg["repo_id"] or "/" not in cfg["repo_id"]:
        raise ValueError("dataset.repo_id must look like owner/name")
    if not cfg["root"].is_dir():
        raise ValueError(f"Local dataset directory does not exist: {cfg['root']}")
    if not (cfg["root"] / "meta/info.json").is_file():
        raise ValueError(f"Not a LeRobot dataset (missing meta/info.json): {cfg['root']}")
    if cfg["episode"] < 0 or cfg["start_frame"] < 0:
        raise ValueError("episode and start-frame must be non-negative")
    if cfg["frames"] is not None and int(cfg["frames"]) <= 0:
        raise ValueError("frames must be positive")
    if cfg["seconds"] is not None and float(cfg["seconds"]) <= 0:
        raise ValueError("seconds must be positive")
    if cfg["rate"] <= 0:
        raise ValueError("rate must be positive")
    if not 1 <= cfg["speed_percent"] <= 100:
        raise ValueError("speed must be between 1 and 100")
    if cfg["max_relative_target"] <= 0 or cfg["gripper_speed_mm_s"] <= 0:
        raise ValueError("max-relative-target and gripper-speed-mm-s must be positive")
    if not 1 <= cfg["align_speed"] <= 100 or cfg["align_timeout"] <= 0:
        raise ValueError("alignment speed must be 1..100 and timeout must be positive")
    if cfg["joint_tolerance"] <= 0 or cfg["gripper_tolerance"] <= 0:
        raise ValueError("alignment tolerances must be positive")
    if args.init_can and not cfg["hardware"]:
        raise ValueError("--init-can is only valid together with --hardware")
    if args.yes and not cfg["hardware"]:
        raise ValueError("--yes is only valid together with --hardware")


def _load_replay_data(cfg: dict[str, Any]) -> ReplayData:
    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(
        cfg["repo_id"],
        root=cfg["root"],
        episodes=[cfg["episode"]],
    )
    if dataset.num_episodes != 1:
        raise ValueError(f"Episode {cfg['episode']} is not present in the dataset")

    action_feature = dataset.features.get("action")
    state_feature = dataset.features.get("observation.state")
    if not isinstance(action_feature, dict) or not isinstance(state_feature, dict):
        raise ValueError("Dataset must contain action and observation.state")
    action_names = tuple(action_feature.get("names") or ())
    state_names = tuple(state_feature.get("names") or ())
    if action_names != MOTOR_NAMES or state_names != MOTOR_NAMES:
        raise ValueError(
            "Dataset state/action names must be joint1.pos..joint6.pos, gripper.pos; "
            f"got state={state_names}, action={action_names}"
        )

    actions = np.asarray(dataset.select_columns("action")[:]["action"], dtype=np.float64)
    states = np.asarray(
        dataset.select_columns("observation.state")[:]["observation.state"],
        dtype=np.float64,
    )
    expected_shape = (dataset.num_frames, len(MOTOR_NAMES))
    if actions.shape != expected_shape or states.shape != expected_shape:
        raise ValueError(
            f"Dataset state/action arrays must have shape {expected_shape}; "
            f"got state={states.shape}, action={actions.shape}"
        )
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError("Dataset contains non-finite state/action values")
    for values, label in ((states, "state"), (actions, "action")):
        if (values[:, :6] < -100).any() or (values[:, :6] > 100).any():
            raise ValueError(f"Dataset {label} joints fall outside normalized [-100, 100]")
        if (values[:, 6] < 0).any() or (values[:, 6] > 100).any():
            raise ValueError(f"Dataset {label} gripper falls outside normalized [0, 100]")

    first_frame = cfg["start_frame"]
    if first_frame >= dataset.num_frames:
        raise ValueError(
            f"start-frame {first_frame} is outside episode {cfg['episode']} "
            f"({dataset.num_frames} frames)"
        )
    requested_frames = dataset.num_frames - first_frame
    if cfg["frames"] is not None:
        requested_frames = min(requested_frames, int(cfg["frames"]))
    if cfg["seconds"] is not None:
        requested_frames = min(
            requested_frames,
            max(1, int(np.ceil(float(cfg["seconds"]) * dataset.fps))),
        )

    image_keys = tuple(
        key for key, feature in dataset.features.items() if feature.get("dtype") in {"video", "image"}
    )
    return ReplayData(
        dataset=dataset,
        actions=actions,
        states=states,
        action_names=action_names,
        state_names=state_names,
        image_keys=image_keys,
        first_frame=first_frame,
        last_frame=first_frame + requested_frames,
    )


def _plan(cfg: dict[str, Any], data: ReplayData, *, dry_run: bool) -> None:
    mode = "hardware replay" if cfg["hardware"] else "Rerun dataset preview"
    if dry_run:
        mode += " dry run"
    duration = data.num_frames / data.dataset.fps
    rows: list[str] = []
    rows += _line("MODE", mode)
    if cfg["profile_name"]:
        rows += _line("PROFILE", cfg["profile_name"])
    rows += _line("DATASET", cfg["repo_id"])
    rows += _line("ROOT", str(cfg["root"]))
    rows += _line(
        "EPISODE",
        f"{cfg['episode']} · frames {data.first_frame}..{data.last_frame - 1} "
        f"({data.num_frames} frames, {duration:.2f}s)",
    )
    rows += _line("TIMING", f"{data.dataset.fps} Hz · {cfg['rate']:g}× playback")
    rows += _line("CAMERAS", " · ".join(data.image_keys) or "none")
    rows += _line(
        "START POSE",
        "[" + ", ".join(f"{value:.2f}" for value in data.start_pose.values()) + "]",
    )
    if data.num_frames > 1:
        max_steps = np.abs(np.diff(data.actions[data.first_frame : data.last_frame], axis=0)).max(axis=0)
        rows += _line("MAX Δ/DATA", "[" + ", ".join(f"{value:.2f}" for value in max_steps) + "]")
    if cfg["hardware"]:
        rows += _line(
            "CONTROL",
            f"speed {cfg['speed_percent']}% · max Δ {cfg['max_relative_target']:g} · "
            f"gripper {cfg['gripper_speed_mm_s']:g} mm/s",
        )
        rows += _line(
            "SAFETY",
            f"align {'ON' if cfg['align_start'] else 'OFF'} · "
            f"return {'ON' if cfg['return_to_start'] else 'OFF'} · "
            f"support confirmation {'ON' if cfg['wait_for_support'] else 'OFF'}",
        )
    else:
        rows += _line(
            "VIEWER",
            f"JPEG {'ON' if cfg['compress_images'] else 'OFF'} · "
            f"hold after replay {'ON' if cfg['hold'] else 'OFF'} · arms/CAN disconnected",
        )

    width = 76
    color = supports_color(sys.stdout)
    print(paint("╭─ PIPER REPLAY " + "─" * (width - 16) + "╮", "cyan", bold=True, enabled=color))
    for row in rows:
        print(f"│ {row:<{width - 2}} │")
    if cfg["hardware"]:
        print("├" + "─" * width + "┤")
        caution = "CAUTION: recorded action replay enables follower torque; keep E-stop ready."
        for row in textwrap.wrap(caution, width=width - 2):
            print(paint(f"│ {row:<{width - 2}} │", "yellow", bold=True, enabled=color))
    print("╰" + "─" * width + "╯")


def _to_image(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.ndim == 3 and image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = np.transpose(image, (1, 2, 0))
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0 if image.size and image.max() <= 1.0 else image, 0, 255).astype(
            np.uint8
        )
    return image


def _preview(cfg: dict[str, Any], data: ReplayData) -> None:
    import rerun as rr
    import rerun.blueprint as rrb

    from lerobot.utils.keyboard_input import create_key_listener
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.visualization_utils import shutdown_visualization
    from lerobot_piper.grid_preview_cli import _init_owned_rerun

    viewer = None
    listener = None
    started = False
    stop = Event()
    paused = Event()

    def on_key(name: str) -> None:
        key = name.lower()
        if key in {"esc", "q"}:
            stop.set()
        elif key in {"space", "spacebar"}:
            if paused.is_set():
                paused.clear()
                phase("RESUME", "Dataset playback resumed")
            else:
                paused.set()
                phase("PAUSE", "Dataset playback paused; Space resumes", "yellow")

    try:
        viewer = _init_owned_rerun()
        started = True
        image_paths = list(data.image_keys)
        state_paths = [f"observation.state/{name}" for name in data.state_names]
        action_paths = [f"action/{name}" for name in data.action_names]
        views = [rrb.Spatial2DView(origin=path, name=path) for path in image_paths]
        views.extend(
            [
                rrb.TimeSeriesView(name="observation.state", contents=state_paths),
                rrb.TimeSeriesView(name="action", contents=action_paths),
            ]
        )
        rr.send_blueprint(rrb.Blueprint(rrb.Grid(*views)))
        listener = create_key_listener(on_key, controls_help="Space=pause/resume, q/Esc=quit")
        phase("READY", "Playing dataset only; no CAN, cameras, or motors are connected", "green")

        interval = 1.0 / (data.dataset.fps * cfg["rate"])
        deadline = time.perf_counter()
        first_timestamp: float | None = None
        for relative_index, frame_index in enumerate(range(data.first_frame, data.last_frame)):
            while paused.is_set() and not stop.is_set():
                time.sleep(0.05)
                deadline = time.perf_counter()
            if stop.is_set():
                break

            frame = data.dataset[frame_index]
            timestamp = float(frame.get("timestamp", relative_index / data.dataset.fps))
            if first_timestamp is None:
                first_timestamp = timestamp
            rr.set_time("frame", sequence=frame_index)
            rr.set_time("episode_time", duration=timestamp - first_timestamp)
            for key in data.image_keys:
                entity = rr.Image(_to_image(frame[key]))
                if cfg["compress_images"]:
                    entity = entity.compress(jpeg_quality=90)
                rr.log(key, entity)
            for name, value in zip(data.state_names, data.states[frame_index], strict=True):
                rr.log(f"observation.state/{name}", rr.Scalars(float(value)))
            for name, value in zip(data.action_names, data.actions[frame_index], strict=True):
                rr.log(f"action/{name}", rr.Scalars(float(value)))

            deadline += interval
            precise_sleep(max(deadline - time.perf_counter(), 0.0))

        if stop.is_set():
            phase("STOPPED", "Dataset preview stopped", "yellow")
        else:
            phase("COMPLETE", f"Loaded {data.num_frames} frames into Rerun", "green")
            if cfg["hold"]:
                phase("VIEWER", "Inspect the timeline; press q or Esc to close")
                while not stop.wait(0.1):
                    pass
    finally:
        try:
            if listener is not None:
                listener.stop()
        finally:
            try:
                if started:
                    shutdown_visualization("rerun")
            finally:
                if viewer is not None:
                    viewer.close()


def _hardware_replay(cfg: dict[str, Any], data: ReplayData) -> None:
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot_robot_piper import PiperFollower, PiperFollowerConfig

    start_pose = data.start_pose
    robot = PiperFollower(
        PiperFollowerConfig(
            id="piper_follower",
            port=cfg["follower_can"],
            cameras={},
            speed_percent=cfg["speed_percent"],
            max_relative_target=cfg["max_relative_target"],
            gripper_speed_mm_s=cfg["gripper_speed_mm_s"],
            terminal_update_hz=0,
            play_sounds=cfg["play_sounds"],
            disable_torque_on_disconnect=True,
            wait_for_enter_on_disconnect=cfg["wait_for_support"],
            startup_pose=start_pose if cfg["align_start"] else None,
            startup_pose_speed_percent=cfg["align_speed"],
            startup_pose_timeout_s=cfg["align_timeout"],
            startup_pose_joint_tolerance_degrees=cfg["joint_tolerance"],
            startup_pose_gripper_tolerance_mm=cfg["gripper_tolerance"],
        )
    )
    connected = False
    sent_frames = 0
    limited_frames = 0
    max_difference = np.zeros(len(data.action_names), dtype=np.float64)
    try:
        phase("CONNECT", "Enabling follower torque and aligning to the selected dataset state")
        robot.connect()
        connected = True
        phase("REPLAY", f"Sending recorded actions at {data.dataset.fps * cfg['rate']:g} Hz", "green")
        interval = 1.0 / (data.dataset.fps * cfg["rate"])
        deadline = time.perf_counter()
        for frame_index in range(data.first_frame, data.last_frame):
            requested = {
                name: float(value)
                for name, value in zip(data.action_names, data.actions[frame_index], strict=True)
            }
            sent = robot.send_action(requested)
            difference = np.asarray(
                [abs(float(sent[name]) - requested[name]) for name in data.action_names]
            )
            max_difference = np.maximum(max_difference, difference)
            limited_frames += int(np.any(difference > 1e-6))
            sent_frames += 1
            if sent_frames % data.dataset.fps == 0:
                phase("PROGRESS", f"{sent_frames}/{data.num_frames} frames")
            deadline += interval
            precise_sleep(max(deadline - time.perf_counter(), 0.0))
        phase("COMPLETE", f"Sent {sent_frames} recorded actions", "green")
        if limited_frames:
            phase(
                "LIMIT",
                f"Safety limiting changed {limited_frames}/{sent_frames} frames; max |sent-requested| "
                + "["
                + ", ".join(f"{value:.2f}" for value in max_difference)
                + "]",
                "yellow",
            )
    finally:
        if connected:
            if cfg["return_to_start"]:
                try:
                    phase("RETURN", "Returning to the selected dataset start state")
                    robot.bus.move_follower_to_normalized_pose(
                        start_pose,
                        speed_percent=cfg["align_speed"],
                        timeout_s=cfg["align_timeout"],
                        joint_tolerance_degrees=cfg["joint_tolerance"],
                        gripper_tolerance_mm=cfg["gripper_tolerance"],
                    )
                except Exception as exc:
                    phase("WARNING", f"Could not return to the dataset start state: {exc}", "yellow")
            try:
                robot.prepare_for_disconnect()
            finally:
                robot.disconnect()


def _failure_hint(message: str) -> str | None:
    lower = message.lower()
    if "missing can" in lower or "unhealthy can" in lower:
        return "Stop other Piper processes, then retry with --hardware --init-can."
    if "video" in lower or "decode" in lower or "av1" in lower:
        return "Verify ffmpeg/PyAV AV1 support and that all dataset video files were copied."
    if "failed to enable follower" in lower:
        return "Release the E-stop, power-cycle the follower controller, then retry."
    return None


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    try:
        if args.list_profiles:
            _print_profiles(args.config)
            return
        cfg = _effective(_load_configuration(args), args)
        _validate_settings(cfg, args)
        phase("DATASET", f"Loading episode {cfg['episode']} metadata and actions")
        data = _load_replay_data(cfg)
        _plan(cfg, data, dry_run=args.dry_run)

        if args.dry_run:
            phase("DRY RUN", "Dataset and selected frame range are valid; nothing was activated", "green")
            return
        if not cfg["hardware"]:
            _preview(cfg, data)
            return

        if args.init_can:
            phase("CAN", "Initializing USB-CAN adapters")
            subprocess.run([str(CAN_INIT)], check=True)
            phase("CAN", "CAN initialization completed", "green")
        phase("CHECK", f"Checking follower CAN interface {cfg['follower_can']}")
        check_can_interfaces((cfg["follower_can"],))
        if not args.yes and not sys.stdin.isatty():
            raise RuntimeError("Non-interactive hardware replay requires explicit --yes")
        if not args.yes:
            answer = input(
                "Clear the workspace, hold the E-stop, then press Enter to align and replay; "
                "type q to cancel: "
            ).strip().lower()
            if answer in {"q", "quit", "n", "no"}:
                phase("CANCELLED", "No hardware was activated", "yellow")
                return
        _hardware_replay(cfg, data)
    except KeyboardInterrupt:
        phase("STOPPED", "Interrupted by user; safe teardown was requested", "yellow", stream=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        phase("ERROR", str(exc), "red", stream=sys.stderr)
        if hint := _failure_hint(str(exc)):
            phase("HINT", hint, "yellow", stream=sys.stderr)
        if args.debug:
            traceback.print_exc()
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()

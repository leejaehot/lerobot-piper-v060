from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from lerobot_piper.cli_utils import (
    check_can_interfaces,
    load_yaml as _load,
    override,
    section as _section,
    wrapped_lines,
)
from lerobot_piper.console_ui import paint, phase, recording_log_style, supports_color
from lerobot_piper.project_paths import PIPER_ROOT

DEFAULT_CONFIG = PIPER_ROOT / "configs/record.yaml"
CAN_INIT = PIPER_ROOT / "scripts/can_init.sh"
DEFAULT_LEROBOT_HOME = Path.home() / ".cache/huggingface/lerobot-v060-piper"


class _ResetOnlyGuide:
    """Force LeRobot's pre-episode reset phase without adding annotations."""

    @property
    def dataset_features(self) -> dict[str, dict[str, Any]]:
        return {}

    def annotations_for_episode(self, episode_index: int) -> dict[str, Any]:
        del episode_index
        return {}

    def on_phase(self, episode_index: int, phase: str) -> None:
        del episode_index, phase

    def log_visualization(self) -> None:
        return


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="piper_record",
        description="Record a Piper dataset with stable CAN and RealSense identities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML defaults")
    parser.add_argument("--init-can", action="store_true", help="initialize and rename CAN adapters first")
    parser.add_argument("--test", action="store_true", help="record one 5-second local smoke-test episode")
    parser.add_argument("--repo-id", help="dataset ID, for example local/pick_cup")
    parser.add_argument("--task", help="task instruction stored with every frame")
    parser.add_argument("--episodes", type=int, help="number of episodes")
    parser.add_argument("--seconds", type=float, help="recording seconds per episode")
    parser.add_argument("--reset-seconds", type=float, help="reset time between episodes")
    parser.add_argument(
        "--wait-for-enter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="wait for Enter after the initial object setup before episode 1",
    )
    parser.add_argument("--dataset-fps", type=int, help="camera and dataset FPS")
    parser.add_argument("--control-fps", type=int, help="leader-to-follower control FPS")
    parser.add_argument("--speed", type=int, help="follower speed percent")
    parser.add_argument(
        "--home-on-reset",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="return only the follower to Piper's zero/home between episodes",
    )
    parser.add_argument(
        "--home-speed",
        type=int,
        help="follower speed percent during the between-episode home move",
    )
    parser.add_argument(
        "--gripper-speed-mm-s",
        type=float,
        help="maximum follower gripper travel speed",
    )
    parser.add_argument(
        "--leader-gripper-friction",
        type=int,
        help="leader gripper teaching friction (1..10)",
    )
    parser.add_argument(
        "--segments",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="store dense Space-pedal sub-task IDs in annotation.segment_id",
    )
    parser.add_argument(
        "--segment-debounce-ms",
        type=int,
        help="minimum interval between accepted segment-boundary presses",
    )
    parser.add_argument(
        "--grid",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="show configured object initial poses on an egoview grid",
    )
    parser.add_argument("--rerun", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--sounds",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="speak recording phase changes",
    )
    parser.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--yes", "-y", action="store_true", help="start without the confirmation prompt")
    parser.add_argument("--dry-run", action="store_true", help="show the effective recording plan only")
    parser.add_argument("--debug", action="store_true", help="show a traceback when startup fails")
    return parser.parse_args()


def _effective(data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    dataset = _section(data, "dataset")
    capture = _section(data, "capture")
    arm = _section(data, "arm")
    video = _section(data, "video")
    cameras = _section(data, "cameras")
    annotations = _section(data, "annotations")
    reset_grid = _section(data, "reset_grid")
    initial_setup = _section(data, "initial_setup")
    audio = _section(data, "audio")

    cfg = {
        "repo_id": args.repo_id or dataset.get("repo_id", "local/piper_doubleport"),
        "task": args.task or dataset.get("task", ""),
        "episodes": override(args.episodes, dataset.get("episodes", 10)),
        "episode_seconds": override(args.seconds, dataset.get("episode_seconds", 90)),
        "reset_seconds": override(args.reset_seconds, dataset.get("reset_seconds", 60)),
        "wait_for_enter": (
            initial_setup.get("wait_for_enter", True)
            if getattr(args, "wait_for_enter", None) is None
            else args.wait_for_enter
        ),
        "push_to_hub": dataset.get("push_to_hub", False)
        if args.push_to_hub is None
        else args.push_to_hub,
        "cameras": cameras,
        "dataset_fps": override(args.dataset_fps, capture.get("dataset_fps", 30)),
        "control_fps": override(args.control_fps, capture.get("control_fps", 200)),
        "width": capture.get("width", 640),
        "height": capture.get("height", 480),
        "rerun": capture.get("rerun", True) if args.rerun is None else args.rerun,
        # Local Jetson viewers can stall while decoding rapidly replaced JPEG
        # EncodedImage entities. Raw images avoid that decoder path; compression
        # remains available as an explicit opt-in for remote visualization.
        "rerun_compress_images": capture.get("rerun_compress_images", False),
        "rerun_fps": capture.get("rerun_fps", 10),
        "play_sounds": (
            audio.get("enabled", True)
            if getattr(args, "sounds", None) is None
            else args.sounds
        ),
        "leader_can": os.getenv("PIPER_LEADER_CAN", arm.get("leader_can", "can_leader")),
        "follower_can": os.getenv("PIPER_FOLLOWER_CAN", arm.get("follower_can", "can_follower")),
        "speed_percent": override(args.speed, arm.get("speed_percent", 100)),
        "home_on_reset": (
            arm.get("home_on_reset", True)
            if getattr(args, "home_on_reset", None) is None
            else args.home_on_reset
        ),
        "home_speed_percent": override(
            getattr(args, "home_speed", None),
            arm.get("home_speed_percent", 20),
        ),
        "home_tolerance_degrees": arm.get("home_tolerance_degrees", 2.0),
        "max_relative_target": arm.get("max_relative_target", 100),
        "gripper_speed_mm_s": override(
            args.gripper_speed_mm_s,
            arm.get("gripper_speed_mm_s", 80),
        ),
        "leader_gripper_friction": override(
            args.leader_gripper_friction,
            arm.get("leader_gripper_friction", 5),
        ),
        "gripper_input_min": arm.get("gripper_input_min", 1_000),
        "gripper_input_max": arm.get("gripper_input_max", 50_000),
        "segments": annotations.get("segments", True)
        if args.segments is None
        else args.segments,
        "segment_debounce_ms": override(
            args.segment_debounce_ms,
            annotations.get("segment_debounce_ms", 400),
        ),
        "reset_grid": {
            "enabled": (
                reset_grid.get("enabled", False)
                if getattr(args, "grid", None) is None
                else args.grid
            ),
            "camera": reset_grid.get("camera", "egoview"),
            "columns": reset_grid.get("columns", 16),
            "rows": reset_grid.get("rows", 12),
            "corners": reset_grid.get(
                "corners",
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            ),
            "initial_poses": reset_grid.get("initial_poses", {}),
        },
        "video": video,
    }
    if args.test:
        cfg.update(
            repo_id="local/piper_smoke",
            episodes=1,
            episode_seconds=5,
            reset_seconds=1,
            push_to_hub=False,
        )
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    if "/" not in str(cfg["repo_id"]):
        raise ValueError("repo_id must look like 'namespace/dataset_name'")
    if not str(cfg["task"]).strip():
        raise ValueError("task must not be empty")
    if int(cfg["episodes"]) < 1 or float(cfg["episode_seconds"]) <= 0:
        raise ValueError("episodes and episode_seconds must be positive")
    if not isinstance(cfg["wait_for_enter"], bool):
        raise ValueError("initial_setup wait_for_enter must be true or false")
    if int(cfg["dataset_fps"]) < 1 or int(cfg["control_fps"]) < int(cfg["dataset_fps"]):
        raise ValueError("control_fps must be at least dataset_fps")
    if not 0 < float(cfg["rerun_fps"]) <= int(cfg["dataset_fps"]):
        raise ValueError("rerun_fps must be positive and no greater than dataset_fps")
    if not 1 <= int(cfg["speed_percent"]) <= 100:
        raise ValueError("speed_percent must be between 1 and 100")
    if not isinstance(cfg["home_on_reset"], bool):
        raise ValueError("home_on_reset must be true or false")
    if not 1 <= int(cfg["home_speed_percent"]) <= 100:
        raise ValueError("home_speed_percent must be between 1 and 100")
    if float(cfg["home_tolerance_degrees"]) <= 0:
        raise ValueError("home_tolerance_degrees must be positive")
    if (
        cfg["home_on_reset"]
        and int(cfg["episodes"]) > 1
        and float(cfg["reset_seconds"]) <= 3
    ):
        raise ValueError(
            "reset_seconds must be greater than 3 when home_on_reset is enabled"
        )
    if float(cfg["gripper_speed_mm_s"]) <= 0:
        raise ValueError("gripper_speed_mm_s must be positive")
    if not 1 <= int(cfg["leader_gripper_friction"]) <= 10:
        raise ValueError("leader_gripper_friction must be between 1 and 10")
    if int(cfg["segment_debounce_ms"]) < 0:
        raise ValueError("segment_debounce_ms must be non-negative")
    cameras = cfg["cameras"]
    if not cameras or any(not str(serial).isdigit() for serial in cameras.values()):
        raise ValueError("cameras must map readable names to numeric RealSense SDK serials")
    if len(set(map(str, cameras.values()))) != len(cameras):
        raise ValueError("camera serials must be unique")
    reset_grid = cfg["reset_grid"]
    if reset_grid["enabled"] and reset_grid["camera"] not in cameras:
        raise ValueError(
            f"reset_grid camera {reset_grid['camera']!r} is not present in cameras"
        )
    if not isinstance(reset_grid["initial_poses"], dict):
        raise ValueError("reset_grid initial_poses must be a YAML mapping")
    if not isinstance(cfg["play_sounds"], bool):
        raise ValueError("audio enabled must be true or false")


def _usb_realsense_count() -> int:
    count = 0
    for product in Path("/sys/bus/usb/devices").glob("*/product"):
        try:
            count += "RealSense" in product.read_text(errors="replace")
        except OSError:
            pass
    return count


def _preflight(cfg: dict[str, Any]) -> None:
    check_can_interfaces((cfg["leader_can"], cfg["follower_can"]))
    detected = _usb_realsense_count()
    expected = len(cfg["cameras"])
    if detected < expected:
        raise RuntimeError(f"Only {detected} RealSense USB device(s) detected; config expects {expected}")


def _line(label: str, value: str, width: int = 68) -> list[str]:
    return wrapped_lines(label, value, width=width, label_width=10)


def _plan(cfg: dict[str, Any], *, test: bool, reset_grid_guide=None) -> None:
    camera_text = "  ·  ".join(f"{name} ({serial})" for name, serial in cfg["cameras"].items())
    output_name = f"{cfg['repo_id']}_{datetime.now():%Y%m%d_%H%M%S}"
    output_hint = f"$HF_LEROBOT_HOME/{output_name}"
    rows: list[str] = []
    rows += _line("MODE", "5-second smoke test" if test else "dataset recording")
    rows += _line("TASK", str(cfg["task"]))
    rows += _line(
        "EPISODES",
        f"{cfg['episodes']} × {cfg['episode_seconds']}s  ·  reset {cfg['reset_seconds']}s",
    )
    rows += _line(
        "FIRST START",
        "object setup → Enter → 3-second countdown"
        if cfg["wait_for_enter"]
        else "timed reset (Enter confirmation OFF)",
    )
    rows += _line("CAMERAS", camera_text)
    rows += _line(
        "CONTROL",
        f"{cfg['control_fps']} Hz  ·  dataset {cfg['dataset_fps']} Hz  ·  speed {cfg['speed_percent']}%",
    )
    rows += _line(
        "GRIPPER",
        (
            f"follower {cfg['gripper_speed_mm_s']:g} mm/s"
            f"  ·  leader friction {cfg['leader_gripper_friction']}/10"
        ),
    )
    rows += _line(
        "HOME RESET",
        (
            f"Follower Piper zero · {cfg['home_speed_percent']}% · leader unchanged"
            f" · tolerance {cfg['home_tolerance_degrees']:g}°"
            if cfg["home_on_reset"]
            else "OFF"
        ),
    )
    rows += _line(
        "PEDALS",
        "← re-record  ·  Space next segment  ·  → next episode"
        if cfg["segments"]
        else "← re-record  ·  → next episode  ·  segments OFF",
    )
    reset_grid = cfg["reset_grid"]
    if reset_grid["enabled"]:
        rows += _line(
            "RESET GRID",
            (
                f"{reset_grid['camera']} · {reset_grid['columns']}×{reset_grid['rows']} points"
                f" · {reset_grid_guide.num_positions} fixed object pose(s)"
            ),
        )
        rows += _line(
            "INITIAL",
            (
                " · ".join(pose.label for pose in reset_grid_guide.initial_poses)
                or "no object poses configured"
            ),
        )
    rows += _line("OUTPUT", str(output_hint))
    rows += _line(
        "OPTIONS",
        (
            f"Rerun {'ON' if cfg['rerun'] else 'OFF'}"
            + (
                f" ({cfg['rerun_fps']} Hz "
                f"{'JPEG' if cfg['rerun_compress_images'] else 'raw'} preview)"
                if cfg["rerun"]
                else ""
            )
            + f"  ·  Hub upload {'ON' if cfg['push_to_hub'] else 'OFF'}"
            + f"  ·  Voice {'ON' if cfg['play_sounds'] else 'OFF'}"
        ),
    )

    width = 72
    color = supports_color(sys.stdout)
    print(
        paint(
            "╭─ PIPER RECORD " + "─" * (width - 16) + "╮",
            "cyan",
            bold=True,
            enabled=color,
        )
    )
    for row in rows:
        print(f"│ {row:<{width - 2}} │")
    print("├" + "─" * width + "┤")
    caution = f"│ {'CAUTION: follower torque enables immediately; clear both workspaces.':<{width - 2}} │"
    print(paint(caution, "yellow", bold=True, enabled=color))
    print("╰" + "─" * width + "╯")


def _record_config(cfg: dict[str, Any]):
    from lerobot.cameras.realsense import RealSenseCameraConfig
    from lerobot.configs.dataset import DatasetRecordConfig
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.scripts.lerobot_record import RecordConfig
    from lerobot_robot_piper import PiperFollowerConfig
    from lerobot_teleoperator_piper import PiperLeaderConfig

    cameras = {
        name: RealSenseCameraConfig(
            serial_number_or_name=str(serial),
            fps=int(cfg["dataset_fps"]),
            width=int(cfg["width"]),
            height=int(cfg["height"]),
            use_rgb=True,
            use_depth=False,
        )
        for name, serial in cfg["cameras"].items()
    }
    video = cfg["video"]
    rgb_encoder = RGBEncoderConfig(
        vcodec=str(video.get("codec", "h264")),
        preset=video.get("preset", "veryfast"),
        crf=video.get("crf", 28),
        g=video.get("keyframe_interval", int(cfg["dataset_fps"])),
    )
    return RecordConfig(
        robot=PiperFollowerConfig(
            id="piper_follower",
            port=str(cfg["follower_can"]),
            cameras=cameras,
            speed_percent=int(cfg["speed_percent"]),
            max_relative_target=float(cfg["max_relative_target"]),
            gripper_speed_mm_s=float(cfg["gripper_speed_mm_s"]),
            terminal_update_hz=0,
            play_sounds=bool(cfg["play_sounds"]),
        ),
        teleop=PiperLeaderConfig(
            id="piper_leader",
            port=str(cfg["leader_can"]),
            gripper_input_min=int(cfg["gripper_input_min"]),
            gripper_input_max=int(cfg["gripper_input_max"]),
            gripper_teaching_friction=int(cfg["leader_gripper_friction"]),
        ),
        dataset=DatasetRecordConfig(
            repo_id=str(cfg["repo_id"]),
            single_task=str(cfg["task"]),
            fps=int(cfg["dataset_fps"]),
            episode_time_s=float(cfg["episode_seconds"]),
            reset_time_s=float(cfg["reset_seconds"]),
            num_episodes=int(cfg["episodes"]),
            push_to_hub=bool(cfg["push_to_hub"]),
            streaming_encoding=bool(video.get("streaming", False)),
            encoder_threads=video.get("encoder_threads", 2),
            rgb_encoder=rgb_encoder,
        ),
        control_fps=int(cfg["control_fps"]),
        display_data=bool(cfg["rerun"]),
        display_mode="rerun",
        display_fps=float(cfg["rerun_fps"]),
        display_compressed_images=bool(cfg["rerun_compress_images"]),
        segment_annotation=bool(cfg["segments"]),
        segment_debounce_s=float(cfg["segment_debounce_ms"]) / 1_000,
        play_sounds=bool(cfg["play_sounds"]),
    )


def _reset_grid_guide(cfg: dict[str, Any]):
    reset_grid = cfg["reset_grid"]
    if not reset_grid["enabled"]:
        return _ResetOnlyGuide() if cfg["wait_for_enter"] else None

    from lerobot_piper.reset_grid import ResetGridGuide

    return ResetGridGuide(
        camera_name=str(reset_grid["camera"]),
        image_width=int(cfg["width"]),
        image_height=int(cfg["height"]),
        columns=int(reset_grid["columns"]),
        rows=int(reset_grid["rows"]),
        corners=reset_grid["corners"],
        initial_poses=reset_grid["initial_poses"],
        rerun_enabled=bool(cfg["rerun"]),
    )


def _failure_hint(message: str) -> str | None:
    lower = message.lower()
    if "failed to set power state" in lower:
        return "Close RealSense Viewer/other camera processes, reconnect the camera hub, then retry directly."
    if "failed to enable follower" in lower:
        return "Release the E-stop, power-cycle the follower controller, then retry."
    if "unhealthy can" in lower:
        return "Reinitialize CAN; if it is still not ERROR-ACTIVE, power-cycle the affected arm controller."
    if "already exists" in lower:
        return "Use a different --repo-id; each recording receives a timestamp but same-second retries can collide."
    return None


def main() -> None:
    args = _arguments()
    try:
        os.environ.setdefault("HF_LEROBOT_HOME", str(DEFAULT_LEROBOT_HOME))
        cfg = _effective(_load(args.config), args)
        _validate(cfg)
        record_cfg = _record_config(cfg)
        reset_grid_guide = _reset_grid_guide(cfg)
        if args.init_can:
            phase("CAN", "Initializing USB-CAN adapters")
            subprocess.run([str(CAN_INIT), *( ["--dry-run"] if args.dry_run else [] )], check=True)
            phase("CAN", "Leader and follower interfaces are ready", "green")
        _plan(cfg, test=args.test, reset_grid_guide=reset_grid_guide)
        if args.dry_run:
            phase("DRY RUN", "No hardware was activated", "green")
            return
        phase("CHECK", "Checking CAN interfaces and RealSense connections")
        _preflight(cfg)
        phase("READY", "Preflight passed; follower torque will enable on start", "green")
        if not args.yes and sys.stdin.isatty():
            answer = input("Press Enter to start, or type q to cancel: ").strip().lower()
            if answer in {"q", "quit", "n", "no"}:
                phase("CANCELLED", "No hardware was activated", "yellow")
                return

        from lerobot.scripts import lerobot_record
        from lerobot_piper.audio import local_record_audio

        phase("CONNECT", "Starting cameras, follower, leader, and Rerun")
        with local_record_audio(
            lerobot_record,
            enabled=bool(cfg["play_sounds"]),
            home_on_reset=bool(cfg["home_on_reset"]),
            home_speed_percent=int(cfg["home_speed_percent"]),
            home_tolerance_degrees=float(cfg["home_tolerance_degrees"]),
            wait_for_enter=bool(cfg["wait_for_enter"]),
        ):
            with recording_log_style(int(cfg["episodes"])):
                dataset = lerobot_record.record(
                    record_cfg,
                    episode_annotation_provider=reset_grid_guide,
                )
        phase("COMPLETE", "Dataset recording finished", "green")
        print(f"  Dataset:  {dataset.root}")
        print(f"  Episodes: {dataset.num_episodes}")
        print(f"  Frames:   {dataset.num_frames}")
    except KeyboardInterrupt:
        phase("STOPPED", "Interrupted by user; finalization was requested", "yellow", stream=sys.stderr)
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

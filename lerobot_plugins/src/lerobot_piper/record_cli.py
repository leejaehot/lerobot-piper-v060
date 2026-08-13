from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from lerobot_piper.console_ui import paint, phase, recording_log_style, supports_color

PIPER_ROOT = Path(os.getenv("PIPER_ROOT", Path(__file__).resolve().parents[3]))
DEFAULT_CONFIG = PIPER_ROOT / "configs/record.yaml"
CAN_INIT = PIPER_ROOT / "scripts/can_init"
DEFAULT_LEROBOT_HOME = Path.home() / ".cache/huggingface/lerobot-v060-piper"


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
    parser.add_argument("--dataset-fps", type=int, help="camera and dataset FPS")
    parser.add_argument("--control-fps", type=int, help="leader-to-follower control FPS")
    parser.add_argument("--speed", type=int, help="follower speed percent")
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
    parser.add_argument("--rerun", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--yes", "-y", action="store_true", help="start without the confirmation prompt")
    parser.add_argument("--dry-run", action="store_true", help="show the effective recording plan only")
    parser.add_argument("--debug", action="store_true", help="show a traceback when startup fails")
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"Config file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return data


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"'{name}' must be a YAML mapping")
    return value


def _effective(data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    dataset = _section(data, "dataset")
    capture = _section(data, "capture")
    arm = _section(data, "arm")
    video = _section(data, "video")
    cameras = _section(data, "cameras")
    annotations = _section(data, "annotations")

    def override(value: Any, fallback: Any) -> Any:
        return fallback if value is None else value

    cfg = {
        "repo_id": args.repo_id or dataset.get("repo_id", "local/piper_doubleport"),
        "task": args.task or dataset.get("task", ""),
        "episodes": override(args.episodes, dataset.get("episodes", 10)),
        "episode_seconds": override(args.seconds, dataset.get("episode_seconds", 90)),
        "reset_seconds": override(args.reset_seconds, dataset.get("reset_seconds", 60)),
        "push_to_hub": dataset.get("push_to_hub", False)
        if args.push_to_hub is None
        else args.push_to_hub,
        "cameras": cameras,
        "dataset_fps": override(args.dataset_fps, capture.get("dataset_fps", 30)),
        "control_fps": override(args.control_fps, capture.get("control_fps", 200)),
        "width": capture.get("width", 640),
        "height": capture.get("height", 480),
        "rerun": capture.get("rerun", True) if args.rerun is None else args.rerun,
        "rerun_compress_images": capture.get("rerun_compress_images", True),
        "rerun_fps": capture.get("rerun_fps", 10),
        "leader_can": os.getenv("PIPER_LEADER_CAN", arm.get("leader_can", "can_leader")),
        "follower_can": os.getenv("PIPER_FOLLOWER_CAN", arm.get("follower_can", "can_follower")),
        "speed_percent": override(args.speed, arm.get("speed_percent", 100)),
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
    if int(cfg["dataset_fps"]) < 1 or int(cfg["control_fps"]) < int(cfg["dataset_fps"]):
        raise ValueError("control_fps must be at least dataset_fps")
    if not 1 <= int(cfg["speed_percent"]) <= 100:
        raise ValueError("speed_percent must be between 1 and 100")
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


def _usb_realsense_count() -> int:
    count = 0
    for product in Path("/sys/bus/usb/devices").glob("*/product"):
        try:
            count += "RealSense" in product.read_text(errors="replace")
        except OSError:
            pass
    return count


def _preflight(cfg: dict[str, Any]) -> None:
    missing_can = [
        name
        for name in (cfg["leader_can"], cfg["follower_can"])
        if not Path(f"/sys/class/net/{name}").exists()
    ]
    if missing_can:
        raise RuntimeError(f"Missing CAN interface(s): {', '.join(missing_can)}; run with --init-can")
    unhealthy_can = []
    for name in (cfg["leader_can"], cfg["follower_can"]):
        status = subprocess.run(
            ["ip", "-details", "link", "show", str(name)],
            check=False,
            capture_output=True,
            text=True,
        )
        if status.returncode != 0 or "can state ERROR-ACTIVE" not in status.stdout:
            unhealthy_can.append(str(name))
    if unhealthy_can:
        raise RuntimeError(
            f"Unhealthy CAN interface(s): {', '.join(unhealthy_can)}; stop other Piper processes "
            "and retry with --init-can"
        )
    detected = _usb_realsense_count()
    expected = len(cfg["cameras"])
    if detected < expected:
        raise RuntimeError(f"Only {detected} RealSense USB device(s) detected; config expects {expected}")


def _line(label: str, value: str, width: int = 68) -> list[str]:
    prefix = f"{label:<10} "
    parts = textwrap.wrap(value, width=width - len(prefix)) or [""]
    return [
        prefix + part if index == 0 else " " * len(prefix) + part
        for index, part in enumerate(parts)
    ]


def _plan(cfg: dict[str, Any], *, test: bool) -> None:
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
        "PEDALS",
        "← re-record  ·  Space next segment  ·  → next episode"
        if cfg["segments"]
        else "← re-record  ·  → next episode  ·  segments OFF",
    )
    rows += _line("OUTPUT", str(output_hint))
    rows += _line(
        "OPTIONS",
        (
            f"Rerun {'ON' if cfg['rerun'] else 'OFF'}"
            + (
                f" ({cfg['rerun_fps']} Hz JPEG preview)"
                if cfg["rerun"] and cfg["rerun_compress_images"]
                else ""
            )
            + f"  ·  Hub upload {'ON' if cfg['push_to_hub'] else 'OFF'}"
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
        play_sounds=False,
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
        if args.init_can:
            phase("CAN", "Initializing USB-CAN adapters")
            subprocess.run([str(CAN_INIT), *( ["--dry-run"] if args.dry_run else [] )], check=True)
            phase("CAN", "Leader and follower interfaces are ready", "green")
        _plan(cfg, test=args.test)
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

        from lerobot.scripts.lerobot_record import record

        phase("CONNECT", "Starting cameras, follower, leader, and Rerun")
        with recording_log_style(int(cfg["episodes"])):
            dataset = record(record_cfg)
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

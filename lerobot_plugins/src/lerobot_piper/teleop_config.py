from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lerobot_piper.reset_grid import load_reset_grid_settings


@dataclass(frozen=True)
class TeleopSettings:
    path: Path
    follower_can: str
    leader_can: str
    control_fps: int
    speed_percent: int
    max_relative_target: float
    gripper_speed_mm_s: float
    leader_gripper_friction: int
    status_hz: float
    rerun: bool
    rerun_fps: float
    play_sounds: bool

    def shell_lines(self) -> tuple[str, ...]:
        """Stable newline-delimited values consumed by piper_teleop.sh."""
        return (
            str(self.path),
            self.follower_can,
            self.leader_can,
            str(self.control_fps),
            str(self.speed_percent),
            f"{self.max_relative_target:g}",
            f"{self.gripper_speed_mm_s:g}",
            str(self.leader_gripper_friction),
            f"{self.status_hz:g}",
            str(self.rerun).lower(),
            f"{self.rerun_fps:g}",
            str(self.play_sounds).lower(),
        )


def _mapping(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return section


def _integer(value: Any, name: str, *, minimum: int, maximum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        limit = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be {limit}")
    return value


def _number(
    value: Any,
    name: str,
    *,
    positive: bool = False,
    minimum: float | None = None,
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _can_name(value: Any, name: str) -> str:
    result = str(value).strip()
    if not result or "\n" in result:
        raise ValueError(f"{name} must be a non-empty CAN interface name")
    return result


def load_teleop_settings(path: str | Path) -> TeleopSettings:
    config_path = Path(path).expanduser().resolve()
    try:
        data = yaml.safe_load(config_path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"Teleop config does not exist: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid teleop YAML in {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Teleop config must contain a YAML mapping: {config_path}")

    capture = _mapping(data, "capture")
    arm = _mapping(data, "arm")
    audio = _mapping(data, "audio")
    rerun = _boolean(capture.get("rerun", True), "capture.rerun")
    if rerun:
        # This also validates cameras/reset_grid and ensures teleop.yaml is a
        # complete, standalone visualization config rather than record.yaml indirection.
        load_reset_grid_settings(config_path)

    return TeleopSettings(
        path=config_path,
        follower_can=_can_name(arm.get("follower_can", "can_follower"), "arm.follower_can"),
        leader_can=_can_name(arm.get("leader_can", "can_leader"), "arm.leader_can"),
        control_fps=_integer(capture.get("control_fps", 200), "capture.control_fps", minimum=1),
        speed_percent=_integer(
            arm.get("speed_percent", 100),
            "arm.speed_percent",
            minimum=1,
            maximum=100,
        ),
        max_relative_target=_number(
            arm.get("max_relative_target", 100),
            "arm.max_relative_target",
            positive=True,
        ),
        gripper_speed_mm_s=_number(
            arm.get("gripper_speed_mm_s", 80),
            "arm.gripper_speed_mm_s",
            positive=True,
        ),
        leader_gripper_friction=_integer(
            arm.get("leader_gripper_friction", 5),
            "arm.leader_gripper_friction",
            minimum=1,
            maximum=10,
        ),
        status_hz=_number(
            capture.get("status_hz", 30),
            "capture.status_hz",
            minimum=0,
        ),
        rerun=rerun,
        rerun_fps=_number(capture.get("rerun_fps", 10), "capture.rerun_fps", positive=True),
        play_sounds=_boolean(audio.get("enabled", True), "audio.enabled"),
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m lerobot_piper.teleop_config CONFIG.yaml")
    try:
        settings = load_teleop_settings(sys.argv[1])
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    print("\n".join(settings.shell_lines()))


if __name__ == "__main__":
    main()

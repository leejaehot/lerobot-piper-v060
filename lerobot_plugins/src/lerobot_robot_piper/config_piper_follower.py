import math
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("piper_follower")
@dataclass(kw_only=True)
class PiperFollowerConfig(RobotConfig):
    port: str
    disable_torque_on_disconnect: bool = True
    wait_for_enter_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = 100.0
    speed_percent: int = 100
    gripper_speed_mm_s: float | None = 80.0
    terminal_update_hz: float = 30.0
    play_sounds: bool = False
    reset_grid_config_path: str | None = None
    startup_pose: dict[str, float] | None = None
    startup_pose_speed_percent: int = 20
    startup_pose_timeout_s: float = 20.0
    startup_pose_joint_tolerance_degrees: float = 2.0
    startup_pose_gripper_tolerance_mm: float = 2.0
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 1 <= self.speed_percent <= 100:
            raise ValueError("speed_percent must be in [1, 100]")
        if self.gripper_speed_mm_s is not None and self.gripper_speed_mm_s <= 0:
            raise ValueError("gripper_speed_mm_s must be positive or None")
        if self.terminal_update_hz < 0:
            raise ValueError("terminal_update_hz must be non-negative")
        if not isinstance(self.play_sounds, bool):
            raise ValueError("play_sounds must be true or false")
        if not 1 <= self.startup_pose_speed_percent <= 100:
            raise ValueError("startup_pose_speed_percent must be in [1, 100]")
        if self.startup_pose_timeout_s <= 0:
            raise ValueError("startup_pose_timeout_s must be positive")
        if self.startup_pose_joint_tolerance_degrees <= 0:
            raise ValueError("startup_pose_joint_tolerance_degrees must be positive")
        if self.startup_pose_gripper_tolerance_mm <= 0:
            raise ValueError("startup_pose_gripper_tolerance_mm must be positive")
        if self.startup_pose is not None:
            expected = {f"joint{index}" for index in range(1, 7)} | {"gripper"}
            if set(self.startup_pose) != expected:
                raise ValueError(
                    "startup_pose must contain exactly joint1..joint6 and gripper"
                )
            for name, value in self.startup_pose.items():
                if not math.isfinite(float(value)):
                    raise ValueError(f"startup_pose.{name} must be finite")
                lower = 0.0 if name == "gripper" else -100.0
                if not lower <= float(value) <= 100.0:
                    raise ValueError(
                        f"startup_pose.{name} must be in [{lower:g}, 100]"
                    )

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("piper_leader")
@dataclass
class PiperLeaderConfig(TeleoperatorConfig):
    port: str
    gripper_input_min: int = 1_000
    gripper_input_max: int = 50_000
    gripper_teaching_friction: int = 5

    def __post_init__(self) -> None:
        if self.gripper_input_min < 0:
            raise ValueError("gripper_input_min must be non-negative")
        if self.gripper_input_max <= self.gripper_input_min:
            raise ValueError("gripper_input_max must be greater than gripper_input_min")
        if not 1 <= self.gripper_teaching_friction <= 10:
            raise ValueError("gripper_teaching_friction must be in [1, 10]")

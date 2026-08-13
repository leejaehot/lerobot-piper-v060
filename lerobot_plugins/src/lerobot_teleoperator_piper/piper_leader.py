from __future__ import annotations

import logging
import time
from typing import Any

from lerobot.motors import MotorCalibration
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from lerobot_piper import PIPER_CALIBRATION, PIPER_MOTORS, PiperMotorsBus

from .config_piper_leader import PiperLeaderConfig

logger = logging.getLogger(__name__)


class PiperLeader(Teleoperator):
    config_class = PiperLeaderConfig
    name = "piper_leader"

    def __init__(
        self,
        config: PiperLeaderConfig,
        *,
        driver_factory: Any | None = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        calibration = dict(PIPER_CALIBRATION)
        calibration["gripper"] = MotorCalibration(
            7,
            0,
            0,
            config.gripper_input_min,
            config.gripper_input_max,
        )
        self.bus = PiperMotorsBus(
            port=config.port,
            calibration=calibration,
            driver_factory=driver_factory,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in PIPER_MOTORS}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self.bus.connect()
        self.bus.enable_fk_cal()
        self.bus.enable_torque()
        logger.info("%s torque on", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        return

    def setup_motors(self) -> None:
        self.bus.connect()
        self.bus.set_master()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        start = time.perf_counter()
        action = {f"{motor}.pos": value for motor, value in self.bus.get_control().items()}
        logger.debug("%s read action: %.1fms", self, (time.perf_counter() - start) * 1e3)
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disable_torque()
        self.bus.disconnect()
        logger.info("%s disconnected", self)

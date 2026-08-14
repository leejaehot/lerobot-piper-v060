from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from lerobot.motors import Motor, MotorCalibration, MotorNormMode

PIPER_MOTORS = {
    "joint1": Motor(1, "AGILEX-M", MotorNormMode.RANGE_M100_100),
    "joint2": Motor(2, "AGILEX-M", MotorNormMode.RANGE_M100_100),
    "joint3": Motor(3, "AGILEX-M", MotorNormMode.RANGE_M100_100),
    "joint4": Motor(4, "AGILEX-S", MotorNormMode.RANGE_M100_100),
    "joint5": Motor(5, "AGILEX-S", MotorNormMode.RANGE_M100_100),
    "joint6": Motor(6, "AGILEX-S", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(7, "AGILEX-S", MotorNormMode.RANGE_0_100),
}

PIPER_CALIBRATION = {
    "joint1": MotorCalibration(1, 0, 0, -150_000, 150_000),
    "joint2": MotorCalibration(2, 0, 0, 0, 180_000),
    "joint3": MotorCalibration(3, 0, 0, -170_000, 0),
    "joint4": MotorCalibration(4, 0, 0, -100_000, 100_000),
    "joint5": MotorCalibration(5, 0, 0, -65_000, 65_000),
    "joint6": MotorCalibration(6, 0, 0, -100_000, 130_000),
    "gripper": MotorCalibration(7, 0, 0, 0, 100_000),
}

PIPER_OFFICIAL_HOME_JOINTS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _make_driver(port: str) -> Any:
    from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config

    config = create_agx_arm_config(
        robot=ArmModel.PIPER,
        firmeware_version="default",
        interface="socketcan",
        channel=port,
        bitrate=1_000_000,
        enable_check_can=True,
        auto_connect=False,
        receive_own_messages=False,
        local_loopback=False,
    )
    return AgxArmFactory.create_arm(config)


class PiperMotorsBus:
    """v0.4.3 PiperMotorsBus behavior backed by the pyAgxArm public API."""

    def __init__(
        self,
        port: str,
        *,
        calibration: dict[str, MotorCalibration] | None = None,
        driver_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.port = port
        self.motors = PIPER_MOTORS
        source_calibration = calibration or PIPER_CALIBRATION
        self.calibration = {name: replace(item) for name, item in source_calibration.items()}
        self._driver = (driver_factory or _make_driver)(port)
        self._gripper = self._driver.init_effector(
            self._driver.OPTIONS.EFFECTOR.AGX_GRIPPER
        )
        self._leader_raw = dict.fromkeys(PIPER_MOTORS, 0.0)
        self._follower_raw = dict.fromkeys(PIPER_MOTORS, 0.0)
        self._target_joint_radians = [0.0] * 6
        self._target_gripper_m = 0.0
        self._follower_pose = [0.0] * 6
        self._configured_speed_percent = 100

    @property
    def is_connected(self) -> bool:
        return bool(self._driver.is_connected())

    def connect(self, handshake: bool = True) -> None:
        del handshake
        self._driver.connect()

        # C_PiperInterface_V2.ConnectPort() performs PiperInit. These public,
        # non-blocking calls produce its equivalent request frames.
        for joint_index in range(1, 7):
            self._driver.get_joint_angle_vel_limits(
                joint_index,
                timeout=0.0,
                min_interval=0.0,
            )
        for joint_index in range(1, 7):
            self._driver.get_joint_acc_limits(joint_index, timeout=0.0, min_interval=0.0)
        self._driver.get_firmware(timeout=0.0, min_interval=0.0)

    def disconnect(self, disable_torque: bool = False) -> None:
        if disable_torque:
            self.disable_torque()
        self._driver.disconnect()

    def enable_fk_cal(self) -> None:
        # EnableFkCal only toggles piper_sdk's local FK calculation. LeRobot
        # teleoperation reads joint control values directly, so no replacement
        # command is required.
        return

    def enable_torque(
        self,
        motors: str | list[str] | None = None,
        num_retry: int = 0,
    ) -> bool:
        del motors, num_retry
        retry = 50
        while not self._driver.enable() and retry:
            retry -= 1
            time.sleep(0.1)
        return retry > 0

    def disable_torque(
        self,
        motors: str | list[str] | None = None,
        num_retry: int = 0,
    ) -> None:
        del motors, num_retry
        self._driver.disable()

    def configure_leader_gripper_friction(self, teaching_friction: int) -> bool:
        """Set leader gripper teaching friction while preserving other parameters."""
        current = self._gripper.get_gripper_teaching_pendant_param(
            timeout=1.0,
            min_interval=0.0,
        )
        if current is None:
            return False
        if int(current.msg.teaching_friction) == teaching_friction:
            return True

        teaching_range_per = int(current.msg.teaching_range_per)
        max_range_config = float(current.msg.max_range_config)
        if max_range_config not in {0.0, 0.07, 0.1}:
            return False
        return bool(
            self._gripper.set_gripper_teaching_pendant_param(
                teaching_range_per=teaching_range_per,
                max_range_config=max_range_config,
                teaching_friction=teaching_friction,
                timeout=1.0,
            )
        )

    def _normalize(self, values: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for name, value in values.items():
            calibration = self.calibration[name]
            bounded = min(calibration.range_max, max(calibration.range_min, value))
            if self.motors[name].norm_mode is MotorNormMode.RANGE_M100_100:
                normalized[name] = (
                    (bounded - calibration.range_min)
                    / (calibration.range_max - calibration.range_min)
                    * 200.0
                    - 100.0
                )
            else:
                normalized[name] = (
                    (bounded - calibration.range_min)
                    / (calibration.range_max - calibration.range_min)
                    * 100.0
                )
        return normalized

    def _unnormalize(self, values: dict[str, float]) -> dict[str, int]:
        raw: dict[str, int] = {}
        for name, value in values.items():
            calibration = self.calibration[name]
            if self.motors[name].norm_mode is MotorNormMode.RANGE_M100_100:
                bounded = min(100.0, max(-100.0, float(value)))
                ratio = (bounded + 100.0) / 200.0
            else:
                bounded = min(100.0, max(0.0, float(value)))
                ratio = bounded / 100.0
            raw[name] = round(
                calibration.range_min
                + ratio * (calibration.range_max - calibration.range_min)
            )
        return raw

    @staticmethod
    def _joint_raw(message: Any) -> dict[str, float]:
        return {
            f"joint{index}": math.degrees(float(message.msg[index - 1])) * 1000.0
            for index in range(1, 7)
        }

    def get_control(self) -> dict[str, float]:
        joints = self._driver.get_leader_joint_angles()
        gripper = self._gripper.get_gripper_ctrl_states()
        if joints is not None:
            self._leader_raw.update(self._joint_raw(joints))
        if gripper is not None:
            self._leader_raw["gripper"] = float(gripper.msg.value) * 1_000_000.0
        return self._normalize(self._leader_raw)

    def get_action(self) -> dict[str, float]:
        joints = self._driver.get_joint_angles()
        gripper = self._gripper.get_gripper_status()
        if joints is not None:
            self._follower_raw.update(self._joint_raw(joints))
        if gripper is not None:
            self._follower_raw["gripper"] = float(gripper.msg.value) * 1_000_000.0
        return self._normalize(self._follower_raw)

    def set_action(self, action: dict[str, float]) -> dict[str, float]:
        raw = self._unnormalize(action)
        joints = [math.radians(raw[f"joint{index}"] * 0.001) for index in range(1, 7)]
        self._target_joint_radians = joints
        self._target_gripper_m = raw["gripper"] * 1e-6
        self._driver.move_j(joints)
        self._gripper.move_gripper_m(value=self._target_gripper_m, force=1.0)
        return self.get_control()

    def get_follower_telemetry(
        self,
    ) -> tuple[list[float], list[float], float, float, list[float]]:
        actual_joints = [
            math.radians(self._follower_raw[f"joint{index}"] * 0.001)
            for index in range(1, 7)
        ]
        actual_gripper_m = self._follower_raw["gripper"] * 1e-6
        pose = self._driver.get_flange_pose()
        if pose is not None:
            self._follower_pose = [float(value) for value in pose.msg]
        else:
            self._follower_pose = [float(value) for value in self._driver.fk(actual_joints)]
        return (
            list(self._target_joint_radians),
            actual_joints,
            self._target_gripper_m,
            actual_gripper_m,
            list(self._follower_pose),
        )

    def configure_follower(self, speed_percent: int) -> None:
        self._configured_speed_percent = speed_percent
        self._driver.set_speed_percent(speed_percent)

    def start_follower_official_home(self, speed_percent: int) -> None:
        """Move the follower to Piper's physical zero pose."""
        if not 1 <= speed_percent <= 100:
            raise ValueError("home speed_percent must be in [1, 100]")
        self._driver.set_speed_percent(speed_percent)
        self._driver.move_j(list(PIPER_OFFICIAL_HOME_JOINTS))

    def restore_follower_speed(self) -> None:
        self._driver.set_speed_percent(self._configured_speed_percent)

    def is_follower_at_official_home(
        self,
        *,
        tolerance_degrees: float,
    ) -> bool:
        if tolerance_degrees <= 0:
            raise ValueError("home tolerance_degrees must be positive")
        message = self._driver.get_joint_angles()
        if message is None:
            return False
        tolerance_radians = math.radians(tolerance_degrees)
        return all(
            abs(float(value) - target) <= tolerance_radians
            for value, target in zip(
                message.msg,
                PIPER_OFFICIAL_HOME_JOINTS,
                strict=True,
            )
        )

    def sync_read(self, data_name: str) -> dict[str, float]:
        if data_name != "Present_Position":
            raise ValueError(f"Unsupported Piper read: {data_name}")
        return self.get_action()

    def set_master(self) -> None:
        self._driver.set_leader_mode()

    def set_slave(self) -> None:
        self._driver.set_follower_mode()

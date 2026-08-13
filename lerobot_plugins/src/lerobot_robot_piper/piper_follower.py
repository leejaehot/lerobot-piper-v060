from __future__ import annotations

import logging
import math
import sys
import time
from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from lerobot_piper import PIPER_MOTORS, PiperMotorsBus

from .config_piper_follower import PiperFollowerConfig

logger = logging.getLogger(__name__)


class PiperFollower(Robot):
    config_class = PiperFollowerConfig
    name = "piper_follower"
    teleop_terminal_fullscreen = True

    def __init__(
        self,
        config: PiperFollowerConfig,
        *,
        driver_factory: Any | None = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.bus = PiperMotorsBus(port=config.port, driver_factory=driver_factory)
        self.cameras = make_cameras_from_configs(config.cameras)
        self._last_terminal_update_s = float("-inf")
        self._last_terminal_pose: list[float] | None = None
        self._terminal_loop_hz: float | None = None
        self._terminal_line_count = 0
        self._terminal_screen_active = False
        self._last_gripper_target: float | None = None
        self._last_action_time_s: float | None = None

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        features: dict[str, type | tuple[int, int, int]] = {
            f"{motor}.pos": float for motor in PIPER_MOTORS
        }
        for name, camera in self.cameras.items():
            if camera.height is None or camera.width is None:
                raise ValueError(f"Camera {name!r} has no configured shape")
            features[name] = (camera.height, camera.width, 3)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in PIPER_MOTORS}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(
            camera.is_connected for camera in self.cameras.values()
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        try:
            self.bus.connect()
            if not self.bus.enable_torque():
                raise RuntimeError(f"{self} failed to enable follower motors")
            self.configure()
            self._last_gripper_target = None
            self._last_action_time_s = None
            for camera in self.cameras.values():
                camera.connect()
        except Exception:
            for camera in self.cameras.values():
                if camera.is_connected:
                    camera.disconnect()
            if self.bus.is_connected:
                self.bus.disconnect(disable_torque=True)
            raise
        logger.info("%s follower motors enabled", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return

    def configure(self) -> None:
        self.bus.configure_follower(self.config.speed_percent)

    def setup_motors(self) -> None:
        self.bus.connect()
        self.bus.set_slave()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        # Each RealSense already has its own 30 Hz background reader. Waiting
        # for another fresh frame here makes a nominal 30 Hz loop take one
        # camera period plus processing time. Snapshot both latest buffers
        # instead; a 100 ms age limit still detects a stalled camera quickly.
        camera_frames = {
            name: camera.read_latest(max_age_ms=100)
            for name, camera in self.cameras.items()
        }

        # Read joint feedback after the camera barrier. This places the arm
        # sample close to the newest frame from both independently-clocked
        # cameras instead of up to two camera periods before it.
        observation: RobotObservation = {
            f"{motor}.pos": value for motor, value in self.bus.get_action().items()
        }
        observation.update(camera_frames)
        logger.debug("%s read state: %.1fms", self, (time.perf_counter() - start) * 1e3)
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal = {
            key.removesuffix(".pos"): float(value)
            for key, value in action.items()
            if key.endswith(".pos")
        }
        present: dict[str, float] | None = None
        if self.config.max_relative_target is not None:
            present = self.bus.sync_read("Present_Position")
            goal = ensure_safe_goal_position(
                {name: (value, present[name]) for name, value in goal.items()},
                self.config.max_relative_target,
            )
        if self.config.gripper_speed_mm_s is not None and "gripper" in goal:
            if present is None:
                present = self.bus.sync_read("Present_Position")
            now = time.monotonic()
            if self._last_gripper_target is None or self._last_action_time_s is None:
                self._last_gripper_target = float(present["gripper"])
                self._last_action_time_s = now
            elapsed_s = min(max(now - self._last_action_time_s, 0.0), 0.05)
            max_delta = self.config.gripper_speed_mm_s * elapsed_s
            requested = float(goal["gripper"])
            goal["gripper"] = min(
                self._last_gripper_target + max_delta,
                max(self._last_gripper_target - max_delta, requested),
            )
            self._last_gripper_target = goal["gripper"]
            self._last_action_time_s = now
        self.bus.set_action(goal)
        return {f"{motor}.pos": value for motor, value in goal.items()}

    def get_teleop_terminal_lines(self, loop_hz: float) -> list[str] | None:
        if self.config.terminal_update_hz == 0:
            return None
        now = time.monotonic()
        if now - self._last_terminal_update_s < 1.0 / self.config.terminal_update_hz:
            return None
        self._last_terminal_update_s = now

        target, actual, target_gripper_m, actual_gripper_m, pose = (
            self.bus.get_follower_telemetry()
        )
        if self._last_terminal_pose is None:
            pose_delta = [0.0] * 6
        else:
            pose_delta = [
                value - previous
                for value, previous in zip(pose, self._last_terminal_pose, strict=True)
            ]
            pose_delta[3:] = [
                math.atan2(math.sin(value), math.cos(value)) for value in pose_delta[3:]
            ]
        self._last_terminal_pose = list(pose)

        if self._terminal_loop_hz is None:
            self._terminal_loop_hz = loop_hz
        else:
            self._terminal_loop_hz = 0.85 * self._terminal_loop_hz + 0.15 * loop_hz

        error = [
            target_value - actual_value
            for target_value, actual_value in zip(target, actual, strict=True)
        ]

        lines = [
            "╭─ PIPER TELEOP ──────────────────────────────────────╮",
            (
                f"│ Control loop {self._terminal_loop_hz:6.1f} Hz"
                f"   │   Monitor {self.config.terminal_update_hz:5.1f} Hz       │"
            ),
            "├────────┬──────────────┬──────────────┬──────────────┤",
            "│ Joint  │ Leader qpos  │ Follower qpos│ Δ (L - F)    │",
            "│        │          rad │          rad │          rad │",
            "├────────┼──────────────┼──────────────┼──────────────┤",
        ]
        lines.extend(
            f"│ J{index:<5} │ {leader:+12.4f} │ {follower:+12.4f} │ {delta:+12.4f} │"
            for index, (leader, follower, delta) in enumerate(
                zip(target, actual, error, strict=True),
                start=1,
            )
        )
        lines.extend(
            [
                "├────────┼──────────────┼──────────────┼──────────────┤",
                (
                    f"│ Grip mm│ {target_gripper_m * 1000:+12.3f} "
                    f"│ {actual_gripper_m * 1000:+12.3f} "
                    f"│ {(target_gripper_m - actual_gripper_m) * 1000:+12.3f} │"
                ),
                "├────────┼──────────────┼──────────────┼──────────────┤",
                "│ EE     │        Value │     Δ/update │         Unit │",
            ]
        )
        ee_rows = (
            ("x", pose[0], pose_delta[0], "m"),
            ("y", pose[1], pose_delta[1], "m"),
            ("z", pose[2], pose_delta[2], "m"),
            ("rx", pose[3], pose_delta[3], "rad"),
            ("ry", pose[4], pose_delta[4], "rad"),
            ("rz", pose[5], pose_delta[5], "rad"),
        )
        lines.extend(
            f"│ {axis:<6} │ {value:+12.4f} │ {delta:+12.4f} │ {unit:>12} │"
            for axis, value, delta, unit in ee_rows
        )
        lines.append("╰────────┴──────────────┴──────────────┴──────────────╯")
        self._terminal_line_count = len(lines)
        if sys.stdout.isatty() and not self._terminal_screen_active:
            sys.stdout.write("\033[?1049h\033[?25l")
            sys.stdout.flush()
            self._terminal_screen_active = True
        return lines

    def close_teleop_terminal(self) -> None:
        if self._terminal_screen_active:
            sys.stdout.write("\033[?25h\033[?1049l")
            sys.stdout.flush()
            self._terminal_screen_active = False
        self._terminal_line_count = 0

    def prepare_for_disconnect(self) -> None:
        """Keep torque on until the operator confirms both arms are supported."""
        self.close_teleop_terminal()
        if not (
            self.config.disable_torque_on_disconnect
            and self.config.wait_for_enter_on_disconnect
        ):
            return
        if not sys.stdin.isatty():
            logger.warning(
                "%s cannot prompt on non-interactive stdin; disconnecting now",
                self,
            )
            return

        print("\n╭─ SAFE DISCONNECT ───────────────────────────────────╮")
        print("│ Follower torque is still ON.                        │")
        print("│ Support both arms, then press Enter to release it.  │")
        print("╰─────────────────────────────────────────────────────╯")
        while True:
            try:
                input("Press Enter when both arms are supported: ")
                return
            except KeyboardInterrupt:
                print("\nTorque remains ON. Support both arms before disconnecting.")
            except EOFError:
                logger.warning("stdin closed before safe-disconnect confirmation")
                return

    @check_if_not_connected
    def disconnect(self) -> None:
        self.close_teleop_terminal()
        for camera in self.cameras.values():
            if camera.is_connected:
                camera.disconnect()
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        logger.info("%s disconnected", self)

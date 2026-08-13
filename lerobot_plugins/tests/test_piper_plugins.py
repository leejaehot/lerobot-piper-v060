from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.utils import keyboard_input
from lerobot_piper import PIPER_CALIBRATION, PiperMotorsBus
from lerobot_piper.console_ui import recording_log_style
from lerobot_piper.record_cli import _effective
from lerobot_robot_piper import PiperFollower, PiperFollowerConfig
from lerobot_teleoperator_piper import PiperLeader, PiperLeaderConfig


class FakeGripper:
    def __init__(self, driver: "FakePyAgxArm") -> None:
        self.driver = driver

    def get_gripper_ctrl_states(self) -> SimpleNamespace:
        return SimpleNamespace(
            msg=SimpleNamespace(value=self.driver.leader_gripper_m),
        )

    def get_gripper_status(self) -> SimpleNamespace:
        return SimpleNamespace(
            msg=SimpleNamespace(value=self.driver.follower_gripper_m),
        )

    def move_gripper_m(self, value: float, force: float) -> None:
        self.driver.gripper_commands.append((value, force))

    def get_gripper_teaching_pendant_param(
        self,
        *,
        timeout: float,
        min_interval: float,
    ) -> SimpleNamespace:
        assert timeout == 1.0
        assert min_interval == 0.0
        return SimpleNamespace(
            msg=SimpleNamespace(
                teaching_range_per=self.driver.gripper_teaching_range_per,
                max_range_config=self.driver.gripper_max_range_config,
                teaching_friction=self.driver.gripper_teaching_friction,
            )
        )

    def set_gripper_teaching_pendant_param(
        self,
        *,
        teaching_range_per: int,
        max_range_config: float,
        teaching_friction: int,
        timeout: float,
    ) -> bool:
        assert timeout == 1.0
        self.driver.gripper_teaching_range_per = teaching_range_per
        self.driver.gripper_max_range_config = max_range_config
        self.driver.gripper_teaching_friction = teaching_friction
        self.driver.calls.append(f"gripper_friction:{teaching_friction}")
        return True


class FakePyAgxArm:
    class OPTIONS:
        class EFFECTOR:
            AGX_GRIPPER = "agx_gripper"

    def __init__(self, *, enable_after: int = 1) -> None:
        self.connected = False
        self.enable_after = enable_after
        self.enable_calls = 0
        self.calls: list[str] = []
        self.leader_joint_raw = [0, 90_000, -85_000, 0, 0, 15_000]
        self.follower_joint_raw = list(self.leader_joint_raw)
        self.leader_gripper_m = 0.0255
        self.follower_gripper_m = 0.05
        self.flange_pose = [0.3, 0.1, 0.2, 0.01, 0.02, 0.03]
        self.joint_commands: list[list[float]] = []
        self.gripper_commands: list[tuple[float, float]] = []
        self.gripper_teaching_range_per = 125
        self.gripper_max_range_config = 0.1
        self.gripper_teaching_friction = 1

    def init_effector(self, effector: str) -> FakeGripper:
        assert effector == "agx_gripper"
        self.calls.append("init_gripper")
        return FakeGripper(self)

    def connect(self) -> None:
        self.connected = True
        self.calls.append("connect")

    def disconnect(self) -> None:
        self.connected = False
        self.calls.append("disconnect")

    def is_connected(self) -> bool:
        return self.connected

    def get_joint_angle_vel_limits(self, joint_index: int, **kwargs: float) -> None:
        assert kwargs == {"timeout": 0.0, "min_interval": 0.0}
        self.calls.append(f"angle_limit:{joint_index}")

    def get_joint_acc_limits(self, joint_index: int, **kwargs: float) -> None:
        assert kwargs == {"timeout": 0.0, "min_interval": 0.0}
        self.calls.append(f"acc_limit:{joint_index}")

    def get_firmware(self, **kwargs: float) -> None:
        assert kwargs == {"timeout": 0.0, "min_interval": 0.0}
        self.calls.append("firmware")

    def enable(self) -> bool:
        self.enable_calls += 1
        self.calls.append("enable")
        return self.enable_calls >= self.enable_after

    def disable(self) -> bool:
        self.calls.append("disable")
        return True

    @staticmethod
    def _joints(raw: list[int]) -> SimpleNamespace:
        return SimpleNamespace(msg=[math.radians(value * 0.001) for value in raw])

    def get_leader_joint_angles(self) -> SimpleNamespace:
        return self._joints(self.leader_joint_raw)

    def get_joint_angles(self) -> SimpleNamespace:
        return self._joints(self.follower_joint_raw)

    def get_flange_pose(self) -> SimpleNamespace:
        return SimpleNamespace(msg=list(self.flange_pose))

    def fk(self, joints: list[float]) -> list[float]:
        del joints
        return list(self.flange_pose)

    def set_speed_percent(self, percent: int) -> None:
        self.calls.append(f"speed:{percent}")

    def move_j(self, joints: list[float]) -> None:
        self.joint_commands.append(joints)

    def set_leader_mode(self) -> None:
        self.calls.append("set_leader")

    def set_follower_mode(self) -> None:
        self.calls.append("set_follower")


def make_bus(driver: FakePyAgxArm) -> PiperMotorsBus:
    return PiperMotorsBus("can_test", driver_factory=lambda _: driver)


def test_normalization_matches_v043_ranges() -> None:
    driver = FakePyAgxArm()
    bus = make_bus(driver)
    raw_midpoints = {
        name: (item.range_min + item.range_max) / 2
        for name, item in PIPER_CALIBRATION.items()
    }

    assert bus._normalize(raw_midpoints) == pytest.approx(
        {**{f"joint{index}": 0.0 for index in range(1, 7)}, "gripper": 50.0}
    )
    assert bus._unnormalize(bus._normalize(raw_midpoints)) == pytest.approx(raw_midpoints)


def test_leader_connect_configures_balanced_gripper_friction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("lerobot_piper.piper.time.sleep", lambda _: None)
    driver = FakePyAgxArm(enable_after=3)
    leader = PiperLeader(
        PiperLeaderConfig(id="leader", port="can_leader"),
        driver_factory=lambda _: driver,
    )

    leader.connect()

    assert driver.calls[0:2] == ["init_gripper", "connect"]
    assert driver.calls.count("enable") == 3
    assert driver.calls.count("gripper_friction:5") == 1
    assert driver.gripper_teaching_range_per == 125
    assert driver.gripper_max_range_config == 0.1
    assert driver.gripper_teaching_friction == 5
    assert "set_leader" not in driver.calls
    assert not hasattr(leader.bus, "wait_for_sample")
    assert leader.get_action() == pytest.approx(
        {**{f"joint{index}.pos": 0.0 for index in range(1, 7)}, "gripper.pos": 50.0}
    )

    leader.disconnect()
    assert driver.calls[-2:] == ["disable", "disconnect"]


def test_leader_does_not_rewrite_matching_gripper_friction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("lerobot_piper.piper.time.sleep", lambda _: None)
    driver = FakePyAgxArm()
    driver.gripper_teaching_friction = 5
    leader = PiperLeader(
        PiperLeaderConfig(id="leader", port="can_leader"),
        driver_factory=lambda _: driver,
    )

    leader.connect()

    assert all(not call.startswith("gripper_friction:") for call in driver.calls)


def test_enable_repeats_exactly_like_v043(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lerobot_piper.piper.time.sleep", lambda _: None)
    driver = FakePyAgxArm(enable_after=999)
    bus = make_bus(driver)

    assert bus.enable_torque() is False
    assert driver.enable_calls == 51


def test_role_command_is_only_in_explicit_setup() -> None:
    driver = FakePyAgxArm()
    leader = PiperLeader(
        PiperLeaderConfig(id="leader", port="can_leader"),
        driver_factory=lambda _: driver,
    )

    leader.setup_motors()

    assert driver.calls[-1] == "set_leader"


def test_follower_enables_reads_and_sends_with_v060_safety_clamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("lerobot_piper.piper.time.sleep", lambda _: None)
    driver = FakePyAgxArm(enable_after=2)
    follower = PiperFollower(
        PiperFollowerConfig(
            id="follower",
            port="can_follower",
            cameras={},
            max_relative_target=2.0,
            speed_percent=30,
            gripper_speed_mm_s=None,
        ),
        driver_factory=lambda _: driver,
    )

    follower.connect()
    assert driver.calls.count("enable") == 2
    assert "speed:30" in driver.calls

    observation = follower.get_observation()
    assert observation == pytest.approx(
        {**{f"joint{index}.pos": 0.0 for index in range(1, 7)}, "gripper.pos": 50.0}
    )

    sent = follower.send_action({name: 100.0 for name in follower.action_features})
    assert sent == pytest.approx(
        {**{f"joint{index}.pos": 2.0 for index in range(1, 7)}, "gripper.pos": 52.0}
    )
    expected_raw = follower.bus._unnormalize(
        {**{f"joint{index}": 2.0 for index in range(1, 7)}, "gripper": 52.0}
    )
    assert driver.joint_commands[-1] == pytest.approx(
        [math.radians(expected_raw[f"joint{index}"] * 0.001) for index in range(1, 7)]
    )
    assert driver.gripper_commands[-1] == pytest.approx((0.052, 1.0))

    lines = follower.get_teleop_terminal_lines(loop_hz=200.0)
    assert lines is not None
    assert len(lines) == 23
    assert lines[3] == "│ Joint  │ Leader qpos  │ Follower qpos│ Δ (L - F)    │"
    assert lines[6].startswith("│ J1")
    assert lines[11].startswith("│ J6")
    assert lines[13].startswith("│ Grip mm")
    assert "+52.000" in lines[13]
    assert "+50.000" in lines[13]
    assert "+2.000" in lines[13]
    assert "+0.3000" in lines[16]
    assert "200.0 Hz" in lines[1]
    assert follower.get_teleop_terminal_lines(loop_hz=200.0) is None

    driver.flange_pose[0] += 0.01
    follower._last_terminal_update_s = float("-inf")
    changed_lines = follower.get_teleop_terminal_lines(loop_hz=200.0)
    assert changed_lines is not None
    assert "+0.0100" in changed_lines[16]

    follower.disconnect()
    assert driver.calls[-2:] == ["disable", "disconnect"]


def test_follower_enable_failure_cleans_up(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lerobot_piper.piper.time.sleep", lambda _: None)
    driver = FakePyAgxArm(enable_after=999)
    follower = PiperFollower(
        PiperFollowerConfig(id="follower", port="can_follower", cameras={}),
        driver_factory=lambda _: driver,
    )

    with pytest.raises(RuntimeError, match="failed to enable follower motors"):
        follower.connect()

    assert driver.enable_calls == 51
    assert driver.calls[-2:] == ["disable", "disconnect"]
    assert not follower.is_connected


def test_terminal_dashboard_uses_and_restores_alternate_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTerminal:
        def __init__(self) -> None:
            self.writes: list[str] = []

        def isatty(self) -> bool:
            return True

        def write(self, value: str) -> int:
            self.writes.append(value)
            return len(value)

        def flush(self) -> None:
            return

    terminal = FakeTerminal()
    monkeypatch.setattr(
        "lerobot_robot_piper.piper_follower.sys",
        SimpleNamespace(stdout=terminal),
    )
    follower = PiperFollower(
        PiperFollowerConfig(port="can_follower", cameras={}),
        driver_factory=lambda _: FakePyAgxArm(),
    )

    assert follower.get_teleop_terminal_lines(loop_hz=200.0) is not None
    assert terminal.writes == ["\033[?1049h\033[?25l"]
    assert follower._terminal_screen_active

    follower.close_teleop_terminal()
    assert terminal.writes[-1] == "\033[?25h\033[?1049l"
    assert not follower._terminal_screen_active


def test_follower_gripper_uses_time_based_smoothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter((10.0, 10.01, 10.11))
    monkeypatch.setattr(
        "lerobot_robot_piper.piper_follower.time.monotonic",
        lambda: next(ticks),
    )
    driver = FakePyAgxArm()
    follower = PiperFollower(
        PiperFollowerConfig(
            port="can_follower",
            cameras={},
            max_relative_target=100.0,
            gripper_speed_mm_s=80.0,
        ),
        driver_factory=lambda _: driver,
    )
    follower.connect()
    action = {name: 0.0 for name in follower.action_features}
    action["gripper.pos"] = 100.0

    first = follower.send_action(action)
    second = follower.send_action(action)
    third = follower.send_action(action)

    assert first["gripper.pos"] == pytest.approx(50.0)
    assert second["gripper.pos"] == pytest.approx(50.8)
    # A stalled loop is capped to a 50 ms smoothing step, avoiding a jump.
    assert third["gripper.pos"] == pytest.approx(54.8)


def test_safe_disconnect_waits_for_enter_before_torque_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeInput:
        @staticmethod
        def isatty() -> bool:
            return True

    prompts: list[str] = []
    monkeypatch.setattr(
        "lerobot_robot_piper.piper_follower.sys.stdin",
        FakeInput(),
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: prompts.append(prompt) or "",
    )
    driver = FakePyAgxArm()
    follower = PiperFollower(
        PiperFollowerConfig(port="can_follower", cameras={}),
        driver_factory=lambda _: driver,
    )
    follower.connect()

    follower.prepare_for_disconnect()

    assert prompts == ["Press Enter when both arms are supported: "]
    assert "disable" not in driver.calls
    assert follower.is_connected

    follower.disconnect()
    assert driver.calls[-2:] == ["disable", "disconnect"]


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="greater than"):
        PiperLeaderConfig(port="can_leader", gripper_input_min=10, gripper_input_max=10)
    with pytest.raises(ValueError, match="gripper_teaching_friction"):
        PiperLeaderConfig(port="can_leader", gripper_teaching_friction=0)
    with pytest.raises(ValueError, match="speed_percent"):
        PiperFollowerConfig(port="can_follower", speed_percent=101)
    with pytest.raises(ValueError, match="terminal_update_hz"):
        PiperFollowerConfig(port="can_follower", terminal_update_hz=-1)
    with pytest.raises(ValueError, match="gripper_speed_mm_s"):
        PiperFollowerConfig(port="can_follower", gripper_speed_mm_s=0)


def test_responsive_defaults() -> None:
    config = PiperFollowerConfig(port="can_follower")
    assert config.max_relative_target == 100.0
    assert config.speed_percent == 100
    assert config.gripper_speed_mm_s == 80.0
    assert config.wait_for_enter_on_disconnect is True
    assert config.terminal_update_hz == 30.0


def test_recording_log_style_marks_phases_and_restores_factory() -> None:
    original = logging.getLogRecordFactory()

    with recording_log_style(3, force_color=True):
        factory = logging.getLogRecordFactory()
        recording = factory("test", logging.INFO, __file__, 1, "Recording episode %d", (0,), None)
        reset = factory("test", logging.INFO, __file__, 1, "Reset the environment", (), None)
        socket = factory("test", logging.INFO, __file__, 1, "Created a socket", (), None)
        warning = factory("test", logging.WARNING, __file__, 1, "Capture is slow", (), None)
        capture = factory(
            "test",
            logging.INFO,
            __file__,
            1,
            "CAPTURE        150 frames / 5.00 s = 30.0 Hz (target 30 Hz)",
            (),
            None,
        )
        segment = factory(
            "test",
            logging.INFO,
            __file__,
            1,
            "SEGMENT %d      Episode %d · starts at frame %d · %.2f s",
            (3, 2, 45, 1.5),
            None,
        )

        assert "Episode 1/3" in recording.getMessage()
        assert "\033[" in recording.getMessage()
        assert "RESET" in reset.getMessage()
        assert socket.levelno == logging.DEBUG
        assert "\033[" in warning.getMessage()
        assert "\033[1;32m" in capture.getMessage()
        assert "◆ SEGMENT 3" in segment.getMessage()
        assert "\033[1;35m" in segment.getMessage()

    assert logging.getLogRecordFactory() is original


def test_piper_record_enables_foot_pedal_segments_by_default() -> None:
    args = SimpleNamespace(
        repo_id=None,
        task=None,
        episodes=None,
        seconds=None,
        reset_seconds=None,
        push_to_hub=None,
        dataset_fps=None,
        control_fps=None,
        speed=None,
        gripper_speed_mm_s=None,
        leader_gripper_friction=None,
        rerun=None,
        segments=None,
        segment_debounce_ms=None,
        test=False,
    )
    cfg = _effective(
        {
            "dataset": {"task": "Test pedal recording"},
            "cameras": {"overview": "123456"},
        },
        args,
    )

    assert cfg["segments"] is True
    assert cfg["segment_debounce_ms"] == 400
    assert cfg["gripper_speed_mm_s"] == 80
    assert cfg["leader_gripper_friction"] == 5


def test_space_pedal_latches_one_debounced_segment_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_create_key_listener(dispatch, *, controls_help: str):
        captured["dispatch"] = dispatch
        captured["help"] = controls_help
        return SimpleNamespace(stop=lambda: None)

    ticks = iter((10.0, 10.1, 10.5, 10.6))
    monkeypatch.setattr(keyboard_input, "create_key_listener", fake_create_key_listener)
    monkeypatch.setattr(keyboard_input.time, "monotonic", lambda: next(ticks))

    _, events = keyboard_input.init_keyboard_listener(
        enable_segments=True,
        segment_debounce_s=0.4,
    )
    dispatch = captured["dispatch"]
    events["recording_active"] = True

    dispatch("space")
    assert events["segment_boundary_requested"].is_set()
    events["segment_boundary_requested"].clear()

    dispatch("space")
    assert not events["segment_boundary_requested"].is_set()
    dispatch("space")
    assert events["segment_boundary_requested"].is_set()

    events["segment_boundary_requested"].clear()
    events["recording_active"] = False
    dispatch("space")
    assert not events["segment_boundary_requested"].is_set()
    assert "Space=next segment" in captured["help"]


def test_two_camera_observation_uses_nonblocking_latest_frames() -> None:
    class FakeCamera:
        is_connected = True

        def read_latest(self, max_age_ms: int) -> np.ndarray:
            assert max_age_ms == 100
            return np.zeros((2, 2, 3), dtype=np.uint8)

    driver = FakePyAgxArm()
    follower = PiperFollower(
        PiperFollowerConfig(port="can_follower", cameras={}),
        driver_factory=lambda _: driver,
    )
    follower.bus.connect()
    follower.cameras = {"overview": FakeCamera(), "side": FakeCamera()}

    observation = follower.get_observation()

    assert observation["overview"].shape == (2, 2, 3)
    assert observation["side"].shape == (2, 2, 3)
    follower.bus.disconnect(disable_torque=False)

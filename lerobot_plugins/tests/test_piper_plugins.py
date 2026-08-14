from __future__ import annotations

import logging
import math
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.utils import keyboard_input
from lerobot_piper import PIPER_CALIBRATION, PiperMotorsBus
from lerobot_piper.audio import _play_countdown, _record_cue, local_record_audio, play_cue
from lerobot_piper.console_ui import announce, recording_log_style
from lerobot_piper.grid_preview_cli import _settings as grid_preview_settings
from lerobot_piper.record_cli import _effective, _validate
from lerobot_piper.reset_grid import (
    GRID_COL,
    GRID_ROW,
    POSITION_ID,
    ResetGridGuide,
    load_reset_grid_settings,
)
from lerobot_piper.teleop_cli import _script_path as teleop_script_path
from lerobot_piper.teleop_config import load_teleop_settings
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
        self.follower_joint_raw = [round(math.degrees(value) * 1000) for value in joints]

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


def test_official_home_moves_only_follower_and_restores_speed() -> None:
    follower_driver = FakePyAgxArm()
    follower_bus = make_bus(follower_driver)
    follower_bus.configure_follower(80)

    follower_bus.start_follower_official_home(20)

    assert follower_driver.calls[-1] == "speed:20"
    assert follower_driver.joint_commands[-1] == pytest.approx([0.0] * 6)
    assert follower_bus.is_follower_at_official_home(tolerance_degrees=2)

    follower_bus.restore_follower_speed()
    assert follower_driver.calls[-1] == "speed:80"


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
    with pytest.raises(ValueError, match="play_sounds"):
        PiperFollowerConfig(port="can_follower", play_sounds="yes")


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
        initial_poses = factory(
            "test",
            logging.INFO,
            __file__,
            1,
            "INITIAL POSES  Episode %d · %s",
            (2, "spam_can · C04-R06 · white_container · C12-R06"),
            None,
        )
        exiting = factory("test", logging.INFO, __file__, 1, "Exiting", (), None)

        assert "Episode 1/3" in recording.getMessage()
        assert "\033[" in recording.getMessage()
        assert "RESET" in reset.getMessage()
        assert socket.levelno == logging.DEBUG
        assert "\033[" in warning.getMessage()
        assert "\033[1;32m" in capture.getMessage()
        assert "◆ SEGMENT 3" in segment.getMessage()
        assert "\033[1;35m" in segment.getMessage()
        assert "◎ INITIAL POSES" in initial_poses.getMessage()
        assert "spam_can · C04-R06" in initial_poses.getMessage()
        assert "◇ CLOSED" in exiting.getMessage()
        assert "DONE" not in exiting.getMessage()

    assert logging.getLogRecordFactory() is original


def test_voice_announcement_is_non_blocking_and_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        "lerobot_piper.audio.play_cue",
        lambda cue, *, enabled: calls.append((cue, enabled)),
    )

    announce("ready", enabled=False)
    announce("ready", enabled=True)

    assert calls == [("ready", False), ("ready", True)]


def test_record_voice_messages_map_to_short_korean_cues() -> None:
    assert _record_cue("Recording episode 0") == ("recording", 1)
    assert _record_cue("Recording episode 19") == ("recording", 20)
    assert _record_cue("Reset the environment") == ("reset", None)
    assert _record_cue("Reset the environment for episode 4") == ("reset", None)
    assert _record_cue("Re-record episode") == ("rerecord", None)
    assert _record_cue("Stop recording") == ("acquisition_end", None)
    assert _record_cue("Exiting") is None


def test_local_record_audio_suppresses_spd_say_and_restores_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs: list[tuple[str, bool, bool]] = []
    cues: list[tuple[str, bool, int | None]] = []

    def original(text: str, play_sounds: bool = True, blocking: bool = False) -> None:
        logs.append((text, play_sounds, blocking))

    def add_segment(*args, **kwargs) -> int:
        return int(kwargs["segment_id"]) + int(kwargs["events"]["accepted"])

    def record_loop(*args, **kwargs) -> None:
        return None

    module = SimpleNamespace(
        log_say=original,
        _add_segment_annotation=add_segment,
        record_loop=record_loop,
    )
    monkeypatch.setattr(
        "lerobot_piper.audio.play_cue",
        lambda cue, *, enabled, index=None: cues.append((cue, enabled, index)),
    )

    with local_record_audio(module, enabled=True):
        module.log_say("Recording episode 2", True, True)
        assert module._add_segment_annotation(segment_id=4, events={"accepted": True}) == 5
        assert module._add_segment_annotation(segment_id=5, events={"accepted": False}) == 5

    assert module.log_say is original
    assert module._add_segment_annotation is add_segment
    assert module.record_loop is record_loop
    assert logs == [("Recording episode 2", False, False)]
    assert cues == [("recording", True, 3), ("keyframe", True, 5)]


def test_reset_countdown_plays_three_two_one_and_honors_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cues: list[int] = []
    waits: list[float] = []

    class FakeCancel:
        def wait(self, timeout: float) -> bool:
            waits.append(timeout)
            return False

    monkeypatch.setattr(
        "lerobot_piper.audio.play_cue",
        lambda cue, *, enabled, index=None: cues.append(index),
    )
    _play_countdown(10, enabled=True, cancel=FakeCancel())

    assert waits == [7, 1, 1]
    assert cues == [3, 2, 1]

    class CancelImmediately:
        def wait(self, timeout: float) -> bool:
            return True

    cues.clear()
    _play_countdown(10, enabled=True, cancel=CancelImmediately())
    assert cues == []


def test_reset_message_schedules_countdown_for_only_the_next_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduled: list[dict[str, object]] = []
    loop_durations: list[float] = []

    class FakeThread:
        def __init__(self, *, target, kwargs, name, daemon) -> None:
            scheduled.append(
                {"target": target, "kwargs": kwargs, "name": name, "daemon": daemon}
            )

        def start(self) -> None:
            return None

        def join(self, timeout: float) -> None:
            assert timeout == 0.1

    def record_loop(*args, **kwargs) -> None:
        loop_durations.append(float(kwargs["control_time_s"]))

    module = SimpleNamespace(
        log_say=lambda *args, **kwargs: None,
        _add_segment_annotation=lambda *args, **kwargs: kwargs["segment_id"],
        record_loop=record_loop,
    )
    monkeypatch.setattr("lerobot_piper.audio.Thread", FakeThread)
    monkeypatch.setattr("lerobot_piper.audio.play_cue", lambda *args, **kwargs: True)

    with local_record_audio(module, enabled=True):
        module.log_say("Reset the environment for episode 1", True)
        module.record_loop(control_time_s=10)
        module.record_loop(control_time_s=30)

    assert loop_durations == [10, 30]
    assert len(scheduled) == 1
    assert scheduled[0]["name"] == "piper-reset-countdown"
    assert scheduled[0]["daemon"] is True
    assert scheduled[0]["kwargs"]["duration_s"] == 10


def test_initial_setup_requires_enter_then_runs_three_second_countdown() -> None:
    loop_durations: list[float] = []
    confirmation = Event()
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "waiting_for_initial_setup": False,
        "initial_setup_confirmed": confirmation,
        "input_listener_available": True,
    }

    def record_loop(*args, **kwargs) -> None:
        loop_durations.append(float(kwargs["control_time_s"]))
        if events["waiting_for_initial_setup"]:
            confirmation.set()
            events["exit_early"] = True

    module = SimpleNamespace(
        log_say=lambda *args, **kwargs: None,
        _add_segment_annotation=lambda *args, **kwargs: kwargs["segment_id"],
        record_loop=record_loop,
    )

    with local_record_audio(module, enabled=False, wait_for_enter=True):
        module.log_say("Reset the environment for episode 0", False)
        module.record_loop(control_time_s=15, events=events)
        # Initial confirmation is one-shot; a later non-recording reset stays timed.
        module.log_say("Reset the environment for episode 0", False)
        module.record_loop(control_time_s=15, events=events)

    assert loop_durations == [86_400, 3, 15]
    assert not events["waiting_for_initial_setup"]
    assert not events["exit_early"]


def test_home_reset_runs_only_after_a_recording_and_preserves_countdown_window() -> None:
    loop_durations: list[float] = []
    loop_robots: list[object] = []
    visualizations: list[dict[str, object]] = []

    class FakeFollower:
        def __init__(self) -> None:
            self.calls: list[object] = []
            self.sent_actions: list[object] = []

        def start_official_home(self, speed_percent: int) -> None:
            self.calls.append(("start", speed_percent))

        def is_at_official_home(self, tolerance_degrees: float) -> bool:
            self.calls.append(("check", tolerance_degrees))
            return True

        def finish_official_home(self) -> None:
            self.calls.append("finish")

        def get_observation(self) -> dict[str, float]:
            return {"joint1.pos": 0.0}

        def send_action(self, action: object) -> object:
            self.sent_actions.append(action)
            return action

    class FakeLeader:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def get_action(self) -> dict[str, float]:
            self.calls.append("get_action")
            return {"joint1.pos": 0.0}

    class FakeProvider:
        def __init__(self) -> None:
            self.calls = 0

        def log_visualization(self) -> None:
            self.calls += 1

    def record_loop(*args, **kwargs) -> None:
        loop_durations.append(float(kwargs["control_time_s"]))
        loop_robots.append(kwargs["robot"])

    module = SimpleNamespace(
        log_say=lambda *args, **kwargs: None,
        _add_segment_annotation=lambda *args, **kwargs: kwargs["segment_id"],
        record_loop=record_loop,
        log_visualization_data=lambda mode, **kwargs: visualizations.append(
            {"mode": mode, **kwargs}
        ),
    )
    follower = FakeFollower()
    leader = FakeLeader()
    provider = FakeProvider()

    with local_record_audio(
        module,
        enabled=False,
        home_on_reset=True,
        home_speed_percent=20,
        home_tolerance_degrees=2,
    ):
        # Initial scene setup before episode 0 must not move either arm.
        module.log_say("Reset the environment for episode 0", False)
        module.record_loop(control_time_s=4, robot=follower, teleop=leader)
        module.log_say("Recording episode 0", False)
        module.record_loop(control_time_s=1, robot=follower, teleop=leader)
        # The next reset homes only the follower, then leaves the final 3 s idle.
        module.log_say("Reset the environment for episode 1", False)
        module.record_loop(
            control_time_s=4,
            robot=follower,
            teleop=leader,
            fps=30,
            display_data=True,
            display_mode="rerun",
            display_fps=10,
            display_compressed_images=False,
            robot_observation_processor=lambda observation: observation,
            teleop_action_processor=lambda action_observation: action_observation[0],
            episode_annotation_provider=provider,
        )

    assert loop_durations[0:2] == [4, 1]
    assert 3 <= loop_durations[2] < 4
    assert loop_robots[0:2] == [follower, follower]
    assert loop_robots[2] is not follower
    loop_robots[2].send_action({"joint1.pos": 50.0})
    assert follower.sent_actions == []
    assert follower.calls[0] == ("start", 20)
    assert follower.calls[-1] == "finish"
    assert leader.calls
    assert set(leader.calls) == {"get_action"}
    assert visualizations
    assert visualizations[0]["mode"] == "rerun"
    assert provider.calls == len(visualizations)


def test_home_reset_failure_stops_before_next_episode() -> None:
    class NeverHome:
        def __init__(self) -> None:
            self.finished = False

        def start_official_home(self, *args) -> None:
            return None

        def is_at_official_home(self, tolerance_degrees: float) -> bool:
            return False

        def finish_official_home(self) -> None:
            self.finished = True

    original_calls: list[float] = []
    module = SimpleNamespace(
        log_say=lambda *args, **kwargs: None,
        _add_segment_annotation=lambda *args, **kwargs: kwargs["segment_id"],
        record_loop=lambda *args, **kwargs: original_calls.append(
            float(kwargs["control_time_s"])
        ),
    )
    follower = NeverHome()
    leader = object()

    with pytest.raises(RuntimeError, match="stopped before the next episode"):
        with local_record_audio(module, enabled=False, home_on_reset=True):
            module.log_say("Recording episode 0", False)
            module.record_loop(control_time_s=0.01, robot=follower, teleop=leader)
            module.log_say("Reset the environment for episode 1", False)
            module.record_loop(control_time_s=3.01, robot=follower, teleop=leader)

    assert original_calls == [0.01]
    assert follower.finished


def test_play_cue_uses_numbered_wav_without_waiting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sound = tmp_path / "recording_3.wav"
    sound.write_bytes(b"RIFF")
    calls: list[tuple[list[str], dict[str, object]]] = []
    monkeypatch.setattr("lerobot_piper.audio.SOUND_DIR", tmp_path)
    monkeypatch.setattr("lerobot_piper.audio._player", lambda: "/usr/bin/paplay")
    monkeypatch.setattr(
        "lerobot_piper.audio.subprocess.Popen",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    assert play_cue("recording", enabled=True, index=3) is True
    assert calls[0][0] == ["/usr/bin/paplay", str(sound)]
    assert calls[0][1]["start_new_session"] is True


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
    assert cfg["home_on_reset"] is True
    assert cfg["home_speed_percent"] == 20
    assert cfg["home_tolerance_degrees"] == 2
    assert cfg["wait_for_enter"] is True
    assert cfg["reset_grid"]["enabled"] is False
    assert cfg["reset_grid"]["rows"] == 12
    assert cfg["rerun_fps"] == 10
    assert cfg["play_sounds"] is True


def test_record_home_reset_config_requires_a_motion_free_countdown_window() -> None:
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
        home_on_reset=None,
        home_speed=None,
        gripper_speed_mm_s=None,
        leader_gripper_friction=None,
        rerun=None,
        segments=None,
        segment_debounce_ms=None,
        test=False,
    )
    cfg = _effective(
        {
            "dataset": {
                "task": "Test home reset",
                "episodes": 2,
                "reset_seconds": 3,
            },
            "cameras": {"overview": "123456"},
        },
        args,
    )

    with pytest.raises(ValueError, match="greater than 3"):
        _validate(cfg)


def test_installed_teleop_entry_point_targets_shared_launcher() -> None:
    script = teleop_script_path()
    expected = Path(__file__).resolve().parents[2] / "scripts" / "piper_teleop.sh"

    assert script == expected
    assert script.is_file()


def test_teleop_yaml_loads_standalone_arm_grid_audio_and_rates(tmp_path: Path) -> None:
    config = tmp_path / "teleop.yaml"
    config.write_text(
        """
cameras:
  egoview: "123456"
reset_grid:
  enabled: true
  camera: egoview
  columns: 16
  rows: 12
  initial_poses:
    object: [4, 6]
capture:
  dataset_fps: 30
  control_fps: 200
  status_hz: 20
  width: 640
  height: 480
  rerun: true
  rerun_fps: 10
arm:
  leader_can: can_leader_test
  follower_can: can_follower_test
  speed_percent: 70
  max_relative_target: 5
  gripper_speed_mm_s: 60
  leader_gripper_friction: 4
audio:
  enabled: false
""".strip()
    )

    settings = load_teleop_settings(config)

    assert settings.path == config.resolve()
    assert settings.leader_can == "can_leader_test"
    assert settings.follower_can == "can_follower_test"
    assert settings.control_fps == 200
    assert settings.status_hz == 20
    assert settings.speed_percent == 70
    assert settings.max_relative_target == 5
    assert settings.gripper_speed_mm_s == 60
    assert settings.leader_gripper_friction == 4
    assert settings.rerun is True
    assert settings.rerun_fps == 10
    assert settings.play_sounds is False
    assert len(settings.shell_lines()) == 12


def test_teleop_yaml_rejects_invalid_arm_value(tmp_path: Path) -> None:
    config = tmp_path / "teleop.yaml"
    config.write_text(
        """
capture:
  rerun: false
arm:
  leader_gripper_friction: 11
""".strip()
    )

    with pytest.raises(ValueError, match="leader_gripper_friction"):
        load_teleop_settings(config)


def test_grid_preview_uses_one_configured_camera_without_arm_settings() -> None:
    cfg = grid_preview_settings(
        {
            "cameras": {"egoview": "123456", "wristcam": "654321"},
            "capture": {"width": 640, "height": 480, "dataset_fps": 30},
            "reset_grid": {
                "camera": "egoview",
                "columns": 16,
                "rows": 12,
                "initial_poses": {"spam_can": [4, 6]},
            },
        },
        SimpleNamespace(camera=None, fps=None),
    )

    assert cfg["camera_name"] == "egoview"
    assert cfg["serial"] == "123456"
    assert (cfg["width"], cfg["height"]) == (640, 480)
    assert (cfg["columns"], cfg["rows"]) == (16, 12)
    assert cfg["preview_fps"] == 10
    assert cfg["initial_poses"] == {"spam_can": [4, 6]}


def test_reset_grid_uses_fixed_object_poses_and_emits_dense_values() -> None:
    guide = ResetGridGuide(
        camera_name="egoview",
        image_width=640,
        image_height=480,
        columns=16,
        rows=12,
        corners=[[0, 0], [1, 0], [1, 1], [0, 1]],
        initial_poses={"spam_can": [4, 6], "white_container": [12, 6]},
        rerun_enabled=False,
    )

    values = guide.annotations_for_episode(0)
    later_values = guide.annotations_for_episode(99)
    assert values[POSITION_ID].tolist() == [100, 108]
    assert values[GRID_COL].tolist() == [4, 12]
    assert values[GRID_ROW].tolist() == [6, 6]
    assert np.array_equal(values[GRID_COL], later_values[GRID_COL])
    assert set(values) == set(guide.dataset_features)
    assert guide.dataset_features[GRID_COL]["names"] == ["spam_can", "white_container"]


def test_reset_grid_box_normalizes_corners_and_preserves_annotations() -> None:
    guide = ResetGridGuide(
        camera_name="egoview",
        image_width=640,
        image_height=480,
        columns=16,
        rows=12,
        corners=[[0, 0], [1, 0], [1, 1], [0, 1]],
        initial_poses={
            "cider": [11, 5],
            "white_container": [[2, 2], [7, 2], [2, 9], [7, 9]],
        },
        rerun_enabled=False,
    )

    values = guide.annotations_for_episode(0)
    assert guide.num_positions == 2
    assert guide.initial_poses[1].is_box is True
    assert guide.initial_poses[1].label == "white_container · BOX C02-R02..C07-R09"
    assert guide.dataset_features[GRID_COL]["shape"] == (5,)
    assert guide.dataset_features[GRID_COL]["names"] == [
        "cider",
        "white_container.corner_1",
        "white_container.corner_2",
        "white_container.corner_3",
        "white_container.corner_4",
    ]
    assert values[GRID_COL].tolist() == [11, 2, 7, 7, 2]
    assert values[GRID_ROW].tolist() == [5, 2, 2, 9, 9]


def test_reset_grid_renders_four_corner_pose_as_closed_box(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rerun as rr

    logged: dict[str, object] = {}
    monkeypatch.setattr(
        rr,
        "LineStrips2D",
        lambda strips, **kwargs: {"strips": strips, **kwargs},
    )
    monkeypatch.setattr(
        rr,
        "Points2D",
        lambda positions, **kwargs: {"positions": positions, **kwargs},
    )
    monkeypatch.setattr(
        rr,
        "log",
        lambda path, value, **kwargs: logged.update({path: value}),
    )
    guide = ResetGridGuide(
        camera_name="egoview",
        image_width=640,
        image_height=480,
        columns=16,
        rows=12,
        corners=[[0, 0], [1, 0], [1, 1], [0, 1]],
        initial_poses={
            "cider": [11, 5],
            "white_container": [[2, 2], [7, 2], [2, 9], [7, 9]],
        },
        rerun_enabled=True,
    )

    guide._log_rerun(phase="reset")

    box = logged["observation.egoview/reset_grid/initial_pose_boxes"]
    strip = box["strips"][0]
    assert len(strip) == 5
    assert strip[0] == strip[-1]
    assert box["labels"] == ["white_container · BOX C02-R02..C07-R09"]
    assert "observation.egoview/reset_grid/initial_pose_points" in logged


def test_reset_grid_rejects_four_points_that_do_not_form_a_box() -> None:
    with pytest.raises(ValueError, match="axis-aligned corners"):
        ResetGridGuide(
            camera_name="egoview",
            image_width=640,
            image_height=480,
            columns=16,
            rows=12,
            corners=[[0, 0], [1, 0], [1, 1], [0, 1]],
            initial_poses={"container": [[2, 2], [7, 2], [3, 9], [7, 9]]},
            rerun_enabled=False,
        )


def test_reset_grid_rejects_initial_pose_outside_grid() -> None:
    with pytest.raises(ValueError, match="outside"):
        ResetGridGuide(
            camera_name="egoview",
            image_width=640,
            image_height=480,
            columns=16,
            rows=12,
            corners=[[0.1, 0.2], [0.9, 0.2], [0.8, 0.9], [0.2, 0.9]],
            initial_poses={"spam_can": [16, 6]},
            rerun_enabled=False,
        )


def test_shared_reset_grid_settings_load_teleop_camera(tmp_path) -> None:
    config = tmp_path / "record.yaml"
    config.write_text(
        """
cameras:
  egoview: "123456"
capture:
  dataset_fps: 30
  width: 640
  height: 480
reset_grid:
  enabled: true
  camera: egoview
  columns: 16
  rows: 12
  corners: [[0, 0], [1, 0], [1, 1], [0, 1]]
  initial_poses:
    spam_can: [4, 6]
    white_container: [12, 6]
""".strip()
    )

    settings = load_reset_grid_settings(config)

    assert settings.camera_name == "egoview"
    assert settings.camera_serial == "123456"
    assert (settings.image_width, settings.image_height) == (640, 480)
    assert (settings.columns, settings.rows) == (16, 12)
    assert settings.initial_poses == {"spam_can": [4, 6], "white_container": [12, 6]}


def test_follower_teleop_grid_uses_fixed_poses() -> None:
    class FakeGuide:
        def __init__(self) -> None:
            self.phases: list[tuple[int, str]] = []
            self.log_calls = 0

        def on_phase(self, episode_index: int, phase: str) -> None:
            self.phases.append((episode_index, phase))

        def log_visualization(self) -> None:
            self.log_calls += 1

    follower = PiperFollower(
        PiperFollowerConfig(port="can_follower", cameras={}),
        driver_factory=lambda _: FakePyAgxArm(),
    )
    guide = FakeGuide()
    follower._reset_grid = guide
    follower.connect()

    follower.log_teleop_visualization("rerun")
    follower.log_teleop_visualization("rerun")

    assert guide.phases == [(0, "reset")]
    assert guide.log_calls == 2

    follower.disconnect()


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
    assert events["input_listener_available"] is True
    events["waiting_for_initial_setup"] = True
    dispatch("enter")
    assert events["initial_setup_confirmed"].is_set()
    assert events["exit_early"] is True
    events["waiting_for_initial_setup"] = False
    events["initial_setup_confirmed"].clear()
    events["exit_early"] = False
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

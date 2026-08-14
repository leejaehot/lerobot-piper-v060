from __future__ import annotations

import logging
import time
from collections.abc import Callable
from types import ModuleType
from typing import Any

COUNTDOWN_RESERVE_S = 3.0
HOME_POLL_INTERVAL_S = 0.05
HOME_STABLE_SAMPLES = 3
INITIAL_SETUP_WAIT_SLICE_S = 86_400.0

_RECORD_LOOP_ARGUMENTS = {
    "robot": 0,
    "events": 1,
    "fps": 2,
    "teleop_action_processor": 3,
    "robot_action_processor": 4,
    "robot_observation_processor": 5,
    "dataset": 6,
    "teleop": 7,
    "control_time_s": 8,
    "single_task": 9,
    "display_data": 10,
    "display_mode": 11,
    "display_fps": 12,
    "display_compressed_images": 13,
    "control_fps": 14,
    "episode_annotations": 15,
    "episode_annotation_provider": 16,
}


def _argument(args: tuple[Any, ...], kwargs: dict[str, Any], name: str, default=None):
    if name in kwargs:
        return kwargs[name]
    index = _RECORD_LOOP_ARGUMENTS[name]
    return args[index] if len(args) > index else default


def _call_with_duration(
    record_loop: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    duration_s: float,
    *,
    overrides: dict[str, Any] | None = None,
) -> Any:
    call_args = list(args)
    call_kwargs = dict(kwargs)
    replacements = {"control_time_s": duration_s, **(overrides or {})}
    for name, value in replacements.items():
        index = _RECORD_LOOP_ARGUMENTS[name]
        if name in call_kwargs or len(call_args) <= index:
            call_kwargs[name] = value
        else:
            call_args[index] = value
    return record_loop(*call_args, **call_kwargs)


class _FollowerHoldRobot:
    """Expose follower observations while suppressing teleop motion commands."""

    def __init__(self, robot: Any) -> None:
        self._robot = robot

    def __getattr__(self, name: str) -> Any:
        return getattr(self._robot, name)

    def send_action(self, action: Any) -> Any:
        # The real follower driver keeps tracking the zero target issued by
        # start_official_home(). Reading the leader remains useful for Rerun,
        # but its pose must not pull the follower away during the countdown.
        return action


def _log_reset_visualization(
    record_module: ModuleType,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    if not bool(_argument(args, kwargs, "display_data", False)):
        return
    robot = _argument(args, kwargs, "robot")
    teleop = _argument(args, kwargs, "teleop")
    observation_processor = _argument(args, kwargs, "robot_observation_processor")
    teleop_processor = _argument(args, kwargs, "teleop_action_processor")
    observation = robot.get_observation()
    processed_observation = observation_processor(observation)
    action = teleop.get_action()
    processed_action = teleop_processor((action, observation))
    record_module.log_visualization_data(
        _argument(args, kwargs, "display_mode", "rerun"),
        observation=processed_observation,
        action=processed_action,
        compress_images=bool(
            _argument(args, kwargs, "display_compressed_images", False)
        ),
    )
    provider = _argument(args, kwargs, "episode_annotation_provider")
    if provider is not None:
        provider.log_visualization()


def wait_for_initial_setup(
    record_loop: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> bool:
    """Keep reset teleoperation/visualization live until Enter is confirmed."""
    events = _argument(args, kwargs, "events")
    confirmed = events.get("initial_setup_confirmed") if events is not None else None
    if (
        events is None
        or confirmed is None
        or not events.get("input_listener_available", False)
    ):
        raise RuntimeError(
            "Initial object setup requires an interactive keyboard listener; "
            "rerun with a focused terminal or use --no-wait-for-enter"
        )

    confirmed.clear()
    events["waiting_for_initial_setup"] = True
    events["exit_early"] = False
    print(
        "\n[INITIAL SETUP] Match the objects to the grid, then press Enter "
        "to start the 3-second countdown.",
        flush=True,
    )
    try:
        while not confirmed.is_set() and not events["stop_recording"]:
            _call_with_duration(
                record_loop,
                args,
                kwargs,
                INITIAL_SETUP_WAIT_SLICE_S,
            )
            if not confirmed.is_set() and not events["stop_recording"]:
                # Right/Left are not confirmations during initial setup. Restart
                # the live reset loop and keep waiting specifically for Enter.
                events["exit_early"] = False
                events["rerecord_episode"] = False
    finally:
        events["waiting_for_initial_setup"] = False

    if events["stop_recording"]:
        return False
    events["exit_early"] = False
    events["rerecord_episode"] = False
    confirmed.clear()
    logging.info("INITIAL SETUP  confirmed · starting 3-second countdown")
    return True


def run_follower_home_reset(
    record_module: ModuleType,
    record_loop: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    speed_percent: int,
    tolerance_degrees: float,
) -> Any:
    """Home only the Piper follower, then finish the reset loop."""
    duration_s = float(_argument(args, kwargs, "control_time_s", 0.0))
    if duration_s <= COUNTDOWN_RESERVE_S:
        raise RuntimeError(
            "Piper home reset needs more than 3 seconds so the countdown remains motion-free"
        )

    robot = _argument(args, kwargs, "robot")
    required_robot_methods = (
        "start_official_home",
        "is_at_official_home",
        "finish_official_home",
    )
    if robot is None or any(not hasattr(robot, name) for name in required_robot_methods):
        raise RuntimeError("Official Piper home reset requires a Piper follower")

    start_s = time.monotonic()
    deadline_s = start_s + duration_s - COUNTDOWN_RESERVE_S
    logging.info(
        "HOME           Follower Piper zero · speed %d%% · leader unchanged",
        speed_percent,
    )

    follower_started = False
    try:
        robot.start_official_home(speed_percent)
        follower_started = True

        display_fps = float(
            _argument(args, kwargs, "display_fps")
            or _argument(args, kwargs, "fps", 30)
        )
        display_interval_s = 1.0 / display_fps
        next_display_s = start_s
        stable_samples = 0
        while True:
            now_s = time.monotonic()
            if now_s >= next_display_s:
                _log_reset_visualization(record_module, args, kwargs)
                next_display_s = now_s + display_interval_s

            follower_home = robot.is_at_official_home(tolerance_degrees)
            stable_samples = stable_samples + 1 if follower_home else 0
            if stable_samples >= HOME_STABLE_SAMPLES:
                break
            if now_s >= deadline_s:
                raise RuntimeError(
                    "Piper follower did not reach the official home pose before the final "
                    "3-second countdown; recording stopped before the next episode"
                )
            time.sleep(min(HOME_POLL_INTERVAL_S, max(deadline_s - now_s, 0.0)))
        elapsed_s = time.monotonic() - start_s
        remaining_s = max(duration_s - elapsed_s, 0.0)
        logging.info(
            "HOME           follower reached in %.2f s · holding zero for %.2f s",
            elapsed_s,
            remaining_s,
        )
        if remaining_s <= 0:
            return None
        return _call_with_duration(
            record_loop,
            args,
            kwargs,
            remaining_s,
            overrides={"robot": _FollowerHoldRobot(robot)},
        )
    finally:
        if follower_started:
            robot.finish_official_home()

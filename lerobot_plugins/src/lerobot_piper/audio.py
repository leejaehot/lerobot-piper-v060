from __future__ import annotations

import logging
import re
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from types import ModuleType

from lerobot_piper.home_reset import (
    _call_with_duration,
    run_follower_home_reset,
    wait_for_initial_setup,
)
from lerobot_piper.project_paths import PIPER_ROOT

SOUND_DIR = PIPER_ROOT / "assets/sounds"

_CUE_FILES = {
    "ready": "ready.wav",
    "recording": "recording_start.wav",
    "keyframe": "keyframe.wav",
    "reset": "environment_reset.wav",
    "rerecord": "rerecord.wav",
    "acquisition_end": "acquisition_end.wav",
    "support_arms": "support_arms.wav",
    "disconnected": "disconnected.wav",
    "upload_complete": "upload_complete.wav",
}


def _cue_path(cue: str, index: int | None = None) -> Path:
    if cue in {"recording", "keyframe", "countdown"} and index is not None:
        numbered = SOUND_DIR / f"{cue}_{index}.wav"
        if numbered.is_file() or cue == "countdown":
            return numbered
    try:
        return SOUND_DIR / _CUE_FILES[cue]
    except KeyError as exc:
        raise ValueError(f"Unknown Piper sound cue: {cue}") from exc


def _player() -> str | None:
    return shutil.which("paplay") or shutil.which("pw-play") or shutil.which("aplay")


def play_cue(cue: str, *, enabled: bool, index: int | None = None) -> bool:
    """Start one local WAV cue without waiting on the robot control thread."""
    if not enabled:
        return False
    path = _cue_path(cue, index)
    player = _player()
    if not path.is_file() or player is None:
        logging.getLogger(__name__).warning(
            "Audio cue unavailable: cue=%s path=%s player=%s",
            cue,
            path,
            player,
        )
        return False
    subprocess.Popen(
        [player, str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return True


def _record_cue(message: str) -> tuple[str, int | None] | None:
    recording = re.fullmatch(r"Recording episode (\d+)", message)
    if recording:
        return "recording", int(recording.group(1)) + 1
    if re.fullmatch(r"Reset the environment for episode \d+", message):
        return "reset", None
    return {
        "Reset the environment": ("reset", None),
        "Re-record episode": ("rerecord", None),
        "Stop recording": ("acquisition_end", None),
        "Dataset uploaded to hub": ("upload_complete", None),
    }.get(message)


def _play_countdown(duration_s: float, *, enabled: bool, cancel: Event) -> None:
    """Play Three/Two/One during the last three seconds of a reset window."""
    if not enabled or duration_s < 3 or cancel.wait(duration_s - 3):
        return
    for number in (3, 2, 1):
        play_cue("countdown", enabled=True, index=number)
        if number > 1 and cancel.wait(1):
            return


@contextmanager
def local_record_audio(
    record_module: ModuleType,
    *,
    enabled: bool,
    home_on_reset: bool = False,
    home_speed_percent: int = 20,
    home_tolerance_degrees: float = 2.0,
    wait_for_enter: bool = False,
) -> Iterator[None]:
    """Install local WAV cues and the optional between-episode home reset."""
    original_log_say = record_module.log_say
    original_add_segment = record_module._add_segment_annotation
    original_record_loop = record_module.record_loop
    reset_loop_pending = False
    home_loop_pending = False
    initial_setup_loop_pending = False
    initial_setup_done = False
    recording_since_reset = False

    def log_cue(text: str, play_sounds: bool = True, blocking: bool = False) -> None:
        nonlocal home_loop_pending, initial_setup_loop_pending
        nonlocal recording_since_reset, reset_loop_pending
        # Keep LeRobot's log message but explicitly suppress its spd-say backend.
        original_log_say(text, play_sounds=False, blocking=False)
        mapped = _record_cue(text)
        if mapped is not None:
            cue, index = mapped
            play_cue(cue, enabled=enabled and play_sounds, index=index)
            if cue == "reset":
                reset_loop_pending = enabled and play_sounds
                home_loop_pending = home_on_reset and recording_since_reset
                initial_setup_loop_pending = (
                    wait_for_enter and not initial_setup_done and not recording_since_reset
                )
                recording_since_reset = False
            elif cue == "recording":
                recording_since_reset = True

    def add_segment_cue(*args, **kwargs) -> int:
        previous_segment = int(kwargs["segment_id"])
        segment = original_add_segment(*args, **kwargs)
        if segment > previous_segment:
            play_cue("keyframe", enabled=enabled, index=segment)
        return segment

    def record_loop_cue(*args, **kwargs) -> None:
        nonlocal home_loop_pending, initial_setup_done
        nonlocal initial_setup_loop_pending, reset_loop_pending
        use_countdown = reset_loop_pending
        use_home = home_loop_pending
        use_initial_setup = initial_setup_loop_pending
        reset_loop_pending = False
        home_loop_pending = False
        initial_setup_loop_pending = False
        if not use_countdown and not use_home and not use_initial_setup:
            return original_record_loop(*args, **kwargs)

        cancel = Event()
        countdown = None
        if use_initial_setup:
            if not wait_for_initial_setup(original_record_loop, args, kwargs):
                return None
            initial_setup_done = True
            countdown_duration_s = 3.0
        else:
            countdown_duration_s = float(
                kwargs.get("control_time_s", args[8] if len(args) > 8 else 0.0)
            )

        if use_countdown:
            countdown = Thread(
                target=_play_countdown,
                kwargs={
                    "duration_s": countdown_duration_s,
                    "enabled": True,
                    "cancel": cancel,
                },
                name="piper-reset-countdown",
                daemon=True,
            )
            countdown.start()
        try:
            if use_initial_setup:
                return _call_with_duration(
                    original_record_loop,
                    args,
                    kwargs,
                    countdown_duration_s,
                )
            if use_home:
                return run_follower_home_reset(
                    record_module,
                    original_record_loop,
                    args,
                    kwargs,
                    speed_percent=home_speed_percent,
                    tolerance_degrees=home_tolerance_degrees,
                )
            return original_record_loop(*args, **kwargs)
        finally:
            cancel.set()
            if countdown is not None:
                countdown.join(timeout=0.1)

    record_module.log_say = log_cue
    record_module._add_segment_annotation = add_segment_cue
    record_module.record_loop = record_loop_cue
    try:
        yield
    finally:
        record_module.log_say = original_log_say
        record_module._add_segment_annotation = original_add_segment
        record_module.record_loop = original_record_loop

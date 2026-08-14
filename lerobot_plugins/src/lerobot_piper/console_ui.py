from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TextIO

_ANSI = {
    "red": "31",
    "green": "32",
    "yellow": "33",
    "cyan": "36",
    "magenta": "35",
    "dim": "90",
}


def supports_color(stream: TextIO = sys.stderr) -> bool:
    """Follow terminal capability and the conventional NO_COLOR opt-out."""
    return (
        os.getenv("NO_COLOR") is None
        and os.getenv("TERM", "") != "dumb"
        and hasattr(stream, "isatty")
        and stream.isatty()
    )


def paint(text: str, tone: str, *, bold: bool = False, enabled: bool = True) -> str:
    if not enabled:
        return text
    codes = ["1"] if bold else []
    codes.append(_ANSI[tone])
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def phase(
    label: str,
    message: str,
    tone: str = "cyan",
    *,
    stream: TextIO = sys.stdout,
) -> None:
    enabled = supports_color(stream)
    tag = paint(f"[{label}]", tone, bold=True, enabled=enabled)
    print(f"{tag} {message}", file=stream, flush=True)


def announce(message: str, *, enabled: bool) -> None:
    """Play a local status cue without making robot operation depend on audio."""
    try:
        from lerobot_piper.audio import play_cue

        play_cue(message, enabled=enabled)
    except Exception as exc:
        logging.getLogger(__name__).warning("Voice announcement unavailable: %s", exc)


def _replace_message(record: logging.LogRecord, message: str) -> None:
    # getMessage() has already applied %-style arguments. Clear them so the
    # formatter does not try to apply them a second time to the styled string.
    record.msg = message
    record.args = ()


@contextmanager
def recording_log_style(
    total_episodes: int,
    *,
    force_color: bool | None = None,
) -> Iterator[None]:
    """Style recording logs without changing LeRobot's recording loop.

    LeRobot recreates its console handler inside ``record()``. A temporary
    LogRecord factory survives that reset, so the Piper plugin can decorate
    phase changes and demote known noise while leaving upstream logging
    configuration untouched.
    """
    previous_factory = logging.getLogRecordFactory()
    enabled = supports_color(sys.stderr) if force_color is None else force_color

    def factory(*args, **kwargs) -> logging.LogRecord:
        record = previous_factory(*args, **kwargs)
        message = record.getMessage()

        # python-can emits this once per interface. It remains available in
        # debug logs but no longer obscures the meaningful arm-ready messages.
        if message == "Created a socket":
            record.levelno = logging.DEBUG
            record.levelname = "DEBUG"
            return record

        match = re.fullmatch(r"Recording episode (\d+)", message)
        if match:
            episode = int(match.group(1)) + 1
            message = f"● RECORDING   Episode {episode}/{total_episodes}"
            _replace_message(record, paint(message, "red", bold=True, enabled=enabled))
            return record

        segment = re.fullmatch(
            r"SEGMENT (\d+)\s+Episode (\d+) · starts at frame (\d+) · ([\d.]+) s",
            message,
        )
        if segment:
            message = (
                f"◆ SEGMENT {segment.group(1)}   Episode {segment.group(2)} · "
                f"starts at frame {segment.group(3)} · {segment.group(4)} s"
            )
            _replace_message(record, paint(message, "magenta", bold=True, enabled=enabled))
            return record

        initial_poses = re.fullmatch(r"INITIAL POSES\s+Episode (\d+) · (.+)", message)
        if initial_poses:
            message = (
                f"◎ INITIAL POSES  Episode {initial_poses.group(1)}/{total_episodes} · "
                f"{initial_poses.group(2)}"
            )
            _replace_message(record, paint(message, "green", bold=True, enabled=enabled))
            return record

        reset_episode = re.fullmatch(r"Reset the environment for episode (\d+)", message)
        if reset_episode:
            episode = int(reset_episode.group(1)) + 1
            message = (
                f"◆ RESET       Episode {episode}/{total_episodes} · use the fixed egoview poses"
            )
            _replace_message(record, paint(message, "cyan", bold=True, enabled=enabled))
            return record

        transitions = {
            "Reset the environment": ("◆ RESET       Reposition objects and arm", "cyan"),
            "Re-record episode": ("↻ RE-RECORD   Discarding the current take", "magenta"),
            "Stop recording": ("■ FINALIZING  Saving videos and metadata", "yellow"),
            # LeRobot emits "Exiting" from its cleanup path on both success
            # and failure. Keep this neutral; record_cli prints COMPLETE only
            # after record() actually returns without an exception.
            "Exiting": ("◇ CLOSED      Hardware disconnected", "cyan"),
        }
        if message in transitions:
            text, tone = transitions[message]
            _replace_message(record, paint(text, tone, bold=True, enabled=enabled))
            return record

        if "follower motors enabled" in message:
            message = "✓ FOLLOWER    Motors enabled"
            _replace_message(record, paint(message, "green", bold=True, enabled=enabled))
            return record
        if "PiperLeader torque on" in message:
            message = "✓ LEADER      Control stream active"
            _replace_message(record, paint(message, "green", bold=True, enabled=enabled))
            return record
        if message.startswith("Streaming encoding is disabled."):
            message = "VIDEO          Buffered H.264 encoding enabled"
            _replace_message(record, paint(message, "dim", enabled=enabled))
            return record

        capture = re.fullmatch(
            r"CAPTURE\s+(\d+) frames / ([\d.]+) s = ([\d.]+) Hz \(target (\d+) Hz\)",
            message,
        )
        if capture:
            actual_hz = float(capture.group(3))
            target_hz = float(capture.group(4))
            tone = "green" if actual_hz >= target_hz * 0.98 else "yellow"
            _replace_message(record, paint(message, tone, bold=True, enabled=enabled))
            return record

        if record.levelno >= logging.ERROR:
            _replace_message(record, paint(message, "red", bold=True, enabled=enabled))
        elif record.levelno >= logging.WARNING:
            _replace_message(record, paint(message, "yellow", bold=True, enabled=enabled))
        return record

    logging.setLogRecordFactory(factory)
    try:
        yield
    finally:
        logging.setLogRecordFactory(previous_factory)

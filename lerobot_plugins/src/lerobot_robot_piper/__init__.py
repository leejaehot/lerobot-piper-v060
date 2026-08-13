"""Piper follower plugin and registration entry point."""

from .config_piper_follower import PiperFollowerConfig
from .piper_follower import PiperFollower

# This distribution contains both sides of the Piper pair. Import the leader
# package here so LeRobot's one distribution discovery registers both configs.
import lerobot_teleoperator_piper as _leader_plugin  # noqa: F401, E402

__all__ = ["PiperFollower", "PiperFollowerConfig"]

from __future__ import annotations

import os
import sys
from pathlib import Path

from lerobot_piper.project_paths import PIPER_ROOT


def _script_path() -> Path:
    return PIPER_ROOT / "scripts" / "teleop_backend.sh"


def main() -> None:
    """Run the internal teleop backend from the installed console entry point."""
    script = _script_path()
    if not script.is_file():
        raise SystemExit(f"Piper teleop launcher does not exist: {script}")
    os.execv(script, [str(script), *sys.argv[1:]])


if __name__ == "__main__":
    main()

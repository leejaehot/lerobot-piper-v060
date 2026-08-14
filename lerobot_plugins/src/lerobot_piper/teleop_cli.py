from __future__ import annotations

import os
import sys
from pathlib import Path


def _script_path() -> Path:
    piper_root = Path(os.getenv("PIPER_ROOT", Path(__file__).resolve().parents[3]))
    return piper_root / "scripts" / "piper_teleop.sh"


def main() -> None:
    """Run the shared Piper teleop launcher from an installed console entry point."""
    script = _script_path()
    if not script.is_file():
        raise SystemExit(f"Piper teleop launcher does not exist: {script}")
    os.execv(script, [str(script), *sys.argv[1:]])


if __name__ == "__main__":
    main()

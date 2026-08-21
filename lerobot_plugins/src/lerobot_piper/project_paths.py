from __future__ import annotations

import os
from pathlib import Path


_ROOT_MARKERS = ("configs", "scripts", "lerobot_plugins")


def _is_piper_root(path: Path) -> bool:
    return all((path / marker).exists() for marker in _ROOT_MARKERS)


def find_piper_root(start: Path | None = None) -> Path:
    """Return the Piper checkout used by console commands.

    A sourced activation script exports ``PIPER_ROOT``. Editable installs can
    also discover the checkout by walking upward from this module. A regular
    wheel remains usable when it is paired with a checkout and ``PIPER_ROOT``.
    """

    configured = os.getenv("PIPER_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not _is_piper_root(root):
            raise RuntimeError(f"PIPER_ROOT is not a Piper checkout: {root}")
        return root

    source = (start or Path(__file__)).resolve()
    for candidate in source.parents:
        if _is_piper_root(candidate):
            return candidate
    raise RuntimeError(
        "Cannot locate the Piper checkout. Source scripts/activate_lerobot_v060.sh "
        "or export PIPER_ROOT=/path/to/piper."
    )


PIPER_ROOT = find_piper_root()

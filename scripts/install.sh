#!/usr/bin/env bash

set -Eeuo pipefail
EXPECTED_LEROBOT_COMMIT="30da8e687a6dfc617fcd94afc367ac7071c376ce"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIPER_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LEROBOT_ROOT="${1:-${PIPER_LEROBOT_ROOT:-$HOME/lerobot_v060}}"
PATCH_FILE="$PIPER_ROOT/patches/lerobot-v0.6.0-piper.patch"
CONSTRAINTS_FILE="$PIPER_ROOT/constraints.txt"

if [[ ! -d "$LEROBOT_ROOT/.git" ]]; then
    echo "ERROR: LeRobot checkout not found: $LEROBOT_ROOT" >&2
    echo "Clone LeRobot v0.6.0 first or pass its path as the first argument." >&2
    exit 1
fi
LEROBOT_ROOT="$(cd -- "$LEROBOT_ROOT" && pwd)"

if [[ "$(git -C "$LEROBOT_ROOT" rev-parse HEAD)" != "$EXPECTED_LEROBOT_COMMIT" ]]; then
    echo "ERROR: LeRobot checkout must be the v0.6.0 release ($EXPECTED_LEROBOT_COMMIT)." >&2
    echo "Current checkout: $(git -C "$LEROBOT_ROOT" rev-parse HEAD)" >&2
    exit 1
fi

if [[ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.12" ]]; then
    echo "ERROR: activate the Python 3.12 environment before running install.sh." >&2
    exit 1
fi

INSTALLED_LEROBOT_ROOT="$(
    python -c 'from pathlib import Path; import lerobot; print(Path(lerobot.__file__).resolve().parents[2])' \
        2>/dev/null || true
)"
if [[ "$INSTALLED_LEROBOT_ROOT" != "$LEROBOT_ROOT" ]]; then
    echo "ERROR: the active Python does not use the requested editable LeRobot checkout." >&2
    echo "Install it first:" >&2
    echo "  cd $LEROBOT_ROOT && python -m pip install -e \".[core_scripts,intelrealsense]\"" >&2
    exit 1
fi

if git -C "$LEROBOT_ROOT" apply --check "$PATCH_FILE" 2>/dev/null; then
    git -C "$LEROBOT_ROOT" apply "$PATCH_FILE"
elif git -C "$LEROBOT_ROOT" apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
    echo "LeRobot compatibility patch is already applied."
else
    echo "ERROR: patch does not apply cleanly; use a clean LeRobot v0.6.0 checkout." >&2
    exit 1
fi

python -m pip install -c "$CONSTRAINTS_FILE" -e "$PIPER_ROOT/lerobot_plugins"

printf 'export PIPER_LEROBOT_ROOT=%q\n' "$LEROBOT_ROOT" > "$PIPER_ROOT/configs/local.env"

if [[ ! -f "$PIPER_ROOT/configs/record.yaml" ]]; then
    cp "$PIPER_ROOT/configs/record.example.yaml" "$PIPER_ROOT/configs/record.yaml"
fi
if [[ ! -f "$PIPER_ROOT/configs/teleop.yaml" ]]; then
    cp "$PIPER_ROOT/configs/teleop.example.yaml" "$PIPER_ROOT/configs/teleop.yaml"
fi
if [[ ! -f "$PIPER_ROOT/configs/rollout.yaml" ]]; then
    cp "$PIPER_ROOT/configs/rollout.example.yaml" "$PIPER_ROOT/configs/rollout.yaml"
fi
if [[ ! -f "$PIPER_ROOT/configs/replay.yaml" ]]; then
    cp "$PIPER_ROOT/configs/replay.example.yaml" "$PIPER_ROOT/configs/replay.yaml"
fi

echo
echo "Installed Piper integration. Next steps:"
echo "  1. source $PIPER_ROOT/scripts/activate_lerobot_v060.sh"
echo "  2. can_init --configure --leader canX --follower canY"
echo "  3. edit configs/teleop.yaml, record.yaml, rollout.yaml, and replay.yaml"
echo "  4. piper_vis"
echo "  5. piper_teleop --help && piper_record --help && piper_rollout --help && piper_replay --help"

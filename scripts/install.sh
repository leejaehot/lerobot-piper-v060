#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIPER_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LEROBOT_ROOT="${1:-${PIPER_LEROBOT_ROOT:-$HOME/lerobot_v060}}"
PATCH_FILE="$PIPER_ROOT/patches/lerobot-v0.6.0-piper.patch"

if [[ ! -d "$LEROBOT_ROOT/.git" ]]; then
    echo "ERROR: LeRobot checkout not found: $LEROBOT_ROOT" >&2
    echo "Clone LeRobot v0.6.0 first or pass its path as the first argument." >&2
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

python -m pip install -e "$PIPER_ROOT/lerobot_plugins"

if [[ ! -f "$PIPER_ROOT/configs/record.yaml" ]]; then
    cp "$PIPER_ROOT/configs/record.example.yaml" "$PIPER_ROOT/configs/record.yaml"
fi
if [[ ! -f "$PIPER_ROOT/configs/teleop.yaml" ]]; then
    cp "$PIPER_ROOT/configs/teleop.example.yaml" "$PIPER_ROOT/configs/teleop.yaml"
fi
if [[ ! -f "$PIPER_ROOT/configs/rollout.yaml" ]]; then
    cp "$PIPER_ROOT/configs/rollout.example.yaml" "$PIPER_ROOT/configs/rollout.yaml"
fi

echo
echo "Installed Piper integration. Next steps:"
echo "  1. can_init --configure --leader canX --follower canY"
echo "  2. edit $PIPER_ROOT/configs/teleop.yaml and configs/record.yaml"
echo "  3. piper_teleop --init-can"
echo "  4. piper_rollout act --check"

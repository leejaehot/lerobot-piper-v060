#!/usr/bin/env bash

# Source this file; executing it cannot activate Conda in the parent shell.
PIPER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PIPER_LEROBOT_ROOT="${PIPER_LEROBOT_ROOT:-$HOME/lerobot_v060}"
PIPER_CONDA_ENV="${PIPER_CONDA_ENV:-lerobot_v060}"

CONDA_SH="${PIPER_CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
if [[ -f "$CONDA_SH" ]]; then
    # `conda` can be present as an executable while `conda activate` is still
    # unavailable in non-interactive shells. Loading conda.sh is idempotent and
    # makes this helper work from terminals, scripts, and container entrypoints.
    # shellcheck disable=SC1090
    source "$CONDA_SH"
elif ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found; set PIPER_CONDA_SH to conda.sh" >&2
    return 1 2>/dev/null || exit 1
fi
conda activate "$PIPER_CONDA_ENV"

# Keep Piper datasets/calibration separate from AI-Worker while sharing one env.
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot-v060-piper}"

# Export the serial-based Piper CAN role mapping when it has been configured.
PIPER_CAN_CONFIG="${PIPER_CAN_CONFIG:-$PIPER_ROOT/configs/can_mapping.env}"
export PIPER_CAN_CONFIG
if [[ -f "$PIPER_CAN_CONFIG" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$PIPER_CAN_CONFIG"
    set +a
fi

case ":$PATH:" in
    *:"$PIPER_ROOT/scripts":*) ;;
    *) export PATH="$PIPER_ROOT/scripts:$PATH" ;;
esac
cd "$PIPER_LEROBOT_ROOT"

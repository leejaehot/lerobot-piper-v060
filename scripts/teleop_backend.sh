#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIPER_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
INVOCATION_DIR="$(pwd -P)"
INIT_CAN=false
DRY_RUN=false
TELEOP_CONFIG="${PIPER_TELEOP_CONFIG:-$PIPER_ROOT/configs/teleop.yaml}"
RERUN_OVERRIDE=""
declare -a EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: piper_teleop [--config PATH] [--init-can] [--rerun|--no-rerun] [--dry-run] [LEROBOT_OPTIONS ...]

Start the LeRobot 0.6 Piper teleop using the same connection flow as the
working LeRobot 0.4.3 implementation, backed by the pyAgxArm public API.

Examples:
  piper_teleop --init-can
  piper_teleop --config ~/piper/configs/teleop.yaml
  piper_teleop --fps=100 --robot.speed_percent=80

Configuration:
  Default file: ~/piper/configs/teleop.yaml
  Override path: PIPER_TELEOP_CONFIG=/path/to/teleop.yaml
  Existing PIPER_TELEOP_* variables still override individual YAML values.
  LEROBOT_TELEOP_CONSOLE_LEVEL=WARNING
EOF
}

while (($#)); do
    case "$1" in
        --config)
            if (($# < 2)); then
                echo "ERROR: --config requires a YAML path" >&2
                exit 2
            fi
            TELEOP_CONFIG="$2"
            shift
            ;;
        --config=*) TELEOP_CONFIG="${1#*=}" ;;
        --init-can) INIT_CAN=true ;;
        --dry-run) DRY_RUN=true ;;
        --rerun) RERUN_OVERRIDE=true ;;
        --no-rerun) RERUN_OVERRIDE=false ;;
        -h|--help)
            usage
            exit 0
            ;;
        *) EXTRA_ARGS+=("$1") ;;
    esac
    shift
done

if [[ "$TELEOP_CONFIG" != /* ]]; then
    TELEOP_CONFIG="$INVOCATION_DIR/$TELEOP_CONFIG"
fi
TELEOP_CONFIG="$(realpath -m -- "$TELEOP_CONFIG")"

# Internal backend for the piper_teleop console entry point.
# Activates lerobot_v060 and exports the serial-based CAN role mapping.
source "$PIPER_ROOT/scripts/activate_lerobot_v060.sh"

if TELEOP_CONFIG_OUTPUT="$(python -m lerobot_piper.teleop_config "$TELEOP_CONFIG")"; then
    mapfile -t TELEOP_VALUES <<<"$TELEOP_CONFIG_OUTPUT"
else
    exit $?
fi
if ((${#TELEOP_VALUES[@]} != 12)); then
    echo "ERROR: teleop config loader returned ${#TELEOP_VALUES[@]} values; expected 12" >&2
    exit 2
fi

TELEOP_CONFIG="${TELEOP_VALUES[0]}"
FOLLOWER_CAN="${PIPER_FOLLOWER_CAN:-${TELEOP_VALUES[1]}}"
LEADER_CAN="${PIPER_LEADER_CAN:-${TELEOP_VALUES[2]}}"
FPS="${PIPER_TELEOP_FPS:-${TELEOP_VALUES[3]}}"
SPEED_PERCENT="${PIPER_TELEOP_SPEED_PERCENT:-${TELEOP_VALUES[4]}}"
MAX_RELATIVE_TARGET="${PIPER_TELEOP_MAX_RELATIVE_TARGET:-${TELEOP_VALUES[5]}}"
GRIPPER_SPEED_MM_S="${PIPER_TELEOP_GRIPPER_SPEED_MM_S:-${TELEOP_VALUES[6]}}"
LEADER_GRIPPER_FRICTION="${PIPER_LEADER_GRIPPER_FRICTION:-${TELEOP_VALUES[7]}}"
STATUS_HZ="${PIPER_TELEOP_STATUS_HZ:-${TELEOP_VALUES[8]}}"
RERUN="${RERUN_OVERRIDE:-${PIPER_TELEOP_RERUN:-${TELEOP_VALUES[9]}}}"
RERUN_FPS="${PIPER_TELEOP_RERUN_FPS:-${TELEOP_VALUES[10]}}"
PLAY_SOUNDS="${PIPER_TELEOP_SOUNDS:-${TELEOP_VALUES[11]}}"
GRID_CONFIG="${PIPER_TELEOP_GRID_CONFIG:-${PIPER_RECORD_CONFIG:-$TELEOP_CONFIG}}"
if [[ "$GRID_CONFIG" != /* ]]; then
    GRID_CONFIG="$INVOCATION_DIR/$GRID_CONFIG"
fi
GRID_CONFIG="$(realpath -m -- "$GRID_CONFIG")"
export LEROBOT_TELEOP_CONSOLE_LEVEL="${LEROBOT_TELEOP_CONSOLE_LEVEL:-WARNING}"

if ! [[ "$LEADER_GRIPPER_FRICTION" =~ ^([1-9]|10)$ ]]; then
    echo "ERROR: PIPER_LEADER_GRIPPER_FRICTION must be an integer from 1 to 10" >&2
    exit 2
fi

if [[ "$RERUN" != "true" && "$RERUN" != "false" ]]; then
    echo "ERROR: PIPER_TELEOP_RERUN must be true or false" >&2
    exit 2
fi
if [[ "$PLAY_SOUNDS" != "true" && "$PLAY_SOUNDS" != "false" ]]; then
    echo "ERROR: PIPER_TELEOP_SOUNDS must be true or false" >&2
    exit 2
fi
if ! [[ "$RERUN_FPS" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "$RERUN_FPS" == "0" ]]; then
    echo "ERROR: PIPER_TELEOP_RERUN_FPS must be positive" >&2
    exit 2
fi
if "$RERUN" && [[ ! -f "$GRID_CONFIG" ]]; then
    echo "ERROR: reset-grid config does not exist: $GRID_CONFIG" >&2
    exit 2
fi

if "$INIT_CAN"; then
    if "$DRY_RUN"; then
        "$PIPER_ROOT/scripts/can_init.sh" --dry-run
    else
        "$PIPER_ROOT/scripts/can_init.sh"
    fi
fi

if ! "$DRY_RUN" || ! "$INIT_CAN"; then
    for interface in "$FOLLOWER_CAN" "$LEADER_CAN"; do
        if [[ ! -e "/sys/class/net/$interface" ]]; then
            echo "ERROR: CAN interface '$interface' does not exist" >&2
            echo "Run: piper_teleop --init-can" >&2
            exit 1
        fi
        if ! "$DRY_RUN"; then
            CAN_DETAILS="$(ip -details link show "$interface")"
            if [[ "$CAN_DETAILS" != *"can state ERROR-ACTIVE"* ]]; then
                echo "ERROR: CAN interface '$interface' is not ERROR-ACTIVE" >&2
                echo "Stop other Piper processes, then run: piper_teleop --init-can" >&2
                echo "If it remains unhealthy, power-cycle that arm controller." >&2
                exit 1
            fi
        fi
    done
fi

COMMAND=(
    lerobot-teleoperate
    --robot.type=piper_follower
    --robot.id=piper_follower
    "--robot.port=$FOLLOWER_CAN"
    "--robot.speed_percent=$SPEED_PERCENT"
    "--robot.max_relative_target=$MAX_RELATIVE_TARGET"
    "--robot.gripper_speed_mm_s=$GRIPPER_SPEED_MM_S"
    "--robot.terminal_update_hz=$STATUS_HZ"
    "--robot.play_sounds=$PLAY_SOUNDS"
    --teleop.type=piper_leader
    --teleop.id=piper_leader
    "--teleop.port=$LEADER_CAN"
    "--teleop.gripper_teaching_friction=$LEADER_GRIPPER_FRICTION"
    "--fps=$FPS"
)

if "$RERUN"; then
    COMMAND+=(
        "--robot.reset_grid_config_path=$GRID_CONFIG"
        --display_data=true
        --display_mode=rerun
        "--display_fps=$RERUN_FPS"
        --display_compressed_images=false
        --display_images_only=true
    )
fi
COMMAND+=("${EXTRA_ARGS[@]}")

if "$DRY_RUN"; then
    printf 'Command:'
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

echo "╭─ PIPER TELEOP ───────────────────────────────────────────────╮"
printf '│ %-60s │\n' "$LEADER_CAN  ->  LeRobot  ->  $FOLLOWER_CAN"
printf '│ %-60s │\n' "Config $(basename -- "$TELEOP_CONFIG")"
printf '│ %-60s │\n' "Control ${FPS} Hz  |  Speed ${SPEED_PERCENT}%  |  Target cap ${MAX_RELATIVE_TARGET}"
printf '│ %-60s │\n' "Follower grip ${GRIPPER_SPEED_MM_S} mm/s  |  Leader friction ${LEADER_GRIPPER_FRICTION}/10"
if "$RERUN"; then
    printf '│ %-60s │\n' "Rerun egoview grid ${RERUN_FPS} Hz  |  Fixed object poses"
else
    printf '│ %-60s │\n' "Rerun OFF  |  Enable with --rerun"
fi
printf '│ %-60s │\n' "Monitor ${STATUS_HZ} Hz  |  Press Ctrl-C to stop"
printf '│ %-60s │\n' "Voice announcements ${PLAY_SOUNDS^^}"
echo "├──────────────────────────────────────────────────────────────┤"
printf '│ %-60s │\n' "CAUTION: follower torque enables immediately."
printf '│ %-60s │\n' "Initializing arms; dashboard appears when ready (up to 5 s)."
echo "╰──────────────────────────────────────────────────────────────╯"
exec "${COMMAND[@]}"

#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIPER_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
INIT_CAN=false
DRY_RUN=false
declare -a EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: piper_teleop [--init-can] [--dry-run] [LEROBOT_OPTIONS ...]

Start the LeRobot 0.6 Piper teleop using the same connection flow as the
working LeRobot 0.4.3 implementation, backed by the pyAgxArm public API.

Examples:
  piper_teleop --init-can
  piper_teleop --fps=100 --robot.speed_percent=80

Environment defaults:
  PIPER_TELEOP_FPS=200
  PIPER_TELEOP_SPEED_PERCENT=100
  PIPER_TELEOP_MAX_RELATIVE_TARGET=100
  PIPER_TELEOP_GRIPPER_SPEED_MM_S=80
  PIPER_LEADER_GRIPPER_FRICTION=5
  PIPER_TELEOP_STATUS_HZ=30
  LEROBOT_TELEOP_CONSOLE_LEVEL=WARNING
EOF
}

while (($#)); do
    case "$1" in
        --init-can) INIT_CAN=true ;;
        --dry-run) DRY_RUN=true ;;
        -h|--help)
            usage
            exit 0
            ;;
        *) EXTRA_ARGS+=("$1") ;;
    esac
    shift
done

# Activates lerobot_v060 and exports the serial-based CAN role mapping.
source "$PIPER_ROOT/scripts/activate_lerobot_v060.sh"

FOLLOWER_CAN="${PIPER_FOLLOWER_CAN:-can_follower}"
LEADER_CAN="${PIPER_LEADER_CAN:-can_leader}"
FPS="${PIPER_TELEOP_FPS:-200}"
SPEED_PERCENT="${PIPER_TELEOP_SPEED_PERCENT:-100}"
MAX_RELATIVE_TARGET="${PIPER_TELEOP_MAX_RELATIVE_TARGET:-100}"
GRIPPER_SPEED_MM_S="${PIPER_TELEOP_GRIPPER_SPEED_MM_S:-80}"
LEADER_GRIPPER_FRICTION="${PIPER_LEADER_GRIPPER_FRICTION:-5}"
STATUS_HZ="${PIPER_TELEOP_STATUS_HZ:-30}"
export LEROBOT_TELEOP_CONSOLE_LEVEL="${LEROBOT_TELEOP_CONSOLE_LEVEL:-WARNING}"

if ! [[ "$LEADER_GRIPPER_FRICTION" =~ ^([1-9]|10)$ ]]; then
    echo "ERROR: PIPER_LEADER_GRIPPER_FRICTION must be an integer from 1 to 10" >&2
    exit 2
fi

if "$INIT_CAN"; then
    if "$DRY_RUN"; then
        "$PIPER_ROOT/scripts/can_init" --dry-run
    else
        "$PIPER_ROOT/scripts/can_init"
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
    --teleop.type=piper_leader
    --teleop.id=piper_leader
    "--teleop.port=$LEADER_CAN"
    "--teleop.gripper_teaching_friction=$LEADER_GRIPPER_FRICTION"
    "--fps=$FPS"
    "${EXTRA_ARGS[@]}"
)

if "$DRY_RUN"; then
    printf 'Command:'
    printf ' %q' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

echo "╭─ PIPER TELEOP ───────────────────────────────────────────────╮"
printf '│ %-60s │\n' "$LEADER_CAN  ->  LeRobot  ->  $FOLLOWER_CAN"
printf '│ %-60s │\n' "Control ${FPS} Hz  |  Speed ${SPEED_PERCENT}%  |  Target cap ${MAX_RELATIVE_TARGET}"
printf '│ %-60s │\n' "Follower grip ${GRIPPER_SPEED_MM_S} mm/s  |  Leader friction ${LEADER_GRIPPER_FRICTION}/10"
printf '│ %-60s │\n' "Monitor ${STATUS_HZ} Hz  |  Press Ctrl-C to stop"
echo "├──────────────────────────────────────────────────────────────┤"
printf '│ %-60s │\n' "CAUTION: follower torque enables immediately."
printf '│ %-60s │\n' "Initializing arms; dashboard appears when ready (up to 5 s)."
echo "╰──────────────────────────────────────────────────────────────╯"
exec "${COMMAND[@]}"

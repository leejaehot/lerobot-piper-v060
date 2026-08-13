#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIPER_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="${PIPER_CAN_CONFIG:-$PIPER_ROOT/configs/can_mapping.env}"
BITRATE="${PIPER_CAN_BITRATE:-}"
FIRMWARE_VERSION="${PIPER_FIRMWARE_VERSION:-}"
ACTION="up"
DRY_RUN=false
NO_RENAME=false
CONFIGURE=false
LEADER_REF=""
FOLLOWER_REF=""
LEADER_NAME=""
FOLLOWER_NAME=""
declare -a REQUESTED_INTERFACES=()

usage() {
    cat <<'EOF'
Usage: can_init [OPTIONS] [INTERFACE ...]

Discover connected gs_usb SocketCAN adapters and initialize them for Piper.
A saved USB-serial mapping can rename adapters to stable role names.

Options:
  --configure             Save leader/follower USB serial mapping, then exit.
  --leader REF            Current interface name or USB serial for the leader.
  --follower REF          Current interface name or USB serial for the follower.
  --leader-name NAME      Stable leader name (default: can_leader).
  --follower-name NAME    Stable follower name (default: can_follower).
  --config FILE           Mapping file (default: ~/piper/configs/can_mapping.env).
  --status                Show identity, role, link state, bitrate, and counters.
  --down                  Bring selected interfaces down.
  --bitrate RATE          CAN bitrate (default: saved value or 1000000).
  --no-rename             Initialize CAN but retain the current kernel names.
  --dry-run               Print privileged commands without executing them.
  -h, --help              Show this help.

Examples:
  can_init --status
  can_init --configure --leader can4 --follower can5
  can_init
  can_init --down
  can_init --dry-run

The configure command only records adapter identities. Initialization never
sends robot enable, home, or motion commands.
EOF
}

while (($#)); do
    case "$1" in
        --configure)
            CONFIGURE=true
            ;;
        --leader|--follower|--leader-name|--follower-name|--config|--bitrate)
            if (($# < 2)); then
                echo "ERROR: $1 requires a value" >&2
                exit 2
            fi
            option="$1"
            value="$2"
            case "$option" in
                --leader) LEADER_REF="$value" ;;
                --follower) FOLLOWER_REF="$value" ;;
                --leader-name) LEADER_NAME="$value" ;;
                --follower-name) FOLLOWER_NAME="$value" ;;
                --config) CONFIG_FILE="$value" ;;
                --bitrate) BITRATE="$value" ;;
            esac
            shift
            ;;
        --status)
            ACTION="status"
            ;;
        --down)
            ACTION="down"
            ;;
        --no-rename)
            NO_RENAME=true
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            REQUESTED_INTERFACES+=("$1")
            ;;
    esac
    shift
done

for command_name in ip udevadm find sed sort head; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command not found: $command_name" >&2
        exit 1
    fi
done

config_value() {
    local key="$1"
    [[ -f "$CONFIG_FILE" ]] || return 0
    sed -n "s/^${key}=//p" "$CONFIG_FILE" | head -n 1
}

CONFIG_LEADER_CAN="$(config_value PIPER_LEADER_CAN)"
CONFIG_LEADER_SERIAL="$(config_value PIPER_LEADER_USB_SERIAL)"
CONFIG_FOLLOWER_CAN="$(config_value PIPER_FOLLOWER_CAN)"
CONFIG_FOLLOWER_SERIAL="$(config_value PIPER_FOLLOWER_USB_SERIAL)"
CONFIG_BITRATE="$(config_value PIPER_CAN_BITRATE)"
CONFIG_FIRMWARE="$(config_value PIPER_FIRMWARE_VERSION)"

MAPPED_LEADER_CAN="${PIPER_LEADER_CAN:-$CONFIG_LEADER_CAN}"
MAPPED_LEADER_SERIAL="${PIPER_LEADER_USB_SERIAL:-$CONFIG_LEADER_SERIAL}"
MAPPED_FOLLOWER_CAN="${PIPER_FOLLOWER_CAN:-$CONFIG_FOLLOWER_CAN}"
MAPPED_FOLLOWER_SERIAL="${PIPER_FOLLOWER_USB_SERIAL:-$CONFIG_FOLLOWER_SERIAL}"
BITRATE="${BITRATE:-${CONFIG_BITRATE:-1000000}}"
FIRMWARE_VERSION="${FIRMWARE_VERSION:-${CONFIG_FIRMWARE:-default}}"

if [[ ! "$BITRATE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: bitrate must be a positive integer, got: $BITRATE" >&2
    exit 2
fi

validate_interface_name() {
    local name="$1"
    if ((${#name} == 0 || ${#name} > 15)) || [[ ! "$name" =~ ^[A-Za-z0-9_.-]+$ ]]; then
        echo "ERROR: invalid interface name '$name' (use 1-15 letters, digits, _, ., or -)" >&2
        exit 2
    fi
}

udev_property() {
    local interface="$1"
    local key="$2"
    udevadm info --query=property --path="/sys/class/net/$interface" 2>/dev/null \
        | sed -n "s/^${key}=//p" \
        | head -n 1
}

is_gs_usb() {
    [[ "$(udev_property "$1" ID_NET_DRIVER)" == "gs_usb" ]]
}

declare -a ALL_INTERFACES=()
while IFS= read -r interface; do
    is_gs_usb "$interface" || continue
    ALL_INTERFACES+=("$interface")
done < <(find /sys/class/net -mindepth 1 -maxdepth 1 -printf '%f\n' | sort -V)

if ((${#ALL_INTERFACES[@]} == 0)); then
    echo "ERROR: no connected gs_usb CAN adapters found" >&2
    exit 1
fi

serial_for() {
    udev_property "$1" ID_SERIAL_SHORT
}

resolve_adapter() {
    local reference="$1"
    local interface serial match=""

    if [[ -e "/sys/class/net/$reference" ]]; then
        if ! is_gs_usb "$reference"; then
            echo "ERROR: $reference is not a gs_usb adapter" >&2
            return 1
        fi
        printf '%s\n' "$reference"
        return 0
    fi

    for interface in "${ALL_INTERFACES[@]}"; do
        serial="$(serial_for "$interface")"
        if [[ -n "$serial" && "$serial" == "$reference" ]]; then
            if [[ -n "$match" ]]; then
                echo "ERROR: USB serial '$reference' is not unique" >&2
                return 1
            fi
            match="$interface"
        fi
    done

    if [[ -z "$match" ]]; then
        echo "ERROR: no connected gs_usb adapter matches '$reference'" >&2
        return 1
    fi
    printf '%s\n' "$match"
}

if "$CONFIGURE"; then
    if ((${#REQUESTED_INTERFACES[@]})); then
        echo "ERROR: positional interfaces cannot be used with --configure" >&2
        exit 2
    fi
    if [[ -z "$LEADER_REF" ]]; then
        if [[ -t 0 ]]; then
            read -r -p "Current leader interface or USB serial: " LEADER_REF
        else
            echo "ERROR: --configure requires --leader in non-interactive use" >&2
            exit 2
        fi
    fi
    if [[ -z "$FOLLOWER_REF" ]]; then
        if [[ -t 0 ]]; then
            read -r -p "Current follower interface or USB serial: " FOLLOWER_REF
        else
            echo "ERROR: --configure requires --follower in non-interactive use" >&2
            exit 2
        fi
    fi

    LEADER_INTERFACE="$(resolve_adapter "$LEADER_REF")"
    FOLLOWER_INTERFACE="$(resolve_adapter "$FOLLOWER_REF")"
    if [[ "$LEADER_INTERFACE" == "$FOLLOWER_INTERFACE" ]]; then
        echo "ERROR: leader and follower must be different adapters" >&2
        exit 2
    fi

    LEADER_SERIAL="$(serial_for "$LEADER_INTERFACE")"
    FOLLOWER_SERIAL="$(serial_for "$FOLLOWER_INTERFACE")"
    if [[ -z "$LEADER_SERIAL" || -z "$FOLLOWER_SERIAL" ]]; then
        echo "ERROR: both adapters must expose a stable USB serial" >&2
        exit 1
    fi

    LEADER_NAME="${LEADER_NAME:-${MAPPED_LEADER_CAN:-can_leader}}"
    FOLLOWER_NAME="${FOLLOWER_NAME:-${MAPPED_FOLLOWER_CAN:-can_follower}}"
    validate_interface_name "$LEADER_NAME"
    validate_interface_name "$FOLLOWER_NAME"
    if [[ "$LEADER_NAME" == "$FOLLOWER_NAME" ]]; then
        echo "ERROR: leader and follower interface names must differ" >&2
        exit 2
    fi

    mkdir -p "$(dirname "$CONFIG_FILE")"
    TEMP_CONFIG="$(mktemp "${CONFIG_FILE}.tmp.XXXXXX")"
    trap 'rm -f "$TEMP_CONFIG"' EXIT
    {
        echo "# Generated by can_init --configure; roles are bound to USB serials."
        printf 'PIPER_FOLLOWER_CAN=%s\n' "$FOLLOWER_NAME"
        printf 'PIPER_FOLLOWER_USB_SERIAL=%s\n' "$FOLLOWER_SERIAL"
        printf 'PIPER_LEADER_CAN=%s\n' "$LEADER_NAME"
        printf 'PIPER_LEADER_USB_SERIAL=%s\n' "$LEADER_SERIAL"
        printf 'PIPER_CAN_BITRATE=%s\n' "$BITRATE"
        printf 'PIPER_FIRMWARE_VERSION=%s\n' "$FIRMWARE_VERSION"
    } > "$TEMP_CONFIG"
    chmod 600 "$TEMP_CONFIG"
    mv "$TEMP_CONFIG" "$CONFIG_FILE"
    trap - EXIT

    echo "Saved Piper CAN role mapping to $CONFIG_FILE"
    printf '  leader:   %s -> serial=%s -> %s\n' "$LEADER_INTERFACE" "$LEADER_SERIAL" "$LEADER_NAME"
    printf '  follower: %s -> serial=%s -> %s\n' "$FOLLOWER_INTERFACE" "$FOLLOWER_SERIAL" "$FOLLOWER_NAME"
    echo "Run 'can_init' to apply the stable names and initialize CAN."
    exit 0
fi

declare -a INTERFACES=()
if ((${#REQUESTED_INTERFACES[@]})); then
    for interface in "${REQUESTED_INTERFACES[@]}"; do
        if [[ ! -e "/sys/class/net/$interface" ]]; then
            echo "ERROR: interface does not exist: $interface" >&2
            exit 2
        fi
        if ! is_gs_usb "$interface"; then
            echo "ERROR: $interface is not a gs_usb adapter; refusing to reconfigure it" >&2
            exit 2
        fi
        INTERFACES+=("$interface")
    done
else
    INTERFACES=("${ALL_INTERFACES[@]}")
fi

role_for_serial() {
    local serial="$1"
    if [[ -n "$MAPPED_LEADER_SERIAL" && "$serial" == "$MAPPED_LEADER_SERIAL" ]]; then
        echo leader
    elif [[ -n "$MAPPED_FOLLOWER_SERIAL" && "$serial" == "$MAPPED_FOLLOWER_SERIAL" ]]; then
        echo follower
    else
        echo unassigned
    fi
}

target_for_serial() {
    local serial="$1"
    if [[ -n "$MAPPED_LEADER_SERIAL" && "$serial" == "$MAPPED_LEADER_SERIAL" ]]; then
        echo "${MAPPED_LEADER_CAN:-can_leader}"
    elif [[ -n "$MAPPED_FOLLOWER_SERIAL" && "$serial" == "$MAPPED_FOLLOWER_SERIAL" ]]; then
        echo "${MAPPED_FOLLOWER_CAN:-can_follower}"
    fi
}

print_identity() {
    local interface="$1"
    local serial usb_path role target
    serial="$(serial_for "$interface")"
    usb_path="$(udev_property "$interface" ID_PATH)"
    role="$(role_for_serial "$serial")"
    target="$(target_for_serial "$serial")"
    printf '%-12s role=%-10s serial=%-24s target=%-12s path=%s\n' \
        "$interface" "$role" "${serial:-unknown}" "${target:--}" "${usb_path:-unknown}"
}

show_status() {
    local interface="$1"
    print_identity "$interface"
    ip -br link show "$interface"
    ip -details -statistics link show "$interface" | sed 's/^/  /'
}

run_root() {
    if "$DRY_RUN"; then
        printf 'sudo'
        printf ' %q' "$@"
        printf '\n'
    else
        sudo "$@"
    fi
}

if [[ "$ACTION" == "status" ]]; then
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "No saved role mapping: $CONFIG_FILE"
        echo "Use: can_init --configure --leader INTERFACE --follower INTERFACE"
        echo
    fi
    for interface in "${INTERFACES[@]}"; do
        show_status "$interface"
    done
    exit 0
fi

echo "Selected Piper USB-CAN adapters:"
for interface in "${INTERFACES[@]}"; do
    print_identity "$interface"
done

if ! "$DRY_RUN"; then
    sudo -v
fi

if [[ "$ACTION" == "down" ]]; then
    for interface in "${INTERFACES[@]}"; do
        run_root ip link set dev "$interface" down
    done
    echo "Selected interfaces are DOWN."
    exit 0
fi

run_root modprobe gs_usb

declare -A DESIRED=()
declare -A SELECTED=()
declare -A USED_TARGET=()
for interface in "${INTERFACES[@]}"; do
    SELECTED["$interface"]=1
    serial="$(serial_for "$interface")"
    target="$(target_for_serial "$serial")"
    if "$NO_RENAME" || [[ -z "$target" ]]; then
        target="$interface"
    else
        validate_interface_name "$target"
    fi
    if [[ -n "${USED_TARGET[$target]:-}" ]]; then
        echo "ERROR: multiple selected adapters would be named '$target'" >&2
        exit 2
    fi
    USED_TARGET["$target"]="$interface"
    DESIRED["$interface"]="$target"
done

# Refuse collisions with interfaces that will not be renamed away.
for interface in "${INTERFACES[@]}"; do
    target="${DESIRED[$interface]}"
    [[ "$target" == "$interface" ]] && continue
    if [[ -e "/sys/class/net/$target" ]]; then
        if [[ -z "${SELECTED[$target]:-}" || "${DESIRED[$target]:-$target}" == "$target" ]]; then
            echo "ERROR: cannot rename $interface to $target; that name is already occupied" >&2
            exit 1
        fi
    fi
done

for interface in "${INTERFACES[@]}"; do
    run_root ip link set dev "$interface" down
done

# Rename through temporary names so leader/follower name swaps are safe.
declare -A CURRENT=()
temp_index=0
for interface in "${INTERFACES[@]}"; do
    target="${DESIRED[$interface]}"
    CURRENT["$interface"]="$interface"
    [[ "$target" == "$interface" ]] && continue
    while :; do
        temporary="pcan_tmp${temp_index}"
        ((temp_index += 1))
        [[ ! -e "/sys/class/net/$temporary" && -z "${USED_TARGET[$temporary]:-}" ]] && break
    done
    run_root ip link set dev "$interface" name "$temporary"
    CURRENT["$interface"]="$temporary"
done

declare -a FINAL_INTERFACES=()
for interface in "${INTERFACES[@]}"; do
    target="${DESIRED[$interface]}"
    current="${CURRENT[$interface]}"
    if [[ "$current" != "$target" ]]; then
        run_root ip link set dev "$current" name "$target"
    fi
    FINAL_INTERFACES+=("$target")
done

for interface in "${FINAL_INTERFACES[@]}"; do
    run_root ip link set dev "$interface" type can bitrate "$BITRATE"
    run_root ip link set dev "$interface" up
done

if "$DRY_RUN"; then
    exit 0
fi

echo
echo "Configured at ${BITRATE} bit/s:"
for interface in "${FINAL_INTERFACES[@]}"; do
    print_identity "$interface"
    ip -br link show "$interface"
    ip -details link show "$interface" | sed -n '/can state/p; /bitrate/p'
done

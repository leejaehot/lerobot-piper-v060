# LeRobot Piper v0.6.0

**English** | [한국어](README.ko.md)

AgileX Piper leader/follower teleoperation, visualization, dataset recording,
and policy rollout for [LeRobot v0.6.0](https://github.com/huggingface/lerobot/tree/v0.6.0).

> [!CAUTION]
> Connecting enables follower torque and may move the arm. Support both arms,
> clear the workspace, and keep the emergency stop ready before starting.

## Commands

| Command | Purpose | Hardware |
| --- | --- | --- |
| `piper_teleop` | Leader → follower teleoperation | Leader + follower |
| `piper_vis` | RealSense and reset-grid preview | Camera only |
| `piper_record` | LeRobot dataset recording | Leader + follower + cameras |
| `piper_rollout` | ACT / Diffusion Policy rollout | Follower + cameras |
| `piper_replay` | Dataset inspection and optional hardware replay | None by default |

## Requirements

- Ubuntu 24.04 (`x86_64` or `aarch64`)
- Python 3.12
- LeRobot `v0.6.0`
- [pyAgxArm](https://github.com/agilexrobotics/pyAgxArm) `799b841`
- Two Piper arms and two `gs_usb` USB-CAN adapters
- Intel RealSense D435 + D405
- NVIDIA CUDA is recommended for rollout.

> [!NOTE]
> The hardware-validated environment is **NVIDIA Jetson AGX Thor** (`aarch64`),
> Ubuntu 24.04.4, Jetson Linux R38.2.2, Python 3.12.13, and PyTorch
> 2.11.0+cu130. Standard `x86_64` Ubuntu uses the same LeRobot commit and Python
> constraints; only the PyTorch, TorchVision, and RealSense wheels are selected
> for that machine's architecture and CUDA stack.

## Quick start

### 1. Host packages

```bash
sudo apt update
sudo apt install -y git git-lfs can-utils iproute2 udev ffmpeg
sudo modprobe gs_usb
git lfs install
```

CAN adapters and RealSense cameras must be visible on the host before installing
Python packages.

```bash
ip -br link
lsusb
```

### 2. Conda environment

Install [Miniforge](https://github.com/conda-forge/miniforge), then run:

```bash
git clone \
  https://github.com/leejaehot/lerobot-piper-v060.git ~/piper
conda env create -f ~/piper/environment.yml
conda activate lerobot_v060
```

`constraints.txt` pins the Python packages shared by both architectures.
PyTorch, TorchVision, TorchCodec, and `pyrealsense2` are excluded because their
binary wheels must match the host platform.

### 3. Clone and install

Keep the LeRobot checkout beside the Piper integration.

```bash
git clone --branch v0.6.0 --depth 1 \
  https://github.com/huggingface/lerobot.git ~/lerobot_v060

cd ~/lerobot_v060
python -m pip install -c ~/piper/constraints.txt \
  -e ".[core_scripts,intelrealsense,diffusion]"

~/piper/scripts/install.sh ~/lerobot_v060
source ~/piper/scripts/activate_lerobot_v060.sh
```

`install.sh` applies the LeRobot v0.6.0 compatibility patch, installs the Piper
plugin in editable mode, and creates local `teleop.yaml`, `record.yaml`,
`rollout.yaml`, and `replay.yaml` files once. It records the selected LeRobot
checkout in the Git-ignored `configs/local.env`.

On standard Ubuntu, the command above resolves wheels for the host architecture.
On Jetson, first install the NVIDIA PyTorch/TorchVision and `pyrealsense2` wheels
that match the installed JetPack/CUDA release.

Start every new terminal with:

```bash
source ~/piper/scripts/activate_lerobot_v060.sh
```

### 4. Verify software

These commands do not connect CAN or motors.

```bash
python -c 'import lerobot_robot_piper, lerobot_teleoperator_piper; print("Piper plugins: OK")'
piper_teleop --help
piper_vis --help
piper_record --help
piper_rollout --help
piper_replay --help
```

Also verify CUDA on machines used for rollout.

```bash
python -c 'import platform, torch; print(platform.machine(), torch.__version__, torch.cuda.is_available())'
```

## One-time hardware setup

### 1. CAN roles

Connect both USB-CAN adapters. Disconnect them one at a time to identify the
current leader and follower interface names, then save their roles.

```bash
can_init --status
can_init --configure --leader can4 --follower can5
can_init
can_init --status
```

The adapters are subsequently restored as `can_leader` and `can_follower` by
USB serial. Machine-local mapping is stored in the Git-ignored
`configs/can_mapping.env`.

### 2. Cameras and local configs

```bash
lerobot-find-cameras realsense
```

Copy the reported serials into:

| File | Values to edit |
| --- | --- |
| `~/piper/configs/teleop.yaml` | `cameras.egoview` |
| `~/piper/configs/record.yaml` | `cameras.egoview`, `cameras.wristcam`, dataset fields |
| `~/piper/configs/rollout.yaml` | Both camera serials and shared checkpoint settings |

Preview the camera and grid first. This does not connect either arm.

```bash
piper_vis
```

## Run

### Teleoperation

Use conservative limits for the first motion test.

```bash
PIPER_TELEOP_SPEED_PERCENT=20 \
PIPER_TELEOP_MAX_RELATIVE_TARGET=5 \
piper_teleop --init-can --no-rerun
```

After checking pose and direction, run with the configured defaults.

```bash
piper_teleop --init-can
```

### Visualization

```bash
piper_vis
piper_vis --config ~/piper/configs/teleop.yaml
```

Press `q` or `Esc` to close the viewer.

### Record

Start with a five-second local smoke test.

```bash
piper_record --dry-run
piper_record --init-can --test
piper_record
```

Frequently changed values can be overridden from the CLI.

```bash
piper_record \
  --repo-id local/pick_and_place \
  --task "Pick the object and place it in the container." \
  --episodes 20 \
  --seconds 60
```

| Input | Action |
| --- | --- |
| `Right` / `n` | Save the current episode and continue |
| `Left` / `r` | Discard and re-record the current episode |
| `Space` | Start a new segment on the next frame |
| `Esc` / `q` | Stop recording |

### Rollout

Checkpoint-specific settings live in `configs/rollouts/*.yaml` profiles.

```bash
piper_rollout --list-profiles
cp ~/piper/configs/rollout-profile.example.yaml \
  ~/piper/configs/rollouts/my_act.yaml
```

Edit the checkpoint, task, and—when needed—camera aliases and startup pose in
`my_act.yaml`. Always validate in this order:

```bash
# Files and configuration only
piper_rollout my_act --dry-run

# One synthetic CUDA inference; no hardware connection
piper_rollout my_act --check

# Bounded 30-second rollout on the follower
piper_rollout my_act --init-can
```

Every live rollout, regardless of policy type, first opens the camera-only
`piper_vis` reset grid. Press `Enter` after confirming the object setup to
continue with CAN initialization and rollout, or `q`/`Esc` to cancel without
enabling the motors. This step is skipped by `--dry-run` and `--check`.

### Replay

Inspect recorded video and action traces before policy debugging.

```bash
piper_replay --list-profiles
cp ~/piper/configs/replay-profile.example.yaml \
  ~/piper/configs/replays/my_dataset.yaml
piper_replay my_dataset --episode 0
```

Edit `repo_id` and `root` in `my_dataset.yaml` first. Sending actions to the
follower requires an explicit `--hardware` flag.

```bash
piper_replay my_dataset --episode 0 --hardware --dry-run
piper_replay my_dataset --episode 0 --hardware --init-can
```

## Configuration

- Shared defaults and profile templates: `configs/*.example.yaml`
- Machine-local values: `configs/teleop.yaml`, `record.yaml`, `rollout.yaml`, `replay.yaml`
- Per-checkpoint rollout profiles: `configs/rollouts/*.yaml`
- Per-dataset replay profiles: `configs/replays/*.yaml`

Machine-local YAML files and CAN mappings are ignored by Git. Keep lab-specific
values in those local files rather than editing the shared examples.

## Safety and shutdown

- Validate first teleop motion at speed 20% and relative target 5 or lower.
- Use rollout in the order `--dry-run` → `--check` → hardware run.
- After `Ctrl-C`, support both arms—especially the follower—and press Enter to release torque.
- `piper_vis`, `--help`, and rollout `--dry-run`/`--check` do not move motors.

## Update

Update the integration without replacing machine-local configuration.

```bash
cd ~/piper
git pull --ff-only
source ~/piper/scripts/activate_lerobot_v060.sh
~/piper/scripts/install.sh ~/lerobot_v060
```

## Common issues

- `no connected gs_usb CAN adapters`: check `sudo modprobe gs_usb`, USB cabling, and `ip -br link`.
- `REPLACE_WITH_..._SERIAL`: update local YAML using `lerobot-find-cameras realsense`.
- `torch.cuda.is_available() == False`: install the NVIDIA driver and PyTorch wheel for that machine.
- Commands missing in a new terminal: source `~/piper/scripts/activate_lerobot_v060.sh` again.

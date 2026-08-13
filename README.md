# LeRobot Piper v0.6.0

**English** | [한국어](README.ko.md)

AgileX Piper leader/follower teleoperation and synchronized dual-RealSense
dataset recording for [LeRobot 0.6.0](https://github.com/huggingface/lerobot/tree/v0.6.0).

The integration uses the official
[pyAgxArm](https://github.com/agilexrobotics/pyAgxArm) API as a pinned dependency.
It does not vendor or modify the SDK.

## Highlights

- Stable leader/follower SocketCAN names bound to USB adapter serials
- Piper leader activation compatible with the proven LeRobot 0.4.3 startup flow
- 200 Hz leader-to-follower control independent of 30 Hz dataset capture
- D435 + D405 recording with non-blocking latest-frame sampling
- 30 Hz absolute-deadline dataset scheduler and capture-rate summary
- 10 Hz compressed Rerun preview without reducing recorded image quality
- Color-coded terminal phases and a compact teleoperation dashboard
- YAML recording configuration with small CLI overrides

## Safety

The follower torque is enabled during connection and the arm can move abruptly.
Support the leader, align both arm poses, clear both workspaces, and prepare the
emergency stop before starting teleoperation or recording.

The default 200 Hz control, 100% speed, and relative target cap of 100 are
aggressive. Reduce them for initial validation.

## Layout

```text
configs/                 Local CAN and recording configuration templates
lerobot_plugins/         Piper robot, teleoperator, CAN bus, and record CLI
patches/                 Minimal compatibility patch for LeRobot v0.6.0
scripts/                 Installation, CAN setup, teleop, and environment helpers
```

Local hardware identities and datasets are ignored by Git. The repository does
not include CAN dumps, USB serial mappings, camera serials, or recorded data.

## Requirements

- Ubuntu with SocketCAN and the `gs_usb` driver
- Python 3.12 environment containing LeRobot 0.6.0
- Two Piper arms and two USB-CAN adapters
- Intel RealSense cameras supported by `pyrealsense2`
- `iproute2`, `udev`, Git, and FFmpeg

## Environment strategy

Native Conda installation is recommended for a hardware workstation. It keeps
SocketCAN, USB permissions, udev, RealSense, and NVIDIA drivers visible without
an additional container boundary while isolating Python packages.

Docker is useful when the complete user-space stack must be reproduced on
multiple machines, but it does **not** package the host kernel, `gs_usb`, udev
rules, USB permissions, or the NVIDIA driver. A Piper container would still
need a compatible Ubuntu/JetPack host, host-side CAN initialization,
`--network=host` for SocketCAN, USB device passthrough for RealSense, and the
NVIDIA Container Toolkit when a GPU is used. For one robot workstation, start
with Conda; add a platform-specific Docker image when the environment must be
deployed repeatedly.

This repository pins LeRobot to `v0.6.0` and pyAgxArm to a specific Git commit.
Compatible transitive pip packages are not yet locked byte-for-byte, so the
current native setup is version-bounded rather than a complete environment
snapshot.

## Install (native Conda, recommended)

### 1. Prepare the Ubuntu host

```bash
sudo apt update
sudo apt install -y git git-lfs can-utils iproute2 udev ffmpeg
sudo modprobe gs_usb
git lfs install
```

The CAN interfaces and RealSense cameras must be visible on the host before
debugging Python. The `gs_usb` module and device permissions are host settings;
they are not installed into a Conda environment.

### 2. Create the Python environment

Install [Miniforge](https://github.com/conda-forge/miniforge) if `conda` is not
available, then create the environment:

```bash
conda create -y -n lerobot_v060 python=3.12
conda activate lerobot_v060
conda install -y -c conda-forge ffmpeg=7.1.1
python -m pip install --upgrade pip
```

### 3. Install LeRobot v0.6.0 from source

Clone LeRobot at the supported version and this repository alongside it:

```bash
git clone --branch v0.6.0 https://github.com/huggingface/lerobot.git ~/lerobot_v060
cd ~/lerobot_v060
python -m pip install -e ".[core_scripts,intelrealsense]"
```

The `core_scripts` extra installs dataset recording, hardware, and Rerun
dependencies. The `intelrealsense` extra installs `pyrealsense2`. If no
`pyrealsense2` wheel exists for the target Python/aarch64 platform, build or
obtain a matching wheel first and install it into this same environment.

### 4. Apply the Piper integration

```bash
git clone https://github.com/leejaehot/lerobot-piper-v060.git ~/piper

~/piper/scripts/install.sh ~/lerobot_v060
source ~/piper/scripts/activate_lerobot_v060.sh
```

`install.sh` applies [the small LeRobot patch](patches/lerobot-v0.6.0-piper.patch)
to the local editable LeRobot checkout, installs this repository's
`lerobot_robot_piper` distribution in editable mode, and creates a local
recording configuration from the example when needed. Re-running it is safe
when the patch is already present.

LeRobot discovers installed distributions whose names begin with
`lerobot_robot_` and imports them automatically. Importing
`lerobot_robot_piper` registers both `piper_follower` and `piper_leader`, so the
plugin source does not need to be copied into `src/lerobot`. Because both
projects are editable installs, local source changes take effect immediately.

Source the activation helper in every new terminal:

```bash
source ~/piper/scripts/activate_lerobot_v060.sh
```

It activates `lerobot_v060`, adds the Piper scripts to `PATH`, loads the local
CAN mapping, selects a separate LeRobot data directory, and changes to the
LeRobot checkout. Verify the installation without moving either arm:

```bash
python -c 'import lerobot_robot_piper, lerobot_teleoperator_piper; print("Piper plugins: OK")'
python -m pip show lerobot lerobot_robot_piper pyAgxArm pyrealsense2
piper_teleop --help
piper_record --help
```

Expected locations are the editable LeRobot checkout (`~/lerobot_v060`) and
this repository (`~/piper/lerobot_plugins`), not an unrelated global Python
installation.

## Configure CAN roles

Connect both adapters, identify which current interface belongs to each arm,
then save the mapping:

```bash
can_init --status
can_init --configure --leader can4 --follower can5
can_init
```

The generated `configs/can_mapping.env` is intentionally ignored by Git.
Initialization configures 1 Mbit/s CAN and renames the adapters to
`can_leader` and `can_follower`; it does not send robot motion commands.

## Teleoperate

```bash
piper_teleop --init-can
```

Safer initial settings can be supplied through environment variables:

```bash
PIPER_TELEOP_SPEED_PERCENT=20 \
PIPER_TELEOP_MAX_RELATIVE_TARGET=5 \
piper_teleop
```

The full-screen terminal dashboard displays leader target, follower qpos,
joint tracking error, gripper position, flange pose, and control-loop rate.

## Configure recording

Create the local configuration and replace the two RealSense serials:

```bash
cp ~/piper/configs/record.example.yaml ~/piper/configs/record.yaml
lerobot-find-cameras realsense
```

The most frequently edited settings are:

```yaml
dataset:
  repo_id: local/piper_doubleport
  task: Put the object in the target container.
  episodes: 10
  episode_seconds: 60

cameras:
  egoview: "D435_SERIAL"
  wristcam: "D405_SERIAL"
```

Run a five-second validation before collecting a real dataset:

```bash
piper_record --init-can --test
piper_record
```

A successful smoke test at 30 Hz ends with approximately:

```text
CAPTURE        150 frames / 5.00 s = 30.0 Hz (target 30 Hz)
```

Common values can also be overridden without editing YAML:

```bash
piper_record \
  --repo-id local/pepper_to_cup \
  --task "Put the bell pepper in the right cup." \
  --episodes 20 \
  --seconds 60
```

Use `--dry-run` to inspect the effective plan, `--no-rerun` to isolate
visualization overhead, and `NO_COLOR=1` to disable ANSI colors.

## Timing model

The control worker relays leader commands at 200 Hz. Dataset rows are sampled on
a separate 30 Hz absolute schedule. Each RealSense camera continuously captures
in its own background thread; the recorder snapshots the newest frames and then
reads follower state for the same row. Rerun receives a compressed 10 Hz preview
while the stored frames retain their configured capture quality.

This provides software synchronization suitable for standard LeRobot training.
It does not provide hardware-triggered simultaneous exposure between independent
RealSense cameras; their physical capture times can differ by up to one camera
period (about 33 ms at 30 Hz).

## Test

```bash
conda activate lerobot_v060
HF_HOME=/tmp/lerobot-test-hf \
HF_LEROBOT_HOME=/tmp/lerobot-test-data \
pytest -q ~/piper/lerobot_plugins/tests ~/lerobot_v060/tests/test_control_robot.py
```

The current integration test suite passes 16 tests, including high-rate control,
plugin startup flow, camera sampling, terminal UI, recording, and resume paths.

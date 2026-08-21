# LeRobot Piper v0.6.0

[English](README.md) | **한국어**

[LeRobot v0.6.0](https://github.com/huggingface/lerobot/tree/v0.6.0) 기반
AgileX Piper leader/follower teleoperation, 시각화, 데이터 취득, policy rollout
도구입니다.

> [!CAUTION]
> 연결 시 follower 토크가 켜지며 로봇팔이 움직일 수 있습니다. 두 팔을 지지하고,
> 작업 공간을 비운 뒤, 비상 정지를 준비한 상태에서 시작하세요.

## 명령어

| 명령 | 용도 | 로봇 연결 |
| --- | --- | --- |
| `piper_teleop` | leader → follower teleoperation | leader + follower |
| `piper_vis` | RealSense 및 reset grid 확인 | 카메라만 |
| `piper_record` | LeRobot dataset 취득 | leader + follower + 카메라 |
| `piper_rollout` | ACT / Diffusion Policy 실행 | follower + 카메라 |
| `piper_replay` | dataset 영상/action 확인, 선택적 hardware replay | 기본값은 없음 |

## 환경

- Ubuntu 24.04 (`x86_64` 또는 `aarch64`)
- Python 3.12
- LeRobot `v0.6.0`
- [pyAgxArm](https://github.com/agilexrobotics/pyAgxArm) `799b841`
- Piper 2대, `gs_usb` USB-CAN 2개
- Intel RealSense D435 + D405
- Rollout에는 NVIDIA CUDA 환경을 권장합니다.

> [!NOTE]
> 실제 장비 검증 환경은 **NVIDIA Jetson AGX Thor** (`aarch64`), Ubuntu 24.04.4,
> Jetson Linux R38.2.2, Python 3.12.13, PyTorch 2.11.0+cu130입니다. 일반
> `x86_64` Ubuntu도 같은 LeRobot commit과 Python constraints를 사용하며,
> PyTorch·TorchVision·RealSense wheel만 장비의 architecture/CUDA에 맞게 설치합니다.

## QuickStart

### 1. Ubuntu

```bash
sudo apt update
sudo apt install -y git git-lfs can-utils iproute2 udev ffmpeg
sudo modprobe gs_usb
git lfs install
```

Python을 설치하기 전에 CAN adapter와 RealSense가 host에서 인식되어야 합니다.

```bash
ip -br link
lsusb
```

### 2. Python 환경

[Miniforge](https://github.com/conda-forge/miniforge)를 설치한 뒤:

```bash
git clone \
  https://github.com/leejaehot/lerobot-piper-v060.git ~/piper
conda env create -f ~/piper/environment.yml
conda activate lerobot_v060
```

`constraints.txt`는 두 architecture에서 공유하는 Python package 버전을 고정합니다.
PyTorch, TorchVision, TorchCodec, `pyrealsense2`는 플랫폼별 binary가 달라 제외되어
있습니다.

### 3. LeRobot 설치

LeRobot 저장소는 Piper 저장소와 같은 상위 디렉터리에 둡니다.

```bash
git clone --branch v0.6.0 --depth 1 \
  https://github.com/huggingface/lerobot.git ~/lerobot_v060

cd ~/lerobot_v060
python -m pip install -c ~/piper/constraints.txt \
  -e ".[core_scripts,intelrealsense,diffusion]"

~/piper/scripts/install.sh ~/lerobot_v060
source ~/piper/scripts/activate_lerobot_v060.sh
```

`install.sh`는 LeRobot v0.6.0 호환 patch를 적용하고 Piper plugin을 editable로
설치합니다. `teleop.yaml`, `record.yaml`, `rollout.yaml`, `replay.yaml`도 처음 한
번만 생성하며 선택한 LeRobot 경로는 Git에서 제외되는 `configs/local.env`에
저장합니다.

일반 Ubuntu는 위 명령으로 architecture에 맞는 wheel을 설치합니다. Jetson은 위
명령 전에 사용 중인 JetPack/CUDA에 맞는 NVIDIA PyTorch·TorchVision과
`pyrealsense2` wheel을 설치해야 합니다.

새 terminal에서는 항상 다음 한 줄로 시작합니다.

```bash
source ~/piper/scripts/activate_lerobot_v060.sh
```

### 4. 설치 확인

아래 명령은 CAN이나 모터에 연결하지 않습니다.

```bash
python -c 'import lerobot_robot_piper, lerobot_teleoperator_piper; print("Piper plugins: OK")'
piper_teleop --help
piper_vis --help
piper_record --help
piper_rollout --help
piper_replay --help
```

Rollout을 사용할 장비에서는 CUDA도 확인합니다.

```bash
python -c 'import platform, torch; print(platform.machine(), torch.__version__, torch.cuda.is_available())'
```

## 장비 설정

### 1. CAN

두 USB-CAN adapter를 연결합니다. 한쪽씩 분리해 leader/follower의 현재 interface
이름을 확인한 뒤 역할을 저장합니다.

```bash
can_init --status
can_init --configure --leader can4 --follower can5
can_init
can_init --status
```

이후 USB 순서가 바뀌어도 serial 기준으로 `can_leader`, `can_follower`가 복원됩니다.
장비별 mapping은 Git에서 제외된 `configs/can_mapping.env`에 저장됩니다.

### 2. 카메라

```bash
lerobot-find-cameras realsense
```

출력된 serial을 다음 파일에 입력합니다.

| 파일 | 수정할 항목 |
| --- | --- |
| `~/piper/configs/teleop.yaml` | `cameras.egoview` |
| `~/piper/configs/record.yaml` | `cameras.egoview`, `cameras.wristcam`, dataset 정보 |
| `~/piper/configs/rollout.yaml` | 두 camera serial, checkpoint 공통 설정 |

카메라와 grid만 먼저 확인합니다. 이 명령은 CAN과 두 팔을 연결하지 않습니다.

```bash
piper_vis
```

## 사용법

### Teleop

첫 동작은 저속으로 확인합니다.

```bash
PIPER_TELEOP_SPEED_PERCENT=20 \
PIPER_TELEOP_MAX_RELATIVE_TARGET=5 \
piper_teleop --init-can --no-rerun
```

자세와 방향이 정상임을 확인한 뒤 기본 설정으로 실행합니다.

```bash
piper_teleop --init-can
```

### Camera preview

```bash
piper_vis
piper_vis --config ~/piper/configs/teleop.yaml
```

`q` 또는 `Esc`로 종료합니다.

### Record

먼저 5초짜리 local smoke test를 실행합니다.

```bash
piper_record --dry-run
piper_record --init-can --test
piper_record
```

자주 바꾸는 값은 CLI로 덮어쓸 수 있습니다.

```bash
piper_record \
  --repo-id local/pick_and_place \
  --task "Pick the object and place it in the container." \
  --episodes 20 \
  --seconds 60
```

| 입력 | 동작 |
| --- | --- |
| `Right` / `n` | 현재 episode 저장 후 다음 단계 |
| `Left` / `r` | 현재 episode 폐기 후 재녹화 |
| `Space` | 다음 frame부터 새 segment |
| `Esc` / `q` | 전체 종료 |

### Rollout

Checkpoint별 설정은 `configs/rollouts/*.yaml` profile로 관리합니다.

```bash
piper_rollout --list-profiles
cp ~/piper/configs/rollout-profile.example.yaml \
  ~/piper/configs/rollouts/my_act.yaml
```

`my_act.yaml`의 checkpoint, task, 필요 시 camera alias와 시작 pose를 수정합니다.
검증 순서는 항상 다음과 같습니다.

```bash
# 파일과 설정만 검사
piper_rollout my_act --dry-run

# CUDA에서 synthetic input으로 추론 1회; hardware 연결 없음
piper_rollout my_act --check

# 실제 follower에서 30초 bounded rollout
piper_rollout my_act --init-can
```

실제 rollout은 policy 종류와 관계없이 먼저 camera-only `piper_vis` reset
grid를 엽니다. 물체 배치를 확인한 뒤 `Enter`를 누르면 CAN 초기화와 rollout을
계속하고, `q`/`Esc`를 누르면 모터를 활성화하지 않고 취소합니다. 이 단계는
`--dry-run`과 `--check`에서는 실행되지 않습니다.

### Replay

Dataset을 먼저 영상과 action trace로 확인합니다.

```bash
piper_replay --list-profiles
cp ~/piper/configs/replay-profile.example.yaml \
  ~/piper/configs/replays/my_dataset.yaml
piper_replay my_dataset --episode 0
```

`my_dataset.yaml`의 `repo_id`와 `root`를 먼저 수정합니다. 실제 follower에
action을 보내려면 `--hardware`를 명시해야 합니다.

```bash
piper_replay my_dataset --episode 0 --hardware --dry-run
piper_replay my_dataset --episode 0 --hardware --init-can
```

## 설정 파일

- 공유 기본값 및 profile template: `configs/*.example.yaml`
- 장비별 값: `configs/teleop.yaml`, `record.yaml`, `rollout.yaml`, `replay.yaml`
- Checkpoint별 rollout: `configs/rollouts/*.yaml`
- Dataset별 replay: `configs/replays/*.yaml`

장비별 YAML과 CAN mapping은 Git에서 제외됩니다. 새 장비의 값은 example 파일이
아닌 local YAML에 기록합니다.

## 안전

- 최초 teleop은 speed 20%, relative target 5 이하에서 확인합니다.
- Rollout은 `--dry-run` → `--check` → 실제 실행 순서를 지킵니다.
- `Ctrl-C` 후에는 두 팔, 특히 follower를 지지하고 Enter를 눌러 torque를 해제합니다.
- `piper_vis`, `--help`, rollout `--dry-run`/`--check`는 모터를 움직이지 않습니다.

## 업데이트

로컬 장비 설정은 유지한 채 integration만 업데이트합니다.

```bash
cd ~/piper
git pull --ff-only
source ~/piper/scripts/activate_lerobot_v060.sh
~/piper/scripts/install.sh ~/lerobot_v060
```

## 문제 해결

- `no connected gs_usb CAN adapters`: `sudo modprobe gs_usb`, USB 연결, `ip -br link`를 확인합니다.
- `REPLACE_WITH_..._SERIAL`: `lerobot-find-cameras realsense` 결과로 local YAML을 수정합니다.
- `torch.cuda.is_available() == False`: 해당 장비의 NVIDIA driver와 PyTorch wheel을 먼저 맞춥니다.
- 새 terminal에서 명령을 찾지 못함: `source ~/piper/scripts/activate_lerobot_v060.sh`를 다시 실행합니다.

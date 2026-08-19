# LeRobot Piper v0.6.0

[English](README.md) | **한국어**

[LeRobot 0.6.0](https://github.com/huggingface/lerobot/tree/v0.6.0)을 위한
AgileX Piper 리더/팔로워 텔레오퍼레이션 및 듀얼 RealSense 동기화 데이터셋
취득 도구입니다.

이 통합 패키지는 공식 [pyAgxArm](https://github.com/agilexrobotics/pyAgxArm)
API를 고정 버전 의존성으로 사용합니다. SDK 코드를 저장소에 복사하거나
수정하지 않습니다.

## 주요 기능

- USB 어댑터 시리얼 번호에 바인딩된 안정적인 리더/팔로워 SocketCAN 이름
- 검증된 LeRobot 0.4.3 시작 순서와 호환되는 Piper 리더 활성화
- 30 Hz 데이터 취득과 독립적으로 동작하는 200 Hz 리더-팔로워 제어
- 최신 프레임을 논블로킹으로 샘플링하는 D435 + D405 녹화
- 절대 시간 기준 30 Hz 데이터셋 스케줄러 및 취득 속도 요약
- 저장 이미지 품질을 낮추지 않는 10 Hz 압축 Rerun 미리보기
- 단계별 색상 로그와 간결한 텔레오퍼레이션 대시보드
- 작은 CLI 인자로 덮어쓸 수 있는 YAML 녹화 설정

## 안전 주의사항

연결 과정에서 팔로워 모터의 토크가 활성화되며 로봇팔이 갑자기 움직일 수
있습니다. 텔레오퍼레이션 또는 데이터 취득 전에 리더를 지지하고, 두 로봇팔의
자세를 맞춘 뒤, 양쪽 작업 공간을 비우고 비상 정지를 준비하세요.

기본값인 200 Hz 제어, 100% 속도, 상대 목표 제한 100은 공격적인 설정입니다.
최초 검증 시에는 값을 낮춰 사용하세요.

## 저장소 구조

```text
configs/                 로컬 CAN 및 데이터 취득 설정 템플릿
lerobot_plugins/         Piper 로봇, 텔레오퍼레이터, CAN 버스, record CLI
patches/                 LeRobot v0.6.0용 최소 호환성 패치
scripts/                 설치, CAN 설정, teleop 및 환경 스크립트
```

로컬 하드웨어 식별 정보와 데이터셋은 Git에서 제외됩니다. 이 저장소에는 CAN
덤프, USB 시리얼 매핑, 카메라 시리얼 번호 또는 취득한 데이터가 포함되지
않습니다.

## 요구 사항

- SocketCAN 및 `gs_usb` 드라이버를 지원하는 Ubuntu
- LeRobot 0.6.0이 설치된 Python 3.12 환경
- Piper 로봇팔 2대와 USB-CAN 어댑터 2개
- `pyrealsense2`에서 지원하는 Intel RealSense 카메라
- `iproute2`, `udev`, Git, FFmpeg

## 환경 구성 전략

로봇 하드웨어 워크스테이션에는 네이티브 Conda 설치를 권장합니다. Python
패키지는 격리하면서도 SocketCAN, USB 권한, udev, RealSense 및 NVIDIA
드라이버를 별도의 컨테이너 경계 없이 사용할 수 있기 때문입니다.

여러 장비에서 사용자 공간 전체를 동일하게 복원해야 한다면 Docker가 유용하지만,
Docker는 호스트 커널, `gs_usb`, udev 규칙, USB 권한 또는 NVIDIA 드라이버를
포함하지 **않습니다**. Piper 컨테이너를 사용하더라도 호환되는 Ubuntu/JetPack
호스트, 호스트에서의 CAN 초기화, SocketCAN 접근을 위한 `--network=host`,
RealSense USB 장치 전달, GPU 사용 시 NVIDIA Container Toolkit이 필요합니다.
한 대의 로봇 워크스테이션에서는 Conda로 시작하고, 같은 환경을 반복 배포해야 할
때 플랫폼 전용 Docker 이미지를 추가하는 편이 적합합니다.

이 저장소는 LeRobot을 `v0.6.0`으로, pyAgxArm을 특정 Git 커밋으로
고정합니다. 다만 하위 pip 의존성 전체를 바이트 단위로 잠그지는 않았으므로 현재
네이티브 설치는 완전한 환경 스냅샷이 아니라 호환 버전 범위를 고정한 구성입니다.

## 설치(권장: 네이티브 Conda)

### 1. Ubuntu 호스트 준비

```bash
sudo apt update
sudo apt install -y git git-lfs can-utils iproute2 udev ffmpeg
sudo modprobe gs_usb
git lfs install
```

Python을 디버깅하기 전에 호스트에서 CAN 인터페이스와 RealSense 카메라가 보여야
합니다. `gs_usb` 모듈과 장치 권한은 호스트 설정이며 Conda 환경에 설치되는
항목이 아닙니다.

### 2. Python 환경 생성

`conda`가 없다면 [Miniforge](https://github.com/conda-forge/miniforge)를
설치한 뒤 환경을 생성합니다.

```bash
conda create -y -n lerobot_v060 python=3.12
conda activate lerobot_v060
conda install -y -c conda-forge ffmpeg=7.1.1
python -m pip install --upgrade pip
```

### 3. LeRobot v0.6.0 소스 설치

지원되는 버전의 LeRobot과 이 저장소를 나란히 클론합니다.

```bash
git clone --branch v0.6.0 https://github.com/huggingface/lerobot.git ~/lerobot_v060
cd ~/lerobot_v060
python -m pip install -e ".[core_scripts,intelrealsense]"
```

`core_scripts` extra는 데이터셋 취득, 하드웨어 및 Rerun 의존성을 설치하고,
`intelrealsense` extra는 `pyrealsense2`를 설치합니다. 대상 Python/aarch64
플랫폼용 `pyrealsense2` wheel이 없다면 먼저 호환 wheel을 빌드하거나 준비하여
동일한 환경에 설치해야 합니다.

### 4. Piper 통합 적용

```bash
git clone https://github.com/leejaehot/lerobot-piper-v060.git ~/piper

~/piper/scripts/install.sh ~/lerobot_v060
source ~/piper/scripts/activate_lerobot_v060.sh
```

`install.sh`는 [작은 LeRobot 패치](patches/lerobot-v0.6.0-piper.patch)를
로컬 editable LeRobot 체크아웃에 적용하고, 이 저장소의
`lerobot_robot_piper` 배포 패키지를 editable 모드로 설치하며, 필요한 경우
예제에서 로컬 teleop, 녹화, rollout 설정을 생성합니다. 패치가 이미 적용된 상태에서
다시 실행해도 안전합니다.

LeRobot은 이름이 `lerobot_robot_`으로 시작하는 설치된 배포 패키지를 찾아
자동으로 import합니다. `lerobot_robot_piper`를 import하면
`piper_follower`와 `piper_leader`가 모두 등록되므로 플러그인 소스를
`src/lerobot` 안으로 복사할 필요가 없습니다. 두 프로젝트 모두 editable
설치이므로 로컬 소스 수정 사항이 즉시 반영됩니다.

새 터미널을 열 때마다 활성화 스크립트를 source합니다.

```bash
source ~/piper/scripts/activate_lerobot_v060.sh
```

이 스크립트는 `lerobot_v060` 환경을 활성화하고 Piper 스크립트를 `PATH`에
추가하며, 로컬 CAN 매핑과 별도의 LeRobot 데이터 경로를 불러온 뒤 LeRobot
체크아웃으로 이동합니다. 로봇팔을 움직이지 않고 설치를 검증하려면 다음을
실행합니다.

```bash
python -c 'import lerobot_robot_piper, lerobot_teleoperator_piper; print("Piper plugins: OK")'
python -m pip show lerobot lerobot_robot_piper pyAgxArm pyrealsense2
piper_teleop --help
piper_record --help
```

출력 경로는 관련 없는 전역 Python 설치가 아니라 editable LeRobot 체크아웃
(`~/lerobot_v060`)과 이 저장소(`~/piper/lerobot_plugins`)를 가리켜야 합니다.

## CAN 역할 설정

두 어댑터를 연결하고 현재 인터페이스가 어느 로봇팔에 연결되었는지 확인한 뒤
매핑을 저장합니다.

```bash
can_init --status
can_init --configure --leader can4 --follower can5
can_init
```

생성되는 `configs/can_mapping.env`는 의도적으로 Git에서 제외됩니다. 초기화
과정은 CAN을 1 Mbit/s로 설정하고 어댑터 이름을 `can_leader`와
`can_follower`로 변경하며, 로봇 동작 명령은 전송하지 않습니다.

## 텔레오퍼레이션

```bash
piper_teleop --init-can
```

Teleop 기본값은 독립된 `configs/teleop.yaml`에서 읽습니다. 이 파일에 CAN 역할,
제어·모니터링 주기, follower/gripper 제한, 오디오, egoview 카메라, reset grid,
객체별 고정 pose가 들어갑니다. 다른 파일은 다음처럼 지정합니다.

```bash
piper_teleop --config ~/piper/configs/teleop.yaml
```

우선순위는 CLI, 환경 변수, YAML 순서입니다. 최초 테스트에서는 기존 환경 변수로
더 안전한 값을 일시 지정할 수도 있습니다.

```bash
PIPER_TELEOP_SPEED_PERCENT=20 \
PIPER_TELEOP_MAX_RELATIVE_TARGET=5 \
piper_teleop
```

전체 화면 터미널 대시보드에는 리더 목표값, 팔로워 qpos, 관절 추종 오차,
그리퍼 위치, 플랜지 자세 및 제어 루프 속도가 표시됩니다.

기본적으로 teleop 실행 시 카메라 전용 Rerun 화면도 함께 열립니다.
`configs/teleop.yaml`의 egoview 카메라, 16×12 reset grid, 객체별 고정 initial
pose를 읽고, 로봇팔의 200 Hz 제어 루프와 독립적으로 raw 영상을 10 Hz로
갱신합니다. 이 기능은 시각화 전용이며 teleop 중 reset annotation을 저장하지
않습니다.

```bash
PIPER_TELEOP_RERUN_FPS=15 piper_teleop
piper_teleop --no-rerun
```

기본 config 경로는 `PIPER_TELEOP_CONFIG`로 바꿀 수 있습니다. Teleop은 기본적으로
준비, safe-disconnect, disconnect 상태를 짧은 로컬 한국어 WAV로 안내합니다.
`audio.enabled: false` 또는 `PIPER_TELEOP_SOUNDS=false piper_teleop`으로 끕니다.

리더 그리퍼는 기본 teaching friction `5`를 사용합니다. 이 장비에서는 손으로
조작하는 힘과 손을 놓았을 때의 위치 유지력을 절충한 값입니다. 팔로워
그리퍼에만 `80 mm/s` 속도 제한이 적용되어 리더 입력이 급격해도 부드럽게
따라가며 팔 관절의 200 Hz 제어에는 영향을 주지 않습니다. 필요하면 다음
환경 변수로 조정합니다.

```bash
PIPER_LEADER_GRIPPER_FRICTION=4 \
PIPER_TELEOP_GRIPPER_SPEED_MM_S=60 \
piper_teleop
```

friction 값은 `1~10`이며 높을수록 손힘은 줄지만 자연스럽게 벌어질 가능성이
커집니다. 이 장비에서는 `4~6` 범위를 권장합니다.

`Ctrl-C`로 종료하면 토크를 바로 끄지 않고 safe-disconnect 안내를 표시합니다.
두 팔, 특히 follower를 손으로 지지한 뒤 Enter를 눌러야 토크가 해제되고 CAN이
disconnect됩니다. 파이프나 cron처럼 stdin이 터미널이 아닌 실행에서는 확인을
기다릴 수 없으므로 경고 후 기존처럼 종료합니다.

## 학습된 policy rollout

`piper_rollout`은 LeRobot의 실로봇 `lerobot-rollout` 엔진을 Piper용 안전 설정과
함께 실행합니다. 설치 시 `configs/rollout.example.yaml`을 로컬 전용
`configs/rollout.yaml`로 복사합니다. 이 파일에 장비의 두 RealSense 시리얼과
`~/lerobot_v060/outputs/policies` 아래 ACT/DP 체크포인트 경로를 입력합니다. 먼저
로봇을 연결하지 않는 검증을 실행합니다.

```bash
piper_rollout act --dry-run
piper_rollout act --check
piper_rollout dp --check
```

`--check`는 CUDA에서 모델, safetensors, 저장된 normalizer/unnormalizer를 불러오고
검은 합성 영상으로 추론 한 번을 실행하지만 CAN, 카메라, 모터에는 연결하지 않습니다.
검증 후 작업 공간을 비우고 비상 정지 버튼을 잡은 상태에서 제한된 30초 rollout을
시작합니다.

```bash
piper_rollout act --init-can
piper_rollout dp --init-can
```

초기값은 학습 데이터와 같은 30 Hz이며 follower 속도 30%, tick당 정규화 관절 변화
`5`, gripper `40 mm/s`로 제한됩니다. 최초 rollout에서는 성능 영향을 피하려고
Rerun을 끄며 `--rerun`으로 켤 수 있습니다. Policy 제어 전에는 follower를 데이터
취득의 `piper_teleop` 시작 자세(official home 관절 + 닫힌 gripper)로 20% 속도에서
정렬하고, 허용 오차 안에 3회 연속 들어온 뒤 rollout을 시작합니다. 이 자동 정렬을
생략하려면 `--no-align-start`, 정렬 속도와 timeout을 바꾸려면 각각 `--align-speed`,
`--align-timeout`을 사용합니다. 자세와 허용 오차는 `configs/rollout.yaml`의
`teleop_initial_pose`에서 수정할 수 있습니다. `Ctrl-C` 또는 제한 시간 종료 시 정렬된
시작 pose로 부드럽게 복귀한 다음, follower를 지지하고 Enter를 눌러야 토크가 해제됩니다.
경로, task, 카메라 또는 제한값은 `configs/rollout.yaml`에서 바꾸거나 CLI 옵션으로
덮어쓸 수 있습니다. 비대화형 실행은 우발적인 시작을 막기 위해 명시적인 `--yes`가
없으면 거부됩니다.

## 데이터 취득 설정

로컬 설정 파일을 만들고 두 RealSense 시리얼 번호를 실제 값으로 교체합니다.

```bash
cp ~/piper/configs/record.example.yaml ~/piper/configs/record.yaml
lerobot-find-cameras realsense
```

가장 자주 수정하는 항목은 다음과 같습니다.

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

실제 데이터셋을 취득하기 전에 5초 검증을 실행합니다.

```bash
piper_record --init-can --test
piper_record
```

recording에서도 동일한 safe-disconnect Enter 확인과 그리퍼 설정이 적용됩니다.
로컬 `configs/record.yaml`의 `arm` 항목에서 다음 값을 조정할 수 있습니다.

```yaml
initial_setup:
  wait_for_enter: true

arm:
  gripper_speed_mm_s: 80
  leader_gripper_friction: 5
  home_on_reset: true
  home_speed_percent: 20
  home_tolerance_degrees: 2

audio:
  enabled: true
```

각 take가 끝나거나 폐기된 뒤에는 다음 episode 전에 follower만 Piper 물리
zero/home으로 복귀시킵니다. Follower는 driver의 일반 `move_j()` 경로로
`home_speed_percent` 속도에서 이동합니다. Leader firmware는 조회하지 않고,
leader home 명령도 보내지 않으며, leader가 zero인지도 성공 조건에 넣지 않습니다.

`dataset.reset_seconds`에는 home 복귀, 환경 재배치, 마지막 음성 countdown이 모두
포함됩니다. 마지막 3초는 arm이 움직이지 않는 구간으로 예약합니다. Follower가
`home_tolerance_degrees` 안에 연속 3회 들어와야 복귀 완료로 판단하며, 제한 시간
안에 도달하지 못하면 잘못 정렬된 상태로 다음 episode를 시작하지 않고 recording을
중단합니다. Episode 1 전의 최초 reset에서는 arm을 움직이지 않습니다.
`--no-home-on-reset`을 사용하면 follower 복귀도 끌 수 있습니다.

첫 episode 전에는 reset 화면을 계속 띄운 채 Enter 입력을 기다립니다. Rerun의
10 Hz grid를 보면서 모든 객체를 배치하고 recording 터미널에 포커스를 둔 뒤
Enter를 누르면 됩니다. 대기 중에도 leader→follower 제어는 200 Hz로 유지되며,
Enter 이후 `Three`, `Two`, `One` 전용 countdown이 끝나야 녹화가 시작됩니다.
초기 배치 단계의 Right/Left 발판은 확인 입력으로 처리하지 않습니다. 비대화형
취득에서는 `initial_setup.wait_for_enter: false` 또는
`piper_record --no-wait-for-enter`로 끌 수 있습니다.

동적 에피소드 번호는 또렷한 영어 neural voice로 `Recording episode one`부터
`Recording episode fifty`까지 안내합니다. 나머지는 `키프레임 1`부터
`키프레임 10`, `환경 초기화`, `재녹화`, `취득 종료` 같은 짧은 한국어 상태어를
사용합니다.
환경 초기화의 마지막 3초에는 영어 `Three`, `Two`, `One`을 1초 간격으로 재생한
뒤 다음 `Recording episode ...` 안내로 넘어갑니다. Reset이 조기에 끝나면 남은
countdown도 즉시 취소됩니다. `키프레임`은
Space 발판 입력이 실제 recording frame의
segment boundary로 반영됐을 때만 재생되며, 디바운스로 무시되거나 recording 밖에서
누른 입력에는 재생되지 않습니다. 미리 생성된 PCM WAV를 시스템 기본 출력으로 non-blocking
재생하므로 recording 중에는 Speech Dispatcher나 온라인 TTS를 사용하지 않습니다.
YAML 설정은 `piper_record --sounds` 또는 `piper_record --no-sounds`로 덮어쓸 수
있습니다. 더 많은 에피소드 번호가 필요하면 다음처럼 다시 생성합니다.

```bash
uv run scripts/generate_voice_assets.py --max-episodes 50 --max-keyframes 10
```

### Egocentric reset 위치 grid

`piper_record`는 Rerun의 egoview 위에 16 × 12 tabletop grid와 설정된 객체별 고정
initial pose를 10 Hz로 투영합니다. Grid는 별도의 Rerun entity이므로 취득되는 원본
카메라 영상에는 그려지지 않습니다. 모든 take에 동일한 설정 좌표를 표시하며,
주변 좌표를 생성·추천하거나 순서를 섞지 않습니다.

작업대의 네 모서리를 좌상단, 우상단, 우하단, 좌하단 순서의 정규화된 영상
좌표로 설정합니다.

```yaml
reset_grid:
  enabled: true
  camera: egoview
  columns: 16
  rows: 12
  corners:
    - [0.12, 0.24]
    - [0.88, 0.24]
    - [0.95, 0.90]
    - [0.05, 0.90]
  initial_poses:
    spam_can: [4, 6]
    white_container: [[2, 2], [7, 2], [2, 9], [7, 9]]
```

현재 로컬 기본값의 화면 전체 네 모서리는 overlay 확인용입니다. 최종 데이터
취득 전에는 반드시 실제 작업대/유효 작업영역의 네 모서리로 교체해야 합니다.
Homography를 적용하므로 grid 간격은 영상 픽셀 간격이 아니라 평면 작업영역에서
균등하게 배치됩니다. Initial pose는 0-based `[column, row]` 좌표입니다. 따라서
16 × 12 grid의 column 범위는 `0..15`, row 범위는 `0..11`입니다.
좌표 하나는 점으로 표시합니다. 축에 정렬된 네 모서리 좌표는 label이 있는
박스로 표시하며 입력 순서는 상관없습니다. YAML에서는 `{...}`가 아니라 위와
같은 중첩 list 문법을 사용합니다.

모든 frame에는 동일한 fixed-pose vector가 저장됩니다. 점은 객체 key를 annotation
이름으로 사용합니다. 박스의 네 모서리는 TL, TR, BR, BL 순서로 정규화되어
`object.corner_1`부터 `object.corner_4`까지 저장됩니다.

```text
annotation.reset.initial_pose.position_id
annotation.reset.initial_pose.grid_col
annotation.reset.initial_pose.grid_row
annotation.reset.initial_pose.x_norm
annotation.reset.initial_pose.y_norm
```

`--no-grid`를 사용하면 grid 안내와 reset annotation 열을 함께 끌 수 있습니다.

CAN이나 두 팔을 enable하지 않고 grid만 확인하거나 조정하려면 다음을 실행합니다.

```bash
piper_vis
```

이 명령은 설정된 egoview 카메라 한 대만 연결하고, 관절 plot이 없는 camera-only
Rerun layout을 만듭니다. 설정된 객체 pose를 동일한 10 Hz로 표시하며 `q` 또는
`Esc`로 종료합니다. Jetson native Viewer의 안정성을 위해 preview는 항상 raw
image를 사용합니다. CAN, leader, follower에는 연결하지 않습니다. 다른 설정을
확인하려면 `piper_vis --config configs/teleop.yaml`처럼 지정할 수 있습니다.
기존 `piper_grid_preview` 명령도 호환용 alias로 계속 사용할 수 있습니다.

### 3발판 데이터 취득 제어

USB HID 발판이 다음 키보드 키를 출력하도록 매핑되어 있다면 별도 드라이버 없이
사용할 수 있습니다.

| 발판 | HID 키 | 데이터 취득 동작 |
| --- | --- | --- |
| 왼쪽 | 왼쪽 방향키 | 현재 take를 폐기하고 같은 episode를 다시 녹화 |
| 가운데 | Space | 다음 프레임부터 새로운 sub-task segment 시작 |
| 오른쪽 | 오른쪽 방향키 | 현재 take를 저장하고 다음 episode로 진행 |

오른쪽 발판은 현재 진행 단계를 끝냅니다. 녹화 중 한 번 누르면 reset 단계로
들어가며, 물체와 환경을 재배치한 후 다시 누르면 reset timer를 기다리지 않고 다음
episode를 즉시 시작합니다. 왼쪽 발판도 먼저 reset 단계로 들어간 뒤 현재 take를
폐기하고 같은 episode 번호를 다시 사용합니다.

Segment annotation은 기본으로 활성화됩니다. 모든 dataset row에는 dense scalar
`annotation.segment_id` 열이 포함됩니다. Episode는 segment `0`으로 시작하고,
Space를 누를 때마다 그다음 frame부터 모든 후속 frame이 `1`, `2`, `3` 순서의
값을 가집니다. 새로운 값이 처음 나타나는 frame이 정확한 segment 경계입니다.
다음 episode에서는 다시 `0`부터 시작합니다. 발판 입력은 직후의 다음 30 Hz
dataset frame에 적용되며, 독립적인 200 Hz 로봇팔 제어 루프에는 영향을 주지
않습니다. Reset 구간의 입력은 무시되고, 재녹화할 때 해당 take의 segment를
다른 frame과 함께 폐기한 뒤 번호도 다시 `0`부터 시작합니다.

키보드 자동 반복 입력은 설정 가능한 debounce 시간으로 차단합니다.

```yaml
annotations:
  segments: true
  segment_debounce_ms: 400
```

Annotation 열 없이 취득하려면 `piper_record --no-segments`를 사용합니다.
X11에서는 `pynput` listener가 HID 발판을 전역으로 수신합니다. Wayland 또는
headless terminal에서는 TTY fallback이 키를 받을 수 있도록 녹화 terminal에
focus를 유지해야 합니다. 전체 녹화를 중지하는 키는 기존과 동일하게 `Esc` 또는
`q`입니다.

30 Hz로 정상 동작한 스모크 테스트는 다음과 비슷한 결과로 끝납니다.

```text
CAPTURE        150 frames / 5.00 s = 30.0 Hz (target 30 Hz)
```

YAML을 수정하지 않고 자주 쓰는 값을 CLI에서 덮어쓸 수도 있습니다.

```bash
piper_record \
  --repo-id local/pepper_to_cup \
  --task "Put the bell pepper in the right cup." \
  --episodes 20 \
  --seconds 60
```

`--dry-run`은 최종 실행 계획을 출력하고, `--no-rerun`은 시각화 부하를
분리하며, `NO_COLOR=1`은 ANSI 색상을 비활성화합니다.

## 시간 동기화 방식

제어 워커는 리더 명령을 200 Hz로 전달합니다. 데이터셋 행은 별도의 30 Hz
절대 시간 스케줄에 따라 샘플링됩니다. 각 RealSense 카메라는 독립적인 백그라운드
스레드에서 계속 프레임을 취득하고, 레코더는 가장 최신 프레임들을 스냅샷한 뒤
같은 행에 들어갈 팔로워 상태를 읽습니다. 로컬 Rerun에는 Jetson native Viewer의
JPEG decoder 경로를 피하기 위해 raw 10 Hz 미리보기가 전송되며, 저장되는
프레임은 설정된 취득 품질을 유지합니다.

이 방식은 일반적인 LeRobot 학습에 적합한 소프트웨어 동기화를 제공합니다.
다만 독립된 RealSense 카메라 사이의 하드웨어 트리거 기반 동시 노출은 제공하지
않으므로 물리적인 촬영 시점은 최대 한 카메라 주기, 즉 30 Hz에서 약 33 ms까지
차이 날 수 있습니다.

## 테스트

```bash
conda activate lerobot_v060
HF_HOME=/tmp/lerobot-test-hf \
HF_LEROBOT_HOME=/tmp/lerobot-test-data \
pytest -q ~/piper/lerobot_plugins/tests ~/lerobot_v060/tests/test_control_robot.py
```

통합 테스트는 고속 제어, 발판 segment, 플러그인 시작 순서, 카메라 샘플링,
터미널 UI, 공식 home reset, reset 위치 annotation, 데이터 취득 및 이어받기
경로를 검증합니다.

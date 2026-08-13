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

## 설치

지원되는 버전의 LeRobot과 이 저장소를 나란히 클론합니다.

```bash
git clone --branch v0.6.0 https://github.com/huggingface/lerobot.git ~/lerobot_v060
git clone https://github.com/leejaehot/lerobot-piper-v060.git ~/piper

conda activate lerobot_v060
~/piper/scripts/install.sh ~/lerobot_v060
source ~/piper/scripts/activate_lerobot_v060.sh
```

`install.sh`는 [작은 LeRobot 패치](patches/lerobot-v0.6.0-piper.patch)를
적용하고 Piper 패키지를 editable 모드로 설치합니다. 패치가 이미 적용된
상태에서 다시 실행해도 안전합니다.

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

최초 테스트에서는 환경 변수로 더 안전한 값을 지정할 수 있습니다.

```bash
PIPER_TELEOP_SPEED_PERCENT=20 \
PIPER_TELEOP_MAX_RELATIVE_TARGET=5 \
piper_teleop
```

전체 화면 터미널 대시보드에는 리더 목표값, 팔로워 qpos, 관절 추종 오차,
그리퍼 위치, 플랜지 자세 및 제어 루프 속도가 표시됩니다.

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
같은 행에 들어갈 팔로워 상태를 읽습니다. Rerun에는 압축된 10 Hz 미리보기가
전송되며 저장되는 프레임은 설정된 취득 품질을 유지합니다.

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

현재 통합 테스트는 고속 제어, 플러그인 시작 순서, 카메라 샘플링, 터미널 UI,
데이터 취득 및 이어받기 경로를 포함한 16개 테스트를 통과합니다.

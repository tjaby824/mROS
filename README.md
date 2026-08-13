# mROS — Teensy 4.1 실세계 강화학습 스택

Teensy 4.1 펌웨어(`sac3.cpp`)와 PC 트레이너(`sac_trainer_cpp`)를 micro-ROS로 연결해
실세계에서 SAC 학습을 돌리는 프레임워크.

## 어느 폴더가 어디로 가나

클론한 뒤 **대부분은 제자리에서 그대로 쓴다.** 다른 곳으로 복사되는 건 트레이너 하나뿐이고,
그것도 `setup.sh`가 알아서 한다.

| 폴더 | 목적지 | 설명 |
|---|---|---|
| `setup/` | **이동 없음** | 여기서 `./setup.sh` 실행 |
| `firmware/` | **이동 없음** | PlatformIO 프로젝트. VSCode로 **이 폴더**를 연다 |
| `ros2_ws/src/sac_trainer_cpp/` | **→ `~/ros2_ws/src/`** | `setup.sh trainer`가 복사해 colcon 빌드 |
| `examples/` | **이동 없음** | 참고용 서브모듈 (hippo5329). 설치에 불필요 |

`ros2_ws/`라는 이름은 목적지를 그대로 따른 것이다 —
저장소의 `ros2_ws/src/sac_trainer_cpp`가 `~/ros2_ws/src/sac_trainer_cpp`가 된다.

클론 위치는 어디든 상관없다. 펌웨어를 제자리에서 빌드하므로 `~/Documents` 아래일
필요도 없다.

## 구성

```
setup/                            설치 자동화
  setup.sh                        전체 설치 (8단계)
  verify.sh                       설치 확인
  config.env                      IP · 도메인 · 버전 — 여기만 고치면 된다

firmware/                         Teensy 4.1 펌웨어 (PlatformIO)
  platformio.ini                  빌드 설정
  src/sac3.cpp                    micro-ROS 노드 + 커스텀 UDP 트랜스포트
  include/weights_30k.h           학습된 정책 가중치

ros2_ws/src/sac_trainer_cpp/      PC 학습 노드 (LibTorch SAC)
  src/sac_trainer_node.cpp        dual_sac 실행 파일
  src/HiLH.cpp                    hilh 실행 파일
  scripts/run_*.sh                학습 실행 스크립트

examples/                         참고용 예제 (서브모듈, hippo5329)
```

## 시작하기

```bash
# git (권장)
git clone --recurse-submodules https://github.com/tjaby824/mROS.git
cd mROS/setup

# 또는 zip 다운로드 후
unzip mROS-main.zip && cd mROS-main/setup && chmod +x *.sh
```

`config.env`에서 IP를 환경에 맞게 고친 뒤:

```bash
./setup.sh
```

로그아웃 → 재로그인 후:

```bash
./verify.sh
```

전체 절차와 문제 해결은 **[MANUAL.md](MANUAL.md)** 참고.

## 펌웨어만 빌드

```bash
cd firmware
pio run
```

VSCode로 열 때도 저장소 루트가 아니라 `firmware/`를 연다 — PlatformIO는
`platformio.ini`가 있는 폴더만 프로젝트로 인식한다.

## 요구 사항

Ubuntu 24.04 · ROS 2 Jazzy · Teensy 4.1 + 이더넷 킷 · LibTorch 2.4.0 (CPU, cxx11-ABI)

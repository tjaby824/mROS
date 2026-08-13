# mROS — Teensy 4.1 실세계 강화학습 스택

Teensy 4.1 펌웨어(`sac3.cpp`)와 PC 트레이너(`sac_trainer_cpp`)를 micro-ROS로 연결해
실세계에서 SAC 학습을 돌리는 프레임워크.

## 구성

```
Projects/mROS/                          Teensy 4.1 펌웨어 (PlatformIO)
  src/sac3.cpp                          micro-ROS 노드 + 커스텀 UDP 트랜스포트
  include/weights_30k.h                 학습된 정책 가중치
  platformio.ini                        빌드 설정

setup/                                  설치 자동화
  setup.sh                              전체 설치 (ROS 2 · LibTorch · PlatformIO · 에이전트 · 트레이너)
  verify.sh                             설치 확인
  config.env                            IP · 도메인 · 버전 — 여기만 고치면 된다

micro_ros_arduino_examples_platformio/  참고용 예제 (서브모듈, hippo5329)
```

트레이너(`sac_trainer_cpp`)는 별도 저장소인 [`tjaby824/ros2_ws`](https://github.com/tjaby824/ros2_ws)에 있고,
`setup.sh`가 받아서 빌드한다.

## 시작하기

```bash
git clone --recurse-submodules https://github.com/tjaby824/mROS.git
cd mROS/setup
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
cd Projects/mROS
pio run
```

## 요구 사항

Ubuntu 24.04 · ROS 2 Jazzy · Teensy 4.1 + 이더넷 킷 · LibTorch 2.4.0 (CPU, cxx11-ABI)

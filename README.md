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

trainer/sac_trainer_cpp/                PC 학습 노드 (LibTorch SAC)
  src/sac_trainer_node.cpp              dual_sac 실행 파일
  src/HiLH.cpp                          hilh 실행 파일
  scripts/run_*.sh                       학습 실행 스크립트

micro_ros_arduino_examples_platformio/  참고용 예제 (서브모듈, hippo5329)
```

`setup.sh`가 `trainer/sac_trainer_cpp`를 `~/ros2_ws/src/`로 복사해 빌드한다.
별도 저장소를 받을 필요가 없다.

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
cd Projects/mROS
pio run
```

## 요구 사항

Ubuntu 24.04 · ROS 2 Jazzy · Teensy 4.1 + 이더넷 킷 · LibTorch 2.4.0 (CPU, cxx11-ABI)

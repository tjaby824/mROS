# mROS — Teensy 4.1 실세계 강화학습 스택

Teensy 4.1 펌웨어(`sac3.cpp`)와 PC 트레이너(`sac_trainer_cpp`)를 micro-ROS로 연결해
실세계에서 SAC 학습을 돌리는 프레임워크.

## 어느 폴더가 어디로 가나

클론한 뒤 **대부분은 제자리에서 그대로 쓴다.** 다른 곳으로 복사되는 건 트레이너 하나뿐이고,
그것도 `setup.sh`가 알아서 한다.

**저장소 경로가 곧 홈 디렉터리 경로다.** 최상위 `Documents/`와 `ros2_ws/`는 `$HOME` 아래
같은 이름의 위치를 가리킨다.

| 저장소 | 홈 디렉터리 | 방식 |
|---|---|---|
| `Documents/PlatformIO/mROS/` | `~/Documents/PlatformIO/mROS` | **심링크** |
| `Documents/PlatformIO/examples/` | `~/Documents/PlatformIO/examples` | **심링크** |
| `ros2_ws/src/sac_trainer_cpp/` | `~/ros2_ws/src/sac_trainer_cpp` | **복사** (colcon이 워크스페이스 안을 봐야 함) |
| `setup/` | — | 이동 없음. 여기서 `./setup.sh` 실행 |

심링크는 `setup.sh link`가 만든다. 복사가 아니라 링크이므로 펌웨어 편집은 git 안에서
이뤄지고, `.pio` 빌드 캐시(수백 MB)도 클론 한 곳에만 쌓인다.

**클론은 `~/Documents/PlatformIO` 바깥에 둔다** (예: `~/mROS`). 그 안에 두면 심링크가
자기 자신을 가리키게 되어 `link` 단계가 건너뛴다 — 그래도 저장소 안 경로에서 그대로
빌드할 수 있다.

## 구성

```
setup/                            설치 자동화
  setup.sh                        전체 설치 (9단계)
  verify.sh                       설치 확인
  config.env                      IP · 도메인 · 버전 — 여기만 고치면 된다

Documents/PlatformIO/
  mROS/                           Teensy 4.1 펌웨어 (PlatformIO)
    platformio.ini                빌드 설정
    src/sac3.cpp                  micro-ROS 노드 + 커스텀 UDP 트랜스포트
    include/weights_30k.h         학습된 정책 가중치
  examples/                       참고용 예제 (서브모듈, hippo5329)

ros2_ws/src/sac_trainer_cpp/      PC 학습 노드 (LibTorch SAC)
  src/sac_trainer_node.cpp        dual_sac 실행 파일
  src/HiLH.cpp                    hilh 실행 파일
  scripts/run_*.sh                학습 실행 스크립트
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
cd ~/Documents/PlatformIO/mROS      # 또는 저장소의 Documents/PlatformIO/mROS
pio run
```

VSCode로 열 때도 저장소 루트가 아니라 이 폴더를 연다 — PlatformIO는
`platformio.ini`가 있는 폴더만 프로젝트로 인식한다.

## 요구 사항

Ubuntu 24.04 · ROS 2 Jazzy · Teensy 4.1 + 이더넷 킷 · LibTorch 2.4.0 (CPU, cxx11-ABI)

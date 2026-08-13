# 실세계 RL 프레임워크 빌드 매뉴얼

깨끗한 Ubuntu 24.04에서 `sac3.cpp`(Teensy 4.1)와 `sac_trainer_cpp`(PC)를 micro-ROS로 연결해 학습을 돌리기까지.

---

## 0. 무엇을 만드는가

세 계층이 하나의 파라미터 집합을 공유한다. **하나라도 어긋나면 에러 없이 조용히 연결되지 않는다** — 이 매뉴얼이 매 단계마다 확인을 요구하는 이유다.

```
  [Teensy 4.1]          [PC]                          [PC]
   sac3.cpp    ──UDP──▶  micro_ros_agent  ──토픽──▶  sac_trainer_cpp
   펌웨어 계층   8888     통신 계층          도메인121   학습 계층
      ▲                                                    │
      └──────────────── 정책 가중치 배포 ────────────────────┘
```

| 항목 | 값 | 어디에 박혀 있나 |
|---|---|---|
| `ROS_DOMAIN_ID` | **121** | `sac3.cpp:662`, `run_dual_sac.sh:23`, `rsrl` alias |
| 에이전트 포트 | **8888** (UDP4) | `sac3.cpp:362` `agent_port` |
| Teensy IP | `192.168.1.10` | `sac3.cpp:363` `local_ip` |
| PC IP | `192.168.1.12` | `sac3.cpp:361` `agent_ip` — **PC NIC에 직접 설정할 값** |
| GW / 마스크 | `192.168.1.1` / `255.255.255.0` | `sac3.cpp:364-365` |
| 워크스페이스 | `~/ros2_ws` 하나 | 에이전트·트레이너 공용 |

펌웨어 쪽은 전부 **컴파일 시점에 박히는 값**이다. 바꾸려면 `sac3.cpp`를 고쳐 다시
플래시해야 하고, `setup/config.env`만 고치면 스크립트와 검사 도구만 새 값을 보게 되어
오히려 어긋난다.

값을 바꾸려면 [`setup/config.env`](setup/config.env)와 **펌웨어 양쪽을** 고쳐야 한다.

---

## 1. 빠른 경로

전체는 네 단계다. 설치만 스크립트가 하고, 나머지 셋은 직접 한다.

| | 단계 | 하는 곳 | 절 |
|---|---|---|---|
| ① | 저장소 받기 + `./setup.sh` | 터미널 | 1절 |
| ② | PC NIC에 고정 IP | 설정 또는 터미널 | 3절 |
| ③ | 펌웨어 업로드 | VSCode 또는 터미널 | 4절 |
| ④ | 에이전트 + 학습 실행 | 터미널 3개 | 5절 |

```bash
# 경로 A — git (권장)
git clone --recurse-submodules https://github.com/tjaby824/mROS.git
cd mROS/setup

# 경로 B — zip 다운로드
unzip mROS-main.zip && cd mROS-main/setup && chmod +x *.sh
```

> zip으로 받으면 실행 비트가 보존되지 않아 `chmod +x`가 필요하다(`bash setup.sh`로 실행해도 된다).
> 예제 서브모듈도 zip에는 포함되지 않는데, 참고용일 뿐이라 설치에는 영향이 없다.

`config.env`에서 IP를 본인 환경에 맞게 고친 뒤:

```bash
./setup.sh
```

기본 패키지·로케일·ROS 2 Jazzy·colcon·LibTorch·PlatformIO·VSCode 확장·micro-ROS 에이전트·트레이너·셸 환경이 전부 여기 들어 있다. 8단계로 나뉘어 있고 **재실행해도 안전하다** — 이미 된 단계는 건너뛴다.

처음 실행은 오래 걸린다. LibTorch 500MB 다운로드와 micro-ROS 에이전트 빌드가 대부분을 차지한다.

```bash
# 단계 목록 / 일부만 다시
./setup.sh --list
./setup.sh agent trainer
./setup.sh --from libtorch
```

설치가 끝나면 **로그아웃 → 재로그인**(그룹 적용) 후:

```bash
./verify.sh
```

여기까지 전부 통과하면 남은 건 네트워크(3절), 펌웨어(4절), 실행(5절) 셋뿐이다.

---

## 2. 스크립트가 하는 일

직접 확인하거나 손으로 돌려야 할 때를 위한 설명. 각 단계의 "왜"가 핵심이다.

| 단계 | 하는 일 | 왜 이 순서인가 |
|---|---|---|
| `base` | apt 기본 패키지, UTF-8 로케일 | ROS 2는 UTF-8을 전제한다. 안 맞으면 빌드 중 인코딩 에러 |
| `ros` | ROS 2 Jazzy, colcon, rosdep | 이후 모든 clone이 `$ROS_DISTRO`로 브랜치를 고른다. **colcon은 시스템에** 깔린다 — venv 안에 pip으로 깔면 venv를 끈 순간 `command not found` |
| `libtorch` | LibTorch 2.4.0 CPU → `~/libtorch` | cxx11-ABI가 아니면 ROS 2와 링크가 안 된다. 경로는 `config.env`의 `LIBTORCH_DIR`로 바꿀 수 있다 |
| `platformio` | PlatformIO, udev 규칙, `dialout`/`plugdev` | Teensy 업로드에 USB 접근 권한이 필요하다 |
| `vscode` | PlatformIO IDE 확장 설치 | VSCode가 없으면 안내만 하고 넘어간다(터미널로도 업로드 가능) |
| `agent` | `micro_ros_setup` → `create_agent_ws.sh` → `build_agent.sh` | `micro_ros_agent`는 XRCE-DDS 엔진을 별도 패키지로 찾는다. `micro-ROS-Agent` 저장소만 clone하면 엔진이 없어 `find_package`에서 멈춘다 |
| `trainer` | 저장소의 `trainer/sac_trainer_cpp`를 `~/ros2_ws/src/`로 복사 후 colcon 빌드 | 트레이너가 이 저장소 안에 있으므로 별도 clone이 필요 없다 |
| `shell` | `.bashrc` 블록, `rsrl` alias, venv | venv는 **실행용**이지 빌드용이 아니다. 그래서 마지막에 온다 |

네 가지를 짚어둔다. 전부 이 스택에서 실제로 사람을 붙잡았던 지점이다.

**venv를 먼저 만들지 않는다.** `rsrl` venv는 학습 실행에만 쓴다. 빌드를 venv 안에서 하면 `colcon: command not found`가 난다. 스크립트는 venv가 켜진 상태로 실행되면 아예 거부한다.

**단계를 따로 돌려도 된다.** `ros`는 `curl`과 `add-apt-repository`를, `libtorch`는
`wget`과 `unzip`을 쓰는데 이들은 `base`가 깔아준다. 각 단계가 자기 전제를 먼저 확인해
빠진 게 있으면 무엇이 없는지 이름을 대며 멈춘다(sudo가 이미 확인된 단계에서는 바로 설치한다).
`rosdep` 캐시도 `$HOME/.ros`에 있어 마찬가지라, 필요하면 그 자리에서 만든다.

**`.bashrc`는 마커 블록으로 관리된다.** 손으로 append 하면 재실행할 때마다 `LD_LIBRARY_PATH`가 중복으로 쌓인다. 스크립트는 `# >>> rsrl stack ... >>>` 블록을 통째로 갈아끼운다.

**트레이너는 이 저장소 안에 있다.** `trainer/sac_trainer_cpp/`가 정본이고, `trainer` 단계가 이를 `~/ros2_ws/src/`로 복사해 빌드한다. 코드를 고칠 때는 저장소 쪽을 고치고 `./setup.sh trainer`를 다시 돌린다.

---

## 3. 네트워크

**왜**: 펌웨어가 PC를 `192.168.1.12`로 **고정 지정해** UDP를 쏜다. PC NIC이 다른 IP면 패킷이 도착하지 않는다. DHCP로 받은 주소로는 안 된다.

PC 이더넷 NIC에 고정 IP를 준다 (GUI 설정 또는):

```bash
sudo ip addr add 192.168.1.12/24 dev <NIC이름>
```

`<NIC이름>`은 `ip -br link`로 확인한다.

**확인** (Teensy 전원 인가 상태):

```bash
ping -c3 192.168.1.10
```

---

## 4. 펌웨어 업로드

Teensy를 USB로 연결하고, 이더넷도 함께 물려둔다.

### VSCode (권장)

**여는 폴더는 저장소 루트가 아니라 `Projects/mROS`다.** PlatformIO는 `platformio.ini`가
있는 폴더를 프로젝트로 인식한다. 루트를 열면 프로젝트를 못 찾아 툴바가 나타나지 않는다.

```bash
code Projects/mROS
```

1. 왼쪽 사이드바에 **개미 머리 아이콘**(PlatformIO)이 생기고, 하단에 파란 상태바가 뜬다.
   확장이 없으면 VSCode가 설치를 제안한다(`.vscode/extensions.json`에 등록돼 있다).
   `setup.sh vscode`가 미리 깔아두기도 한다.
2. 첫 실행은 micro-ROS 라이브러리를 통째로 받아 빌드하므로 **10분 이상** 걸린다.
   하단 상태바가 계속 도는 것은 정상이다.
3. 하단 상태바에서 환경이 **`env:teensy41`**인지 확인한다. 환경이 하나뿐이라 자동 선택된다.
4. **✓ (체크)** 버튼 = 빌드, **→ (화살표)** 버튼 = 업로드.
5. 업로드를 누르면 **Teensy Loader 창이 뜬다** (`upload_protocol = teensy-gui`).
   자동 리부팅까지 처리되므로 보통은 그대로 두면 끝난다.
   창이 뜨고 멈춰 있으면 **보드의 program 버튼을 한 번 누른다.**
6. 시리얼 출력은 **플러그 아이콘**(Serial Monitor), 속도는 `115200`.

### 터미널

```bash
cd Projects/mROS
pio run -e teensy41 -t upload
```

**확인**: `.pio/build/teensy41/firmware.hex` 생성 + `SUCCESS` 출력.

### `platformio.ini`에서 건드리면 안 되는 것

```ini
board_microros_transport = custom
```

**이게 빠지면 라이브러리가 기본값인 `serial`로 빌드된다.** `sac3.cpp`는 NativeEthernet UDP
트랜스포트를 직접 구현해 `rmw_uros_set_custom_transport()`로 등록하므로, 라이브러리가
트랜스포트를 하나 더 컴파일하면 안 된다.

트랜스포트는 **`board_microros_transport` 옵션으로만** 선택된다. `build_flags`에
`-D MICRO_ROS_TRANSPORT_...` 를 넣는 방식은 동작하지 않는다 — 실제 매크로는 라이브러리가
`MICRO_ROS_TRANSPORT_ARDUINO_<TRANSPORT>` 형태로 자동 생성한다.

빌드 로그 첫머리에서 확인할 수 있다:

```
Configuring teensy41 with transport custom
```

`serial`이라고 나오면 위 옵션이 안 먹은 것이다.

## 5. 실행

터미널 3개. 순서대로 연다 — **에이전트가 먼저 떠 있어야** 펌웨어 세션이 붙는다.

```bash
# 터미널 1 — 에이전트
rsrl
ros2 run micro_ros_agent micro_ros_agent udp4 --port 8888      # 디버그는 -v6
```

`-v6`로 띄우면 Teensy가 붙는 순간 `session established` 로그가 뜬다. 이게 안 보이면
네트워크(3절)부터 다시 본다.

```bash
# 터미널 2 — 확인
rsrl
ros2 topic list        # rl_observation / encoder_feedback / episode_cmd_teensy
```

```bash
# 터미널 3 — 학습
cd ~/ros2_ws/src/sac_trainer_cpp/scripts
./run_dual_sac.sh                  # RULE+RL 교대
# ./run_rl_only.sh                 # 체크포인트에서 RL 전용 이어하기
```

**`scripts/`로 반드시 `cd` 한 뒤 실행할 것.** `run_dual_sac.sh`는 출력 경로를
`save_dir:=./sac_checkpoints`, `csv_dir:=./episode_logs`, `buffer_dir:=./replay_buffers`
처럼 **상대경로**로 넘긴다. 다른 위치에서 돌리면 체크포인트와 로그가 그 위치에 쌓여
다음 실행에서 이어붙지 않는다.

터미널 3은 `rsrl`이 필요 없다 — `run_dual_sac.sh`가 자체적으로 `ROS_DOMAIN_ID=121`을
export하고 `~/ros2_ws/install/setup.bash`를 source한다. 터미널 1·2는 `rsrl`이 필요하다.

인자로 동작을 바꿀 수 있다 (기본값은 스크립트 상단 참조):

```bash
./run_dual_sac.sh 2000 2 30.0             # 에피소드 수, 난이도, sigma(deg)
./run_dual_sac.sh 2000 2 30.0 <체크포인트>  # 체크포인트에서 이어하기
```

---

## 6. 스모크 테스트

위에서부터 순서대로. **처음 막히는 지점이 원인 지점이다.**

```bash
./setup/verify.sh --live
```

수동으로 하려면:

```bash
ping -c3 192.168.1.10                    # 1. 이더넷 링크
#                                          2. 에이전트 -v6 로그에 session established
ros2 topic hz /encoder_feedback          # 3. 1kHz 텔레메트리
ros2 topic echo /rl_observation --once   # 4. 60차원 관측
ros2 topic echo /policy_ack              # 5. 트레이너 기동 후 가중치 배포 확인
```

---

## 7. 증상별 조치

| 증상 | 원인 | 조치 |
|---|---|---|
| `Configuring ... with transport serial` | `board_microros_transport` 누락 | `platformio.ini`에 `= custom` 추가 (4절) |
| `colcon: command not found` | venv 안에 colcon 설치 | `deactivate` 후 `./setup.sh ros` |
| `rosdep installation has not been initialized` | rosdep 캐시가 `$HOME/.ros`에 없음 | 스크립트가 자동 생성한다. 수동이면 `rosdep update` |
| `이 단계에 필요한 도구가 없다: ...` | 단계를 건너뛰고 뒤 단계만 실행함 | `./setup.sh base` 먼저 |
| `git clone -b`이 엉뚱한 브랜치 | `$ROS_DISTRO` 비어 있음 | `source /opt/ros/jazzy/setup.bash` 먼저 |
| `find_package(microxrcedds_agent)` 실패 | 에이전트를 수동 clone함 | `./setup.sh agent` (`micro_ros_setup` 경로) |
| `find_package(Torch)` 실패 | LibTorch 경로 불일치 | `config.env`의 `LIBTORCH_DIR` 확인 후 `./setup.sh trainer` |
| `libtorch_cpu.so` 못 찾음 | `LD_LIBRARY_PATH` 미설정 | `./setup.sh shell` 후 새 셸 |
| 링크 단계 ABI 에러 | pre-cxx11 LibTorch | cxx11-ABI 빌드로 교체 |
| Teensy 업로드 권한 거부 | 그룹이 세션에 미적용 | **로그아웃 → 재로그인**. 터미널만 닫으면 안 된다 |
| 에이전트는 붙는데 토픽이 안 보임 | 도메인 불일치 | `echo $ROS_DOMAIN_ID` → 121 |
| `ping` 실패 | PC NIC IP 불일치 | 3절 (`192.168.1.12/24`) |
| `rclc pull failed` / `Aborting` | micro-ROS 저장소 캐시 충돌 | `cd Projects/mROS && pio run -t clean_microros` 후 재빌드 |
| `file INSTALL cannot find lib*.a` | micro-ROS 라이브러리 빌드 레이스 | `pio run`을 한 번 더. 대개 이어서 통과한다 |

---

## 부록. 이전 매뉴얼에서 바뀐 점

| 이전 | 현재 |
|---|---|
| `Projects/StTn` | `Projects/mROS` |
| `sac3.cpp`가 `lib/`에 있어 `ln -sf` 심링크 필요 | `src/sac3.cpp`에 있음 — 심링크 불필요 |
| `pio run -e baseline -t upload` | `pio run -e teensy41 -t upload` |
| `-e e0_eigen` / `-e e0_forloop` 백엔드 | 해당 env 제거 (참조하던 `e0_backends.cpp`가 존재하지 않았음) |
| `build_src_filter = +<sac3.cpp>` | 제거 (소스가 하나라 불필요) |
| `extra_configs = ../platformio/platformio.ini` | 제거 (예제 저장소 구조에서 온 잔재로, 경로가 깨져 있었음) |
| 트랜스포트 지정 없음 (→ `serial`로 빌드) | `board_microros_transport = custom` |
| STEP 0~8을 손으로 붙여넣기 | `./setup.sh` |
| 첨부파일 3개 (`sac3.cpp`, `platformio.ini`, `sac_trainer_cpp.zip`) | 저장소 **1개** clone (트레이너 포함) |
| 트레이너가 별도 비공개 저장소 | `trainer/sac_trainer_cpp/`로 합침 |
| VSCode 업로드 안내 없음 | 4절에 VSCode 절 추가 |
| `CMakeLists.txt`가 `set(CMAKE_PREFIX_PATH ...)`로 덮어써 `LIBTORCH_DIR` 설정이 무시됨 | `list(APPEND ...)`로 수정 |

---

## 부록. 이 매뉴얼의 검증 상태

깨끗한 Ubuntu 24.04 머신이 없어 전 과정을 한 번에 통과시켜 본 것은 아니다.
어디까지 실제로 확인했는지 밝혀둔다.

| 항목 | 상태 |
|---|---|
| 저장소 clone → `trainer` → `shell` | **실행 확인.** 새 clone에서 격리 HOME으로 완주(`dual_sac`, `hilh` 생성) |
| `libtorch` 단계 | **실행 확인.** 500MB 실제 다운로드 후 `2.4.0+cpu` 압축 해제 |
| `vscode` 단계 | **실행 확인.** 확장 설치·건너뜀 양쪽 경로 |
| `shell` 단계 멱등성 | **실행 확인.** 두 번 돌려 `.bashrc` 블록 1개 유지 |
| 단계별 전제 검사 | **실행 확인.** 도구를 숨긴 PATH에서 정확한 메시지로 중단 |
| `ros` 단계의 다운로드 경로 | **부분 확인.** 버전 조회(1.2.0) → deb URL(HTTP 200) → 패키지 가용성까지. `apt-get install` 실행 자체는 미확인 |
| `base` 단계 | **미확인.** 이 머신에 이미 설치돼 건너뛴다 |
| `agent` 단계 | **미확인.** `build_agent.sh`가 20분 이상이라 재실행하지 않음 |
| 펌웨어 빌드 | **실행 확인.** `firmware.hex` 생성, `transport custom` 적용 |
| Teensy 업로드 / 학습 실행 | **미확인.** 하드웨어 필요 |

즉 첫 실행에서 눈여겨볼 곳은 `base`·`ros`의 apt 설치와 `agent` 빌드,
그리고 실제 업로드다. 나머지는 실행으로 확인됐다.

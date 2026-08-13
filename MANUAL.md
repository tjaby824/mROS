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
| `ROS_DOMAIN_ID` | **121** | `sac3.cpp` 하드코딩, `run_dual_sac.sh`, `rsrl` alias |
| 에이전트 포트 | **8888** (UDP4) | 펌웨어 `agent_port` |
| Teensy IP | `192.168.1.10` | 펌웨어 `local_ip` |
| PC IP | `192.168.1.12` | 펌웨어 `agent_ip` — **PC NIC에 직접 설정할 값** |
| GW / 마스크 | `192.168.1.1` / `255.255.255.0` | 펌웨어 하드코딩 |
| 워크스페이스 | `~/ros2_ws` 하나 | 에이전트·트레이너 공용 |

값을 바꾸려면 [`setup/config.env`](setup/config.env)와 **펌웨어 양쪽을** 고쳐야 한다. 설정 파일만 고치면 펌웨어는 여전히 옛 주소로 쏜다.

---

## 1. 빠른 경로

```bash
git clone --recurse-submodules https://github.com/tjaby824/mROS.git
cd mROS/setup
```

`config.env`에서 IP를 본인 환경에 맞게 고친 뒤:

```bash
./setup.sh
```

STEP 0~8에 해당하는 설치가 전부 여기 들어 있다(기본 패키지·로케일·ROS 2 Jazzy·colcon·LibTorch·PlatformIO·에이전트·트레이너·셸 환경). 재실행해도 안전하며, 이미 된 단계는 건너뛴다.

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
| `libtorch` | LibTorch 2.4.0 CPU → `~/libtorch` | `sac_trainer_cpp/CMakeLists.txt`가 이 경로로 **고정**돼 있다. cxx11-ABI가 아니면 ROS 2와 링크가 안 된다 |
| `platformio` | PlatformIO, udev 규칙, `dialout`/`plugdev` | Teensy 업로드에 USB 접근 권한이 필요하다 |
| `agent` | `micro_ros_setup` → `create_agent_ws.sh` → `build_agent.sh` | `micro_ros_agent`는 XRCE-DDS 엔진을 별도 패키지로 찾는다. `micro-ROS-Agent` 저장소만 clone하면 엔진이 없어 `find_package`에서 멈춘다 |
| `trainer` | 트레이너 clone + colcon 빌드 | CMakeLists가 `install(DIRECTORY scripts/ ...)`를 하므로 `scripts/`가 없으면 빌드가 실패한다 |
| `shell` | `.bashrc` 블록, `rsrl` alias, venv | venv는 **실행용**이지 빌드용이 아니다. 그래서 마지막에 온다 |

두 가지를 짚어둔다.

**venv를 먼저 만들지 않는다.** `rsrl` venv는 학습 실행에만 쓴다. 빌드를 venv 안에서 하면 `colcon: command not found`가 난다. 스크립트는 venv가 켜진 상태로 실행되면 아예 거부한다.

**`.bashrc`는 마커 블록으로 관리된다.** 손으로 append 하면 재실행할 때마다 `LD_LIBRARY_PATH`가 중복으로 쌓인다. 스크립트는 `# >>> rsrl stack ... >>>` 블록을 통째로 갈아끼운다.

> **트레이너 저장소 접근** — `sac_trainer_cpp`는 `tjaby824/ros2_ws`에 있다. 비공개 저장소이므로 다른 사람이 clone하려면 `gh auth login`으로 인증돼 있거나 collaborator로 추가돼 있어야 한다. 안 되면 `trainer` 단계가 그 안내와 함께 멈춘다.

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

## 4. 펌웨어

```bash
cd Projects/mROS
pio run -e teensy41 -t upload
```

첫 빌드는 micro-ROS 라이브러리를 통째로 받아 빌드하므로 오래 걸리고 네트워크가 필요하다. 이후는 1분 내외다.

**확인**: `.pio/build/teensy41/firmware.hex` 생성, 그리고 `SUCCESS` 출력.

### `platformio.ini`에서 건드리면 안 되는 것

```ini
board_microros_transport = custom
```

**이게 빠지면 라이브러리가 기본값인 `serial`로 빌드된다.** `sac3.cpp`는 NativeEthernet UDP 트랜스포트를 직접 구현해 `rmw_uros_set_custom_transport()`로 등록하므로, 라이브러리가 트랜스포트를 하나 더 컴파일하면 안 된다.

트랜스포트는 **`board_microros_transport` 옵션으로만** 선택된다. `build_flags`에 `-D MICRO_ROS_TRANSPORT_...` 를 넣는 방식은 동작하지 않는다 — 실제 매크로는 라이브러리가 `MICRO_ROS_TRANSPORT_ARDUINO_<TRANSPORT>` 형태로 자동 생성한다.

빌드 로그 첫머리에서 확인할 수 있다:

```
Configuring teensy41 with transport custom
```

`serial`이라고 나오면 위 옵션이 안 먹은 것이다.

### 업로드 방식

현재 `upload_protocol = teensy-gui`로, **Teensy Loader GUI 앱이 떠 있어야** 동작한다. 터미널만으로 끝내려면:

```ini
upload_protocol = teensy-cli
```

로 바꾼다. 이 경우 `dialout` 그룹이 현재 세션에 적용돼 있어야 한다(`verify.sh`가 검사한다).

---

## 5. 실행

터미널 3개, 전부 `rsrl`로 연다.

```bash
# 터미널 1 — 에이전트
rsrl
ros2 run micro_ros_agent micro_ros_agent udp4 --port 8888      # 디버그는 -v6
```

```bash
# 터미널 2 — 확인
rsrl
ros2 topic list        # rl_observation / encoder_feedback / episode_cmd_teensy
```

```bash
# 터미널 3 — 학습
rsrl
cd ~/ros2_ws/src/sac_trainer_cpp/scripts
./run_dual_sac.sh                  # RULE+RL 교대
# ./run_rl_only.sh                 # 체크포인트에서 RL 전용 이어하기
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
| `git clone -b`이 엉뚱한 브랜치 | `$ROS_DISTRO` 비어 있음 | `source /opt/ros/jazzy/setup.bash` 먼저 |
| `find_package(microxrcedds_agent)` 실패 | 에이전트를 수동 clone함 | `./setup.sh agent` (`micro_ros_setup` 경로) |
| `find_package(Torch)` 실패 | LibTorch가 `~/libtorch`가 아님 | 경로 이동 (CMakeLists가 고정) |
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
| 첨부파일 3개 (`sac3.cpp`, `platformio.ini`, `sac_trainer_cpp.zip`) | 저장소 2개 clone |

#!/usr/bin/env bash
# =============================================================================
# 실세계 RL 스택 설치 — 깨끗한 Ubuntu 24.04 기준
#
#   ./setup.sh              전체 설치
#   ./setup.sh --list       단계 목록
#   ./setup.sh ros libtorch 특정 단계만
#   ./setup.sh --from agent agent 단계부터 끝까지
#
# 모든 단계는 재실행해도 안전하다(이미 된 건 건너뛴다).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=config.env
source "$SCRIPT_DIR/config.env"

PHASES=(base ros libtorch platformio link vscode agent trainer shell)
# sudo가 실제로 필요한 단계. 나머지는 전부 $HOME 안에서만 작업하므로
# 암호를 요구하지 않는다(./setup.sh shell 하나 돌리려고 sudo를 묻지 않도록).
SUDO_PHASES=(base ros platformio)

# ROS의 setup.bash / local_setup.bash 는 nounset(set -u) 환경에서 깨진다.
# 내부에서 AMENT_TRACE_SETUP_FILES 같은 미정의 변수를 참조하기 때문.
src_ros() { set +u; # shellcheck disable=SC1090
            source "$1"; set -u; }

# 단계를 따로 실행해도 그 단계가 쓰는 명령이 있어야 한다. base 단계가 깔아주는
# 도구(curl, add-apt-repository, wget, unzip)를 뒤 단계가 먼저 쓰기 때문에,
# ./setup.sh ros 처럼 단독 실행하면 깨끗한 머신에서 엉뚱한 곳에서 죽는다.
# 인자는 "명령:패키지" 쌍.
need_cmds() {
  local missing=() pair cmd pkg
  for pair in "$@"; do
    cmd="${pair%%:*}"; pkg="${pair##*:}"
    command -v "$cmd" >/dev/null 2>&1 || missing+=("$pkg")
  done
  [[ ${#missing[@]} -eq 0 ]] && return 0

  if [[ "${SUDO_OK:-0}" == "1" ]]; then
    c_warn "빠진 도구를 설치한다: ${missing[*]}"
    sudo apt-get install -y -qq "${missing[@]}"
  else
    c_die "이 단계에 필요한 도구가 없다: ${missing[*]}
       먼저 기본 패키지를 설치할 것:  ./setup.sh base"
  fi
}

# rosdep 캐시는 시스템이 아니라 $HOME/.ros 아래에 있다. 그래서 ros 단계를 거치지 않고
# agent/trainer 단계만 따로 돌리면 "rosdep installation has not been initialized"로
# 죽는다. 캐시가 없을 때만 만들어 준다.
ensure_rosdep() {
  [[ -d "$HOME/.ros/rosdep/sources.cache" ]] && return 0
  echo "  rosdep 캐시가 없다. 생성한다 (처음 한 번, 네트워크 필요)..."
  rosdep update --rosdistro "${ROS_DISTRO_NAME}" >/dev/null \
    || c_die "rosdep update 실패. 네트워크를 확인하거나 './setup.sh ros'를 먼저 실행할 것."
}

# --- 출력 -------------------------------------------------------------------
c_ok()   { printf '\033[32m  ✓ %s\033[0m\n' "$*"; }
c_skip() { printf '\033[90m  · %s (이미 됨, 건너뜀)\033[0m\n' "$*"; }
c_step() { printf '\n\033[1;36m━━ %s\033[0m\n' "$*"; }
c_warn() { printf '\033[33m  ! %s\033[0m\n' "$*"; }
c_die()  { printf '\033[31m\n실패: %s\033[0m\n' "$*" >&2; exit 1; }

# .bashrc 를 마커로 감싸 관리한다. 재실행 시 블록을 통째로 갈아끼우므로
# 매뉴얼대로 손으로 append 하다 생기는 중복(LD_LIBRARY_PATH 두 줄 같은)이 없다.
BASHRC_BEGIN='# >>> rsrl stack (managed by setup.sh) >>>'
BASHRC_END='# <<< rsrl stack (managed by setup.sh) <<<'

# --- 사전 점검 ---------------------------------------------------------------
# $@ = 이번에 돌릴 단계 목록
preflight() {
  [[ $EUID -eq 0 ]] && c_die "root로 실행하지 말 것. 필요한 곳에서만 sudo를 쓴다.
       (root로 돌리면 ~/libtorch, ~/.platformio 가 root 소유가 되어 나중에 깨진다)"

  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    c_die "가상환경이 켜져 있다. colcon은 시스템 python으로 빌드해야 한다.
       'deactivate' 후 다시 실행할 것."
  fi

  . /etc/os-release
  [[ "${VERSION_CODENAME:-}" == "noble" ]] \
    || c_warn "Ubuntu 24.04(noble) 기준으로 작성됨. 현재: ${PRETTY_NAME:-unknown}"

  # zip으로 받으면 실행 비트가 보존되지 않는다. bash로 호출하면 여기까지 오므로
  # 나머지 스크립트에도 실행 권한을 붙여 verify.sh 등이 바로 돌게 한다.
  chmod +x "$SCRIPT_DIR"/*.sh 2>/dev/null || true

  # 서브모듈은 예제 참고용일 뿐 설치에 필요 없다. zip 경로에선 비어 있는 게 정상이므로
  # 경고만 하고 진행한다.
  if [[ -d "${REPO_ROOT}/examples" ]] \
     && [[ -z "$(ls -A "${REPO_ROOT}/examples" 2>/dev/null)" ]]; then
    c_warn "예제 서브모듈이 비어 있다(zip으로 받으면 정상). 설치에는 영향 없다.
       필요하면: git submodule update --init"
  fi

  # 이번 실행에 sudo가 필요한 단계가 하나라도 있을 때만 미리 확인한다.
  local needs_sudo=0 p q
  for p in "$@"; do
    for q in "${SUDO_PHASES[@]}"; do
      [[ "$p" == "$q" ]] && needs_sudo=1
    done
  done

  if [[ $needs_sudo -eq 1 ]]; then
    command -v sudo >/dev/null || c_die "sudo가 없다."
    echo "sudo 권한을 미리 확인한다 (설치 중 암호 재입력을 피하기 위함)"
    sudo -v || c_die "sudo 인증 실패."
    SUDO_OK=1
  fi
}

# --- 1. 기본 패키지 + 로케일 --------------------------------------------------
phase_base() {
  c_step "1/9  기본 패키지와 로케일"

  sudo apt-get update -qq
  sudo apt-get install -y -qq \
    wget curl git build-essential cmake unzip net-tools \
    python3-pip python3-venv software-properties-common locales
  c_ok "기본 패키지"

  # ROS 2는 UTF-8 로케일을 전제한다. 안 맞으면 빌드 중 인코딩 에러가 난다.
  if locale | grep -q 'UTF-8'; then
    c_skip "UTF-8 로케일"
  else
    sudo locale-gen en_US en_US.UTF-8
    sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
    c_ok "로케일을 en_US.UTF-8로 설정"
    c_warn "로케일은 새 셸부터 적용된다."
  fi
}

# --- 2. ROS 2 + 빌드 툴체인 ---------------------------------------------------
phase_ros() {
  c_step "2/9  ROS 2 ${ROS_DISTRO_NAME} 와 빌드 툴체인"

  need_cmds curl:curl add-apt-repository:software-properties-common

  if [[ -d "/opt/ros/${ROS_DISTRO_NAME}" ]]; then
    c_skip "ROS 2 ${ROS_DISTRO_NAME}"
  else
    sudo add-apt-repository -y universe
    sudo apt-get update -qq

    # ros-apt-source 최신 릴리스를 받아 apt 소스를 등록한다.
    local ver deb_url codename
    ver="$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
           | grep -F '"tag_name"' | awk -F'"' '{print $4}')"
    [[ -n "$ver" ]] || c_die "ros-apt-source 최신 버전을 못 가져왔다 (네트워크/GitHub API 확인)"

    codename="$(. /etc/os-release && echo "${UBUNTU_CODENAME:-${VERSION_CODENAME}}")"
    deb_url="https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ver}/ros2-apt-source_${ver}.${codename}_all.deb"

    curl -fsSL -o /tmp/ros2-apt-source.deb "$deb_url" \
      || c_die "apt 소스 deb 다운로드 실패: $deb_url"
    sudo dpkg -i /tmp/ros2-apt-source.deb
    sudo apt-get update -qq
    sudo apt-get install -y "ros-${ROS_DISTRO_NAME}-desktop"
    c_ok "ROS 2 ${ROS_DISTRO_NAME}"
  fi

  # colcon은 반드시 '시스템에' 있어야 한다. venv 안에 pip으로 깔면
  # venv를 끈 순간 command not found가 된다.
  sudo apt-get install -y -qq python3-colcon-common-extensions python3-rosdep
  c_ok "colcon / rosdep"

  if [[ -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    c_skip "rosdep init"
  else
    sudo rosdep init
  fi
  rosdep update --rosdistro "${ROS_DISTRO_NAME}" >/dev/null
  c_ok "rosdep update"

  # 이후 단계가 $ROS_DISTRO로 브랜치를 고르므로 여기서 반드시 확인한다.
  src_ros "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
  [[ "${ROS_DISTRO:-}" == "${ROS_DISTRO_NAME}" ]] \
    || c_die "\$ROS_DISTRO가 '${ROS_DISTRO_NAME}'이 아니다 (현재: '${ROS_DISTRO:-빈값}'). 여기서 멈춘다."
  c_ok "\$ROS_DISTRO = ${ROS_DISTRO}"
}

# --- 3. LibTorch --------------------------------------------------------------
phase_libtorch() {
  c_step "3/9  LibTorch ${LIBTORCH_VERSION} (CPU, cxx11-ABI)"

  need_cmds wget:wget unzip:unzip

  # sac_trainer_cpp/CMakeLists.txt가 CMAKE_PREFIX_PATH를 $HOME/libtorch로
  # 고정한다. 다른 경로에 풀면 find_package(Torch)가 실패한다.
  if [[ -f "${LIBTORCH_DIR}/build-version" ]]; then
    c_skip "LibTorch ($(cat "${LIBTORCH_DIR}/build-version"))"
    return
  fi

  local zip="/tmp/libtorch-${LIBTORCH_VERSION}.zip"
  # '+'는 URL에서 %2B로 인코딩해야 한다.
  local url="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip"

  echo "  다운로드 중 (약 500MB)..."
  wget -q --show-progress -O "$zip" "$url" || c_die "LibTorch 다운로드 실패: $url"

  # 압축을 풀면 libtorch/ 가 나오므로 상위인 $HOME에 푼다.
  unzip -q "$zip" -d "$(dirname "${LIBTORCH_DIR}")"
  rm -f "$zip"

  [[ -f "${LIBTORCH_DIR}/build-version" ]] || c_die "압축 해제 후 ${LIBTORCH_DIR}가 없다."
  c_ok "LibTorch $(cat "${LIBTORCH_DIR}/build-version") → ${LIBTORCH_DIR}"
}

# --- 4. PlatformIO + 장치 권한 ------------------------------------------------
phase_platformio() {
  c_step "4/9  PlatformIO 와 USB 장치 권한"

  need_cmds curl:curl python3:python3

  if [[ -x "$HOME/.platformio/penv/bin/pio" ]]; then
    c_skip "PlatformIO"
  else
    curl -fsSL -o /tmp/get-platformio.py \
      https://raw.githubusercontent.com/platformio/platformio-core-installer/master/get-platformio.py
    python3 /tmp/get-platformio.py
    c_ok "PlatformIO"
  fi

  if [[ -f /etc/udev/rules.d/99-platformio-udev.rules ]]; then
    c_skip "udev 규칙"
  else
    curl -fsSL https://raw.githubusercontent.com/platformio/platformio-core/develop/platformio/assets/system/99-platformio-udev.rules \
      | sudo tee /etc/udev/rules.d/99-platformio-udev.rules >/dev/null
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    c_ok "udev 규칙"
  fi

  # Teensy 업로드에 필요. 그룹 변경은 재로그인해야 적용된다.
  local need_relogin=0
  for g in dialout plugdev; do
    if id -nG "$USER" | tr ' ' '\n' | grep -qx "$g"; then
      c_skip "그룹 $g"
    else
      sudo usermod -a -G "$g" "$USER"
      c_ok "그룹 $g 추가"
      need_relogin=1
    fi
  done
  [[ $need_relogin -eq 1 ]] && c_warn "그룹 변경은 로그아웃 → 재로그인해야 적용된다. 터미널만 닫는 걸로는 안 된다."
  return 0
}

# --- 5. 홈 디렉터리 심링크 --------------------------------------------------
phase_link() {
  c_step "5/9  ~/Documents/PlatformIO 심링크"

  # 저장소의 Documents/PlatformIO/... 는 홈 아래 같은 경로를 가리키는 이름이다.
  # 실제로 복사하지 않고 심링크로 잇는다 — 그래야 편집이 git 안에서 이뤄지고
  # .pio 캐시(수백 MB)도 한 군데에만 쌓인다.
  local src_base="${REPO_ROOT}/Documents/PlatformIO"
  local dst_base="$HOME/Documents/PlatformIO"

  # 클론이 목적지 안에 있으면 자기 자신을 가리키게 되므로 만들지 않는다.
  case "${REPO_ROOT}/" in
    "$dst_base"/*)
      c_warn "저장소가 ${dst_base} 안에 있어 심링크를 만들지 않는다.
       클론을 그 바깥(예: ~/mROS)에 두면 심링크가 생성된다.
       지금도 ${src_base} 에서 그대로 빌드할 수 있다."
      return 0 ;;
  esac

  mkdir -p "$dst_base"
  local name target
  for name in mROS examples; do
    target="$src_base/$name"
    [[ -e "$target" ]] || continue

    if [[ -L "$dst_base/$name" ]]; then
      ln -sfn "$target" "$dst_base/$name"          # 링크면 갱신
      c_ok "$dst_base/$name → $target"
    elif [[ -e "$dst_base/$name" ]]; then
      c_warn "$dst_base/$name 가 이미 실제 폴더로 존재한다. 건드리지 않는다.
       저장소 쪽을 쓰려면 옮기거나 지운 뒤 './setup.sh link'를 다시 실행할 것."
    else
      ln -s "$target" "$dst_base/$name"
      c_ok "$dst_base/$name → $target"
    fi
  done
}

# --- 6. VSCode + PlatformIO IDE ------------------------------------------------
phase_vscode() {
  c_step "6/9  VSCode 와 PlatformIO IDE 확장"

  if ! command -v code >/dev/null 2>&1; then
    c_warn "VSCode(code)가 없다. 펌웨어는 터미널(pio run -t upload)로도 올릴 수 있으므로
       설치를 강제하지 않는다. VSCode로 쓰려면 먼저 설치한 뒤 이 단계를 다시 실행할 것:
         ./setup.sh vscode"
    return 0
  fi

  # extensions.json이 이 확장을 recommend 하므로 폴더를 열면 VSCode도 제안하지만,
  # 미리 깔아두면 참조자가 제안 팝업을 놓쳐도 바로 쓸 수 있다.
  if code --list-extensions 2>/dev/null | grep -qx 'platformio.platformio-ide'; then
    c_skip "PlatformIO IDE 확장"
  else
    code --install-extension platformio.platformio-ide --force >/dev/null \
      || c_warn "확장 설치 실패. VSCode에서 수동으로 'PlatformIO IDE'를 설치할 것."
    c_ok "PlatformIO IDE 확장"
  fi

  c_ok "VSCode 준비됨 — 열 폴더는 ${REPO_ROOT}/Documents/PlatformIO/mROS"
}

# --- 6. micro-ROS 에이전트 ----------------------------------------------------
phase_agent() {
  c_step "7/9  micro-ROS 에이전트"

  src_ros "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"

  if [[ -d "${WORKSPACE}/install/micro_ros_agent" ]]; then
    c_skip "micro_ros_agent"
    return
  fi

  need_cmds git:git
  mkdir -p "${WORKSPACE}/src"
  cd "${WORKSPACE}"

  # micro_ros_agent는 XRCE-DDS 엔진을 별도 패키지로 찾는다.
  # micro-ROS-Agent 저장소만 clone하면 엔진이 없어 find_package에서 멈춘다.
  # create_agent_ws.sh가 엔진까지 src/uros/ 아래로 한꺼번에 받아온다.
  if [[ ! -d "src/micro_ros_setup" ]]; then
    git clone -b "${ROS_DISTRO}" \
      https://github.com/micro-ROS/micro_ros_setup.git src/micro_ros_setup
  fi

  ensure_rosdep
  rosdep install --from-paths src --ignore-src -y >/dev/null
  colcon build --packages-select micro_ros_setup
  src_ros install/local_setup.bash

  [[ -d src/uros ]] || ros2 run micro_ros_setup create_agent_ws.sh
  echo "  에이전트 빌드 중 (첫 빌드는 오래 걸리고 네트워크가 필요하다)..."
  ros2 run micro_ros_setup build_agent.sh
  src_ros install/local_setup.bash

  c_ok "micro_ros_agent"
}

# --- 6. 트레이너 --------------------------------------------------------------
phase_trainer() {
  c_step "8/9  sac_trainer_cpp 트레이너"

  src_ros "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"

  # 트레이너 소스는 이 저장소 안에 있다. 별도 clone이 필요 없다.
  local repo_src="${REPO_ROOT}/ros2_ws/src/sac_trainer_cpp"
  local ws_src="${WORKSPACE}/src/sac_trainer_cpp"

  [[ -d "$repo_src" ]] || c_die "트레이너 소스가 없다: $repo_src
       저장소를 통째로 받았는지 확인할 것 (zip이라면 압축을 다 풀었는지)."

  # colcon은 워크스페이스 src/ 아래를 본다. 저장소에서 그리로 복사한다.
  # 재실행 시 저장소 쪽이 정본이므로 항상 덮어쓴다(런타임 출력은 scripts/ 아래에
  # 쌓이므로 --delete 없이 복사해 체크포인트를 보존한다).
  mkdir -p "${WORKSPACE}/src"
  cp -r "$repo_src" "${WORKSPACE}/src/"
  c_ok "트레이너 소스 배치 → $ws_src"

  # CMakeLists가 install(DIRECTORY scripts/ ...)를 하므로 scripts/가 없으면 빌드가 실패한다.
  [[ -d "$ws_src/scripts" ]] || c_die "$ws_src/scripts/ 가 없다. 저장소가 온전한지 확인할 것."

  cd "${WORKSPACE}"
  ensure_rosdep
  rosdep install --from-paths src --ignore-src -y >/dev/null
  rm -rf build/sac_trainer_cpp install/sac_trainer_cpp
  colcon build --symlink-install --packages-select sac_trainer_cpp \
    --cmake-args "-DCMAKE_PREFIX_PATH=${LIBTORCH_DIR}"

  chmod +x "$ws_src/scripts/"*.sh
  c_ok "sac_trainer_cpp"
}

# --- 7. 셸 환경 ---------------------------------------------------------------
phase_shell() {
  c_step "9/9  셸 환경 (.bashrc, venv)"

  if [[ -d "${VENV_DIR}" ]]; then
    c_skip "venv ${VENV_DIR}"
  else
    python3 -m venv "${VENV_DIR}"
    c_ok "venv ${VENV_DIR} (학습 실행용. 빌드에는 쓰지 않는다)"
  fi

  # 마커 블록을 통째로 갈아끼운다 → 재실행해도 중복되지 않는다.
  local rc="$HOME/.bashrc"
  if grep -qF "$BASHRC_BEGIN" "$rc" 2>/dev/null; then
    # 기존 블록 삭제
    sed -i "/$(printf '%s' "$BASHRC_BEGIN" | sed 's/[][\.*^$/]/\\&/g')/,/$(printf '%s' "$BASHRC_END" | sed 's/[][\.*^$/]/\\&/g')/d" "$rc"
  fi

  cat >> "$rc" <<EOF
${BASHRC_BEGIN}
export PATH="\$PATH:\$HOME/.platformio/penv/bin:\$HOME/.local/bin"
export LD_LIBRARY_PATH="${LIBTORCH_DIR}/lib:\${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${LIBTORCH_DIR}:\${CMAKE_PREFIX_PATH:-}"
export RCUTILS_CONSOLE_OUTPUT_FORMAT="{message}"

# 학습 스택을 한 번에 켠다. 오버레이와 도메인은 여기서만 잡는다
# (.bashrc 끝에서 무조건 source 하지 않는 이유: 빌드 셸을 오염시키지 않기 위해).
alias rsrl="source ${VENV_DIR}/bin/activate; \\
            source /opt/ros/${ROS_DISTRO_NAME}/setup.bash; \\
            source ${WORKSPACE}/install/setup.bash; \\
            export ROS_DOMAIN_ID=${ROS_DOMAIN}; \\
            echo 'RSRL stack ready (${WORKSPACE}, domain ${ROS_DOMAIN})'"
alias ved="deactivate"
${BASHRC_END}
EOF
  c_ok ".bashrc 갱신 (마커 블록으로 관리 — 재실행해도 중복되지 않음)"
}

# --- 실행 --------------------------------------------------------------------
usage() {
  cat <<EOF
사용법: ./setup.sh [단계...]

  (인자 없음)        전체 설치
  --list             단계 목록
  --from <단계>      해당 단계부터 끝까지
  <단계> [단계...]   지정한 단계만

단계: ${PHASES[*]}
EOF
}

main() {
  local run=()
  if [[ $# -eq 0 ]]; then
    run=("${PHASES[@]}")
  elif [[ "$1" == "--list" ]]; then
    printf '%s\n' "${PHASES[@]}"; exit 0
  elif [[ "$1" == "--help" || "$1" == "-h" ]]; then
    usage; exit 0
  elif [[ "$1" == "--from" ]]; then
    [[ $# -ge 2 ]] || c_die "--from 뒤에 단계 이름이 필요하다."
    local found=0
    for p in "${PHASES[@]}"; do
      [[ "$p" == "$2" ]] && found=1
      [[ $found -eq 1 ]] && run+=("$p")
    done
    [[ $found -eq 1 ]] || c_die "알 수 없는 단계: $2 (가능: ${PHASES[*]})"
  else
    for arg in "$@"; do
      printf '%s\n' "${PHASES[@]}" | grep -qx "$arg" \
        || c_die "알 수 없는 단계: $arg (가능: ${PHASES[*]})"
      run+=("$arg")
    done
  fi

  preflight "${run[@]}"
  for p in "${run[@]}"; do
    "phase_$p"
  done

  cat <<EOF

$(printf '\033[1;32m설치 완료\033[0m')

다음 순서로 진행할 것:

  1. 로그아웃 → 재로그인          (dialout 그룹 적용. 터미널만 닫으면 안 된다)
  2. ./setup/verify.sh            설치 확인
  3. PC NIC IP를 ${PC_IP} 로 설정   (매뉴얼 '네트워크' 절)
  4. 펌웨어 플래시                 (매뉴얼 '펌웨어' 절)
  5. rsrl                         셸에서 스택 활성화

EOF
}

main "$@"

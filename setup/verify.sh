#!/usr/bin/env bash
# =============================================================================
# 설치 확인 — 위에서부터 순서대로 검사한다.
# 처음 실패하는 지점이 원인 지점이다.
#
#   ./verify.sh          설치 상태만 확인 (하드웨어 불필요)
#   ./verify.sh --live   Teensy 연결·통신까지 확인 (전원 인가 + 에이전트 실행 상태)
# =============================================================================
set -uo pipefail   # -e 없음: 실패해도 끝까지 검사해 전체 그림을 보여준다

# ROS의 setup.bash는 nounset(set -u) 환경에서 깨진다(미정의 변수 참조).
src_ros() { set +u; # shellcheck disable=SC1090
            source "$1" 2>/dev/null; set -u; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.env
source "$SCRIPT_DIR/config.env"

PASS=0; FAIL=0
ok()   { printf '\033[32m  ✓\033[0m %-38s %s\n' "$1" "${2:-}"; PASS=$((PASS+1)); }
no()   { printf '\033[31m  ✗\033[0m %-38s \033[31m%s\033[0m\n' "$1" "${2:-}"; FAIL=$((FAIL+1)); }
warn() { printf '\033[33m  !\033[0m %-38s \033[33m%s\033[0m\n' "$1" "${2:-}"; FAIL=$((FAIL+1)); }
sec()  { printf '\n\033[1;36m━━ %s\033[0m\n' "$*"; }

sec "1. 토대"

if locale 2>/dev/null | grep -q 'UTF-8'; then
  ok "UTF-8 로케일"
else
  no "UTF-8 로케일" "locale-gen 필요 → setup.sh base"
fi

if [[ -d "/opt/ros/${ROS_DISTRO_NAME}" ]]; then
  src_ros "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
  ok "ROS 2 ${ROS_DISTRO_NAME}" "\$ROS_DISTRO=${ROS_DISTRO:-빈값}"
else
  no "ROS 2 ${ROS_DISTRO_NAME}" "미설치 → setup.sh ros"
fi

if command -v colcon >/dev/null; then
  # venv 안의 colcon은 venv를 끄면 사라진다. 시스템 설치인지 확인한다.
  if [[ "$(command -v colcon)" == /usr/bin/* ]]; then
    ok "colcon (시스템)" "$(command -v colcon)"
  else
    no "colcon 위치" "$(command -v colcon) — venv 안이면 deactivate 후 setup.sh ros"
  fi
else
  no "colcon" "미설치 → setup.sh ros"
fi

sec "2. PC 계층"

if [[ -f "${LIBTORCH_DIR}/build-version" ]]; then
  ver="$(cat "${LIBTORCH_DIR}/build-version")"
  if [[ "$ver" == *"+cpu"* || "$ver" == *"cu"* ]]; then
    ok "LibTorch" "$ver → ${LIBTORCH_DIR}"
  else
    ok "LibTorch" "$ver"
  fi
else
  no "LibTorch" "${LIBTORCH_DIR} 없음 → setup.sh libtorch"
fi

if [[ -d "${WORKSPACE}/install" ]]; then
  src_ros "${WORKSPACE}/install/setup.bash"
fi

if ros2 pkg executables sac_trainer_cpp 2>/dev/null | grep -q .; then
  ok "트레이너" "$(ros2 pkg executables sac_trainer_cpp | awk '{print $2}' | paste -sd' ')"
else
  no "트레이너" "빌드 안 됨 → setup.sh trainer"
fi

sec "3. 통신 계층"

if ros2 pkg executables micro_ros_agent 2>/dev/null | grep -q .; then
  ok "micro-ROS 에이전트"
else
  no "micro-ROS 에이전트" "빌드 안 됨 → setup.sh agent"
fi

sec "4. 펌웨어 계층"

if command -v pio >/dev/null || [[ -x "$HOME/.platformio/penv/bin/pio" ]]; then
  ok "PlatformIO" "$("${HOME}/.platformio/penv/bin/pio" --version 2>/dev/null || pio --version)"
else
  no "PlatformIO" "미설치 → setup.sh platformio"
fi

if [[ -f /etc/udev/rules.d/99-platformio-udev.rules ]]; then
  ok "udev 규칙"
else
  no "udev 규칙" "→ setup.sh platformio"
fi

# id -nG는 그룹 DB를, groups는 '현재 세션'을 본다. usermod만 하고 재로그인을
# 안 하면 둘이 어긋나고, 업로드는 세션 권한으로 동작하므로 실패한다.
# 이 차이를 구분하지 않으면 '설정은 됐는데 업로드가 안 되는' 상태를 통과로 오판한다.
for g in dialout plugdev; do
  if ! id -nG "$USER" | tr ' ' '\n' | grep -qx "$g"; then
    no "그룹 $g" "미설정 → setup.sh platformio"
  elif ! groups | tr ' ' '\n' | grep -qx "$g"; then
    warn "그룹 $g" "설정됐으나 현재 세션에 미적용 — 로그아웃 → 재로그인 필요"
  else
    ok "그룹 $g"
  fi
done

sec "5. 셸 환경"

if grep -q 'alias rsrl=' "$HOME/.bashrc" 2>/dev/null; then
  ok "rsrl alias"
else
  no "rsrl alias" "→ setup.sh shell"
fi

[[ -d "${VENV_DIR}" ]] && ok "venv" "${VENV_DIR}" || no "venv" "→ setup.sh shell"

# --- 하드웨어가 붙어 있을 때만 --------------------------------------------------
if [[ "${1:-}" == "--live" ]]; then
  sec "6. 실동작 (Teensy 전원 인가 + 에이전트 실행 상태)"

  if ip -4 addr show 2>/dev/null | grep -qF "${PC_IP}"; then
    ok "PC NIC IP" "${PC_IP}"
  else
    no "PC NIC IP" "${PC_IP} 가 어느 NIC에도 없음 — 펌웨어가 이 주소로 쏜다"
  fi

  if ping -c2 -W2 "${TEENSY_IP}" >/dev/null 2>&1; then
    ok "Teensy 응답" "${TEENSY_IP}"
  else
    no "Teensy 응답" "ping ${TEENSY_IP} 실패 — 이더넷 링크/펌웨어 확인"
  fi

  export ROS_DOMAIN_ID="${ROS_DOMAIN}"
  topics="$(timeout 5 ros2 topic list 2>/dev/null)"
  for t in /encoder_feedback /rl_observation; do
    if grep -qx "$t" <<<"$topics"; then
      ok "토픽 $t"
    else
      no "토픽 $t" "에이전트에 세션이 붙었는지, 도메인이 ${ROS_DOMAIN}인지 확인"
    fi
  done
fi

# --- 요약 --------------------------------------------------------------------
printf '\n'
if [[ $FAIL -eq 0 ]]; then
  printf '\033[1;32m전부 통과 (%d)\033[0m\n' "$PASS"
  exit 0
else
  printf '\033[1;31m%d개 실패\033[0m / %d개 통과 — 위에서 처음 ✗ 난 곳이 원인 지점이다.\n' "$FAIL" "$PASS"
  exit 1
fi

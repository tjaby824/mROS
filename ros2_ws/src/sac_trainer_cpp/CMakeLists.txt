cmake_minimum_required(VERSION 3.8)
project(sac_trainer_cpp)

# C++17 required
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# LibTorch
# set()이 아니라 list(APPEND)여야 한다. set이면 외부에서 넘긴
# -DCMAKE_PREFIX_PATH 가 조용히 덮여써져 setup.sh의 --cmake-args가 무효가 된다.
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include directories
include_directories(include)

# Dual SAC Trainer Node
add_executable(dual_sac
  src/sac_trainer_node.cpp
  src/sac_agent.cpp
  src/replay_buffer.cpp
)

ament_target_dependencies(dual_sac
  rclcpp
  std_msgs
)

target_link_libraries(dual_sac
  "${TORCH_LIBRARIES}"
)

# HiLH Node (Hardware-in-the-Learning-Harness) — Phase 1 reward harness
add_executable(hilh
  src/HiLH.cpp
  src/sac_agent.cpp
  src/replay_buffer.cpp
)

ament_target_dependencies(hilh
  rclcpp
  std_msgs
)

target_link_libraries(hilh
  "${TORCH_LIBRARIES}"
)

# Install
install(TARGETS
  dual_sac
  hilh
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  scripts/
  DESTINATION lib/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
)

ament_package()

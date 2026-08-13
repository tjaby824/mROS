#!/bin/bash

# RL-Only SAC Training (no rule-based episodes)
# Uses best policy checkpoint as starting point
# Usage: ./run_rl_only.sh [max_episodes] [difficulty] [sigma_deg] [checkpoint]

MAX_EPISODES=${1:-200}
DIFFICULTY=${2:-2}
SIGMA_DEG=${3:-10.0}
CHECKPOINT=${4:-"./sac_checkpoints/sac_ep50"}

export ROS_DOMAIN_ID=121
source ~/ros2_ws/install/setup.bash

echo "┌─── SAC RL-Only Training ─────────────────────────────┐"
printf "│ Episodes  %-6s Sigma    %-7s Curriculum  ON      │\n" "$MAX_EPISODES" "${SIGMA_DEG}°"
printf "│ Batch     256    Iters    5000    Entropy     0.005  │\n"
printf "│ Checkpoint %-42s│\n" "$CHECKPOINT"
echo "└──────────────────────────────────────────────────────┘"

ros2 run sac_trainer_cpp dual_sac --ros-args \
  -p max_episodes:=$MAX_EPISODES \
  -p difficulty:=$DIFFICULTY \
  -p sigma_deg:=$SIGMA_DEG \
  -p rl_only_mode:=true \
  -p load_buffer:=true \
  -p load_checkpoint:=$CHECKPOINT \
  -p target_entropy:=0.15 \
  -p train_iterations:=1500 \
  -p batch_size:=256 \
  -p save_dir:=./sac_checkpoints \
  -p csv_dir:=./episode_logs \
  -p buffer_dir:=./replay_buffers \
  -p buffer_save_interval:=50 \
  -p sigma_curriculum_enabled:=true \
  -p sigma_window:=10 \
  -p sigma_shrink_ratio:=0.7 \
  -p sigma_shrink_factor:=0.80 \
  -p sigma_min:=0.3

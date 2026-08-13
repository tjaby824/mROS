#!/bin/bash

# SAC Trainer Launch Script
# Usage: ./run_sac_train.sh [max_episodes] [difficulty] [train_after_collect] [checkpoint]

MAX_EPISODES=${1:-5000}
DIFFICULTY=${2:-4}
TRAIN_AFTER_COLLECT=${3:-0}
CHECKPOINT=${4:-""}
LOAD_BUFFER=${5:-"true"}
BUFFER_SAVE_INTERVAL=${6:-50}

echo "=========================================="
echo "SAC Training Configuration"
echo "=========================================="
echo "Max Episodes:       $MAX_EPISODES"
echo "Difficulty:         $DIFFICULTY"
echo "Train After:        $TRAIN_AFTER_COLLECT episodes"
echo "Load Buffer:        $LOAD_BUFFER"
echo "Buffer Save Int:    $BUFFER_SAVE_INTERVAL"
echo "Batch Size:         256"
echo "Train Iterations:   30000"
echo ""

if [ "$TRAIN_AFTER_COLLECT" -gt 0 ]; then
    echo "★ Episodes 1-${TRAIN_AFTER_COLLECT}: Data Collection (no training)"
    echo "★ Episodes $(($TRAIN_AFTER_COLLECT + 1)) +: Training Active"
else
    echo "★ Training Mode: Active from Episode 1"
fi
echo "=========================================="
echo ""

# Source ROS2 (adjust if needed)
source ~/ros2_ws/install/setup.bash

# Build parameter list
PARAMS=(
  "--ros-args"
  "-p" "max_episodes:=$MAX_EPISODES"
  "-p" "difficulty:=$DIFFICULTY"
  "-p" "train_after_collect:=$TRAIN_AFTER_COLLECT"
  "-p" "load_buffer:=$LOAD_BUFFER"
  "-p" "buffer_save_interval:=$BUFFER_SAVE_INTERVAL"
  "-p" "batch_size:=256"
  "-p" "train_iterations:=30000"
  "-p" "save_dir:=./sac_checkpoints"
  "-p" "csv_dir:=./episode_logs"
  "-p" "buffer_dir:=./replay_buffers"
)

# Add checkpoint parameter only if provided
if [ -n "$CHECKPOINT" ]; then
  echo "Loading checkpoint: $CHECKPOINT"
  PARAMS+=("-p" "load_checkpoint:=$CHECKPOINT")
fi

echo ""
echo "Starting SAC Trainer Node..."
echo "Command: ros2 run sac_trainer_cpp sac_trainer_node ${PARAMS[@]}"
echo ""

# Run trainer
ros2 run sac_trainer_cpp sac_trainer_node "${PARAMS[@]}"

#!/bin/bash

# Dual SAC Training Launch Script
# Rule-based (half-step) + RL alternating episodes
# Usage: ./run_dual_sac.sh [max_episodes] [difficulty] [checkpoint]

MAX_EPISODES=${1:-2000}
DIFFICULTY=${2:-2}
SIGMA_DEG=${3:-30.0}
CHECKPOINT=${4:-""}
LOAD_BUFFER=${5:-"false"}
RULE_PERIOD=${6:-2}      # 1 RULE every N episodes (10 = RULE:RL = 1:9)
TARGET_ENTROPY=${7:-0.7}  # Discrete SAC target entropy (1.386=50% of log16; lower=less random)
SYMMETRIC=${8:-"true"}    # RLPD: symmetric sampling (each batch = 50% RULE + 50% online)
UTD_RATIO=${9:-1}         # RLPD: update-to-data ratio (gradient steps multiplier)
# Pull-out soft-barrier penalty: penalty += w_pullout*max(0,|theta_e|-theta_safe)^2
# w derived by anchoring: w=1/(theta_ref-theta_safe)^2 so penalty==tracking_max(1) at theta_ref.
# theta_safe=128 (clean-RL p99), theta_ref=158 (halfway to 180 full-slip) -> w=1/30^2=1.1e-3.
# reward_scale=1.0 (default) so no rescale needed. Set W_PULLOUT=0 to disable.
W_PULLOUT=${10:-0.0}          # default OFF; set 0.001 to enable derived barrier
THETA_SAFE_DEG=${11:-128.0}

export ROS_DOMAIN_ID=121
source ~/ros2_ws/install/setup.bash

echo "=========================================="
echo "Dual SAC Training (Rule-based + RL)"
echo "=========================================="
echo "Max Episodes:       $MAX_EPISODES"
echo "Difficulty:         $DIFFICULTY"
echo "Sigma (deg):        $SIGMA_DEG"
echo "Load Buffer:        $LOAD_BUFFER"
echo "Train Iterations:   3000"
echo "Batch Size:         256"
echo ""
echo "Rule Period:        $RULE_PERIOD  (1 RULE every $RULE_PERIOD eps)"
echo "Target Entropy:     $TARGET_ENTROPY"
echo "Symmetric Sampling: $SYMMETRIC  (RLPD 50% RULE + 50% online)"
echo "UTD Ratio:          $UTD_RATIO"
echo "Pull-out penalty:   w=$W_PULLOUT  theta_safe=$THETA_SAFE_DEG deg  (0=off)"
echo "★ 1 RULE episode every $RULE_PERIOD episodes, rest are RL"
echo "★ RLPD: critic LayerNorm + symmetric expert replay"
echo "=========================================="
echo ""

PARAMS=(
  "--ros-args"
  "-p" "max_episodes:=$MAX_EPISODES"
  "-p" "difficulty:=$DIFFICULTY"
  "-p" "load_buffer:=$LOAD_BUFFER"
  "-p" "buffer_save_interval:=50"
  "-p" "batch_size:=256"
  "-p" "train_iterations:=5000"
  "-p" "save_dir:=./sac_checkpoints"
  "-p" "csv_dir:=./episode_logs"
  "-p" "buffer_dir:=./replay_buffers"
  "-p" "sigma_deg:=$SIGMA_DEG"
  "-p" "rule_period:=$RULE_PERIOD"
  "-p" "target_entropy:=$TARGET_ENTROPY"
  "-p" "symmetric_sampling:=$SYMMETRIC"
  "-p" "utd_ratio:=$UTD_RATIO"
  "-p" "w_pullout:=$W_PULLOUT"
  "-p" "theta_safe_deg:=$THETA_SAFE_DEG"
)

if [ -n "$CHECKPOINT" ]; then
  echo "Loading checkpoint: $CHECKPOINT"
  PARAMS+=("-p" "load_checkpoint:=$CHECKPOINT")
fi

echo "Starting Dual SAC Trainer..."
echo ""

ros2 run sac_trainer_cpp dual_sac "${PARAMS[@]}"

/**
 * SAC Trainer Node (Continuous 4D Action + Dynamic Reward Tuning)
 *
 * ★★★ Training runs in a separate thread ★★★
 *   → ros2 param set works DURING training (30k iterations)
 *   → sigma, lr, reward_type etc. can be changed mid-training
 *
 * ★★★ Reward Functions (switchable via reward_type param) ★★★
 *   Type 0: Gaussian + Hamming (default, matches SB3 sim)
 *   Type 1: Linear MAE
 *   Type 2: Huber-like
 *
 * Action Space: Continuous Box(4,) → Discrete (0-15)
 */

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int32.hpp>

#include "sac_trainer_cpp/sac_agent.hpp"
#include "sac_trainer_cpp/replay_buffer.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <atomic>
#include <mutex>
#include <iomanip>
#include <cmath>
#include <algorithm>

using namespace std::chrono_literals;

class SACTrainerNode : public rclcpp::Node {
public:
    SACTrainerNode() : Node("sac_trainer_node") {
        // ===== Static Parameters (set at launch) =====
        this->declare_parameter("max_episodes", 1000);
        this->declare_parameter("batch_size", 256);
        this->declare_parameter("save_dir", "./sac_checkpoints");
        this->declare_parameter("csv_dir", "./episode_logs");
        this->declare_parameter("buffer_dir", "./replay_buffers");
        this->declare_parameter("load_checkpoint", "");
        this->declare_parameter("load_buffer", true);
        this->declare_parameter("buffer_save_interval", 50);

        // ===== Dynamic Parameters (can be changed DURING training via ros2 param set) =====
        this->declare_parameter("train_iterations", 3000);   // Reduced: less overfitting on small buffer
        this->declare_parameter("reward_type", 0);          // 0=gaussian, 1=linear, 2=huber
        this->declare_parameter("sigma_deg", 45.0);         // Gaussian sigma (curriculum: start large)
        this->declare_parameter("w_change", 0.0);           // No hamming penalty (stepper needs action changes!)
        this->declare_parameter("reward_scale", 1.0);       // Scale reward before storing
        this->declare_parameter("error_cap_rad", 0.5);      // For linear reward: error cap
        this->declare_parameter("huber_delta", 0.1);        // For huber reward: delta
        this->declare_parameter("learning_rate", 3e-4);     // Can adjust lr mid-training
        this->declare_parameter("target_entropy", 1.39);    // 0.5*log(16) for discrete SAC
        this->declare_parameter("freeze_alpha", true);       // Cold start: hold alpha fixed (auto-entropy off)
        this->declare_parameter("frozen_alpha_value", 0.15); // Fixed alpha while frozen (negative = hold current)
        this->declare_parameter("use_expert_data", true);   // Add rule-based data to replay buffer
        this->declare_parameter("rl_only_mode", false);      // RL-only: skip rule-based episodes
        this->declare_parameter("rule_period", 2);           // 1 RULE every N episodes (2=1:1, 10=1:9)
        this->declare_parameter("symmetric_sampling", true); // RLPD: each batch = 50% RULE buffer + 50% online buffer
        this->declare_parameter("utd_ratio", 1);             // RLPD: update-to-data ratio (gradient steps multiplier)
        this->declare_parameter("vel_reward_weight", 0.0);   // Velocity tracking weight (0=disabled)
        this->declare_parameter("vel_sigma_deg", 30.0);      // Velocity reward sigma (deg/s)
        this->declare_parameter("w_pullout", 0.0);           // Pull-out soft-barrier weight (0=disabled). penalty += w*max(0,|theta_e|-theta_safe)^2
        this->declare_parameter("theta_safe_deg", 128.0);    // Load-angle safe threshold (deg). clean-RL p99; >90 electrical = pull-out
        this->declare_parameter("pullout_over_max_deg", 30.0); // Barrier saturation: (|theta_e|-theta_safe) clamped here so penalty<=w*over_max^2 (prevents critic blowup)

        // Auto sigma curriculum
        this->declare_parameter("sigma_curriculum_enabled", false);
        this->declare_parameter("sigma_window", 10);          // MAE moving average window
        this->declare_parameter("sigma_shrink_ratio", 0.7);   // Shrink when MAE < sigma * ratio
        this->declare_parameter("sigma_shrink_factor", 0.80);  // sigma *= factor
        this->declare_parameter("sigma_min", 0.5);            // Minimum sigma (degrees)

        max_episodes_ = this->get_parameter("max_episodes").as_int();
        batch_size_ = this->get_parameter("batch_size").as_int();
        save_dir_ = this->get_parameter("save_dir").as_string();
        csv_dir_ = this->get_parameter("csv_dir").as_string();
        buffer_dir_ = this->get_parameter("buffer_dir").as_string();
        load_checkpoint_ = this->get_parameter("load_checkpoint").as_string();
        load_buffer_ = this->get_parameter("load_buffer").as_bool();
        buffer_save_interval_ = this->get_parameter("buffer_save_interval").as_int();

        rl_only_mode_ = this->get_parameter("rl_only_mode").as_bool();
        rule_period_ = this->get_parameter("rule_period").as_int();
        if (rule_period_ < 1) rule_period_ = 1;

        symmetric_sampling_ = this->get_parameter("symmetric_sampling").as_bool();
        utd_ratio_ = this->get_parameter("utd_ratio").as_int();
        if (utd_ratio_ < 1) utd_ratio_ = 1;

        reload_dynamic_params();

        std::filesystem::create_directories(save_dir_);
        std::filesystem::create_directories(csv_dir_);
        std::filesystem::create_directories(buffer_dir_);

        // SAC Agent & Replay Buffer
        agent_ = std::make_unique<sac::SACAgent>();
        replay_buffer_ = std::make_unique<sac::ReplayBuffer>(1000000);
        // RLPD: dedicated RULE (expert) buffer, kept separate from online RL data
        rule_buffer_ = std::make_unique<sac::ReplayBuffer>(300000);

        // Apply target_entropy ROS param at init (agent ctor hardcodes 0.5*log|A|)
        agent_->set_target_entropy(target_entropy_);

        if (!load_checkpoint_.empty()) {
            RCLCPP_INFO(this->get_logger(), "[Load] checkpoint: %s", load_checkpoint_.c_str());
            agent_->load(load_checkpoint_);
            // ★ Sync target_entropy & alpha with ROS params immediately
            agent_->set_target_entropy(target_entropy_);
            use_pretrained_policy_ = true;
        }

        // Apply initial alpha freeze state (after checkpoint load so it takes precedence).
        // Frozen: hold alpha fixed so the (unreliable) cold-start critic can't drive
        // auto-entropy into collapse/burst. Unfreeze later via `ros2 param set freeze_alpha false`.
        if (freeze_alpha_) {
            if (frozen_alpha_value_ >= 0.0) {
                agent_->reset_alpha(frozen_alpha_value_);
            }
            agent_->set_auto_entropy(false);
        } else {
            agent_->set_auto_entropy(true);
        }

        if (load_buffer_) {
            load_all_buffers();
        }

        auto qos = rclcpp::QoS(10).best_effort();

        obs_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "rl_observation", qos,
            std::bind(&SACTrainerNode::obs_callback, this, std::placeholders::_1));

        encoder_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "encoder_feedback", qos,
            std::bind(&SACTrainerNode::encoder_callback, this, std::placeholders::_1));

        episode_cmd_teensy_sub_ = this->create_subscription<std_msgs::msg::Int32>(
            "episode_cmd_teensy", qos,
            std::bind(&SACTrainerNode::episode_cmd_callback, this, std::placeholders::_1));

        policy_ack_sub_ = this->create_subscription<std_msgs::msg::Float32>(
            "policy_ack", qos,
            std::bind(&SACTrainerNode::policy_ack_callback, this, std::placeholders::_1));

        episode_cmd_pc_pub_ = this->create_publisher<std_msgs::msg::Int32>(
            "episode_cmd_pc", qos);

        policy_chunk_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
            "policy_chunk", qos);

        // Dynamic parameter callback
        param_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&SACTrainerNode::on_param_change, this, std::placeholders::_1));

        // Timer to poll training completion (checks every 500ms)
        train_poll_timer_ = this->create_wall_timer(500ms,
            std::bind(&SACTrainerNode::check_training_done, this));

        startup_timer_ = this->create_wall_timer(
            3s, std::bind(&SACTrainerNode::start_first_episode, this));

        RCLCPP_INFO(this->get_logger(), "╔══ SAC Trainer ═══════════════════════════════════════════════╗");
        RCLCPP_INFO(this->get_logger(), "║ Mode      %-10s Sigma    %6.2f°   Curriculum  %-3s      ║",
            rl_only_mode_ ? "RL-Only" : "Dual", sigma_deg_, sigma_curriculum_enabled_ ? "ON" : "OFF");
        RCLCPP_INFO(this->get_logger(), "║ Reward    type=%-2d     w=%5.3f  scale=%4.2f  vel=%4.2f          ║",
            reward_type_, w_change_, reward_scale_, vel_reward_weight_);
        RCLCPP_INFO(this->get_logger(), "║ Alpha     freeze=%-3s  frozen_val=%5.2f  target_ent=%5.3f       ║",
            freeze_alpha_ ? "ON" : "OFF", frozen_alpha_value_, target_entropy_);
        RCLCPP_INFO(this->get_logger(), "║ Train     iters=%-5d batch=%-4d buf=%-6zu save=%-3d        ║",
            train_iterations_, batch_size_, replay_buffer_->size(), buffer_save_interval_);
        RCLCPP_INFO(this->get_logger(), "║ Curriculum  window=%-3d ratio=%4.2f  factor=%4.2f  min=%4.2f°    ║",
            sigma_window_, sigma_shrink_ratio_, sigma_shrink_factor_, sigma_min_);
        RCLCPP_INFO(this->get_logger(), "╚══════════════════════════════════════════════════════════════╝");
    }

    ~SACTrainerNode() {
        // Wait for training thread to finish
        if (train_thread_.joinable()) {
            train_thread_.join();
        }
    }

private:
    struct TelemetryRecord {
        double timestamp;
        float target_angle;
        float current_angle;
        float error;
        float error_dot;
        int action;
        float tracking;
        float penalty;
        float r_total;
        uint32_t policy_version;
        float remote_checksum;
        float expected_checksum;
        float teensy_time;
        float control_period_us;
        float inference_time_us;
        float isr_execution_time_us;
        float exp_delta_us;      // encoder[10]: commanded actuation phase Δ
        float load_angle_deg;    // encoder[11]: electrical load angle θ_e
        float desync_flag;       // encoder[12]: 1 if |θ_e|>90° (pull-out)
        float compute_us;        // encoder[13]: DWT measured compute time
    };

    void reload_dynamic_params() {
        train_iterations_ = this->get_parameter("train_iterations").as_int();
        reward_type_ = this->get_parameter("reward_type").as_int();
        sigma_deg_ = this->get_parameter("sigma_deg").as_double();
        sigma_rad_ = sigma_deg_ * M_PI / 180.0;
        w_change_ = this->get_parameter("w_change").as_double();
        reward_scale_ = this->get_parameter("reward_scale").as_double();
        error_cap_rad_ = this->get_parameter("error_cap_rad").as_double();
        huber_delta_ = this->get_parameter("huber_delta").as_double();
        learning_rate_ = this->get_parameter("learning_rate").as_double();
        target_entropy_ = this->get_parameter("target_entropy").as_double();
        freeze_alpha_ = this->get_parameter("freeze_alpha").as_bool();
        frozen_alpha_value_ = this->get_parameter("frozen_alpha_value").as_double();
        use_expert_data_ = this->get_parameter("use_expert_data").as_bool();
        vel_reward_weight_ = this->get_parameter("vel_reward_weight").as_double();
        vel_sigma_deg_ = this->get_parameter("vel_sigma_deg").as_double();
        vel_sigma_rad_ = vel_sigma_deg_ * M_PI / 180.0;
        w_pullout_ = this->get_parameter("w_pullout").as_double();
        theta_safe_deg_ = this->get_parameter("theta_safe_deg").as_double();
        pullout_over_max_deg_ = this->get_parameter("pullout_over_max_deg").as_double();

        sigma_curriculum_enabled_ = this->get_parameter("sigma_curriculum_enabled").as_bool();
        sigma_window_ = this->get_parameter("sigma_window").as_int();
        sigma_shrink_ratio_ = this->get_parameter("sigma_shrink_ratio").as_double();
        sigma_shrink_factor_ = this->get_parameter("sigma_shrink_factor").as_double();
        sigma_min_ = this->get_parameter("sigma_min").as_double();
    }

    rcl_interfaces::msg::SetParametersResult on_param_change(
        const std::vector<rclcpp::Parameter>& params)
    {
        auto result = rcl_interfaces::msg::SetParametersResult();
        result.successful = true;

        for (const auto& p : params) {
            const auto& name = p.get_name();

            if (name == "sigma_deg") {
                sigma_deg_ = p.as_double();
                sigma_rad_ = sigma_deg_ * M_PI / 180.0;
                RCLCPP_WARN(this->get_logger(), "[Param] sigma=%.2f°", sigma_deg_);
            } else if (name == "w_change") {
                w_change_ = p.as_double();
                RCLCPP_WARN(this->get_logger(), "[Param] w_change=%.4f", w_change_);
            } else if (name == "reward_type") {
                reward_type_ = p.as_int();
                RCLCPP_WARN(this->get_logger(), "[Param] reward_type=%d", reward_type_);
            } else if (name == "reward_scale") {
                reward_scale_ = p.as_double();
                RCLCPP_WARN(this->get_logger(), "[Param] reward_scale=%.2f", reward_scale_);
            } else if (name == "train_iterations") {
                train_iterations_ = p.as_int();
                RCLCPP_WARN(this->get_logger(), "[Param] train_iters=%d", train_iterations_);
            } else if (name == "error_cap_rad") {
                error_cap_rad_ = p.as_double();
                RCLCPP_WARN(this->get_logger(), "[Param] error_cap=%.4f", error_cap_rad_);
            } else if (name == "huber_delta") {
                huber_delta_ = p.as_double();
                RCLCPP_WARN(this->get_logger(), "[Param] huber_delta=%.4f", huber_delta_);
            } else if (name == "learning_rate") {
                learning_rate_ = p.as_double();
                lr_changed_.store(true);
                RCLCPP_WARN(this->get_logger(), "[Param] lr=%.6f", learning_rate_);
            } else if (name == "target_entropy") {
                target_entropy_ = p.as_double();
                te_changed_.store(true);
                RCLCPP_WARN(this->get_logger(), "[Param] target_entropy=%.4f", target_entropy_);
            } else if (name == "freeze_alpha") {
                freeze_alpha_ = p.as_bool();
                freeze_changed_.store(true);
                RCLCPP_WARN(this->get_logger(), "[Param] freeze_alpha=%s", freeze_alpha_ ? "ON" : "OFF");
            } else if (name == "frozen_alpha_value") {
                frozen_alpha_value_ = p.as_double();
                freeze_changed_.store(true);
                RCLCPP_WARN(this->get_logger(), "[Param] frozen_alpha_value=%.4f", frozen_alpha_value_);
            } else if (name == "use_expert_data") {
                use_expert_data_ = p.as_bool();
                RCLCPP_WARN(this->get_logger(), "[Param] expert_data=%s", use_expert_data_ ? "ON" : "OFF");
            } else if (name == "vel_reward_weight") {
                vel_reward_weight_ = p.as_double();
                RCLCPP_WARN(this->get_logger(), "[Param] vel_weight=%.3f", vel_reward_weight_);
            } else if (name == "vel_sigma_deg") {
                vel_sigma_deg_ = p.as_double();
                vel_sigma_rad_ = vel_sigma_deg_ * M_PI / 180.0;
                RCLCPP_WARN(this->get_logger(), "[Param] vel_sigma=%.1f°", vel_sigma_deg_);
            } else if (name == "w_pullout") {
                w_pullout_ = p.as_double();
                RCLCPP_WARN(this->get_logger(), "[Param] w_pullout=%.4f", w_pullout_);
            } else if (name == "theta_safe_deg") {
                theta_safe_deg_ = p.as_double();
                RCLCPP_WARN(this->get_logger(), "[Param] theta_safe=%.1f°", theta_safe_deg_);
            } else if (name == "pullout_over_max_deg") {
                pullout_over_max_deg_ = p.as_double();
                RCLCPP_WARN(this->get_logger(), "[Param] pullout_over_max=%.1f°", pullout_over_max_deg_);
            } else if (name == "sigma_curriculum_enabled") {
                sigma_curriculum_enabled_ = p.as_bool();
                RCLCPP_WARN(this->get_logger(), "[Param] curriculum=%s", sigma_curriculum_enabled_ ? "ON" : "OFF");
            } else if (name == "sigma_window") {
                sigma_window_ = p.as_int();
                RCLCPP_WARN(this->get_logger(), "[Param] sigma_window=%d", sigma_window_);
            } else if (name == "sigma_shrink_ratio") {
                sigma_shrink_ratio_ = p.as_double();
                RCLCPP_WARN(this->get_logger(), "[Param] shrink_ratio=%.2f", sigma_shrink_ratio_);
            } else if (name == "sigma_shrink_factor") {
                sigma_shrink_factor_ = p.as_double();
                RCLCPP_WARN(this->get_logger(), "[Param] shrink_factor=%.2f", sigma_shrink_factor_);
            } else if (name == "sigma_min") {
                sigma_min_ = p.as_double();
                RCLCPP_WARN(this->get_logger(), "[Param] sigma_min=%.2f°", sigma_min_);
            }
        }

        return result;
    }

    void maybe_update_sigma(float mae_deg) {
        if (!sigma_curriculum_enabled_) return;

        recent_mae_history_.push_back(mae_deg);

        if (static_cast<int>(recent_mae_history_.size()) < sigma_window_) return;

        // Compute moving average of last sigma_window_ episodes
        float sum = 0.0f;
        int start = recent_mae_history_.size() - sigma_window_;
        for (int i = start; i < static_cast<int>(recent_mae_history_.size()); i++) {
            sum += recent_mae_history_[i];
        }
        float avg_mae = sum / sigma_window_;

        float threshold = sigma_deg_ * sigma_shrink_ratio_;
        if (avg_mae < threshold) {
            double new_sigma = std::max(sigma_deg_ * sigma_shrink_factor_, sigma_min_);
            if (new_sigma < sigma_deg_) {
                double old_sigma = sigma_deg_;
                sigma_deg_ = new_sigma;
                sigma_rad_ = sigma_deg_ * M_PI / 180.0;
                recent_mae_history_.clear();

                // Alpha 리셋: sigma 축소 시 reward 급변으로 인한 alpha burst 방지
                double old_alpha = agent_->get_alpha();
                agent_->reset_alpha(0.1);

                RCLCPP_WARN(this->get_logger(),
                    "[Curriculum] sigma %.2f° -> %.2f°  (avg_MAE=%.2f° < thr=%.2f°)  alpha %.3f -> 0.100",
                    old_sigma, sigma_deg_, avg_mae, threshold, old_alpha);
            }
        }
    }

    int hamming_distance(uint8_t a, uint8_t b) {
        uint8_t xor_val = a ^ b;
        int count = 0;
        for (int i = 0; i < 4; i++) {
            if (xor_val & (1 << i)) count++;
        }
        return count;
    }

    float compute_reward(float error_rad, float vel_error_rad,
                         int prev_action, int curr_action,
                         float load_angle_deg,
                         float& tracking_out, float& penalty_out) {

        float abs_error = std::abs(error_rad);

        int ham = hamming_distance(static_cast<uint8_t>(prev_action),
                                   static_cast<uint8_t>(curr_action));
        penalty_out = static_cast<float>(w_change_) * ham;

        // Pull-out soft barrier (SATURATING): punish load angle beyond the safe well.
        // theta_e is the electrical load angle (rotor lag). |theta_e|>90 electrical = pull-out.
        // Quadratic barrier only bites past theta_safe (default 128deg = clean-RL p99),
        // so healthy transients are not penalized (avoids reward hacking).
        // CRITICAL: `over` is clamped to over_max so the penalty saturates. Without this,
        // post-pull-out theta_e winds to thousands of deg -> unbounded penalty -> critic blowup.
        // over_max = theta_ref - theta_safe (same anchor as w=1/(theta_ref-theta_safe)^2),
        // so max penalty = w*over_max^2 = 1.0 = tracking_max. Barrier brakes, never explodes.
        if (w_pullout_ > 1e-9) {
            float over = std::abs(load_angle_deg) - static_cast<float>(theta_safe_deg_);
            if (over > 0.0f) {
                float over_max = static_cast<float>(pullout_over_max_deg_);
                if (over > over_max) over = over_max;      // saturate
                penalty_out += static_cast<float>(w_pullout_) * over * over;
            }
        }

        // Position tracking reward
        float pos_tracking = 0.0f;
        switch (reward_type_) {
            case 0: {
                float normalized = error_rad / static_cast<float>(sigma_rad_ + 1e-9);
                pos_tracking = std::exp(-0.5f * normalized * normalized);
                break;
            }
            case 1: {
                pos_tracking = std::max(0.0f, 1.0f - abs_error / static_cast<float>(error_cap_rad_));
                break;
            }
            case 2: {
                float delta = static_cast<float>(huber_delta_);
                if (abs_error <= delta) {
                    pos_tracking = 1.0f - 0.5f * (abs_error / delta) * (abs_error / delta);
                } else {
                    pos_tracking = std::max(0.0f, 0.5f * delta / abs_error);
                }
                break;
            }
            default:
                float normalized = error_rad / static_cast<float>(sigma_rad_ + 1e-9);
                pos_tracking = std::exp(-0.5f * normalized * normalized);
                break;
        }

        // Velocity tracking reward (Gaussian, optional)
        if (vel_reward_weight_ > 1e-6) {
            float vel_norm = vel_error_rad / static_cast<float>(vel_sigma_rad_ + 1e-9);
            float vel_tracking = std::exp(-0.5f * vel_norm * vel_norm);
            float w_pos = 1.0f - static_cast<float>(vel_reward_weight_);
            tracking_out = w_pos * pos_tracking + static_cast<float>(vel_reward_weight_) * vel_tracking;
        } else {
            tracking_out = pos_tracking;
        }

        float reward = (tracking_out - penalty_out) * static_cast<float>(reward_scale_);
        return reward;
    }

    void load_all_buffers() {
        std::vector<std::string> buffer_files;

        if (std::filesystem::exists(buffer_dir_)) {
            for (const auto& entry : std::filesystem::directory_iterator(buffer_dir_)) {
                if (entry.path().extension() == ".buf") {
                    buffer_files.push_back(entry.path().string());
                }
            }
        }

        if (buffer_files.empty()) {
            RCLCPP_INFO(this->get_logger(), "[Buffer] no saved buffers in %s", buffer_dir_.c_str());
            return;
        }

        std::sort(buffer_files.begin(), buffer_files.end());
        RCLCPP_INFO(this->get_logger(), "[Buffer] found %zu files", buffer_files.size());

        replay_buffer_->load_and_merge(buffer_files);
        RCLCPP_INFO(this->get_logger(), "[Buffer] loaded  %zu transitions", replay_buffer_->size());
    }

    void save_buffer(int episode_num) {
        std::string filename = buffer_dir_ + "/buffer_ep" + std::to_string(episode_num) + ".buf";

        if (replay_buffer_->save(filename)) {
            RCLCPP_INFO(this->get_logger(), "[Buffer] saved   %s  %zu transitions",
                filename.c_str(), replay_buffer_->size());
        } else {
            RCLCPP_ERROR(this->get_logger(), "[Buffer] FAILED to save");
        }
    }

    void start_first_episode() {
        startup_timer_->cancel();

        if (use_pretrained_policy_) {
            deploy_policy_and_start();
        } else {
            send_start_command();
        }
    }

    void deploy_policy_and_start() {
        auto weights = agent_->get_policy_weights();
        size_t total_weights = weights.size();

        const int CHUNK_SIZE = 100;
        int total_chunks = (total_weights + CHUNK_SIZE - 1) / CHUNK_SIZE;

        RCLCPP_INFO(this->get_logger(), "[Deploy] v%d  %zu weights  %d chunks",
            policy_version_ + 1, total_weights, total_chunks);

        for (int chunk_idx = 0; chunk_idx < total_chunks; chunk_idx++) {
            size_t start = chunk_idx * CHUNK_SIZE;
            size_t end = std::min(start + static_cast<size_t>(CHUNK_SIZE), total_weights);

            auto msg = std_msgs::msg::Float32MultiArray();
            msg.data.push_back(static_cast<float>(chunk_idx));
            msg.data.push_back(static_cast<float>(total_chunks));
            msg.data.insert(msg.data.end(), weights.begin() + start, weights.begin() + end);

            policy_chunk_pub_->publish(msg);
            std::this_thread::sleep_for(50ms);
        }

        policy_version_++;

        next_episode_timer_ = this->create_wall_timer(2s, [this]() {
            next_episode_timer_->cancel();
            send_start_command();
        });
    }

    void send_start_command() {
        if (episode_count_ >= max_episodes_) {
            RCLCPP_INFO(this->get_logger(), "[Done] Training completed  %d episodes", max_episodes_);
            agent_->save(save_dir_ + "/sac_final");
            save_buffer(episode_count_);
            save_summary_csv();
            return;
        }

        episode_data_.clear();
        telemetry_data_.clear();
        last_obs_.clear();
        last_action_discrete_ = 0;
        step_count_ = 0;
        episode_reward_ = 0.0;
        episode_running_ = false;
        current_encoder_.clear();

        episode_start_time_ = std::chrono::steady_clock::now();

        auto msg = std_msgs::msg::Int32();

        // Episode type selection
        if (rl_only_mode_) {
            current_episode_type_ = RL_EPISODE;
        } else {
            // Periodic: 1 RULE episode every rule_period_ episodes, rest are RL.
            // episode_count_ starts at 0, so with rule_period_=10:
            //   ep1 (count=0) = RULE, ep2..ep10 (count=1..9) = RL, ep11 (count=10) = RULE, ...
            current_episode_type_ = (episode_count_ % rule_period_ == 0) ? RULE_BASED_EPISODE : RL_EPISODE;
        }

        if (current_episode_type_ == RULE_BASED_EPISODE) {
            msg.data = 2;  // cmd=2: Rule-based half-step
            RCLCPP_INFO(this->get_logger(),
                "[Ep%3d] START  RULE  buf=%zu",
                episode_count_ + 1, replay_buffer_->size());
        } else {
            msg.data = 0;  // cmd=0: RL policy
            RCLCPP_INFO(this->get_logger(),
                "[Ep%3d] START  RL    buf=%zu  sigma=%.2f°  alpha=%.4f  v%d",
                episode_count_ + 1, replay_buffer_->size(),
                sigma_deg_, agent_->get_alpha(), policy_version_);
        }

        episode_cmd_pc_pub_->publish(msg);
        episode_running_ = true;

        retry_timer_ = this->create_wall_timer(120s, [this]() {
            retry_timer_->cancel();
            if (episode_running_) {
                RCLCPP_WARN(this->get_logger(), "[Ep%3d] TIMEOUT  retrying...", episode_count_ + 1);
                send_start_command();
            }
        });
    }

    void obs_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
        if (!episode_running_) return;
        if (msg->data.size() < static_cast<size_t>(sac::OBS_SIZE)) return;

        std::vector<float> obs(msg->data.begin(), msg->data.begin() + sac::OBS_SIZE);

        int action_discrete = last_action_discrete_;

        float current_angle = 0.0f;
        float target_angle = 0.0f;
        float error = 0.0f;
        float error_dot = 0.0f;

        if (!current_encoder_.empty() && current_encoder_.size() >= 10) {
            current_angle = current_encoder_[0];
            target_angle = current_encoder_[2];
            error = current_angle - target_angle;

            float actual_velocity = current_encoder_[1];
            float target_velocity = current_encoder_[3];
            error_dot = actual_velocity - target_velocity;

            if (current_encoder_.size() > 4) {
                action_discrete = static_cast<int>(current_encoder_[4]);
            }
        }

        float vel_error = 0.0f;
        if (!current_encoder_.empty() && current_encoder_.size() >= 10) {
            float actual_velocity = current_encoder_[1];
            float target_velocity = current_encoder_[3];
            vel_error = actual_velocity - target_velocity;
        }

        // Load angle (theta_e) must be read before compute_reward (pull-out barrier needs it).
        float load_angle_deg = 0.0f;
        if (current_encoder_.size() >= 14) {
            load_angle_deg = current_encoder_[11];
        }

        float tracking = 0.0f, penalty = 0.0f;
        float reward = compute_reward(error, vel_error, last_action_discrete_, action_discrete, load_angle_deg, tracking, penalty);

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - episode_start_time_).count();

        float remote_checksum = 0.0f;
        float teensy_time = 0.0f;
        float control_period_us = 0.0f;
        float inference_time_us = 0.0f;
        float isr_execution_time_us = 0.0f;
        float exp_delta_us = 0.0f;
        float desync_flag = 0.0f;
        float compute_us = 0.0f;
        if (current_encoder_.size() >= 10) {
            remote_checksum = current_encoder_[9];
        }
        if (current_encoder_.size() >= 7) {
            teensy_time = current_encoder_[6];
        }
        if (current_encoder_.size() >= 9) {
            control_period_us = current_encoder_[5];
            inference_time_us = current_encoder_[7];
            isr_execution_time_us = current_encoder_[8];
        }
        if (current_encoder_.size() >= 14) {
            exp_delta_us = current_encoder_[10];
            desync_flag = current_encoder_[12];
            compute_us = current_encoder_[13];
        }

        TelemetryRecord record;
        record.timestamp = elapsed;
        record.target_angle = target_angle;
        record.current_angle = current_angle;
        record.error = error;
        record.error_dot = error_dot;
        record.action = action_discrete;
        record.tracking = tracking;
        record.penalty = penalty;
        record.r_total = reward;
        record.policy_version = policy_version_;
        record.remote_checksum = remote_checksum;
        record.expected_checksum = 0.0f;
        record.teensy_time = teensy_time;
        record.control_period_us = control_period_us;
        record.inference_time_us = inference_time_us;
        record.isr_execution_time_us = isr_execution_time_us;
        record.exp_delta_us = exp_delta_us;
        record.load_angle_deg = load_angle_deg;
        record.desync_flag = desync_flag;
        record.compute_us = compute_us;
        telemetry_data_.push_back(record);

        if (!last_obs_.empty()) {
            episode_data_.push_back({last_obs_, last_action_discrete_, reward, obs, false});
        }

        last_obs_ = obs;
        last_action_discrete_ = action_discrete;
        step_count_++;
        episode_reward_ += reward;
    }

    void encoder_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
        current_encoder_ = std::vector<float>(msg->data.begin(), msg->data.end());
    }

    void episode_cmd_callback(const std_msgs::msg::Int32::SharedPtr msg) {
        int cmd = msg->data;

        if (cmd == 1 && episode_running_) {
            if (retry_timer_) retry_timer_->cancel();
            episode_running_ = false;

            // Trim data to chirp duration (30s) — exclude settling time
            const float CHIRP_DURATION = 30.0f;
            size_t trim_idx = telemetry_data_.size();
            for (size_t i = 0; i < telemetry_data_.size(); i++) {
                if (telemetry_data_[i].teensy_time > CHIRP_DURATION) {
                    trim_idx = i;
                    break;
                }
            }
            if (trim_idx < telemetry_data_.size()) {
                size_t removed = telemetry_data_.size() - trim_idx;
                telemetry_data_.resize(trim_idx);
                // episode_data_ has 1 fewer entry than telemetry (first step has no prev obs)
                // trim_idx in telemetry corresponds to trim_idx-1 in episode_data
                size_t ep_trim = (trim_idx > 0) ? (trim_idx - 1) : 0;
                if (episode_data_.size() > ep_trim) {
                    episode_data_.resize(ep_trim);
                }
                RCLCPP_INFO(this->get_logger(),
                    "[Trim] removed %zu settling steps (kept %zu telem, %zu transitions @ %.1fs)",
                    removed, trim_idx, ep_trim, CHIRP_DURATION);
            }

            if (!episode_data_.empty()) {
                episode_data_.back().done = true;
            }

            // Calculate statistics (common for both types)
            float avg_error = 0.0f;
            float avg_tracking = 0.0f, avg_penalty = 0.0f;
            int total_hamming = 0;

            if (!telemetry_data_.empty()) {
                for (size_t i = 0; i < telemetry_data_.size(); i++) {
                    const auto& r = telemetry_data_[i];
                    avg_error += std::abs(r.error);
                    avg_tracking += r.tracking;
                    avg_penalty += r.penalty;

                    if (i > 0) {
                        int hamming = hamming_distance(
                            static_cast<uint8_t>(telemetry_data_[i-1].action),
                            static_cast<uint8_t>(r.action)
                        );
                        total_hamming += hamming;
                    }
                }
                size_t n = telemetry_data_.size();
                avg_error /= n;
                avg_tracking /= n;
                avg_penalty /= n;
            }

            float avg_hamming = telemetry_data_.size() > 1 ?
                (float)total_hamming / (telemetry_data_.size() - 1) : 0.0f;

            float mae_deg = avg_error * 180.0f / M_PI;

            save_episode_csv(episode_count_ + 1);

            EpisodeSummary summary;
            summary.episode = episode_count_ + 1;
            summary.steps = step_count_;
            summary.total_reward = episode_reward_;
            summary.avg_error = avg_error;
            summary.mae_deg = mae_deg;
            summary.avg_tracking = avg_tracking;
            summary.avg_penalty = avg_penalty;
            summary.avg_hamming = avg_hamming;
            summary.policy_version = policy_version_;
            summary.alpha = agent_->get_alpha();
            summary.sigma_deg = sigma_deg_;
            summary.w_change = w_change_;
            summary.reward_type = reward_type_;
            summary.episode_type = static_cast<int>(current_episode_type_);
            episode_summaries_.push_back(summary);

            // ========== DUAL-EPISODE LOGIC ==========
            if (current_episode_type_ == RULE_BASED_EPISODE) {
                // Rule-based episode finished → save baseline
                rule_based_mae_deg_ = mae_deg;

                RCLCPP_INFO(this->get_logger(),
                    "[Ep%3d] DONE   RULE  MAE=%6.2f°  track=%.3f  hamming=%.2f",
                    episode_count_ + 1, mae_deg, avg_tracking, avg_hamming);

                // Optionally add expert (RULE) data to replay buffer.
                // RLPD symmetric mode: keep RULE data in a SEPARATE buffer so online
                // RL data never dilutes the expert set (each batch mixes 50/50 later).
                if (use_expert_data_) {
                    if (symmetric_sampling_) {
                        for (const auto& t : episode_data_) {
                            rule_buffer_->push(t.state, t.action, t.reward, t.next_state, t.done);
                        }
                    } else {
                        for (const auto& t : episode_data_) {
                            replay_buffer_->push(t.state, t.action, t.reward, t.next_state, t.done);
                        }
                    }
                }

                episode_count_++;

                // Next: RL episode (deploy policy first, no training)
                deploy_policy_and_start();

            } else {
                // RL episode finished → compare with rule-based baseline
                for (const auto& t : episode_data_) {
                    replay_buffer_->push(t.state, t.action, t.reward, t.next_state, t.done);
                }

                float rl_mae_deg = mae_deg;

                if (!rl_only_mode_ && rule_based_mae_deg_ > 0) {
                    float delta = rule_based_mae_deg_ - rl_mae_deg;
                    bool rl_wins = rl_mae_deg < rule_based_mae_deg_;

                    RCLCPP_WARN(this->get_logger(),
                        "[Ep%3d] >>>    RULE=%5.2f° vs RL=%5.2f°  d=%+.2f°  %s",
                        episode_count_ + 1, rule_based_mae_deg_, rl_mae_deg, delta,
                        rl_wins ? "RL WINS" : "RULE WINS");
                }

                RCLCPP_INFO(this->get_logger(),
                    "[Ep%3d] DONE   RL    MAE=%6.2f°  track=%.3f  hamming=%.2f  alpha=%.4f  sigma=%.2f°",
                    episode_count_ + 1, rl_mae_deg, avg_tracking, avg_hamming,
                    agent_->get_alpha(), sigma_deg_);

                // Auto sigma curriculum check
                maybe_update_sigma(rl_mae_deg);

                episode_count_++;

                if (episode_count_ % buffer_save_interval_ == 0) {
                    save_buffer(episode_count_);
                }

                // Train then start next rule-based episode
                start_async_training();
            }
        }
    }

    /**
     * ★★★ Async Training: runs in a separate thread ★★★
     * Main thread continues to spin (handles param set, subscriptions, etc.)
     */
    void start_async_training() {
        // Join any previous thread
        if (train_thread_.joinable()) {
            train_thread_.join();
        }

        training_in_progress_.store(true);
        waiting_for_training_ = true;  // Set immediately to avoid race condition when iters=0

        // Capture current training params (snapshot)
        int iters = train_iterations_;
        int bs = batch_size_;
        int ep = episode_count_;

        train_thread_ = std::thread([this, iters, bs, ep]() {
            train_sac_thread(iters, bs, ep);
        });
    }

    /**
     * Training function that runs in a separate thread
     */
    void train_sac_thread(int iterations, int batch_size, int episode_num) {
        const size_t LEARNING_STARTS = 1000;
        if (replay_buffer_->size() < LEARNING_STARTS) {
            std::cout << "[Train] warmup  buf=" << replay_buffer_->size()
                      << "/" << LEARNING_STARTS << "  skip" << std::endl;
            training_in_progress_.store(false);
            return;
        }
        if (replay_buffer_->size() < static_cast<size_t>(batch_size)) {
            std::cout << "[Train] skip    buf=" << replay_buffer_->size()
                      << " < batch=" << batch_size << std::endl;
            training_in_progress_.store(false);
            return;
        }
        // RLPD symmetric mode: warn if RULE buffer is thin (still works via sampling
        // with replacement, but low diversity in the expert half).
        if (symmetric_sampling_ && rule_buffer_->size() > 0 &&
            rule_buffer_->size() < static_cast<size_t>(batch_size / 2)) {
            std::cout << "[Train] note  rule_buf=" << rule_buffer_->size()
                      << " < half_batch=" << (batch_size / 2)
                      << "  (expert half sampled with replacement)" << std::endl;
        }

        // RLPD: total gradient steps = iterations * UTD ratio
        const int total_steps = iterations * utd_ratio_;
        // Symmetric batch split: half from RULE (expert) buffer, half from online
        const bool use_sym = symmetric_sampling_ && (rule_buffer_->size() > 0);
        const size_t rule_half = use_sym ? (batch_size / 2) : 0;
        const size_t online_half = batch_size - rule_half;

        auto t_start = std::chrono::steady_clock::now();
        std::cout << "[Train] start   steps=" << total_steps
                  << " (iters=" << iterations << " x utd=" << utd_ratio_ << ")"
                  << "  online=" << replay_buffer_->size()
                  << "  rule=" << rule_buffer_->size()
                  << (use_sym ? "  [SYM 50/50]" : "  [single]") << std::endl;

        std::vector<std::vector<float>> states, next_states;
        std::vector<int> actions;
        std::vector<float> rewards, dones;

        for (int i = 0; i < total_steps; i++) {
            // Check for dynamic lr/target_entropy changes
            if (lr_changed_.exchange(false)) {
                agent_->set_learning_rate(learning_rate_);
            }
            if (te_changed_.exchange(false)) {
                agent_->set_target_entropy(target_entropy_);
            }
            if (freeze_changed_.exchange(false)) {
                if (freeze_alpha_) {
                    if (frozen_alpha_value_ >= 0.0) agent_->reset_alpha(frozen_alpha_value_);
                    agent_->set_auto_entropy(false);   // freeze: skip alpha update
                } else {
                    agent_->set_auto_entropy(true);    // unfreeze: resume from current alpha
                }
            }

            if (use_sym) {
                // Build mixed batch: rule_half from RULE buffer + online_half from online buffer
                states.clear(); actions.clear(); rewards.clear();
                next_states.clear(); dones.clear();
                rule_buffer_->sample_append(rule_half, states, actions, rewards, next_states, dones);
                replay_buffer_->sample_append(online_half, states, actions, rewards, next_states, dones);
                if (states.size() == static_cast<size_t>(batch_size)) {
                    agent_->update(states, actions, rewards, next_states, dones);
                }
            } else {
                if (replay_buffer_->sample(batch_size, states, actions, rewards, next_states, dones)) {
                    agent_->update(states, actions, rewards, next_states, dones);
                }
            }

            // Progress log every 10000 steps
            if ((i + 1) % 10000 == 0) {
                std::cout << "[Train] " << (i + 1) << "/" << total_steps
                          << "  alpha=" << agent_->get_alpha() << std::endl;
            }
        }

        auto t_end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();

        auto st = agent_->get_last_stats();
        std::cout << "[Train] done    " << std::fixed << std::setprecision(1) << elapsed
                  << "s  alpha=" << agent_->get_alpha()
                  << "  steps=" << agent_->get_training_steps() << std::endl;
        std::cout << "[Diag]  closs=" << std::setprecision(4) << st.critic_loss
                  << "  Qmean=" << st.q_mean << "  Qmax=" << st.q_max
                  << "  tgtQ=" << st.target_q_mean
                  << "  rew=" << st.reward_mean
                  << "  ent=" << st.entropy_mean
                  << "  (tgt_ent=" << target_entropy_ << ")" << std::endl;

        // Save checkpoint
        if (episode_num % 10 == 0) {
            agent_->save(save_dir_ + "/sac_ep" + std::to_string(episode_num));
        }

        training_in_progress_.store(false);
    }

    /**
     * Timer callback: checks if async training is done, then deploys policy
     */
    void check_training_done() {
        if (waiting_for_training_ && !training_in_progress_.load()) {
            waiting_for_training_ = false;

            // Join training thread
            if (train_thread_.joinable()) {
                train_thread_.join();
            }

            if (rl_only_mode_) {
                RCLCPP_INFO(this->get_logger(), "[Train] deploy -> next RL");
                deploy_policy_and_start();
            } else {
                RCLCPP_INFO(this->get_logger(), "[Train] done -> next RULE");
                // Next episode is rule-based → no policy deploy needed, just start
                send_start_command();
            }
        }

        // Set waiting flag when training starts
        if (training_in_progress_.load() && !waiting_for_training_) {
            waiting_for_training_ = true;
        }
    }

    void save_episode_csv(int episode_num) {
        std::string filename = csv_dir_ + "/episode_" +
            std::to_string(episode_num) + ".csv";

        std::ofstream file(filename);
        if (!file.is_open()) return;

        file << "timestamp,target_angle,current_angle,error,error_dot,"
             << "action,tracking,penalty,r_total,"
             << "policy_version,remote_checksum,expected_checksum,teensy_time,"
             << "control_period_us,inference_time_us,isr_execution_time_us,"
             << "exp_delta_us,load_angle_deg,desync_flag,compute_us\n";

        file << std::fixed << std::setprecision(6);
        for (const auto& r : telemetry_data_) {
            file << r.timestamp << ","
                 << r.target_angle << ","
                 << r.current_angle << ","
                 << r.error << ","
                 << r.error_dot << ","
                 << r.action << ","
                 << r.tracking << ","
                 << r.penalty << ","
                 << r.r_total << ","
                 << r.policy_version << ","
                 << r.remote_checksum << ","
                 << r.expected_checksum << ","
                 << r.teensy_time << ","
                 << r.control_period_us << ","
                 << r.inference_time_us << ","
                 << r.isr_execution_time_us << ","
                 << r.exp_delta_us << ","
                 << r.load_angle_deg << ","
                 << r.desync_flag << ","
                 << r.compute_us << "\n";
        }

        file.close();
    }

    void save_summary_csv() {
        std::string filename = csv_dir_ + "/training_summary.csv";

        std::ofstream file(filename);
        if (!file.is_open()) return;

        file << "episode,steps,total_reward,avg_error_rad,mae_deg,"
             << "avg_tracking,avg_penalty,avg_hamming,"
             << "policy_version,alpha,sigma_deg,w_change,reward_type,episode_type\n";

        file << std::fixed << std::setprecision(6);
        for (const auto& s : episode_summaries_) {
            file << s.episode << ","
                 << s.steps << ","
                 << s.total_reward << ","
                 << s.avg_error << ","
                 << s.mae_deg << ","
                 << s.avg_tracking << ","
                 << s.avg_penalty << ","
                 << s.avg_hamming << ","
                 << s.policy_version << ","
                 << s.alpha << ","
                 << s.sigma_deg << ","
                 << s.w_change << ","
                 << s.reward_type << ","
                 << s.episode_type << "\n";
        }

        file.close();
        RCLCPP_INFO(this->get_logger(), "[Save] %s", filename.c_str());
    }

    void deploy_policy() {
        auto weights = agent_->get_policy_weights();
        size_t total_weights = weights.size();

        const int CHUNK_SIZE = 100;
        int total_chunks = (total_weights + CHUNK_SIZE - 1) / CHUNK_SIZE;

        RCLCPP_INFO(this->get_logger(), "[Deploy] v%d  %zu weights",
            policy_version_ + 1, total_weights);

        for (int chunk_idx = 0; chunk_idx < total_chunks; chunk_idx++) {
            size_t start = chunk_idx * CHUNK_SIZE;
            size_t end = std::min(start + static_cast<size_t>(CHUNK_SIZE), total_weights);

            auto msg = std_msgs::msg::Float32MultiArray();
            msg.data.push_back(static_cast<float>(chunk_idx));
            msg.data.push_back(static_cast<float>(total_chunks));
            msg.data.insert(msg.data.end(), weights.begin() + start, weights.begin() + end);

            policy_chunk_pub_->publish(msg);
            std::this_thread::sleep_for(50ms);
        }

        policy_version_++;

        next_episode_timer_ = this->create_wall_timer(2s, [this]() {
            next_episode_timer_->cancel();
            if (!episode_running_ && episode_count_ < max_episodes_) {
                send_start_command();
            }
        });
    }

    void policy_ack_callback(const std_msgs::msg::Float32::SharedPtr msg) {
        float value = msg->data;
        if (value > 0) {
            RCLCPP_INFO(this->get_logger(), "[Teensy] policy v%d confirmed", (int)value);
        }
    }

    struct Transition {
        std::vector<float> state;
        int action;  // discrete 0-15
        float reward;
        std::vector<float> next_state;
        bool done;
    };

    struct EpisodeSummary {
        int episode;
        int steps;
        double total_reward;
        float avg_error;
        float mae_deg;
        float avg_tracking;
        float avg_penalty;
        float avg_hamming;
        int policy_version;
        double alpha;
        double sigma_deg;
        double w_change;
        int reward_type;
        int episode_type;  // 0=RL, 1=rule-based
    };

    std::unique_ptr<sac::SACAgent> agent_;
    std::unique_ptr<sac::ReplayBuffer> replay_buffer_;
    std::unique_ptr<sac::ReplayBuffer> rule_buffer_;  // RLPD: dedicated expert (RULE) buffer

    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr obs_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr encoder_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr episode_cmd_teensy_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr policy_ack_sub_;

    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr episode_cmd_pc_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr policy_chunk_pub_;

    rclcpp::TimerBase::SharedPtr startup_timer_;
    rclcpp::TimerBase::SharedPtr next_episode_timer_;
    rclcpp::TimerBase::SharedPtr retry_timer_;
    rclcpp::TimerBase::SharedPtr train_poll_timer_;

    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;

    // Async training
    std::thread train_thread_;
    std::atomic<bool> training_in_progress_{false};
    bool waiting_for_training_ = false;
    std::atomic<bool> lr_changed_{false};
    std::atomic<bool> te_changed_{false};
    std::atomic<bool> freeze_changed_{false};

    int max_episodes_;
    int batch_size_;
    std::string save_dir_;
    std::string csv_dir_;
    std::string buffer_dir_;
    std::string load_checkpoint_;
    bool use_pretrained_policy_ = false;
    bool load_buffer_ = true;
    int buffer_save_interval_ = 50;

    // Dynamic parameters (changeable via ros2 param set ANYTIME)
    int train_iterations_ = 30000;
    int reward_type_ = 0;
    double sigma_deg_ = 45.0;
    double sigma_rad_ = 45.0 * M_PI / 180.0;
    double w_change_ = 0.02;
    double reward_scale_ = 1.0;
    double error_cap_rad_ = 0.5;
    double huber_delta_ = 0.1;
    double learning_rate_ = 3e-4;
    double target_entropy_ = 1.39;  // 0.5*log(16) for discrete SAC
    bool freeze_alpha_ = true;              // Cold start: hold alpha fixed (auto-entropy off)
    double frozen_alpha_value_ = 0.15;      // Fixed alpha while frozen (negative = hold current)

    // Velocity reward
    double vel_reward_weight_ = 0.0;
    double vel_sigma_deg_ = 30.0;
    double vel_sigma_rad_ = 30.0 * M_PI / 180.0;

    // Pull-out (load-angle) soft-barrier penalty
    double w_pullout_ = 0.0;                 // 0=disabled
    double theta_safe_deg_ = 128.0;          // clean-RL p99; |theta_e|>90 electrical = pull-out
    double pullout_over_max_deg_ = 30.0;     // barrier saturation: over clamped here; max penalty = w*over_max^2 = 1

    // Auto sigma curriculum
    bool sigma_curriculum_enabled_ = false;
    int sigma_window_ = 10;
    double sigma_shrink_ratio_ = 0.7;
    double sigma_shrink_factor_ = 0.80;
    double sigma_min_ = 0.5;
    std::vector<float> recent_mae_history_;

    // Dual-episode system
    enum EpisodeType { RL_EPISODE = 0, RULE_BASED_EPISODE = 1 };
    EpisodeType current_episode_type_ = RULE_BASED_EPISODE;
    float rule_based_mae_deg_ = -1.0f;
    bool use_expert_data_ = true;
    bool rl_only_mode_ = false;
    int rule_period_ = 2;
    bool symmetric_sampling_ = true;  // RLPD: 50% RULE + 50% online per batch
    int utd_ratio_ = 1;               // RLPD: update-to-data ratio

    int episode_count_ = 0;
    int policy_version_ = 0;
    bool episode_running_ = false;

    std::vector<Transition> episode_data_;
    std::vector<TelemetryRecord> telemetry_data_;
    std::vector<EpisodeSummary> episode_summaries_;

    std::vector<float> last_obs_;
    std::vector<float> current_encoder_;
    int last_action_discrete_ = 0;
    int step_count_ = 0;
    double episode_reward_ = 0.0;

    std::chrono::steady_clock::time_point episode_start_time_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SACTrainerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

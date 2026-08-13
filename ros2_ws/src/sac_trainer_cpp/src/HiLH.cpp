/**
 * HiLH — Hardware-in-the-Learning-Harness (Discrete SAC Trainer)
 *
 * Derived from sac_trainer_node.cpp. Re-architects the hardcoded reward switch
 * into a hot-swappable, data-driven "Reward Harness" (HarnessX D4 + HarnessForge
 * Archive-Guided variant management), controllable at runtime via ROS params and
 * file hot-swap so an external LLM (Claude Code, via ROS_MCP) can author/select
 * reward specifications without recompilation.
 *
 * ===== PHASE 1 SCOPE =====
 *   - compute_reward() is now a weighted sum over a closed set of named terms
 *     (pos_tracking / vel_tracking / hamming / settling / disturbance_rejection /
 *      energy). The LLM composes terms + weights; it cannot inject formulas
 *     (closed vocabulary = safety boundary).
 *   - Reward specs are flat key=value files (.rwd) parsed by a self-contained
 *     parser (NO new build dependency).
 *   - Multiple harnesses are registered as named "variants"; each tracks its own
 *     MAE leaderboard (last/ema/best/delta).
 *   - Harness swaps happen ONLY at episode boundaries (one episode == one
 *     spec_version) so a buffer transition is never computed with a mixed reward.
 *   - Status/events are published on std_msgs/String topics for the LLM loop.
 *
 * ===== INVARIANTS =====
 *   - default harness reproduces the original compute_reward (switch 0/1/2 +
 *     vel + hamming + scale) BYTE-IDENTICALLY (regression guarantee vs dual_sac).
 *   - get_policy_weights() / Teensy packet format unchanged (25744 floats).
 *   - reward/hooks run only on the main spin thread (obs_callback); the bg
 *     training thread reads frozen buffer rewards only (never recomputes).
 *
 * Node name: hilh_node   Executable: hilh
 */

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/string.hpp>

#include "sac_trainer_cpp/sac_agent.hpp"
#include "sac_trainer_cpp/replay_buffer.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>
#include <atomic>
#include <mutex>
#include <map>
#include <array>
#include <memory>
#include <functional>
#include <iomanip>
#include <cmath>
#include <algorithm>

using namespace std::chrono_literals;

// ==================== Reward Harness data model (Phase 1) ====================
// Closed-vocabulary reward terms. The LLM may only choose terms + parameters
// from this set; arbitrary expressions cannot be injected.
enum class TermType {
    PosTracking,
    VelTracking,
    Hamming,
    Settling,
    DisturbanceRejection,
    Energy
};

enum class PosKernel { Gaussian, Linear, Huber };

struct RewardTerm {
    TermType type = TermType::PosTracking;
    float weight = 0.0f;
    PosKernel kernel = PosKernel::Gaussian;   // pos_tracking only
    float sigma_rad = 0.785398f;              // gaussian (45 deg default)
    float error_cap_rad = 0.5f;               // linear
    float huber_delta = 0.1f;                 // huber
    float vel_sigma_rad = 0.523599f;          // vel_tracking gaussian (30 deg/s)
    float tol_rad = 0.0174533f;               // settling tolerance (1 deg)
    float window_s = 2.0f;                     // settling window (reserved)
    bool is_penalty = false;                   // subtracted instead of added
};

struct RewardHarness {
    std::string name = "default";
    std::vector<RewardTerm> terms;
    float reward_scale = 1.0f;
    int spec_version = 0;
    std::string source_path;
};

// ==================== H = (P, A, M) harness bundle (Phase 3) ====================
// HarnessForge models a harness as three coupled sub-bundles so that fault
// attribution can target the right dimension:
//   P (Planning)  : training strategy + curriculum            (D5 train, D6 curriculum)
//   A (Action)    : reward shaping + deployment policy        (D3 deploy, D4 reward)
//   M (Memory)    : replay / expert data / trace retention    (D1 trace, D2 memory)
// Defaults reproduce Phase 1/2 behavior exactly: every field is read-only here
// and only consumed by future processors when a variant explicitly overrides it.
struct PlanningBundle {
    int train_iterations = -1;        // -1 = inherit node default
    bool curriculum_enabled = false;  // sigma curriculum
    float sigma_shrink_ratio = -1.0f; // -1 = inherit
};
struct ActionBundle {
    // reward lives in HarnessConfig::reward (the closed-vocabulary RewardHarness).
    bool deploy_merge_lora = false;   // Phase 4 hook; OFF here
};
struct MemoryBundle {
    bool use_expert_data = true;      // inherit node default semantics
    int replay_trim_settling = -1;    // -1 = inherit
};

struct HarnessConfig {
    std::string name;
    RewardHarness reward;             // A: reward shaping (closed vocabulary)
    // H = (P, A, M) bundle (Phase 3). Defaults => byte-identical to Phase 1/2.
    PlanningBundle planning;          // P
    ActionBundle   action;            // A (deploy side; reward is `reward` above)
    MemoryBundle   memory;            // M
    std::string parent;               // lineage for fork-on-conflict (Variant Isolation)
    // --- performance bookkeeping (Phase 1 + Phase 3 Pareto axes) ---
    int episodes_run = 0;
    float ema_mae_deg = -1.0f;        // axis 1: tracking accuracy (minimize)
    float best_mae_deg = 1e9f;
    float last_mae_deg = -1.0f;
    float last_delta_deg = 0.0f;
    float ema_energy = -1.0f;         // axis 2: actuation energy ~ mean hamming (minimize)
    float last_energy = -1.0f;
    float ema_tau_us = -1.0f;         // mean on-MCU inference time (us)
    float tau_robustness = -1.0f;     // axis 3: 1/(1+stdev(tau)) proxy (maximize)
    bool on_pareto = false;           // last-computed Pareto membership
};

// ==================== Hook Point + Processor architecture (Phase 2) ====================
// HarnessX decomposes a harness into Processors that subscribe to Hook Points in
// the learning loop. The paper's async process() is adapted to *synchronous* C++:
// each Processor exposes process(Hook, HookContext&) -> void and runs in-order on
// a single dispatcher. Only the train-affinity hooks (BeforeTrain/AfterTrain) run
// on the background training thread; all others run on the main spin thread.
//
// substitution algebra: each Hook owns an ordered registry of Processors. Swapping
// a Processor at a Hook never touches the others (Variant Isolation foundation).
//
// INVARIANT: with no Processors registered a Hook is a no-op. The default build
// registers exactly one RewardProcessor at AfterAction, reproducing Phase 1 reward
// behavior byte-identically.

enum class Hook {
    TaskStart,      // episode about to start (main thread)
    BeforeModel,    // before policy inference (reserved; inference is on-MCU)
    AfterModel,     // after policy inference (reserved)
    BeforeAction,   // before applying action (reserved)
    AfterAction,    // after action observed -> compute reward (main thread)
    BeforeTrain,    // before a training batch loop (TRAIN affinity: bg thread)
    AfterTrain,     // after the training batch loop (TRAIN affinity: bg thread)
    EpisodeEnd      // episode finished, stats ready (main thread)
};

inline const char* hook_name(Hook h) {
    switch (h) {
        case Hook::TaskStart:    return "TaskStart";
        case Hook::BeforeModel:  return "BeforeModel";
        case Hook::AfterModel:   return "AfterModel";
        case Hook::BeforeAction: return "BeforeAction";
        case Hook::AfterAction:  return "AfterAction";
        case Hook::BeforeTrain:  return "BeforeTrain";
        case Hook::AfterTrain:   return "AfterTrain";
        case Hook::EpisodeEnd:   return "EpisodeEnd";
    }
    return "?";
}

// Mutable context passed by reference through a hook chain. Processors read the
// fields they need and write outputs in place. Fields are a superset across all
// hooks; only the relevant ones are populated for a given hook.
struct HookContext {
    Hook hook;
    int episode = 0;
    int step = 0;

    // --- AfterAction inputs (reward computation) ---
    float error_rad = 0.0f;
    float vel_error_rad = 0.0f;
    int prev_action = 0;
    int curr_action = 0;
    const RewardHarness* reward_harness = nullptr;  // active harness snapshot
    // --- AfterAction outputs ---
    float tracking = 0.0f;
    float penalty = 0.0f;
    float reward = 0.0f;

    // --- EpisodeEnd inputs ---
    float mae_deg = 0.0f;
    int episode_type = 0;  // 0=RL, 1=rule-based

    // --- Train hooks ---
    int train_iteration = 0;
};

// Base Processor: one of HarnessX's 9 dimensions (D1..D9) lives behind this.
// scope() returns the dimension tag for tracing/introspection.
struct Processor {
    virtual ~Processor() = default;
    virtual const char* name() const = 0;
    virtual const char* scope() const { return "D?"; }  // e.g. "D4-reward"
    virtual void process(HookContext& ctx) = 0;
};

// D4 reward (Phase 1 logic, now a Processor at the AfterAction hook).
// Delegates the actual term math to the node's compute_reward via a std::function
// so the Processor stays decoupled from the node class definition.
struct RewardProcessor : public Processor {
    using RewardFn = std::function<float(float, float, int, int, float&, float&)>;
    explicit RewardProcessor(RewardFn fn) : fn_(std::move(fn)) {}
    const char* name() const override { return "RewardProcessor"; }
    const char* scope() const override { return "D4-reward"; }
    void process(HookContext& ctx) override {
        float tracking = 0.0f, penalty = 0.0f;
        ctx.reward = fn_(ctx.error_rad, ctx.vel_error_rad,
                         ctx.prev_action, ctx.curr_action, tracking, penalty);
        ctx.tracking = tracking;
        ctx.penalty = penalty;
    }
private:
    RewardFn fn_;
};

// Ordered registry of Processors per Hook (substitution algebra container).
class HookRegistry {
public:
    void add(Hook h, std::shared_ptr<Processor> p) {
        registry_[static_cast<int>(h)].push_back(std::move(p));
    }
    // Replace all Processors at a hook (substitution); returns count removed.
    size_t clear_hook(Hook h) {
        auto& v = registry_[static_cast<int>(h)];
        size_t n = v.size();
        v.clear();
        return n;
    }
    const std::vector<std::shared_ptr<Processor>>& at(Hook h) const {
        return registry_[static_cast<int>(h)];
    }
    bool empty(Hook h) const { return registry_[static_cast<int>(h)].empty(); }
    size_t size(Hook h) const { return registry_[static_cast<int>(h)].size(); }

private:
    static constexpr int NUM_HOOKS = 8;
    std::array<std::vector<std::shared_ptr<Processor>>, NUM_HOOKS> registry_;
};

class HiLHNode : public rclcpp::Node {
public:
    HiLHNode() : Node("hilh_node") {
        // ===== Static Parameters (set at launch) =====
        this->declare_parameter("max_episodes", 1000);
        this->declare_parameter("batch_size", 256);
        this->declare_parameter("save_dir", "./sac_checkpoints");
        this->declare_parameter("csv_dir", "./episode_logs");
        this->declare_parameter("buffer_dir", "./replay_buffers");
        this->declare_parameter("load_checkpoint", "");
        this->declare_parameter("load_buffer", true);
        this->declare_parameter("buffer_save_interval", 50);

        // ===== Dynamic Parameters (changeable DURING training via ros2 param set) =====
        this->declare_parameter("train_iterations", 3000);
        this->declare_parameter("reward_type", 0);          // 0=gaussian, 1=linear, 2=huber
        this->declare_parameter("sigma_deg", 45.0);
        this->declare_parameter("w_change", 0.0);
        this->declare_parameter("reward_scale", 1.0);
        this->declare_parameter("error_cap_rad", 0.5);
        this->declare_parameter("huber_delta", 0.1);
        this->declare_parameter("learning_rate", 3e-4);
        this->declare_parameter("target_entropy", 1.39);
        this->declare_parameter("use_expert_data", true);
        this->declare_parameter("rl_only_mode", false);
        this->declare_parameter("vel_reward_weight", 0.0);
        this->declare_parameter("vel_sigma_deg", 30.0);

        // Auto sigma curriculum
        this->declare_parameter("sigma_curriculum_enabled", false);
        this->declare_parameter("sigma_window", 10);
        this->declare_parameter("sigma_shrink_ratio", 0.7);
        this->declare_parameter("sigma_shrink_factor", 0.80);
        this->declare_parameter("sigma_min", 0.5);

        // ===== PHASE 1: Reward Harness control surface =====
        this->declare_parameter("harness_dir", "./harnesses");
        this->declare_parameter("reward_spec_path", "");       // path to a .rwd file to load
        this->declare_parameter("reload_harness", 0);          // bump counter to re-read reward_spec_path
        this->declare_parameter("active_harness", "default");  // which registered variant is live
        this->declare_parameter("harness_autowatch", false);   // (reserved) auto re-read spec mtime

        // PHASE 2: hook introspection (default OFF -> behavior identical to Phase 1)
        this->declare_parameter("hooks_trace", false);         // verbose hook dispatch logging

        // PHASE 3: Variant Isolation + Pareto archive (all default OFF)
        this->declare_parameter("harness_isolation", false);   // fork-on-conflict vs overwrite
        this->declare_parameter("archive_enabled", false);     // persist Pareto archive to disk
        this->declare_parameter("archive_dir", "./harness_archive");
        // Status (read-back) params, set by the node:
        this->declare_parameter("current_harness", "default");
        this->declare_parameter("current_spec_version", 0);
        this->declare_parameter("last_mae_deg", -1.0);
        this->declare_parameter("best_mae_deg", -1.0);

        max_episodes_ = this->get_parameter("max_episodes").as_int();
        batch_size_ = this->get_parameter("batch_size").as_int();
        save_dir_ = this->get_parameter("save_dir").as_string();
        csv_dir_ = this->get_parameter("csv_dir").as_string();
        buffer_dir_ = this->get_parameter("buffer_dir").as_string();
        load_checkpoint_ = this->get_parameter("load_checkpoint").as_string();
        load_buffer_ = this->get_parameter("load_buffer").as_bool();
        buffer_save_interval_ = this->get_parameter("buffer_save_interval").as_int();

        rl_only_mode_ = this->get_parameter("rl_only_mode").as_bool();
        harness_dir_ = this->get_parameter("harness_dir").as_string();
        hooks_trace_ = this->get_parameter("hooks_trace").as_bool();
        harness_isolation_ = this->get_parameter("harness_isolation").as_bool();
        archive_enabled_ = this->get_parameter("archive_enabled").as_bool();
        archive_dir_ = this->get_parameter("archive_dir").as_string();

        reload_dynamic_params();

        std::filesystem::create_directories(save_dir_);
        std::filesystem::create_directories(csv_dir_);
        std::filesystem::create_directories(buffer_dir_);
        std::filesystem::create_directories(harness_dir_);

        // SAC Agent & Replay Buffer
        agent_ = std::make_unique<sac::SACAgent>();
        replay_buffer_ = std::make_unique<sac::ReplayBuffer>(1000000);

        if (!load_checkpoint_.empty()) {
            RCLCPP_INFO(this->get_logger(), "[Load] checkpoint: %s", load_checkpoint_.c_str());
            agent_->load(load_checkpoint_);
            agent_->set_target_entropy(target_entropy_);
            use_pretrained_policy_ = true;
        }

        if (load_buffer_) {
            load_all_buffers();
        }

        auto qos = rclcpp::QoS(10).best_effort();

        obs_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "rl_observation", qos,
            std::bind(&HiLHNode::obs_callback, this, std::placeholders::_1));

        encoder_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
            "encoder_feedback", qos,
            std::bind(&HiLHNode::encoder_callback, this, std::placeholders::_1));

        episode_cmd_teensy_sub_ = this->create_subscription<std_msgs::msg::Int32>(
            "episode_cmd_teensy", qos,
            std::bind(&HiLHNode::episode_cmd_callback, this, std::placeholders::_1));

        policy_ack_sub_ = this->create_subscription<std_msgs::msg::Float32>(
            "policy_ack", qos,
            std::bind(&HiLHNode::policy_ack_callback, this, std::placeholders::_1));

        episode_cmd_pc_pub_ = this->create_publisher<std_msgs::msg::Int32>(
            "episode_cmd_pc", qos);

        policy_chunk_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
            "policy_chunk", qos);

        // PHASE 1: LLM-facing harness telemetry topics
        harness_status_pub_ = this->create_publisher<std_msgs::msg::String>(
            "harness_status", rclcpp::QoS(10));
        harness_event_pub_ = this->create_publisher<std_msgs::msg::String>(
            "harness_event", rclcpp::QoS(10));

        // PHASE 2: hook registry introspection topic (LLM-facing).
        // Latched (transient_local) so any late subscriber gets the current wiring.
        hooks_status_pub_ = this->create_publisher<std_msgs::msg::String>(
            "hooks_status", rclcpp::QoS(1).transient_local());

        // PHASE 3: Pareto archive introspection topic (LLM-facing, latched).
        harness_archive_pub_ = this->create_publisher<std_msgs::msg::String>(
            "harness_archive", rclcpp::QoS(1).transient_local());

        // Dynamic parameter callback
        param_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&HiLHNode::on_param_change, this, std::placeholders::_1));

        train_poll_timer_ = this->create_wall_timer(500ms,
            std::bind(&HiLHNode::check_training_done, this));

        startup_timer_ = this->create_wall_timer(
            3s, std::bind(&HiLHNode::start_first_episode, this));

        // PHASE 1: build + register the default harness (regression-faithful).
        register_default_harness();
        // If a reward_spec_path was provided at launch, load it now.
        {
            std::string spec = this->get_parameter("reward_spec_path").as_string();
            if (!spec.empty()) {
                request_harness_load(spec);
            }
        }

        // PHASE 2: install the default Processor set (RewardProcessor @ AfterAction).
        install_default_processors();
        publish_hooks_status();

        RCLCPP_INFO(this->get_logger(), "╔══ HiLH (Harness-in-Learning) ════════════════════════════════╗");
        RCLCPP_INFO(this->get_logger(), "║ Mode      %-10s Sigma    %6.2f°   Curriculum  %-3s      ║",
            rl_only_mode_ ? "RL-Only" : "Dual", sigma_deg_, sigma_curriculum_enabled_ ? "ON" : "OFF");
        RCLCPP_INFO(this->get_logger(), "║ Harness   active=%-12s variants=%-2zu  dir=%-10s ║",
            active_variant_.c_str(), variants_.size(), harness_dir_.c_str());
        RCLCPP_INFO(this->get_logger(), "║ Train     iters=%-5d batch=%-4d buf=%-6zu save=%-3d        ║",
            train_iterations_, batch_size_, replay_buffer_->size(), buffer_save_interval_);
        RCLCPP_INFO(this->get_logger(), "╚══════════════════════════════════════════════════════════════╝");
    }

    ~HiLHNode() {
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
        use_expert_data_ = this->get_parameter("use_expert_data").as_bool();
        vel_reward_weight_ = this->get_parameter("vel_reward_weight").as_double();
        vel_sigma_deg_ = this->get_parameter("vel_sigma_deg").as_double();
        vel_sigma_rad_ = vel_sigma_deg_ * M_PI / 180.0;

        sigma_curriculum_enabled_ = this->get_parameter("sigma_curriculum_enabled").as_bool();
        sigma_window_ = this->get_parameter("sigma_window").as_int();
        sigma_shrink_ratio_ = this->get_parameter("sigma_shrink_ratio").as_double();
        sigma_shrink_factor_ = this->get_parameter("sigma_shrink_factor").as_double();
        sigma_min_ = this->get_parameter("sigma_min").as_double();
    }

    // ==================== PHASE 1: Harness construction ====================

    // Build a RewardHarness from the current scalar ROS params that exactly
    // reproduces the original compute_reward (regression guarantee).
    RewardHarness build_default_reward() {
        RewardHarness h;
        h.name = "default";
        h.reward_scale = static_cast<float>(reward_scale_);
        h.spec_version = 0;
        h.source_path = "(params)";

        // pos_tracking term: kernel chosen by reward_type_ (0/1/2)
        RewardTerm pos;
        pos.type = TermType::PosTracking;
        // original: if vel_reward_weight_>1e-6, w_pos = 1 - vel_weight, else 1.0
        bool vel_on = vel_reward_weight_ > 1e-6;
        pos.weight = vel_on ? (1.0f - static_cast<float>(vel_reward_weight_)) : 1.0f;
        pos.kernel = (reward_type_ == 1) ? PosKernel::Linear
                   : (reward_type_ == 2) ? PosKernel::Huber
                   : PosKernel::Gaussian;
        pos.sigma_rad = static_cast<float>(sigma_rad_);
        pos.error_cap_rad = static_cast<float>(error_cap_rad_);
        pos.huber_delta = static_cast<float>(huber_delta_);
        h.terms.push_back(pos);

        // vel_tracking term (only when enabled, matching original weighting)
        if (vel_on) {
            RewardTerm vel;
            vel.type = TermType::VelTracking;
            vel.weight = static_cast<float>(vel_reward_weight_);
            vel.vel_sigma_rad = static_cast<float>(vel_sigma_rad_);
            h.terms.push_back(vel);
        }

        // hamming penalty term: penalty = w_change * hamming
        RewardTerm ham;
        ham.type = TermType::Hamming;
        ham.weight = static_cast<float>(w_change_);
        ham.is_penalty = true;
        h.terms.push_back(ham);

        return h;
    }

    // Returns the variant name actually registered (may differ from reward.name
    // when isolation forks a conflicting variant to "<name>#v2").
    std::string register_variant(const RewardHarness& reward) {
        auto it = variants_.find(reward.name);
        if (it == variants_.end()) {
            HarnessConfig cfg;
            cfg.name = reward.name;
            cfg.reward = reward;
            variants_[reward.name] = cfg;
            RCLCPP_INFO(this->get_logger(), "[Harness] registered variant '%s' (v%d, %zu terms)",
                reward.name.c_str(), reward.spec_version, reward.terms.size());
            return reward.name;
        }

        // PHASE 3 Variant Isolation: if this variant already has measured history
        // and isolation is ON, fork instead of overwriting so we never corrupt an
        // existing variant's leaderboard. Default OFF => Phase 1 overwrite behavior.
        if (harness_isolation_ && it->second.episodes_run > 0) {
            std::string fork = next_fork_name(reward.name);
            HarnessConfig cfg = it->second;       // inherit P/A/M defaults + lineage
            cfg.name = fork;
            cfg.parent = reward.name;
            cfg.reward = reward;
            cfg.reward.name = fork;
            // fresh performance ledger for the fork
            cfg.episodes_run = 0;
            cfg.ema_mae_deg = -1.0f; cfg.best_mae_deg = 1e9f;
            cfg.last_mae_deg = -1.0f; cfg.last_delta_deg = 0.0f;
            cfg.ema_energy = -1.0f; cfg.last_energy = -1.0f;
            cfg.ema_tau_us = -1.0f; cfg.tau_robustness = -1.0f;
            cfg.on_pareto = false;
            variants_[fork] = cfg;
            RCLCPP_WARN(this->get_logger(),
                "[Harness] ISOLATION fork '%s' -> '%s' (parent kept, v%d, %zu terms)",
                reward.name.c_str(), fork.c_str(), reward.spec_version, reward.terms.size());
            publish_harness_event("FORKED", fork + " parent=" + reward.name);
            return fork;
        }

        it->second.reward = reward;  // update spec, keep accumulated stats
        RCLCPP_INFO(this->get_logger(), "[Harness] updated variant '%s' -> v%d (%zu terms)",
            reward.name.c_str(), reward.spec_version, reward.terms.size());
        return reward.name;
    }

    // Generate the next free "<base>#vN" name (N starts at 2).
    std::string next_fork_name(const std::string& base) {
        for (int v = 2; v < 10000; ++v) {
            std::string cand = base + "#v" + std::to_string(v);
            if (variants_.find(cand) == variants_.end()) return cand;
        }
        return base + "#vX";
    }

    void register_default_harness() {
        RewardHarness def = build_default_reward();
        register_variant(def);
        active_variant_ = "default";
        // publish live pointer for compute_reward
        std::atomic_store(&active_reward_,
            std::make_shared<const RewardHarness>(variants_["default"].reward));
        publish_status_params();
        publish_harness_event("LOADED", "default ok");
    }

    // ---- flat .rwd parser (self-contained, no JSON dependency) ----
    // Grammar (one key=value per line, '#' comments, blank lines ignored):
    //   name = <variant_name>
    //   reward_scale = <float>
    //   term.<i>.type = pos_tracking|vel_tracking|hamming|settling|
    //                   disturbance_rejection|energy
    //   term.<i>.weight = <float>
    //   term.<i>.kernel = gaussian|linear|huber
    //   term.<i>.sigma_deg = <float>          (-> rad)
    //   term.<i>.error_cap_rad = <float>
    //   term.<i>.huber_delta = <float>
    //   term.<i>.vel_sigma_deg = <float>      (-> rad)
    //   term.<i>.tol_deg = <float>            (-> rad)
    //   term.<i>.window_s = <float>
    //   term.<i>.is_penalty = true|false
    // Returns true on success; on any parse/enum error returns false and
    // 'err' is set (caller keeps previous harness, never crashes).
    bool parse_flat_spec(const std::string& path, RewardHarness& out, std::string& err) {
        std::ifstream f(path);
        if (!f.is_open()) { err = "cannot open " + path; return false; }

        RewardHarness h;
        h.name = "";
        h.reward_scale = 1.0f;
        std::map<int, RewardTerm> term_map;  // index -> term (sparse-safe, ordered)
        std::map<int, bool> term_seen;

        std::string line;
        int lineno = 0;
        while (std::getline(f, line)) {
            lineno++;
            // strip comments
            auto hash = line.find('#');
            if (hash != std::string::npos) line = line.substr(0, hash);
            // trim
            auto l = line.find_first_not_of(" \t\r\n");
            if (l == std::string::npos) continue;
            auto r = line.find_last_not_of(" \t\r\n");
            line = line.substr(l, r - l + 1);
            if (line.empty()) continue;

            auto eq = line.find('=');
            if (eq == std::string::npos) {
                RCLCPP_WARN(this->get_logger(), "[Spec] line %d: no '=' -> skip", lineno);
                continue;
            }
            std::string key = trim(line.substr(0, eq));
            std::string val = trim(line.substr(eq + 1));
            if (key.empty()) continue;

            if (key == "name") {
                h.name = val;
            } else if (key == "reward_scale") {
                if (!parse_float(val, h.reward_scale)) { err = "bad reward_scale"; return false; }
            } else if (key.rfind("term.", 0) == 0) {
                // term.<i>.<field>
                std::string rest = key.substr(5);
                auto dot = rest.find('.');
                if (dot == std::string::npos) {
                    RCLCPP_WARN(this->get_logger(), "[Spec] line %d: malformed term key -> skip", lineno);
                    continue;
                }
                int idx = 0;
                if (!parse_int(rest.substr(0, dot), idx)) {
                    RCLCPP_WARN(this->get_logger(), "[Spec] line %d: bad term index -> skip", lineno);
                    continue;
                }
                std::string field = rest.substr(dot + 1);
                RewardTerm& t = term_map[idx];
                term_seen[idx] = true;

                if (field == "type") {
                    if (!parse_term_type(val, t.type)) { err = "unknown term type '" + val + "'"; return false; }
                } else if (field == "weight") {
                    if (!parse_float(val, t.weight)) { err = "bad weight"; return false; }
                } else if (field == "kernel") {
                    if (!parse_kernel(val, t.kernel)) { err = "unknown kernel '" + val + "'"; return false; }
                } else if (field == "sigma_deg") {
                    float d; if (!parse_float(val, d)) { err = "bad sigma_deg"; return false; }
                    t.sigma_rad = d * static_cast<float>(M_PI) / 180.0f;
                } else if (field == "error_cap_rad") {
                    if (!parse_float(val, t.error_cap_rad)) { err = "bad error_cap_rad"; return false; }
                } else if (field == "huber_delta") {
                    if (!parse_float(val, t.huber_delta)) { err = "bad huber_delta"; return false; }
                } else if (field == "vel_sigma_deg") {
                    float d; if (!parse_float(val, d)) { err = "bad vel_sigma_deg"; return false; }
                    t.vel_sigma_rad = d * static_cast<float>(M_PI) / 180.0f;
                } else if (field == "tol_deg") {
                    float d; if (!parse_float(val, d)) { err = "bad tol_deg"; return false; }
                    t.tol_rad = d * static_cast<float>(M_PI) / 180.0f;
                } else if (field == "window_s") {
                    if (!parse_float(val, t.window_s)) { err = "bad window_s"; return false; }
                } else if (field == "is_penalty") {
                    t.is_penalty = (val == "true" || val == "1");
                } else {
                    RCLCPP_WARN(this->get_logger(), "[Spec] line %d: unknown field '%s' -> skip",
                        lineno, field.c_str());
                    publish_harness_event("UNKNOWN_TERM", "field=" + field);
                }
            } else {
                RCLCPP_WARN(this->get_logger(), "[Spec] line %d: unknown key '%s' -> skip",
                    lineno, key.c_str());
            }
        }

        // assemble terms in index order
        for (auto& kv : term_map) {
            h.terms.push_back(kv.second);
        }

        if (h.name.empty()) {
            // derive a name from the filename stem
            h.name = std::filesystem::path(path).stem().string();
        }
        if (h.terms.empty()) {
            err = "no terms defined";
            return false;
        }

        h.source_path = path;
        out = h;
        return true;
    }

    // Load spec from disk into a registered variant (does NOT swap live reward;
    // swap happens at the episode boundary in send_start_command()).
    void request_harness_load(const std::string& path) {
        RewardHarness h;
        std::string err;
        if (!parse_flat_spec(path, h, err)) {
            RCLCPP_ERROR(this->get_logger(), "[Harness] LOAD_FAILED %s: %s", path.c_str(), err.c_str());
            publish_harness_event("LOAD_FAILED", "reason=" + err);
            return;  // previous harness retained
        }
        // assign a fresh spec_version (monotonic per variant)
        auto it = variants_.find(h.name);
        int prev_ver = (it != variants_.end()) ? it->second.reward.spec_version : 0;
        h.spec_version = prev_ver + 1;
        // register_variant may fork to "<name>#vN" under Variant Isolation; swap to
        // whatever name was actually registered.
        std::string registered = register_variant(h);
        pending_active_ = registered; // becomes live next episode boundary
        harness_changed_.store(true);
        RCLCPP_INFO(this->get_logger(), "[Harness] LOADED '%s' v%d (%zu terms) -> pending swap",
            registered.c_str(), h.spec_version, h.terms.size());
        publish_harness_event("LOADED", registered + " v" + std::to_string(h.spec_version) + " ok");
    }

    // Performed at episode boundary only (one episode == one spec_version).
    void swap_active_harness() {
        if (!harness_changed_.exchange(false)) return;
        std::string target = pending_active_.empty() ? active_variant_ : pending_active_;
        auto it = variants_.find(target);
        if (it == variants_.end()) {
            RCLCPP_WARN(this->get_logger(), "[Harness] swap target '%s' not found -> keep '%s'",
                target.c_str(), active_variant_.c_str());
            publish_harness_event("LOAD_FAILED", "reason=unknown variant " + target);
            return;
        }
        active_variant_ = target;
        std::atomic_store(&active_reward_,
            std::make_shared<const RewardHarness>(it->second.reward));
        publish_status_params();
        RCLCPP_WARN(this->get_logger(), "[Harness] SWAPPED -> '%s' v%d",
            active_variant_.c_str(), it->second.reward.spec_version);
        publish_harness_event("SWAPPED", active_variant_ + " v" +
            std::to_string(it->second.reward.spec_version));
    }

    // ---- small parse helpers ----
    static std::string trim(const std::string& s) {
        auto l = s.find_first_not_of(" \t\r\n");
        if (l == std::string::npos) return "";
        auto r = s.find_last_not_of(" \t\r\n");
        return s.substr(l, r - l + 1);
    }
    static bool parse_float(const std::string& s, float& out) {
        try { size_t pos; out = std::stof(s, &pos); return pos == s.size(); }
        catch (...) { return false; }
    }
    static bool parse_int(const std::string& s, int& out) {
        try { size_t pos; out = std::stoi(s, &pos); return pos == s.size(); }
        catch (...) { return false; }
    }
    static bool parse_term_type(const std::string& s, TermType& out) {
        if (s == "pos_tracking") out = TermType::PosTracking;
        else if (s == "vel_tracking") out = TermType::VelTracking;
        else if (s == "hamming") out = TermType::Hamming;
        else if (s == "settling") out = TermType::Settling;
        else if (s == "disturbance_rejection") out = TermType::DisturbanceRejection;
        else if (s == "energy") out = TermType::Energy;
        else return false;
        return true;
    }
    static bool parse_kernel(const std::string& s, PosKernel& out) {
        if (s == "gaussian") out = PosKernel::Gaussian;
        else if (s == "linear") out = PosKernel::Linear;
        else if (s == "huber") out = PosKernel::Huber;
        else return false;
        return true;
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
            // ===== PHASE 1: harness control =====
            else if (name == "harness_dir") {
                harness_dir_ = p.as_string();
                std::filesystem::create_directories(harness_dir_);
                RCLCPP_WARN(this->get_logger(), "[Param] harness_dir=%s", harness_dir_.c_str());
            } else if (name == "reward_spec_path") {
                std::string path = p.as_string();
                if (!path.empty()) {
                    RCLCPP_WARN(this->get_logger(), "[Param] reward_spec_path=%s", path.c_str());
                    request_harness_load(path);
                }
            } else if (name == "reload_harness") {
                int counter = p.as_int();
                if (counter != reload_counter_) {
                    reload_counter_ = counter;
                    std::string path = this->get_parameter("reward_spec_path").as_string();
                    if (!path.empty()) {
                        RCLCPP_WARN(this->get_logger(), "[Param] reload_harness=%d -> reread %s",
                            counter, path.c_str());
                        request_harness_load(path);
                    }
                }
            } else if (name == "active_harness") {
                std::string target = p.as_string();
                if (variants_.find(target) != variants_.end()) {
                    pending_active_ = target;
                    harness_changed_.store(true);
                    RCLCPP_WARN(this->get_logger(), "[Param] active_harness=%s -> pending swap",
                        target.c_str());
                } else {
                    RCLCPP_WARN(this->get_logger(), "[Param] active_harness=%s NOT registered",
                        target.c_str());
                    publish_harness_event("LOAD_FAILED", "reason=unknown variant " + target);
                }
            } else if (name == "harness_autowatch") {
                harness_autowatch_ = p.as_bool();
                RCLCPP_WARN(this->get_logger(), "[Param] harness_autowatch=%s",
                    harness_autowatch_ ? "ON" : "OFF");
            } else if (name == "hooks_trace") {
                // PHASE 2: toggle verbose hook dispatch logging at runtime.
                hooks_trace_ = p.as_bool();
                RCLCPP_WARN(this->get_logger(), "[Param] hooks_trace=%s",
                    hooks_trace_ ? "ON" : "OFF");
                publish_hooks_status();
            } else if (name == "harness_isolation") {
                // PHASE 3: fork-on-conflict vs overwrite (affects future reloads only).
                harness_isolation_ = p.as_bool();
                RCLCPP_WARN(this->get_logger(), "[Param] harness_isolation=%s",
                    harness_isolation_ ? "ON (fork)" : "OFF (overwrite)");
            } else if (name == "archive_enabled") {
                // PHASE 3: enable disk persistence of the Pareto archive.
                archive_enabled_ = p.as_bool();
                RCLCPP_WARN(this->get_logger(), "[Param] archive_enabled=%s",
                    archive_enabled_ ? "ON" : "OFF");
                if (archive_enabled_) update_pareto_archive();  // flush current frontier
            } else if (name == "archive_dir") {
                archive_dir_ = p.as_string();
                RCLCPP_WARN(this->get_logger(), "[Param] archive_dir=%s", archive_dir_.c_str());
            }
        }

        return result;
    }

    void publish_status_params() {
        // read-back params for ROS_MCP get_parameter
        this->set_parameter(rclcpp::Parameter("current_harness", active_variant_));
        auto sp = std::atomic_load(&active_reward_);
        int ver = sp ? sp->spec_version : 0;
        this->set_parameter(rclcpp::Parameter("current_spec_version", ver));
        auto it = variants_.find(active_variant_);
        if (it != variants_.end()) {
            this->set_parameter(rclcpp::Parameter("last_mae_deg",
                static_cast<double>(it->second.last_mae_deg)));
            this->set_parameter(rclcpp::Parameter("best_mae_deg",
                static_cast<double>(it->second.best_mae_deg >= 1e8f ? -1.0f : it->second.best_mae_deg)));
        }
    }

    void publish_harness_event(const std::string& kind, const std::string& detail) {
        auto msg = std_msgs::msg::String();
        msg.data = kind + " " + detail;
        harness_event_pub_->publish(msg);
    }

    void publish_harness_status(int episode, float last_mae, float ema_mae,
                                float best_mae, float rule_mae, float delta) {
        auto sp = std::atomic_load(&active_reward_);
        int ver = sp ? sp->spec_version : 0;
        std::ostringstream ss;
        ss << "harness=" << active_variant_
           << ";spec_version=" << ver
           << ";episode=" << episode
           << ";last_mae_deg=" << last_mae
           << ";ema_mae_deg=" << ema_mae
           << ";best_mae_deg=" << best_mae
           << ";rule_mae_deg=" << rule_mae
           << ";delta_deg=" << delta;
        auto msg = std_msgs::msg::String();
        msg.data = ss.str();
        harness_status_pub_->publish(msg);
    }

    // PHASE 2: publish the current hook -> processor wiring for LLM introspection.
    // Format: trace=ON|OFF;<HookName>=p1/scope1,p2/scope2;...
    void publish_hooks_status() {
        static const Hook all_hooks[] = {
            Hook::TaskStart, Hook::BeforeModel, Hook::AfterModel, Hook::BeforeAction,
            Hook::AfterAction, Hook::BeforeTrain, Hook::AfterTrain, Hook::EpisodeEnd
        };
        std::ostringstream ss;
        ss << "trace=" << (hooks_trace_ ? "ON" : "OFF");
        for (Hook h : all_hooks) {
            ss << ";" << hook_name(h) << "=";
            const auto& procs = hooks_.at(h);
            for (size_t i = 0; i < procs.size(); ++i) {
                if (i) ss << ",";
                ss << procs[i]->name() << "/" << procs[i]->scope();
            }
        }
        auto msg = std_msgs::msg::String();
        msg.data = ss.str();
        hooks_status_pub_->publish(msg);
    }

    void write_leaderboard_csv() {
        std::string filename = csv_dir_ + "/harness_leaderboard.csv";
        std::ofstream file(filename);
        if (!file.is_open()) return;
        file << "variant,episodes_run,last_mae_deg,ema_mae_deg,best_mae_deg,last_delta_deg,spec_version\n";
        file << std::fixed << std::setprecision(4);
        for (const auto& kv : variants_) {
            const auto& c = kv.second;
            file << c.name << ","
                 << c.episodes_run << ","
                 << c.last_mae_deg << ","
                 << c.ema_mae_deg << ","
                 << (c.best_mae_deg >= 1e8f ? -1.0f : c.best_mae_deg) << ","
                 << c.last_delta_deg << ","
                 << c.reward.spec_version << "\n";
        }
        file.close();
    }

    void maybe_update_sigma(float mae_deg) {
        if (!sigma_curriculum_enabled_) return;

        recent_mae_history_.push_back(mae_deg);

        if (static_cast<int>(recent_mae_history_.size()) < sigma_window_) return;

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

    int popcount4(int a) {
        int count = 0;
        for (int i = 0; i < 4; i++) {
            if (a & (1 << i)) count++;
        }
        return count;
    }

    // ==================== PHASE 2: Hook dispatch ====================
    // Runs every Processor registered at a Hook, in registration order, passing
    // the same HookContext by reference (synchronous, in-order). With no
    // Processors registered the call is a cheap no-op. Train-affinity hooks
    // (BeforeTrain/AfterTrain) are invoked from the bg training thread; all other
    // hooks are invoked from the main spin thread (see call sites).
    void run_hook(Hook h, HookContext& ctx) {
        ctx.hook = h;
        const auto& procs = hooks_.at(h);
        if (procs.empty()) return;
        if (hooks_trace_) {
            RCLCPP_INFO(this->get_logger(), "[Hook] %s -> %zu processor(s)",
                hook_name(h), procs.size());
        }
        for (const auto& p : procs) {
            p->process(ctx);
            if (hooks_trace_) {
                RCLCPP_INFO(this->get_logger(), "[Hook]   %s/%s",
                    p->scope(), p->name());
            }
        }
    }

    // Wire the default Processor set. Default == exactly one RewardProcessor at
    // AfterAction, preserving Phase 1 behavior. Other hooks remain empty (no-op)
    // until later phases register Processors.
    void install_default_processors() {
        auto reward_fn = [this](float e, float ve, int pa, int ca,
                                float& trk, float& pen) -> float {
            return this->compute_reward(e, ve, pa, ca, trk, pen);
        };
        hooks_.add(Hook::AfterAction, std::make_shared<RewardProcessor>(reward_fn));
        RCLCPP_INFO(this->get_logger(),
            "[Hook] installed defaults: AfterAction=RewardProcessor (D4-reward)");
    }

    // ==================== PHASE 1: data-driven reward ====================
    // Weighted sum over the active harness's named terms. The "default" harness
    // reproduces the original switch-based reward exactly.
    float compute_reward(float error_rad, float vel_error_rad,
                         int prev_action, int curr_action,
                         float& tracking_out, float& penalty_out) {

        auto harness = std::atomic_load(&active_reward_);
        if (!harness) {
            // should never happen; fall back to neutral
            tracking_out = 0.0f;
            penalty_out = 0.0f;
            return 0.0f;
        }

        float abs_error = std::abs(error_rad);
        int ham = hamming_distance(static_cast<uint8_t>(prev_action),
                                   static_cast<uint8_t>(curr_action));

        float tracking_acc = 0.0f;
        float penalty_acc = 0.0f;

        for (const auto& t : harness->terms) {
            float val = 0.0f;
            switch (t.type) {
                case TermType::PosTracking: {
                    switch (t.kernel) {
                        case PosKernel::Gaussian: {
                            float n = error_rad / (t.sigma_rad + 1e-9f);
                            val = std::exp(-0.5f * n * n);
                            break;
                        }
                        case PosKernel::Linear: {
                            val = std::max(0.0f, 1.0f - abs_error / (t.error_cap_rad + 1e-9f));
                            break;
                        }
                        case PosKernel::Huber: {
                            float d = t.huber_delta;
                            if (abs_error <= d) {
                                val = 1.0f - 0.5f * (abs_error / d) * (abs_error / d);
                            } else {
                                val = std::max(0.0f, 0.5f * d / abs_error);
                            }
                            break;
                        }
                    }
                    break;
                }
                case TermType::VelTracking: {
                    float n = vel_error_rad / (t.vel_sigma_rad + 1e-9f);
                    val = std::exp(-0.5f * n * n);
                    break;
                }
                case TermType::Hamming: {
                    val = static_cast<float>(ham);
                    break;
                }
                case TermType::Settling: {
                    // bonus when within tolerance
                    val = (abs_error <= t.tol_rad) ? 1.0f : 0.0f;
                    break;
                }
                case TermType::DisturbanceRejection: {
                    // reward returning toward target: positive when error shrinking.
                    // error_dot is not passed here; approximate with -|error| normalized.
                    float n = error_rad / (t.sigma_rad + 1e-9f);
                    val = std::exp(-0.5f * n * n);
                    break;
                }
                case TermType::Energy: {
                    // penalize number of active DI bits (actuation cost)
                    val = static_cast<float>(popcount4(curr_action));
                    break;
                }
            }

            if (t.is_penalty) {
                penalty_acc += t.weight * val;
            } else {
                tracking_acc += t.weight * val;
            }
        }

        tracking_out = tracking_acc;
        penalty_out = penalty_acc;
        float reward = (tracking_out - penalty_out) * harness->reward_scale;
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

        // PHASE 1: apply any pending harness swap at the episode boundary so a
        // single episode is computed with exactly one spec_version.
        swap_active_harness();

        // PHASE 2: TaskStart hook (main thread). No-op unless processors registered.
        {
            HookContext ctx;
            ctx.episode = episode_count_ + 1;
            run_hook(Hook::TaskStart, ctx);
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

        if (rl_only_mode_) {
            current_episode_type_ = RL_EPISODE;
        } else {
            current_episode_type_ = (episode_count_ % 2 == 0) ? RULE_BASED_EPISODE : RL_EPISODE;
        }

        if (current_episode_type_ == RULE_BASED_EPISODE) {
            msg.data = 2;  // cmd=2: Rule-based half-step
            RCLCPP_INFO(this->get_logger(),
                "[Ep%3d] START  RULE  buf=%zu  harness=%s",
                episode_count_ + 1, replay_buffer_->size(), active_variant_.c_str());
        } else {
            msg.data = 0;  // cmd=0: RL policy
            RCLCPP_INFO(this->get_logger(),
                "[Ep%3d] START  RL    buf=%zu  harness=%s  alpha=%.4f  v%d",
                episode_count_ + 1, replay_buffer_->size(),
                active_variant_.c_str(), agent_->get_alpha(), policy_version_);
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

        // PHASE 2: AfterAction hook (main thread) runs the RewardProcessor chain.
        // Default registry = single RewardProcessor -> identical to Phase 1 reward.
        HookContext ctx;
        ctx.episode = episode_count_ + 1;
        ctx.step = step_count_;
        ctx.error_rad = error;
        ctx.vel_error_rad = vel_error;
        ctx.prev_action = last_action_discrete_;
        ctx.curr_action = action_discrete;
        {
            auto sp = std::atomic_load(&active_reward_);
            ctx.reward_harness = sp.get();
        }
        run_hook(Hook::AfterAction, ctx);
        float tracking = ctx.tracking;
        float penalty = ctx.penalty;
        float reward = ctx.reward;

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - episode_start_time_).count();

        float remote_checksum = 0.0f;
        float teensy_time = 0.0f;
        float control_period_us = 0.0f;
        float inference_time_us = 0.0f;
        float isr_execution_time_us = 0.0f;
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
            summary.harness_name = active_variant_;
            {
                auto sp = std::atomic_load(&active_reward_);
                summary.spec_version = sp ? sp->spec_version : 0;
            }
            episode_summaries_.push_back(summary);

            // PHASE 2: EpisodeEnd hook (main thread). Stats ready, fires once for
            // both RULE and RL episodes. No-op unless processors registered.
            {
                HookContext ctx;
                ctx.episode = episode_count_ + 1;
                ctx.mae_deg = mae_deg;
                ctx.episode_type = static_cast<int>(current_episode_type_);
                run_hook(Hook::EpisodeEnd, ctx);
            }

            if (current_episode_type_ == RULE_BASED_EPISODE) {
                rule_based_mae_deg_ = mae_deg;

                RCLCPP_INFO(this->get_logger(),
                    "[Ep%3d] DONE   RULE  MAE=%6.2f°  track=%.3f  hamming=%.2f",
                    episode_count_ + 1, mae_deg, avg_tracking, avg_hamming);

                if (use_expert_data_) {
                    for (const auto& t : episode_data_) {
                        replay_buffer_->push(t.state, t.action, t.reward, t.next_state, t.done);
                    }
                }

                episode_count_++;

                deploy_policy_and_start();

            } else {
                for (const auto& t : episode_data_) {
                    replay_buffer_->push(t.state, t.action, t.reward, t.next_state, t.done);
                }

                float rl_mae_deg = mae_deg;
                float delta = 0.0f;

                if (!rl_only_mode_ && rule_based_mae_deg_ > 0) {
                    delta = rule_based_mae_deg_ - rl_mae_deg;
                    bool rl_wins = rl_mae_deg < rule_based_mae_deg_;

                    RCLCPP_WARN(this->get_logger(),
                        "[Ep%3d] >>>    RULE=%5.2f° vs RL=%5.2f°  d=%+.2f°  %s",
                        episode_count_ + 1, rule_based_mae_deg_, rl_mae_deg, delta,
                        rl_wins ? "RL WINS" : "RULE WINS");
                }

                RCLCPP_INFO(this->get_logger(),
                    "[Ep%3d] DONE   RL    MAE=%6.2f°  track=%.3f  hamming=%.2f  alpha=%.4f  harness=%s",
                    episode_count_ + 1, rl_mae_deg, avg_tracking, avg_hamming,
                    agent_->get_alpha(), active_variant_.c_str());

                // PHASE 1: update the active variant's MAE leaderboard.
                // PHASE 3: also feed energy (mean hamming) + tau stats for Pareto axes.
                float tau_mean = 0.0f, tau_robust = -1.0f;
                episode_tau_stats(tau_mean, tau_robust);
                update_variant_perf(active_variant_, rl_mae_deg, delta,
                                    avg_hamming, tau_mean, tau_robust);
                publish_harness_status(episode_count_ + 1, rl_mae_deg,
                    variant_ema(active_variant_), variant_best(active_variant_),
                    rule_based_mae_deg_, delta);
                write_leaderboard_csv();
                publish_status_params();

                // PHASE 3: refresh the Pareto archive (default OFF -> no-op unless enabled).
                update_pareto_archive();

                maybe_update_sigma(rl_mae_deg);

                episode_count_++;

                if (episode_count_ % buffer_save_interval_ == 0) {
                    save_buffer(episode_count_);
                }

                start_async_training();
            }
        }
    }

    // PHASE 1+3: per-variant bookkeeping. MAE (last/ema/best/delta) plus the
    // Phase 3 Pareto axes: energy (mean hamming) and tau (mean + robustness).
    void update_variant_perf(const std::string& name, float mae_deg, float delta_deg,
                             float energy = -1.0f, float tau_mean = -1.0f,
                             float tau_robust = -1.0f) {
        auto it = variants_.find(name);
        if (it == variants_.end()) return;
        HarnessConfig& c = it->second;
        c.episodes_run++;
        c.last_mae_deg = mae_deg;
        c.last_delta_deg = delta_deg;
        if (c.ema_mae_deg < 0.0f) c.ema_mae_deg = mae_deg;
        else c.ema_mae_deg = 0.8f * c.ema_mae_deg + 0.2f * mae_deg;
        if (mae_deg < c.best_mae_deg) c.best_mae_deg = mae_deg;

        if (energy >= 0.0f) {
            c.last_energy = energy;
            if (c.ema_energy < 0.0f) c.ema_energy = energy;
            else c.ema_energy = 0.8f * c.ema_energy + 0.2f * energy;
        }
        if (tau_mean >= 0.0f) {
            if (c.ema_tau_us < 0.0f) c.ema_tau_us = tau_mean;
            else c.ema_tau_us = 0.8f * c.ema_tau_us + 0.2f * tau_mean;
        }
        if (tau_robust >= 0.0f) c.tau_robustness = tau_robust;
    }

    // PHASE 3: mean tau and a robustness proxy (1/(1+stdev)) over this episode's
    // on-MCU inference times. Returns (-1, -1) when tau telemetry is unavailable.
    void episode_tau_stats(float& tau_mean_out, float& tau_robust_out) {
        tau_mean_out = -1.0f;
        tau_robust_out = -1.0f;
        double sum = 0.0; size_t n = 0;
        for (const auto& r : telemetry_data_) {
            if (r.inference_time_us > 0.0f) { sum += r.inference_time_us; n++; }
        }
        if (n == 0) return;
        double mean = sum / static_cast<double>(n);
        double var = 0.0;
        for (const auto& r : telemetry_data_) {
            if (r.inference_time_us > 0.0f) {
                double d = r.inference_time_us - mean;
                var += d * d;
            }
        }
        var /= static_cast<double>(n);
        double sd = std::sqrt(var);
        tau_mean_out = static_cast<float>(mean);
        tau_robust_out = static_cast<float>(1.0 / (1.0 + sd));  // higher = steadier tau
    }

    // ==================== Pareto archive (Phase 3) ====================
    // Archive-Guided search keeps the non-dominated frontier over three axes:
    //   minimize ema_mae_deg, minimize ema_energy, maximize tau_robustness.
    // A variant A dominates B iff A is no worse on all axes and strictly better
    // on at least one. Only variants with measured episodes participate.
    // Default OFF (archive_enabled_) so nothing is written unless requested; the
    // in-memory frontier is always recomputed for /harness_archive introspection.

    // true if 'a' dominates 'b' (all-axes >= and at-least-one strictly better).
    static bool dominates(const HarnessConfig& a, const HarnessConfig& b) {
        // missing axis -> treat as non-discriminating (tie) so we never spuriously
        // dominate on an unmeasured dimension.
        auto le = [](float x, float y) { return x <= y + 1e-6f; }; // x no worse (min)
        auto lt = [](float x, float y) { return x <  y - 1e-6f; }; // x better (min)
        auto ge = [](float x, float y) { return x >= y - 1e-6f; }; // x no worse (max)
        auto gt = [](float x, float y) { return x >  y + 1e-6f; }; // x better (max)

        bool no_worse =
            le(a.ema_mae_deg, b.ema_mae_deg) &&
            (a.ema_energy < 0 || b.ema_energy < 0 || le(a.ema_energy, b.ema_energy)) &&
            (a.tau_robustness < 0 || b.tau_robustness < 0 || ge(a.tau_robustness, b.tau_robustness));
        bool strictly_better =
            lt(a.ema_mae_deg, b.ema_mae_deg) ||
            (a.ema_energy >= 0 && b.ema_energy >= 0 && lt(a.ema_energy, b.ema_energy)) ||
            (a.tau_robustness >= 0 && b.tau_robustness >= 0 && gt(a.tau_robustness, b.tau_robustness));
        return no_worse && strictly_better;
    }

    void update_pareto_archive() {
        // gather measured candidates
        std::vector<std::string> cands;
        for (auto& kv : variants_) {
            if (kv.second.episodes_run > 0 && kv.second.ema_mae_deg >= 0.0f)
                cands.push_back(kv.first);
        }
        // recompute frontier
        pareto_front_.clear();
        for (const auto& name : cands) {
            const HarnessConfig& c = variants_[name];
            bool dominated = false;
            for (const auto& other : cands) {
                if (other == name) continue;
                if (dominates(variants_[other], c)) { dominated = true; break; }
            }
            variants_[name].on_pareto = !dominated;
            if (!dominated) pareto_front_.push_back(name);
        }
        publish_harness_archive();
        if (archive_enabled_) write_archive_to_disk();
    }

    void publish_harness_archive() {
        std::ostringstream ss;
        ss << "pareto=" << pareto_front_.size() << "/" << variants_.size();
        for (const auto& name : pareto_front_) {
            const HarnessConfig& c = variants_[name];
            ss << ";" << name
               << "(mae=" << c.ema_mae_deg
               << ",energy=" << c.ema_energy
               << ",tau_robust=" << c.tau_robustness << ")";
        }
        auto msg = std_msgs::msg::String();
        msg.data = ss.str();
        if (harness_archive_pub_) harness_archive_pub_->publish(msg);
    }

    // Persist each on-frontier variant as a .harness file plus a scannable index.
    void write_archive_to_disk() {
        std::error_code ec;
        std::filesystem::create_directories(archive_dir_, ec);

        for (const auto& name : pareto_front_) {
            const HarnessConfig& c = variants_[name];
            std::string safe = name;
            std::replace(safe.begin(), safe.end(), '#', '_');
            std::replace(safe.begin(), safe.end(), '/', '_');
            std::ofstream f(archive_dir_ + "/" + safe + ".harness");
            if (!f.is_open()) continue;
            f << std::fixed << std::setprecision(6);
            f << "# Pareto-frontier harness bundle (Phase 3)\n";
            f << "name = " << c.name << "\n";
            if (!c.parent.empty()) f << "parent = " << c.parent << "\n";
            f << "reward_scale = " << c.reward.reward_scale << "\n";
            // A: reward terms (re-loadable .rwd grammar)
            for (size_t i = 0; i < c.reward.terms.size(); ++i) {
                const auto& t = c.reward.terms[i];
                f << "term." << i << ".type = " << term_type_name(t.type) << "\n";
                f << "term." << i << ".weight = " << t.weight << "\n";
                f << "term." << i << ".is_penalty = " << (t.is_penalty ? "true" : "false") << "\n";
            }
            // P / M bundle snapshot
            f << "P.train_iterations = " << c.planning.train_iterations << "\n";
            f << "P.curriculum_enabled = " << (c.planning.curriculum_enabled ? "true" : "false") << "\n";
            f << "M.use_expert_data = " << (c.memory.use_expert_data ? "true" : "false") << "\n";
            // measured metrics
            f << "metric.ema_mae_deg = " << c.ema_mae_deg << "\n";
            f << "metric.ema_energy = " << c.ema_energy << "\n";
            f << "metric.tau_robustness = " << c.tau_robustness << "\n";
            f << "metric.episodes_run = " << c.episodes_run << "\n";
        }

        // LLM-scannable index
        std::ofstream idx(archive_dir_ + "/archive_index.txt");
        if (idx.is_open()) {
            idx << "# variant,on_pareto,episodes,ema_mae_deg,ema_energy,tau_robustness,parent\n";
            idx << std::fixed << std::setprecision(4);
            for (auto& kv : variants_) {
                const HarnessConfig& c = kv.second;
                idx << c.name << ","
                    << (c.on_pareto ? 1 : 0) << ","
                    << c.episodes_run << ","
                    << c.ema_mae_deg << ","
                    << c.ema_energy << ","
                    << c.tau_robustness << ","
                    << (c.parent.empty() ? "-" : c.parent) << "\n";
            }
        }
    }

    static const char* term_type_name(TermType t) {
        switch (t) {
            case TermType::PosTracking:          return "pos_tracking";
            case TermType::VelTracking:          return "vel_tracking";
            case TermType::Hamming:              return "hamming";
            case TermType::Settling:             return "settling";
            case TermType::DisturbanceRejection: return "disturbance_rejection";
            case TermType::Energy:               return "energy";
        }
        return "pos_tracking";
    }

    float variant_ema(const std::string& name) {
        auto it = variants_.find(name);
        return (it != variants_.end()) ? it->second.ema_mae_deg : -1.0f;
    }
    float variant_best(const std::string& name) {
        auto it = variants_.find(name);
        if (it == variants_.end()) return -1.0f;
        return it->second.best_mae_deg >= 1e8f ? -1.0f : it->second.best_mae_deg;
    }

    void start_async_training() {
        if (train_thread_.joinable()) {
            train_thread_.join();
        }

        training_in_progress_.store(true);
        waiting_for_training_ = true;

        int iters = train_iterations_;
        int bs = batch_size_;
        int ep = episode_count_;

        train_thread_ = std::thread([this, iters, bs, ep]() {
            train_sac_thread(iters, bs, ep);
        });
    }

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

        auto t_start = std::chrono::steady_clock::now();
        std::cout << "[Train] start   iters=" << iterations
                  << "  buf=" << replay_buffer_->size() << std::endl;

        // PHASE 2: BeforeTrain hook (bg thread, TRAIN affinity). No-op unless
        // processors registered. Reads only frozen buffer rewards (no recompute).
        {
            HookContext ctx;
            ctx.episode = episode_num;
            run_hook(Hook::BeforeTrain, ctx);
        }

        std::vector<std::vector<float>> states, next_states;
        std::vector<int> actions;
        std::vector<float> rewards, dones;

        for (int i = 0; i < iterations; i++) {
            if (lr_changed_.exchange(false)) {
                agent_->set_learning_rate(learning_rate_);
            }
            if (te_changed_.exchange(false)) {
                agent_->set_target_entropy(target_entropy_);
            }

            if (replay_buffer_->sample(batch_size, states, actions, rewards, next_states, dones)) {
                agent_->update(states, actions, rewards, next_states, dones);
            }

            if ((i + 1) % 10000 == 0) {
                std::cout << "[Train] " << (i + 1) << "/" << iterations
                          << "  alpha=" << agent_->get_alpha() << std::endl;
            }
        }

        auto t_end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();

        std::cout << "[Train] done    " << std::fixed << std::setprecision(1) << elapsed
                  << "s  alpha=" << agent_->get_alpha()
                  << "  steps=" << agent_->get_training_steps() << std::endl;

        // PHASE 2: AfterTrain hook (bg thread, TRAIN affinity). No-op unless
        // processors registered.
        {
            HookContext ctx;
            ctx.episode = episode_num;
            ctx.train_iteration = iterations;
            run_hook(Hook::AfterTrain, ctx);
        }

        if (episode_num % 10 == 0) {
            agent_->save(save_dir_ + "/sac_ep" + std::to_string(episode_num));
        }

        training_in_progress_.store(false);
    }

    void check_training_done() {
        if (waiting_for_training_ && !training_in_progress_.load()) {
            waiting_for_training_ = false;

            if (train_thread_.joinable()) {
                train_thread_.join();
            }

            if (rl_only_mode_) {
                RCLCPP_INFO(this->get_logger(), "[Train] deploy -> next RL");
                deploy_policy_and_start();
            } else {
                RCLCPP_INFO(this->get_logger(), "[Train] done -> next RULE");
                send_start_command();
            }
        }

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
             << "control_period_us,inference_time_us,isr_execution_time_us\n";

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
                 << r.isr_execution_time_us << "\n";
        }

        file.close();
    }

    void save_summary_csv() {
        std::string filename = csv_dir_ + "/training_summary.csv";

        std::ofstream file(filename);
        if (!file.is_open()) return;

        file << "episode,steps,total_reward,avg_error_rad,mae_deg,"
             << "avg_tracking,avg_penalty,avg_hamming,"
             << "policy_version,alpha,sigma_deg,w_change,reward_type,episode_type,"
             << "harness_name,spec_version\n";

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
                 << s.episode_type << ","
                 << s.harness_name << ","
                 << s.spec_version << "\n";
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
        std::string harness_name;  // PHASE 1
        int spec_version = 0;      // PHASE 1
    };

    std::unique_ptr<sac::SACAgent> agent_;
    std::unique_ptr<sac::ReplayBuffer> replay_buffer_;

    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr obs_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr encoder_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr episode_cmd_teensy_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr policy_ack_sub_;

    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr episode_cmd_pc_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr policy_chunk_pub_;

    // PHASE 1: LLM-facing harness telemetry
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr harness_status_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr harness_event_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr hooks_status_pub_;     // PHASE 2
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr harness_archive_pub_;  // PHASE 3

    rclcpp::TimerBase::SharedPtr startup_timer_;
    rclcpp::TimerBase::SharedPtr next_episode_timer_;
    rclcpp::TimerBase::SharedPtr retry_timer_;
    rclcpp::TimerBase::SharedPtr train_poll_timer_;

    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;

    std::thread train_thread_;
    std::atomic<bool> training_in_progress_{false};
    bool waiting_for_training_ = false;
    std::atomic<bool> lr_changed_{false};
    std::atomic<bool> te_changed_{false};

    int max_episodes_;
    int batch_size_;
    std::string save_dir_;
    std::string csv_dir_;
    std::string buffer_dir_;
    std::string load_checkpoint_;
    bool use_pretrained_policy_ = false;
    bool load_buffer_ = true;
    int buffer_save_interval_ = 50;

    int train_iterations_ = 30000;
    int reward_type_ = 0;
    double sigma_deg_ = 45.0;
    double sigma_rad_ = 45.0 * M_PI / 180.0;
    double w_change_ = 0.02;
    double reward_scale_ = 1.0;
    double error_cap_rad_ = 0.5;
    double huber_delta_ = 0.1;
    double learning_rate_ = 3e-4;
    double target_entropy_ = 1.39;

    double vel_reward_weight_ = 0.0;
    double vel_sigma_deg_ = 30.0;
    double vel_sigma_rad_ = 30.0 * M_PI / 180.0;

    bool sigma_curriculum_enabled_ = false;
    int sigma_window_ = 10;
    double sigma_shrink_ratio_ = 0.7;
    double sigma_shrink_factor_ = 0.80;
    double sigma_min_ = 0.5;
    std::vector<float> recent_mae_history_;

    // ===== PHASE 1: Reward Harness state =====
    std::string harness_dir_ = "./harnesses";
    bool harness_autowatch_ = false;
    int reload_counter_ = 0;
    std::map<std::string, HarnessConfig> variants_;
    std::string active_variant_ = "default";
    std::string pending_active_;
    std::atomic<bool> harness_changed_{false};
    // live reward pointer (atomic load/store; immutable contents)
    std::shared_ptr<const RewardHarness> active_reward_;

    // ===== PHASE 2: Hook Point + Processor architecture =====
    HookRegistry hooks_;
    bool hooks_trace_ = false;  // verbose hook dispatch logging (ROS param)

    // ===== PHASE 3: Variant Isolation + Pareto archive =====
    bool harness_isolation_ = false;       // fork-on-conflict (default OFF = overwrite)
    bool archive_enabled_ = false;         // persist Pareto archive to disk (default OFF)
    std::string archive_dir_ = "./harness_archive";
    std::vector<std::string> pareto_front_; // variant names currently on the frontier

    enum EpisodeType { RL_EPISODE = 0, RULE_BASED_EPISODE = 1 };
    EpisodeType current_episode_type_ = RULE_BASED_EPISODE;
    float rule_based_mae_deg_ = -1.0f;
    bool use_expert_data_ = true;
    bool rl_only_mode_ = false;

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
    auto node = std::make_shared<HiLHNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

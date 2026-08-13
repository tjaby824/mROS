/**
 * Discrete SAC Agent Header (Christodoulou 2019)
 *
 * Categorical policy over 16 discrete actions (4-bit DI for stepper motor)
 *   Actor: actor_obs(55) → fc1→ReLU→fc2→ReLU → logits(16) → softmax
 *   Critic: full_obs(60) → fc1→ReLU→fc2→ReLU → Q-values(16)
 *
 * Actor obs (55): target(1), target_dot(1), preview(32), last_action(1),
 *                 action_hist(16), encoder_feats(4)
 * Critic obs (60): actor_obs(55) + time(1) + prev_DI_bits(4)
 *
 * Network: Actor 55→128→128→16, Critic 60→128→128→16
 * Deployed to Teensy: logits weights (55×128+128+128×128+128+128×16+16 = 25,744 floats)
 */

#ifndef SAC_AGENT_HPP
#define SAC_AGENT_HPP

#include <torch/torch.h>
#include <vector>
#include <memory>
#include <string>

namespace sac {

// Constants - Asymmetric Actor-Critic
constexpr int ACTOR_OBS_SIZE = 55;   // Actor observation (no critic extras)
constexpr int CRITIC_OBS_SIZE = 60;  // Critic observation (full obs including time + prev_DI)
constexpr int OBS_SIZE = 60;         // Full observation stored in replay buffer
constexpr int NN_HIDDEN = 128;       // Hidden layer size
constexpr int NN_OUTPUT = 16;        // Number of discrete actions
constexpr int NUM_ACTIONS = 16;      // Alias for readability

constexpr float EPSILON = 1e-6f;

// ==================== Actor (Categorical Policy) ====================
// Input: actor_obs only (55 dims, no critic extras)
// Output: logits over 16 discrete actions
struct ActorImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::Linear logits_layer{nullptr};  // 128 → 16

    ActorImpl();

    // Forward: actor_obs(55) → logits [batch, 16] (for Teensy: argmax)
    torch::Tensor forward(torch::Tensor x);

    // Forward with action probabilities (for SAC training)
    // Returns: (probs [batch,16], log_probs [batch,16])
    std::tuple<torch::Tensor, torch::Tensor> forward_with_probs(torch::Tensor x);
};
TORCH_MODULE(Actor);

// ==================== Critic (Q-Function for all actions) ====================
// Input: full_obs (60 dims) → Q-values for all 16 actions
struct CriticImpl : torch::nn::Module {
    torch::nn::Linear q1_fc1{nullptr}, q1_fc2{nullptr}, q1_fc3{nullptr};
    torch::nn::Linear q2_fc1{nullptr}, q2_fc2{nullptr}, q2_fc3{nullptr};
    // RLPD: LayerNorm after each hidden layer curbs Q-value overestimation/divergence
    torch::nn::LayerNorm q1_ln1{nullptr}, q1_ln2{nullptr};
    torch::nn::LayerNorm q2_ln1{nullptr}, q2_ln2{nullptr};

    CriticImpl();

    // Forward: full_obs(60) → (Q1 [batch,16], Q2 [batch,16])
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor state);
};
TORCH_MODULE(Critic);

// ==================== SAC Agent (Discrete) ====================
class SACAgent {
public:
    SACAgent(double lr = 3e-4, double gamma = 0.99, double tau = 0.005,
             double alpha = 0.2, bool auto_entropy = true);

    // Select discrete action (0-15)
    struct ActionResult {
        int discrete;  // 0-15
    };
    ActionResult select_action(const std::vector<float>& obs, bool deterministic = false);

    // Update networks from replay buffer batch (discrete actions)
    void update(const std::vector<std::vector<float>>& states,
                const std::vector<int>& actions,  // discrete 0-15
                const std::vector<float>& rewards,
                const std::vector<std::vector<float>>& next_states,
                const std::vector<float>& dones);

    // Get logits_layer weights for Teensy deployment
    std::vector<float> get_policy_weights();

    // Save/Load
    void save(const std::string& path);
    void load(const std::string& path);

    // Getters
    double get_alpha() const { return alpha_; }
    int get_training_steps() const { return training_steps_; }

    // Diagnostics from the most recent update() call (RLPD observability)
    struct UpdateStats {
        float critic_loss = 0.0f;
        float q_mean = 0.0f;     // mean of min(Q1,Q2) over batch & actions
        float q_max = 0.0f;      // max Q value seen (divergence indicator)
        float reward_mean = 0.0f;
        float entropy_mean = 0.0f;
        float target_q_mean = 0.0f;
    };
    UpdateStats get_last_stats() const { return last_stats_; }

    // Dynamic setters (for runtime tuning)
    void set_learning_rate(double lr);
    void set_target_entropy(double target_entropy);
    void reset_alpha(double new_alpha);

    // Freeze/unfreeze auto-entropy (alpha) tuning at runtime.
    // Disabling holds alpha fixed (update() skips the alpha step); re-enabling
    // resumes tuning from the current alpha (log_alpha_ left untouched).
    void set_auto_entropy(bool enable);
    bool get_auto_entropy() const { return auto_entropy_tuning_; }

private:
    Actor actor_;
    Critic critic_;
    Critic critic_target_;

    std::unique_ptr<torch::optim::Adam> actor_optimizer_;
    std::unique_ptr<torch::optim::Adam> critic_optimizer_;
    std::unique_ptr<torch::optim::Adam> alpha_optimizer_;

    double alpha_;
    double gamma_;
    double tau_;
    double target_entropy_;
    bool auto_entropy_tuning_;

    torch::Tensor log_alpha_;
    torch::Device device_;

    int training_steps_;
    UpdateStats last_stats_;

    void soft_update_target();
};

} // namespace sac

#endif // SAC_AGENT_HPP

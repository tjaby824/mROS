/**
 * Discrete SAC Agent Implementation (Christodoulou 2019)
 *
 * Categorical policy over 16 discrete actions:
 *   Actor: actor_obs(55) → fc1(ReLU) → fc2(ReLU) → logits(16) → softmax
 *   Critic: full_obs(60) → fc1(ReLU) → fc2(ReLU) → Q-values(16)
 *
 * Actor Network:
 *   fc1: 55 × 128 + 128 = 7,168
 *   fc2: 128 × 128 + 128 = 16,512
 *   logits_layer: 128 × 16 + 16 = 2,064
 *   Deployed: 25,744 floats (~100KB)
 *
 * Critic Network:
 *   Input: 60 → Q-values for all 16 actions
 */

#include "sac_trainer_cpp/sac_agent.hpp"
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>

namespace sac {

// ==================== Partial checkpoint loader ====================
// Loads only the tensors present in an archive into a module, matching by
// parameter/buffer name. Tensors absent from the archive (e.g. LayerNorm
// added after the checkpoint was saved) keep their current init values.
// Returns {matched, total} counts. Never throws on missing keys.
// Read a dotted key (e.g. "q1_fc1.weight") from a hierarchically-saved
// archive by descending into each submodule sub-archive, then reading the
// leaf tensor. Returns false if any level is absent (tensor left untouched).
static bool read_nested_tensor(torch::serialize::InputArchive& root,
                               const std::string& dotted,
                               torch::Tensor& out, bool is_buffer) {
    std::vector<std::string> parts;
    size_t start = 0, dot;
    while ((dot = dotted.find('.', start)) != std::string::npos) {
        parts.push_back(dotted.substr(start, dot - start));
        start = dot + 1;
    }
    parts.push_back(dotted.substr(start));
    if (parts.empty()) return false;

    torch::serialize::InputArchive cur = std::move(root);
    // descend through all but the last component (submodules)
    for (size_t i = 0; i + 1 < parts.size(); ++i) {
        torch::serialize::InputArchive next;
        if (!cur.try_read(parts[i], next)) return false;
        cur = std::move(next);
    }
    return cur.try_read(parts.back(), out, is_buffer);
}

static std::pair<int,int> load_matching_tensors(
        const std::shared_ptr<torch::nn::Module>& module,
        const std::string& file) {
    int matched = 0, total = 0;
    torch::NoGradGuard no_grad;

    auto try_copy = [&](const std::string& name, torch::Tensor dst, bool is_buffer) {
        total++;
        // Fresh archive per key: descending consumes the archive object.
        torch::serialize::InputArchive archive;
        archive.load_from(file);
        torch::Tensor tmp;
        try {
            if (read_nested_tensor(archive, name, tmp, is_buffer)
                    && tmp.sizes() == dst.sizes()) {
                dst.copy_(tmp);
                matched++;
            }
        } catch (const std::exception&) {
            // key absent or unreadable -> keep current init
        }
    };

    for (auto& p : module->named_parameters(/*recurse=*/true)) {
        try_copy(p.key(), p.value(), /*is_buffer=*/false);
    }
    for (auto& b : module->named_buffers(/*recurse=*/true)) {
        try_copy(b.key(), b.value(), /*is_buffer=*/true);
    }
    return {matched, total};
}

// ==================== Actor (Categorical Policy) ====================
// Actor uses ACTOR_OBS_SIZE (55) — no critic extras (time, prev_DI)
ActorImpl::ActorImpl() {
    fc1 = register_module("fc1", torch::nn::Linear(ACTOR_OBS_SIZE, NN_HIDDEN));
    fc2 = register_module("fc2", torch::nn::Linear(NN_HIDDEN, NN_HIDDEN));
    logits_layer = register_module("logits_layer", torch::nn::Linear(NN_HIDDEN, NUM_ACTIONS));

    // Orthogonal initialization
    torch::nn::init::orthogonal_(fc1->weight, std::sqrt(2.0));
    torch::nn::init::zeros_(fc1->bias);
    torch::nn::init::orthogonal_(fc2->weight, std::sqrt(2.0));
    torch::nn::init::zeros_(fc2->bias);
    torch::nn::init::orthogonal_(logits_layer->weight, 0.01);
    torch::nn::init::zeros_(logits_layer->bias);
}

torch::Tensor ActorImpl::forward(torch::Tensor x) {
    // Returns raw logits [batch, 16] — Teensy uses argmax on these
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    return logits_layer->forward(x);
}

std::tuple<torch::Tensor, torch::Tensor> ActorImpl::forward_with_probs(torch::Tensor x) {
    auto logits = forward(x);                           // [batch, 16]
    auto probs = torch::softmax(logits, -1);            // [batch, 16]
    auto log_probs = torch::log(probs + EPSILON);       // [batch, 16]
    return {probs, log_probs};
}

// ==================== Critic (Q-function for all actions) ====================
// Critic uses CRITIC_OBS_SIZE (60, full obs) → Q-values for all 16 actions
CriticImpl::CriticImpl() {
    // Q(s): full_obs → Q-value for each action
    // Input: CRITIC_OBS_SIZE = 60
    q1_fc1 = register_module("q1_fc1", torch::nn::Linear(CRITIC_OBS_SIZE, NN_HIDDEN));
    q1_fc2 = register_module("q1_fc2", torch::nn::Linear(NN_HIDDEN, NN_HIDDEN));
    q1_fc3 = register_module("q1_fc3", torch::nn::Linear(NN_HIDDEN, NUM_ACTIONS));

    q2_fc1 = register_module("q2_fc1", torch::nn::Linear(CRITIC_OBS_SIZE, NN_HIDDEN));
    q2_fc2 = register_module("q2_fc2", torch::nn::Linear(NN_HIDDEN, NN_HIDDEN));
    q2_fc3 = register_module("q2_fc3", torch::nn::Linear(NN_HIDDEN, NUM_ACTIONS));

    // RLPD: LayerNorm on hidden activations (bounds Q magnitude, prevents divergence)
    q1_ln1 = register_module("q1_ln1", torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({NN_HIDDEN})));
    q1_ln2 = register_module("q1_ln2", torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({NN_HIDDEN})));
    q2_ln1 = register_module("q2_ln1", torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({NN_HIDDEN})));
    q2_ln2 = register_module("q2_ln2", torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({NN_HIDDEN})));

    for (auto& m : {q1_fc1, q1_fc2, q2_fc1, q2_fc2}) {
        torch::nn::init::orthogonal_(m->weight, std::sqrt(2.0));
        torch::nn::init::zeros_(m->bias);
    }
    for (auto& m : {q1_fc3, q2_fc3}) {
        torch::nn::init::orthogonal_(m->weight, 1.0);
        torch::nn::init::zeros_(m->bias);
    }
}

std::tuple<torch::Tensor, torch::Tensor> CriticImpl::forward(torch::Tensor state) {
    // No action input — outputs Q-values for ALL 16 actions
    // RLPD: Linear → LayerNorm → ReLU on each hidden layer
    auto q1 = torch::relu(q1_ln1->forward(q1_fc1->forward(state)));
    q1 = torch::relu(q1_ln2->forward(q1_fc2->forward(q1)));
    q1 = q1_fc3->forward(q1);  // [batch, 16]

    auto q2 = torch::relu(q2_ln1->forward(q2_fc1->forward(state)));
    q2 = torch::relu(q2_ln2->forward(q2_fc2->forward(q2)));
    q2 = q2_fc3->forward(q2);  // [batch, 16]

    return {q1, q2};
}

// ==================== SAC Agent (Discrete) ====================
SACAgent::SACAgent(double lr, double gamma, double tau, double alpha, bool auto_entropy)
    : actor_(), critic_(), critic_target_(),
      alpha_(alpha), gamma_(gamma), tau_(tau),
      auto_entropy_tuning_(auto_entropy),
      device_(torch::kCPU), training_steps_(0) {

    // Copy critic to target
    std::stringstream buffer;
    torch::save(critic_, buffer);
    buffer.seekg(0);
    torch::load(critic_target_, buffer);

    // Optimizers
    actor_optimizer_ = std::make_unique<torch::optim::Adam>(
        actor_->parameters(), torch::optim::AdamOptions(lr));
    critic_optimizer_ = std::make_unique<torch::optim::Adam>(
        critic_->parameters(), torch::optim::AdamOptions(lr));

    // Entropy tuning - Discrete SAC: target = 0.5 * log(|A|) = 0.5 * log(16) ≈ 1.39
    target_entropy_ = 0.5 * std::log(static_cast<double>(NUM_ACTIONS));
    log_alpha_ = torch::tensor(std::log(alpha),
        torch::TensorOptions().requires_grad(true).device(device_));

    if (auto_entropy_tuning_) {
        alpha_optimizer_ = std::make_unique<torch::optim::Adam>(
            std::vector<torch::Tensor>{log_alpha_},
            torch::optim::AdamOptions(lr));
    }

    std::cout << "SACAgent initialized (Discrete SAC, Asymmetric Actor-Critic)" << std::endl;
    std::cout << "  Actor obs: " << ACTOR_OBS_SIZE << " (no critic extras)" << std::endl;
    std::cout << "  Critic obs: " << CRITIC_OBS_SIZE << " (full obs)" << std::endl;
    std::cout << "  NUM_ACTIONS: " << NUM_ACTIONS << " (discrete)" << std::endl;
    std::cout << "  Action: softmax(logits) → categorical sample / argmax" << std::endl;
    std::cout << "  Target entropy: " << target_entropy_ << std::endl;
}

SACAgent::ActionResult SACAgent::select_action(const std::vector<float>& obs, bool deterministic) {
    torch::NoGradGuard no_grad;

    // Actor only sees first ACTOR_OBS_SIZE (55) dims — no critic extras
    auto actor_obs_t = torch::from_blob(const_cast<float*>(obs.data()), {1, ACTOR_OBS_SIZE},
        torch::kFloat32).to(device_);

    ActionResult result;

    if (deterministic) {
        // Deterministic: argmax of logits
        auto logits = actor_->forward(actor_obs_t);  // [1, 16]
        result.discrete = logits.argmax(-1).item<int>();
    } else {
        // Stochastic: sample from categorical distribution
        auto [probs, log_probs] = actor_->forward_with_probs(actor_obs_t);
        auto action_t = torch::multinomial(probs, 1);  // [1, 1]
        result.discrete = action_t.item<int>();
    }

    return result;
}

void SACAgent::update(const std::vector<std::vector<float>>& states,
                      const std::vector<int>& actions,  // discrete 0-15
                      const std::vector<float>& rewards,
                      const std::vector<std::vector<float>>& next_states,
                      const std::vector<float>& dones) {

    int batch_size = states.size();

    // Convert discrete actions to int64 tensor for gather()
    std::vector<int64_t> actions_i64(batch_size);
    for (int i = 0; i < batch_size; i++) {
        actions_i64[i] = static_cast<int64_t>(actions[i]);
    }

    // Flatten full states (60 dim) for critic, and actor states (first 55 dim) for actor
    std::vector<float> states_full_flat, next_states_full_flat;
    std::vector<float> states_actor_flat, next_states_actor_flat;
    states_full_flat.reserve(batch_size * CRITIC_OBS_SIZE);
    next_states_full_flat.reserve(batch_size * CRITIC_OBS_SIZE);
    states_actor_flat.reserve(batch_size * ACTOR_OBS_SIZE);
    next_states_actor_flat.reserve(batch_size * ACTOR_OBS_SIZE);

    for (int i = 0; i < batch_size; i++) {
        // Full obs (60) for critic
        for (int j = 0; j < CRITIC_OBS_SIZE; j++) {
            states_full_flat.push_back(states[i][j]);
            next_states_full_flat.push_back(next_states[i][j]);
        }
        // Actor obs (first 55) — no critic extras
        for (int j = 0; j < ACTOR_OBS_SIZE; j++) {
            states_actor_flat.push_back(states[i][j]);
            next_states_actor_flat.push_back(next_states[i][j]);
        }
    }

    // Critic uses full obs (60)
    auto states_t = torch::from_blob(states_full_flat.data(), {batch_size, CRITIC_OBS_SIZE},
        torch::kFloat32).clone().to(device_);
    auto next_states_t = torch::from_blob(next_states_full_flat.data(), {batch_size, CRITIC_OBS_SIZE},
        torch::kFloat32).clone().to(device_);

    // Actor uses actor obs (55)
    auto actor_states_t = torch::from_blob(states_actor_flat.data(), {batch_size, ACTOR_OBS_SIZE},
        torch::kFloat32).clone().to(device_);
    auto actor_next_states_t = torch::from_blob(next_states_actor_flat.data(), {batch_size, ACTOR_OBS_SIZE},
        torch::kFloat32).clone().to(device_);

    // Actions as int64 tensor [batch, 1] for gather
    auto actions_t = torch::from_blob(actions_i64.data(), {batch_size, 1},
        torch::kInt64).clone().to(device_);

    auto rewards_t = torch::from_blob(const_cast<float*>(rewards.data()), {batch_size, 1},
        torch::kFloat32).clone().to(device_);
    auto dones_t = torch::from_blob(const_cast<float*>(dones.data()), {batch_size, 1},
        torch::kFloat32).clone().to(device_);

    float alpha = std::exp(log_alpha_.item<float>());

    // ===== Update Critic (Discrete SAC) =====
    torch::Tensor target_q;
    {
        torch::NoGradGuard no_grad;

        // Get action probs from current policy for next states
        auto [next_probs, next_log_probs] = actor_->forward_with_probs(actor_next_states_t);
        // next_probs: [batch, 16], next_log_probs: [batch, 16]

        // Target Q-values for all actions
        auto [next_q1, next_q2] = critic_target_->forward(next_states_t);
        auto next_q = torch::min(next_q1, next_q2);  // [batch, 16]

        // V(s') = sum_a π(a|s') * (Q(s',a) - α * log π(a|s'))
        auto next_v = (next_probs * (next_q - alpha * next_log_probs)).sum(-1, /*keepdim=*/true);
        // [batch, 1]

        target_q = rewards_t + (1.0 - dones_t) * gamma_ * next_v;
    }

    // Current Q-values for all actions
    auto [q1_all, q2_all] = critic_->forward(states_t);  // [batch, 16] each

    // Gather Q-values for the actions actually taken
    auto q1 = q1_all.gather(1, actions_t);  // [batch, 1]
    auto q2 = q2_all.gather(1, actions_t);  // [batch, 1]

    auto critic_loss = torch::mse_loss(q1, target_q) + torch::mse_loss(q2, target_q);

    critic_optimizer_->zero_grad();
    critic_loss.backward();
    torch::nn::utils::clip_grad_norm_(critic_->parameters(), 1.0);
    critic_optimizer_->step();

    // ===== Update Actor (Discrete SAC) =====
    auto [probs, log_probs] = actor_->forward_with_probs(actor_states_t);
    // probs: [batch, 16], log_probs: [batch, 16]

    // Q-values for all actions (detached from critic graph)
    auto [qa1, qa2] = critic_->forward(states_t);
    auto q_for_actor = torch::min(qa1, qa2).detach();  // [batch, 16]

    // Discrete actor loss: E_s[ sum_a π(a|s) * (α * log π(a|s) - Q(s,a)) ]
    auto actor_loss = (probs * (alpha * log_probs - q_for_actor)).sum(-1).mean();

    actor_optimizer_->zero_grad();
    actor_loss.backward();
    torch::nn::utils::clip_grad_norm_(actor_->parameters(), 1.0);
    actor_optimizer_->step();

    // ===== Update Alpha (Entropy Temperature) =====
    if (auto_entropy_tuning_) {
        // Entropy = -sum_a π(a|s) * log π(a|s)  (always positive for discrete)
        auto entropy = -(probs.detach() * log_probs.detach()).sum(-1);  // [batch]
        // When entropy < target: alpha increases (more exploration)
        // When entropy > target: alpha decreases (more exploitation)
        auto alpha_loss = (log_alpha_ * (entropy - target_entropy_)).mean();

        alpha_optimizer_->zero_grad();
        alpha_loss.backward();
        alpha_optimizer_->step();

        // Clamp log_alpha to prevent alpha collapse (min=0.01) or explosion (max=0.5)
        {
            torch::NoGradGuard no_grad;
            log_alpha_.clamp_(std::log(0.01), std::log(0.5));
        }
        alpha_ = std::exp(log_alpha_.item<float>());
    }

    // ===== Diagnostics (RLPD observability) =====
    {
        torch::NoGradGuard no_grad;
        auto q_min_all = torch::min(q1_all, q2_all);  // [batch, 16]
        last_stats_.critic_loss = critic_loss.item<float>();
        last_stats_.q_mean = q_min_all.mean().item<float>();
        last_stats_.q_max = q_min_all.max().item<float>();
        last_stats_.target_q_mean = target_q.mean().item<float>();
        last_stats_.reward_mean = rewards_t.mean().item<float>();
        auto entropy = -(probs.detach() * log_probs.detach()).sum(-1);  // [batch]
        last_stats_.entropy_mean = entropy.mean().item<float>();
    }

    // ===== Soft Update Target =====
    soft_update_target();
    training_steps_++;
}

void SACAgent::soft_update_target() {
    torch::NoGradGuard no_grad;

    auto params = critic_->parameters();
    auto target_params = critic_target_->parameters();

    for (size_t i = 0; i < params.size(); i++) {
        target_params[i].data().copy_(
            tau_ * params[i].data() + (1.0 - tau_) * target_params[i].data()
        );
    }
}

std::vector<float> SACAgent::get_policy_weights() {
    std::vector<float> weights;

    // Deploy logits_layer weights for Teensy (argmax inference)
    // fc1: [NN_HIDDEN, ACTOR_OBS_SIZE] → transpose → [ACTOR_OBS_SIZE, NN_HIDDEN]
    auto fc1_w = actor_->fc1->weight.data().cpu().t().contiguous();
    auto fc1_b = actor_->fc1->bias.data().cpu();

    auto fc2_w = actor_->fc2->weight.data().cpu().t().contiguous();
    auto fc2_b = actor_->fc2->bias.data().cpu();

    // logits_layer: [NUM_ACTIONS, NN_HIDDEN] → transpose → [NN_HIDDEN, NUM_ACTIONS]
    auto logits_w = actor_->logits_layer->weight.data().cpu().t().contiguous();
    auto logits_b = actor_->logits_layer->bias.data().cpu();

    auto append_tensor = [&weights](const torch::Tensor& t) {
        auto ptr = t.data_ptr<float>();
        weights.insert(weights.end(), ptr, ptr + t.numel());
    };

    append_tensor(fc1_w);     // 55 × 128 = 7,040
    append_tensor(fc1_b);     // 128
    append_tensor(fc2_w);     // 128 × 128 = 16,384
    append_tensor(fc2_b);     // 128
    append_tensor(logits_w);  // 128 × 16 = 2,048
    append_tensor(logits_b);  // 16
    // Total: 25,744

    return weights;
}

void SACAgent::save(const std::string& path) {
    std::ofstream meta(path + ".meta");
    if (meta.is_open()) {
        meta << training_steps_ << "\n";
        meta << alpha_ << "\n";
        meta << OBS_SIZE << "\n";
        meta << NN_OUTPUT << "\n";
        meta.close();
    }

    torch::save(actor_, path + "_actor.pt");
    torch::save(critic_, path + "_critic.pt");

    // Save optimizer states for stable checkpoint resume
    torch::save(*actor_optimizer_, path + "_actor_opt.pt");
    torch::save(*critic_optimizer_, path + "_critic_opt.pt");
    if (auto_entropy_tuning_) {
        torch::save(*alpha_optimizer_, path + "_alpha_opt.pt");
    }

    std::cout << "Saved: " << path << " (OBS=" << OBS_SIZE
              << ", ACTIONS=" << NUM_ACTIONS << ", step=" << training_steps_ << ")" << std::endl;
}

void SACAgent::load(const std::string& path) {
    std::ifstream meta(path + ".meta");
    if (meta.is_open()) {
        meta >> training_steps_;
        meta >> alpha_;

        int saved_obs = 0, saved_output = 0;
        meta >> saved_obs >> saved_output;

        if (saved_obs != OBS_SIZE || saved_output != NN_OUTPUT) {
            std::cerr << "WARNING: Dimension mismatch!" << std::endl;
            std::cerr << "  Saved: OBS=" << saved_obs << ", OUTPUT=" << saved_output << std::endl;
            std::cerr << "  Current: OBS=" << OBS_SIZE << ", OUTPUT=" << NN_OUTPUT << std::endl;
        }
        meta.close();
    }

    try {
        torch::load(actor_, path + "_actor.pt");

        // Critic may come from an older checkpoint whose structure differs
        // (e.g. saved before RLPD LayerNorm was added). A strict torch::load
        // throws on the first missing submodule ("No such serialized
        // submodule: q1_ln1"), losing ALL critic weights. Try strict load
        // first; on failure, fall back to name-matched partial load so the
        // Q-network Linear weights (q1_fc*, q2_fc*) still warm-start and only
        // the new LayerNorm params keep their fresh init.
        try {
            torch::load(critic_, path + "_critic.pt");
            std::cout << "[Load] critic: full load OK" << std::endl;
        } catch (const std::exception& ce) {
            auto [m, t] = load_matching_tensors(critic_.ptr(), path + "_critic.pt");
            std::cout << "[Load] critic: structure mismatch (" << ce.what()
                      << ") -> partial load " << m << "/" << t
                      << " tensors matched (missing keep fresh init)"
                      << std::endl;
        }

        // Copy to target
        std::stringstream buffer;
        torch::save(critic_, buffer);
        buffer.seekg(0);
        torch::load(critic_target_, buffer);

        // ★ Sync log_alpha_ with loaded alpha_ from .meta
        {
            torch::NoGradGuard no_grad;
            log_alpha_.fill_(std::log(alpha_));
        }

        // Restore optimizer states (backward-compatible: skip if files don't exist)
        bool optimizers_loaded = false;
        try {
            torch::load(*actor_optimizer_, path + "_actor_opt.pt");
            torch::load(*critic_optimizer_, path + "_critic_opt.pt");
            if (auto_entropy_tuning_) {
                torch::load(*alpha_optimizer_, path + "_alpha_opt.pt");
            }
            optimizers_loaded = true;
        } catch (const std::exception& oe) {
            std::cerr << "[Load] Optimizer states not found (will use fresh optimizers): "
                      << oe.what() << std::endl;
        }

        std::cout << "Loaded: " << path << " (step=" << training_steps_
                  << ", alpha=" << alpha_
                  << ", optimizers=" << (optimizers_loaded ? "restored" : "fresh")
                  << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load: " << path << " - " << e.what() << std::endl;
    }
}

void SACAgent::set_learning_rate(double lr) {
    for (auto& group : actor_optimizer_->param_groups()) {
        static_cast<torch::optim::AdamOptions&>(group.options()).lr(lr);
    }
    for (auto& group : critic_optimizer_->param_groups()) {
        static_cast<torch::optim::AdamOptions&>(group.options()).lr(lr);
    }
    if (alpha_optimizer_) {
        for (auto& group : alpha_optimizer_->param_groups()) {
            static_cast<torch::optim::AdamOptions&>(group.options()).lr(lr);
        }
    }
    std::cout << "Learning rate updated to " << lr << std::endl;
}

void SACAgent::set_target_entropy(double target_entropy) {
    target_entropy_ = target_entropy;
    std::cout << "Target entropy updated to " << target_entropy << std::endl;
}

void SACAgent::reset_alpha(double new_alpha) {
    {
        torch::NoGradGuard no_grad;
        log_alpha_.fill_(std::log(new_alpha));
    }
    alpha_ = new_alpha;
    std::cout << "[Alpha] reset to " << new_alpha << std::endl;
}

void SACAgent::set_auto_entropy(bool enable) {
    if (enable && !auto_entropy_tuning_) {
        // Lazy-create the alpha optimizer if the agent was built without one.
        // Resumes tuning from the current log_alpha_ (no discontinuity).
        if (!alpha_optimizer_) {
            alpha_optimizer_ = std::make_unique<torch::optim::Adam>(
                std::vector<torch::Tensor>{log_alpha_},
                torch::optim::AdamOptions(3e-4));
        }
    }
    auto_entropy_tuning_ = enable;
    std::cout << "[Alpha] auto-entropy tuning " << (enable ? "ENABLED (alpha unfrozen)"
                                                          : "DISABLED (alpha frozen)")
              << "  alpha=" << alpha_ << std::endl;
}

} // namespace sac

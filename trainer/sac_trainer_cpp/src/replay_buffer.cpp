/**
 * Replay Buffer Implementation (Discrete Actions, with Save/Load)
 *
 * File format (Binary, v3 = discrete int actions):
 *   [Header]
 *     - magic: 4 bytes ("RBUF")
 *     - version: uint32_t (3)
 *     - obs_size: uint32_t
 *     - capacity: uint64_t
 *     - size: uint64_t
 *     - pos: uint64_t
 *   [Data]
 *     - states: size × obs_size × float
 *     - actions: size × int32_t (discrete)
 *     - rewards: size × float
 *     - next_states: size × obs_size × float
 *     - dones: size × float
 */

#include "sac_trainer_cpp/replay_buffer.hpp"
#include <fstream>
#include <iostream>
#include <cstring>
#include <filesystem>

namespace sac {

constexpr uint32_t BUFFER_FILE_VERSION = 3;  // v3: discrete actions (int32)
constexpr char BUFFER_MAGIC[4] = {'R', 'B', 'U', 'F'};

ReplayBuffer::ReplayBuffer(size_t capacity)
    : capacity_(capacity), size_(0), pos_(0),
      rng_(std::random_device{}()) {

    states_.resize(capacity);
    actions_.resize(capacity, 0);
    rewards_.resize(capacity);
    next_states_.resize(capacity);
    dones_.resize(capacity);
}

void ReplayBuffer::push(const std::vector<float>& state,
                        int action,  // discrete 0-15
                        float reward,
                        const std::vector<float>& next_state,
                        bool done) {
    states_[pos_] = state;
    actions_[pos_] = action;
    rewards_[pos_] = reward;
    next_states_[pos_] = next_state;
    dones_[pos_] = done ? 1.0f : 0.0f;

    pos_ = (pos_ + 1) % capacity_;
    if (size_ < capacity_) {
        size_++;
    }
}

bool ReplayBuffer::sample(size_t batch_size,
                          std::vector<std::vector<float>>& states,
                          std::vector<int>& actions,  // discrete 0-15
                          std::vector<float>& rewards,
                          std::vector<std::vector<float>>& next_states,
                          std::vector<float>& dones) {
    if (size_ < batch_size) {
        return false;
    }

    states.resize(batch_size);
    actions.resize(batch_size);
    rewards.resize(batch_size);
    next_states.resize(batch_size);
    dones.resize(batch_size);

    std::uniform_int_distribution<size_t> dist(0, size_ - 1);

    for (size_t i = 0; i < batch_size; i++) {
        size_t idx = dist(rng_);
        states[i] = states_[idx];
        actions[i] = actions_[idx];
        rewards[i] = rewards_[idx];
        next_states[i] = next_states_[idx];
        dones[i] = dones_[idx];
    }

    return true;
}

size_t ReplayBuffer::sample_append(size_t n,
                                   std::vector<std::vector<float>>& states,
                                   std::vector<int>& actions,
                                   std::vector<float>& rewards,
                                   std::vector<std::vector<float>>& next_states,
                                   std::vector<float>& dones) {
    if (size_ == 0 || n == 0) {
        return 0;
    }

    std::uniform_int_distribution<size_t> dist(0, size_ - 1);

    for (size_t i = 0; i < n; i++) {
        size_t idx = dist(rng_);
        states.push_back(states_[idx]);
        actions.push_back(actions_[idx]);
        rewards.push_back(rewards_[idx]);
        next_states.push_back(next_states_[idx]);
        dones.push_back(dones_[idx]);
    }

    return n;
}

void ReplayBuffer::clear() {
    size_ = 0;
    pos_ = 0;
}

bool ReplayBuffer::save(const std::string& filepath) const {
    if (size_ == 0) {
        std::cerr << "ReplayBuffer::save - Buffer is empty, nothing to save" << std::endl;
        return false;
    }

    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "ReplayBuffer::save - Failed to open: " << filepath << std::endl;
        return false;
    }

    uint32_t obs_size = 0;
    if (!states_.empty() && !states_[0].empty()) {
        obs_size = static_cast<uint32_t>(states_[0].size());
    }

    // Header
    file.write(BUFFER_MAGIC, 4);
    file.write(reinterpret_cast<const char*>(&BUFFER_FILE_VERSION), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&obs_size), sizeof(uint32_t));

    uint64_t cap = capacity_;
    uint64_t sz = size_;
    uint64_t p = pos_;
    file.write(reinterpret_cast<const char*>(&cap), sizeof(uint64_t));
    file.write(reinterpret_cast<const char*>(&sz), sizeof(uint64_t));
    file.write(reinterpret_cast<const char*>(&p), sizeof(uint64_t));

    // States
    for (size_t i = 0; i < size_; i++) {
        file.write(reinterpret_cast<const char*>(states_[i].data()), obs_size * sizeof(float));
    }

    // Actions (discrete int32)
    for (size_t i = 0; i < size_; i++) {
        int32_t a = static_cast<int32_t>(actions_[i]);
        file.write(reinterpret_cast<const char*>(&a), sizeof(int32_t));
    }

    // Rewards
    file.write(reinterpret_cast<const char*>(rewards_.data()), size_ * sizeof(float));

    // Next states
    for (size_t i = 0; i < size_; i++) {
        file.write(reinterpret_cast<const char*>(next_states_[i].data()), obs_size * sizeof(float));
    }

    // Dones
    file.write(reinterpret_cast<const char*>(dones_.data()), size_ * sizeof(float));

    file.close();

    auto file_size = std::filesystem::file_size(filepath);
    std::cout << "ReplayBuffer::save - Saved " << size_ << " transitions to " << filepath
              << " (" << (file_size / 1024 / 1024) << " MB)" << std::endl;

    return true;
}

bool ReplayBuffer::load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "ReplayBuffer::load - Failed to open: " << filepath << std::endl;
        return false;
    }

    char magic[4];
    file.read(magic, 4);
    if (std::memcmp(magic, BUFFER_MAGIC, 4) != 0) {
        std::cerr << "ReplayBuffer::load - Invalid file format (bad magic)" << std::endl;
        return false;
    }

    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    if (version != BUFFER_FILE_VERSION) {
        std::cerr << "ReplayBuffer::load - Version mismatch (file: " << version
                  << ", expected: " << BUFFER_FILE_VERSION << ")" << std::endl;
        return false;
    }

    uint32_t obs_size;
    uint64_t saved_capacity, saved_size, saved_pos;

    file.read(reinterpret_cast<char*>(&obs_size), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&saved_capacity), sizeof(uint64_t));
    file.read(reinterpret_cast<char*>(&saved_size), sizeof(uint64_t));
    file.read(reinterpret_cast<char*>(&saved_pos), sizeof(uint64_t));

    std::cout << "ReplayBuffer::load - Loading " << saved_size << " transitions (obs_size="
              << obs_size << ")" << std::endl;

    clear();

    // States
    for (size_t i = 0; i < saved_size; i++) {
        std::vector<float> state(obs_size);
        file.read(reinterpret_cast<char*>(state.data()), obs_size * sizeof(float));
        if (i < capacity_) {
            states_[i] = std::move(state);
        }
    }

    // Actions (discrete int32)
    for (size_t i = 0; i < saved_size; i++) {
        int32_t a;
        file.read(reinterpret_cast<char*>(&a), sizeof(int32_t));
        if (i < capacity_) {
            actions_[i] = static_cast<int>(a);
        }
    }

    // Rewards
    std::vector<float> temp_rewards(saved_size);
    file.read(reinterpret_cast<char*>(temp_rewards.data()), saved_size * sizeof(float));
    for (size_t i = 0; i < std::min(saved_size, capacity_); i++) {
        rewards_[i] = temp_rewards[i];
    }

    // Next states
    for (size_t i = 0; i < saved_size; i++) {
        std::vector<float> next_state(obs_size);
        file.read(reinterpret_cast<char*>(next_state.data()), obs_size * sizeof(float));
        if (i < capacity_) {
            next_states_[i] = std::move(next_state);
        }
    }

    // Dones
    std::vector<float> temp_dones(saved_size);
    file.read(reinterpret_cast<char*>(temp_dones.data()), saved_size * sizeof(float));
    for (size_t i = 0; i < std::min(saved_size, capacity_); i++) {
        dones_[i] = temp_dones[i];
    }

    size_ = std::min(saved_size, capacity_);
    pos_ = size_ % capacity_;

    file.close();

    std::cout << "ReplayBuffer::load - Loaded " << size_ << " transitions" << std::endl;

    return true;
}

bool ReplayBuffer::load_and_merge(const std::vector<std::string>& filepaths) {
    size_t total_loaded = 0;

    for (const auto& filepath : filepaths) {
        if (!std::filesystem::exists(filepath)) {
            std::cout << "ReplayBuffer::load_and_merge - Skipping (not found): " << filepath << std::endl;
            continue;
        }

        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "ReplayBuffer::load_and_merge - Failed to open: " << filepath << std::endl;
            continue;
        }

        char magic[4];
        file.read(magic, 4);
        if (std::memcmp(magic, BUFFER_MAGIC, 4) != 0) {
            std::cerr << "ReplayBuffer::load_and_merge - Invalid file: " << filepath << std::endl;
            file.close();
            continue;
        }

        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
        if (version != BUFFER_FILE_VERSION) {
            std::cerr << "ReplayBuffer::load_and_merge - Version mismatch: " << filepath << std::endl;
            file.close();
            continue;
        }

        uint32_t obs_size;
        uint64_t saved_capacity, saved_size, saved_pos;

        file.read(reinterpret_cast<char*>(&obs_size), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&saved_capacity), sizeof(uint64_t));
        file.read(reinterpret_cast<char*>(&saved_size), sizeof(uint64_t));
        file.read(reinterpret_cast<char*>(&saved_pos), sizeof(uint64_t));

        std::cout << "ReplayBuffer::load_and_merge - Reading " << saved_size
                  << " transitions from " << filepath << std::endl;

        // Load into temp buffers
        std::vector<std::vector<float>> temp_states(saved_size);
        std::vector<int> temp_actions(saved_size);
        std::vector<float> temp_rewards(saved_size);
        std::vector<std::vector<float>> temp_next_states(saved_size);
        std::vector<float> temp_dones(saved_size);

        // States
        for (size_t i = 0; i < saved_size; i++) {
            temp_states[i].resize(obs_size);
            file.read(reinterpret_cast<char*>(temp_states[i].data()), obs_size * sizeof(float));
        }

        // Actions (discrete int32)
        for (size_t i = 0; i < saved_size; i++) {
            int32_t a;
            file.read(reinterpret_cast<char*>(&a), sizeof(int32_t));
            temp_actions[i] = static_cast<int>(a);
        }

        // Rewards
        file.read(reinterpret_cast<char*>(temp_rewards.data()), saved_size * sizeof(float));

        // Next states
        for (size_t i = 0; i < saved_size; i++) {
            temp_next_states[i].resize(obs_size);
            file.read(reinterpret_cast<char*>(temp_next_states[i].data()), obs_size * sizeof(float));
        }

        // Dones
        file.read(reinterpret_cast<char*>(temp_dones.data()), saved_size * sizeof(float));

        file.close();

        // Push to buffer
        size_t loaded_from_file = 0;
        for (size_t i = 0; i < saved_size; i++) {
            push(temp_states[i], temp_actions[i], temp_rewards[i],
                 temp_next_states[i], temp_dones[i] > 0.5f);
            loaded_from_file++;
        }

        total_loaded += loaded_from_file;
        std::cout << "ReplayBuffer::load_and_merge - Merged " << loaded_from_file
                  << " transitions (buffer size: " << size_ << ")" << std::endl;
    }

    std::cout << "ReplayBuffer::load_and_merge - Total merged: " << total_loaded
              << " transitions, final buffer size: " << size_ << std::endl;

    return total_loaded > 0;
}

} // namespace sac

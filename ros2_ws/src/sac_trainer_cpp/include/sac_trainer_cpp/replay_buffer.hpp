/**
 * Replay Buffer Header (Discrete Actions, with Save/Load)
 */

#ifndef REPLAY_BUFFER_HPP
#define REPLAY_BUFFER_HPP

#include <vector>
#include <random>
#include <string>

namespace sac {

class ReplayBuffer {
public:
    ReplayBuffer(size_t capacity);

    void push(const std::vector<float>& state,
              int action,  // discrete 0-15
              float reward,
              const std::vector<float>& next_state,
              bool done);

    bool sample(size_t batch_size,
                std::vector<std::vector<float>>& states,
                std::vector<int>& actions,  // discrete 0-15
                std::vector<float>& rewards,
                std::vector<std::vector<float>>& next_states,
                std::vector<float>& dones);

    // RLPD symmetric sampling: append `n` random transitions to the END of the
    // provided vectors (does NOT clear them). Returns number actually appended
    // (0 if buffer empty). Used to build a mixed batch (e.g. half RULE, half online).
    size_t sample_append(size_t n,
                         std::vector<std::vector<float>>& states,
                         std::vector<int>& actions,
                         std::vector<float>& rewards,
                         std::vector<std::vector<float>>& next_states,
                         std::vector<float>& dones);

    void clear();
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }

    bool save(const std::string& filepath) const;
    bool load(const std::string& filepath);
    bool load_and_merge(const std::vector<std::string>& filepaths);

private:
    size_t capacity_;
    size_t size_;
    size_t pos_;

    std::vector<std::vector<float>> states_;
    std::vector<int> actions_;  // discrete 0-15
    std::vector<float> rewards_;
    std::vector<std::vector<float>> next_states_;
    std::vector<float> dones_;

    std::mt19937 rng_;
};

} // namespace sac

#endif // REPLAY_BUFFER_HPP

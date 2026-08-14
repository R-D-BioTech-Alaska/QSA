#pragma once

#include "qubit/qstate.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qubit {

enum class BehaviorJoin : std::uint8_t {
    And = 0,
    Or = 1,
    Xor = 2,
};

struct BehaviorFoldConfig {
    std::size_t max_states_per_level{8U << 20U};
    std::size_t max_index_bytes{512U << 20U};
    std::uint64_t max_destination_lookups{100000000ULL};
};

struct BehaviorFoldPlan {
    std::vector<std::uint32_t> branch_indices{};
    std::vector<BehaviorJoin> joins{};

    [[nodiscard]] std::size_t leaf_count() const noexcept { return branch_indices.size(); }
    friend bool operator==(const BehaviorFoldPlan&, const BehaviorFoldPlan&) = default;
};

struct BehaviorFoldStats {
    std::size_t branch_count{0U};
    std::size_t probe_bit_count{0U};
    std::size_t max_leaves{0U};
    std::size_t behavior_class_count{0U};
    std::size_t deep_behavior_count{0U};
    std::vector<std::size_t> depth_state_counts{};
    std::vector<std::uint64_t> structural_count_by_leaf{};
    std::uint64_t eligible_pairs{0U};
    std::uint64_t destination_lookups{0U};
    std::size_t estimated_bytes{0U};
    double fold_seconds{0.0};
    double index_seconds{0.0};
    double build_seconds{0.0};
};

struct BehaviorFoldUniqueMatch {
    std::size_t count{0U};
    std::optional<std::uint64_t> behavior{};
    std::optional<BehaviorFoldPlan> plan{};
};

namespace detail {

class BehaviorSha256 {
public:
    BehaviorSha256() { reset(); }

    void update(const void* data, std::size_t size) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        if (size > (std::numeric_limits<std::uint64_t>::max() - bit_count_) / 8U) {
            throw QStateError("Behavior SHA-256 input is too large");
        }
        bit_count_ += static_cast<std::uint64_t>(size) * 8U;
        while (size != 0U) {
            const std::size_t copied = std::min(size, block_.size() - block_size_);
            std::copy_n(bytes, copied, block_.begin() + static_cast<std::ptrdiff_t>(block_size_));
            block_size_ += copied;
            bytes += copied;
            size -= copied;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0U;
            }
        }
    }

    [[nodiscard]] std::string finish() {
        const std::uint64_t message_bits = bit_count_;
        const std::uint8_t marker = 0x80U;
        update_padding(&marker, 1U);
        const std::uint8_t zero = 0U;
        while (block_size_ != 56U) update_padding(&zero, 1U);
        std::array<std::uint8_t, 8> length{};
        for (std::size_t i = 0U; i < length.size(); ++i) {
            length[7U - i] = static_cast<std::uint8_t>((message_bits >> (i * 8U)) & 0xFFU);
        }
        update_padding(length.data(), length.size());

        constexpr char digits[] = "0123456789abcdef";
        std::string result(64U, '0');
        for (std::size_t i = 0U; i < state_.size(); ++i) {
            for (std::size_t byte = 0U; byte < 4U; ++byte) {
                const std::uint8_t value = static_cast<std::uint8_t>(
                    (state_[i] >> ((3U - byte) * 8U)) & 0xFFU);
                const std::size_t offset = (i * 4U + byte) * 2U;
                result[offset] = digits[value >> 4U];
                result[offset + 1U] = digits[value & 0x0FU];
            }
        }
        return result;
    }

private:
    static constexpr std::array<std::uint32_t, 64> constants_{
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
    };

    void reset() noexcept {
        state_ = {
            0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
            0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
        };
        block_.fill(0U);
        block_size_ = 0U;
        bit_count_ = 0U;
    }

    void update_padding(const void* data, std::size_t size) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        while (size != 0U) {
            const std::size_t copied = std::min(size, block_.size() - block_size_);
            std::copy_n(bytes, copied, block_.begin() + static_cast<std::ptrdiff_t>(block_size_));
            block_size_ += copied;
            bytes += copied;
            size -= copied;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0U;
            }
        }
    }

    void transform(const std::uint8_t* block) noexcept {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t i = 0U; i < 16U; ++i) {
            const std::size_t offset = i * 4U;
            words[i] = (static_cast<std::uint32_t>(block[offset]) << 24U) |
                       (static_cast<std::uint32_t>(block[offset + 1U]) << 16U) |
                       (static_cast<std::uint32_t>(block[offset + 2U]) << 8U) |
                       static_cast<std::uint32_t>(block[offset + 3U]);
        }
        for (std::size_t i = 16U; i < words.size(); ++i) {
            const std::uint32_t s0 = std::rotr(words[i - 15U], 7) ^
                                     std::rotr(words[i - 15U], 18) ^
                                     (words[i - 15U] >> 3U);
            const std::uint32_t s1 = std::rotr(words[i - 2U], 17) ^
                                     std::rotr(words[i - 2U], 19) ^
                                     (words[i - 2U] >> 10U);
            words[i] = words[i - 16U] + s0 + words[i - 7U] + s1;
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (std::size_t i = 0U; i < words.size(); ++i) {
            const std::uint32_t s1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
            const std::uint32_t choose = (e & f) ^ ((~e) & g);
            const std::uint32_t temp1 = h + s1 + choose + constants_[i] + words[i];
            const std::uint32_t s0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> block_{};
    std::size_t block_size_{0U};
    std::uint64_t bit_count_{0U};
};

inline void behavior_hash_u32(BehaviorSha256& hash, std::uint32_t value) {
    const std::array<std::uint8_t, 4> bytes{
        static_cast<std::uint8_t>(value & 0xFFU),
        static_cast<std::uint8_t>((value >> 8U) & 0xFFU),
        static_cast<std::uint8_t>((value >> 16U) & 0xFFU),
        static_cast<std::uint8_t>((value >> 24U) & 0xFFU),
    };
    hash.update(bytes.data(), bytes.size());
}

inline void behavior_hash_u64(BehaviorSha256& hash, std::uint64_t value) {
    std::array<std::uint8_t, 8> bytes{};
    for (std::size_t i = 0U; i < bytes.size(); ++i) {
        bytes[i] = static_cast<std::uint8_t>((value >> (i * 8U)) & 0xFFU);
    }
    hash.update(bytes.data(), bytes.size());
}

}  // namespace detail

class ExactBehaviorFoldUniverse {
public:
    ExactBehaviorFoldUniverse(
        std::span<const std::uint64_t> branch_masks,
        std::size_t probe_bit_count,
        std::size_t max_leaves = 4U,
        std::span<const BehaviorJoin> joins = default_joins(),
        BehaviorFoldConfig config = {})
        : branch_masks_(branch_masks.begin(), branch_masks.end()),
          probe_bit_count_(probe_bit_count),
          max_leaves_(max_leaves),
          joins_(joins.begin(), joins.end()),
          config_(config) {
        validate_contract();
        const auto started = std::chrono::steady_clock::now();
        build_levels();
        const auto folded = std::chrono::steady_clock::now();
        build_canonical_population();
        const auto index_started = std::chrono::steady_clock::now();
        build_index();
        const auto finished = std::chrono::steady_clock::now();
        stats_.fold_seconds = std::chrono::duration<double>(folded - started).count();
        stats_.index_seconds = std::chrono::duration<double>(finished - index_started).count();
        stats_.build_seconds = std::chrono::duration<double>(finished - started).count();
        finalize_stats();
    }

    [[nodiscard]] static std::span<const BehaviorJoin> default_joins() noexcept {
        static constexpr std::array<BehaviorJoin, 3> values{
            BehaviorJoin::And,
            BehaviorJoin::Or,
            BehaviorJoin::Xor,
        };
        return values;
    }

    [[nodiscard]] const BehaviorFoldStats& stats() const noexcept { return stats_; }

    [[nodiscard]] bool contains_at_depth(std::size_t leaf_count, std::uint64_t behavior) const {
        validate_depth(leaf_count);
        return levels_[leaf_count - 1U].find(behavior) != levels_[leaf_count - 1U].end();
    }

    [[nodiscard]] std::optional<BehaviorFoldPlan> canonical_plan(std::uint64_t behavior) const {
        for (std::size_t depth = 0U; depth < levels_.size(); ++depth) {
            const auto found = levels_[depth].find(behavior);
            if (found != levels_[depth].end()) return decode_plan(found->second, depth + 1U);
        }
        return std::nullopt;
    }

    [[nodiscard]] std::size_t conditioned_count(
        std::span<const std::size_t> positions,
        std::span<const std::uint8_t> bits) const {
        return matching(positions, bits).first;
    }

    [[nodiscard]] BehaviorFoldUniqueMatch conditioned_unique(
        std::span<const std::size_t> positions,
        std::span<const std::uint8_t> bits) const {
        const auto [count, index] = matching(positions, bits);
        BehaviorFoldUniqueMatch result;
        result.count = count;
        if (count == 1U && index.has_value()) {
            result.behavior = behaviors_[*index];
            result.plan = canonical_plan(*result.behavior);
            if (!result.plan.has_value()) throw QStateError("Behavior index lost canonical plan identity");
        }
        return result;
    }

    [[nodiscard]] std::string state_digest(std::size_t leaf_count) const {
        validate_depth(leaf_count);
        std::optional<std::string>& cached = state_digest_cache_[leaf_count - 1U];
        if (!cached.has_value()) cached = compute_state_digest(leaf_count);
        return *cached;
    }

    [[nodiscard]] std::string canonical_digest() const {
        if (!canonical_digest_cache_.has_value()) canonical_digest_cache_ = compute_canonical_digest();
        return *canonical_digest_cache_;
    }

private:
    struct State {
        std::uint32_t last_index{0U};
        std::uint64_t branch_code{0U};
        std::uint64_t join_code{0U};
    };

    struct Row {
        std::uint64_t behavior{0U};
        State state{};
    };

    static std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right, const char* message) {
        if (right != 0U && left > std::numeric_limits<std::uint64_t>::max() / right) {
            throw QStateError(message);
        }
        return left * right;
    }

    static std::uint64_t binomial(std::size_t n, std::size_t k) {
        if (k > n) return 0U;
        k = std::min(k, n - k);
        std::uint64_t value = 1U;
        for (std::size_t i = 1U; i <= k; ++i) {
            std::uint64_t numerator = static_cast<std::uint64_t>(n - k + i);
            std::uint64_t denominator = static_cast<std::uint64_t>(i);
            const std::uint64_t first = std::gcd(numerator, denominator);
            numerator /= first;
            denominator /= first;
            const std::uint64_t second = std::gcd(value, denominator);
            value /= second;
            denominator /= second;
            if (denominator != 1U) throw QStateError("Behavior structural count division is not exact");
            value = checked_mul(value, numerator, "Behavior structural count overflow");
        }
        return value;
    }

    static std::uint64_t pow_checked(std::uint64_t base, std::size_t exponent) {
        std::uint64_t value = 1U;
        for (std::size_t i = 0U; i < exponent; ++i) {
            value = checked_mul(value, base, "Behavior structural count overflow");
        }
        return value;
    }

    void validate_contract() {
        if (branch_masks_.empty() || branch_masks_.size() > std::numeric_limits<std::uint32_t>::max() ||
            probe_bit_count_ == 0U || probe_bit_count_ > 64U || max_leaves_ == 0U ||
            branch_masks_.size() < max_leaves_ || joins_.empty() || joins_.size() > 3U ||
            config_.max_states_per_level == 0U || config_.max_index_bytes == 0U ||
            config_.max_destination_lookups == 0U) {
            throw QStateError("Behavior fold construction contract is invalid");
        }
        std::array<bool, 3> seen{};
        for (const BehaviorJoin join : joins_) {
            const std::size_t code = static_cast<std::size_t>(join);
            if (code >= seen.size() || seen[code]) throw QStateError("Behavior join set is invalid");
            seen[code] = true;
        }
        if (probe_bit_count_ != 64U) {
            const std::uint64_t allowed = (std::uint64_t{1U} << probe_bit_count_) - 1U;
            for (const std::uint64_t mask : branch_masks_) {
                if ((mask & ~allowed) != 0U) throw QStateError("Behavior branch mask exceeds probe width");
            }
        }
        state_digest_cache_.resize(max_leaves_);
    }

    [[nodiscard]] std::uint64_t apply_join(
        std::uint64_t left,
        std::uint64_t right,
        BehaviorJoin join) const noexcept {
        switch (join) {
            case BehaviorJoin::And:
                return left & right;
            case BehaviorJoin::Or:
                return left | right;
            case BehaviorJoin::Xor:
                return left ^ right;
        }
        return 0U;
    }

    void build_levels() {
        const std::size_t branch_count = branch_masks_.size();
        std::vector<std::int64_t> previous_same(branch_count, -1);
        std::unordered_map<std::uint64_t, std::uint32_t> last_by_mask;
        last_by_mask.reserve(branch_count * 2U);
        for (std::size_t index = 0U; index < branch_count; ++index) {
            const auto found = last_by_mask.find(branch_masks_[index]);
            if (found != last_by_mask.end()) {
                previous_same[index] = static_cast<std::int64_t>(found->second);
                found->second = static_cast<std::uint32_t>(index);
            } else {
                last_by_mask.emplace(branch_masks_[index], static_cast<std::uint32_t>(index));
            }
        }

        levels_.resize(max_leaves_);
        auto& level_one = levels_[0];
        level_one.max_load_factor(0.85F);
        level_one.reserve(std::min(branch_count * 2U, config_.max_states_per_level));
        for (std::size_t index = 0U; index < branch_count; ++index) {
            level_one.try_emplace(
                branch_masks_[index],
                State{static_cast<std::uint32_t>(index), static_cast<std::uint64_t>(index), 0U});
            if (level_one.size() > config_.max_states_per_level) {
                throw QStateError("Behavior state cap exceeded at depth one");
            }
        }

        for (std::size_t depth = 2U; depth <= max_leaves_; ++depth) {
            const auto& prior = levels_[depth - 2U];
            std::vector<Row> previous;
            previous.reserve(prior.size());
            for (const auto& [behavior, state] : prior) previous.push_back(Row{behavior, state});
            std::sort(previous.begin(), previous.end(), [](const Row& left, const Row& right) {
                if (left.state.branch_code != right.state.branch_code) {
                    return left.state.branch_code < right.state.branch_code;
                }
                return left.state.join_code < right.state.join_code;
            });

            auto& current = levels_[depth - 1U];
            current.max_load_factor(0.85F);
            const std::size_t upper = previous.size() > std::numeric_limits<std::size_t>::max() / branch_count
                ? config_.max_states_per_level
                : std::min(previous.size() * branch_count, config_.max_states_per_level);
            current.reserve(upper);

            for (std::size_t branch_index = 0U; branch_index < branch_count; ++branch_index) {
                const std::int64_t lower = previous_same[branch_index];
                const std::uint64_t branch_mask = branch_masks_[branch_index];
                for (const Row& row : previous) {
                    const std::int64_t last = static_cast<std::int64_t>(row.state.last_index);
                    if (last >= static_cast<std::int64_t>(branch_index) || last < lower) continue;
                    ++stats_.eligible_pairs;
                    const std::uint64_t next_branch_code = checked_mul(
                        row.state.branch_code,
                        static_cast<std::uint64_t>(branch_count),
                        "Behavior branch rank overflow");
                    if (next_branch_code > std::numeric_limits<std::uint64_t>::max() - branch_index) {
                        throw QStateError("Behavior branch rank overflow");
                    }
                    std::array<std::uint64_t, 3> emitted{};
                    std::size_t emitted_count = 0U;
                    for (std::size_t join_index = 0U; join_index < joins_.size(); ++join_index) {
                        const std::uint64_t output = apply_join(row.behavior, branch_mask, joins_[join_index]);
                        if (std::find(emitted.begin(), emitted.begin() + static_cast<std::ptrdiff_t>(emitted_count), output) !=
                            emitted.begin() + static_cast<std::ptrdiff_t>(emitted_count)) {
                            continue;
                        }
                        emitted[emitted_count++] = output;
                        if (stats_.destination_lookups >= config_.max_destination_lookups) {
                            throw QStateError("Behavior destination lookup cap exceeded");
                        }
                        ++stats_.destination_lookups;
                        const std::uint64_t next_join_code = checked_mul(
                            row.state.join_code,
                            static_cast<std::uint64_t>(joins_.size()),
                            "Behavior join rank overflow");
                        if (next_join_code > std::numeric_limits<std::uint64_t>::max() - join_index) {
                            throw QStateError("Behavior join rank overflow");
                        }
                        current.try_emplace(
                            output,
                            State{
                                static_cast<std::uint32_t>(branch_index),
                                next_branch_code + static_cast<std::uint64_t>(branch_index),
                                next_join_code + static_cast<std::uint64_t>(join_index),
                            });
                        if (current.size() > config_.max_states_per_level) {
                            throw QStateError("Behavior state cap exceeded");
                        }
                    }
                }
            }
        }
    }

    void build_canonical_population() {
        std::size_t total = 0U;
        for (const auto& level : levels_) {
            if (level.size() > std::numeric_limits<std::size_t>::max() - total) {
                throw QStateError("Behavior canonical population size overflow");
            }
            total += level.size();
        }
        behaviors_.reserve(total);
        for (const auto& level : levels_) {
            for (const auto& [behavior, state] : level) {
                (void)state;
                behaviors_.push_back(behavior);
            }
        }
        std::sort(behaviors_.begin(), behaviors_.end());
        behaviors_.erase(std::unique(behaviors_.begin(), behaviors_.end()), behaviors_.end());
    }

    void build_index() {
        const std::size_t words = (behaviors_.size() + 63U) / 64U;
        if (words != 0U && probe_bit_count_ > std::numeric_limits<std::size_t>::max() / words) {
            throw QStateError("Behavior index size overflow");
        }
        const std::size_t word_count = probe_bit_count_ * words;
        if (word_count > config_.max_index_bytes / sizeof(std::uint64_t)) {
            throw QStateError("Behavior index byte cap exceeded");
        }
        index_words_ = words;
        slices_.assign(word_count, 0U);
        for (std::size_t index = 0U; index < behaviors_.size(); ++index) {
            const std::size_t word = index >> 6U;
            const std::uint64_t bit = std::uint64_t{1U} << (index & 63U);
            const std::uint64_t behavior = behaviors_[index];
            for (std::size_t position = 0U; position < probe_bit_count_; ++position) {
                if (((behavior >> position) & 1U) != 0U) slices_[position * index_words_ + word] |= bit;
            }
        }
    }

    void finalize_stats() {
        stats_.branch_count = branch_masks_.size();
        stats_.probe_bit_count = probe_bit_count_;
        stats_.max_leaves = max_leaves_;
        stats_.behavior_class_count = behaviors_.size();
        stats_.depth_state_counts.reserve(levels_.size());
        stats_.structural_count_by_leaf.reserve(levels_.size());
        for (std::size_t depth = 1U; depth <= levels_.size(); ++depth) {
            stats_.depth_state_counts.push_back(levels_[depth - 1U].size());
            const std::uint64_t combinations = binomial(branch_masks_.size(), depth);
            const std::uint64_t join_choices = pow_checked(joins_.size(), depth - 1U);
            stats_.structural_count_by_leaf.push_back(
                checked_mul(combinations, join_choices, "Behavior structural count overflow"));
        }
        for (const auto& [behavior, state] : levels_.back()) {
            (void)state;
            bool shallow = false;
            for (std::size_t depth = 0U; depth + 1U < levels_.size(); ++depth) {
                if (levels_[depth].find(behavior) != levels_[depth].end()) {
                    shallow = true;
                    break;
                }
            }
            if (!shallow) ++stats_.deep_behavior_count;
        }
        std::size_t bytes = behaviors_.capacity() * sizeof(std::uint64_t) +
                            slices_.capacity() * sizeof(std::uint64_t);
        for (const auto& level : levels_) {
            bytes += level.bucket_count() * sizeof(void*);
            bytes += level.size() * sizeof(typename decltype(level)::value_type);
        }
        stats_.estimated_bytes = bytes;
    }

    void validate_depth(std::size_t leaf_count) const {
        if (leaf_count == 0U || leaf_count > levels_.size()) {
            throw QStateError("Behavior leaf count is outside the constructed universe");
        }
    }

    [[nodiscard]] BehaviorFoldPlan decode_plan(const State& state, std::size_t leaf_count) const {
        BehaviorFoldPlan result;
        result.branch_indices.resize(leaf_count);
        std::uint64_t branch_code = state.branch_code;
        for (std::size_t offset = leaf_count; offset != 0U; --offset) {
            result.branch_indices[offset - 1U] = static_cast<std::uint32_t>(branch_code % branch_masks_.size());
            branch_code /= branch_masks_.size();
        }
        if (branch_code != 0U) throw QStateError("Behavior branch rank exceeds plan width");

        result.joins.resize(leaf_count - 1U);
        std::uint64_t join_code = state.join_code;
        for (std::size_t offset = result.joins.size(); offset != 0U; --offset) {
            const std::size_t index = static_cast<std::size_t>(join_code % joins_.size());
            join_code /= joins_.size();
            result.joins[offset - 1U] = joins_[index];
        }
        if (join_code != 0U) throw QStateError("Behavior join rank exceeds plan width");
        return result;
    }

    [[nodiscard]] std::pair<std::size_t, std::optional<std::size_t>> matching(
        std::span<const std::size_t> positions,
        std::span<const std::uint8_t> bits) const {
        if (positions.size() != bits.size()) throw QStateError("Behavior observation lengths differ");
        std::vector<bool> seen(probe_bit_count_, false);
        for (std::size_t i = 0U; i < positions.size(); ++i) {
            if (positions[i] >= probe_bit_count_ || seen[positions[i]] || bits[i] > 1U) {
                throw QStateError("Behavior observation is invalid");
            }
            seen[positions[i]] = true;
        }

        std::size_t count = 0U;
        std::optional<std::size_t> only;
        for (std::size_t word = 0U; word < index_words_; ++word) {
            std::uint64_t matches = std::numeric_limits<std::uint64_t>::max();
            if (word + 1U == index_words_ && (behaviors_.size() & 63U) != 0U) {
                matches = (std::uint64_t{1U} << (behaviors_.size() & 63U)) - 1U;
            }
            for (std::size_t i = 0U; i < positions.size() && matches != 0U; ++i) {
                const std::uint64_t ones = slices_[positions[i] * index_words_ + word];
                matches &= bits[i] != 0U ? ones : ~ones;
            }
            const std::size_t local = static_cast<std::size_t>(std::popcount(matches));
            if (local != 0U && !only.has_value()) {
                only = word * 64U + static_cast<std::size_t>(std::countr_zero(matches));
            }
            count += local;
        }
        if (count != 1U) only.reset();
        return {count, only};
    }

    void hash_plan(detail::BehaviorSha256& hash, const State& state, std::size_t leaf_count) const {
        detail::behavior_hash_u32(hash, static_cast<std::uint32_t>(leaf_count));
        const BehaviorFoldPlan plan = decode_plan(state, leaf_count);
        for (const std::uint32_t branch : plan.branch_indices) detail::behavior_hash_u32(hash, branch);
        detail::behavior_hash_u32(hash, static_cast<std::uint32_t>(plan.joins.size()));
        for (const BehaviorJoin join : plan.joins) {
            const std::uint8_t code = static_cast<std::uint8_t>(join);
            hash.update(&code, 1U);
        }
    }

    [[nodiscard]] std::string compute_state_digest(std::size_t leaf_count) const {
        const auto& level = levels_[leaf_count - 1U];
        std::vector<Row> rows;
        rows.reserve(level.size());
        for (const auto& [behavior, state] : level) rows.push_back(Row{behavior, state});
        std::sort(rows.begin(), rows.end(), [](const Row& left, const Row& right) {
            return left.behavior < right.behavior;
        });
        detail::BehaviorSha256 hash;
        for (const Row& row : rows) {
            detail::behavior_hash_u64(hash, row.behavior);
            detail::behavior_hash_u32(hash, row.state.last_index);
            hash_plan(hash, row.state, leaf_count);
        }
        return hash.finish();
    }

    [[nodiscard]] std::string compute_canonical_digest() const {
        detail::BehaviorSha256 hash;
        for (const std::uint64_t behavior : behaviors_) {
            bool found = false;
            for (std::size_t depth = 0U; depth < levels_.size(); ++depth) {
                const auto state = levels_[depth].find(behavior);
                if (state == levels_[depth].end()) continue;
                detail::behavior_hash_u64(hash, behavior);
                hash_plan(hash, state->second, depth + 1U);
                found = true;
                break;
            }
            if (!found) throw QStateError("Behavior canonical population lost plan identity");
        }
        return hash.finish();
    }

    std::vector<std::uint64_t> branch_masks_{};
    std::size_t probe_bit_count_{0U};
    std::size_t max_leaves_{0U};
    std::vector<BehaviorJoin> joins_{};
    BehaviorFoldConfig config_{};
    std::vector<std::unordered_map<std::uint64_t, State>> levels_{};
    std::vector<std::uint64_t> behaviors_{};
    std::size_t index_words_{0U};
    std::vector<std::uint64_t> slices_{};
    BehaviorFoldStats stats_{};
    mutable std::vector<std::optional<std::string>> state_digest_cache_{};
    mutable std::optional<std::string> canonical_digest_cache_{};
};

}  // namespace qubit

#pragma once

#include "qubit/qstate.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace qubit {

struct TropicalQTTEntry {
    std::int64_t cost{0};
    bool supported{false};

    [[nodiscard]] static constexpr TropicalQTTEntry edge(std::int64_t value) noexcept {
        return TropicalQTTEntry{value, true};
    }

    [[nodiscard]] static constexpr TropicalQTTEntry missing() noexcept {
        return TropicalQTTEntry{};
    }
};

struct TropicalQTTCore {
    std::size_t left_rank{0U};
    std::size_t right_rank{0U};
    std::vector<TropicalQTTEntry> zero{};
    std::vector<TropicalQTTEntry> one{};
};

struct TropicalQTTConfig {
    std::size_t max_rank{1024U};
    std::size_t max_core_entries{1U << 20U};
    std::size_t max_total_entries{1U << 24U};
    std::size_t max_backpointer_entries{1U << 22U};
};

struct TropicalQTTStats {
    std::size_t logical_bits{0U};
    std::size_t maximum_rank{0U};
    std::size_t descriptor_entries{0U};
    std::size_t supported_entries{0U};
};

struct TropicalQTTMinimum {
    std::vector<std::uint8_t> bits{};
    std::int64_t energy{0};
    std::size_t candidate_evaluations{0U};
    std::size_t maximum_active_bond_states{0U};
};

class ExactTropicalQTT {
public:
    [[nodiscard]] static ExactTropicalQTT from_certified_cores(
        std::vector<TropicalQTTCore> cores,
        TropicalQTTConfig config = {}) {
        return ExactTropicalQTT(std::move(cores), config);
    }

    [[nodiscard]] static ExactTropicalQTT weighted_bit_sum(
        std::span<const std::int64_t> weights,
        std::int64_t offset = 0,
        TropicalQTTConfig config = {}) {
        if (weights.empty()) {
            throw QStateError("Tropical QTT weighted bit sum requires at least one bit");
        }
        preflight_rank_one(weights.size(), config);

        std::vector<TropicalQTTCore> cores;
        cores.reserve(weights.size());
        for (std::size_t position = 0U; position < weights.size(); ++position) {
            const std::int64_t zero = position == 0U ? offset : 0;
            const std::int64_t one = position == 0U
                ? checked_cost_sum(offset, weights[position])
                : weights[position];
            cores.push_back(TropicalQTTCore{
                1U,
                1U,
                std::vector<TropicalQTTEntry>{TropicalQTTEntry::edge(zero)},
                std::vector<TropicalQTTEntry>{TropicalQTTEntry::edge(one)},
            });
        }
        return ExactTropicalQTT(std::move(cores), config);
    }

    [[nodiscard]] static ExactTropicalQTT structured_markov_energy(
        std::size_t logical_bits,
        std::size_t rank,
        std::int64_t seed = 17,
        TropicalQTTConfig config = {}) {
        if (logical_bits == 0U || rank == 0U) {
            throw QStateError("Tropical QTT logical bits and rank must be positive");
        }
        preflight_markov(logical_bits, rank, config);
        const std::int64_t seed_mod = positive_mod(seed, 29);

        std::vector<TropicalQTTCore> cores;
        cores.reserve(logical_bits);
        for (std::size_t position = 0U; position < logical_bits; ++position) {
            const std::size_t left_rank = position == 0U ? 1U : rank;
            const std::size_t right_rank = position + 1U == logical_bits ? 1U : rank;
            const std::size_t matrix_size = left_rank * right_rank;
            TropicalQTTCore core{
                left_rank,
                right_rank,
                std::vector<TropicalQTTEntry>(matrix_size),
                std::vector<TropicalQTTEntry>(matrix_size),
            };
            for (std::uint8_t bit = 0U; bit < 2U; ++bit) {
                std::vector<TropicalQTTEntry>& matrix = bit == 0U ? core.zero : core.one;
                for (std::size_t left = 0U; left < left_rank; ++left) {
                    for (std::size_t right = 0U; right < right_rank; ++right) {
                        const std::size_t distance = left > right ? left - right : right - left;
                        const std::int64_t distance_cost = checked_size_to_cost(distance);
                        const std::int64_t square = checked_nonnegative_cost_product(
                            distance_cost, distance_cost);
                        const std::int64_t base = checked_nonnegative_cost_product(
                            checked_size_to_cost(position + 3U),
                            static_cast<std::int64_t>(bit) + 1);
                        const std::int64_t left_mod = static_cast<std::int64_t>(
                            ((left % 29U) + 1U) % 29U);
                        const std::int64_t right_mod = static_cast<std::int64_t>(
                            ((right % 29U) + 1U) % 29U);
                        const std::int64_t mix = positive_mod(
                            seed_mod * left_mod + 7 * right_mod +
                                11 * static_cast<std::int64_t>(bit) +
                                static_cast<std::int64_t>(position % 29U),
                            29);
                        const std::int64_t cost = checked_cost_sum(
                            checked_cost_sum(base, square), mix);
                        matrix[left * right_rank + right] = TropicalQTTEntry::edge(cost);
                    }
                }
            }
            cores.push_back(std::move(core));
        }
        return ExactTropicalQTT(std::move(cores), config);
    }

    [[nodiscard]] std::size_t logical_bits() const noexcept { return cores_.size(); }
    [[nodiscard]] const TropicalQTTConfig& config() const noexcept { return config_; }
    [[nodiscard]] const TropicalQTTStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const std::vector<TropicalQTTCore>& cores() const noexcept { return cores_; }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) + cores_.capacity() * sizeof(TropicalQTTCore);
        for (const TropicalQTTCore& core : cores_) {
            bytes += core.zero.capacity() * sizeof(TropicalQTTEntry);
            bytes += core.one.capacity() * sizeof(TropicalQTTEntry);
        }
        return bytes;
    }

    [[nodiscard]] std::int64_t energy_bits(std::span<const std::uint8_t> bits) const {
        if (bits.size() != cores_.size()) {
            throw QStateError("Tropical QTT bit string does not match logical shape");
        }
        std::vector<CostState> costs(1U, CostState{true, 0});
        for (std::size_t index = 0U; index < cores_.size(); ++index) {
            const std::uint8_t bit = bits[index];
            if (bit > 1U) {
                throw QStateError("Tropical QTT sector bits must be 0 or 1");
            }
            const TropicalQTTCore& core = cores_[index];
            const std::vector<TropicalQTTEntry>& matrix = bit == 0U ? core.zero : core.one;
            std::vector<CostState> next(core.right_rank);
            for (std::size_t following = 0U; following < core.right_rank; ++following) {
                for (std::size_t previous = 0U; previous < core.left_rank; ++previous) {
                    const TropicalQTTEntry& edge = matrix[previous * core.right_rank + following];
                    if (!costs[previous].supported || !edge.supported) {
                        continue;
                    }
                    const std::int64_t candidate =
                        checked_cost_sum(costs[previous].value, edge.cost);
                    if (!next[following].supported || candidate < next[following].value) {
                        next[following] = CostState{true, candidate};
                    }
                }
            }
            costs = std::move(next);
        }
        if (!costs.front().supported) {
            throw QStateError("Tropical QTT requested sector has no supported path");
        }
        return costs.front().value;
    }

    [[nodiscard]] std::int64_t energy(BasisIndex index) const {
        if (cores_.size() > std::numeric_limits<BasisIndex>::digits) {
            throw QStateError("Tropical QTT BasisIndex query cannot address more than 64 bits");
        }
        if (cores_.size() < std::numeric_limits<BasisIndex>::digits &&
            index >= (BasisIndex{1} << cores_.size())) {
            throw QStateError("Tropical QTT sector index lies outside logical range");
        }
        std::vector<std::uint8_t> bits(cores_.size());
        for (std::size_t position = 0U; position < cores_.size(); ++position) {
            const std::size_t shift = cores_.size() - 1U - position;
            bits[position] = static_cast<std::uint8_t>((index >> shift) & BasisIndex{1});
        }
        return energy_bits(bits);
    }

    [[nodiscard]] TropicalQTTMinimum global_minimum() const {
        std::size_t backpointer_entries = 0U;
        for (const TropicalQTTCore& core : cores_) {
            backpointer_entries = checked_size_sum(
                backpointer_entries,
                core.right_rank,
                "Tropical QTT backpointer count overflowed");
            if (backpointer_entries > config_.max_backpointer_entries) {
                throw QStateError("Tropical QTT minimum exceeds configured backpointer cap");
            }
        }

        std::vector<CostState> costs(1U, CostState{true, 0});
        std::vector<std::vector<Backpointer>> backpointers;
        backpointers.reserve(cores_.size());
        std::size_t candidate_evaluations = 0U;
        std::size_t maximum_active = 1U;

        for (const TropicalQTTCore& core : cores_) {
            std::vector<CostState> next(core.right_rank);
            std::vector<Backpointer> pointers(core.right_rank);
            for (std::size_t following = 0U; following < core.right_rank; ++following) {
                for (std::uint8_t bit = 0U; bit < 2U; ++bit) {
                    const std::vector<TropicalQTTEntry>& matrix =
                        bit == 0U ? core.zero : core.one;
                    for (std::size_t previous = 0U; previous < core.left_rank; ++previous) {
                        candidate_evaluations = checked_size_sum(
                            candidate_evaluations,
                            1U,
                            "Tropical QTT candidate evaluation count overflowed");
                        const TropicalQTTEntry& edge =
                            matrix[previous * core.right_rank + following];
                        if (!costs[previous].supported || !edge.supported) {
                            continue;
                        }
                        const std::int64_t candidate =
                            checked_cost_sum(costs[previous].value, edge.cost);
                        if (!next[following].supported || candidate < next[following].value ||
                            (candidate == next[following].value &&
                             earlier_pointer(bit, previous, pointers[following]))) {
                            next[following] = CostState{true, candidate};
                            pointers[following] = Backpointer{true, previous, bit};
                        }
                    }
                }
            }
            const std::size_t active = static_cast<std::size_t>(std::count_if(
                next.begin(), next.end(), [](const CostState& state) { return state.supported; }));
            if (active == 0U) {
                throw QStateError("Tropical QTT has no supported global sector");
            }
            maximum_active = std::max(maximum_active, active);
            costs = std::move(next);
            backpointers.push_back(std::move(pointers));
        }
        if (!costs.front().supported) {
            throw QStateError("Tropical QTT final boundary is unreachable");
        }

        std::vector<std::uint8_t> bits(cores_.size());
        std::size_t bond = 0U;
        for (std::size_t reverse = cores_.size(); reverse-- > 0U;) {
            const Backpointer& pointer = backpointers[reverse][bond];
            if (!pointer.valid) {
                throw QStateError("Tropical QTT minimum backpointer is missing");
            }
            bits[reverse] = pointer.bit;
            bond = pointer.previous;
        }
        if (bond != 0U) {
            throw QStateError("Tropical QTT minimum backtracking did not reach left boundary");
        }
        const std::int64_t energy_value = costs.front().value;
        if (energy_bits(bits) != energy_value) {
            throw QStateError("Tropical QTT minimum reconstruction does not reproduce optimum");
        }
        return TropicalQTTMinimum{
            std::move(bits),
            energy_value,
            candidate_evaluations,
            maximum_active,
        };
    }

private:
    struct CostState {
        bool supported{false};
        std::int64_t value{0};
    };

    struct Backpointer {
        bool valid{false};
        std::size_t previous{0U};
        std::uint8_t bit{0U};
    };

    std::vector<TropicalQTTCore> cores_{};
    TropicalQTTConfig config_{};
    TropicalQTTStats stats_{};

    explicit ExactTropicalQTT(
        std::vector<TropicalQTTCore> cores,
        TropicalQTTConfig config)
        : cores_(std::move(cores)), config_(config) {
        validate();
    }

    [[nodiscard]] static bool earlier_pointer(
        std::uint8_t bit,
        std::size_t previous,
        const Backpointer& current) noexcept {
        if (!current.valid) {
            return true;
        }
        if (bit != current.bit) {
            return bit < current.bit;
        }
        return previous < current.previous;
    }

    [[nodiscard]] static std::size_t checked_size_product(
        std::size_t left,
        std::size_t right,
        const char* message) {
        if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
            throw QStateError(message);
        }
        return left * right;
    }

    [[nodiscard]] static std::size_t checked_size_sum(
        std::size_t left,
        std::size_t right,
        const char* message) {
        if (right > std::numeric_limits<std::size_t>::max() - left) {
            throw QStateError(message);
        }
        return left + right;
    }

    [[nodiscard]] static std::int64_t checked_cost_sum(
        std::int64_t left,
        std::int64_t right) {
        if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
            (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right)) {
            throw QStateError("Tropical QTT integer cost overflowed");
        }
        return left + right;
    }

    [[nodiscard]] static std::int64_t checked_nonnegative_cost_product(
        std::int64_t left,
        std::int64_t right) {
        if (left < 0 || right < 0) {
            throw QStateError("Tropical QTT nonnegative cost product received a negative term");
        }
        if (left != 0 && right > std::numeric_limits<std::int64_t>::max() / left) {
            throw QStateError("Tropical QTT integer cost overflowed");
        }
        return left * right;
    }

    [[nodiscard]] static std::int64_t checked_size_to_cost(std::size_t value) {
        if (value > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
            throw QStateError("Tropical QTT size exceeds exact integer cost range");
        }
        return static_cast<std::int64_t>(value);
    }

    [[nodiscard]] static std::int64_t positive_mod(
        std::int64_t value,
        std::int64_t modulus) noexcept {
        const std::int64_t remainder = value % modulus;
        return remainder < 0 ? remainder + modulus : remainder;
    }

    static void validate_config(const TropicalQTTConfig& config) {
        if (config.max_rank == 0U || config.max_core_entries == 0U ||
            config.max_total_entries == 0U || config.max_backpointer_entries == 0U) {
            throw QStateError("Tropical QTT configuration contains a zero resource cap");
        }
    }

    static void preflight_rank_one(
        std::size_t logical_bits,
        const TropicalQTTConfig& config) {
        validate_config(config);
        if (config.max_rank < 1U || config.max_core_entries < 2U) {
            throw QStateError("Tropical QTT rank-one generator exceeds configured core cap");
        }
        const std::size_t total = checked_size_product(
            logical_bits, 2U, "Tropical QTT rank-one descriptor size overflowed");
        if (total > config.max_total_entries) {
            throw QStateError("Tropical QTT rank-one generator exceeds configured total entry cap");
        }
    }

    static void preflight_markov(
        std::size_t logical_bits,
        std::size_t rank,
        const TropicalQTTConfig& config) {
        validate_config(config);
        if (rank > config.max_rank) {
            throw QStateError("Tropical QTT structured generator exceeds configured rank cap");
        }
        std::size_t total = 0U;
        for (std::size_t position = 0U; position < logical_bits; ++position) {
            const std::size_t left_rank = position == 0U ? 1U : rank;
            const std::size_t right_rank = position + 1U == logical_bits ? 1U : rank;
            const std::size_t matrix = checked_size_product(
                left_rank, right_rank, "Tropical QTT structured core size overflowed");
            const std::size_t entries = checked_size_product(
                matrix, 2U, "Tropical QTT structured core entry count overflowed");
            if (entries > config.max_core_entries) {
                throw QStateError("Tropical QTT structured core exceeds configured entry cap");
            }
            total = checked_size_sum(
                total, entries, "Tropical QTT structured total entry count overflowed");
            if (total > config.max_total_entries) {
                throw QStateError("Tropical QTT structured generator exceeds configured total entry cap");
            }
        }
    }

    void validate() {
        validate_config(config_);
        if (cores_.empty()) {
            throw QStateError("Tropical QTT requires at least one binary core");
        }
        if (cores_.front().left_rank != 1U) {
            throw QStateError("Tropical QTT first left rank must be one");
        }

        std::size_t previous_right = 0U;
        std::size_t maximum_rank = 1U;
        std::size_t descriptor_entries = 0U;
        std::size_t supported_entries = 0U;
        std::vector<bool> active(1U, true);

        for (std::size_t index = 0U; index < cores_.size(); ++index) {
            const TropicalQTTCore& core = cores_[index];
            if (core.left_rank == 0U || core.right_rank == 0U) {
                throw QStateError("Tropical QTT ranks must be positive");
            }
            if (index != 0U && core.left_rank != previous_right) {
                throw QStateError("Tropical QTT adjacent bond ranks do not match");
            }
            if (active.size() != core.left_rank) {
                throw QStateError("Tropical QTT reachability rank mismatch");
            }
            if (core.left_rank > config_.max_rank || core.right_rank > config_.max_rank) {
                throw QStateError("Tropical QTT core exceeds configured rank cap");
            }
            const std::size_t matrix = checked_size_product(
                core.left_rank, core.right_rank, "Tropical QTT core shape overflowed");
            if (core.zero.size() != matrix || core.one.size() != matrix) {
                throw QStateError("Tropical QTT physical slices do not match declared ranks");
            }
            const std::size_t entries = checked_size_product(
                matrix, 2U, "Tropical QTT core entry count overflowed");
            if (entries > config_.max_core_entries) {
                throw QStateError("Tropical QTT core exceeds configured entry cap");
            }
            descriptor_entries = checked_size_sum(
                descriptor_entries,
                entries,
                "Tropical QTT total entry count overflowed");
            if (descriptor_entries > config_.max_total_entries) {
                throw QStateError("Tropical QTT exceeds configured total entry cap");
            }

            std::vector<bool> next_active(core.right_rank, false);
            for (std::uint8_t bit = 0U; bit < 2U; ++bit) {
                const std::vector<TropicalQTTEntry>& matrix_values =
                    bit == 0U ? core.zero : core.one;
                for (std::size_t previous = 0U; previous < core.left_rank; ++previous) {
                    for (std::size_t following = 0U; following < core.right_rank; ++following) {
                        const TropicalQTTEntry& entry =
                            matrix_values[previous * core.right_rank + following];
                        if (entry.supported) {
                            supported_entries = checked_size_sum(
                                supported_entries,
                                1U,
                                "Tropical QTT supported entry count overflowed");
                            if (active[previous]) {
                                next_active[following] = true;
                            }
                        }
                    }
                }
            }
            if (std::none_of(next_active.begin(), next_active.end(), [](bool value) { return value; })) {
                throw QStateError("Tropical QTT has no complete supported sector path");
            }
            active = std::move(next_active);
            maximum_rank = std::max(maximum_rank, std::max(core.left_rank, core.right_rank));
            previous_right = core.right_rank;
        }
        if (previous_right != 1U || active.size() != 1U || !active.front()) {
            throw QStateError("Tropical QTT final boundary is unreachable");
        }
        stats_ = TropicalQTTStats{
            cores_.size(), maximum_rank, descriptor_entries, supported_entries,
        };
    }
};

}  // namespace qubit

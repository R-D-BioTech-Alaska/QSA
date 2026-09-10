#pragma once

#include "qubit/qstate.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <vector>

namespace qubit {

struct FixedExcitationConfig {
    std::size_t max_modes{1U << 20U};
    std::size_t max_excitation_count{64U};
    std::size_t max_sector_amplitudes{1U << 24U};
    std::size_t max_occupied_entries{1U << 27U};
};

struct FixedExcitationStats {
    std::size_t modes{0U};
    std::size_t excitations{0U};
    std::size_t sector_dimension{0U};
    std::size_t occupied_entries{0U};
    std::size_t scalar_operations{0U};
    std::size_t estimated_bytes{0U};
};

class ExactFixedExcitationState {
public:
    [[nodiscard]] static ExactFixedExcitationState basis(
        std::size_t mode_count,
        std::span<const QubitId> occupied_modes,
        FixedExcitationConfig config = {}) {
        ExactFixedExcitationState state(mode_count, occupied_modes.size(), config);
        std::vector<QubitId> occupied(occupied_modes.begin(), occupied_modes.end());
        std::sort(occupied.begin(), occupied.end());
        state.validate_combination(occupied);
        state.amplitudes_[state.rank_of(occupied)] = QComplex{1.0};
        return state;
    }

    [[nodiscard]] std::size_t mode_count() const noexcept { return mode_count_; }
    [[nodiscard]] std::size_t excitation_count() const noexcept { return excitation_count_; }
    [[nodiscard]] std::size_t sector_dimension() const noexcept { return amplitudes_.size(); }
    [[nodiscard]] const FixedExcitationStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const FixedExcitationConfig& config() const noexcept { return config_; }

    [[nodiscard]] double norm_squared() const noexcept {
        double total = 0.0;
        for (const QComplex amplitude : amplitudes_) {
            total += amplitude.norm2();
        }
        return total;
    }

    [[nodiscard]] QComplex amplitude(std::span<const QubitId> occupied_modes) const {
        std::vector<QubitId> occupied(occupied_modes.begin(), occupied_modes.end());
        std::sort(occupied.begin(), occupied.end());
        validate_combination(occupied);
        return amplitudes_[rank_of(occupied)];
    }

    [[nodiscard]] double sector_log2_dimension() const noexcept {
        return std::log2(static_cast<double>(amplitudes_.size()));
    }

    [[nodiscard]] double dense_to_sector_log2_ratio() const noexcept {
        return static_cast<double>(mode_count_) - sector_log2_dimension();
    }

    void apply_mode_phase(std::size_t mode, double angle) {
        validate_mode(mode);
        if (!std::isfinite(angle)) {
            throw QStateError("Fixed-excitation phase angle must be finite");
        }
        const QComplex phase = QComplex::from_polar(1.0, angle);
        for (std::size_t basis = 0U; basis < amplitudes_.size(); ++basis) {
            if (contains(combination(basis), static_cast<QubitId>(mode))) {
                amplitudes_[basis] *= phase;
            }
        }
        record_work();
    }

    void apply_givens(std::size_t left, std::size_t right, double angle, double phase_angle = 0.0) {
        validate_mode(left);
        validate_mode(right);
        if (left == right) {
            throw QStateError("Fixed-excitation Givens rotation requires distinct modes");
        }
        if (!std::isfinite(angle) || !std::isfinite(phase_angle)) {
            throw QStateError("Fixed-excitation Givens parameters must be finite");
        }

        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        const QComplex phase = QComplex::from_polar(1.0, phase_angle);
        const QComplex phase_conjugate = phase.conjugate();
        const QubitId left_id = static_cast<QubitId>(left);
        const QubitId right_id = static_cast<QubitId>(right);
        std::vector<QubitId> partner(excitation_count_);

        for (std::size_t basis = 0U; basis < amplitudes_.size(); ++basis) {
            const std::span<const QubitId> occupied = combination(basis);
            if (!contains(occupied, left_id) || contains(occupied, right_id)) {
                continue;
            }
            std::copy(occupied.begin(), occupied.end(), partner.begin());
            const auto found = std::lower_bound(partner.begin(), partner.end(), left_id);
            if (found == partner.end() || *found != left_id) {
                throw QStateError("Fixed-excitation Givens support bookkeeping failed");
            }
            *found = right_id;
            std::sort(partner.begin(), partner.end());
            const std::size_t partner_basis = rank_of(partner);
            const QComplex a = amplitudes_[basis];
            const QComplex b = amplitudes_[partner_basis];
            amplitudes_[basis] = a * cosine - phase_conjugate * b * sine;
            amplitudes_[partner_basis] = phase * a * sine + b * cosine;
        }
        record_work();
    }

    [[nodiscard]] double occupation_probability(std::size_t mode) const {
        validate_mode(mode);
        const QubitId id = static_cast<QubitId>(mode);
        double total = 0.0;
        for (std::size_t basis = 0U; basis < amplitudes_.size(); ++basis) {
            if (contains(combination(basis), id)) {
                total += amplitudes_[basis].norm2();
            }
        }
        return total;
    }

    [[nodiscard]] double marginal_probability(
        std::span<const QubitId> modes,
        std::span<const std::uint8_t> bits) const {
        if (modes.size() != bits.size()) {
            throw QStateError("Fixed-excitation marginal mode/bit sizes differ");
        }
        std::vector<QubitId> sorted(modes.begin(), modes.end());
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            throw QStateError("Fixed-excitation marginal contains duplicate modes");
        }
        for (std::size_t index = 0U; index < modes.size(); ++index) {
            validate_mode(static_cast<std::size_t>(modes[index]));
            if (bits[index] > 1U) {
                throw QStateError("Fixed-excitation marginal bit must be zero or one");
            }
        }

        double total = 0.0;
        for (std::size_t basis = 0U; basis < amplitudes_.size(); ++basis) {
            const std::span<const QubitId> occupied = combination(basis);
            bool match = true;
            for (std::size_t query = 0U; query < modes.size(); ++query) {
                const bool present = contains(occupied, modes[query]);
                if (present != (bits[query] != 0U)) {
                    match = false;
                    break;
                }
            }
            if (match) {
                total += amplitudes_[basis].norm2();
            }
        }
        return total;
    }

    void apply_x(std::size_t mode) {
        validate_mode(mode);
        throw QStateError("X changes excitation number and leaves the certified fixed-excitation sector");
    }

private:
    std::size_t mode_count_{0U};
    std::size_t excitation_count_{0U};
    FixedExcitationConfig config_{};
    std::vector<QubitId> occupied_flat_{};
    std::vector<QComplex> amplitudes_{};
    FixedExcitationStats stats_{};

    ExactFixedExcitationState(
        std::size_t mode_count,
        std::size_t excitation_count,
        FixedExcitationConfig config)
        : mode_count_(mode_count), excitation_count_(excitation_count), config_(config) {
        if (mode_count_ == 0U || mode_count_ > config_.max_modes ||
            mode_count_ > static_cast<std::size_t>(std::numeric_limits<QubitId>::max()) ||
            excitation_count_ > mode_count_ ||
            excitation_count_ > config_.max_excitation_count ||
            config_.max_sector_amplitudes == 0U || config_.max_occupied_entries == 0U) {
            throw QStateError("Fixed-excitation dimensions or configuration are invalid");
        }
        if (excitation_count_ > mode_count_ / 2U && excitation_count_ != 0U) {
            throw QStateError(
                "Fixed-excitation direct representation is limited to the low-excitation side of half filling");
        }

        const std::size_t dimension = choose_bounded(
            mode_count_, excitation_count_, config_.max_sector_amplitudes);
        if (dimension == 0U) {
            throw QStateError("Fixed-excitation sector dimension exceeds configured amplitude cap");
        }
        const std::size_t occupied_entries = checked_product(
            dimension, excitation_count_, "Fixed-excitation occupied-entry count overflowed");
        if (occupied_entries > config_.max_occupied_entries) {
            throw QStateError("Fixed-excitation occupied-entry count exceeds configured cap");
        }

        amplitudes_.assign(dimension, QComplex{});
        occupied_flat_.reserve(occupied_entries);
        generate_combinations(dimension);
        if (occupied_flat_.size() != occupied_entries) {
            throw QStateError("Fixed-excitation combination generation failed");
        }
        stats_ = FixedExcitationStats{
            mode_count_,
            excitation_count_,
            dimension,
            occupied_entries,
            0U,
            sizeof(*this) + occupied_flat_.capacity() * sizeof(QubitId) +
                amplitudes_.capacity() * sizeof(QComplex),
        };
    }

    void generate_combinations(std::size_t dimension) {
        if (excitation_count_ == 0U) {
            return;
        }
        std::vector<QubitId> current(excitation_count_);
        for (std::size_t index = 0U; index < excitation_count_; ++index) {
            current[index] = static_cast<QubitId>(index);
        }

        for (std::size_t emitted = 0U; emitted < dimension; ++emitted) {
            occupied_flat_.insert(occupied_flat_.end(), current.begin(), current.end());
            if (emitted + 1U == dimension) {
                break;
            }
            std::size_t pivot = excitation_count_;
            while (pivot != 0U) {
                const std::size_t position = pivot - 1U;
                const std::size_t maximum = mode_count_ - excitation_count_ + position;
                if (static_cast<std::size_t>(current[position]) < maximum) {
                    break;
                }
                --pivot;
            }
            if (pivot == 0U) {
                throw QStateError("Fixed-excitation combination enumeration ended early");
            }
            ++current[pivot - 1U];
            for (std::size_t position = pivot; position < excitation_count_; ++position) {
                current[position] = static_cast<QubitId>(
                    static_cast<std::size_t>(current[position - 1U]) + 1U);
            }
        }
    }

    [[nodiscard]] std::span<const QubitId> combination(std::size_t basis) const {
        if (basis >= amplitudes_.size()) {
            throw QStateError("Fixed-excitation basis index is out of range");
        }
        if (excitation_count_ == 0U) {
            return {};
        }
        return std::span<const QubitId>(
            occupied_flat_.data() + basis * excitation_count_, excitation_count_);
    }

    [[nodiscard]] std::size_t rank_of(std::span<const QubitId> occupied) const {
        validate_combination(occupied);
        if (excitation_count_ == 0U) {
            return 0U;
        }
        std::size_t rank = 0U;
        std::size_t first = 0U;
        for (std::size_t position = 0U; position < excitation_count_; ++position) {
            const std::size_t current = static_cast<std::size_t>(occupied[position]);
            const std::size_t remaining = excitation_count_ - position;
            if (current > first) {
                const std::size_t high = choose_exact(mode_count_ - first, remaining);
                const std::size_t low = choose_exact(mode_count_ - current, remaining);
                if (high < low || rank > amplitudes_.size() - 1U - (high - low)) {
                    throw QStateError("Fixed-excitation combinatorial rank overflowed");
                }
                rank += high - low;
            }
            first = current + 1U;
        }
        if (rank >= amplitudes_.size()) {
            throw QStateError("Fixed-excitation combinatorial rank is out of range");
        }
        return rank;
    }

    void validate_combination(std::span<const QubitId> occupied) const {
        if (occupied.size() != excitation_count_) {
            throw QStateError("Fixed-excitation occupied-mode count does not match sector");
        }
        QubitId previous = 0U;
        for (std::size_t index = 0U; index < occupied.size(); ++index) {
            if (static_cast<std::size_t>(occupied[index]) >= mode_count_) {
                throw QStateError("Fixed-excitation occupied mode is out of range");
            }
            if (index != 0U && occupied[index] <= previous) {
                throw QStateError("Fixed-excitation occupied modes must be unique");
            }
            previous = occupied[index];
        }
    }

    void validate_mode(std::size_t mode) const {
        if (mode >= mode_count_) {
            throw QStateError("Fixed-excitation mode is out of range");
        }
    }

    void record_work() noexcept {
        if (stats_.scalar_operations <=
            std::numeric_limits<std::size_t>::max() - amplitudes_.size()) {
            stats_.scalar_operations += amplitudes_.size();
        } else {
            stats_.scalar_operations = std::numeric_limits<std::size_t>::max();
        }
    }

    [[nodiscard]] static bool contains(std::span<const QubitId> occupied, QubitId mode) noexcept {
        return std::binary_search(occupied.begin(), occupied.end(), mode);
    }

    [[nodiscard]] std::size_t choose_exact(std::size_t n, std::size_t k) const {
        const std::size_t value = choose_bounded(n, k, amplitudes_.size());
        if (value == 0U) {
            throw QStateError("Fixed-excitation combinatorial coefficient exceeded sector dimension");
        }
        return value;
    }

    [[nodiscard]] static std::size_t choose_bounded(
        std::size_t n, std::size_t k, std::size_t cap) {
        if (k > n) {
            return 0U;
        }
        k = std::min(k, n - k);
        std::size_t value = 1U;
        for (std::size_t index = 1U; index <= k; ++index) {
            std::size_t numerator = n - k + index;
            std::size_t denominator = index;
            const std::size_t common = std::gcd(numerator, denominator);
            numerator /= common;
            denominator /= common;
            const std::size_t from_value = std::gcd(value, denominator);
            value /= from_value;
            denominator /= from_value;
            if (denominator != 1U) {
                throw QStateError("Fixed-excitation binomial reduction failed");
            }
            if (numerator != 0U && value > cap / numerator) {
                return 0U;
            }
            value *= numerator;
        }
        return value;
    }

    [[nodiscard]] static std::size_t checked_product(
        std::size_t left, std::size_t right, const char* message) {
        if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
            throw QStateError(message);
        }
        return left * right;
    }
};

}  // namespace qubit

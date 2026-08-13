#pragma once

#include "qubit/qstate.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <vector>

namespace qubit {

struct EncodedRepetitionConfig {
    std::size_t max_logical_qubits{24U};
    std::size_t max_logical_amplitudes{1U << 24U};
    std::size_t max_physical_qubits{1U << 30U};
};

struct EncodedRepetitionStats {
    std::size_t logical_qubits{0U};
    std::size_t block_size{0U};
    std::size_t physical_qubits{0U};
    std::size_t logical_amplitudes{0U};
    std::size_t scalar_operations{0U};
    std::size_t estimated_bytes{0U};
};

class EncodedRepetitionState {
public:
    [[nodiscard]] static EncodedRepetitionState zero(
        std::size_t logical_qubits,
        std::size_t block_size,
        EncodedRepetitionConfig config = {}) {
        EncodedRepetitionState state(logical_qubits, block_size, config);
        state.amplitudes_[0] = QComplex{1.0};
        return state;
    }

    [[nodiscard]] static EncodedRepetitionState plus(
        std::size_t logical_qubits,
        std::size_t block_size,
        EncodedRepetitionConfig config = {}) {
        EncodedRepetitionState state(logical_qubits, block_size, config);
        const double scale = 1.0 / std::sqrt(static_cast<double>(state.amplitudes_.size()));
        for (QComplex& amplitude : state.amplitudes_) {
            amplitude = QComplex{scale};
        }
        return state;
    }

    [[nodiscard]] std::size_t logical_qubit_count() const noexcept { return logical_qubits_; }
    [[nodiscard]] std::size_t block_size() const noexcept { return block_size_; }
    [[nodiscard]] std::size_t physical_qubit_count() const noexcept { return physical_qubits_; }
    [[nodiscard]] std::size_t logical_dimension() const noexcept { return amplitudes_.size(); }
    [[nodiscard]] const std::vector<QComplex>& amplitudes() const noexcept { return amplitudes_; }
    [[nodiscard]] const EncodedRepetitionStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const EncodedRepetitionConfig& config() const noexcept { return config_; }

    [[nodiscard]] double norm_squared() const noexcept {
        double total = 0.0;
        for (const QComplex amplitude : amplitudes_) {
            total += amplitude.norm2();
        }
        return total;
    }

    [[nodiscard]] QComplex amplitude(std::size_t logical_basis) const {
        validate_logical_basis(logical_basis);
        return amplitudes_[logical_basis];
    }

    [[nodiscard]] double logical_basis_probability(std::size_t logical_basis) const {
        return amplitude(logical_basis).norm2();
    }

    void logical_x(std::size_t logical) {
        const std::size_t bit = logical_bit(logical);
        for (std::size_t index = 0U; index < amplitudes_.size(); ++index) {
            if ((index & bit) == 0U) {
                const std::size_t partner = index | bit;
                std::swap(amplitudes_[index], amplitudes_[partner]);
            }
        }
        record_work();
    }

    void logical_z(std::size_t logical) {
        const std::size_t bit = logical_bit(logical);
        for (std::size_t index = 0U; index < amplitudes_.size(); ++index) {
            if ((index & bit) != 0U) {
                amplitudes_[index] = -amplitudes_[index];
            }
        }
        record_work();
    }

    void logical_ry(std::size_t logical, double angle) {
        require_finite(angle, "Encoded logical Ry angle must be finite");
        const std::size_t bit = logical_bit(logical);
        const double cosine = std::cos(0.5 * angle);
        const double sine = std::sin(0.5 * angle);
        for (std::size_t index = 0U; index < amplitudes_.size(); ++index) {
            if ((index & bit) != 0U) {
                continue;
            }
            const std::size_t partner = index | bit;
            const QComplex zero = amplitudes_[index];
            const QComplex one = amplitudes_[partner];
            amplitudes_[index] = zero * cosine - one * sine;
            amplitudes_[partner] = zero * sine + one * cosine;
        }
        record_work();
    }

    void logical_rz(std::size_t logical, double angle) {
        require_finite(angle, "Encoded logical Rz angle must be finite");
        const std::size_t bit = logical_bit(logical);
        const QComplex zero_phase = QComplex::from_polar(1.0, -0.5 * angle);
        const QComplex one_phase = QComplex::from_polar(1.0, 0.5 * angle);
        for (std::size_t index = 0U; index < amplitudes_.size(); ++index) {
            amplitudes_[index] *= (index & bit) == 0U ? zero_phase : one_phase;
        }
        record_work();
    }

    void logical_cnot(std::size_t control, std::size_t target) {
        const std::size_t control_bit = logical_bit(control);
        const std::size_t target_bit = logical_bit(target);
        if (control == target) {
            throw QStateError("Encoded logical CNOT requires distinct logical qubits");
        }
        for (std::size_t index = 0U; index < amplitudes_.size(); ++index) {
            if ((index & control_bit) != 0U && (index & target_bit) == 0U) {
                const std::size_t partner = index | target_bit;
                std::swap(amplitudes_[index], amplitudes_[partner]);
            }
        }
        record_work();
    }

    void physical_z(std::size_t physical_qubit) {
        logical_z(logical_for_physical(physical_qubit));
    }

    void physical_rz(std::size_t physical_qubit, double angle) {
        logical_rz(logical_for_physical(physical_qubit), angle);
    }

    void physical_x(std::size_t physical_qubit) {
        const std::size_t logical = logical_for_physical(physical_qubit);
        if (block_size_ != 1U) {
            throw QStateError(
                "Single physical X leaves the certified repetition-code subspace");
        }
        logical_x(logical);
    }

    [[nodiscard]] double logical_z_expectation(std::size_t logical) const {
        const std::size_t bit = logical_bit(logical);
        double total = 0.0;
        for (std::size_t index = 0U; index < amplitudes_.size(); ++index) {
            total += ((index & bit) == 0U ? 1.0 : -1.0) * amplitudes_[index].norm2();
        }
        return total;
    }

    [[nodiscard]] double logical_x_expectation(std::size_t logical) const {
        const std::size_t bit = logical_bit(logical);
        QComplex total{};
        for (std::size_t index = 0U; index < amplitudes_.size(); ++index) {
            total += amplitudes_[index ^ bit].conjugate() * amplitudes_[index];
        }
        const double scale = 1.0 + std::abs(total.re);
        if (!finite(total) || std::abs(total.im) > 1e-10 * scale) {
            throw QStateError("Encoded logical X expectation became complex/non-finite");
        }
        return total.re;
    }

    [[nodiscard]] double physical_z_expectation(std::size_t physical_qubit) const {
        return logical_z_expectation(logical_for_physical(physical_qubit));
    }

    [[nodiscard]] double physical_marginal_probability(
        std::span<const QubitId> physical_qubits,
        std::span<const std::uint8_t> bits) const {
        if (physical_qubits.size() != bits.size()) {
            throw QStateError("Encoded physical marginal qubit/bit sizes differ");
        }
        std::vector<std::int8_t> constraints(logical_qubits_, -1);
        std::vector<QubitId> sorted(physical_qubits.begin(), physical_qubits.end());
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            throw QStateError("Encoded physical marginal contains duplicate physical qubits");
        }
        for (std::size_t index = 0U; index < physical_qubits.size(); ++index) {
            const std::size_t physical = static_cast<std::size_t>(physical_qubits[index]);
            if (physical >= physical_qubits_) {
                throw QStateError("Encoded physical marginal qubit is out of range");
            }
            if (bits[index] > 1U) {
                throw QStateError("Encoded physical marginal bit must be zero or one");
            }
            const std::size_t logical = physical / block_size_;
            const std::int8_t value = static_cast<std::int8_t>(bits[index]);
            if (constraints[logical] != -1 && constraints[logical] != value) {
                return 0.0;
            }
            constraints[logical] = value;
        }

        double probability = 0.0;
        for (std::size_t basis = 0U; basis < amplitudes_.size(); ++basis) {
            bool match = true;
            for (std::size_t logical = 0U; logical < logical_qubits_; ++logical) {
                if (constraints[logical] == -1) {
                    continue;
                }
                const std::int8_t value = static_cast<std::int8_t>((basis >> logical) & 1U);
                if (value != constraints[logical]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                probability += amplitudes_[basis].norm2();
            }
        }
        return probability;
    }

private:
    std::size_t logical_qubits_{0U};
    std::size_t block_size_{0U};
    std::size_t physical_qubits_{0U};
    EncodedRepetitionConfig config_{};
    std::vector<QComplex> amplitudes_{};
    EncodedRepetitionStats stats_{};

    EncodedRepetitionState(
        std::size_t logical_qubits,
        std::size_t block_size,
        EncodedRepetitionConfig config)
        : logical_qubits_(logical_qubits), block_size_(block_size), config_(config) {
        if (logical_qubits_ == 0U || block_size_ == 0U ||
            config_.max_logical_qubits == 0U || config_.max_logical_amplitudes == 0U ||
            config_.max_physical_qubits == 0U ||
            logical_qubits_ > config_.max_logical_qubits ||
            logical_qubits_ >= std::numeric_limits<std::size_t>::digits) {
            throw QStateError("Encoded repetition dimensions or configuration are invalid");
        }
        const std::size_t dimension = std::size_t{1U} << logical_qubits_;
        if (dimension > config_.max_logical_amplitudes) {
            throw QStateError("Encoded repetition logical state exceeds amplitude cap");
        }
        physical_qubits_ = checked_product(
            logical_qubits_, block_size_, "Encoded repetition physical qubit count overflowed");
        if (physical_qubits_ > config_.max_physical_qubits ||
            physical_qubits_ > static_cast<std::size_t>(std::numeric_limits<QubitId>::max())) {
            throw QStateError("Encoded repetition physical qubit count exceeds configured/index cap");
        }
        amplitudes_.assign(dimension, QComplex{});
        stats_ = EncodedRepetitionStats{
            logical_qubits_,
            block_size_,
            physical_qubits_,
            dimension,
            0U,
            sizeof(*this) + amplitudes_.capacity() * sizeof(QComplex),
        };
    }

    [[nodiscard]] std::size_t logical_bit(std::size_t logical) const {
        if (logical >= logical_qubits_) {
            throw QStateError("Encoded logical qubit is out of range");
        }
        return std::size_t{1U} << logical;
    }

    [[nodiscard]] std::size_t logical_for_physical(std::size_t physical) const {
        if (physical >= physical_qubits_) {
            throw QStateError("Encoded physical qubit is out of range");
        }
        return physical / block_size_;
    }

    void validate_logical_basis(std::size_t logical_basis) const {
        if (logical_basis >= amplitudes_.size()) {
            throw QStateError("Encoded logical basis index is out of range");
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

    [[nodiscard]] static bool finite(QComplex value) noexcept {
        return std::isfinite(value.re) && std::isfinite(value.im);
    }

    static void require_finite(double value, const char* message) {
        if (!std::isfinite(value)) {
            throw QStateError(message);
        }
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

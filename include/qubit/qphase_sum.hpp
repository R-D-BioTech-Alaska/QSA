#pragma once

#include "qubit/qphase_graph.hpp"
#include "qubit/qplan.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace qubit {

struct PhaseGraphBranchSumConfig {
    std::size_t max_branches{4096U};
    std::size_t max_retained_estimated_bytes{1U << 30U};
    std::size_t max_materialize_qubits{24U};
    PhaseGraphConfig phase_graph{};
};

struct PhaseGraphBranchSumStats {
    std::size_t qubits{0U};
    std::size_t branches{0U};
    std::size_t hadamard_defects{0U};
    std::size_t max_live_branches{0U};
    std::size_t total_phase_edges{0U};
    std::size_t retained_estimated_bytes{0U};
};

struct ScaledPhaseGraphAmplitude {
    QComplex mantissa{};
    double log2_scale{0.0};
};

struct PhaseGraphBranch {
    QComplex coefficient{};
    PhaseGraphState state;
};

class ExactPhaseGraphBranchSum {
public:
    explicit ExactPhaseGraphBranchSum(
        std::size_t qubit_count,
        PhaseGraphBranchSumConfig config = {})
        : qubit_count_(qubit_count), config_(config) {
        validate_configuration();
        branches_.push_back(PhaseGraphBranch{
            QComplex{1.0}, PhaseGraphState(qubit_count_, config_.phase_graph),
        });
        refresh_stats();
        enforce_retained_cap(stats_.retained_estimated_bytes);
    }

    explicit ExactPhaseGraphBranchSum(
        PhaseGraphState state,
        PhaseGraphBranchSumConfig config = {})
        : qubit_count_(state.qubit_count()), config_(config) {
        validate_configuration();
        branches_.push_back(PhaseGraphBranch{QComplex{1.0}, std::move(state)});
        refresh_stats();
        enforce_retained_cap(stats_.retained_estimated_bytes);
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::size_t branch_count() const noexcept { return branches_.size(); }
    [[nodiscard]] const std::vector<PhaseGraphBranch>& branches() const noexcept {
        return branches_;
    }
    [[nodiscard]] const PhaseGraphBranchSumStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const PhaseGraphBranchSumConfig& config() const noexcept { return config_; }

    void apply_x(QubitId qubit) {
        mutate([qubit](PhaseGraphState& state) { state.apply_x(qubit); });
    }

    void apply_y(QubitId qubit) {
        mutate([qubit](PhaseGraphState& state) { state.apply_y(qubit); });
    }

    void apply_z(QubitId qubit) {
        mutate([qubit](PhaseGraphState& state) { state.apply_z(qubit); });
    }

    void apply_s(QubitId qubit) {
        mutate([qubit](PhaseGraphState& state) { state.apply_s(qubit); });
    }

    void apply_sdg(QubitId qubit) {
        mutate([qubit](PhaseGraphState& state) { state.apply_sdg(qubit); });
    }

    void apply_t(QubitId qubit) {
        mutate([qubit](PhaseGraphState& state) { state.apply_t(qubit); });
    }

    void apply_tdg(QubitId qubit) {
        mutate([qubit](PhaseGraphState& state) { state.apply_tdg(qubit); });
    }

    void apply_rz(QubitId qubit, double angle) {
        mutate([qubit, angle](PhaseGraphState& state) { state.apply_rz(qubit, angle); });
    }

    void apply_cz(QubitId first, QubitId second) {
        mutate([first, second](PhaseGraphState& state) { state.apply_cz(first, second); });
    }

    void apply_controlled_phase(QubitId first, QubitId second, double angle) {
        mutate([first, second, angle](PhaseGraphState& state) {
            state.apply_controlled_phase(first, second, angle);
        });
    }

    void apply_swap(QubitId first, QubitId second) {
        mutate([first, second](PhaseGraphState& state) { state.apply_swap(first, second); });
    }

    void apply_h(QubitId qubit) {
        if (branches_.size() > config_.max_branches / 2U) {
            throw QStateError("Phase-graph branch-sum Hadamard exceeds configured branch cap");
        }
        const std::size_t next_count = checked_product(
            branches_.size(), 2U, "Phase-graph branch-sum branch count overflowed");
        if (next_count > config_.max_branches) {
            throw QStateError("Phase-graph branch-sum Hadamard exceeds configured branch cap");
        }

        constexpr double inverse_sqrt_two = 0.707106781186547524400844362104849039;
        std::vector<PhaseGraphBranch> next;
        next.reserve(next_count);
        for (const PhaseGraphBranch& branch : branches_) {
            PhaseGraphState x_state = branch.state;
            PhaseGraphState z_state = branch.state;
            x_state.apply_x(qubit);
            z_state.apply_z(qubit);
            const QComplex coefficient = branch.coefficient * inverse_sqrt_two;
            next.push_back(PhaseGraphBranch{coefficient, std::move(x_state)});
            next.push_back(PhaseGraphBranch{coefficient, std::move(z_state)});
        }
        const std::size_t retained = estimated_bytes(next);
        enforce_retained_cap(retained);
        branches_.swap(next);
        ++hadamard_defects_;
        max_live_branches_ = std::max(max_live_branches_, branches_.size());
        refresh_stats(retained);
    }

    void apply(const Operation& operation) {
        switch (operation.code) {
            case OperationCode::X:
                apply_x(operation.first);
                return;
            case OperationCode::Y:
                apply_y(operation.first);
                return;
            case OperationCode::Z:
                apply_z(operation.first);
                return;
            case OperationCode::H:
                apply_h(operation.first);
                return;
            case OperationCode::S:
                apply_s(operation.first);
                return;
            case OperationCode::Sdg:
                apply_sdg(operation.first);
                return;
            case OperationCode::T:
                apply_t(operation.first);
                return;
            case OperationCode::Tdg:
                apply_tdg(operation.first);
                return;
            case OperationCode::Rz:
                apply_rz(operation.first, operation.parameter);
                return;
            case OperationCode::Cz:
                apply_cz(operation.first, operation.second);
                return;
            case OperationCode::Swap:
                apply_swap(operation.first, operation.second);
                return;
            case OperationCode::Rx:
            case OperationCode::Ry:
            case OperationCode::Cnot:
            case OperationCode::BitFlipTrajectory:
            case OperationCode::PhaseFlipTrajectory:
            case OperationCode::DepolarizingTrajectory:
            case OperationCode::AmplitudeDampingTrajectory:
                throw QStateError(
                    "Operation is outside the exact low-H phase-graph branch-sum contract");
            default:
                throw QStateError("Phase-graph branch-sum received an unknown opcode");
        }
    }

    [[nodiscard]] ScaledPhaseGraphAmplitude scaled_amplitude_bits(
        std::span<const std::uint8_t> bits) const {
        if (bits.size() != qubit_count_) {
            throw QStateError("Phase-graph branch-sum bit-vector length does not match qubit count");
        }
        QComplex mantissa{};
        for (const PhaseGraphBranch& branch : branches_) {
            mantissa += branch.coefficient * branch.state.unit_phase_bits(bits);
        }
        if (!finite(mantissa)) {
            throw QStateError("Phase-graph branch-sum scaled amplitude became non-finite");
        }
        return ScaledPhaseGraphAmplitude{
            mantissa, -0.5 * static_cast<double>(qubit_count_),
        };
    }

    [[nodiscard]] double log2_probability_bits(
        std::span<const std::uint8_t> bits) const {
        const ScaledPhaseGraphAmplitude scaled = scaled_amplitude_bits(bits);
        const double norm2 = scaled.mantissa.norm2();
        if (!std::isfinite(norm2)) {
            throw QStateError("Phase-graph branch-sum probability became non-finite");
        }
        if (norm2 == 0.0) {
            return -std::numeric_limits<double>::infinity();
        }
        return std::log2(norm2) + 2.0 * scaled.log2_scale;
    }

    [[nodiscard]] QComplex amplitude_bits(
        std::span<const std::uint8_t> bits) const {
        const ScaledPhaseGraphAmplitude scaled = scaled_amplitude_bits(bits);
        const double scale = std::exp2(scaled.log2_scale);
        if (scale == 0.0) {
            throw QStateError(
                "Phase-graph branch-sum amplitude underflows; use scaled_amplitude_bits or log2_probability_bits");
        }
        return scaled.mantissa * scale;
    }

    [[nodiscard]] QComplex amplitude(BasisIndex basis) const {
        if (qubit_count_ >= 63U) {
            throw QStateError(
                "Integer phase-graph branch-sum amplitude supports at most 62 qubits; use bit-vector queries");
        }
        const BasisIndex dimension = BasisIndex{1} << qubit_count_;
        if (basis >= dimension) {
            throw QStateError("Phase-graph branch-sum basis index is out of range");
        }
        std::vector<std::uint8_t> bits(qubit_count_);
        for (std::size_t qubit = 0U; qubit < qubit_count_; ++qubit) {
            bits[qubit] = static_cast<std::uint8_t>((basis >> qubit) & 1U);
        }
        return amplitude_bits(bits);
    }

    [[nodiscard]] std::vector<QComplex> materialize(
        std::size_t max_qubits = 24U) const {
        const std::size_t limit = std::min(max_qubits, config_.max_materialize_qubits);
        if (qubit_count_ > limit || qubit_count_ >= 63U) {
            throw QStateError("Phase-graph branch-sum materialization exceeds configured qubit cap");
        }
        const BasisIndex dimension = BasisIndex{1} << qubit_count_;
        std::vector<QComplex> output(static_cast<std::size_t>(dimension));
        for (BasisIndex basis = 0U; basis < dimension; ++basis) {
            output[static_cast<std::size_t>(basis)] = amplitude(basis);
        }
        return output;
    }

private:
    std::size_t qubit_count_{0U};
    PhaseGraphBranchSumConfig config_{};
    std::vector<PhaseGraphBranch> branches_{};
    std::size_t hadamard_defects_{0U};
    std::size_t max_live_branches_{1U};
    PhaseGraphBranchSumStats stats_{};

    void validate_configuration() const {
        if (qubit_count_ == 0U ||
            qubit_count_ > static_cast<std::size_t>(std::numeric_limits<QubitId>::max()) ||
            config_.max_branches == 0U ||
            config_.max_retained_estimated_bytes == 0U ||
            config_.max_materialize_qubits == 0U) {
            throw QStateError("Phase-graph branch-sum dimensions or configuration are invalid");
        }
    }

    template <typename Mutation>
    void mutate(Mutation&& mutation) {
        std::vector<PhaseGraphBranch> next = branches_;
        for (PhaseGraphBranch& branch : next) {
            mutation(branch.state);
        }
        const std::size_t retained = estimated_bytes(next);
        enforce_retained_cap(retained);
        branches_.swap(next);
        refresh_stats(retained);
    }

    void refresh_stats() {
        refresh_stats(estimated_bytes(branches_));
    }

    void refresh_stats(std::size_t retained) {
        std::size_t edges = 0U;
        for (const PhaseGraphBranch& branch : branches_) {
            edges = checked_sum(
                edges, branch.state.edge_count(),
                "Phase-graph branch-sum total edge count overflowed");
        }
        stats_ = PhaseGraphBranchSumStats{
            qubit_count_,
            branches_.size(),
            hadamard_defects_,
            max_live_branches_,
            edges,
            retained,
        };
    }

    [[nodiscard]] std::size_t estimated_bytes(
        const std::vector<PhaseGraphBranch>& branches) const {
        std::size_t total = sizeof(*this);
        total = checked_sum(
            total,
            checked_product(branches.capacity(), sizeof(PhaseGraphBranch),
                "Phase-graph branch-sum vector storage overflowed"),
            "Phase-graph branch-sum retained storage overflowed");
        for (const PhaseGraphBranch& branch : branches) {
            const std::size_t graph_bytes = branch.state.estimated_bytes();
            if (graph_bytes > sizeof(PhaseGraphState)) {
                total = checked_sum(
                    total,
                    graph_bytes - sizeof(PhaseGraphState),
                    "Phase-graph branch-sum retained storage overflowed");
            }
        }
        return total;
    }

    void enforce_retained_cap(std::size_t retained) const {
        if (retained > config_.max_retained_estimated_bytes) {
            throw QStateError("Phase-graph branch-sum exceeds configured retained-byte estimate cap");
        }
    }

    [[nodiscard]] static bool finite(QComplex value) noexcept {
        return std::isfinite(value.re) && std::isfinite(value.im);
    }

    [[nodiscard]] static std::size_t checked_product(
        std::size_t left, std::size_t right, const char* message) {
        if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
            throw QStateError(message);
        }
        return left * right;
    }

    [[nodiscard]] static std::size_t checked_sum(
        std::size_t left, std::size_t right, const char* message) {
        if (right > std::numeric_limits<std::size_t>::max() - left) {
            throw QStateError(message);
        }
        return left + right;
    }
};

}  // namespace qubit

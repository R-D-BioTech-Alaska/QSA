#include "qubit/qadaptive_basis_amplitude.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct DenseReference {
    std::size_t qubits{0U};
    std::vector<qubit::QComplex> amplitudes{};

    explicit DenseReference(std::span<const std::uint8_t> input_bits)
        : qubits(input_bits.size()) {
        amplitudes.assign(std::size_t{1U} << qubits, {});
        std::size_t basis = 0U;
        for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
            basis |= static_cast<std::size_t>(input_bits[qubit]) << qubit;
        }
        amplitudes[basis] = {1.0, 0.0};
    }

    void single(qubit::QubitId qubit, const qubit::QMatrix2& matrix) {
        const std::size_t mask = std::size_t{1U} << qubit;
        for (std::size_t basis = 0U; basis < amplitudes.size(); ++basis) {
            if ((basis & mask) != 0U) {
                continue;
            }
            const std::size_t one = basis | mask;
            const auto zero_value = amplitudes[basis];
            const auto one_value = amplitudes[one];
            amplitudes[basis] = matrix(0U, 0U) * zero_value + matrix(0U, 1U) * one_value;
            amplitudes[one] = matrix(1U, 0U) * zero_value + matrix(1U, 1U) * one_value;
        }
    }

    void cz(qubit::QubitId first, qubit::QubitId second) {
        const std::size_t mask =
            (std::size_t{1U} << first) | (std::size_t{1U} << second);
        for (std::size_t basis = 0U; basis < amplitudes.size(); ++basis) {
            if ((basis & mask) == mask) {
                amplitudes[basis] = -amplitudes[basis];
            }
        }
    }

    void swap(qubit::QubitId first, qubit::QubitId second) {
        const std::size_t first_mask = std::size_t{1U} << first;
        const std::size_t second_mask = std::size_t{1U} << second;
        for (std::size_t basis = 0U; basis < amplitudes.size(); ++basis) {
            const bool first_bit = (basis & first_mask) != 0U;
            const bool second_bit = (basis & second_mask) != 0U;
            if (first_bit == second_bit || first_bit) {
                continue;
            }
            const std::size_t other = basis ^ first_mask ^ second_mask;
            std::swap(amplitudes[basis], amplitudes[other]);
        }
    }

    void apply(const qubit::Operation& operation) {
        using qubit::OperationCode;
        switch (operation.code) {
            case OperationCode::X: single(operation.first, qubit::gates::x()); return;
            case OperationCode::Y: single(operation.first, qubit::gates::y()); return;
            case OperationCode::Z: single(operation.first, qubit::gates::z()); return;
            case OperationCode::H: single(operation.first, qubit::gates::h()); return;
            case OperationCode::S: single(operation.first, qubit::gates::s()); return;
            case OperationCode::Sdg: single(operation.first, qubit::gates::sdg()); return;
            case OperationCode::T: single(operation.first, qubit::gates::t()); return;
            case OperationCode::Tdg: single(operation.first, qubit::gates::tdg()); return;
            case OperationCode::Rz:
                single(operation.first, qubit::gates::rz(operation.parameter));
                return;
            case OperationCode::Cz: cz(operation.first, operation.second); return;
            case OperationCode::Swap: swap(operation.first, operation.second); return;
            default:
                throw std::runtime_error("dense adaptive basis reference received unsupported operation");
        }
    }
};

qubit::ExactAdaptiveAmplitudeConfig config() {
    qubit::ExactAdaptiveAmplitudeConfig result;
    result.advisor.max_qubits = 4096U;
    result.advisor.max_operations = 20000U;
    result.advisor.max_phase_h_defects = 20U;
    result.advisor.max_phase_branches = 1U << 20U;
    result.advisor.max_hpath_events = 4096U;
    result.advisor.minimum_hpath_log2_margin = 4U;
    result.hpath.factor.max_variables = 4096U;
    result.hpath.factor.max_factors = 20000U;
    result.hpath.factor.max_factor_entries = 4096U;
    result.hpath.factor.max_compiled_index_entries = 1U << 20U;
    result.hpath.factor.reuse_workspace_slots = true;
    result.hpath.max_qubits = 4096U;
    result.hpath.max_operations = 20000U;
    result.hpath.max_h_events = 4096U;
    result.hpath.max_metadata_bytes = 64U * 1024U * 1024U;
    result.phase_graph.max_branches = 1U << 20U;
    result.phase_graph.max_retained_estimated_bytes = 256U * 1024U * 1024U;
    return result;
}

void compare_all(
    qubit::ExactAdaptiveBasisAmplitudePlan& plan,
    qubit::ExactAdaptiveBasisAmplitudeWorkspace& workspace,
    const DenseReference& dense,
    double tolerance) {
    std::vector<std::uint8_t> bits(dense.qubits, 0U);
    for (std::size_t basis = 0U; basis < dense.amplitudes.size(); ++basis) {
        for (std::size_t qubit = 0U; qubit < dense.qubits; ++qubit) {
            bits[qubit] = static_cast<std::uint8_t>((basis >> qubit) & 1U);
        }
        require(qubit::almost_equal(
                    plan.amplitude_bits(bits, workspace),
                    dense.amplitudes[basis],
                    tolerance),
            "adaptive basis amplitude differs from dense reference");
    }
}

void phase_only_route_is_exact() {
    using qubit::ExactAdaptiveBasisAmplitudePlan;
    using qubit::ExactAmplitudeRoute;
    using qubit::Operation;
    using qubit::OperationCode;

    const std::array<std::uint8_t, 3> input{1U, 0U, 1U};
    const std::vector<Operation> operations{
        {OperationCode::X, 0U},
        {OperationCode::Swap, 0U, 2U},
        {OperationCode::Y, 1U},
        {OperationCode::T, 2U},
    };
    auto settings = config();
    ExactAdaptiveBasisAmplitudePlan plan(input, operations, settings);
    require(plan.route() == ExactAmplitudeRoute::PhaseGraphBranchSum,
        "adaptive basis did not select its phase-only route");
    require(plan.decision().phase_graph_eligible && !plan.decision().hpath_eligible,
        "adaptive basis phase-only eligibility mismatch");
    require(plan.decision().phase_h_defects == input.size(),
        "adaptive basis preparation defect count mismatch");

    DenseReference dense(input);
    for (const auto& operation : operations) {
        dense.apply(operation);
    }
    auto workspace = plan.workspace();
    compare_all(plan, workspace, dense, 3e-11);
}

void structurally_lower_hpath_route_is_exact() {
    using qubit::ExactAdaptiveBasisAmplitudePlan;
    using qubit::ExactAmplitudeRoute;
    using qubit::ExactPreparedBasisHadamardPathPlan;
    using qubit::Operation;
    using qubit::OperationCode;

    constexpr std::size_t qubits = 6U;
    const std::array<std::uint8_t, qubits> input{1U, 0U, 1U, 1U, 0U, 1U};
    std::vector<Operation> operations;
    for (std::size_t round = 0U; round < 2U; ++round) {
        for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
            operations.push_back({OperationCode::H, static_cast<qubit::QubitId>(qubit)});
            operations.push_back({
                OperationCode::Rz,
                static_cast<qubit::QubitId>(qubit),
                0U,
                0.071 * static_cast<double>(1U + round + qubit),
            });
        }
        for (std::size_t qubit = 0U; qubit + 1U < qubits; ++qubit) {
            operations.push_back({
                OperationCode::Cz,
                static_cast<qubit::QubitId>(qubit),
                static_cast<qubit::QubitId>(qubit + 1U),
            });
        }
    }

    auto settings = config();
    ExactAdaptiveBasisAmplitudePlan adaptive(input, operations, settings);
    require(adaptive.route() == ExactAmplitudeRoute::HadamardPathFactor,
        "adaptive basis did not select lower-width Hpath");
    require(adaptive.decision().phase_graph_eligible && adaptive.decision().hpath_eligible,
        "adaptive basis both-route eligibility mismatch");
    require(adaptive.decision().phase_h_defects == 18U,
        "adaptive basis total phase-defect count mismatch");
    require(adaptive.decision().structural_log2_margin >= 4U,
        "adaptive basis Hpath structural margin was not certified");

    ExactPreparedBasisHadamardPathPlan direct(input, operations, settings.hpath);
    auto adaptive_workspace = adaptive.workspace();
    auto direct_workspace = direct.workspace();
    std::array<std::uint8_t, qubits> bits{};
    for (std::size_t basis = 0U; basis < (std::size_t{1U} << qubits); ++basis) {
        for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
            bits[qubit] = static_cast<std::uint8_t>((basis >> qubit) & 1U);
        }
        require(qubit::almost_equal(
                    adaptive.amplitude_bits(bits, adaptive_workspace),
                    direct.amplitude_bits(bits, direct_workspace),
                    2e-12),
            "adaptive basis Hpath route differs from direct basis Hpath");
    }
}

void resource_rejection_falls_back_to_phase() {
    using qubit::ExactAdaptiveBasisAmplitudePlan;
    using qubit::ExactAmplitudeRoute;
    using qubit::Operation;
    using qubit::OperationCode;

    constexpr std::size_t qubits = 4U;
    const std::array<std::uint8_t, qubits> input{0U, 1U, 0U, 1U};
    std::vector<Operation> operations;
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({OperationCode::H, static_cast<qubit::QubitId>(qubit)});
    }
    for (std::size_t first = 0U; first < qubits; ++first) {
        for (std::size_t second = first + 1U; second < qubits; ++second) {
            operations.push_back({
                OperationCode::Cz,
                static_cast<qubit::QubitId>(first),
                static_cast<qubit::QubitId>(second),
            });
        }
    }
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({OperationCode::H, static_cast<qubit::QubitId>(qubit)});
    }

    auto settings = config();
    settings.advisor.minimum_hpath_log2_margin = 1U;
    settings.hpath.factor.max_factor_entries = 4U;
    settings.phase_graph.max_branches = 4096U;
    settings.advisor.max_phase_branches = 4096U;
    ExactAdaptiveBasisAmplitudePlan adaptive(input, operations, settings);
    require(adaptive.route() == ExactAmplitudeRoute::PhaseGraphBranchSum,
        "adaptive basis did not fall back to phase after Hpath width rejection");
    require(adaptive.decision().phase_graph_eligible,
        "adaptive basis phase fallback was not eligible");
    require(!adaptive.decision().hpath_eligible && adaptive.decision().hpath_resource_rejected,
        "adaptive basis did not expose Hpath resource rejection");

    DenseReference dense(input);
    for (const auto& operation : operations) {
        dense.apply(operation);
    }
    auto workspace = adaptive.workspace();
    compare_all(adaptive, workspace, dense, 5e-11);
}

void unresolved_and_invalid_cases_fail_closed() {
    using qubit::ExactAdaptiveBasisAmplitudePlan;
    using qubit::Operation;
    using qubit::OperationCode;
    using qubit::QStateError;

    const std::array<std::uint8_t, 2> input{0U, 1U};
    const std::vector<Operation> operations{
        {OperationCode::H, 0U},
        {OperationCode::H, 0U},
    };
    auto unresolved = config();
    unresolved.advisor.minimum_hpath_log2_margin = 8U;
    bool rejected = false;
    try {
        (void)ExactAdaptiveBasisAmplitudePlan(input, operations, unresolved);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "adaptive basis forced an unresolved common-route envelope");

    rejected = false;
    try {
        const std::array<std::uint8_t, 2> bad_input{0U, 2U};
        (void)ExactAdaptiveBasisAmplitudePlan(bad_input, operations, config());
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "adaptive basis accepted non-binary input bits");

    rejected = false;
    try {
        const std::vector<Operation> unsupported{{OperationCode::Cnot, 0U, 1U}};
        (void)ExactAdaptiveBasisAmplitudePlan(input, unsupported, config());
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "adaptive basis accepted a circuit with no eligible exact route");

    rejected = false;
    try {
        const std::vector<Operation> bad_rz{{
            OperationCode::Rz, 0U, 0U, std::numeric_limits<double>::quiet_NaN()}};
        (void)ExactAdaptiveBasisAmplitudePlan(input, bad_rz, config());
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "adaptive basis accepted a non-finite rotation");
}

}  // namespace

int main() {
    phase_only_route_is_exact();
    structurally_lower_hpath_route_is_exact();
    resource_rejection_falls_back_to_phase();
    unresolved_and_invalid_cases_fail_closed();
    return 0;
}

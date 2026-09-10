#include "qubit/qadaptive_amplitude.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

qubit::ExactAdaptiveAmplitudeConfig config() {
    qubit::ExactAdaptiveAmplitudeConfig result;
    result.advisor.max_qubits = 4096U;
    result.advisor.max_operations = 20000U;
    result.advisor.max_phase_h_defects = 20U;
    result.advisor.max_hpath_events = 4096U;
    result.advisor.minimum_hpath_log2_margin = 8U;
    result.phase_graph.max_branches = 1U << 20U;
    result.phase_graph.max_retained_estimated_bytes = 256U * 1024U * 1024U;
    result.hpath.factor.max_variables = 4096U;
    result.hpath.factor.max_factors = 20000U;
    result.hpath.factor.max_factor_entries = 4096U;
    result.hpath.factor.max_compiled_index_entries = 1U << 20U;
    result.hpath.max_qubits = 4096U;
    result.hpath.max_operations = 20000U;
    result.hpath.max_h_events = 4096U;
    return result;
}

std::vector<qubit::Operation> low_width_grid() {
    using qubit::Operation;
    using qubit::OperationCode;
    std::vector<Operation> operations;
    constexpr std::size_t qubits = 8U;
    constexpr std::size_t rounds = 4U;
    for (std::size_t round = 0U; round < rounds; ++round) {
        for (std::size_t target = 0U; target < qubits; ++target) {
            operations.push_back({OperationCode::H, static_cast<qubit::QubitId>(target)});
            operations.push_back({
                OperationCode::Rz,
                static_cast<qubit::QubitId>(target),
                0U,
                0.031 * static_cast<double>(1U + round * qubits + target),
            });
        }
        for (std::size_t target = 0U; target + 1U < qubits; ++target) {
            operations.push_back({
                OperationCode::Cz,
                static_cast<qubit::QubitId>(target),
                static_cast<qubit::QubitId>(target + 1U),
            });
        }
    }
    return operations;
}

void executes_hpath_route() {
    using qubit::ExactAdaptiveAmplitudePlan;
    using qubit::ExactAmplitudeRoute;
    using qubit::ExactHadamardPathAmplitudePlan;

    const auto operations = low_width_grid();
    const auto settings = config();
    ExactAdaptiveAmplitudePlan adaptive(8U, operations, settings);
    require(adaptive.route() == ExactAmplitudeRoute::HadamardPathFactor,
        "adaptive amplitude did not select Hpath on low-width grid");
    ExactHadamardPathAmplitudePlan direct(8U, operations, settings.hpath);
    std::array<std::uint8_t, 8> bits{};
    for (std::size_t basis = 0U; basis < 256U; ++basis) {
        for (std::size_t qubit = 0U; qubit < bits.size(); ++qubit) {
            bits[qubit] = static_cast<std::uint8_t>((basis >> qubit) & 1U);
        }
        require(qubit::almost_equal(
                    adaptive.amplitude_bits(bits), direct.amplitude_bits(bits), 2e-12),
            "adaptive Hpath amplitude differs from direct Hpath route");
    }
}

void executes_phase_route() {
    using qubit::ExactAdaptiveAmplitudePlan;
    using qubit::ExactAmplitudeRoute;
    using qubit::ExactPhaseGraphBranchSum;
    using qubit::Operation;
    using qubit::OperationCode;

    const std::vector<Operation> operations{
        {OperationCode::H, 0U},
        {OperationCode::X, 0U},
        {OperationCode::T, 1U},
        {OperationCode::Cz, 0U, 1U},
    };
    const auto settings = config();
    ExactAdaptiveAmplitudePlan adaptive(4U, operations, settings);
    require(adaptive.route() == ExactAmplitudeRoute::PhaseGraphBranchSum,
        "adaptive amplitude did not select PhaseGraph-only circuit");
    ExactPhaseGraphBranchSum direct(4U, settings.phase_graph);
    for (const Operation& operation : operations) {
        direct.apply(operation);
    }
    const std::array<std::uint8_t, 4> bits{1U, 0U, 1U, 0U};
    require(qubit::almost_equal(
                adaptive.amplitude_bits(bits), direct.amplitude_bits(bits), 2e-12),
        "adaptive PhaseGraph amplitude differs from direct route");
}

void refuses_unresolved_and_overcap() {
    using qubit::ExactAdaptiveAmplitudePlan;
    using qubit::Operation;
    using qubit::OperationCode;
    using qubit::QStateError;

    const std::vector<Operation> close_case{
        {OperationCode::H, 0U},
        {OperationCode::H, 1U},
        {OperationCode::Rz, 0U, 0U, 0.2},
    };
    bool rejected = false;
    try {
        (void)ExactAdaptiveAmplitudePlan(4U, close_case, config());
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "adaptive amplitude forced an unresolved close route");

    auto settings = config();
    settings.phase_graph.max_branches = 4U;
    settings.hpath.max_h_events = 1U;
    const std::vector<Operation> overcap{
        {OperationCode::H, 0U},
        {OperationCode::H, 1U},
        {OperationCode::H, 2U},
        {OperationCode::X, 0U},
    };
    rejected = false;
    try {
        (void)ExactAdaptiveAmplitudePlan(4U, overcap, settings);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "adaptive amplitude ignored aligned route caps");
}

}  // namespace

int main() {
    executes_hpath_route();
    executes_phase_route();
    refuses_unresolved_and_overcap();
    return 0;
}

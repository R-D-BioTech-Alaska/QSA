#include "qubit/qamplitude_advisor.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

qubit::ExactAmplitudeAdvisorConfig wide_config() {
    qubit::ExactAmplitudeAdvisorConfig config;
    config.factor.max_variables = 4096U;
    config.factor.max_factors = 20000U;
    config.factor.max_factor_entries = 4096U;
    config.factor.max_compiled_index_entries = 1U << 20U;
    config.max_qubits = 4096U;
    config.max_operations = 20000U;
    config.max_phase_h_defects = 20U;
    config.max_phase_branches = 1U << 20U;
    config.max_hpath_events = 4096U;
    config.minimum_hpath_log2_margin = 8U;
    return config;
}

std::vector<qubit::Operation> low_width_grid(
    std::size_t qubits,
    std::size_t rounds) {
    using qubit::Operation;
    using qubit::OperationCode;
    std::vector<Operation> operations;
    for (std::size_t round = 0U; round < rounds; ++round) {
        for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
            operations.push_back({OperationCode::H, static_cast<qubit::QubitId>(qubit)});
            operations.push_back({
                OperationCode::Rz,
                static_cast<qubit::QubitId>(qubit),
                0U,
                0.07 * static_cast<double>(1U + round * qubits + qubit),
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
    return operations;
}

std::vector<qubit::Operation> comparable_chain(std::size_t qubits) {
    using qubit::Operation;
    using qubit::OperationCode;
    std::vector<Operation> operations;
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({
            OperationCode::Rz,
            static_cast<qubit::QubitId>(qubit),
            0U,
            0.031 * static_cast<double>(qubit + 1U),
        });
    }
    for (std::size_t qubit = 0U; qubit + 1U < qubits; ++qubit) {
        operations.push_back({
            OperationCode::Cz,
            static_cast<qubit::QubitId>(qubit),
            static_cast<qubit::QubitId>(qubit + 1U),
        });
    }
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({OperationCode::H, static_cast<qubit::QubitId>(qubit)});
    }
    return operations;
}

void hpath_dominates_flat_branch_envelope() {
    using qubit::ExactAmplitudeRepresentationAdvisor;
    using qubit::ExactAmplitudeRoute;

    const auto operations = comparable_chain(12U);
    const auto decision = ExactAmplitudeRepresentationAdvisor::analyze_plus_state(
        12U, operations, wide_config());
    require(decision.phase_graph_eligible, "comparable chain lost PhaseGraph eligibility");
    require(decision.hpath_eligible, "comparable chain lost Hpath eligibility");
    require(decision.phase_h_defects == 12U, "comparable chain H count mismatch");
    require(decision.hpath.h_events == 12U, "comparable chain Hpath event mismatch");
    require(decision.hpath.peak_factor_entries <= 16U,
        "comparable chain factor envelope unexpectedly wide");
    require(decision.route == ExactAmplitudeRoute::HadamardPathFactor,
        "advisor did not select Hpath under a certified structural margin");
    require(decision.structural_log2_margin >= 8U,
        "advisor structural margin fell below configured gate");
}

void route_boundaries() {
    using qubit::ExactAmplitudeRepresentationAdvisor;
    using qubit::ExactAmplitudeRoute;
    using qubit::Operation;
    using qubit::OperationCode;

    const std::vector<Operation> phase_only{
        {OperationCode::H, 0U},
        {OperationCode::X, 0U},
        {OperationCode::T, 1U},
    };
    const auto phase = ExactAmplitudeRepresentationAdvisor::analyze_plus_state(
        4U, phase_only, wide_config());
    require(phase.route == ExactAmplitudeRoute::PhaseGraphBranchSum,
        "phase-only circuit did not select PhaseGraph");

    const std::vector<Operation> unsupported{{OperationCode::Cnot, 0U, 1U}};
    const auto none = ExactAmplitudeRepresentationAdvisor::analyze_plus_state(
        4U, unsupported, wide_config());
    require(none.route == ExactAmplitudeRoute::None,
        "unsupported circuit did not reject both amplitude routes");

    const std::vector<Operation> close_case{
        {OperationCode::H, 0U},
        {OperationCode::H, 1U},
        {OperationCode::Rz, 0U, 0U, 0.23},
    };
    const auto unresolved = ExactAmplitudeRepresentationAdvisor::analyze_plus_state(
        4U, close_case, wide_config());
    require(unresolved.route == ExactAmplitudeRoute::Indeterminate,
        "close structural envelopes should remain unresolved");

    auto hpath_capped = wide_config();
    hpath_capped.max_hpath_events = 1U;
    const std::vector<Operation> phase_survives{
        {OperationCode::H, 0U},
        {OperationCode::H, 1U},
    };
    const auto capped = ExactAmplitudeRepresentationAdvisor::analyze_plus_state(
        4U, phase_survives, hpath_capped);
    require(capped.route == ExactAmplitudeRoute::PhaseGraphBranchSum,
        "Hpath event cap should leave an eligible PhaseGraph route available");

    auto branch_capped = wide_config();
    branch_capped.max_phase_branches = 4U;
    const std::vector<Operation> too_many_phase_branches{
        {OperationCode::H, 0U},
        {OperationCode::H, 1U},
        {OperationCode::H, 2U},
        {OperationCode::X, 0U},
    };
    const auto branch_rejected = ExactAmplitudeRepresentationAdvisor::analyze_plus_state(
        4U, too_many_phase_branches, branch_capped);
    require(!branch_rejected.phase_graph_eligible,
        "PhaseGraph remained eligible above its actual branch cap");
}

void fail_closed_validation() {
    using qubit::ExactAmplitudeRepresentationAdvisor;
    using qubit::Operation;
    using qubit::OperationCode;
    using qubit::QStateError;

    bool rejected = false;
    try {
        const std::vector<Operation> bad{{
            OperationCode::Rz,
            0U,
            0U,
            std::numeric_limits<double>::infinity(),
        }};
        (void)ExactAmplitudeRepresentationAdvisor::analyze_plus_state(
            2U, bad, wide_config());
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "amplitude advisor accepted a non-finite Rz angle");

    rejected = false;
    try {
        const std::vector<Operation> bad{{OperationCode::Cz, 0U, 0U}};
        (void)ExactAmplitudeRepresentationAdvisor::analyze_plus_state(
            2U, bad, wide_config());
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "amplitude advisor accepted invalid CZ support");
}

}  // namespace

int main() {
    hpath_dominates_flat_branch_envelope();
    route_boundaries();
    fail_closed_validation();
    return 0;
}

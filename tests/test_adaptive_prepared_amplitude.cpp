#include "qubit/qadaptive_prepared_amplitude.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
    result.advisor.max_phase_h_defects = 32U;
    result.advisor.max_phase_branches = 1U << 20U;
    result.advisor.max_hpath_events = 4096U;
    result.advisor.minimum_hpath_log2_margin = 4U;
    result.phase_graph.max_branches = 1U << 20U;
    result.phase_graph.max_retained_estimated_bytes = 256U * 1024U * 1024U;
    result.hpath.factor.max_variables = 4096U;
    result.hpath.factor.max_factors = 20000U;
    result.hpath.factor.max_factor_entries = 4096U;
    result.hpath.factor.max_compiled_index_entries = 1U << 20U;
    result.hpath.factor.reuse_workspace_slots = true;
    result.hpath.max_qubits = 4096U;
    result.hpath.max_operations = 20000U;
    result.hpath.max_h_events = 4096U;
    result.hpath.max_metadata_bytes = 64U * 1024U * 1024U;
    return result;
}

std::vector<qubit::Operation> comparable_chain(std::size_t qubits) {
    using qubit::Operation;
    using qubit::OperationCode;
    std::vector<Operation> operations;
    for (std::size_t target = 0U; target < qubits; ++target) {
        operations.push_back({
            OperationCode::Rz,
            static_cast<qubit::QubitId>(target),
            0U,
            0.041 * static_cast<double>(target + 1U),
        });
    }
    for (std::size_t target = 0U; target + 1U < qubits; ++target) {
        operations.push_back({
            OperationCode::Cz,
            static_cast<qubit::QubitId>(target),
            static_cast<qubit::QubitId>(target + 1U),
        });
    }
    for (std::size_t target = 0U; target < qubits; ++target) {
        operations.push_back({OperationCode::H, static_cast<qubit::QubitId>(target)});
    }
    return operations;
}

void shared_hpath_matches_existing_adaptive() {
    using qubit::ExactAdaptiveAmplitudePlan;
    using qubit::ExactAdaptivePreparedAmplitudePlan;
    using qubit::ExactAmplitudeRoute;

    constexpr std::size_t qubits = 8U;
    const auto operations = comparable_chain(qubits);
    auto settings = config();
    ExactAdaptiveAmplitudePlan existing(qubits, operations, settings);
    ExactAdaptivePreparedAmplitudePlan prepared(qubits, operations, settings);
    auto workspace = prepared.workspace();
    require(existing.route() == ExactAmplitudeRoute::HadamardPathFactor,
        "existing adaptive plan did not select Hpath on matched carrier");
    require(prepared.route() == ExactAmplitudeRoute::HadamardPathFactor,
        "prepared adaptive plan did not select Hpath on matched carrier");
    require(prepared.has_prepared_hpath(),
        "prepared adaptive plan selected Hpath without retaining prepared topology");
    require(prepared.decision().structural_log2_margin >= 4U,
        "prepared adaptive Hpath selection lost structural margin");

    std::array<std::uint8_t, qubits> bits{};
    for (std::size_t basis = 0U; basis < (std::size_t{1U} << qubits); ++basis) {
        for (std::size_t bit = 0U; bit < qubits; ++bit) {
            bits[bit] = static_cast<std::uint8_t>((basis >> bit) & 1U);
        }
        const qubit::QComplex left = existing.amplitude_bits(bits);
        const qubit::QComplex right = prepared.amplitude_bits(bits, workspace);
        require(qubit::almost_equal(left, right, 3e-12),
            "prepared adaptive Hpath differs from existing adaptive execution");
    }
    require(workspace.hpath_rebind_count() == qubits * (std::size_t{1U} << qubits),
        "prepared adaptive targeted rebind count mismatch");
}

void phase_only_route_is_preserved() {
    using qubit::ExactAdaptiveAmplitudePlan;
    using qubit::ExactAdaptivePreparedAmplitudePlan;
    using qubit::ExactAmplitudeRoute;
    using qubit::Operation;
    using qubit::OperationCode;

    const std::vector<Operation> operations{
        {OperationCode::H, 0U},
        {OperationCode::X, 0U},
        {OperationCode::T, 1U},
        {OperationCode::Cz, 0U, 1U},
    };
    auto settings = config();
    ExactAdaptiveAmplitudePlan existing(4U, operations, settings);
    ExactAdaptivePreparedAmplitudePlan prepared(4U, operations, settings);
    auto workspace = prepared.workspace();
    require(existing.route() == ExactAmplitudeRoute::PhaseGraphBranchSum,
        "existing adaptive phase-only route changed");
    require(prepared.route() == ExactAmplitudeRoute::PhaseGraphBranchSum,
        "prepared adaptive did not preserve phase-only route");
    require(!prepared.has_prepared_hpath(),
        "phase-only prepared adaptive retained an unused Hpath plan");
    const std::array<std::uint8_t, 4> bits{1U, 0U, 0U, 0U};
    require(qubit::almost_equal(
                existing.amplitude_bits(bits), prepared.amplitude_bits(bits, workspace), 2e-12),
        "prepared adaptive phase-only execution differs from existing route");
}

void hpath_event_cap_leaves_phase_route() {
    using qubit::ExactAdaptivePreparedAmplitudePlan;
    using qubit::ExactAmplitudeRoute;
    using qubit::Operation;
    using qubit::OperationCode;

    auto settings = config();
    settings.hpath.max_h_events = 1U;
    settings.advisor.max_hpath_events = 1U;
    const std::vector<Operation> operations{
        {OperationCode::H, 0U},
        {OperationCode::H, 1U},
    };
    ExactAdaptivePreparedAmplitudePlan prepared(4U, operations, settings);
    auto workspace = prepared.workspace();
    require(prepared.route() == ExactAmplitudeRoute::PhaseGraphBranchSum,
        "Hpath event cap did not preserve eligible PhaseGraph route");
    require(!prepared.has_prepared_hpath(),
        "over-cap Hpath candidate retained a prepared Hpath plan");
    const std::array<std::uint8_t, 4> bits{0U, 0U, 0U, 0U};
    require(std::isfinite(prepared.log2_probability_bits(bits, workspace)),
        "phase fallback returned non-finite log probability");
}

void unresolved_and_workspace_mismatch_reject() {
    using qubit::ExactAdaptivePreparedAmplitudePlan;
    using qubit::Operation;
    using qubit::OperationCode;
    using qubit::QStateError;

    auto unresolved_settings = config();
    unresolved_settings.advisor.minimum_hpath_log2_margin = 8U;
    const std::vector<Operation> unresolved{
        {OperationCode::H, 0U},
        {OperationCode::H, 1U},
        {OperationCode::Rz, 0U, 0U, 0.23},
    };
    bool rejected = false;
    try {
        (void)ExactAdaptivePreparedAmplitudePlan(4U, unresolved, unresolved_settings);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "prepared adaptive forced an unresolved representation choice");

    const auto operations = comparable_chain(8U);
    auto settings = config();
    ExactAdaptivePreparedAmplitudePlan first(8U, operations, settings);
    ExactAdaptivePreparedAmplitudePlan second(8U, operations, settings);
    auto wrong_workspace = second.workspace();
    std::array<std::uint8_t, 8> bits{};
    rejected = false;
    try {
        (void)first.log2_probability_bits(bits, wrong_workspace);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "prepared adaptive accepted a workspace from another plan");
}

void broad_hpath_width_rejects() {
    using qubit::ExactAdaptivePreparedAmplitudePlan;
    using qubit::Operation;
    using qubit::OperationCode;
    using qubit::QStateError;

    constexpr std::size_t qubits = 16U;
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
    settings.hpath.factor.max_factor_entries = 1024U;
    bool rejected = false;
    try {
        (void)ExactAdaptivePreparedAmplitudePlan(qubits, operations, settings);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "prepared adaptive Hpath did not reject broad induced width");
}

}  // namespace

int main() {
    shared_hpath_matches_existing_adaptive();
    phase_only_route_is_preserved();
    hpath_event_cap_leaves_phase_route();
    unresolved_and_workspace_mismatch_reject();
    broad_hpath_width_rejects();
    return 0;
}

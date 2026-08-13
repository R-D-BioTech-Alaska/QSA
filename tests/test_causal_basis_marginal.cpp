#include "qubit/qcausal_basis_marginal.hpp"
#include "qubit/qcausal_basis_born.hpp"

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

qubit::ExactCausalBasisMarginalConfig config() {
    qubit::ExactCausalBasisMarginalConfig result;
    result.light_cone.max_qubits = 4096U;
    result.light_cone.max_operations = 20000U;
    result.light_cone.max_active_qubits = 128U;
    result.light_cone.max_active_operations = 4096U;
    result.born.factor.max_variables = 4096U;
    result.born.factor.max_factors = 20000U;
    result.born.factor.max_factor_entries = 4096U;
    result.born.factor.max_compiled_index_entries = 1U << 20U;
    result.born.factor.reuse_workspace_slots = true;
    result.born.max_qubits = 4096U;
    result.born.max_operations = 20000U;
    result.born.max_h_events = 4096U;
    result.born.max_retained_qubits = 8U;
    result.local_state.max_component_qubits = 20U;
    result.local_state.max_dense_amplitudes = 1ULL << 20U;
    result.local_state.max_sparse_entries = 1'000'000U;
    result.max_local_qubits = 20U;
    result.local_preference_max_amplitudes = 1ULL << 12U;
    result.max_local_dense_bytes = 64U * 1024U * 1024U;
    return result;
}

std::vector<qubit::QComplex> full_register_marginal(
    std::span<const std::uint8_t> input,
    std::span<const qubit::Operation> operations,
    std::span<const std::size_t> retained) {
    qubit::QRegister state(input.size());
    for (std::size_t qubit = 0U; qubit < input.size(); ++qubit) {
        if (input[qubit] != 0U) {
            state.apply_x(static_cast<qubit::QubitId>(qubit));
        }
    }
    qubit::OperationPlan plan(operations, true);
    plan.execute(state);

    std::vector<qubit::QubitId> retained_ids;
    retained_ids.reserve(retained.size());
    for (const std::size_t qubit : retained) {
        retained_ids.push_back(static_cast<qubit::QubitId>(qubit));
    }
    std::vector<std::uint8_t> bits(retained.size(), 0U);
    std::vector<qubit::QComplex> result(
        std::size_t{1U} << retained.size(), qubit::QComplex{});
    for (std::size_t index = 0U; index < result.size(); ++index) {
        for (std::size_t position = 0U; position < bits.size(); ++position) {
            bits[position] = static_cast<std::uint8_t>((index >> position) & 1U);
        }
        result[index] = {
            state.marginal_probability(retained_ids, bits),
            0.0,
        };
    }
    return result;
}

void compare(
    std::span<const qubit::QComplex> observed,
    std::span<const qubit::QComplex> expected,
    double tolerance,
    const char* message) {
    require(observed.size() == expected.size(), message);
    for (std::size_t index = 0U; index < observed.size(); ++index) {
        require(qubit::almost_equal(observed[index], expected[index], tolerance), message);
    }
}

void small_causal_problem_prefers_local_register() {
    using qubit::ExactCausalBasisMarginalPlan;
    using qubit::ExactCausalBasisMarginalRoute;
    using qubit::Operation;
    using qubit::OperationCode;

    const std::array<std::uint8_t, 6> input{1U, 0U, 1U, 0U, 1U, 1U};
    const std::vector<Operation> operations{
        {OperationCode::H, 0U},
        {OperationCode::Rx, 1U, 0U, 0.27},
        {OperationCode::Cnot, 0U, 1U},
        {OperationCode::Ry, 2U, 0U, -0.19},
        {OperationCode::Cz, 1U, 2U},
        {OperationCode::Swap, 0U, 2U},
        {OperationCode::Rz, 0U, 0U, 0.33},
        {OperationCode::H, 4U},
        {OperationCode::Cz, 4U, 5U},
        {OperationCode::T, 5U},
    };
    const std::array<std::size_t, 2> retained{0U, 1U};
    ExactCausalBasisMarginalPlan plan(input, operations, retained, config());
    require(plan.route() == ExactCausalBasisMarginalRoute::LocalRegister,
        "causal basis marginal did not choose the small-cone local register");
    require(plan.stats().local_eligible,
        "causal basis marginal local route was not certified eligible");
    require(!plan.stats().born_eligible,
        "causal basis marginal claimed Hpath eligibility for Rx/Ry/CNOT/Swap circuit");
    require(plan.stats().light_cone.active_qubits == 3U,
        "causal basis marginal local cone size mismatch");
    require(plan.stats().local_worst_case_amplitudes == 8U,
        "causal basis marginal local amplitude preflight mismatch");

    auto workspace = plan.workspace();
    const auto observed = plan.marginal(workspace);
    const auto expected = full_register_marginal(input, operations, retained);
    compare(observed, expected, 5e-11,
        "causal local-register marginal differs from full exact QRegister");
}

void hpath_handles_large_causal_support() {
    using qubit::ExactCausalBasisHadamardBornPlan;
    using qubit::ExactCausalBasisMarginalPlan;
    using qubit::ExactCausalBasisMarginalRoute;
    using qubit::Operation;
    using qubit::OperationCode;

    constexpr std::size_t qubits = 16U;
    std::array<std::uint8_t, qubits> input{};
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        input[qubit] = static_cast<std::uint8_t>((qubit * 5U + 1U) & 1U);
    }
    std::vector<Operation> operations;
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({OperationCode::H, static_cast<qubit::QubitId>(qubit)});
        operations.push_back({
            OperationCode::Rz,
            static_cast<qubit::QubitId>(qubit),
            0U,
            0.031 * static_cast<double>(qubit + 1U),
        });
    }
    for (std::size_t first = qubits - 1U; first != 0U; --first) {
        operations.push_back({
            OperationCode::Cz,
            static_cast<qubit::QubitId>(first - 1U),
            static_cast<qubit::QubitId>(first),
        });
    }
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({OperationCode::H, static_cast<qubit::QubitId>(qubit)});
    }
    const std::array<std::size_t, 1> retained{0U};

    auto settings = config();
    settings.max_local_qubits = 8U;
    settings.local_state.max_component_qubits = 8U;
    settings.local_state.max_dense_amplitudes = 1ULL << 8U;
    ExactCausalBasisMarginalPlan broker(input, operations, retained, settings);
    require(broker.route() == ExactCausalBasisMarginalRoute::HadamardBorn,
        "causal basis marginal did not choose Hpath for large causal support");
    require(!broker.stats().local_eligible && broker.stats().born_eligible,
        "causal basis marginal large-support route eligibility mismatch");
    require(broker.stats().light_cone.active_qubits == qubits,
        "causal basis marginal chain did not recover the full relevant support");

    qubit::ExactCausalBasisHadamardBornConfig born_settings;
    born_settings.light_cone = settings.light_cone;
    born_settings.born = settings.born;
    ExactCausalBasisHadamardBornPlan direct(input, operations, retained, born_settings);
    auto broker_workspace = broker.workspace();
    auto direct_workspace = direct.workspace();
    compare(
        broker.marginal(broker_workspace),
        direct.marginal(direct_workspace),
        2e-12,
        "causal Hpath broker differs from direct causal Hpath Born");
}

void nonpreferred_local_falls_back_when_hpath_is_incompatible() {
    using qubit::ExactCausalBasisMarginalPlan;
    using qubit::ExactCausalBasisMarginalRoute;
    using qubit::Operation;
    using qubit::OperationCode;

    const std::array<std::uint8_t, 4> input{0U, 1U, 0U, 1U};
    const std::vector<Operation> operations{
        {OperationCode::H, 0U},
        {OperationCode::Cnot, 0U, 1U},
        {OperationCode::Rx, 1U, 0U, 0.4},
        {OperationCode::Cnot, 1U, 2U},
        {OperationCode::Ry, 2U, 0U, -0.2},
    };
    const std::array<std::size_t, 1> retained{2U};
    auto settings = config();
    settings.local_preference_max_amplitudes = 2U;
    ExactCausalBasisMarginalPlan plan(input, operations, retained, settings);
    require(plan.route() == ExactCausalBasisMarginalRoute::LocalRegister,
        "causal basis marginal did not fall back to bounded local execution");
    require(plan.stats().local_eligible && !plan.stats().born_eligible,
        "causal basis marginal local fallback eligibility mismatch");
    auto workspace = plan.workspace();
    compare(
        plan.marginal(workspace),
        full_register_marginal(input, operations, retained),
        5e-11,
        "causal local fallback differs from full exact QRegister");
}

void fail_closed_when_neither_route_is_bounded() {
    using qubit::ExactCausalBasisMarginalPlan;
    using qubit::Operation;
    using qubit::OperationCode;
    using qubit::QStateError;

    const std::array<std::uint8_t, 4> input{0U, 0U, 0U, 0U};
    const std::vector<Operation> operations{
        {OperationCode::H, 0U},
        {OperationCode::Cnot, 0U, 1U},
        {OperationCode::Cnot, 1U, 2U},
        {OperationCode::Rx, 2U, 0U, 0.17},
    };
    const std::array<std::size_t, 1> retained{2U};
    auto settings = config();
    settings.max_local_qubits = 2U;
    settings.local_state.max_component_qubits = 2U;
    settings.local_state.max_dense_amplitudes = 4U;
    bool rejected = false;
    try {
        (void)ExactCausalBasisMarginalPlan(input, operations, retained, settings);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "causal basis marginal forced a route outside both resource certificates");

    rejected = false;
    try {
        auto noisy = operations;
        noisy.push_back({OperationCode::DepolarizingTrajectory, 3U, 0U, 0.1, 0.5});
        (void)ExactCausalBasisMarginalPlan(input, noisy, retained, config());
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "causal basis marginal accepted nonunitary trajectory noise");
}

}  // namespace

int main() {
    small_causal_problem_prefers_local_register();
    hpath_handles_large_causal_support();
    nonpreferred_local_falls_back_when_hpath_is_incompatible();
    fail_closed_when_neither_route_is_bounded();
    return 0;
}

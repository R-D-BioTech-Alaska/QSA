#include "qubit/qhpath_born.hpp"
#include "qubit/qhpath_factor.hpp"

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

qubit::ExactHadamardBornConfig born_config() {
    qubit::ExactHadamardBornConfig config;
    config.factor.max_variables = 4096U;
    config.factor.max_factors = 20000U;
    config.factor.max_factor_entries = 1U << 16U;
    config.factor.max_compiled_index_entries = 1U << 20U;
    config.factor.reuse_workspace_slots = true;
    config.max_qubits = 4096U;
    config.max_operations = 20000U;
    config.max_h_events = 1024U;
    config.max_retained_qubits = 12U;
    return config;
}

qubit::ExactHadamardPathConfig hpath_config() {
    qubit::ExactHadamardPathConfig config;
    config.factor.max_variables = 1024U;
    config.factor.max_factors = 20000U;
    config.factor.max_factor_entries = 1U << 16U;
    config.factor.max_compiled_index_entries = 1U << 20U;
    config.factor.reuse_workspace_slots = true;
    config.max_qubits = 4096U;
    config.max_operations = 20000U;
    config.max_h_events = 1024U;
    config.max_metadata_bytes = 64U * 1024U * 1024U;
    return config;
}

std::vector<qubit::Operation> circuit() {
    using qubit::Operation;
    using qubit::OperationCode;
    return {
        {OperationCode::H, 0U},
        {OperationCode::Rz, 0U, 0U, 0.37},
        {OperationCode::Cz, 0U, 3U},
        {OperationCode::T, 3U},
        {OperationCode::H, 0U},
        {OperationCode::H, 1U},
        {OperationCode::S, 1U},
        {OperationCode::Cz, 1U, 2U},
        {OperationCode::H, 2U},
        {OperationCode::Rz, 2U, 0U, -0.21},
        {OperationCode::Cz, 2U, 3U},
        {OperationCode::H, 2U},
        {OperationCode::Cz, 1U, 3U},
        {OperationCode::Z, 3U},
    };
}

std::array<double, 16> direct_probabilities(
    const std::vector<qubit::Operation>& operations) {
    qubit::ExactHadamardPathAmplitudePlan direct(4U, operations, hpath_config());
    std::array<double, 16> result{};
    std::array<std::uint8_t, 4> bits{};
    for (std::size_t basis = 0U; basis < result.size(); ++basis) {
        for (std::size_t qubit = 0U; qubit < bits.size(); ++qubit) {
            bits[qubit] = static_cast<std::uint8_t>((basis >> qubit) & 1U);
        }
        result[basis] = direct.amplitude_bits(bits).norm2();
    }
    return result;
}

void full_born_matches_direct_amplitudes() {
    const auto operations = circuit();
    const auto expected = direct_probabilities(operations);
    const std::array<std::size_t, 4> retained{0U, 1U, 2U, 3U};
    qubit::ExactHadamardBornMarginalPlan born(
        4U, operations, retained, born_config());
    auto workspace = born.workspace();
    const auto actual = born.marginal(workspace);
    require(actual.size() == expected.size(), "Born full output size mismatch");
    double total = 0.0;
    for (std::size_t basis = 0U; basis < actual.size(); ++basis) {
        require(std::abs(actual[basis].im) <= 2e-11,
            "Born full probability retained imaginary residue");
        require(std::abs(actual[basis].re - expected[basis]) <= 4e-11,
            "Born full probability differs from direct Hpath amplitude norm");
        total += actual[basis].re;
    }
    require(std::abs(total - 1.0) <= 5e-11,
        "Born full probability mass does not normalize to one");
}

void selected_marginal_matches_direct_sum() {
    const auto operations = circuit();
    const auto direct = direct_probabilities(operations);
    const std::array<std::size_t, 2> retained{0U, 2U};
    qubit::ExactHadamardBornMarginalPlan born(
        4U, operations, retained, born_config());
    auto workspace = born.workspace();
    const auto actual = born.marginal(workspace);
    std::array<double, 4> expected{};
    for (std::size_t basis = 0U; basis < direct.size(); ++basis) {
        const std::size_t index = (basis & 1U) | (((basis >> 2U) & 1U) << 1U);
        expected[index] += direct[basis];
    }
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        require(std::abs(actual[index].im) <= 2e-11,
            "Born selected marginal retained imaginary residue");
        require(std::abs(actual[index].re - expected[index]) <= 5e-11,
            "Born selected marginal differs from direct probability sum");
    }
    const auto normalized = born.normalized_marginal(workspace);
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        require(std::abs(normalized[index].re - expected[index]) <= 5e-11,
            "Born normalized marginal differs from direct probability sum");
    }
}

void diagonal_no_h_is_uniform() {
    using qubit::Operation;
    using qubit::OperationCode;
    const std::vector<Operation> operations{
        {OperationCode::Rz, 0U, 0U, 0.31},
        {OperationCode::T, 1U},
        {OperationCode::Sdg, 2U},
        {OperationCode::Cz, 0U, 1U},
        {OperationCode::Cz, 1U, 3U},
    };
    const std::array<std::size_t, 2> retained{1U, 3U};
    qubit::ExactHadamardBornMarginalPlan born(
        4U, operations, retained, born_config());
    auto workspace = born.workspace();
    const auto marginal = born.marginal(workspace);
    require(born.stats().h_events == 0U, "no-H Born plan created H events");
    for (const auto& value : marginal) {
        require(std::abs(value.re - 0.25) <= 2e-12 && std::abs(value.im) <= 2e-12,
            "diagonal no-H Born marginal is not uniform");
    }

    const std::array<std::size_t, 0> none{};
    qubit::ExactHadamardBornMarginalPlan total_plan(
        4U, operations, none, born_config());
    auto total_workspace = total_plan.workspace();
    const auto total = total_plan.marginal(total_workspace);
    require(total.size() == 1U && std::abs(total[0].re - 1.0) <= 2e-12,
        "zero-retained Born partition is not one");
}

void fail_closed_boundaries() {
    using qubit::ExactHadamardBornMarginalPlan;
    using qubit::Operation;
    using qubit::OperationCode;
    using qubit::QStateError;

    const auto operations = circuit();
    bool rejected = false;
    try {
        const std::array<std::size_t, 2> duplicate{0U, 0U};
        (void)ExactHadamardBornMarginalPlan(4U, operations, duplicate, born_config());
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "Born compiler accepted duplicate retained qubits");

    rejected = false;
    try {
        auto limited = born_config();
        limited.max_retained_qubits = 1U;
        const std::array<std::size_t, 2> retained{0U, 1U};
        (void)ExactHadamardBornMarginalPlan(4U, operations, retained, limited);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "Born compiler exceeded retained-qubit cap");

    rejected = false;
    try {
        auto limited = born_config();
        limited.max_h_events = 1U;
        const std::array<std::size_t, 1> retained{0U};
        (void)ExactHadamardBornMarginalPlan(4U, operations, retained, limited);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "Born compiler exceeded H-event cap");

    rejected = false;
    try {
        const std::vector<Operation> unsupported{{OperationCode::X, 0U}};
        const std::array<std::size_t, 1> retained{0U};
        (void)ExactHadamardBornMarginalPlan(4U, unsupported, retained, born_config());
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "Born compiler accepted an operation outside its exact contract");
}

void workspace_identity_is_enforced() {
    using qubit::ExactHadamardBornMarginalPlan;
    using qubit::QStateError;
    const auto operations = circuit();
    const std::array<std::size_t, 1> first_retained{0U};
    const std::array<std::size_t, 1> second_retained{1U};
    ExactHadamardBornMarginalPlan first(4U, operations, first_retained, born_config());
    ExactHadamardBornMarginalPlan second(4U, operations, second_retained, born_config());
    auto wrong_workspace = second.workspace();
    bool rejected = false;
    try {
        (void)first.marginal(wrong_workspace);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "Born plan accepted a workspace from another plan");
}

}  // namespace

int main() {
    full_born_matches_direct_amplitudes();
    selected_marginal_matches_direct_sum();
    diagonal_no_h_is_uniform();
    fail_closed_boundaries();
    workspace_identity_is_enforced();
    return 0;
}

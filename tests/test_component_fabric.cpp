#include "qubit/qfabric.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace qubit;

namespace {

bool close(double left, double right, double tolerance = 2e-11) {
    return std::abs(left - right) <= tolerance;
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw QStateError(message);
    }
}

std::vector<Operation> two_island_program() {
    return {
        {OperationCode::H, 0U},
        {OperationCode::Cnot, 0U, 1U},
        {OperationCode::Ry, 2U, 0U, 0.37},
        {OperationCode::Cz, 1U, 2U},
        {OperationCode::H, 4U},
        {OperationCode::Cnot, 4U, 5U},
        {OperationCode::Rz, 6U, 0U, -0.23},
        {OperationCode::Cnot, 6U, 7U},
    };
}

void exact_small_control() {
    const std::vector<Operation> operations = two_island_program();
    ExactComponentProbabilityPlan fabric(8U, operations);
    ExactPreparedProbabilityPlan global = ExactPreparedProbabilityPlan::for_marginals(
        8U, operations);

    const std::vector<std::uint8_t> basis{0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U};
    std::vector<QubitId> all_qubits(8U);
    for (std::size_t index = 0U; index < all_qubits.size(); ++index) {
        all_qubits[index] = static_cast<QubitId>(index);
    }
    const double expected_basis = global.marginal_probability(all_qubits, basis).value;
    const double actual_basis = fabric.basis_probability(basis);
    require(close(actual_basis, expected_basis), "component fabric basis probability mismatch");

    const std::vector<QubitId> query{0U, 2U, 4U, 7U};
    const std::vector<std::uint8_t> bits{0U, 1U, 0U, 0U};
    const double expected_marginal = global.marginal_probability(query, bits).value;
    const double actual_marginal = fabric.marginal_probability(query, bits);
    require(close(actual_marginal, expected_marginal), "component fabric marginal mismatch");

    const ExactComponentProbabilityPlan pruned = ExactComponentProbabilityPlan::for_marginals(
        8U, operations, query);
    require(!pruned.full_basis_capable(), "query-pruned fabric incorrectly reports full-basis capability");
    require(pruned.stats().active_components == 3U, "query-pruned fabric active component count mismatch");
    require(pruned.stats().prepared_components == 3U, "query-pruned fabric backend count mismatch");
    require(pruned.stats().estimated_bytes < fabric.stats().estimated_bytes,
            "query-pruned fabric did not reduce retained storage");
    require(close(pruned.marginal_probability(query, bits), expected_marginal),
            "query-pruned component fabric marginal mismatch");

    require(fabric.stats().qubits == 8U, "component fabric qubit count mismatch");
    require(fabric.stats().operations == operations.size(), "component fabric operation count mismatch");
    require(fabric.stats().components == 4U, "component fabric did not preserve independent islands");
    require(fabric.stats().largest_component_qubits == 3U, "component fabric largest island mismatch");
    require(fabric.stats().active_components == 4U, "full fabric active component count mismatch");
}

void large_structured_carrier() {
    constexpr std::size_t qubits = 4096U;
    constexpr std::size_t block = 4U;
    constexpr std::size_t blocks = qubits / block;

    std::vector<Operation> operations;
    operations.reserve(blocks * 4U + 1U);
    for (std::size_t index = 0U; index < blocks; ++index) {
        const QubitId base = static_cast<QubitId>(index * block);
        operations.push_back({OperationCode::H, base});
        operations.push_back({OperationCode::Cnot, base, static_cast<QubitId>(base + 1U)});
        operations.push_back({OperationCode::Cnot, static_cast<QubitId>(base + 1U), static_cast<QubitId>(base + 2U)});
        operations.push_back({OperationCode::Cnot, static_cast<QubitId>(base + 2U), static_cast<QubitId>(base + 3U)});
    }
    operations.push_back({OperationCode::T, static_cast<QubitId>(qubits - block)});

    ExactComponentFabricConfig config;
    config.max_qubits = 8192U;
    config.max_components = 2048U;
    config.max_operations = 8192U;
    ExactComponentProbabilityPlan fabric(qubits, operations, config);

    require(fabric.stats().components == blocks, "large fabric component count mismatch");
    require(fabric.stats().largest_component_qubits == block, "large fabric component width mismatch");
    require(fabric.stats().active_components == blocks, "large full fabric active component count mismatch");
    require(fabric.stats().prepared_components == blocks, "large fabric prepared component count mismatch");

    const std::vector<QubitId> first_block{0U, 1U, 2U, 3U};
    const std::vector<std::uint8_t> zero_block(4U, 0U);
    require(close(fabric.marginal_probability(first_block, zero_block), 0.5),
            "large fabric GHZ block probability mismatch");

    const std::vector<QubitId> two_blocks{0U, 1U, 2U, 3U, 2048U, 2049U, 2050U, 2051U};
    const std::vector<std::uint8_t> two_zero_blocks(8U, 0U);
    const ExactComponentProbabilityPlan pruned = ExactComponentProbabilityPlan::for_marginals(
        qubits, operations, two_blocks, config);
    require(pruned.stats().active_components == 2U, "large pruned fabric did not isolate queried components");
    require(pruned.stats().prepared_components == 2U, "large pruned fabric prepared excess backends");
    require(pruned.stats().estimated_bytes < fabric.stats().estimated_bytes,
            "large pruned fabric did not reduce retained storage");
    require(close(pruned.marginal_probability(two_blocks, two_zero_blocks), 0.25),
            "large pruned fabric independent-block product mismatch");
}

void rejection_cases() {
    bool rejected = false;
    try {
        const std::vector<Operation> operations{{OperationCode::Cnot, 0U, 4U}};
        ExactComponentProbabilityPlan invalid(4U, operations);
        (void)invalid;
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "component fabric accepted an out-of-range dependency");

    rejected = false;
    try {
        const std::vector<Operation> operations{{OperationCode::H, 0U}};
        ExactComponentProbabilityPlan fabric(2U, operations);
        const std::vector<QubitId> duplicate{0U, 0U};
        const std::vector<std::uint8_t> bits{0U, 1U};
        (void)fabric.marginal_probability(duplicate, bits);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "component fabric accepted a duplicate marginal qubit");

    rejected = false;
    try {
        const std::vector<Operation> operations{{OperationCode::H, 0U}, {OperationCode::H, 1U}};
        const std::vector<QubitId> prepared{0U};
        const ExactComponentProbabilityPlan pruned = ExactComponentProbabilityPlan::for_marginals(
            2U, operations, prepared);
        const std::vector<QubitId> outside{1U};
        const std::vector<std::uint8_t> bit{0U};
        (void)pruned.marginal_probability(outside, bit);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "query-pruned fabric answered outside prepared support");

    rejected = false;
    try {
        const std::vector<Operation> operations{{OperationCode::H, 0U}};
        const std::vector<QubitId> prepared{0U};
        const ExactComponentProbabilityPlan pruned = ExactComponentProbabilityPlan::for_marginals(
            2U, operations, prepared);
        const std::vector<std::uint8_t> basis{0U, 0U};
        (void)pruned.basis_probability(basis);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "query-pruned fabric answered a full-basis query");
}

}  // namespace

int main() {
    exact_small_control();
    large_structured_carrier();
    rejection_cases();
    std::cout << "component fabric tests passed\n";
    return 0;
}

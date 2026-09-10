#include "qubit/qbroker.hpp"
#include "qubit/qcausal_index.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
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

std::vector<Operation> layered_chain(std::size_t qubits, std::size_t layers, bool non_clifford) {
    std::vector<Operation> operations;
    operations.reserve(layers * (qubits - 1U) + 1U);
    if (non_clifford) {
        operations.push_back({OperationCode::Ry, 0U, 0U, 0.37});
    } else {
        operations.push_back({OperationCode::H, 0U});
    }
    for (std::size_t layer = 0U; layer < layers; ++layer) {
        for (std::size_t qubit = 0U; qubit + 1U < qubits; ++qubit) {
            operations.push_back({
                OperationCode::Cnot,
                static_cast<QubitId>(qubit),
                static_cast<QubitId>(qubit + 1U),
            });
        }
    }
    return operations;
}

void exact_small_control() {
    constexpr std::size_t qubits = 12U;
    const std::vector<Operation> operations = layered_chain(qubits, 3U, true);
    ExactCausalOperationIndex index(qubits, operations);
    const std::vector<QubitId> query{0U};
    const ExactCausalSlice slice = index.slice(query);

    require(slice.global_qubits.size() == 4U, "causal index small slice qubit count mismatch");
    require(slice.operations.size() == 7U, "causal index small slice operation count mismatch");
    require(slice.local_query_qubits.size() == 1U && slice.local_query_qubits[0] == 0U,
            "causal index small query remap mismatch");

    ExactPreparedProbabilityPlan local = ExactPreparedProbabilityPlan::for_marginals(
        slice.global_qubits.size(), slice.operations);
    ExactPreparedProbabilityPlan global = ExactPreparedProbabilityPlan::for_marginals(
        qubits, operations);
    const std::vector<std::uint8_t> zero{0U};
    const std::vector<std::uint8_t> one{1U};
    require(close(
                local.marginal_probability(slice.local_query_qubits, zero).value,
                global.marginal_probability(query, zero).value),
            "causal index small zero probability mismatch");
    require(close(
                local.marginal_probability(slice.local_query_qubits, one).value,
                global.marginal_probability(query, one).value),
            "causal index small one probability mismatch");
}

void static_connectivity_is_not_temporal_causality() {
    constexpr std::size_t qubits = 32U;
    const std::vector<Operation> operations = layered_chain(qubits, 5U, false);
    ExactCausalOperationIndex index(qubits, operations);
    const std::vector<QubitId> left_query{0U};
    const ExactCausalSlice left = index.slice(left_query);
    require(left.global_qubits.size() == 6U,
            "causal index failed to preserve the narrow temporal left cone");
    require(left.operations.size() == 16U,
            "causal index left temporal operation count mismatch");

    const std::vector<QubitId> right_query{31U};
    const ExactCausalSlice right = index.slice(right_query);
    require(right.global_qubits.size() == qubits,
            "causal index unexpectedly pruned the right-edge light cone");
    require(right.operations.size() > left.operations.size(),
            "causal index did not distinguish temporal query direction");
}

void untouched_qubit_and_empty_query() {
    const std::vector<Operation> operations{
        {OperationCode::H, 0U},
        {OperationCode::Cnot, 0U, 1U},
    };
    ExactCausalOperationIndex index(4U, operations);

    const std::vector<QubitId> query{3U};
    const ExactCausalSlice untouched = index.slice(query);
    require(untouched.global_qubits.size() == 1U && untouched.global_qubits[0] == 3U,
            "causal index untouched query support mismatch");
    require(untouched.operations.empty(), "causal index retained work for untouched qubit");
    require(untouched.local_query_qubits.size() == 1U && untouched.local_query_qubits[0] == 0U,
            "causal index untouched query remap mismatch");

    const std::vector<QubitId> empty{};
    const ExactCausalSlice identity = index.slice(empty);
    require(identity.global_qubits.empty() && identity.local_query_qubits.empty() &&
                identity.operations.empty(),
            "causal index empty query must have an empty slice");
}

void rejection_cases() {
    bool rejected = false;
    try {
        const std::vector<Operation> operations{{OperationCode::Cnot, 0U, 2U}};
        ExactCausalOperationIndex invalid(2U, operations);
        (void)invalid;
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "causal index accepted an out-of-range dependency");

    rejected = false;
    try {
        const std::vector<Operation> operations{{
            OperationCode::Ry, 0U, 0U, std::numeric_limits<double>::infinity(),
        }};
        ExactCausalOperationIndex invalid(1U, operations);
        (void)invalid;
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "causal index accepted a non-finite rotation");

    rejected = false;
    try {
        const std::vector<Operation> operations{{
            OperationCode::AmplitudeDampingTrajectory, 0U, 0U, 0.1, 0.25,
        }};
        ExactCausalOperationIndex invalid(1U, operations);
        (void)invalid;
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "causal index accepted trajectory semantics");

    rejected = false;
    try {
        const std::vector<Operation> operations{{OperationCode::H, 0U}};
        ExactCausalOperationIndex index(2U, operations);
        const std::vector<QubitId> duplicate{0U, 0U};
        (void)index.slice(duplicate);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "causal index accepted a duplicate query qubit");

    rejected = false;
    try {
        const std::vector<Operation> operations{{OperationCode::H, 0U}};
        ExactCausalOperationIndex index(2U, operations);
        const std::vector<QubitId> outside{2U};
        (void)index.slice(outside);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "causal index accepted an out-of-range query qubit");
}

}  // namespace

int main() {
    exact_small_control();
    static_connectivity_is_not_temporal_causality();
    untouched_qubit_and_empty_query();
    rejection_cases();
    std::cout << "causal index tests passed\n";
    return 0;
}

#include "qubit/qcompiler.hpp"

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

void exact_small_global_control() {
    constexpr std::size_t qubits = 12U;
    constexpr std::size_t layers = 3U;
    const std::vector<Operation> operations = layered_chain(qubits, layers, true);
    const std::vector<QubitId> query{0U};
    const std::vector<std::uint8_t> zero{0U};
    const std::vector<std::uint8_t> one{1U};

    ExactMarginalCompilerPlan compiler(qubits, operations, query);
    ExactPreparedProbabilityPlan global = ExactPreparedProbabilityPlan::for_marginals(
        qubits, operations);

    const double expected_zero = global.marginal_probability(query, zero).value;
    const double expected_one = global.marginal_probability(query, one).value;
    require(close(compiler.probability(zero), expected_zero),
            "marginal compiler zero probability mismatch");
    require(close(compiler.probability(one), expected_one),
            "marginal compiler one probability mismatch");
    require(close(expected_zero + expected_one, 1.0),
            "marginal compiler global control is not normalized");

    require(compiler.stats().original_qubits == qubits,
            "marginal compiler original qubit count mismatch");
    require(compiler.stats().original_operations == operations.size(),
            "marginal compiler original operation count mismatch");
    require(compiler.stats().causal_qubits == 4U,
            "marginal compiler did not reduce the layered chain to four causal qubits");
    require(compiler.stats().causal_operations == 7U,
            "marginal compiler causal operation count mismatch");
    require(compiler.stats().removed_qubits == 8U,
            "marginal compiler removed qubit count mismatch");
    require(compiler.stats().removed_operations == operations.size() - 7U,
            "marginal compiler removed operation count mismatch");
    require(compiler.stats().components == 1U,
            "marginal compiler unexpectedly split a connected causal cone");
    require(compiler.stats().prepared_components == 1U,
            "marginal compiler did not prepare its causal component");
}

void indexed_compiler_matches_one_shot() {
    constexpr std::size_t qubits = 24U;
    const std::vector<Operation> operations = layered_chain(qubits, 6U, true);
    ExactCausalOperationIndex index(qubits, operations);
    const std::vector<std::uint8_t> zero{0U};
    const std::vector<std::uint8_t> one{1U};

    for (const QubitId qubit : std::vector<QubitId>{0U, 3U, 11U}) {
        const std::vector<QubitId> query{qubit};
        ExactMarginalCompilerPlan one_shot(qubits, operations, query);
        ExactIndexedMarginalCompilerPlan indexed(index, query);
        require(indexed.stats().causal_qubits == one_shot.stats().causal_qubits,
                "indexed marginal compiler causal qubit count differs from one-shot compiler");
        require(indexed.stats().causal_operations == one_shot.stats().causal_operations,
                "indexed marginal compiler causal operation count differs from one-shot compiler");
        require(indexed.causal_global_qubits() == one_shot.causal_global_qubits(),
                "indexed marginal compiler causal support differs from one-shot compiler");
        require(close(indexed.probability(zero), one_shot.probability(zero)),
                "indexed marginal compiler zero probability differs from one-shot compiler");
        require(close(indexed.probability(one), one_shot.probability(one)),
                "indexed marginal compiler one probability differs from one-shot compiler");
    }

    const std::vector<QubitId> empty{};
    const std::vector<std::uint8_t> empty_bits{};
    ExactIndexedMarginalCompilerPlan identity(index, empty);
    require(close(identity.probability(empty_bits), 1.0),
            "indexed marginal compiler empty query must equal one");
}

void temporal_cone_beats_static_connectivity() {
    constexpr std::size_t qubits = 32U;
    constexpr std::size_t layers = 5U;
    const std::vector<Operation> operations = layered_chain(qubits, layers, false);
    const std::vector<QubitId> query{0U};
    const std::vector<std::uint8_t> zero{0U};

    ExactMarginalCompilerPlan compiler(qubits, operations, query);
    ExactComponentProbabilityPlan static_fabric = ExactComponentProbabilityPlan::for_marginals(
        qubits, operations, query);

    require(static_fabric.stats().components == 1U,
            "static component control should see the full connected chain");
    require(compiler.stats().causal_qubits == 6U,
            "marginal compiler temporal cone width mismatch");
    require(compiler.stats().causal_operations == 16U,
            "marginal compiler temporal cone operation count mismatch");
    require(compiler.stats().causal_qubits < static_fabric.stats().largest_component_qubits,
            "marginal compiler did not beat static dependency connectivity");
    require(close(compiler.probability(zero), 0.5),
            "marginal compiler layered Clifford probability mismatch");
}

void empty_query_identity() {
    const std::vector<Operation> operations{
        {OperationCode::H, 0U},
        {OperationCode::Cnot, 0U, 1U},
    };
    const std::vector<QubitId> query{};
    const std::vector<std::uint8_t> bits{};
    ExactMarginalCompilerPlan compiler(2U, operations, query);
    require(close(compiler.probability(bits), 1.0),
            "empty marginal compiler query must equal one");
    require(compiler.stats().causal_qubits == 0U,
            "empty marginal compiler query retained qubits");
    require(compiler.stats().causal_operations == 0U,
            "empty marginal compiler query retained operations");
}

void rejection_cases() {
    bool rejected = false;
    try {
        const std::vector<Operation> operations{{OperationCode::H, 0U}};
        const std::vector<QubitId> duplicate{0U, 0U};
        ExactMarginalCompilerPlan invalid(2U, operations, duplicate);
        (void)invalid;
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "marginal compiler accepted a duplicate query qubit");

    rejected = false;
    try {
        const std::vector<Operation> operations{{OperationCode::H, 0U}};
        const std::vector<QubitId> outside{2U};
        ExactMarginalCompilerPlan invalid(2U, operations, outside);
        (void)invalid;
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "marginal compiler accepted an out-of-range query qubit");

    rejected = false;
    try {
        const std::vector<Operation> operations{{OperationCode::Cnot, 0U, 2U}};
        const std::vector<QubitId> query{0U};
        ExactMarginalCompilerPlan invalid(2U, operations, query);
        (void)invalid;
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "marginal compiler accepted an out-of-range operation dependency");

    rejected = false;
    try {
        const std::vector<Operation> operations{{OperationCode::H, 0U}};
        const std::vector<QubitId> query{0U};
        ExactMarginalCompilerPlan compiler(2U, operations, query);
        const std::vector<std::uint8_t> bits{0U, 1U};
        (void)compiler.probability(bits);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "marginal compiler accepted a mismatched query bit count");

    rejected = false;
    try {
        const std::vector<Operation> operations = layered_chain(8U, 3U, false);
        const std::vector<QubitId> query{0U};
        ExactMarginalCompilerConfig config;
        config.max_causal_operations = 2U;
        ExactMarginalCompilerPlan invalid(8U, operations, query, config);
        (void)invalid;
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "marginal compiler ignored its causal operation cap");

    rejected = false;
    try {
        const std::vector<Operation> operations = layered_chain(8U, 3U, false);
        ExactCausalOperationIndex index(8U, operations);
        const std::vector<QubitId> query{0U};
        ExactMarginalCompilerConfig config;
        config.max_causal_operations = 2U;
        ExactIndexedMarginalCompilerPlan invalid(index, query, config);
        (void)invalid;
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "indexed marginal compiler ignored its causal operation cap");

    rejected = false;
    try {
        const std::vector<Operation> operations{
            {OperationCode::H, 0U},
            {OperationCode::AmplitudeDampingTrajectory, 1U, 0U, 0.1, 0.25},
        };
        const std::vector<QubitId> query{0U};
        ExactMarginalCompilerPlan invalid(2U, operations, query);
        (void)invalid;
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected,
            "marginal compiler accepted trajectory semantics outside its exact causal contract");
}

}  // namespace

int main() {
    exact_small_global_control();
    indexed_compiler_matches_one_shot();
    temporal_cone_beats_static_connectivity();
    empty_query_identity();
    rejection_cases();
    std::cout << "exact marginal compiler tests passed\n";
    return 0;
}

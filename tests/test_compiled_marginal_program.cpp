#include "qubit/qprogram.hpp"

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

std::vector<Operation> layered_chain(std::size_t qubits, std::size_t layers) {
    std::vector<Operation> operations;
    operations.reserve(layers * (qubits - 1U) + 1U);
    operations.push_back({OperationCode::Ry, 0U, 0U, 0.37});
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

void exact_cached_queries() {
    constexpr std::size_t qubits = 16U;
    const std::vector<Operation> operations = layered_chain(qubits, 4U);
    ExactCompiledMarginalProgram program(qubits, operations);
    ExactPreparedProbabilityPlan global = ExactPreparedProbabilityPlan::for_marginals(
        qubits, operations);

    const std::vector<QubitId> query0{0U};
    const std::vector<std::uint8_t> zero{0U};
    const std::vector<std::uint8_t> one{1U};
    const ExactIndexedMarginalCompilerPlan& first = program.prepare(query0);
    const ExactIndexedMarginalCompilerPlan& second = program.prepare(query0);
    require(&first == &second, "compiled marginal program did not reuse cached plan identity");
    require(first.stats().causal_qubits == 5U,
            "compiled marginal program unexpected q0 causal width");
    require(first.stats().causal_operations == 9U,
            "compiled marginal program unexpected q0 causal operation count");
    require(close(
                program.probability(query0, zero),
                global.marginal_probability(query0, zero).value),
            "compiled marginal program q0 zero probability mismatch");
    require(close(
                program.probability(query0, one),
                global.marginal_probability(query0, one).value),
            "compiled marginal program q0 one probability mismatch");

    const std::vector<QubitId> query3{3U};
    require(close(
                program.probability(query3, zero),
                global.marginal_probability(query3, zero).value),
            "compiled marginal program q3 zero probability mismatch");

    require(program.stats().compile_count == 2U,
            "compiled marginal program compile count mismatch");
    require(program.stats().cache_misses == 2U,
            "compiled marginal program cache miss count mismatch");
    require(program.stats().cache_hits >= 3U,
            "compiled marginal program did not record repeated query hits");
    require(program.stats().cached_plans == 2U && program.stats().cached_bytes > 0U,
            "compiled marginal program cache accounting mismatch");
}

void query_order_is_bound_to_bits() {
    const std::vector<Operation> operations{
        {OperationCode::Ry, 0U, 0U, 0.31},
        {OperationCode::Ry, 1U, 0U, -0.47},
        {OperationCode::Cnot, 0U, 1U},
    };
    ExactCompiledMarginalProgram program(2U, operations);
    const std::vector<QubitId> forward{0U, 1U};
    const std::vector<QubitId> reverse{1U, 0U};
    const std::vector<std::uint8_t> forward_bits{0U, 1U};
    const std::vector<std::uint8_t> reverse_bits{1U, 0U};
    require(close(
                program.probability(forward, forward_bits),
                program.probability(reverse, reverse_bits)),
            "compiled marginal program did not preserve query-order bit semantics");
    require(program.stats().cached_plans == 2U,
            "compiled marginal program incorrectly aliased differently ordered supports");
}

void explicit_cache_lifecycle() {
    const std::vector<Operation> operations{
        {OperationCode::H, 0U},
        {OperationCode::Cnot, 0U, 1U},
    };
    ExactCompiledMarginalProgram program(4U, operations);
    const std::vector<QubitId> query{0U};
    const std::vector<std::uint8_t> bits{0U};
    (void)program.probability(query, bits);
    require(program.stats().cached_plans == 1U,
            "compiled marginal program failed to retain prepared plan");
    const std::size_t compile_count = program.stats().compile_count;
    program.clear_cache();
    require(program.stats().cached_plans == 0U && program.stats().cached_bytes == 0U,
            "compiled marginal program cache clear did not release accounting");
    (void)program.probability(query, bits);
    require(program.stats().compile_count == compile_count + 1U,
            "compiled marginal program did not rebuild after explicit cache clear");
}

void rejection_cases() {
    bool rejected = false;
    try {
        const std::vector<Operation> operations{{OperationCode::H, 0U}};
        ExactCompiledMarginalProgramConfig config;
        config.max_cached_plans = 1U;
        ExactCompiledMarginalProgram program(2U, operations, config);
        const std::vector<QubitId> q0{0U};
        const std::vector<QubitId> q1{1U};
        (void)program.prepare(q0);
        (void)program.prepare(q1);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "compiled marginal program ignored its cached-plan cap");

    rejected = false;
    try {
        const std::vector<Operation> operations{{OperationCode::H, 0U}};
        ExactCompiledMarginalProgramConfig config;
        config.max_cached_bytes = 1U;
        ExactCompiledMarginalProgram program(2U, operations, config);
        const std::vector<QubitId> q0{0U};
        (void)program.prepare(q0);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "compiled marginal program ignored its cached-byte cap");

    rejected = false;
    try {
        const std::vector<Operation> operations{{OperationCode::H, 0U}};
        ExactCompiledMarginalProgram program(2U, operations);
        const std::vector<QubitId> duplicate{0U, 0U};
        (void)program.prepare(duplicate);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "compiled marginal program accepted duplicate query qubits");

    rejected = false;
    try {
        const std::vector<Operation> operations{{
            OperationCode::AmplitudeDampingTrajectory, 0U, 0U, 0.1, 0.25,
        }};
        ExactCompiledMarginalProgram program(1U, operations);
        (void)program;
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "compiled marginal program accepted trajectory semantics");
}

}  // namespace

int main() {
    exact_cached_queries();
    query_order_is_bound_to_bits();
    explicit_cache_lifecycle();
    rejection_cases();
    std::cout << "compiled marginal program tests passed\n";
    return 0;
}

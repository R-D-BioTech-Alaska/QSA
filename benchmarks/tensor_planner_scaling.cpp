#include "qubit/qplan.hpp"
#include "qubit/qtensor.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::Operation;
using qubit::OperationCode;
using qubit::QubitId;
using qubit::TensorContractionPlan;
using qubit::TensorNetworkCircuit;
using qubit::TensorNetworkConfig;

[[nodiscard]] std::vector<Operation> chain_operations(std::size_t qubits) {
    std::vector<Operation> operations;
    operations.reserve(2U * qubits + 2U);
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({
            OperationCode::H,
            static_cast<QubitId>(qubit),
            0U,
            0.0,
            0.0,
        });
    }
    for (std::size_t qubit = 0U; qubit + 1U < qubits; ++qubit) {
        operations.push_back({
            OperationCode::Cz,
            static_cast<QubitId>(qubit),
            static_cast<QubitId>(qubit + 1U),
            0.0,
            0.0,
        });
    }
    operations.push_back({
        OperationCode::Rz,
        static_cast<QubitId>(qubits / 2U),
        0U,
        0.37,
        0.0,
    });
    operations.push_back({
        OperationCode::Ry,
        static_cast<QubitId>(qubits / 2U),
        0U,
        0.19,
        0.0,
    });
    return operations;
}

struct Result {
    double circuit_ms{0.0};
    double compile_ms{0.0};
    std::size_t operations{0U};
    std::size_t factors{0U};
    std::size_t steps{0U};
    std::size_t peak_entries{0U};
    std::size_t plan_bytes{0U};
};

[[nodiscard]] Result measure(std::size_t qubits) {
    const std::vector<Operation> operations = chain_operations(qubits);
    TensorNetworkConfig config;
    config.max_contraction_entries = 16U;
    config.max_factors = 1'000'000U;

    std::optional<TensorNetworkCircuit> circuit;
    const auto circuit_start = Clock::now();
    circuit.emplace(qubits, operations, config);
    const auto circuit_stop = Clock::now();

    std::optional<TensorContractionPlan> plan;
    const auto compile_start = Clock::now();
    plan.emplace(circuit->compile());
    const auto compile_stop = Clock::now();

    Result result;
    result.circuit_ms = std::chrono::duration<double, std::milli>(
        circuit_stop - circuit_start).count();
    result.compile_ms = std::chrono::duration<double, std::milli>(
        compile_stop - compile_start).count();
    result.operations = operations.size();
    result.factors = circuit->factor_count();
    result.steps = plan->step_count();
    result.peak_entries = plan->stats().peak_contraction_entries;
    result.plan_bytes = plan->estimated_bytes();
    return result;
}

[[nodiscard]] bool certified(std::size_t qubits, const Result& result) noexcept {
    return result.operations == 2U * qubits + 1U &&
           result.factors == 3U * qubits + 1U &&
           result.steps == 4U * qubits &&
           result.peak_entries == 16U;
}

void print(std::size_t qubits, const Result& result) {
    std::cout << "tensor_planner_qubits=" << qubits << '\n';
    std::cout << "tensor_planner_operations=" << result.operations << '\n';
    std::cout << "tensor_planner_factors=" << result.factors << '\n';
    std::cout << "tensor_planner_steps=" << result.steps << '\n';
    std::cout << "tensor_planner_peak_entries=" << result.peak_entries << '\n';
    std::cout << "tensor_planner_circuit_ms=" << result.circuit_ms << '\n';
    std::cout << "tensor_planner_compile_ms=" << result.compile_ms << '\n';
    std::cout << "tensor_planner_plan_bytes=" << result.plan_bytes << '\n';
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);
    const Result small = measure(1'000U);
    const Result medium = measure(5'000U);
    const Result large = measure(30'000U);
    if (!certified(1'000U, small) ||
        !certified(5'000U, medium) ||
        !certified(30'000U, large)) {
        std::cerr << "tensor planner scaling certificate changed\n";
        return 1;
    }
    print(1'000U, small);
    print(5'000U, medium);
    print(30'000U, large);
    std::cout << "tensor_planner_compile_growth_5k_over_1k="
              << medium.compile_ms / small.compile_ms << '\n';
    std::cout << "tensor_planner_compile_growth_30k_over_5k="
              << large.compile_ms / medium.compile_ms << '\n';
    return 0;
}

#include "qubit/qcompiler.hpp"
#include "qubit/qrepresentation_compiler.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace qubit;

int main() {
    constexpr std::size_t qubits = 10000U;
    constexpr std::size_t layers = 10U;
    constexpr std::size_t repeats = 1000U;

    std::vector<Operation> operations;
    operations.reserve(99998U);
    for (std::size_t repeat = 0U; repeat < 8U; ++repeat) {
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
    if (operations.size() != 99998U) {
        throw std::runtime_error("10k carrier operation count changed");
    }

    const std::vector<QubitId> query{0U};
    const std::vector<std::uint8_t> bits{0U};

    const auto index_start = std::chrono::steady_clock::now();
    ExactCausalOperationIndex index(qubits, operations);
    const auto index_end = std::chrono::steady_clock::now();
    const auto control_start = std::chrono::steady_clock::now();
    ExactIndexedMarginalCompilerPlan control(index, query);
    const auto control_end = std::chrono::steady_clock::now();

    const auto fabric_build_start = std::chrono::steady_clock::now();
    ExactRepresentationFabric fabric(qubits);
    for (const Operation& operation : operations) {
        fabric.append(operation);
    }
    const auto fabric_build_end = std::chrono::steady_clock::now();

    const auto cold_start = std::chrono::steady_clock::now();
    const ExactRepresentationProbabilityResult cold = fabric.marginal_probability(query, bits);
    const auto cold_end = std::chrono::steady_clock::now();
    if (cold.islands.size() != 1U) {
        throw std::runtime_error("10k fabric query did not isolate one exact island");
    }

    ExactRepresentationProbabilityResult warm;
    const auto warm_start = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0U; repeat < repeats; ++repeat) {
        warm = fabric.marginal_probability(query, bits);
    }
    const auto warm_end = std::chrono::steady_clock::now();

    double control_probability = 0.0;
    const auto control_query_start = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0U; repeat < repeats; ++repeat) {
        control_probability = control.probability(bits);
    }
    const auto control_query_end = std::chrono::steady_clock::now();

    const double error = std::abs(warm.value - control_probability);
    if (error > 2e-11 || std::abs(cold.value - control_probability) > 2e-11) {
        throw std::runtime_error("10k persistent fabric differs from exact indexed control");
    }

    const auto& island = cold.islands.front();
    const double index_ms = std::chrono::duration<double, std::milli>(index_end - index_start).count();
    const double control_setup_ms =
        std::chrono::duration<double, std::milli>(control_end - control_start).count();
    const double fabric_build_ms =
        std::chrono::duration<double, std::milli>(fabric_build_end - fabric_build_start).count();
    const double fabric_cold_ms =
        std::chrono::duration<double, std::milli>(cold_end - cold_start).count();
    const double fabric_warm_us =
        std::chrono::duration<double, std::micro>(warm_end - warm_start).count() /
        static_cast<double>(repeats);
    const double control_query_us =
        std::chrono::duration<double, std::micro>(control_query_end - control_query_start).count() /
        static_cast<double>(repeats);

    std::cout << std::setprecision(17);
    std::cout << "carrier_qubits=" << qubits << '\n';
    std::cout << "carrier_operations=" << operations.size() << '\n';
    std::cout << "index_build_ms=" << index_ms << '\n';
    std::cout << "control_setup_ms=" << control_setup_ms << '\n';
    std::cout << "control_query_us=" << control_query_us << '\n';
    std::cout << "fabric_build_ms=" << fabric_build_ms << '\n';
    std::cout << "fabric_cold_query_ms=" << fabric_cold_ms << '\n';
    std::cout << "fabric_total_cold_ms=" << fabric_build_ms + fabric_cold_ms << '\n';
    std::cout << "fabric_warm_query_us=" << fabric_warm_us << '\n';
    std::cout << "fabric_probability_error=" << error << '\n';
    std::cout << "fabric_active_components=" << fabric.stats().active_components << '\n';
    std::cout << "fabric_largest_component_qubits=" << fabric.stats().largest_component_qubits << '\n';
    std::cout << "fabric_causal_qubits=" << island.causal_qubits << '\n';
    std::cout << "fabric_causal_operations=" << island.causal_operations << '\n';
    std::cout << "fabric_execution=" << static_cast<unsigned>(island.execution) << '\n';
    std::cout << "fabric_local_dense_eligible=" << (island.local_dense_eligible ? 1 : 0) << '\n';
    std::cout << "fabric_local_dense_state_scalars=" << island.local_dense_state_scalars << '\n';
    std::cout << "fabric_local_dense_payload_bytes=" << island.local_dense_payload_bytes << '\n';
    std::cout << "fabric_local_dense_scalar_sweep_units=" << island.local_dense_scalar_sweep_units << '\n';
    std::cout << "fabric_estimated_executor_bytes=" << island.estimated_executor_bytes << '\n';
    std::cout << "fabric_warm_cache_hits=" << warm.receipt.cache_hits << '\n';
    std::cout << "fabric_warm_cache_misses=" << warm.receipt.cache_misses << '\n';
    return 0;
}

#include "qubit/qtensor.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::Operation;
using qubit::OperationCode;
using qubit::QComplex;
using qubit::QubitId;
using qubit::TensorContractionPlan;
using qubit::TensorContractionWorkspace;
using qubit::TensorNetworkCircuit;
using qubit::TensorNetworkConfig;

template <typename Function>
double milliseconds(Function&& function) {
    const auto start = Clock::now();
    function();
    const auto finish = Clock::now();
    return std::chrono::duration<double, std::milli>(finish - start).count();
}

std::vector<Operation> brickwork(std::size_t qubits, std::size_t layers) {
    std::vector<Operation> operations;
    operations.reserve(qubits * 2U + layers * qubits / 2U);
    for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
        operations.push_back({
            OperationCode::Ry,
            static_cast<QubitId>(qubit),
            0U,
            0.004 * static_cast<double>(1U + qubit % 19U),
            0.0,
        });
        operations.push_back({
            OperationCode::Rz,
            static_cast<QubitId>(qubit),
            0U,
            -0.003 * static_cast<double>(1U + qubit % 13U),
            0.0,
        });
    }
    for (std::size_t layer = 0; layer < layers; ++layer) {
        const std::size_t offset = layer & 1U;
        for (std::size_t qubit = offset; qubit + 1U < qubits; qubit += 2U) {
            operations.push_back({
                OperationCode::Cnot,
                static_cast<QubitId>(qubit),
                static_cast<QubitId>(qubit + 1U),
                0.0,
                0.0,
            });
        }
    }
    return operations;
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);

    {
        constexpr std::size_t qubits = 18U;
        constexpr std::size_t query_count = 16U;
        TensorNetworkCircuit tensor(qubits);
        tensor.apply(brickwork(qubits, 5U));

        TensorContractionPlan plan = tensor.compile();
        double compile_ms = milliseconds([&] { plan = tensor.compile(); });
        TensorContractionWorkspace workspace = plan.workspace();
        QComplex direct_checksum{};
        QComplex compiled_checksum{};
        const std::uint64_t seed = 0x15555U;

        const double direct_ms = milliseconds([&] {
            for (std::size_t index = 0; index < query_count; ++index) {
                const std::uint64_t basis =
                    (seed + index * 4051U) & ((std::uint64_t{1} << qubits) - 1U);
                direct_checksum += tensor.amplitude(basis);
            }
        });
        const double compiled_ms = milliseconds([&] {
            for (std::size_t index = 0; index < query_count; ++index) {
                const std::uint64_t basis =
                    (seed + index * 4051U) & ((std::uint64_t{1} << qubits) - 1U);
                compiled_checksum += plan.amplitude(basis, workspace);
            }
        });

        std::cout << "small_qubits=" << qubits << '\n';
        std::cout << "small_queries=" << query_count << '\n';
        std::cout << "small_compile_ms=" << compile_ms << '\n';
        std::cout << "small_direct_query_total_ms=" << direct_ms << '\n';
        std::cout << "small_compiled_query_total_ms=" << compiled_ms << '\n';
        std::cout << "small_compiled_query_speedup=" << direct_ms / compiled_ms << '\n';
        std::cout << "small_error=" << (direct_checksum - compiled_checksum).magnitude() << '\n';
        std::cout << "small_plan_bytes=" << plan.estimated_bytes() << '\n';
        std::cout << "small_workspace_bytes=" << workspace.estimated_bytes() << '\n';
    }

    {
        constexpr std::size_t qubits = 100U;
        constexpr std::size_t query_count = 2U;
        TensorNetworkConfig config;
        config.max_contraction_entries = 1U << 16U;
        TensorNetworkCircuit tensor(qubits, config);
        tensor.apply(brickwork(qubits, 6U));

        TensorContractionPlan plan = tensor.compile();
        double compile_ms = milliseconds([&] { plan = tensor.compile(); });
        TensorContractionWorkspace workspace = plan.workspace();
        std::vector<std::uint8_t> bits(qubits, 0U);

        const double direct_ms = milliseconds([&] {
            for (std::size_t index = 0; index < query_count; ++index) {
                bits[index] ^= 1U;
                static_cast<void>(tensor.amplitude(bits));
            }
        });
        std::fill(bits.begin(), bits.end(), 0U);
        const double compiled_ms = milliseconds([&] {
            for (std::size_t index = 0; index < query_count; ++index) {
                bits[index] ^= 1U;
                static_cast<void>(plan.amplitude(bits, workspace));
            }
        });

        std::cout << "large_qubits=" << qubits << '\n';
        std::cout << "large_queries=" << query_count << '\n';
        std::cout << "large_compile_ms=" << compile_ms << '\n';
        std::cout << "large_direct_query_total_ms=" << direct_ms << '\n';
        std::cout << "large_compiled_query_total_ms=" << compiled_ms << '\n';
        std::cout << "large_compiled_query_speedup=" << direct_ms / compiled_ms << '\n';
        std::cout << "large_peak_entries=" << plan.stats().peak_contraction_entries << '\n';
        std::cout << "large_peak_variables=" << plan.stats().peak_union_variables << '\n';
        std::cout << "large_plan_bytes=" << plan.estimated_bytes() << '\n';
        std::cout << "large_workspace_bytes=" << workspace.estimated_bytes() << '\n';
    }

    return 0;
}

#include "qubit/qtensor.hpp"

#include <chrono>
#include <cmath>
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
using qubit::QRegister;
using qubit::QubitId;
using qubit::TensorContractionStats;
using qubit::TensorNetworkCircuit;
using qubit::TensorNetworkConfig;

void apply(QRegister& state, const Operation& operation) {
    switch (operation.code) {
        case OperationCode::Ry:
            state.apply_ry(operation.first, operation.parameter);
            break;
        case OperationCode::Rz:
            state.apply_rz(operation.first, operation.parameter);
            break;
        case OperationCode::Cnot:
            state.apply_cnot(operation.first, operation.second);
            break;
        default:
            throw std::runtime_error("benchmark operation is unsupported");
    }
}

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
        const std::vector<Operation> operations = brickwork(qubits, 5U);
        QRegister reference(qubits);
        TensorNetworkCircuit tensor(qubits);

        const double reference_setup_ms = milliseconds([&] {
            for (const Operation& operation : operations) {
                apply(reference, operation);
            }
        });
        const double tensor_setup_ms = milliseconds([&] {
            tensor.apply(operations);
        });

        const std::uint64_t basis = 0x15555U;
        QComplex reference_value{};
        QComplex tensor_value{};
        TensorContractionStats stats;
        const double reference_query_ms = milliseconds([&] {
            reference_value = reference.amplitude(basis);
        });
        const double tensor_query_ms = milliseconds([&] {
            tensor_value = tensor.amplitude(basis, &stats);
        });

        std::cout << "small_qubits=" << qubits << '\n';
        std::cout << "small_operations=" << operations.size() << '\n';
        std::cout << "qregister_setup_ms=" << reference_setup_ms << '\n';
        std::cout << "tensor_setup_ms=" << tensor_setup_ms << '\n';
        std::cout << "qregister_query_ms=" << reference_query_ms << '\n';
        std::cout << "tensor_query_ms=" << tensor_query_ms << '\n';
        std::cout << "small_amplitude_error="
                  << (reference_value - tensor_value).magnitude() << '\n';
        std::cout << "small_tensor_peak_entries=" << stats.peak_contraction_entries << '\n';
        std::cout << "small_tensor_peak_variables=" << stats.peak_union_variables << '\n';
        std::cout << "qregister_bytes=" << reference.estimated_bytes() << '\n';
        std::cout << "tensor_bytes=" << tensor.estimated_bytes() << '\n';
    }

    {
        constexpr std::size_t qubits = 100U;
        const std::vector<Operation> operations = brickwork(qubits, 6U);
        TensorNetworkConfig config;
        config.max_contraction_entries = 1U << 16U;
        TensorNetworkCircuit tensor(qubits, config);
        const double setup_ms = milliseconds([&] {
            tensor.apply(operations);
        });
        std::vector<std::uint8_t> bits(qubits, 0U);
        TensorContractionStats stats;
        QComplex amplitude{};
        const double query_ms = milliseconds([&] {
            amplitude = tensor.amplitude(bits, &stats);
        });

        std::cout << "large_qubits=" << qubits << '\n';
        std::cout << "large_operations=" << operations.size() << '\n';
        std::cout << "large_setup_ms=" << setup_ms << '\n';
        std::cout << "large_query_ms=" << query_ms << '\n';
        std::cout << "large_peak_entries=" << stats.peak_contraction_entries << '\n';
        std::cout << "large_peak_variables=" << stats.peak_union_variables << '\n';
        std::cout << "large_eliminated_variables=" << stats.eliminated_variables << '\n';
        std::cout << "large_tensor_bytes=" << tensor.estimated_bytes() << '\n';
        std::cout << "large_amplitude_real=" << amplitude.re << '\n';
        std::cout << "large_amplitude_imag=" << amplitude.im << '\n';
    }

    return 0;
}

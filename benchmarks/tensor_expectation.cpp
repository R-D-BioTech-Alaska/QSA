#include "qubit/qtensor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::Operation;
using qubit::OperationCode;
using qubit::OperationPlan;
using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
using qubit::QComplex;
using qubit::QRegister;
using qubit::TensorContractionStats;
using qubit::TensorNetworkCircuit;
using qubit::TensorNetworkConfig;

template <typename Function>
[[nodiscard]] double milliseconds(Function&& function) {
    const auto start = Clock::now();
    function();
    const auto finish = Clock::now();
    return std::chrono::duration<double, std::milli>(finish - start).count();
}

[[nodiscard]] std::vector<Operation> brickwork(std::size_t qubits, std::size_t layers) {
    std::vector<Operation> operations;
    operations.reserve(layers * (qubits * 2U + qubits / 2U));
    for (std::size_t layer = 0; layer < layers; ++layer) {
        for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
            operations.push_back({
                OperationCode::Ry,
                static_cast<qubit::QubitId>(qubit),
                0U,
                0.007 * static_cast<double>((layer + 1U) * (qubit + 3U)),
                0.0,
            });
            operations.push_back({
                OperationCode::Rz,
                static_cast<qubit::QubitId>(qubit),
                0U,
                -0.005 * static_cast<double>((layer + 2U) * (qubit + 1U)),
                0.0,
            });
        }
        const std::size_t start = layer & 1U;
        for (std::size_t qubit = start; qubit + 1U < qubits; qubit += 2U) {
            operations.push_back({
                OperationCode::Cnot,
                static_cast<qubit::QubitId>(qubit),
                static_cast<qubit::QubitId>(qubit + 1U),
                0.0,
                0.0,
            });
        }
    }
    return operations;
}

[[nodiscard]] std::vector<PauliObservable> observables(std::size_t qubits, std::size_t count) {
    std::vector<PauliObservable> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        PauliObservable observable(qubits);
        const std::size_t first = (index * 7U + 3U) % qubits;
        const std::size_t second = (first + 1U + index % 3U) % qubits;
        const std::size_t third = (second + 2U) % qubits;
        std::vector<PauliFactor> factors{
            {static_cast<qubit::QubitId>(first), static_cast<PauliAxis>(1U + index % 3U)},
            {static_cast<qubit::QubitId>(second), static_cast<PauliAxis>(1U + (index + 1U) % 3U)},
        };
        if (index % 2U != 0U && third != first && third != second) {
            factors.push_back({
                static_cast<qubit::QubitId>(third),
                static_cast<PauliAxis>(1U + (index + 2U) % 3U),
            });
        }
        observable.add_term({1.0, 0.0}, factors);
        result.push_back(std::move(observable));
    }
    return result;
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);

    {
        constexpr std::size_t qubits = 18U;
        const std::vector<Operation> operations = brickwork(qubits, 5U);
        const auto queries = observables(qubits, 24U);

        QRegister state(qubits);
        const OperationPlan qregister_plan(operations, false);
        const double qregister_setup_ms = milliseconds([&] { qregister_plan.execute(state); });
        std::vector<QComplex> qregister_values(queries.size());
        const double qregister_query_ms = milliseconds([&] {
            for (std::size_t index = 0; index < queries.size(); ++index) {
                qregister_values[index] = queries[index].expectation(state);
            }
        });

        TensorNetworkCircuit circuit(
            qubits, operations, TensorNetworkConfig{1U << 20U, 1'000'000U});
        qubit::TensorExpectationPlan tensor_plan(circuit, queries.front());
        const double tensor_compile_ms = milliseconds([&] {
            tensor_plan = qubit::TensorExpectationPlan(
                circuit,
                std::span<const PauliObservable>(queries.data(), queries.size()));
        });
        auto tensor_workspace = tensor_plan.workspace();
        std::vector<QComplex> tensor_values(queries.size());
        const double tensor_query_ms = milliseconds([&] {
            tensor_plan.expectations(tensor_values, tensor_workspace);
        });

        double max_error = 0.0;
        for (std::size_t index = 0; index < queries.size(); ++index) {
            max_error = std::max(
                max_error,
                (qregister_values[index] - tensor_values[index]).magnitude());
        }
        const std::size_t plan_bytes = tensor_plan.estimated_bytes();
        const std::size_t workspace_bytes = tensor_workspace.estimated_bytes();
        const std::size_t peak_entries = tensor_plan.stats().peak_contraction_entries;
        const std::size_t peak_variables = tensor_plan.stats().peak_union_variables;
        std::cout << "expect18_qubits=" << qubits << '\n';
        std::cout << "expect18_operations=" << operations.size() << '\n';
        std::cout << "expect18_queries=" << queries.size() << '\n';
        std::cout << "expect18_qregister_setup_ms=" << qregister_setup_ms << '\n';
        std::cout << "expect18_qregister_query_ms=" << qregister_query_ms << '\n';
        std::cout << "expect18_tensor_compile_ms=" << tensor_compile_ms << '\n';
        std::cout << "expect18_tensor_query_ms=" << tensor_query_ms << '\n';
        std::cout << "expect18_query_ratio=" << qregister_query_ms / tensor_query_ms << '\n';
        std::cout << "expect18_max_error=" << max_error << '\n';
        std::cout << "expect18_qregister_bytes=" << state.estimated_bytes() << '\n';
        std::cout << "expect18_plan_bytes=" << plan_bytes << '\n';
        std::cout << "expect18_workspace_bytes=" << workspace_bytes << '\n';
        std::cout << "expect18_source_factors=" << tensor_plan.stats().source_factors << '\n';
        std::cout << "expect18_peak_entries=" << peak_entries << '\n';
        std::cout << "expect18_peak_variables=" << peak_variables << '\n';
        std::cout << "expect18_end_to_end_ratio="
                  << (qregister_setup_ms + qregister_query_ms) /
                         (tensor_compile_ms + tensor_query_ms)
                  << '\n';
    }

    {
        constexpr std::size_t qubits = 100U;
        const std::vector<Operation> operations = brickwork(qubits, 5U);
        const auto queries = observables(qubits, 8U);
        TensorNetworkCircuit circuit(
            qubits, operations, TensorNetworkConfig{1U << 20U, 1'000'000U});
        qubit::TensorExpectationPlan tensor_plan(circuit, queries.front());
        const double compile_ms = milliseconds([&] {
            tensor_plan = qubit::TensorExpectationPlan(
                circuit,
                std::span<const PauliObservable>(queries.data(), queries.size()));
        });
        auto tensor_workspace = tensor_plan.workspace();
        std::vector<QComplex> values(queries.size());
        QComplex checksum{};
        const double query_ms = milliseconds([&] {
            tensor_plan.expectations(values, tensor_workspace);
            checksum = {};
            for (const QComplex value : values) {
                checksum += value;
            }
        });
        const std::size_t plan_bytes = tensor_plan.estimated_bytes();
        const std::size_t workspace_bytes = tensor_workspace.estimated_bytes();
        const std::size_t peak_entries = tensor_plan.stats().peak_contraction_entries;
        const std::size_t peak_variables = tensor_plan.stats().peak_union_variables;
        std::cout << "expect100_qubits=" << qubits << '\n';
        std::cout << "expect100_operations=" << operations.size() << '\n';
        std::cout << "expect100_queries=" << queries.size() << '\n';
        std::cout << "expect100_compile_ms=" << compile_ms << '\n';
        std::cout << "expect100_query_ms=" << query_ms << '\n';
        std::cout << "expect100_plan_bytes=" << plan_bytes << '\n';
        std::cout << "expect100_workspace_bytes=" << workspace_bytes << '\n';
        std::cout << "expect100_source_factors=" << tensor_plan.stats().source_factors << '\n';
        std::cout << "expect100_peak_entries=" << peak_entries << '\n';
        std::cout << "expect100_peak_variables=" << peak_variables << '\n';
        std::cout << "expect100_checksum_real=" << checksum.re << '\n';
        std::cout << "expect100_checksum_imag=" << checksum.im << '\n';
    }

    return 0;
}

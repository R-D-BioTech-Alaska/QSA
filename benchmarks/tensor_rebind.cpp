#include "qubit/qtensor_rebind.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::Operation;
using qubit::OperationCode;
using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
using qubit::PauliPropagationConfig;
using qubit::QComplex;
using qubit::TensorExpectationPlan;
using qubit::TensorExpectationRebindPlan;
using qubit::TensorNetworkCircuit;


template <typename Function>
[[nodiscard]] double milliseconds(Function&& function) {
    const auto start = Clock::now();
    function();
    const auto finish = Clock::now();
    return std::chrono::duration<double, std::milli>(finish - start).count();
}

std::vector<Operation> brickwork(
    std::size_t qubits,
    std::size_t layers,
    double scale) {
    std::vector<Operation> operations;
    operations.reserve(layers * (qubits * 2U + qubits / 2U));
    for (std::size_t layer = 0; layer < layers; ++layer) {
        for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
            operations.push_back({
                OperationCode::Ry,
                static_cast<qubit::QubitId>(qubit),
                0U,
                scale * 0.007 * static_cast<double>((layer + 1U) * (qubit + 3U)),
                0.0,
            });
            operations.push_back({
                OperationCode::Rz,
                static_cast<qubit::QubitId>(qubit),
                0U,
                -scale * 0.005 * static_cast<double>((layer + 2U) * (qubit + 1U)),
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

std::vector<PauliObservable> observables(std::size_t qubits, std::size_t count) {
    std::vector<PauliObservable> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        PauliObservable observable(qubits, PauliPropagationConfig{65'536U});
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

void run_case(
    std::size_t qubits,
    std::size_t query_count,
    std::size_t sweeps) {
    const std::vector<PauliObservable> queries = observables(qubits, query_count);
    const TensorNetworkCircuit initial(
        qubits,
        brickwork(qubits, 5U, 1.0));
    TensorExpectationRebindPlan reusable(initial, queries);
    auto reusable_workspace = reusable.workspace();
    std::vector<QComplex> reusable_values(query_count);
    std::vector<QComplex> fresh_values(query_count);

    double fresh_compile_ms = 0.0;
    double fresh_query_ms = 0.0;
    double rebind_ms = 0.0;
    double rebind_query_ms = 0.0;
    double max_error = 0.0;

    for (std::size_t sweep = 0; sweep < sweeps; ++sweep) {
        const double scale = 0.55 + 0.073 * static_cast<double>(sweep + 1U);
        const TensorNetworkCircuit circuit(
            qubits,
            brickwork(qubits, 5U, scale));

        std::unique_ptr<TensorExpectationPlan> fresh;
        fresh_compile_ms += milliseconds([&] {
            fresh = std::make_unique<TensorExpectationPlan>(circuit, queries);
        });
        auto fresh_workspace = fresh->workspace();
        fresh_query_ms += milliseconds([&] {
            fresh->expectations(fresh_values, fresh_workspace);
        });

        rebind_ms += milliseconds([&] {
            reusable.rebind(circuit);
        });
        rebind_query_ms += milliseconds([&] {
            reusable.expectations(reusable_values, reusable_workspace);
        });

        for (std::size_t index = 0; index < query_count; ++index) {
            max_error = std::max(
                max_error,
                (fresh_values[index] - reusable_values[index]).magnitude());
        }
    }

    const std::string prefix = "rebind" + std::to_string(qubits);
    std::cout << prefix << "_qubits=" << qubits << '\n';
    std::cout << prefix << "_queries=" << query_count << '\n';
    std::cout << prefix << "_sweeps=" << sweeps << '\n';
    std::cout << prefix << "_fresh_compile_total_ms=" << fresh_compile_ms << '\n';
    std::cout << prefix << "_rebind_total_ms=" << rebind_ms << '\n';
    std::cout << prefix << "_compile_speedup=" << fresh_compile_ms / rebind_ms << '\n';
    std::cout << prefix << "_fresh_query_total_ms=" << fresh_query_ms << '\n';
    std::cout << prefix << "_rebind_query_total_ms=" << rebind_query_ms << '\n';
    std::cout << prefix << "_fresh_total_ms=" << fresh_compile_ms + fresh_query_ms << '\n';
    std::cout << prefix << "_reused_total_ms=" << rebind_ms + rebind_query_ms << '\n';
    std::cout << prefix << "_total_speedup="
              << (fresh_compile_ms + fresh_query_ms) /
                     (rebind_ms + rebind_query_ms)
              << '\n';
    std::cout << prefix << "_max_error=" << max_error << '\n';
    std::cout << prefix << "_rebind_count=" << reusable.rebind_count() << '\n';
    std::cout << prefix << "_plan_bytes=" << reusable.estimated_bytes() << '\n';
    std::cout << prefix << "_workspace_bytes="
              << reusable_workspace.estimated_bytes() << '\n';
    std::cout << prefix << "_peak_entries="
              << reusable.stats().peak_contraction_entries << '\n';
    std::cout << prefix << "_peak_variables="
              << reusable.stats().peak_union_variables << '\n';
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);
    run_case(18U, 24U, 12U);
    run_case(100U, 8U, 8U);
    return 0;
}

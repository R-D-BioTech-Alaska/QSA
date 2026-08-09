#include "qubit/qtensor_adjoint_causal.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::ExactAdjointGradientConfig;
using qubit::ExactCausalAdjointGradientConfig;
using qubit::ExactCausalAdjointGradientPlan;
using qubit::OperationCode;
using qubit::ParameterizedOperation;
using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
using qubit::PauliPropagationConfig;
using qubit::QComplex;
using qubit::TensorNetworkConfig;

template <typename Function>
[[nodiscard]] double milliseconds(Function&& function) {
    const auto start = Clock::now();
    function();
    const auto finish = Clock::now();
    return std::chrono::duration<double, std::milli>(finish - start).count();
}

[[nodiscard]] std::vector<ParameterizedOperation> parameterized_brickwork(
    std::size_t qubits,
    std::size_t layers,
    std::size_t parameter_count) {
    std::vector<ParameterizedOperation> operations;
    operations.reserve(layers * (qubits * 2U + qubits / 2U));
    std::size_t next_parameter = 0U;
    for (std::size_t layer = 0U; layer < layers; ++layer) {
        for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
            ParameterizedOperation ry;
            ry.operation = {
                OperationCode::Ry,
                static_cast<qubit::QubitId>(qubit),
                0U,
                0.007 * static_cast<double>((layer + 1U) * (qubit + 3U)),
                0.0,
            };
            if (next_parameter < parameter_count) {
                ry.parameter_slot = static_cast<std::int32_t>(next_parameter++);
            }
            operations.push_back(ry);

            ParameterizedOperation rz;
            rz.operation = {
                OperationCode::Rz,
                static_cast<qubit::QubitId>(qubit),
                0U,
                -0.005 * static_cast<double>((layer + 2U) * (qubit + 1U)),
                0.0,
            };
            if (next_parameter < parameter_count) {
                rz.parameter_slot = static_cast<std::int32_t>(next_parameter++);
            }
            operations.push_back(rz);
        }
        const std::size_t start = layer & 1U;
        for (std::size_t qubit = start; qubit + 1U < qubits; qubit += 2U) {
            ParameterizedOperation cnot;
            cnot.operation = {
                OperationCode::Cnot,
                static_cast<qubit::QubitId>(qubit),
                static_cast<qubit::QubitId>(qubit + 1U),
                0.0,
                0.0,
            };
            operations.push_back(cnot);
        }
    }
    return operations;
}

[[nodiscard]] std::vector<PauliObservable> observables(
    std::size_t qubits,
    std::size_t count) {
    std::vector<PauliObservable> result;
    result.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        PauliObservable observable(qubits, PauliPropagationConfig{65'536U});
        const std::size_t first = (index * 7U + 3U) % qubits;
        const std::size_t second = (first + 1U + index % 3U) % qubits;
        const std::size_t third = (second + 2U) % qubits;
        std::vector<PauliFactor> factors{
            {static_cast<qubit::QubitId>(first),
             static_cast<PauliAxis>(1U + index % 3U)},
            {static_cast<qubit::QubitId>(second),
             static_cast<PauliAxis>(1U + (index + 1U) % 3U)},
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

[[nodiscard]] std::vector<double> parameter_point(std::size_t parameter_count) {
    std::vector<double> result(parameter_count);
    for (std::size_t parameter = 0U; parameter < parameter_count; ++parameter) {
        result[parameter] =
            -0.47 + 0.059 + 0.031 * static_cast<double>(parameter + 1U);
    }
    return result;
}

void run_case(
    std::size_t qubits,
    std::size_t query_count,
    std::size_t parameter_count) {
    constexpr std::size_t repetitions = 3U;
    const TensorNetworkConfig tensor{1U << 16U, 1'000'000U};

    std::vector<ParameterizedOperation> operations;
    std::vector<PauliObservable> queries;
    std::vector<double> parameters;
    const double workload_build_ms = milliseconds([&] {
        operations = parameterized_brickwork(qubits, 5U, parameter_count);
        queries = observables(qubits, query_count);
        parameters = parameter_point(parameter_count);
    });

    std::unique_ptr<ExactCausalAdjointGradientPlan> plan;
    const double compile_ms = milliseconds([&] {
        plan = std::make_unique<ExactCausalAdjointGradientPlan>(
            qubits,
            operations,
            queries,
            ExactCausalAdjointGradientConfig{
                ExactAdjointGradientConfig{tensor, 4U},
            });
    });
    const auto plan_stats = plan->stats();

    const auto workspace_start = Clock::now();
    auto workspace = plan->workspace();
    const double workspace_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - workspace_start)
            .count();

    const auto output_start = Clock::now();
    std::vector<QComplex> values(query_count);
    std::vector<QComplex> gradients(query_count * parameter_count);
    const double output_allocation_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - output_start)
            .count();

    std::vector<double> run_times;
    run_times.reserve(repetitions);
    for (std::size_t repetition = 0U; repetition < repetitions; ++repetition) {
        run_times.push_back(milliseconds([&] {
            plan->value_and_gradient(
                parameters,
                values,
                gradients,
                workspace);
        }));
    }

    QComplex value_checksum{};
    QComplex gradient_checksum{};
    for (const QComplex value : values) {
        value_checksum += value;
    }
    for (const QComplex value : gradients) {
        gradient_checksum += value;
    }

    const double first_execution_ms = run_times.front();
    const double best_execution_ms =
        *std::min_element(run_times.begin(), run_times.end());
    const double setup_ms =
        workload_build_ms + compile_ms + workspace_ms + output_allocation_ms;

    const std::string prefix = "qsa_gradient" + std::to_string(qubits);
    std::cout << prefix << "_qubits=" << qubits << '\n';
    std::cout << prefix << "_operations=" << operations.size() << '\n';
    std::cout << prefix << "_queries=" << query_count << '\n';
    std::cout << prefix << "_parameters=" << parameter_count << '\n';
    std::cout << prefix << "_repetitions=" << repetitions << '\n';
    std::cout << prefix << "_selected_route=causal_term_parallel\n";
    std::cout << prefix << "_selected_workers="
              << workspace.execution_worker_count() << '\n';
    std::cout << prefix << "_dynamic_terms="
              << plan_stats.dynamic_term_count << '\n';
    std::cout << prefix << "_static_terms="
              << plan_stats.static_term_count << '\n';
    std::cout << prefix << "_estimated_critical_work="
              << plan_stats.dynamic.balanced_peak_estimated_work << '\n';
    std::cout << prefix << "_estimated_workspace_bytes="
              << workspace.estimated_bytes() << '\n';
    std::cout << prefix << "_plan_bytes=" << plan->estimated_bytes() << '\n';
    std::cout << prefix << "_workspace_bytes=" << workspace.estimated_bytes() << '\n';
    std::cout << prefix << "_workload_build_ms=" << workload_build_ms << '\n';
    std::cout << prefix << "_compile_ms=" << compile_ms << '\n';
    std::cout << prefix << "_workspace_ms=" << workspace_ms << '\n';
    std::cout << prefix << "_output_allocation_ms=" << output_allocation_ms << '\n';
    std::cout << prefix << "_setup_ms=" << setup_ms << '\n';
    std::cout << prefix << "_first_execution_ms=" << first_execution_ms << '\n';
    std::cout << prefix << "_best_execution_ms=" << best_execution_ms << '\n';
    std::cout << prefix << "_setup_plus_first_ms="
              << setup_ms + first_execution_ms << '\n';
    std::cout << prefix << "_value_checksum_real=" << value_checksum.re << '\n';
    std::cout << prefix << "_value_checksum_imag=" << value_checksum.im << '\n';
    std::cout << prefix << "_gradient_checksum_real=" << gradient_checksum.re << '\n';
    std::cout << prefix << "_gradient_checksum_imag=" << gradient_checksum.im << '\n';

    for (std::size_t observable = 0U; observable < query_count; ++observable) {
        const QComplex value = values[observable];
        std::cout << prefix << "_value_" << observable << "_real=" << value.re << '\n';
        std::cout << prefix << "_value_" << observable << "_imag=" << value.im << '\n';
        for (std::size_t parameter = 0U; parameter < parameter_count; ++parameter) {
            const QComplex gradient =
                gradients[observable * parameter_count + parameter];
            std::cout << prefix << "_gradient_" << observable << '_' << parameter
                      << "_real=" << gradient.re << '\n';
            std::cout << prefix << "_gradient_" << observable << '_' << parameter
                      << "_imag=" << gradient.im << '\n';
        }
    }
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    run_case(18U, 24U, 6U);
    run_case(100U, 8U, 4U);
    return 0;
}

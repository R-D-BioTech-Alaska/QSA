#include "qubit/qtensor_adjoint_batch.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::ExactAdjointGradientBatchConfig;
using qubit::ExactAdjointGradientBatchPlan;
using qubit::ExactAdjointGradientConfig;
using qubit::ExactAdjointGradientPlan;
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

std::vector<ParameterizedOperation> parameterized_brickwork(
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

std::vector<PauliObservable> observables(std::size_t qubits, std::size_t count) {
    std::vector<PauliObservable> result;
    result.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
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

std::vector<double> parameter_points(
    std::size_t point_count,
    std::size_t parameter_count) {
    std::vector<double> result(point_count * parameter_count);
    for (std::size_t point = 0U; point < point_count; ++point) {
        for (std::size_t parameter = 0U;
             parameter < parameter_count;
             ++parameter) {
            result[point * parameter_count + parameter] =
                -0.53 +
                0.061 * static_cast<double>(point + 1U) +
                0.037 * static_cast<double>(parameter + 1U);
        }
    }
    return result;
}

void evaluate_points(
    const ExactAdjointGradientPlan& plan,
    ExactAdjointGradientWorkspace& workspace,
    std::span<const double> parameters,
    std::size_t point_count,
    std::span<QComplex> values,
    std::span<QComplex> gradients) {
    const std::size_t parameter_count = plan.parameter_count();
    const std::size_t observable_count = plan.observable_count();
    const std::size_t point_gradient_entries =
        observable_count * parameter_count;
    for (std::size_t point = 0U; point < point_count; ++point) {
        plan.value_and_gradient(
            std::span<const double>(
                parameters.data() + point * parameter_count,
                parameter_count),
            std::span<QComplex>(
                values.data() + point * observable_count,
                observable_count),
            std::span<QComplex>(
                gradients.data() + point * point_gradient_entries,
                point_gradient_entries),
            workspace);
    }
}

void run_case(
    std::size_t qubits,
    std::size_t query_count,
    std::size_t parameter_count,
    std::size_t point_count) {
    const TensorNetworkConfig tensor{1U << 16U, 1'000'000U};
    const std::vector<ParameterizedOperation> operations =
        parameterized_brickwork(qubits, 5U, parameter_count);
    const std::vector<PauliObservable> queries = observables(qubits, query_count);
    const std::vector<double> parameters =
        parameter_points(point_count, parameter_count);
    const std::size_t point_gradient_entries = query_count * parameter_count;

    std::unique_ptr<ExactAdjointGradientPlan> serial;
    const double serial_compile_ms = milliseconds([&] {
        serial = std::make_unique<ExactAdjointGradientPlan>(
            qubits,
            operations,
            queries,
            ExactAdjointGradientConfig{tensor, 1U});
    });
    auto serial_workspace = serial->workspace();
    std::vector<QComplex> serial_values(point_count * query_count);
    std::vector<QComplex> serial_gradients(
        point_count * point_gradient_entries);
    const double serial_sweep_ms = milliseconds([&] {
        evaluate_points(
            *serial,
            serial_workspace,
            parameters,
            point_count,
            serial_values,
            serial_gradients);
    });

    std::unique_ptr<ExactAdjointGradientPlan> term_parallel;
    const double term_compile_ms = milliseconds([&] {
        term_parallel = std::make_unique<ExactAdjointGradientPlan>(
            qubits,
            operations,
            queries,
            ExactAdjointGradientConfig{tensor, 4U});
    });
    auto term_workspace = term_parallel->workspace();
    std::vector<QComplex> term_values(serial_values.size());
    std::vector<QComplex> term_gradients(serial_gradients.size());
    const double term_sweep_ms = milliseconds([&] {
        evaluate_points(
            *term_parallel,
            term_workspace,
            parameters,
            point_count,
            term_values,
            term_gradients);
    });

    std::unique_ptr<ExactAdjointGradientBatchPlan> point_batch;
    const double point_compile_ms = milliseconds([&] {
        point_batch = std::make_unique<ExactAdjointGradientBatchPlan>(
            qubits,
            operations,
            queries,
            ExactAdjointGradientBatchConfig{tensor, 4U});
    });
    auto point_workspace = point_batch->workspace();
    std::vector<QComplex> point_values(serial_values.size());
    std::vector<QComplex> point_gradients(serial_gradients.size());
    const double point_sweep_ms = milliseconds([&] {
        point_batch->value_and_gradient_batch(
            parameters,
            point_count,
            point_values,
            point_gradients,
            point_workspace);
    });

    double term_value_error = 0.0;
    double term_gradient_error = 0.0;
    double point_value_error = 0.0;
    double point_gradient_error = 0.0;
    for (std::size_t index = 0U; index < serial_values.size(); ++index) {
        term_value_error = std::max(
            term_value_error,
            (serial_values[index] - term_values[index]).magnitude());
        point_value_error = std::max(
            point_value_error,
            (serial_values[index] - point_values[index]).magnitude());
    }
    for (std::size_t index = 0U; index < serial_gradients.size(); ++index) {
        term_gradient_error = std::max(
            term_gradient_error,
            (serial_gradients[index] - term_gradients[index]).magnitude());
        point_gradient_error = std::max(
            point_gradient_error,
            (serial_gradients[index] - point_gradients[index]).magnitude());
    }

    const std::string prefix = "pointbatch" + std::to_string(qubits);
    std::cout << prefix << "_qubits=" << qubits << '\n';
    std::cout << prefix << "_queries=" << query_count << '\n';
    std::cout << prefix << "_parameters=" << parameter_count << '\n';
    std::cout << prefix << "_points=" << point_count << '\n';
    std::cout << prefix << "_serial_compile_ms=" << serial_compile_ms << '\n';
    std::cout << prefix << "_serial_sweep_ms=" << serial_sweep_ms << '\n';
    std::cout << prefix << "_term_workers=" << term_parallel->worker_count() << '\n';
    std::cout << prefix << "_term_compile_ms=" << term_compile_ms << '\n';
    std::cout << prefix << "_term_sweep_ms=" << term_sweep_ms << '\n';
    std::cout << prefix << "_term_parallel_ratio="
              << serial_sweep_ms / term_sweep_ms << '\n';
    std::cout << prefix << "_point_workers="
              << point_batch->point_worker_count() << '\n';
    std::cout << prefix << "_point_compile_ms=" << point_compile_ms << '\n';
    std::cout << prefix << "_point_sweep_ms=" << point_sweep_ms << '\n';
    std::cout << prefix << "_point_parallel_ratio="
              << serial_sweep_ms / point_sweep_ms << '\n';
    std::cout << prefix << "_point_vs_term_ratio="
              << term_sweep_ms / point_sweep_ms << '\n';
    std::cout << prefix << "_serial_first_total_ms="
              << serial_compile_ms + serial_sweep_ms << '\n';
    std::cout << prefix << "_term_first_total_ms="
              << term_compile_ms + term_sweep_ms << '\n';
    std::cout << prefix << "_point_first_total_ms="
              << point_compile_ms + point_sweep_ms << '\n';
    std::cout << prefix << "_point_first_total_speedup="
              << (serial_compile_ms + serial_sweep_ms) /
                     (point_compile_ms + point_sweep_ms)
              << '\n';
    std::cout << prefix << "_term_value_error=" << term_value_error << '\n';
    std::cout << prefix << "_term_gradient_error=" << term_gradient_error << '\n';
    std::cout << prefix << "_point_value_error=" << point_value_error << '\n';
    std::cout << prefix << "_point_gradient_error=" << point_gradient_error << '\n';
    std::cout << prefix << "_serial_rebinds="
              << serial_workspace.rebind_count() << '\n';
    std::cout << prefix << "_term_rebinds="
              << term_workspace.rebind_count() << '\n';
    std::cout << prefix << "_point_rebinds="
              << point_workspace.rebind_count() << '\n';
    std::cout << prefix << "_serial_workspace_bytes="
              << serial_workspace.estimated_bytes() << '\n';
    std::cout << prefix << "_term_workspace_bytes="
              << term_workspace.estimated_bytes() << '\n';
    std::cout << prefix << "_point_workspace_bytes="
              << point_workspace.estimated_bytes() << '\n';
    std::cout << prefix << "_point_plan_bytes="
              << point_batch->estimated_bytes() << '\n';
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);
    run_case(18U, 24U, 6U, 12U);
    run_case(100U, 8U, 4U, 8U);
    return 0;
}

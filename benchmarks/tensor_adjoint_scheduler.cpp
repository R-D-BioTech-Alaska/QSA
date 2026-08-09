#include "qubit/qtensor_adjoint_batch.hpp"
#include "qubit/qtensor_adjoint_scheduler.hpp"

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
using qubit::ExactAdjointGradientBatchConfig;
using qubit::ExactAdjointGradientBatchPlan;
using qubit::ExactAdjointGradientConfig;
using qubit::ExactAdjointGradientPlan;
using qubit::ExactAdjointGradientSchedulerConfig;
using qubit::ExactAdjointGradientSchedulerPlan;
using qubit::ExactAdjointGradientWorkspace;
using qubit::ExactAdjointScheduleRoute;
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
                -0.47 +
                0.059 * static_cast<double>(point + 1U) +
                0.031 * static_cast<double>(parameter + 1U);
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

const char* route_name(ExactAdjointScheduleRoute route) {
    switch (route) {
        case ExactAdjointScheduleRoute::Serial: return "serial";
        case ExactAdjointScheduleRoute::TermParallel: return "term_parallel";
        case ExactAdjointScheduleRoute::PointParallel: return "point_parallel";
    }
    return "invalid";
}

void run_case(
    std::size_t qubits,
    std::size_t query_count,
    std::size_t parameter_count,
    std::size_t point_count) {
    const TensorNetworkConfig tensor{1U << 16U, 1'000'000U};
    const auto operations = parameterized_brickwork(qubits, 5U, parameter_count);
    const auto queries = observables(qubits, query_count);
    const auto parameters = parameter_points(point_count, parameter_count);
    const std::size_t value_count = point_count * query_count;
    const std::size_t gradient_count =
        point_count * query_count * parameter_count;

    std::unique_ptr<ExactAdjointGradientPlan> serial;
    const double serial_compile_ms = milliseconds([&] {
        serial = std::make_unique<ExactAdjointGradientPlan>(
            qubits, operations, queries, ExactAdjointGradientConfig{tensor, 1U});
    });
    auto serial_workspace = serial->workspace();
    std::vector<QComplex> serial_values(value_count);
    std::vector<QComplex> serial_gradients(gradient_count);
    const double serial_ms = milliseconds([&] {
        evaluate_points(
            *serial, serial_workspace, parameters, point_count,
            serial_values, serial_gradients);
    });

    std::unique_ptr<ExactAdjointGradientPlan> term;
    const double term_compile_ms = milliseconds([&] {
        term = std::make_unique<ExactAdjointGradientPlan>(
            qubits, operations, queries, ExactAdjointGradientConfig{tensor, 4U});
    });
    auto term_workspace = term->workspace();
    std::vector<QComplex> term_values(value_count);
    std::vector<QComplex> term_gradients(gradient_count);
    const double term_ms = milliseconds([&] {
        evaluate_points(
            *term, term_workspace, parameters, point_count,
            term_values, term_gradients);
    });

    std::unique_ptr<ExactAdjointGradientBatchPlan> point;
    const double point_compile_ms = milliseconds([&] {
        point = std::make_unique<ExactAdjointGradientBatchPlan>(
            qubits, operations, queries,
            ExactAdjointGradientBatchConfig{tensor, 4U});
    });
    auto point_workspace = point->workspace();
    std::vector<QComplex> point_values(value_count);
    std::vector<QComplex> point_gradients(gradient_count);
    const double point_ms = milliseconds([&] {
        point->value_and_gradient_batch(
            parameters, point_count, point_values, point_gradients, point_workspace);
    });

    std::unique_ptr<ExactAdjointGradientSchedulerPlan> scheduler;
    const double scheduler_compile_ms = milliseconds([&] {
        scheduler = std::make_unique<ExactAdjointGradientSchedulerPlan>(
            qubits, operations, queries,
            ExactAdjointGradientSchedulerConfig{tensor, 4U, 0U});
    });
    const auto predicted = scheduler->choose(point_count);
    auto scheduler_workspace = scheduler->workspace();
    std::vector<QComplex> scheduler_values(value_count);
    std::vector<QComplex> scheduler_gradients(gradient_count);
    qubit::ExactAdjointGradientSchedule completed;
    const double scheduler_ms = milliseconds([&] {
        scheduler->value_and_gradient_batch(
            parameters,
            point_count,
            scheduler_values,
            scheduler_gradients,
            scheduler_workspace,
            &completed);
    });

    double term_error = 0.0;
    double point_error = 0.0;
    double scheduler_error = 0.0;
    for (std::size_t index = 0U; index < value_count; ++index) {
        term_error = std::max(term_error, (serial_values[index] - term_values[index]).magnitude());
        point_error = std::max(point_error, (serial_values[index] - point_values[index]).magnitude());
        scheduler_error = std::max(scheduler_error, (serial_values[index] - scheduler_values[index]).magnitude());
    }
    for (std::size_t index = 0U; index < gradient_count; ++index) {
        term_error = std::max(term_error, (serial_gradients[index] - term_gradients[index]).magnitude());
        point_error = std::max(point_error, (serial_gradients[index] - point_gradients[index]).magnitude());
        scheduler_error = std::max(scheduler_error, (serial_gradients[index] - scheduler_gradients[index]).magnitude());
    }

    QComplex value_checksum{};
    QComplex gradient_checksum{};
    for (const QComplex value : scheduler_values) {
        value_checksum += value;
    }
    for (const QComplex value : scheduler_gradients) {
        gradient_checksum += value;
    }

    const double best_fixed_ms = std::min({serial_ms, term_ms, point_ms});
    double selected_fixed_ms = serial_ms;
    if (completed.route == ExactAdjointScheduleRoute::TermParallel) selected_fixed_ms = term_ms;
    if (completed.route == ExactAdjointScheduleRoute::PointParallel) selected_fixed_ms = point_ms;

    const std::string prefix =
        "scheduler" + std::to_string(qubits) + "p" + std::to_string(point_count);
    std::cout << prefix << "_qubits=" << qubits << '\n';
    std::cout << prefix << "_queries=" << query_count << '\n';
    std::cout << prefix << "_parameters=" << parameter_count << '\n';
    std::cout << prefix << "_points=" << point_count << '\n';
    std::cout << prefix << "_selected_route=" << route_name(completed.route) << '\n';
    std::cout << prefix << "_selected_workers=" << completed.worker_count << '\n';
    std::cout << prefix << "_serial_work=" << predicted.serial_estimated_critical_work << '\n';
    std::cout << prefix << "_term_work=" << predicted.term_estimated_critical_work << '\n';
    std::cout << prefix << "_point_work=" << predicted.point_estimated_critical_work << '\n';
    std::cout << prefix << "_selected_work=" << predicted.estimated_critical_work << '\n';
    std::cout << prefix << "_serial_workspace_bytes=" << predicted.serial_estimated_workspace_bytes << '\n';
    std::cout << prefix << "_term_workspace_bytes=" << predicted.term_estimated_workspace_bytes << '\n';
    std::cout << prefix << "_point_workspace_bytes=" << predicted.point_estimated_workspace_bytes << '\n';
    std::cout << prefix << "_selected_workspace_bytes=" << predicted.estimated_workspace_bytes << '\n';
    std::cout << prefix << "_serial_compile_ms=" << serial_compile_ms << '\n';
    std::cout << prefix << "_term_compile_ms=" << term_compile_ms << '\n';
    std::cout << prefix << "_point_compile_ms=" << point_compile_ms << '\n';
    std::cout << prefix << "_scheduler_compile_ms=" << scheduler_compile_ms << '\n';
    std::cout << prefix << "_serial_ms=" << serial_ms << '\n';
    std::cout << prefix << "_term_ms=" << term_ms << '\n';
    std::cout << prefix << "_point_ms=" << point_ms << '\n';
    std::cout << prefix << "_scheduler_ms=" << scheduler_ms << '\n';
    std::cout << prefix << "_best_fixed_ms=" << best_fixed_ms << '\n';
    std::cout << prefix << "_selected_fixed_ms=" << selected_fixed_ms << '\n';
    std::cout << prefix << "_scheduler_vs_best=" << best_fixed_ms / scheduler_ms << '\n';
    std::cout << prefix << "_scheduler_vs_selected_fixed=" << selected_fixed_ms / scheduler_ms << '\n';
    std::cout << prefix << "_term_error=" << term_error << '\n';
    std::cout << prefix << "_point_error=" << point_error << '\n';
    std::cout << prefix << "_scheduler_error=" << scheduler_error << '\n';
    std::cout << prefix << "_value_checksum_real=" << value_checksum.re << '\n';
    std::cout << prefix << "_value_checksum_imag=" << value_checksum.im << '\n';
    std::cout << prefix << "_gradient_checksum_real=" << gradient_checksum.re << '\n';
    std::cout << prefix << "_gradient_checksum_imag=" << gradient_checksum.im << '\n';
    std::cout << prefix << "_scheduler_successful_points="
              << scheduler_workspace.successful_point_count() << '\n';
    std::cout << prefix << "_scheduler_workspace_bytes="
              << scheduler_workspace.estimated_bytes() << '\n';
    std::cout << prefix << "_scheduler_plan_bytes="
              << scheduler->estimated_bytes() << '\n';
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);
    run_case(18U, 24U, 6U, 1U);
    run_case(18U, 24U, 6U, 12U);
    run_case(100U, 8U, 4U, 1U);
    run_case(100U, 8U, 4U, 8U);
    return 0;
}

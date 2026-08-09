#include "qubit/qparameterized_estimator.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::ExactExecutionRoute;
using qubit::ExactParameterizedEstimatorConfig;
using qubit::ExactParameterizedEstimatorPlan;
using qubit::OperationCode;
using qubit::ParameterizedOperation;
using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
using qubit::QComplex;

struct Workload {
    std::vector<ParameterizedOperation> operations{};
    std::vector<PauliObservable> observables{};
    std::vector<double> parameters{};
};

template <typename Function>
[[nodiscard]] double milliseconds(Function&& function) {
    const auto start = Clock::now();
    function();
    const auto finish = Clock::now();
    return std::chrono::duration<double, std::milli>(finish - start).count();
}

[[nodiscard]] Workload build_workload(
    std::size_t qubits,
    std::size_t layers,
    std::size_t parameter_count,
    std::size_t observable_count,
    std::size_t point_count) {
    Workload result;
    result.operations.reserve(layers * (2U * qubits + qubits / 2U));
    std::size_t next_parameter = 0U;
    for (std::size_t layer = 0U; layer < layers; ++layer) {
        for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
            ParameterizedOperation ry;
            ry.operation.code = OperationCode::Ry;
            ry.operation.first = static_cast<qubit::QubitId>(qubit);
            ry.operation.parameter =
                0.007 * static_cast<double>((layer + 1U) * (qubit + 3U));
            if (next_parameter < parameter_count) {
                ry.parameter_slot = static_cast<std::int32_t>(next_parameter++);
            }
            result.operations.push_back(ry);

            ParameterizedOperation rz;
            rz.operation.code = OperationCode::Rz;
            rz.operation.first = static_cast<qubit::QubitId>(qubit);
            rz.operation.parameter =
                -0.005 * static_cast<double>((layer + 2U) * (qubit + 1U));
            if (next_parameter < parameter_count) {
                rz.parameter_slot = static_cast<std::int32_t>(next_parameter++);
            }
            result.operations.push_back(rz);
        }
        const std::size_t start = layer & 1U;
        for (std::size_t qubit = start; qubit + 1U < qubits; qubit += 2U) {
            ParameterizedOperation cnot;
            cnot.operation.code = OperationCode::Cnot;
            cnot.operation.first = static_cast<qubit::QubitId>(qubit);
            cnot.operation.second = static_cast<qubit::QubitId>(qubit + 1U);
            result.operations.push_back(cnot);
        }
    }
    if (next_parameter != parameter_count) {
        throw std::runtime_error("parameterized estimator benchmark did not assign all parameters");
    }

    result.observables.reserve(observable_count);
    for (std::size_t index = 0U; index < observable_count; ++index) {
        PauliObservable observable(qubits);
        const std::size_t first = (index * 7U + 3U) % qubits;
        const std::size_t second = (first + 1U + index % 3U) % qubits;
        const std::size_t third = (second + 2U) % qubits;
        std::vector<PauliFactor> factors{
            {static_cast<qubit::QubitId>(first), static_cast<PauliAxis>(1U + index % 3U)},
            {static_cast<qubit::QubitId>(second), static_cast<PauliAxis>(1U + (index + 1U) % 3U)},
        };
        if ((index & 1U) != 0U && third != first && third != second) {
            factors.push_back({
                static_cast<qubit::QubitId>(third),
                static_cast<PauliAxis>(1U + (index + 2U) % 3U),
            });
        }
        observable.add_term({1.0, 0.0}, factors);
        result.observables.push_back(std::move(observable));
    }

    result.parameters.resize(point_count * parameter_count);
    for (std::size_t point = 0U; point < point_count; ++point) {
        for (std::size_t parameter = 0U; parameter < parameter_count; ++parameter) {
            result.parameters[point * parameter_count + parameter] =
                -0.47 + 0.021 * static_cast<double>(point + 1U) +
                0.031 * static_cast<double>(parameter + 1U);
        }
    }
    return result;
}

void run_case(
    std::size_t qubits,
    std::size_t observable_count,
    std::size_t parameter_count,
    std::size_t point_count,
    std::size_t repetitions) {
    const std::string prefix = "pest" + std::to_string(qubits);
    Workload workload;
    const double workload_build_ms = milliseconds([&] {
        workload = build_workload(
            qubits, 5U, parameter_count, observable_count, point_count);
    });

    ExactParameterizedEstimatorConfig config;
    config.point_worker_count = 4U;

    std::unique_ptr<ExactParameterizedEstimatorPlan> plan;
    const double compile_ms = milliseconds([&] {
        plan = std::make_unique<ExactParameterizedEstimatorPlan>(
            qubits, workload.operations, workload.observables, config);
    });
    if (plan->route() != ExactExecutionRoute::TensorNetwork) {
        throw std::runtime_error(
            "parameterized estimator benchmark unexpectedly fell back from tensor reuse");
    }

    qubit::ExactParameterizedEstimatorWorkspace workspace;
    const double workspace_ms = milliseconds([&] {
        workspace = plan->workspace(point_count);
    });
    std::vector<QComplex> values(point_count * observable_count);

    std::vector<double> sweep_times;
    sweep_times.reserve(repetitions);
    for (std::size_t repetition = 0U; repetition < repetitions; ++repetition) {
        const double elapsed = milliseconds([&] {
            plan->estimate_points(
                workload.parameters, point_count, values, workspace);
        });
        sweep_times.push_back(elapsed);
    }

    const QComplex checksum = std::accumulate(
        values.begin(), values.end(), QComplex{});
    const double first_ms = sweep_times.front();
    const double best_ms = *std::min_element(sweep_times.begin(), sweep_times.end());
    const double setup_ms = workload_build_ms + compile_ms + workspace_ms;

    std::cout << prefix << "_qubits=" << qubits << '\n';
    std::cout << prefix << "_operations=" << workload.operations.size() << '\n';
    std::cout << prefix << "_queries=" << observable_count << '\n';
    std::cout << prefix << "_parameters=" << parameter_count << '\n';
    std::cout << prefix << "_points=" << point_count << '\n';
    std::cout << prefix << "_worker_count=" << workspace.worker_count() << '\n';
    std::cout << prefix << "_route=" << static_cast<int>(plan->route()) << '\n';
    std::cout << prefix << "_workload_build_ms=" << workload_build_ms << '\n';
    std::cout << prefix << "_compile_ms=" << compile_ms << '\n';
    std::cout << prefix << "_workspace_ms=" << workspace_ms << '\n';
    std::cout << prefix << "_first_sweep_ms=" << first_ms << '\n';
    std::cout << prefix << "_best_sweep_ms=" << best_ms << '\n';
    std::cout << prefix << "_setup_plus_first_ms=" << setup_ms + first_ms << '\n';
    std::cout << prefix << "_plan_bytes=" << plan->estimated_bytes() << '\n';
    std::cout << prefix << "_workspace_bytes=" << workspace.estimated_bytes() << '\n';
    std::cout << prefix << "_rebind_count=" << workspace.rebind_count() << '\n';
    std::cout << prefix << "_checksum_real=" << checksum.re << '\n';
    std::cout << prefix << "_checksum_imag=" << checksum.im << '\n';

    for (std::size_t point = 0U; point < point_count; ++point) {
        for (std::size_t observable = 0U; observable < observable_count; ++observable) {
            const QComplex value = values[point * observable_count + observable];
            std::cout << prefix << "_value_" << point << '_' << observable
                      << "_real=" << value.re << '\n';
            std::cout << prefix << "_value_" << point << '_' << observable
                      << "_imag=" << value.im << '\n';
        }
    }
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    run_case(18U, 24U, 6U, 12U, 3U);
    run_case(100U, 8U, 4U, 8U, 3U);
    return 0;
}

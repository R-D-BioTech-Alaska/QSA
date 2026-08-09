#include "qubit/qparameterized_estimator.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using qubit::ExactExecutionRoute;
using qubit::ExactParameterizedEstimatorConfig;
using qubit::ExactParameterizedEstimatorPlan;
using qubit::Operation;
using qubit::OperationCode;
using qubit::ParameterizedOperation;
using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
using qubit::QComplex;
using qubit::QRegister;
using qubit::QStateError;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(QComplex actual, QComplex expected, double tolerance, const char* message) {
    if (!qubit::almost_equal(actual, expected, tolerance)) {
        throw std::runtime_error(message);
    }
}

[[nodiscard]] std::vector<ParameterizedOperation> parameterized_brickwork(
    std::size_t qubits,
    std::size_t layers,
    std::size_t parameter_count) {
    std::vector<ParameterizedOperation> operations;
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
    require(next_parameter == parameter_count,
            "test circuit did not assign every requested parameter");
    return operations;
}

[[nodiscard]] std::vector<PauliObservable> observables(
    std::size_t qubits,
    std::size_t count) {
    std::vector<PauliObservable> result;
    result.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        PauliObservable observable(qubits);
        const qubit::QubitId first = static_cast<qubit::QubitId>((index * 3U + 1U) % qubits);
        const qubit::QubitId second = static_cast<qubit::QubitId>((index * 5U + 2U) % qubits);
        std::vector<PauliFactor> factors{
            {first, static_cast<PauliAxis>(1U + index % 3U)},
        };
        if (second != first) {
            factors.push_back({
                second,
                static_cast<PauliAxis>(1U + (index + 1U) % 3U),
            });
        }
        observable.add_term({1.0, 0.0}, factors);
        result.push_back(std::move(observable));
    }
    return result;
}

[[nodiscard]] std::vector<Operation> bind(
    const std::vector<ParameterizedOperation>& operations,
    std::span<const double> parameters) {
    std::vector<Operation> result;
    result.reserve(operations.size());
    for (const ParameterizedOperation& templated : operations) {
        Operation operation = templated.operation;
        if (templated.parameter_slot >= 0) {
            operation.parameter = parameters[static_cast<std::size_t>(templated.parameter_slot)];
        }
        result.push_back(operation);
    }
    return result;
}

[[nodiscard]] std::vector<QComplex> register_reference(
    std::size_t qubits,
    const std::vector<ParameterizedOperation>& operations,
    const std::vector<PauliObservable>& query,
    std::span<const double> parameters) {
    QRegister state(qubits);
    const std::vector<Operation> concrete = bind(operations, parameters);
    qubit::OperationPlan plan(concrete, false);
    plan.execute(state);
    std::vector<QComplex> result;
    result.reserve(query.size());
    for (const PauliObservable& observable : query) {
        result.push_back(observable.expectation(state));
    }
    return result;
}

}  // namespace

int main() {
    {
        constexpr std::size_t qubits = 6U;
        constexpr std::size_t parameter_count = 5U;
        constexpr std::size_t point_count = 7U;
        const std::vector<ParameterizedOperation> operations =
            parameterized_brickwork(qubits, 3U, parameter_count);
        const std::vector<PauliObservable> query = observables(qubits, 5U);

        ExactParameterizedEstimatorConfig config;
        config.point_worker_count = 4U;
        ExactParameterizedEstimatorPlan estimator(qubits, operations, query, config);
        require(estimator.route() == ExactExecutionRoute::TensorNetwork,
                "bounded parameterized circuit did not select tensor topology reuse");
        require(estimator.parameter_count() == parameter_count,
                "parameterized estimator parameter count changed");
        require(estimator.parameterized_operation_count() == parameter_count,
                "parameterized estimator operation count changed");
        require(estimator.point_worker_limit() == 4U,
                "parameterized estimator worker limit changed");
        require(estimator.fallback_reason().empty(),
                "tensor-routable parameterized estimator recorded a fallback");

        std::vector<double> points(point_count * parameter_count);
        for (std::size_t point = 0U; point < point_count; ++point) {
            for (std::size_t parameter = 0U; parameter < parameter_count; ++parameter) {
                points[point * parameter_count + parameter] =
                    -0.41 + 0.037 * static_cast<double>(point + 1U) +
                    0.029 * static_cast<double>(parameter + 1U);
            }
        }

        auto workspace = estimator.workspace(point_count);
        require(workspace.point_count() == point_count,
                "parameterized estimator workspace point count changed");
        require(workspace.worker_count() == 4U,
                "parameterized estimator did not use the bounded point workers");
        require(workspace.estimated_bytes() > 0U,
                "parameterized estimator workspace did not report retained storage");

        std::vector<QComplex> results(point_count * query.size());
        estimator.estimate_points(points, point_count, results, workspace);
        require(workspace.rebind_count() == point_count,
                "parameterized estimator did not perform one topology rebind per point");

        for (std::size_t point = 0U; point < point_count; ++point) {
            const std::span<const double> parameters(
                points.data() + point * parameter_count,
                parameter_count);
            const std::vector<QComplex> expected =
                register_reference(qubits, operations, query, parameters);
            for (std::size_t observable = 0U; observable < query.size(); ++observable) {
                require_close(
                    results[point * query.size() + observable],
                    expected[observable],
                    5e-11,
                    "parameterized tensor estimator differs from QRegister");
            }
        }

        std::vector<QComplex> single(query.size());
        const std::span<const double> first(points.data(), parameter_count);
        estimator.estimate(first, single);
        for (std::size_t observable = 0U; observable < query.size(); ++observable) {
            require_close(single[observable], results[observable], 1e-12,
                          "single-point and batch parameterized estimates differ");
        }

        std::vector<QComplex> guarded(results.size(), {123.0, -456.0});
        std::vector<double> invalid = points;
        invalid[parameter_count + 1U] = std::numeric_limits<double>::quiet_NaN();
        bool rejected = false;
        try {
            estimator.estimate_points(invalid, point_count, guarded, workspace);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "parameterized estimator accepted a nonfinite point");
        for (const QComplex value : guarded) {
            require_close(value, {123.0, -456.0}, 0.0,
                          "failed parameterized estimate mutated caller output");
        }

        ExactParameterizedEstimatorPlan foreign_plan(qubits, operations, query, config);
        auto foreign_workspace = foreign_plan.workspace(point_count);
        rejected = false;
        try {
            estimator.estimate_points(points, point_count, guarded, foreign_workspace);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "parameterized estimator accepted a foreign workspace");

        rejected = false;
        try {
            estimator.estimate_points(points, point_count - 1U, guarded, workspace);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "parameterized estimator accepted a malformed batch shape");

        auto empty_workspace = estimator.workspace(0U);
        std::vector<QComplex> empty_results;
        std::vector<double> empty_parameters;
        estimator.estimate_points(empty_parameters, 0U, empty_results, empty_workspace);
        require(empty_workspace.worker_count() == 0U,
                "empty parameterized batch allocated point workers");
    }

    {
        constexpr std::size_t qubits = 4U;
        constexpr std::size_t point_count = 6U;
        std::vector<ParameterizedOperation> operations;

        ParameterizedOperation ry;
        ry.operation = {OperationCode::Ry, 0U, 0U, 0.0, 0.0};
        ry.parameter_slot = 0;
        operations.push_back(ry);

        ParameterizedOperation rz;
        rz.operation = {OperationCode::Rz, 1U, 0U, 0.0, 0.0};
        rz.parameter_slot = 0;
        operations.push_back(rz);

        ParameterizedOperation cnot;
        cnot.operation = {OperationCode::Cnot, 0U, 2U, 0.0, 0.0};
        operations.push_back(cnot);

        ParameterizedOperation rx;
        rx.operation = {OperationCode::Rx, 3U, 0U, 0.0, 0.0};
        rx.parameter_slot = 1;
        operations.push_back(rx);

        const std::vector<PauliObservable> query = observables(qubits, 4U);
        ExactParameterizedEstimatorConfig config;
        config.point_worker_count = 3U;
        ExactParameterizedEstimatorPlan estimator(qubits, operations, query, config);
        require(estimator.route() == ExactExecutionRoute::TensorNetwork,
                "shared-slot circuit did not retain tensor route");
        require(estimator.parameter_count() == 2U,
                "shared-slot circuit parameter count changed");
        require(estimator.parameterized_operation_count() == 3U,
                "shared-slot circuit occurrence count changed");

        const std::vector<double> points{
            -0.8, 0.3,
             0.7, -0.5,
            -0.2, 1.1,
             1.0, 0.0,
            -1.3, 0.6,
             0.4, -0.9,
        };
        auto workspace = estimator.workspace(point_count);
        std::vector<QComplex> results(point_count * query.size());
        estimator.estimate_points(points, point_count, results, workspace);
        require(workspace.rebind_count() == point_count,
                "shared-slot estimator did not rebind every point");

        for (std::size_t point = 0U; point < point_count; ++point) {
            const std::span<const double> parameters(points.data() + 2U * point, 2U);
            const std::vector<QComplex> expected =
                register_reference(qubits, operations, query, parameters);
            for (std::size_t observable = 0U; observable < query.size(); ++observable) {
                require_close(
                    results[point * query.size() + observable],
                    expected[observable],
                    5e-12,
                    "shared parameter slot did not update every bound tensor source");
            }
        }
    }

    {
        constexpr std::size_t qubits = 4U;
        constexpr std::size_t point_count = 5U;
        std::vector<ParameterizedOperation> operations;

        ParameterizedOperation ry;
        ry.operation = {OperationCode::Ry, 0U, 0U, 0.0, 0.0};
        ry.parameter_slot = 0;
        operations.push_back(ry);

        ParameterizedOperation h;
        h.operation = {OperationCode::H, 2U, 0U, 0.0, 0.0};
        operations.push_back(h);

        PauliObservable dynamic_observable(qubits);
        const PauliFactor dynamic_factor{0U, PauliAxis::Z};
        dynamic_observable.add_term(
            {1.0, 0.0},
            std::span<const PauliFactor>(&dynamic_factor, 1U));

        PauliObservable static_observable(qubits);
        const PauliFactor static_factor{3U, PauliAxis::Z};
        static_observable.add_term(
            {1.0, 0.0},
            std::span<const PauliFactor>(&static_factor, 1U));

        const std::vector<PauliObservable> query{
            dynamic_observable,
            static_observable,
        };
        ExactParameterizedEstimatorConfig config;
        config.point_worker_count = 2U;
        ExactParameterizedEstimatorPlan estimator(qubits, operations, query, config);
        require(estimator.route() == ExactExecutionRoute::TensorNetwork,
                "causal-cache test did not retain tensor route");
        const auto stats = estimator.stats();
        require(stats.dynamic_term_count == 1U,
                "causal-cache test did not identify the parameter-dependent term");
        require(stats.static_term_count == 1U,
                "causal-cache test did not identify the parameter-independent term");
        require(stats.dynamic_observable_count == 1U,
                "causal-cache test did not identify the dynamic observable");
        require(stats.static_observable_count == 1U,
                "causal-cache test did not identify the static observable");

        const std::vector<double> points{-1.1, -0.3, 0.2, 0.8, 1.4};
        auto workspace = estimator.workspace(point_count);
        std::vector<QComplex> results(point_count * query.size());
        estimator.estimate_points(points, point_count, results, workspace);
        for (std::size_t point = 0U; point < point_count; ++point) {
            const std::span<const double> parameters(points.data() + point, 1U);
            const std::vector<QComplex> expected =
                register_reference(qubits, operations, query, parameters);
            for (std::size_t observable = 0U; observable < query.size(); ++observable) {
                require_close(
                    results[point * query.size() + observable],
                    expected[observable],
                    5e-12,
                    "causal static-term cache differs from QRegister");
            }
            require_close(
                results[point * query.size() + 1U],
                results[1U],
                0.0,
                "cached static observable changed across parameter points");
        }
    }

    {
        constexpr std::size_t qubits = 3U;
        std::vector<ParameterizedOperation> operations;
        ParameterizedOperation ry;
        ry.operation = {OperationCode::Ry, 0U, 0U, 0.0, 0.0};
        ry.parameter_slot = 0;
        operations.push_back(ry);
        ParameterizedOperation noise;
        noise.operation = {
            OperationCode::BitFlipTrajectory,
            1U,
            0U,
            0.75,
            0.25,
        };
        operations.push_back(noise);
        ParameterizedOperation cnot;
        cnot.operation = {OperationCode::Cnot, 0U, 2U, 0.0, 0.0};
        operations.push_back(cnot);

        const std::vector<PauliObservable> query = observables(qubits, 3U);
        ExactParameterizedEstimatorConfig config;
        config.point_worker_count = 2U;
        ExactParameterizedEstimatorPlan estimator(qubits, operations, query, config);
        require(estimator.route() == ExactExecutionRoute::Register,
                "trajectory parameterized estimator did not fail closed to QRegister");
        require(!estimator.fallback_reason().empty(),
                "register fallback did not retain its tensor rejection reason");

        const std::vector<double> points{-0.3, 0.2, 0.7};
        auto workspace = estimator.workspace(3U);
        std::vector<QComplex> results(3U * query.size());
        estimator.estimate_points(points, 3U, results, workspace);
        require(workspace.rebind_count() == 0U,
                "register fallback unexpectedly materialized tensor rebinds");
        for (std::size_t point = 0U; point < 3U; ++point) {
            const std::span<const double> parameters(points.data() + point, 1U);
            const std::vector<QComplex> expected =
                register_reference(qubits, operations, query, parameters);
            for (std::size_t observable = 0U; observable < query.size(); ++observable) {
                require_close(
                    results[point * query.size() + observable],
                    expected[observable],
                    1e-12,
                    "parameterized register fallback differs from direct QRegister");
            }
        }
    }

    {
        PauliObservable z(2U);
        const PauliFactor factor{0U, PauliAxis::Z};
        z.add_term({1.0, 0.0}, std::span<const PauliFactor>(&factor, 1U));
        const std::vector<PauliObservable> query{z};

        ParameterizedOperation invalid_gate;
        invalid_gate.operation = {OperationCode::Cnot, 0U, 1U, 0.0, 0.0};
        invalid_gate.parameter_slot = 0;
        bool rejected = false;
        try {
            const std::vector<ParameterizedOperation> operations{invalid_gate};
            ExactParameterizedEstimatorPlan plan(2U, operations, query);
            (void)plan;
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "parameterized estimator accepted an unsupported parameter generator");

        ParameterizedOperation stochastic;
        stochastic.operation = {OperationCode::Ry, 0U, 0U, 0.0, 0.0};
        stochastic.sample_slot = 0;
        rejected = false;
        try {
            const std::vector<ParameterizedOperation> operations{stochastic};
            ExactParameterizedEstimatorPlan plan(2U, operations, query);
            (void)plan;
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "parameterized estimator accepted a stochastic sample slot");

        ExactParameterizedEstimatorConfig too_many_workers;
        too_many_workers.point_worker_count = 129U;
        rejected = false;
        try {
            const std::vector<ParameterizedOperation> operations;
            ExactParameterizedEstimatorPlan plan(2U, operations, query, too_many_workers);
            (void)plan;
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "parameterized estimator accepted an unbounded worker request");
    }

    return 0;
}

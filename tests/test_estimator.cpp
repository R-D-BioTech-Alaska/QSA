#include "qubit/qestimator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using qubit::ExactEstimatorPlan;
using qubit::ExactEstimatorResult;
using qubit::ExactEstimatorConfig;
using qubit::ExactExecutionRoute;
using qubit::Operation;
using qubit::OperationCode;
using qubit::OperationPlan;
using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
using qubit::PauliPropagationConfig;
using qubit::QComplex;
using qubit::QRegister;
using qubit::QStateError;
using qubit::TensorNetworkConfig;

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

Operation random_operation(std::mt19937_64& rng, std::size_t qubits) {
    std::uniform_int_distribution<int> kind(0, 8);
    std::uniform_int_distribution<std::size_t> qubit(0U, qubits - 1U);
    std::uniform_real_distribution<double> angle(-0.8, 0.8);
    Operation operation;
    const int selected = kind(rng);
    operation.first = static_cast<qubit::QubitId>(qubit(rng));
    switch (selected) {
        case 0: operation.code = OperationCode::H; break;
        case 1: operation.code = OperationCode::X; break;
        case 2: operation.code = OperationCode::S; break;
        case 3: operation.code = OperationCode::T; break;
        case 4:
            operation.code = OperationCode::Rx;
            operation.parameter = angle(rng);
            break;
        case 5:
            operation.code = OperationCode::Ry;
            operation.parameter = angle(rng);
            break;
        case 6:
            operation.code = OperationCode::Rz;
            operation.parameter = angle(rng);
            break;
        default: {
            operation.code = selected == 7 ? OperationCode::Cnot : OperationCode::Cz;
            std::size_t second = qubit(rng);
            while (second == static_cast<std::size_t>(operation.first)) {
                second = qubit(rng);
            }
            operation.second = static_cast<qubit::QubitId>(second);
            break;
        }
    }
    return operation;
}

PauliObservable random_observable(std::mt19937_64& rng, std::size_t qubits) {
    PauliObservable observable(qubits, PauliPropagationConfig{65'536U});
    std::uniform_int_distribution<std::size_t> qubit(0U, qubits - 1U);
    std::uniform_int_distribution<int> axis(1, 3);
    std::uniform_real_distribution<double> coefficient(-1.0, 1.0);
    for (std::size_t term = 0; term < 3U; ++term) {
        std::vector<PauliFactor> factors;
        std::vector<std::uint8_t> used(qubits, 0U);
        const std::size_t support = 1U + term % 2U;
        while (factors.size() < support) {
            const std::size_t selected = qubit(rng);
            if (used[selected] != 0U) {
                continue;
            }
            used[selected] = 1U;
            factors.push_back({
                static_cast<qubit::QubitId>(selected),
                static_cast<PauliAxis>(axis(rng)),
            });
        }
        observable.add_term({coefficient(rng), 0.2 * coefficient(rng)}, factors);
    }
    return observable;
}

}  // namespace

int main() {
    {
        const std::vector<Operation> bell{
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::Cnot, 0U, 1U, 0.0, 0.0},
        };
        PauliObservable xx(2U);
        PauliObservable yy(2U);
        PauliObservable zz(2U);
        const std::vector<PauliFactor> xxf{{0U, PauliAxis::X}, {1U, PauliAxis::X}};
        const std::vector<PauliFactor> yyf{{0U, PauliAxis::Y}, {1U, PauliAxis::Y}};
        const std::vector<PauliFactor> zzf{{0U, PauliAxis::Z}, {1U, PauliAxis::Z}};
        xx.add_term({1.0, 0.0}, xxf);
        yy.add_term({1.0, 0.0}, yyf);
        zz.add_term({1.0, 0.0}, zzf);
        const std::vector<PauliObservable> observables{xx, yy, zz};
        ExactEstimatorPlan plan(2U, bell);
        const qubit::ExactEstimatorBatchPlan batch = plan.compile(observables);
        require(batch.observable_count() == observables.size(),
                "compiled estimator observable count changed");
        require(batch.tensor_observable_count() == 0U &&
                    batch.register_observable_count() == 0U,
                "compiled Bell estimator selected an unnecessary fallback");
        auto workspace = batch.workspace();
        std::vector<ExactEstimatorResult> results(observables.size());
        batch.estimate(results, workspace);
        require_close(results[0].value, {1.0, 0.0}, 1e-12, "Bell XX estimator");
        require_close(results[1].value, {-1.0, 0.0}, 1e-12, "Bell YY estimator");
        require_close(results[2].value, {1.0, 0.0}, 1e-12, "Bell ZZ estimator");
        for (const ExactEstimatorResult& result : results) {
            require(result.route == ExactExecutionRoute::CausalPauli,
                    "Bell estimator did not use exact causal Pauli propagation");
        }
    }

    {
        constexpr std::size_t qubits = 6U;
        std::mt19937_64 rng(0x657374696d61746fULL);
        for (std::size_t trial = 0; trial < 8U; ++trial) {
            std::vector<Operation> operations;
            operations.reserve(24U);
            for (std::size_t index = 0; index < 24U; ++index) {
                operations.push_back(random_operation(rng, qubits));
            }
            ExactEstimatorPlan estimator(qubits, operations);
            QRegister state(qubits);
            OperationPlan state_plan(operations, false);
            state_plan.execute(state);

            std::vector<PauliObservable> observables;
            observables.reserve(6U);
            for (std::size_t index = 0; index < 6U; ++index) {
                observables.push_back(random_observable(rng, qubits));
            }
            std::vector<ExactEstimatorResult> results(observables.size());
            estimator.estimate(observables, results);
            for (std::size_t index = 0; index < observables.size(); ++index) {
                require_close(
                    results[index].value,
                    observables[index].expectation(state),
                    5e-11,
                    "random exact estimator differs from QRegister");
            }
        }
    }

    {
        constexpr std::size_t qubits = 8U;
        std::vector<Operation> operations;
        for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
            operations.push_back({
                OperationCode::Ry,
                static_cast<qubit::QubitId>(qubit),
                0U,
                0.11 + 0.017 * static_cast<double>(qubit),
                0.0,
            });
        }
        for (std::size_t qubit = 0; qubit + 1U < qubits; qubit += 2U) {
            operations.push_back({
                OperationCode::Cnot,
                static_cast<qubit::QubitId>(qubit),
                static_cast<qubit::QubitId>(qubit + 1U),
                0.0,
                0.0,
            });
        }

        std::vector<PauliObservable> observables;
        for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
            PauliObservable observable(qubits, PauliPropagationConfig{1U});
            const PauliFactor z{static_cast<qubit::QubitId>(qubit), PauliAxis::Z};
            observable.add_term({1.0, 0.0}, std::span<const PauliFactor>(&z, 1U));
            observables.push_back(std::move(observable));
        }

        ExactEstimatorConfig serial_config;
        serial_config.max_pauli_terms = 1U;
        serial_config.tensor_worker_count = 1U;
        ExactEstimatorPlan serial_estimator(qubits, operations, serial_config);
        auto serial_batch = serial_estimator.compile(observables);
        require(serial_batch.tensor_observable_count() == observables.size(),
                "serial estimator did not route the bounded batch through tensor execution");
        require(serial_batch.tensor_worker_count() == 1U,
                "serial estimator did not preserve the requested tensor worker count");

        ExactEstimatorConfig parallel_config = serial_config;
        parallel_config.tensor_worker_count = 4U;
        ExactEstimatorPlan parallel_estimator(qubits, operations, parallel_config);
        auto parallel_batch = parallel_estimator.compile(observables);
        require(parallel_batch.tensor_observable_count() == observables.size(),
                "parallel estimator lost tensor-routable observables");
        require(parallel_batch.tensor_worker_count() == 4U,
                "parallel estimator did not materialize the requested tensor lanes");

        auto serial_workspace = serial_batch.workspace();
        auto parallel_workspace = parallel_batch.workspace();
        std::vector<ExactEstimatorResult> serial_results(observables.size());
        std::vector<ExactEstimatorResult> parallel_results(observables.size());
        serial_batch.estimate(serial_results, serial_workspace);
        parallel_batch.estimate(parallel_results, parallel_workspace);

        QRegister state(qubits);
        OperationPlan state_plan(operations, false);
        state_plan.execute(state);
        for (std::size_t index = 0; index < observables.size(); ++index) {
            const QComplex reference = observables[index].expectation(state);
            require_close(serial_results[index].value, reference, 2e-12,
                          "serial tensor estimator differs from QRegister");
            require_close(parallel_results[index].value, reference, 2e-12,
                          "parallel tensor estimator differs from QRegister");
            require_close(parallel_results[index].value, serial_results[index].value, 0.0,
                          "parallel tensor estimator changed exact arithmetic");
        }
        require(parallel_workspace.estimated_bytes() > serial_workspace.estimated_bytes(),
                "parallel tensor estimator did not account for lane workspaces");

        bool rejected = false;
        try {
            qubit::ExactEstimatorBatchWorkspace wrong_workspace;
            parallel_batch.estimate(parallel_results, wrong_workspace);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "parallel estimator accepted an unbound workspace");
    }

    {
        const std::vector<Operation> operations{
            {OperationCode::Rx, 0U, 0U, 0.37, 0.0},
        };
        PauliObservable observable(1U, PauliPropagationConfig{1U});
        const PauliFactor z{0U, PauliAxis::Z};
        observable.add_term({1.0, 0.0}, std::span<const PauliFactor>(&z, 1U));
        ExactEstimatorPlan estimator(1U, operations);
        const ExactEstimatorResult result = estimator.estimate(observable);
        require(result.route == ExactExecutionRoute::TensorNetwork,
                "estimator did not fall back from Pauli growth to tensor contraction");
        require_close(result.value, {std::cos(0.37), 0.0}, 1e-12,
                      "tensor estimator fallback changed the expectation");
        require(!result.fallback_reason.empty(),
                "tensor estimator fallback did not retain its causal failure reason");
    }

    {
        const std::vector<Operation> operations{
            {OperationCode::Rx, 0U, 0U, 0.37, 0.0},
        };
        PauliObservable observable(1U, PauliPropagationConfig{1U});
        const PauliFactor z{0U, PauliAxis::Z};
        observable.add_term({1.0, 0.0}, std::span<const PauliFactor>(&z, 1U));
        ExactEstimatorConfig config;
        config.execution.tensor = TensorNetworkConfig{2U, 100U};
        config.max_pauli_terms = 1U;
        ExactEstimatorPlan estimator(1U, operations, config);
        const ExactEstimatorResult result = estimator.estimate(observable);
        require(result.route == ExactExecutionRoute::Register,
                "estimator did not fail closed to QRegister after structural collapse");
        require_close(result.value, {std::cos(0.37), 0.0}, 1e-12,
                      "QRegister estimator fallback changed the expectation");
        require(result.fallback_reason.find("CausalPauli") != std::string::npos &&
                    result.fallback_reason.find("TensorNetwork") != std::string::npos,
                "QRegister fallback did not retain both structural failure reasons");
    }

    {
        const std::vector<Operation> operations{
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::PhaseFlipTrajectory, 0U, 0U, 0.25, 0.1},
        };
        PauliObservable observable(1U);
        const PauliFactor x{0U, PauliAxis::X};
        observable.add_term({1.0, 0.0}, std::span<const PauliFactor>(&x, 1U));
        ExactEstimatorPlan estimator(1U, operations);
        const ExactEstimatorResult result = estimator.estimate(observable);
        QRegister state(1U);
        OperationPlan state_plan(operations, false);
        state_plan.execute(state);
        require(result.route == ExactExecutionRoute::Register,
                "trajectory estimator did not route to QRegister");
        require_close(result.value, observable.expectation(state), 1e-12,
                      "trajectory estimator fallback changed the sampled state");
    }

    {
        constexpr std::size_t qubits = 100U;
        std::vector<Operation> operations;
        operations.reserve(qubits + qubits / 2U);
        std::vector<double> angles(qubits);
        for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
            angles[qubit] = 0.0017 * static_cast<double>(qubit + 1U);
            operations.push_back({
                OperationCode::Ry,
                static_cast<qubit::QubitId>(qubit),
                0U,
                angles[qubit],
                0.0,
            });
        }
        for (std::size_t qubit = 0; qubit < qubits; qubit += 2U) {
            operations.push_back({
                OperationCode::Cnot,
                static_cast<qubit::QubitId>(qubit),
                static_cast<qubit::QubitId>(qubit + 1U),
                0.0,
                0.0,
            });
        }
        PauliObservable observable(qubits, PauliPropagationConfig{65'536U});
        const PauliFactor z{73U, PauliAxis::Z};
        observable.add_term({1.0, 0.0}, std::span<const PauliFactor>(&z, 1U));
        ExactEstimatorPlan estimator(qubits, operations);
        const ExactEstimatorResult result = estimator.estimate(observable);
        const double expected = std::cos(angles[72U]) * std::cos(angles[73U]);
        require_close(result.value, {expected, 0.0}, 2e-11,
                      "100-qubit exact estimator changed the analytical expectation");
        require(result.route != ExactExecutionRoute::Register,
                "100-qubit local estimator materialized the general state fallback");
    }

    {
        ExactEstimatorPlan estimator(2U, {});
        PauliObservable observable(2U);
        observable.add_term({1.0, 0.0});
        bool rejected = false;
        try {
            std::vector<ExactEstimatorResult> results(2U);
            estimator.estimate(
                std::span<const PauliObservable>(&observable, 1U), results);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "estimator accepted a mismatched result span");

        PauliObservable wrong_width(3U);
        wrong_width.add_term({1.0, 0.0});
        rejected = false;
        try {
            static_cast<void>(estimator.estimate(wrong_width));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "estimator accepted a mismatched observable width");

        ExactEstimatorConfig invalid_config;
        invalid_config.tensor_worker_count = 129U;
        rejected = false;
        try {
            static_cast<void>(ExactEstimatorPlan(2U, {}, invalid_config));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "estimator accepted an unbounded tensor worker count");
    }

    return 0;
}
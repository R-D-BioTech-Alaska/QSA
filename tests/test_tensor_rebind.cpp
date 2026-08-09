#include "qubit/qtensor_rebind.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

using qubit::ExactParameterShiftConfig;
using qubit::ExactParameterShiftPlan;
using qubit::Operation;
using qubit::OperationCode;
using qubit::OperationPlan;
using qubit::ParameterizedOperation;
using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
using qubit::PauliPropagationConfig;
using qubit::QComplex;
using qubit::QRegister;
using qubit::QStateError;
using qubit::TensorExpectationPlan;
using qubit::TensorExpectationRebindPlan;
using qubit::TensorNetworkCircuit;
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

std::vector<Operation> circuit_operations(
    std::size_t qubits,
    double scale,
    bool alternate_single_qubit = false) {
    std::vector<Operation> operations;
    for (std::size_t layer = 0; layer < 3U; ++layer) {
        for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
            operations.push_back({
                alternate_single_qubit ? OperationCode::Rx : OperationCode::Ry,
                static_cast<qubit::QubitId>(qubit),
                0U,
                scale * 0.017 * static_cast<double>((layer + 1U) * (qubit + 2U)),
                0.0,
            });
            operations.push_back({
                OperationCode::Rz,
                static_cast<qubit::QubitId>(qubit),
                0U,
                -scale * 0.011 * static_cast<double>((layer + 2U) * (qubit + 1U)),
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

std::vector<PauliObservable> observables(std::size_t qubits) {
    std::vector<PauliObservable> result;

    PauliObservable first(qubits, PauliPropagationConfig{65'536U});
    const std::vector<PauliFactor> first_factors{
        {0U, PauliAxis::X},
        {1U, PauliAxis::Y},
    };
    first.add_term({0.7, -0.1}, first_factors);
    result.push_back(std::move(first));

    PauliObservable second(qubits, PauliPropagationConfig{65'536U});
    const PauliFactor z{2U, PauliAxis::Z};
    second.add_term({-0.4, 0.0}, std::span<const PauliFactor>(&z, 1U));
    second.add_term({0.125, 0.0});
    result.push_back(std::move(second));

    PauliObservable third(qubits, PauliPropagationConfig{65'536U});
    const std::vector<PauliFactor> third_factors{
        {3U, PauliAxis::X},
        {4U, PauliAxis::Z},
        {5U, PauliAxis::Y},
    };
    third.add_term({0.55, 0.05}, third_factors);
    result.push_back(std::move(third));

    PauliObservable identity(qubits);
    identity.add_term({0.75, -0.2});
    result.push_back(std::move(identity));

    return result;
}

std::vector<QComplex> qregister_values(
    std::size_t qubits,
    const std::vector<Operation>& operations,
    const std::vector<PauliObservable>& queries) {
    QRegister state(qubits);
    OperationPlan plan(operations, false);
    plan.execute(state);
    std::vector<QComplex> values(queries.size());
    for (std::size_t index = 0; index < queries.size(); ++index) {
        values[index] = queries[index].expectation(state);
    }
    return values;
}

std::vector<Operation> bind_parameterized(
    const std::vector<ParameterizedOperation>& operations,
    const std::vector<double>& parameters) {
    std::vector<Operation> result;
    result.reserve(operations.size());
    for (const ParameterizedOperation& templated : operations) {
        Operation operation = templated.operation;
        if (templated.parameter_slot >= 0) {
            operation.parameter =
                parameters[static_cast<std::size_t>(templated.parameter_slot)];
        }
        result.push_back(operation);
    }
    return result;
}

std::vector<QComplex> qregister_parameter_shift(
    std::size_t qubits,
    const std::vector<ParameterizedOperation>& operations,
    const std::vector<double>& parameters,
    const std::vector<PauliObservable>& queries,
    std::size_t parameter_count) {
    constexpr double shift = 1.57079632679489661923;
    std::vector<QComplex> gradient(queries.size() * parameter_count);
    std::vector<Operation> bound = bind_parameterized(operations, parameters);
    for (std::size_t operation_index = 0U;
         operation_index < operations.size();
         ++operation_index) {
        if (operations[operation_index].parameter_slot < 0) {
            continue;
        }
        const std::size_t parameter =
            static_cast<std::size_t>(operations[operation_index].parameter_slot);
        const double original = bound[operation_index].parameter;

        bound[operation_index].parameter = original + shift;
        const std::vector<QComplex> plus =
            qregister_values(qubits, bound, queries);
        bound[operation_index].parameter = original - shift;
        const std::vector<QComplex> minus =
            qregister_values(qubits, bound, queries);
        bound[operation_index].parameter = original;

        for (std::size_t observable = 0U; observable < queries.size(); ++observable) {
            gradient[observable * parameter_count + parameter] +=
                (plus[observable] - minus[observable]) * 0.5;
        }
    }
    return gradient;
}

}  // namespace

int main() {
    constexpr std::size_t qubits = 6U;
    const std::vector<PauliObservable> queries = observables(qubits);
    const std::vector<Operation> initial_operations = circuit_operations(qubits, 1.0);
    const TensorNetworkCircuit initial_circuit(qubits, initial_operations);
    TensorExpectationRebindPlan reusable(initial_circuit, queries);
    auto reusable_workspace = reusable.workspace();

    const std::size_t initial_steps = reusable.step_count();
    const auto initial_stats = reusable.stats();
    std::vector<QComplex> reusable_values(queries.size());
    reusable.expectations(reusable_values, reusable_workspace);
    const std::vector<QComplex> initial_reference =
        qregister_values(qubits, initial_operations, queries);
    for (std::size_t index = 0; index < queries.size(); ++index) {
        require_close(reusable_values[index], initial_reference[index], 5e-11,
                      "initial reusable expectation differs from QRegister");
    }

    for (std::size_t sweep = 0; sweep < 24U; ++sweep) {
        const double scale = 0.35 + 0.071 * static_cast<double>(sweep + 1U);
        const bool alternate = (sweep % 5U) == 4U;
        const std::vector<Operation> operations =
            circuit_operations(qubits, scale, alternate);
        const TensorNetworkCircuit circuit(qubits, operations);
        reusable.rebind(circuit);
        reusable.expectations(reusable_values, reusable_workspace);

        TensorExpectationPlan fresh(circuit, queries);
        auto fresh_workspace = fresh.workspace();
        std::vector<QComplex> fresh_values(queries.size());
        fresh.expectations(fresh_values, fresh_workspace);
        const std::vector<QComplex> register_values =
            qregister_values(qubits, operations, queries);

        for (std::size_t index = 0; index < queries.size(); ++index) {
            require_close(reusable_values[index], fresh_values[index], 0.0,
                          "rebound expectation differs from a fresh tensor plan");
            require_close(reusable_values[index], register_values[index], 5e-11,
                          "rebound expectation differs from QRegister");
        }
        require(reusable.step_count() == initial_steps,
                "rebind changed the compiled contraction schedule");
        require(reusable.stats().peak_contraction_entries ==
                    initial_stats.peak_contraction_entries,
                "rebind changed the contraction resource certificate");
    }
    require(reusable.rebind_count() == 24U,
            "rebind count does not match successful topology updates");

    std::vector<QComplex> before_failure(queries.size());
    reusable.expectations(before_failure, reusable_workspace);

    {
        std::vector<Operation> changed = circuit_operations(qubits, 2.3);
        auto gate = std::find_if(
            changed.begin(), changed.end(),
            [](const Operation& operation) { return operation.code == OperationCode::Cnot; });
        require(gate != changed.end(), "topology test did not find a CNOT");
        gate->second = static_cast<qubit::QubitId>((gate->second + 2U) % qubits);
        if (gate->second == gate->first) {
            gate->second = static_cast<qubit::QubitId>((gate->second + 1U) % qubits);
        }
        bool rejected = false;
        try {
            reusable.rebind(TensorNetworkCircuit(qubits, changed));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "rebind accepted a changed circuit topology");
    }

    {
        const std::vector<Operation> same = circuit_operations(qubits, 2.3);
        bool rejected = false;
        try {
            reusable.rebind(TensorNetworkCircuit(
                qubits,
                same,
                TensorNetworkConfig{1U << 19U, 1'000'000U}));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "rebind accepted a changed resource contract");
    }

    {
        std::vector<Operation> changed = circuit_operations(qubits, 2.3);
        changed.push_back({OperationCode::H, 0U, 0U, 0.0, 0.0});
        bool rejected = false;
        try {
            reusable.rebind(TensorNetworkCircuit(qubits, changed));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "rebind accepted a changed operation count");
    }

    std::vector<QComplex> after_failure(queries.size());
    reusable.expectations(after_failure, reusable_workspace);
    for (std::size_t index = 0; index < queries.size(); ++index) {
        require_close(after_failure[index], before_failure[index], 0.0,
                      "failed rebind mutated the accepted tensor plan");
    }
    require(reusable.rebind_count() == 24U,
            "failed rebind changed the successful rebind count");

    {
        const std::vector<ParameterizedOperation> operations{
            {{OperationCode::Ry, 0U, 0U, 0.0, 0.0}, 0, -1},
            {{OperationCode::Ry, 1U, 0U, 0.0, 0.0}, 0, -1},
        };
        PauliObservable zz(2U);
        const std::vector<PauliFactor> factors{
            {0U, PauliAxis::Z},
            {1U, PauliAxis::Z},
        };
        zz.add_term({1.0, 0.0}, factors);
        const std::vector<PauliObservable> gradient_queries{zz};
        ExactParameterShiftPlan plan(
            2U,
            operations,
            gradient_queries,
            ExactParameterShiftConfig{{}, 2U});
        auto workspace = plan.workspace();
        const std::vector<double> parameters{0.37};
        std::vector<QComplex> values(1U);
        std::vector<QComplex> gradients(1U);
        plan.value_and_gradient(parameters, values, gradients, workspace);

        require_close(
            values[0],
            {std::cos(parameters[0]) * std::cos(parameters[0]), 0.0},
            2e-12,
            "shared-parameter value is incorrect");
        require_close(
            gradients[0],
            {-std::sin(2.0 * parameters[0]), 0.0},
            2e-12,
            "shared-parameter gradient did not sum gate occurrences");
        require(plan.parameter_count() == 1U,
                "shared parameter count changed");
        require(plan.parameterized_operation_count() == 2U,
                "shared parameter occurrence count changed");
        require(plan.stats().value_and_gradient_evaluations == 5U,
                "shared parameter evaluation count is incorrect");
        require(workspace.rebind_count() == 5U,
                "shared parameter workspace rebind count is incorrect");
    }

    {
        const std::vector<ParameterizedOperation> operations{
            {{OperationCode::Ry, 0U, 0U, 0.0, 0.0}, 0, -1},
            {{OperationCode::Rz, 1U, 0U, 0.0, 0.0}, 1, -1},
            {{OperationCode::Cnot, 0U, 1U, 0.0, 0.0}, -1, -1},
            {{OperationCode::Rx, 2U, 0U, 0.0, 0.0}, 0, -1},
            {{OperationCode::Ry, 3U, 0U, 0.0, 0.0}, 2, -1},
            {{OperationCode::Cnot, 2U, 3U, 0.0, 0.0}, -1, -1},
            {{OperationCode::Rz, 4U, 0U, 0.0, 0.0}, 1, -1},
            {{OperationCode::Cnot, 4U, 5U, 0.0, 0.0}, -1, -1},
            {{OperationCode::Ry, 5U, 0U, 0.0, 0.0}, 2, -1},
        };
        const std::vector<double> parameters{0.31, -0.52, 0.73};
        ExactParameterShiftPlan plan(
            qubits,
            operations,
            queries,
            ExactParameterShiftConfig{{1U << 18U, 100'000U}, 4U});
        auto workspace = plan.workspace();
        std::vector<QComplex> values(queries.size());
        std::vector<QComplex> gradients(queries.size() * parameters.size());
        plan.value_and_gradient(parameters, values, gradients, workspace);

        const std::vector<Operation> bound = bind_parameterized(operations, parameters);
        const std::vector<QComplex> reference_values =
            qregister_values(qubits, bound, queries);
        const std::vector<QComplex> reference_gradients =
            qregister_parameter_shift(
                qubits, operations, parameters, queries, parameters.size());

        for (std::size_t observable = 0U; observable < queries.size(); ++observable) {
            require_close(
                values[observable], reference_values[observable], 5e-11,
                "parameter-shift value differs from QRegister");
            for (std::size_t parameter = 0U;
                 parameter < parameters.size();
                 ++parameter) {
                const std::size_t index = plan.gradient_index(observable, parameter);
                require_close(
                    gradients[index], reference_gradients[index], 7e-11,
                    "parameter-shift gradient differs from direct QRegister shifts");
            }
        }
        require(plan.worker_count() == 4U,
                "parameter-shift worker count did not honor its bound");
        require(plan.stats().tensor.peak_contraction_entries <= (1U << 18U),
                "parameter-shift plan exceeded its tensor certificate");

        std::vector<QComplex> preserved_values = values;
        std::vector<QComplex> preserved_gradients = gradients;
        bool rejected = false;
        try {
            const std::vector<double> too_short{parameters[0], parameters[1]};
            plan.value_and_gradient(
                too_short, values, gradients, workspace);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "parameter-shift plan accepted too few parameters");
        require(values == preserved_values,
                "rejected parameter-shift call mutated value output");
        require(gradients == preserved_gradients,
                "rejected parameter-shift call mutated gradient output");

        ExactParameterShiftPlan foreign(
            qubits,
            operations,
            queries,
            ExactParameterShiftConfig{{1U << 18U, 100'000U}, 4U});
        auto foreign_workspace = foreign.workspace();
        rejected = false;
        try {
            plan.value_and_gradient(
                parameters, values, gradients, foreign_workspace);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "parameter-shift plan accepted a foreign workspace");
    }

    {
        const std::vector<ParameterizedOperation> invalid{
            {{OperationCode::H, 0U, 0U, 0.0, 0.0}, 0, -1},
        };
        PauliObservable z(1U);
        const PauliFactor factor{0U, PauliAxis::Z};
        z.add_term({1.0, 0.0}, std::span<const PauliFactor>(&factor, 1U));
        const std::vector<PauliObservable> single{z};
        bool rejected = false;
        try {
            static_cast<void>(ExactParameterShiftPlan(1U, invalid, single));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "parameter-shift plan accepted a parameterized H gate");
    }

    {
        const std::vector<ParameterizedOperation> invalid{
            {{OperationCode::Ry, 0U, 0U, 0.0, 0.0}, 0, 1},
        };
        PauliObservable z(1U);
        const PauliFactor factor{0U, PauliAxis::Z};
        z.add_term({1.0, 0.0}, std::span<const PauliFactor>(&factor, 1U));
        const std::vector<PauliObservable> single{z};
        bool rejected = false;
        try {
            static_cast<void>(ExactParameterShiftPlan(1U, invalid, single));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "parameter-shift plan accepted a stochastic sample slot");
    }

    return 0;
}

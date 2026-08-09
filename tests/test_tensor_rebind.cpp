#include "qubit/qtensor_rebind.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

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

    return 0;
}

#include "qubit/qtensor.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using qubit::Operation;
using qubit::OperationCode;
using qubit::OperationPlan;
using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
using qubit::QComplex;
using qubit::QRegister;
using qubit::QStateError;
using qubit::TensorContractionStats;
using qubit::TensorExpectationPlan;
using qubit::TensorNetworkCircuit;
using qubit::TensorNetworkConfig;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void require_close(QComplex actual, QComplex expected, double tolerance, const char* message) {
    if (!qubit::almost_equal(actual, expected, tolerance)) throw std::runtime_error(message);
}

Operation random_operation(std::mt19937_64& rng, std::size_t qubits) {
    std::uniform_int_distribution<int> kind(0, 8);
    std::uniform_int_distribution<std::size_t> qubit(0U, qubits - 1U);
    std::uniform_real_distribution<double> angle(-1.2, 1.2);
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
            while (second == static_cast<std::size_t>(operation.first)) second = qubit(rng);
            operation.second = static_cast<qubit::QubitId>(second);
            break;
        }
    }
    return operation;
}

PauliObservable random_observable(std::mt19937_64& rng, std::size_t qubits) {
    PauliObservable observable(qubits);
    std::uniform_int_distribution<std::size_t> qubit(0U, qubits - 1U);
    std::uniform_int_distribution<int> axis(1, 3);
    std::uniform_real_distribution<double> coefficient(-1.0, 1.0);
    for (std::size_t term = 0; term < 4U; ++term) {
        std::vector<PauliFactor> factors;
        const std::size_t support = 1U + term % 3U;
        std::vector<std::uint8_t> used(qubits, 0U);
        while (factors.size() < support) {
            const std::size_t selected = qubit(rng);
            if (used[selected] != 0U) continue;
            used[selected] = 1U;
            factors.push_back({
                static_cast<qubit::QubitId>(selected),
                static_cast<PauliAxis>(axis(rng)),
            });
        }
        observable.add_term({coefficient(rng), 0.25 * coefficient(rng)}, factors);
    }
    return observable;
}

}  // namespace

int main() {
    {
        const Operation h{OperationCode::H, 0U, 0U, 0.0, 0.0};
        TensorNetworkCircuit circuit(1U, std::span<const Operation>(&h, 1U));

        for (const auto [axis, expected] : std::vector<std::pair<PauliAxis, QComplex>>{
                 {PauliAxis::X, {1.0, 0.0}},
                 {PauliAxis::Y, {0.0, 0.0}},
                 {PauliAxis::Z, {0.0, 0.0}},
             }) {
            PauliObservable observable(1U);
            const PauliFactor factor{0U, axis};
            observable.add_term({1.0, 0.0}, std::span<const PauliFactor>(&factor, 1U));
            const TensorExpectationPlan plan(circuit, observable);
            require_close(plan.expectation(), expected, 1e-12, "single-qubit tensor expectation");
        }
    }

    {
        const std::vector<Operation> bell{
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::Cnot, 0U, 1U, 0.0, 0.0},
        };
        TensorNetworkCircuit circuit(2U, bell);
        PauliObservable observable(2U);
        const std::vector<PauliFactor> xx{{0U, PauliAxis::X}, {1U, PauliAxis::X}};
        const std::vector<PauliFactor> yy{{0U, PauliAxis::Y}, {1U, PauliAxis::Y}};
        const std::vector<PauliFactor> zz{{0U, PauliAxis::Z}, {1U, PauliAxis::Z}};
        observable.add_term({0.5, 0.25}, xx);
        observable.add_term({0.25, -0.5}, yy);
        observable.add_term({1.5, 0.0}, zz);
        observable.add_term({0.125, 0.0});
        const TensorExpectationPlan plan(circuit, observable);
        auto workspace = plan.workspace();
        const QComplex expected =
            QComplex{0.5, 0.25} - QComplex{0.25, -0.5} +
            QComplex{1.5, 0.0} + QComplex{0.125, 0.0};
        require_close(plan.expectation(workspace), expected, 1e-12,
                      "Bell Pauli combination expectation");
        require(workspace.estimated_bytes() > 0U, "expectation workspace byte estimate");
    }

    {
        constexpr std::size_t qubits = 6U;
        std::mt19937_64 rng(0x7173615f657870ULL);
        for (std::size_t trial = 0; trial < 8U; ++trial) {
            std::vector<Operation> operations;
            operations.reserve(28U);
            for (std::size_t index = 0; index < 28U; ++index) {
                operations.push_back(random_operation(rng, qubits));
            }
            TensorNetworkCircuit circuit(
                qubits, operations, TensorNetworkConfig{1U << 18U, 100'000U});
            QRegister state(qubits);
            OperationPlan qregister_plan(operations, false);
            qregister_plan.execute(state);

            std::vector<PauliObservable> observables;
            observables.reserve(6U);
            for (std::size_t observable_index = 0; observable_index < 6U; ++observable_index) {
                observables.push_back(random_observable(rng, qubits));
                const PauliObservable& observable = observables.back();
                const TensorExpectationPlan plan(circuit, observable);
                auto workspace = plan.workspace();
                const QComplex reference = observable.expectation(state);
                const QComplex tensor = plan.expectation(workspace);
                require_close(tensor, reference, 4e-11,
                              "random tensor expectation differs from QRegister");
            }

            const TensorExpectationPlan batch_plan = circuit.compile_expectations(
                std::span<const PauliObservable>(observables.data(), observables.size()));
            require(batch_plan.observable_count() == observables.size(),
                    "batched expectation observable count changed");
            auto batch_workspace = batch_plan.workspace();
            std::vector<QComplex> batch_results(observables.size());
            batch_plan.expectations(batch_results, batch_workspace);
            for (std::size_t observable_index = 0;
                 observable_index < observables.size();
                 ++observable_index) {
                require_close(
                    batch_results[observable_index],
                    observables[observable_index].expectation(state),
                    4e-11,
                    "batched tensor expectation differs from QRegister");
            }

            bool rejected = false;
            try {
                static_cast<void>(batch_plan.expectation(batch_workspace));
            } catch (const QStateError&) {
                rejected = true;
            }
            require(rejected,
                    "batched expectation accepted a single-result query");

            rejected = false;
            try {
                std::vector<QComplex> wrong_results(observables.size() - 1U);
                batch_plan.expectations(wrong_results, batch_workspace);
            } catch (const QStateError&) {
                rejected = true;
            }
            require(rejected,
                    "batched expectation accepted a mismatched result span");
        }
    }

    {
        constexpr std::size_t qubits = 100U;
        std::vector<Operation> operations;
        operations.reserve(qubits + qubits / 2U);
        std::vector<double> angles(qubits);
        for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
            angles[qubit] = 0.0025 * static_cast<double>(qubit + 1U);
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
        TensorNetworkCircuit circuit(
            qubits, operations, TensorNetworkConfig{1U << 16U, 100'000U});
        PauliObservable observable(qubits);
        const PauliFactor target{73U, PauliAxis::Z};
        observable.add_term({1.0, 0.0}, std::span<const PauliFactor>(&target, 1U));
        const TensorExpectationPlan plan(circuit, observable);
        TensorContractionStats stats;
        const double expected = std::cos(angles[72U]) * std::cos(angles[73U]);
        require_close(plan.expectation(&stats), {expected, 0.0}, 2e-11,
                      "100-qubit pair-circuit expectation");
        require(stats.peak_contraction_entries <= (1U << 16U),
                "100-qubit expectation exceeded its resource certificate");
        require(stats.eliminated_variables > 0U,
                "100-qubit expectation did not execute a contraction");
        require(stats.source_factors < qubits,
                "100-qubit expectation failed to trim the causal cone");
    }

    {
        const std::vector<Operation> operations{
            {OperationCode::Ry, 1U, 0U, 0.41, 0.0},
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::Cnot, 1U, 0U, 0.0, 0.0},
            {OperationCode::Rz, 1U, 0U, -0.73, 0.0},
            {OperationCode::Ry, 2U, 0U, 0.52, 0.0},
            {OperationCode::Cnot, 2U, 3U, 0.0, 0.0},
            {OperationCode::Rx, 3U, 0U, -0.29, 0.0},
        };
        TensorNetworkCircuit circuit(4U, operations);
        QRegister state(4U);
        OperationPlan qregister_plan(operations, false);
        qregister_plan.execute(state);

        PauliObservable observable(4U);
        const PauliFactor target{0U, PauliAxis::Z};
        observable.add_term({1.0, 0.0}, std::span<const PauliFactor>(&target, 1U));
        const TensorExpectationPlan plan(circuit, observable);
        TensorContractionStats stats;
        require_close(
            plan.expectation(&stats),
            observable.expectation(state),
            2e-12,
            "causal tensor trimming changed an exact expectation");
        require(
            stats.source_factors < 2U * circuit.factor_count() + 1U,
            "causal tensor trimming retained the complete doubled circuit");
    }

    {
        TensorNetworkCircuit circuit(3U);
        PauliObservable zero(3U);
        PauliObservable identity(3U);
        identity.add_term({0.375, -0.125});
        const std::vector<PauliObservable> observables{zero, identity};
        const TensorExpectationPlan plan(
            circuit,
            std::span<const PauliObservable>(observables.data(), observables.size()));
        auto workspace = plan.workspace();
        std::vector<QComplex> results(observables.size());
        plan.expectations(results, workspace);
        require_close(results[0], {}, 0.0, "empty observable did not evaluate to zero");
        require_close(results[1], {0.375, -0.125}, 0.0,
                      "identity-only observable changed in batch evaluation");
    }

    {
        const std::vector<Operation> operations{
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::Cnot, 0U, 1U, 0.0, 0.0},
            {OperationCode::Cnot, 1U, 2U, 0.0, 0.0},
            {OperationCode::Cnot, 2U, 3U, 0.0, 0.0},
        };
        PauliObservable observable(4U);
        const PauliFactor x{0U, PauliAxis::X};
        observable.add_term({1.0, 0.0}, std::span<const PauliFactor>(&x, 1U));
        bool rejected = false;
        try {
            TensorNetworkCircuit constrained(
                4U, operations, TensorNetworkConfig{4U, 1'000U});
            static_cast<void>(constrained.compile_expectation(observable));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "expectation compiler ignored contraction resource limit");
    }

    {
        TensorNetworkCircuit two(2U);
        PauliObservable observable(2U);
        const PauliFactor z{0U, PauliAxis::Z};
        observable.add_term({1.0, 0.0}, std::span<const PauliFactor>(&z, 1U));
        const TensorExpectationPlan two_plan(two, observable);

        PauliObservable different(2U);
        const std::vector<PauliFactor> xx{{0U, PauliAxis::X}, {1U, PauliAxis::X}};
        different.add_term({1.0, 0.0}, xx);
        auto wrong_workspace = TensorExpectationPlan(two, different).workspace();
        bool rejected = false;
        try {
            static_cast<void>(two_plan.expectation(wrong_workspace));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "expectation plan accepted a foreign workspace");

        PauliObservable wrong_width(3U);
        wrong_width.add_term({1.0, 0.0});
        rejected = false;
        try {
            static_cast<void>(TensorExpectationPlan(two, wrong_width));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "expectation plan accepted a mismatched observable width");
    }

    return 0;
}

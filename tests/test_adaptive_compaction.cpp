#include "qubit/qadaptive.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using qubit::AdaptiveCompactionConfig;
using qubit::AdaptiveOperationPlan;
using qubit::Operation;
using qubit::OperationCode;
using qubit::QComplex;
using qubit::QRegister;
using qubit::QStateConfig;
using qubit::QStateError;
using qubit::QubitId;

[[nodiscard]] QRegister make_entangled(std::size_t qubits) {
    QStateConfig config;
    config.max_component_qubits = 30;
    config.max_dense_amplitudes = 1ULL << 22;
    QRegister state(qubits, config);
    for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
        state.apply_ry(
            static_cast<QubitId>(qubit),
            0.19 + 0.037 * static_cast<double>(qubit));
        state.apply_rz(
            static_cast<QubitId>(qubit),
            -0.31 + 0.021 * static_cast<double>(qubit));
    }
    for (std::size_t qubit = 1; qubit < qubits; ++qubit) {
        state.apply_cnot(
            static_cast<QubitId>(qubit - 1U),
            static_cast<QubitId>(qubit));
    }
    for (std::size_t qubit = 2; qubit < qubits; qubit += 3U) {
        state.apply_cz(0, static_cast<QubitId>(qubit));
    }
    return state;
}

[[nodiscard]] std::vector<Operation> make_operations(
    std::size_t qubits,
    std::size_t count) {
    std::mt19937_64 generator(0xB4A9351AULL);
    std::vector<Operation> operations;
    operations.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        QubitId first = static_cast<QubitId>(generator() % qubits);
        QubitId second = static_cast<QubitId>(generator() % qubits);
        if (first == second) {
            second = static_cast<QubitId>((second + 1U) % qubits);
        }
        Operation operation;
        operation.code = (generator() & 1U) == 0U ? OperationCode::Cnot : OperationCode::Cz;
        operation.first = first;
        operation.second = second;
        operations.push_back(operation);
    }
    return operations;
}

void execute_reference(QRegister& state, const std::vector<Operation>& operations) {
    for (const Operation& operation : operations) {
        if (operation.code == OperationCode::Cnot) {
            state.apply_cnot_structured(operation.first, operation.second);
        } else {
            state.apply_cz_structured(operation.first, operation.second);
        }
    }
}

[[nodiscard]] double fidelity(
    const std::vector<QComplex>& first,
    const std::vector<QComplex>& second) {
    if (first.size() != second.size()) {
        return 0.0;
    }
    QComplex overlap{};
    for (std::size_t index = 0; index < first.size(); ++index) {
        overlap += second[index].conjugate() * first[index];
    }
    return overlap.norm2();
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    {
        constexpr std::size_t qubits = 16U;
        constexpr std::size_t operation_count = 1'600U;
        const QRegister seed = make_entangled(qubits);
        const auto operations = make_operations(qubits, operation_count);

        QRegister reference = seed;
        execute_reference(reference, operations);

        AdaptiveCompactionConfig config;
        config.enabled = true;
        config.warmup_checks = 8U;
        config.audit_interval = 64U;
        config.prior_success = 1.0;
        config.prior_failure = 3.0;
        config.minimum_posterior_success = 0.20;
        AdaptiveOperationPlan plan(operations, config);

        QRegister adaptive = seed;
        plan.execute(adaptive);
        const auto metrics = plan.metrics();
        require(adaptive.validate(), "adaptive plan produced an invalid state");
        require(
            fidelity(reference.materialize(20), adaptive.materialize(20)) > 1.0 - 2e-11,
            "adaptive compaction changed the quantum state");
        require(metrics.skips > 0U, "adaptive compaction did not skip any checks");
        require(metrics.checks < operation_count / 2U,
                "adaptive compaction did not reduce check count");
    }

    {
        QRegister adaptive(4);
        adaptive.apply_h(0);
        adaptive.apply_cnot(0, 1);
        adaptive.apply_cnot(0, 2);
        adaptive.apply_cnot(0, 3);
        QRegister reference = adaptive;

        std::vector<Operation> operations(4);
        operations[0] = Operation{OperationCode::Cnot, 0, 3};
        operations[1] = Operation{OperationCode::Cnot, 0, 2};
        operations[2] = Operation{OperationCode::Cnot, 0, 1};
        operations[3] = Operation{OperationCode::Cnot, 1, 2};

        AdaptiveCompactionConfig config;
        config.warmup_checks = 0U;
        config.audit_interval = 3U;
        config.prior_success = 1.0;
        config.prior_failure = 100.0;
        config.minimum_posterior_success = 0.20;
        AdaptiveOperationPlan plan(operations, config);
        plan.execute(adaptive);
        execute_reference(reference, operations);

        const auto metrics = plan.metrics();
        require(
            fidelity(reference.materialize(8), adaptive.materialize(8)) > 1.0 - 2e-12,
            "audit recovery changed the quantum state");
        require(metrics.skips == 3U, "audit recovery did not skip the expected checks");
        require(metrics.audit_checks == 1U, "audit recovery did not force an exact audit");
        require(metrics.successes >= 1U, "audit recovery did not recover separability");
        require(adaptive.component_count() > 1U,
                "audit recovery did not split the component");
    }

    {
        QRegister state(8);
        std::vector<Operation> operations{
            Operation{OperationCode::Cnot, 0, 4},
            Operation{OperationCode::Cz, 1, 5},
            Operation{OperationCode::Swap, 2, 6},
        };
        AdaptiveCompactionConfig config;
        config.enabled = false;
        AdaptiveOperationPlan plan(operations, config);
        QRegister reference = state;
        execute_reference(reference, {operations[0], operations[1]});
        reference.apply_swap_structured(2, 6);
        plan.execute(state);
        require(
            fidelity(reference.materialize(12), state.materialize(12)) > 1.0 - 1e-12,
            "disabled adaptive plan changed the exact result");
    }

    {
        bool rejected = false;
        try {
            AdaptiveCompactionConfig config;
            config.prior_success = 0.0;
            const std::vector<Operation> operations;
            AdaptiveOperationPlan invalid(operations, config);
            (void)invalid;
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "invalid adaptive prior was accepted");
    }

    std::cout << "adaptive compaction tests passed\n";
    return 0;
}

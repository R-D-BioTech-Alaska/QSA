#include "qubit/qadaptive.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::AdaptiveCompactionConfig;
using qubit::AdaptiveOperationPlan;
using qubit::Operation;
using qubit::OperationCode;
using qubit::QRegister;
using qubit::QStateConfig;
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
        operations.push_back(Operation{
            (generator() & 1U) == 0U ? OperationCode::Cnot : OperationCode::Cz,
            first,
            second,
        });
    }
    return operations;
}

template <class Function>
[[nodiscard]] double timed_ms(Function&& function) {
    const auto start = Clock::now();
    function();
    const auto stop = Clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);
    for (const auto [qubits, operation_count] : {
             std::pair<std::size_t, std::size_t>{16U, 1'600U},
             {18U, 1'200U},
             {20U, 600U},
         }) {
        const QRegister seed = make_entangled(qubits);
        const auto operations = make_operations(qubits, operation_count);

        QRegister always = seed;
        const double always_ms = timed_ms([&] {
            for (const Operation& operation : operations) {
                if (operation.code == OperationCode::Cnot) {
                    always.apply_cnot_structured(operation.first, operation.second);
                } else {
                    always.apply_cz_structured(operation.first, operation.second);
                }
            }
        });

        AdaptiveCompactionConfig config;
        AdaptiveOperationPlan plan(operations, config);
        QRegister adaptive = seed;
        const double adaptive_ms = timed_ms([&] { plan.execute(adaptive); });
        const auto metrics = plan.metrics();
        std::cout << "qubits=" << qubits
                  << " operations=" << operation_count
                  << " always_ms=" << always_ms
                  << " adaptive_ms=" << adaptive_ms
                  << " speedup=" << always_ms / adaptive_ms
                  << " checks=" << metrics.checks
                  << " successes=" << metrics.successes
                  << " skips=" << metrics.skips
                  << " audits=" << metrics.audit_checks << '\n';
    }
    return 0;
}

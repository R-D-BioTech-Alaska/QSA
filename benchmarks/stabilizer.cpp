#include "qubit/qstabilizer.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::QRegister;
using qubit::QStateConfig;
using qubit::QubitId;
using qubit::StabilizerConfig;
using qubit::StabilizerOperation;
using qubit::StabilizerOperationCode;
using qubit::StabilizerState;

struct CliffordGate {
    std::uint8_t kind;
    QubitId first;
    QubitId second;
};

[[nodiscard]] std::vector<CliffordGate> gates(
    std::size_t qubits,
    std::size_t count,
    std::uint64_t seed) {
    std::mt19937_64 generator(seed);
    std::vector<CliffordGate> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        QubitId first = static_cast<QubitId>(generator() % qubits);
        QubitId second = static_cast<QubitId>(generator() % qubits);
        if (first == second) {
            second = static_cast<QubitId>((second + 1U) % qubits);
        }
        result.push_back(CliffordGate{
            static_cast<std::uint8_t>(generator() % 5U), first, second});
    }
    return result;
}

[[nodiscard]] std::vector<StabilizerOperation> batch_operations(
    const std::vector<CliffordGate>& operations) {
    std::vector<StabilizerOperation> result;
    result.reserve(operations.size());
    for (const CliffordGate& gate : operations) {
        StabilizerOperationCode code = StabilizerOperationCode::H;
        switch (gate.kind) {
            case 0U:
                code = StabilizerOperationCode::H;
                break;
            case 1U:
                code = StabilizerOperationCode::S;
                break;
            case 2U:
                code = StabilizerOperationCode::Cnot;
                break;
            case 3U:
                code = StabilizerOperationCode::Cz;
                break;
            default:
                code = StabilizerOperationCode::Swap;
                break;
        }
        result.push_back(StabilizerOperation{code, gate.first, gate.second});
    }
    return result;
}

void execute(StabilizerState& state, const std::vector<CliffordGate>& operations) {
    for (const CliffordGate& gate : operations) {
        switch (gate.kind) {
            case 0U:
                state.apply_h(gate.first);
                break;
            case 1U:
                state.apply_s(gate.first);
                break;
            case 2U:
                state.apply_cnot(gate.first, gate.second);
                break;
            case 3U:
                state.apply_cz(gate.first, gate.second);
                break;
            default:
                state.apply_swap(gate.first, gate.second);
                break;
        }
    }
}

void execute(QRegister& state, const std::vector<CliffordGate>& operations) {
    for (const CliffordGate& gate : operations) {
        switch (gate.kind) {
            case 0U:
                state.apply_h(gate.first);
                break;
            case 1U:
                state.apply_s(gate.first);
                break;
            case 2U:
                state.apply_cnot_structured(gate.first, gate.second);
                break;
            case 3U:
                state.apply_cz_structured(gate.first, gate.second);
                break;
            default:
                state.apply_swap_structured(gate.first, gate.second);
                break;
        }
    }
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
    const auto operations = gates(18U, 800U, 0x434C4946464F5244ULL);
    StabilizerState stabilizer(18U);
    const double stabilizer_ms = timed_ms([&] { execute(stabilizer, operations); });

    QStateConfig config;
    config.max_component_qubits = 24U;
    config.max_dense_amplitudes = 1ULL << 20;
    QRegister register_state(18U, config);
    const double register_ms = timed_ms([&] { execute(register_state, operations); });
    std::cout << "clifford qubits=18 gates=800 qregister_ms=" << register_ms
              << " stabilizer_ms=" << stabilizer_ms
              << " speedup=" << register_ms / stabilizer_ms
              << " stabilizer_bytes=" << stabilizer.estimated_bytes()
              << " qregister_bytes=" << register_state.estimated_bytes() << '\n';

    StabilizerConfig large_config;
    large_config.max_tableau_bytes = 1ULL << 29;
    large_config.max_batch_scratch_bytes = 1ULL << 29;
    const auto large_operations = gates(4096U, 100'000U, 0x4C41524745434C49ULL);
    const auto large_batch = batch_operations(large_operations);
    StabilizerState large_scalar(4096U, large_config);
    const double large_scalar_ms = timed_ms([&] {
        execute(large_scalar, large_operations);
    });
    StabilizerState large_batched(4096U, large_config);
    const double large_batch_ms = timed_ms([&] {
        large_batched.apply_batch(large_batch);
    });
    std::cout << "clifford qubits=4096 gates=100000 scalar_ms=" << large_scalar_ms
              << " batch_ms=" << large_batch_ms
              << " batch_speedup=" << large_scalar_ms / large_batch_ms
              << " bytes=" << large_batched.estimated_bytes() << '\n';
    return 0;
}

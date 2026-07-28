#include "qubit/qphase_graph.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::PhaseGraphConfig;
using qubit::PhaseGraphState;
using qubit::QComplex;
using qubit::QMatrix4;
using qubit::QRegister;
using qubit::QStateConfig;
using qubit::QubitId;

struct PhaseGate {
    std::uint8_t kind;
    QubitId first;
    QubitId second;
    double angle;
};

[[nodiscard]] std::vector<PhaseGate> gates(
    std::size_t qubits,
    std::size_t count,
    std::uint64_t seed) {
    std::mt19937_64 generator(seed);
    std::vector<PhaseGate> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        QubitId first = static_cast<QubitId>(generator() % qubits);
        QubitId second = static_cast<QubitId>(generator() % qubits);
        if (second == first) {
            second = static_cast<QubitId>((second + 1U) % qubits);
        }
        const double angle = -1.0 + 2.0 * static_cast<double>(generator() >> 11U) /
                                      9007199254740992.0;
        result.push_back(PhaseGate{
            static_cast<std::uint8_t>(generator() % 5U), first, second, angle});
    }
    return result;
}

void execute(PhaseGraphState& state, const std::vector<PhaseGate>& operations) {
    for (const PhaseGate& gate : operations) {
        switch (gate.kind) {
            case 0U:
                state.apply_rz(gate.first, gate.angle);
                break;
            case 1U:
                state.apply_controlled_phase(gate.first, gate.second, gate.angle);
                break;
            case 2U:
                state.apply_x(gate.first);
                break;
            case 3U:
                state.apply_t(gate.first);
                break;
            default:
                state.apply_swap(gate.first, gate.second);
                break;
        }
    }
}

void execute(QRegister& state, const std::vector<PhaseGate>& operations) {
    for (const PhaseGate& gate : operations) {
        switch (gate.kind) {
            case 0U:
                state.apply_rz(gate.first, gate.angle);
                break;
            case 1U: {
                QMatrix4 matrix{};
                matrix.values[0] = {1.0, 0.0};
                matrix.values[5] = {1.0, 0.0};
                matrix.values[10] = {1.0, 0.0};
                matrix.values[15] = QComplex::from_polar(1.0, gate.angle);
                state.apply_two(gate.first, gate.second, matrix);
                break;
            }
            case 2U:
                state.apply_x(gate.first);
                break;
            case 3U:
                state.apply_t(gate.first);
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
    const auto operations = gates(14U, 120U, 0x504841534542454EULL);
    PhaseGraphState graph(14U);
    const double graph_ms = timed_ms([&] { execute(graph, operations); });

    QStateConfig config;
    config.max_component_qubits = 20U;
    config.max_dense_amplitudes = 1ULL << 18;
    QRegister reference(14U, config);
    for (QubitId qubit = 0; qubit < 14U; ++qubit) {
        reference.apply_h(qubit);
    }
    const double register_ms = timed_ms([&] { execute(reference, operations); });
    std::cout << "phase_graph qubits=14 gates=120 qregister_ms=" << register_ms
              << " phase_graph_ms=" << graph_ms
              << " speedup=" << register_ms / graph_ms
              << " phase_graph_bytes=" << graph.estimated_bytes()
              << " qregister_bytes=" << reference.estimated_bytes() << '\n';

    PhaseGraphConfig large_config;
    large_config.max_edges = 300'000U;
    PhaseGraphState large(100'000U, large_config);
    const double large_ms = timed_ms([&] {
        for (QubitId qubit = 0; qubit < 100'000U; ++qubit) {
            large.apply_rz(qubit, 0.0001 * static_cast<double>(qubit % 17U));
        }
        for (QubitId qubit = 1; qubit < 100'000U; ++qubit) {
            large.apply_controlled_phase(qubit - 1U, qubit, 0.01);
        }
    });
    std::cout << "phase_graph qubits=100000 gates=199999 milliseconds=" << large_ms
              << " edges=" << large.edge_count()
              << " bytes=" << large.estimated_bytes() << '\n';
    return 0;
}

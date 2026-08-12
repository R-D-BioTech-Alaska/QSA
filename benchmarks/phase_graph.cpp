#include "qubit/qbroker.hpp"
#include "qubit/qphase_graph.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::ExactExecutionBroker;
using qubit::ExactExecutionBrokerConfig;
using qubit::ExactExecutionRoute;
using qubit::Operation;
using qubit::OperationCode;
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

template <class Function>
[[nodiscard]] double median_ms(Function&& function, int repeats = 7) {
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        samples.push_back(timed_ms(function));
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

[[nodiscard]] std::vector<Operation> broker_phase_operations(std::size_t qubits) {
    std::vector<Operation> operations;
    operations.reserve(4U * qubits);
    for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
        operations.push_back({
            OperationCode::H,
            static_cast<QubitId>(qubit),
            0U,
            0.0,
            0.0,
        });
    }
    for (std::size_t qubit = 1U; qubit < qubits; ++qubit) {
        operations.push_back({
            OperationCode::Cz,
            0U,
            static_cast<QubitId>(qubit),
            0.0,
            0.0,
        });
    }
    for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
        operations.push_back({
            OperationCode::Rz,
            static_cast<QubitId>(qubit),
            0U,
            0.0003 * static_cast<double>((qubit % 17U) + 1U),
            0.0,
        });
    }
    for (std::size_t qubit = 0; qubit < qubits; qubit += 3U) {
        operations.push_back({
            OperationCode::T,
            static_cast<QubitId>(qubit),
            0U,
            0.0,
            0.0,
        });
    }
    return operations;
}

[[nodiscard]] PhaseGraphState direct_phase_graph(
    std::size_t qubits,
    std::span<const Operation> operations,
    PhaseGraphConfig config = {}) {
    PhaseGraphState state(qubits, config);
    for (std::size_t index = qubits; index < operations.size(); ++index) {
        const Operation& operation = operations[index];
        switch (operation.code) {
            case OperationCode::Cz:
                state.apply_cz(operation.first, operation.second);
                break;
            case OperationCode::Rz:
                state.apply_rz(operation.first, operation.parameter);
                break;
            case OperationCode::T:
                state.apply_t(operation.first);
                break;
            default:
                break;
        }
    }
    return state;
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

    {
        constexpr std::size_t qubits = 18U;
        const std::vector<Operation> broker_operations = broker_phase_operations(qubits);
        const std::vector<std::uint8_t> bits(qubits, 0U);
        ExactExecutionBrokerConfig broker_config;
        broker_config.tensor.max_contraction_entries = 8U;
        ExactExecutionBroker broker(broker_config);
        qubit::ExactProbabilityResult broker_result;
        const double broker_ms = median_ms([&] {
            broker_result = broker.basis_probability_from_zero(
                qubits,
                broker_operations,
                bits);
        });

        double reference_probability = 0.0;
        std::size_t reference_bytes = 0U;
        const double qregister_ms = median_ms([&] {
            QRegister state(qubits);
            qubit::OperationPlan plan(broker_operations);
            plan.execute(state);
            reference_probability = state.amplitude_bits(bits).norm2();
            reference_bytes = state.estimated_bytes();
        });
        const PhaseGraphState direct = direct_phase_graph(qubits, broker_operations);
        if (broker_result.route != ExactExecutionRoute::UniformMagnitude) {
            std::cerr << "uniform broker did not select UniformMagnitude\n";
            return 2;
        }
        std::cout << "uniform_broker_qubits=" << qubits << '\n';
        std::cout << "uniform_broker_route="
                  << qubit::exact_execution_route_name(broker_result.route) << '\n';
        std::cout << "uniform_broker_ms=" << broker_ms << '\n';
        std::cout << "uniform_broker_qregister_ms=" << qregister_ms << '\n';
        std::cout << "uniform_broker_speed_ratio=" << qregister_ms / broker_ms << '\n';
        std::cout << "uniform_broker_value_error="
                  << std::abs(broker_result.value - reference_probability) << '\n';
        std::cout << "uniform_broker_retained_state_bytes=0\n";
        std::cout << "uniform_broker_phase_graph_bytes=" << direct.estimated_bytes() << '\n';
        std::cout << "uniform_broker_phase_graph_edges=" << direct.edge_count() << '\n';
        std::cout << "uniform_broker_qregister_bytes=" << reference_bytes << '\n';
    }

    {
        constexpr std::size_t qubits = 1'000U;
        const std::vector<Operation> broker_operations = broker_phase_operations(qubits);
        const std::vector<std::uint8_t> bits(qubits, 0U);
        ExactExecutionBrokerConfig broker_config;
        broker_config.tensor.max_contraction_entries = 8U;
        broker_config.phase_graph.max_edges = 1U;
        ExactExecutionBroker broker(broker_config);
        qubit::ExactProbabilityResult result;
        const double broker_ms = median_ms([&] {
            result = broker.basis_probability_from_zero(
                qubits,
                broker_operations,
                bits);
        }, 3);
        const double expected = std::exp2(-static_cast<double>(qubits));
        if (result.route != ExactExecutionRoute::UniformMagnitude) {
            std::cerr << "wide uniform broker did not select UniformMagnitude\n";
            return 3;
        }
        std::cout << "uniform_broker_wide_qubits=" << qubits << '\n';
        std::cout << "uniform_broker_wide_ms=" << broker_ms << '\n';
        std::cout << "uniform_broker_wide_error=" << std::abs(result.value - expected) << '\n';
        std::cout << "uniform_broker_wide_retained_state_bytes=0\n";
    }

    {
        constexpr std::size_t qubits = 100'000U;
        const std::vector<Operation> broker_operations = broker_phase_operations(qubits);
        const std::vector<std::uint8_t> bits(qubits, 0U);
        ExactExecutionBrokerConfig broker_config;
        broker_config.tensor.max_contraction_entries = 8U;
        broker_config.phase_graph.max_edges = 1U;
        ExactExecutionBroker broker(broker_config);
        qubit::ExactProbabilityResult result;
        const double broker_ms = median_ms([&] {
            result = broker.basis_probability_from_zero(
                qubits,
                broker_operations,
                bits);
        }, 3);
        if (result.route != ExactExecutionRoute::UniformMagnitude || result.value != 0.0) {
            std::cerr << "100000-qubit uniform broker certificate failed\n";
            return 4;
        }
        std::cout << "uniform_broker_huge_qubits=" << qubits << '\n';
        std::cout << "uniform_broker_huge_operations=" << broker_operations.size() << '\n';
        std::cout << "uniform_broker_huge_ms=" << broker_ms << '\n';
        std::cout << "uniform_broker_huge_retained_state_bytes=0\n";
    }
    return 0;
}

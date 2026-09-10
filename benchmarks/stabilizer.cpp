#include "qubit/qbroker.hpp"
#include "qubit/qstabilizer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::ExactExecutionBroker;
using qubit::ExactExecutionRoute;
using qubit::ExactPreparedProbabilityPlan;
using qubit::Operation;
using qubit::OperationCode;
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

[[nodiscard]] std::vector<Operation> broker_operations(
    const std::vector<CliffordGate>& operations) {
    std::vector<Operation> result;
    result.reserve(operations.size());
    for (const CliffordGate& gate : operations) {
        OperationCode code = OperationCode::H;
        switch (gate.kind) {
            case 0U:
                code = OperationCode::H;
                break;
            case 1U:
                code = OperationCode::S;
                break;
            case 2U:
                code = OperationCode::Cnot;
                break;
            case 3U:
                code = OperationCode::Cz;
                break;
            default:
                code = OperationCode::Swap;
                break;
        }
        result.push_back({code, gate.first, gate.second, 0.0, 0.0});
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

template <class Function>
[[nodiscard]] double median_ms(
    Function&& function,
    int repeats = 7,
    int iterations = 1) {
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto start = Clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            function();
        }
        const auto stop = Clock::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count() /
            static_cast<double>(iterations));
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

[[nodiscard]] double break_even_queries(
    double setup_ms,
    double one_shot_ms,
    double prepared_query_ms) noexcept {
    if (one_shot_ms <= prepared_query_ms) {
        return -1.0;
    }
    return setup_ms / (one_shot_ms - prepared_query_ms);
}

[[nodiscard]] double marginal_probability(
    StabilizerState state,
    std::span<const QubitId> qubits,
    std::span<const std::uint8_t> bits) {
    double probability = 1.0;
    for (std::size_t index = 0U; index < qubits.size(); ++index) {
        const double p1 = state.probability_one(qubits[index]);
        const double factor = bits[index] != 0U ? p1 : 1.0 - p1;
        if (factor == 0.0) {
            return 0.0;
        }
        probability *= factor;
        const int outcome = state.measure_z(
            qubits[index], bits[index] != 0U ? 0.0 : 0.75);
        if (outcome != static_cast<int>(bits[index])) {
            return 0.0;
        }
    }
    return probability;
}

template <std::size_t N>
[[nodiscard]] std::array<std::uint8_t, N> supported_assignment(
    StabilizerState state,
    const std::array<QubitId, N>& qubits) {
    std::array<std::uint8_t, N> bits{};
    for (std::size_t index = 0U; index < N; ++index) {
        const double p1 = state.probability_one(qubits[index]);
        bits[index] = p1 == 1.0 ? 1U : 0U;
        const int outcome = state.measure_z(
            qubits[index], bits[index] != 0U ? 0.0 : 0.75);
        if (outcome != static_cast<int>(bits[index])) {
            throw std::runtime_error("stabilizer support selection failed");
        }
    }
    return bits;
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);
    const auto operations = gates(18U, 800U, 0x434C4946464F5244ULL);
    const auto operations_batch = batch_operations(operations);
    const auto operations_broker = broker_operations(operations);
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

    StabilizerState sampled = stabilizer;
    const std::vector<int> measured = sampled.measure_all(0x5354414242415349ULL);
    std::vector<std::uint8_t> basis_bits(measured.size(), 0U);
    for (std::size_t qubit = 0U; qubit < measured.size(); ++qubit) {
        basis_bits[qubit] = static_cast<std::uint8_t>(measured[qubit]);
    }

    ExactExecutionBroker broker;
    qubit::ExactProbabilityResult broker_result;
    const double broker_ms = median_ms([&] {
        broker_result = broker.basis_probability_from_zero(
            18U, operations_broker, basis_bits);
    }, 5);
    double register_probability = 0.0;
    const double register_probability_ms = median_ms([&] {
        QRegister state(18U, config);
        execute(state, operations);
        register_probability = state.amplitude_bits(basis_bits).norm2();
    }, 5);
    std::optional<ExactPreparedProbabilityPlan> prepared;
    const double prepared_setup_ms = median_ms([&] {
        prepared.emplace(18U, operations_broker);
    }, 5);
    qubit::ExactProbabilityResult prepared_result;
    const double prepared_query_ms = median_ms([&] {
        prepared_result = prepared->probability(basis_bits);
    }, 11, 1000);
    if (broker_result.route != ExactExecutionRoute::Stabilizer ||
        prepared->prepared_route() != ExactExecutionRoute::Stabilizer ||
        prepared_result.route != ExactExecutionRoute::Stabilizer ||
        std::abs(broker_result.value - register_probability) > 2e-11 ||
        std::abs(prepared_result.value - register_probability) > 2e-11) {
        std::cerr << "stabilizer broker probability certificate failed\n";
        return 2;
    }
    std::cout << "stabilizer_broker_qubits=18\n";
    std::cout << "stabilizer_broker_operations=" << operations_broker.size() << '\n';
    std::cout << "stabilizer_broker_route=Stabilizer\n";
    std::cout << "stabilizer_broker_one_shot_ms=" << broker_ms << '\n';
    std::cout << "stabilizer_broker_qregister_ms=" << register_probability_ms << '\n';
    std::cout << "stabilizer_broker_vs_qregister=" << register_probability_ms / broker_ms << '\n';
    std::cout << "stabilizer_broker_value_error="
              << std::abs(broker_result.value - register_probability) << '\n';
    std::cout << "stabilizer_prepared_setup_ms=" << prepared_setup_ms << '\n';
    std::cout << "stabilizer_prepared_query_ms=" << prepared_query_ms << '\n';
    std::cout << "stabilizer_prepared_over_one_shot_speed=" << broker_ms / prepared_query_ms << '\n';
    std::cout << "stabilizer_prepared_break_even_queries="
              << break_even_queries(prepared_setup_ms, broker_ms, prepared_query_ms) << '\n';
    std::cout << "stabilizer_prepared_bytes=" << prepared->estimated_bytes() << '\n';

    StabilizerConfig large_config;
    large_config.max_tableau_bytes = 1ULL << 29;
    large_config.max_batch_scratch_bytes = 1ULL << 29;
    const auto large_operations = gates(4096U, 100'000U, 0x4C41524745434C49ULL);
    const auto large_batch = batch_operations(large_operations);
    const auto large_broker = broker_operations(large_operations);
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

    const std::array<QubitId, 8> marginal_qubits{{
        0U, 1U, 17U, 255U, 1023U, 2048U, 3071U, 4095U,
    }};
    const auto marginal_bits = supported_assignment(large_batched, marginal_qubits);
    const double direct_probability = marginal_probability(
        large_batched, marginal_qubits, marginal_bits);

    qubit::ExactProbabilityResult large_one_shot;
    const double large_one_shot_ms = median_ms([&] {
        large_one_shot = broker.marginal_probability_from_zero(
            4096U, large_broker, marginal_qubits, marginal_bits);
    }, 3);
    std::optional<ExactPreparedProbabilityPlan> large_prepared;
    const double large_prepared_setup_ms = median_ms([&] {
        large_prepared.emplace(4096U, large_broker);
    }, 3);
    qubit::ExactProbabilityResult large_prepared_result;
    const double large_prepared_query_ms = median_ms([&] {
        large_prepared_result = large_prepared->marginal_probability(
            marginal_qubits, marginal_bits);
    }, 7, 20);
    double direct_query_value = 0.0;
    const double large_direct_query_ms = median_ms([&] {
        direct_query_value = marginal_probability(
            large_batched, marginal_qubits, marginal_bits);
    }, 7, 20);
    if (large_one_shot.route != ExactExecutionRoute::Stabilizer ||
        large_prepared->prepared_route() != ExactExecutionRoute::Stabilizer ||
        large_prepared_result.route != ExactExecutionRoute::Stabilizer ||
        std::abs(large_one_shot.value - direct_probability) > 2e-11 ||
        std::abs(large_prepared_result.value - direct_probability) > 2e-11 ||
        std::abs(direct_query_value - direct_probability) > 2e-11) {
        std::cerr << "wide stabilizer marginal certificate failed\n";
        return 3;
    }
    std::cout << "stabilizer_marginal_qubits=4096\n";
    std::cout << "stabilizer_marginal_operations=" << large_broker.size() << '\n';
    std::cout << "stabilizer_marginal_selected_qubits=" << marginal_qubits.size() << '\n';
    std::cout << "stabilizer_marginal_route=Stabilizer\n";
    std::cout << "stabilizer_marginal_one_shot_ms=" << large_one_shot_ms << '\n';
    std::cout << "stabilizer_marginal_prepared_setup_ms=" << large_prepared_setup_ms << '\n';
    std::cout << "stabilizer_marginal_prepared_query_ms=" << large_prepared_query_ms << '\n';
    std::cout << "stabilizer_marginal_direct_query_ms=" << large_direct_query_ms << '\n';
    std::cout << "stabilizer_marginal_prepared_over_one_shot_speed="
              << large_one_shot_ms / large_prepared_query_ms << '\n';
    std::cout << "stabilizer_marginal_prepared_over_direct_ratio="
              << large_direct_query_ms / large_prepared_query_ms << '\n';
    std::cout << "stabilizer_marginal_break_even_queries="
              << break_even_queries(
                     large_prepared_setup_ms,
                     large_one_shot_ms,
                     large_prepared_query_ms)
              << '\n';
    std::cout << "stabilizer_marginal_prepared_bytes=" << large_prepared->estimated_bytes() << '\n';
    std::cout << "stabilizer_marginal_value_error="
              << std::abs(large_prepared_result.value - direct_probability) << '\n';
    return 0;
}

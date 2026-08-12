#include "qubit/qbroker.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::ExactExecutionBroker;
using qubit::ExactExecutionBrokerConfig;
using qubit::ExactExecutionRoute;
using qubit::ExactPreparedProbabilityPlan;
using qubit::Operation;
using qubit::OperationCode;
using qubit::QubitId;

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
        const auto finish = Clock::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(finish - start).count() /
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

[[nodiscard]] std::vector<Operation> uniform_operations(std::size_t qubits) {
    std::vector<Operation> operations;
    operations.reserve(4U * qubits);
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
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
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({
            OperationCode::Rz,
            static_cast<QubitId>(qubit),
            0U,
            0.0003 * static_cast<double>((qubit % 17U) + 1U),
            0.0,
        });
    }
    for (std::size_t qubit = 0U; qubit < qubits; qubit += 3U) {
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

[[nodiscard]] std::vector<Operation> basis_operations(std::size_t qubits) {
    std::vector<Operation> operations;
    operations.reserve(3U * qubits);
    operations.push_back({OperationCode::X, 0U, 0U, 0.0, 0.0});
    for (std::size_t qubit = 0U; qubit + 1U < qubits; ++qubit) {
        operations.push_back({
            OperationCode::Cnot,
            static_cast<QubitId>(qubit),
            static_cast<QubitId>(qubit + 1U),
            0.0,
            0.0,
        });
    }
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({
            OperationCode::Rz,
            static_cast<QubitId>(qubit),
            0U,
            0.0007 * static_cast<double>((qubit % 19U) + 1U),
            0.0,
        });
    }
    for (std::size_t qubit = 0U; qubit + 1U < qubits; qubit += 2U) {
        operations.push_back({
            OperationCode::Cz,
            static_cast<QubitId>(qubit),
            static_cast<QubitId>(qubit + 1U),
            0.0,
            0.0,
        });
        operations.push_back({
            OperationCode::Swap,
            static_cast<QubitId>(qubit),
            static_cast<QubitId>(qubit + 1U),
            0.0,
            0.0,
        });
    }
    return operations;
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);
    constexpr std::size_t qubits = 100'000U;
    const std::array<QubitId, 8> query_qubits{{
        0U, 1U, 17U, 999U, 5'000U, 40'000U, 75'000U, 99'999U,
    }};

    {
        const std::vector<Operation> operations = uniform_operations(qubits);
        const std::array<std::uint8_t, 8> query_bits{{0U, 1U, 0U, 1U, 1U, 0U, 1U, 0U}};
        ExactExecutionBrokerConfig config;
        config.tensor.max_contraction_entries = 8U;
        config.phase_graph.max_edges = 1U;
        ExactExecutionBroker broker(config);

        qubit::ExactProbabilityResult one_shot;
        const double one_shot_ms = median_ms([&] {
            one_shot = broker.marginal_probability_from_zero(
                qubits, operations, query_qubits, query_bits);
        }, 3);

        std::optional<ExactPreparedProbabilityPlan> prepared;
        const double setup_ms = median_ms([&] {
            prepared.emplace(qubits, operations, config);
        }, 3);
        qubit::ExactProbabilityResult prepared_result;
        const double query_ms = median_ms([&] {
            prepared_result = prepared->marginal_probability(query_qubits, query_bits);
        }, 11, 1000);

        const double expected = std::ldexp(1.0, -static_cast<int>(query_bits.size()));
        if (one_shot.route != ExactExecutionRoute::UniformMagnitude ||
            prepared->prepared_route() != ExactExecutionRoute::UniformMagnitude ||
            prepared_result.route != ExactExecutionRoute::UniformMagnitude ||
            std::abs(one_shot.value - expected) > 1e-15 ||
            std::abs(prepared_result.value - expected) > 1e-15) {
            std::cerr << "uniform marginal certificate failed\n";
            return 2;
        }

        std::cout << "uniform_marginal_qubits=" << qubits << '\n';
        std::cout << "uniform_marginal_operations=" << operations.size() << '\n';
        std::cout << "uniform_marginal_selected_qubits=" << query_bits.size() << '\n';
        std::cout << "uniform_marginal_route=UniformMagnitude\n";
        std::cout << "uniform_marginal_one_shot_ms=" << one_shot_ms << '\n';
        std::cout << "uniform_marginal_prepared_setup_ms=" << setup_ms << '\n';
        std::cout << "uniform_marginal_prepared_query_ms=" << query_ms << '\n';
        std::cout << "uniform_marginal_prepared_over_one_shot_speed="
                  << one_shot_ms / query_ms << '\n';
        std::cout << "uniform_marginal_break_even_queries="
                  << break_even_queries(setup_ms, one_shot_ms, query_ms) << '\n';
        std::cout << "uniform_marginal_prepared_bytes=" << prepared->estimated_bytes() << '\n';
        std::cout << "uniform_marginal_value_error="
                  << std::abs(prepared_result.value - expected) << '\n';
    }

    {
        const std::vector<Operation> operations = basis_operations(qubits);
        const std::array<std::uint8_t, 8> hit_bits{{1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U}};
        std::array<std::uint8_t, 8> miss_bits = hit_bits;
        miss_bits[3] = 0U;
        ExactExecutionBroker broker;

        qubit::ExactProbabilityResult one_shot;
        const double one_shot_ms = median_ms([&] {
            one_shot = broker.marginal_probability_from_zero(
                qubits, operations, query_qubits, hit_bits);
        }, 3);

        std::optional<ExactPreparedProbabilityPlan> prepared;
        const double setup_ms = median_ms([&] {
            prepared.emplace(qubits, operations);
        }, 3);
        qubit::ExactProbabilityResult prepared_result;
        const double query_ms = median_ms([&] {
            prepared_result = prepared->marginal_probability(query_qubits, hit_bits);
        }, 11, 1000);
        const auto miss = prepared->marginal_probability(query_qubits, miss_bits);

        if (one_shot.route != ExactExecutionRoute::BasisPermutation ||
            prepared->prepared_route() != ExactExecutionRoute::BasisPermutation ||
            prepared_result.route != ExactExecutionRoute::BasisPermutation ||
            one_shot.value != 1.0 || prepared_result.value != 1.0 || miss.value != 0.0) {
            std::cerr << "basis-permutation marginal certificate failed\n";
            return 3;
        }

        std::cout << "basis_marginal_qubits=" << qubits << '\n';
        std::cout << "basis_marginal_operations=" << operations.size() << '\n';
        std::cout << "basis_marginal_selected_qubits=" << hit_bits.size() << '\n';
        std::cout << "basis_marginal_route=BasisPermutation\n";
        std::cout << "basis_marginal_one_shot_ms=" << one_shot_ms << '\n';
        std::cout << "basis_marginal_prepared_setup_ms=" << setup_ms << '\n';
        std::cout << "basis_marginal_prepared_query_ms=" << query_ms << '\n';
        std::cout << "basis_marginal_prepared_over_one_shot_speed="
                  << one_shot_ms / query_ms << '\n';
        std::cout << "basis_marginal_break_even_queries="
                  << break_even_queries(setup_ms, one_shot_ms, query_ms) << '\n';
        std::cout << "basis_marginal_prepared_bytes=" << prepared->estimated_bytes() << '\n';
        std::cout << "basis_marginal_hit=" << prepared_result.value << '\n';
        std::cout << "basis_marginal_miss=" << miss.value << '\n';
    }

    return 0;
}

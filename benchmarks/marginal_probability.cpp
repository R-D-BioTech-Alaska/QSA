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
using qubit::MatrixProductState;
using qubit::Operation;
using qubit::OperationCode;
using qubit::QRegister;
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

[[nodiscard]] std::vector<Operation> low_bond_operations(std::size_t qubits) {
    std::vector<Operation> operations;
    operations.reserve(2U * qubits);
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({
            OperationCode::Ry,
            static_cast<QubitId>(qubit),
            0U,
            0.0005 * static_cast<double>((qubit % 23U) + 1U),
            0.0,
        });
    }
    for (std::size_t qubit = 0U; qubit + 1U < qubits; ++qubit) {
        operations.push_back({
            OperationCode::Cz,
            static_cast<QubitId>(qubit),
            static_cast<QubitId>(qubit + 1U),
            0.0,
            0.0,
        });
    }
    return operations;
}

[[nodiscard]] MatrixProductState low_bond_reference(std::size_t qubits) {
    MatrixProductState state = MatrixProductState::zero(qubits);
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        state.apply_unitary(
            static_cast<QubitId>(qubit),
            qubit::gates::ry(0.0005 * static_cast<double>((qubit % 23U) + 1U)));
    }
    for (std::size_t qubit = 0U; qubit + 1U < qubits; ++qubit) {
        state.apply_cz(
            static_cast<QubitId>(qubit),
            static_cast<QubitId>(qubit + 1U));
    }
    return state;
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
            prepared.emplace(ExactPreparedProbabilityPlan::for_marginals(
                qubits, operations, config));
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
            prepared.emplace(ExactPreparedProbabilityPlan::for_marginals(
                qubits, operations));
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

    {
        constexpr std::size_t low_bond_qubits = 20'000U;
        const std::vector<Operation> operations = low_bond_operations(low_bond_qubits);
        const std::array<QubitId, 3> selected{{9'999U, 10'000U, 10'001U}};
        const std::array<std::uint8_t, 3> selected_bits{{1U, 0U, 1U}};
        ExactExecutionBroker broker;

        ExactExecutionBrokerConfig full_config;
        full_config.tensor_planning_defer_variables = 0U;
        std::optional<ExactPreparedProbabilityPlan> full;
        const double full_setup_ms = median_ms([&] {
            full.emplace(low_bond_qubits, operations, full_config);
        }, 3);
        if (full->prepared_route() != ExactExecutionRoute::TensorNetwork) {
            std::cerr << "full-basis route-conflict certificate changed\n";
            return 4;
        }

        MatrixProductState reference = low_bond_reference(low_bond_qubits);
        const double expected = reference.marginal_probability(selected, selected_bits);

        qubit::ExactProbabilityResult one_shot;
        const double one_shot_ms = median_ms([&] {
            one_shot = broker.marginal_probability_from_zero(
                low_bond_qubits, operations, selected, selected_bits);
        }, 3);

        std::optional<ExactPreparedProbabilityPlan> prepared;
        const double setup_ms = median_ms([&] {
            prepared.emplace(ExactPreparedProbabilityPlan::for_marginals(
                low_bond_qubits, operations));
        }, 3);
        qubit::ExactProbabilityResult prepared_result;
        const double query_ms = median_ms([&] {
            prepared_result = prepared->marginal_probability(selected, selected_bits);
        }, 11, 1000);

        if (one_shot.route != ExactExecutionRoute::PersistentMPS ||
            prepared->prepared_route() != ExactExecutionRoute::PersistentMPS ||
            prepared_result.route != ExactExecutionRoute::PersistentMPS ||
            prepared_result.fallback_reason.find(
                "tensor: route does not support marginal probability") == std::string::npos ||
            std::abs(one_shot.value - expected) > 2e-11 ||
            std::abs(prepared_result.value - expected) > 2e-11) {
            std::cerr << "query-capability marginal route certificate failed\n";
            return 5;
        }

        std::cout << "capability_marginal_qubits=" << low_bond_qubits << '\n';
        std::cout << "capability_marginal_operations=" << operations.size() << '\n';
        std::cout << "capability_full_basis_route=TensorNetwork\n";
        std::cout << "capability_full_basis_deferral_disabled=1\n";
        std::cout << "capability_marginal_route=PersistentMPS\n";
        std::cout << "capability_full_basis_setup_ms=" << full_setup_ms << '\n';
        std::cout << "capability_full_basis_plan_bytes=" << full->estimated_bytes() << '\n';
        std::cout << "capability_marginal_one_shot_ms=" << one_shot_ms << '\n';
        std::cout << "capability_marginal_prepared_setup_ms=" << setup_ms << '\n';
        std::cout << "capability_marginal_prepared_query_ms=" << query_ms << '\n';
        std::cout << "capability_marginal_prepared_over_one_shot_speed="
                  << one_shot_ms / query_ms << '\n';
        std::cout << "capability_marginal_break_even_queries="
                  << break_even_queries(setup_ms, one_shot_ms, query_ms) << '\n';
        std::cout << "capability_marginal_prepared_bytes=" << prepared->estimated_bytes() << '\n';
        std::cout << "capability_marginal_value_error="
                  << std::abs(prepared_result.value - expected) << '\n';
    }

    {
        const std::array<Operation, 3> operations{{
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::Cnot, 0U, 5U, 0.0, 0.0},
            {OperationCode::Ry, 0U, 0U, 0.19, 0.0},
        }};
        const std::array<QubitId, 2> selected{{0U, 5U}};
        const std::array<std::uint8_t, 2> selected_bits{{1U, 1U}};
        ExactExecutionBroker broker;
        QRegister reference(6U);
        qubit::OperationPlan plan(operations);
        plan.execute(reference);
        const double expected = reference.marginal_probability(selected, selected_bits);

        const auto result = broker.marginal_probability_from_zero(
            6U, operations, selected, selected_bits);
        ExactPreparedProbabilityPlan prepared =
            ExactPreparedProbabilityPlan::for_marginals(6U, operations);
        const auto prepared_result = prepared.marginal_probability(selected, selected_bits);
        if (result.route != ExactExecutionRoute::Register ||
            prepared.prepared_route() != ExactExecutionRoute::Register ||
            prepared_result.route != ExactExecutionRoute::Register ||
            std::abs(result.value - expected) > 2e-11 ||
            std::abs(prepared_result.value - expected) > 2e-11) {
            std::cerr << "generic QRegister marginal fallback certificate failed\n";
            return 6;
        }
        std::cout << "register_marginal_route=QRegister\n";
        std::cout << "register_marginal_value_error="
                  << std::abs(prepared_result.value - expected) << '\n';
    }

    return 0;
}
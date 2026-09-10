#include "qubit/qfactor_chain.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <span>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::ExactFactorChainPlan;
using qubit::ExactFactorConfig;
using qubit::ExactFactorGraph;
using qubit::ExactFactorPlan;
using qubit::FactorId;
using qubit::FactorVariableId;
using qubit::QComplex;

template <class Function>
double median_ms(Function&& function, int repeats) {
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto start = Clock::now();
        function();
        const auto stop = Clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

std::array<QComplex, 8> nonlinear_table(std::size_t factor, std::size_t point = 0U) {
    std::array<QComplex, 8> values{};
    for (std::size_t index = 0U; index < values.size(); ++index) {
        const std::size_t a = index & 1U;
        const std::size_t b = (index >> 1U) & 1U;
        const std::size_t c = (index >> 2U) & 1U;
        const std::size_t expected = a & b;
        const double phase = 0.0001 * static_cast<double>((factor + 3U * point) % 97U);
        values[index] = {
            c == expected ? 1.0 + phase : 0.15 + 0.5 * phase,
            0.0,
        };
    }
    return values;
}

std::array<QComplex, 4> specialized_chain(
    std::span<const std::array<QComplex, 8>> tables) {
    std::array<QComplex, 4> current{{
        QComplex{1.0}, QComplex{1.0}, QComplex{1.0}, QComplex{1.0},
    }};
    for (const auto& table : tables) {
        std::array<QComplex, 4> next{};
        for (std::size_t a = 0U; a < 2U; ++a) {
            for (std::size_t b = 0U; b < 2U; ++b) {
                const QComplex prefix = current[a + 2U * b];
                for (std::size_t c = 0U; c < 2U; ++c) {
                    next[b + 2U * c] += prefix * table[a + 2U * b + 4U * c];
                }
            }
        }
        current = next;
    }
    return current;
}

double max_error(
    std::span<const QComplex> actual,
    std::span<const QComplex> expected) {
    double error = 0.0;
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        error = std::max(error, (actual[index] - expected[index]).magnitude());
    }
    return error;
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);
    constexpr std::size_t variables = 4096U;
    constexpr std::size_t factors = variables - 2U;

    ExactFactorConfig config;
    config.max_factor_entries = 64U;
    ExactFactorGraph graph(config);
    for (std::size_t variable = 0U; variable < variables; ++variable) {
        static_cast<void>(graph.add_variable(2U));
    }

    std::vector<std::array<QComplex, 8>> tables(factors);
    std::vector<FactorId> factor_ids;
    factor_ids.reserve(factors);
    for (std::size_t factor = 0U; factor < factors; ++factor) {
        tables[factor] = nonlinear_table(factor);
        const std::array<FactorVariableId, 3> scope{{
            static_cast<FactorVariableId>(factor),
            static_cast<FactorVariableId>(factor + 1U),
            static_cast<FactorVariableId>(factor + 2U),
        }};
        factor_ids.push_back(graph.add_dense_factor(scope, tables[factor]));
    }

    const std::array<FactorVariableId, 2> retained{{
        static_cast<FactorVariableId>(variables - 2U),
        static_cast<FactorVariableId>(variables - 1U),
    }};

    ExactFactorPlan generic = graph.compile(retained);
    auto generic_workspace = generic.workspace();
    std::vector<QComplex> generic_values;
    const double generic_query_ms = median_ms([&] {
        generic_values = generic.evaluate(generic_workspace);
    }, 101);

    ExactFactorChainPlan chain(graph, retained);
    auto chain_workspace = chain.workspace();
    std::vector<QComplex> chain_values;
    const double chain_query_ms = median_ms([&] {
        chain_values = chain.evaluate(chain_workspace);
    }, 101);
    const double chain_compile_ms = median_ms([&] {
        ExactFactorChainPlan candidate(graph, retained);
        static_cast<void>(candidate.stats().factor_count);
    }, 11);

    std::array<QComplex, 4> control{};
    const double specialized_ms = median_ms([&] {
        control = specialized_chain(tables);
    }, 101);

    const double chain_control_error = max_error(chain_values, control);
    const double chain_generic_error = max_error(chain_values, generic_values);
    if (chain_control_error > 2e-9 || chain_generic_error > 2e-9) {
        std::cerr << "exact factor chain benchmark mismatch\n";
        return 1;
    }

    constexpr std::size_t points = 16U;
    const std::size_t rebound_factor = factors / 2U;
    ExactFactorChainPlan chain_updates(graph, retained);
    ExactFactorPlan generic_updates = graph.compile(retained);
    auto chain_update_workspace = chain_updates.workspace();
    auto generic_update_workspace = generic_updates.workspace();
    auto updated_tables = tables;
    double chain_update_error = 0.0;
    const auto chain_update_start = Clock::now();
    for (std::size_t point = 1U; point <= points; ++point) {
        updated_tables[rebound_factor] = nonlinear_table(rebound_factor, point);
        chain_updates.rebind_dense_factor(
            factor_ids[rebound_factor], updated_tables[rebound_factor]);
        chain_values = chain_updates.evaluate(chain_update_workspace);
        control = specialized_chain(updated_tables);
        chain_update_error =
            std::max(chain_update_error, max_error(chain_values, control));
    }
    const auto chain_update_stop = Clock::now();
    const double chain_update_ms =
        std::chrono::duration<double, std::milli>(chain_update_stop - chain_update_start).count();

    updated_tables = tables;
    double generic_update_error = 0.0;
    const auto generic_update_start = Clock::now();
    for (std::size_t point = 1U; point <= points; ++point) {
        updated_tables[rebound_factor] = nonlinear_table(rebound_factor, point);
        generic_updates.rebind_dense_factor(
            factor_ids[rebound_factor], updated_tables[rebound_factor]);
        generic_values = generic_updates.evaluate(generic_update_workspace);
        control = specialized_chain(updated_tables);
        generic_update_error =
            std::max(generic_update_error, max_error(generic_values, control));
    }
    const auto generic_update_stop = Clock::now();
    const double generic_update_ms =
        std::chrono::duration<double, std::milli>(generic_update_stop - generic_update_start).count();

    if (chain_update_error > 2e-9 || generic_update_error > 2e-9) {
        std::cerr << "exact factor chain update benchmark mismatch\n";
        return 1;
    }

    std::cout << "chain_variables=" << variables << '\n';
    std::cout << "chain_factors=" << factors << '\n';
    std::cout << "chain_width=" << chain.stats().factor_width << '\n';
    std::cout << "chain_route=" << chain.route_name() << '\n';
    std::cout << "chain_compile_ms=" << chain_compile_ms << '\n';
    std::cout << "chain_query_ms=" << chain_query_ms << '\n';
    std::cout << "chain_generic_query_ms=" << generic_query_ms << '\n';
    std::cout << "chain_over_generic_speed=" << generic_query_ms / chain_query_ms << '\n';
    std::cout << "chain_specialized_control_ms=" << specialized_ms << '\n';
    std::cout << "chain_specialized_over_chain_ratio=" << specialized_ms / chain_query_ms << '\n';
    std::cout << "chain_control_error=" << chain_control_error << '\n';
    std::cout << "chain_generic_error=" << chain_generic_error << '\n';
    std::cout << "chain_max_separator_entries=" << chain.stats().max_separator_entries << '\n';
    std::cout << "chain_plan_bytes=" << chain.estimated_bytes() << '\n';
    std::cout << "chain_workspace_bytes=" << chain_workspace.estimated_bytes() << '\n';
    std::cout << "chain_generic_plan_bytes=" << generic.estimated_bytes() << '\n';
    std::cout << "chain_generic_workspace_bytes=" << generic_workspace.estimated_bytes() << '\n';
    std::cout << "chain_update_points=" << points << '\n';
    std::cout << "chain_update_total_ms=" << chain_update_ms << '\n';
    std::cout << "chain_generic_update_total_ms=" << generic_update_ms << '\n';
    std::cout << "chain_update_over_generic_speed=" << generic_update_ms / chain_update_ms << '\n';
    std::cout << "chain_update_error=" << chain_update_error << '\n';
    std::cout << "chain_generic_update_error=" << generic_update_error << '\n';
    return 0;
}

#include "qubit/qfactor.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
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

std::array<QComplex, 4> chain_control(
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

    ExactFactorPlan plan = graph.compile(retained);
    const double compile_ms = median_ms([&] {
        plan = graph.compile(retained);
    }, 7);
    auto workspace = plan.workspace();
    std::vector<QComplex> qsa_values;
    const double prepared_query_ms = median_ms([&] {
        qsa_values = plan.evaluate(workspace);
    }, 101);

    std::array<QComplex, 4> control{};
    const double control_ms = median_ms([&] {
        control = chain_control(tables);
    }, 101);
    const double initial_error = max_error(qsa_values, control);

    std::vector<QComplex> one_shot_values;
    const double one_shot_ms = median_ms([&] {
        ExactFactorPlan one_shot = graph.compile(retained);
        one_shot_values = one_shot.evaluate();
    }, 11);
    const double one_shot_error = max_error(one_shot_values, control);

    constexpr std::size_t points = 16U;
    const std::size_t rebound_factor = factors / 2U;
    double rebind_error = 0.0;
    const auto rebind_start = Clock::now();
    for (std::size_t point = 1U; point <= points; ++point) {
        tables[rebound_factor] = nonlinear_table(rebound_factor, point);
        graph.set_dense_factor(factor_ids[rebound_factor], tables[rebound_factor]);
        plan.rebind(graph);
        qsa_values = plan.evaluate(workspace);
        control = chain_control(tables);
        rebind_error = std::max(rebind_error, max_error(qsa_values, control));
    }
    const auto rebind_stop = Clock::now();
    const double rebind_total_ms =
        std::chrono::duration<double, std::milli>(rebind_stop - rebind_start).count();

    double one_shot_point_error = 0.0;
    const auto one_shot_start = Clock::now();
    for (std::size_t point = points + 1U; point <= 2U * points; ++point) {
        tables[rebound_factor] = nonlinear_table(rebound_factor, point);
        graph.set_dense_factor(factor_ids[rebound_factor], tables[rebound_factor]);
        ExactFactorPlan one_shot = graph.compile(retained);
        one_shot_values = one_shot.evaluate();
        control = chain_control(tables);
        one_shot_point_error =
            std::max(one_shot_point_error, max_error(one_shot_values, control));
    }
    const auto one_shot_stop = Clock::now();
    const double one_shot_total_ms =
        std::chrono::duration<double, std::milli>(one_shot_stop - one_shot_start).count();

    if (initial_error > 2e-9 || one_shot_error > 2e-9 ||
        rebind_error > 2e-9 || one_shot_point_error > 2e-9) {
        std::cerr << "exact factor benchmark control mismatch\n";
        return 1;
    }

    std::cout << "factor_variables=" << variables << '\n';
    std::cout << "factor_source_factors=" << factors << '\n';
    std::cout << "factor_retained_variables=" << retained.size() << '\n';
    std::cout << "factor_route=" << qubit::exact_factor_route_name(plan.route()) << '\n';
    std::cout << "factor_compile_ms=" << compile_ms << '\n';
    std::cout << "factor_prepared_query_ms=" << prepared_query_ms << '\n';
    std::cout << "factor_one_shot_ms=" << one_shot_ms << '\n';
    std::cout << "factor_prepared_over_one_shot_speed=" << one_shot_ms / prepared_query_ms << '\n';
    std::cout << "factor_specialized_control_ms=" << control_ms << '\n';
    std::cout << "factor_prepared_over_specialized_control_ratio=" << control_ms / prepared_query_ms << '\n';
    std::cout << "factor_peak_union_variables=" << plan.stats().peak_union_variables << '\n';
    std::cout << "factor_peak_entries=" << plan.stats().peak_factor_entries << '\n';
    std::cout << "factor_plan_bytes=" << plan.estimated_bytes() << '\n';
    std::cout << "factor_workspace_bytes=" << workspace.estimated_bytes() << '\n';
    std::cout << "factor_initial_error=" << initial_error << '\n';
    std::cout << "factor_rebind_points=" << points << '\n';
    std::cout << "factor_rebind_total_ms=" << rebind_total_ms << '\n';
    std::cout << "factor_recompile_total_ms=" << one_shot_total_ms << '\n';
    std::cout << "factor_rebind_over_recompile_speed=" << one_shot_total_ms / rebind_total_ms << '\n';
    std::cout << "factor_rebind_error=" << rebind_error << '\n';
    std::cout << "factor_recompile_error=" << one_shot_point_error << '\n';
    std::cout << "factor_rebind_count=" << plan.rebind_count() << '\n';
    return 0;
}

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

void populate_chain(
    ExactFactorGraph& graph,
    std::span<const std::array<QComplex, 8>> tables,
    std::vector<FactorId>* factor_ids = nullptr) {
    for (std::size_t variable = 0U; variable < tables.size() + 2U; ++variable) {
        static_cast<void>(graph.add_variable(2U));
    }
    if (factor_ids != nullptr) {
        factor_ids->reserve(tables.size());
    }
    for (std::size_t factor = 0U; factor < tables.size(); ++factor) {
        const std::array<FactorVariableId, 3> scope{{
            static_cast<FactorVariableId>(factor),
            static_cast<FactorVariableId>(factor + 1U),
            static_cast<FactorVariableId>(factor + 2U),
        }};
        const FactorId id = graph.add_dense_factor(scope, tables[factor]);
        if (factor_ids != nullptr) {
            factor_ids->push_back(id);
        }
    }
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);
    constexpr std::size_t variables = 4096U;
    constexpr std::size_t factors = variables - 2U;
    ExactFactorConfig config;
    config.max_factor_entries = 64U;

    std::vector<std::array<QComplex, 8>> tables(factors);
    for (std::size_t factor = 0U; factor < factors; ++factor) {
        tables[factor] = nonlinear_table(factor);
    }

    ExactFactorGraph graph(config);
    std::vector<FactorId> factor_ids;
    populate_chain(graph, tables, &factor_ids);

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

    ExactFactorConfig unique_config = config;
    unique_config.reuse_workspace_slots = false;
    ExactFactorGraph unique_graph(unique_config);
    populate_chain(unique_graph, tables);
    ExactFactorPlan unique_plan = unique_graph.compile(retained);
    auto unique_workspace = unique_plan.workspace();
    std::vector<QComplex> unique_values;
    const double unique_query_ms = median_ms([&] {
        unique_values = unique_plan.evaluate(unique_workspace);
    }, 101);

    ExactFactorConfig generic_config = config;
    generic_config.max_compiled_index_entries = 0U;
    ExactFactorGraph generic_graph(generic_config);
    populate_chain(generic_graph, tables);
    ExactFactorPlan generic_plan = generic_graph.compile(retained);
    auto generic_workspace = generic_plan.workspace();
    std::vector<QComplex> generic_values;
    const double generic_query_ms = median_ms([&] {
        generic_values = generic_plan.evaluate(generic_workspace);
    }, 101);

    std::array<QComplex, 4> control{};
    const double control_ms = median_ms([&] {
        control = chain_control(tables);
    }, 101);
    const double initial_error = max_error(qsa_values, control);
    const double unique_error = max_error(unique_values, control);
    const double reuse_unique_error = max_error(qsa_values, unique_values);
    const double generic_error = max_error(generic_values, control);
    const double compiled_generic_error = max_error(qsa_values, generic_values);

    std::vector<QComplex> one_shot_values;
    const double one_shot_ms = median_ms([&] {
        ExactFactorPlan one_shot = graph.compile(retained);
        one_shot_values = one_shot.evaluate();
    }, 11);
    const double one_shot_error = max_error(one_shot_values, control);

    constexpr std::size_t points = 16U;
    const std::size_t rebound_factor = factors / 2U;

    ExactFactorPlan targeted_plan = graph.compile(retained);
    auto targeted_workspace = targeted_plan.workspace();
    auto targeted_tables = tables;
    double targeted_error = 0.0;
    const auto targeted_start = Clock::now();
    for (std::size_t point = 1U; point <= points; ++point) {
        targeted_tables[rebound_factor] = nonlinear_table(rebound_factor, point);
        targeted_plan.rebind_dense_factor(
            factor_ids[rebound_factor], targeted_tables[rebound_factor]);
        qsa_values = targeted_plan.evaluate(targeted_workspace);
        control = chain_control(targeted_tables);
        targeted_error = std::max(targeted_error, max_error(qsa_values, control));
    }
    const auto targeted_stop = Clock::now();
    const double targeted_total_ms =
        std::chrono::duration<double, std::milli>(targeted_stop - targeted_start).count();

    ExactFactorGraph full_graph = graph;
    ExactFactorPlan full_plan = full_graph.compile(retained);
    auto full_workspace = full_plan.workspace();
    auto full_tables = tables;
    double full_rebind_error = 0.0;
    const auto full_start = Clock::now();
    for (std::size_t point = 1U; point <= points; ++point) {
        full_tables[rebound_factor] = nonlinear_table(rebound_factor, point);
        full_graph.set_dense_factor(factor_ids[rebound_factor], full_tables[rebound_factor]);
        full_plan.rebind(full_graph);
        qsa_values = full_plan.evaluate(full_workspace);
        control = chain_control(full_tables);
        full_rebind_error = std::max(full_rebind_error, max_error(qsa_values, control));
    }
    const auto full_stop = Clock::now();
    const double full_rebind_total_ms =
        std::chrono::duration<double, std::milli>(full_stop - full_start).count();

    ExactFactorGraph one_shot_graph = graph;
    auto recompile_tables = tables;
    double one_shot_point_error = 0.0;
    const auto one_shot_start = Clock::now();
    for (std::size_t point = 1U; point <= points; ++point) {
        recompile_tables[rebound_factor] = nonlinear_table(rebound_factor, point);
        one_shot_graph.set_dense_factor(
            factor_ids[rebound_factor], recompile_tables[rebound_factor]);
        ExactFactorPlan one_shot = one_shot_graph.compile(retained);
        one_shot_values = one_shot.evaluate();
        control = chain_control(recompile_tables);
        one_shot_point_error =
            std::max(one_shot_point_error, max_error(one_shot_values, control));
    }
    const auto one_shot_stop = Clock::now();
    const double one_shot_total_ms =
        std::chrono::duration<double, std::milli>(one_shot_stop - one_shot_start).count();

    if (initial_error > 2e-9 || unique_error > 2e-9 ||
        reuse_unique_error > 2e-9 || generic_error > 2e-9 ||
        compiled_generic_error > 2e-9 || one_shot_error > 2e-9 ||
        targeted_error > 2e-9 || full_rebind_error > 2e-9 ||
        one_shot_point_error > 2e-9) {
        std::cerr << "exact factor benchmark control mismatch\n";
        return 1;
    }

    std::cout << "factor_variables=" << variables << '\n';
    std::cout << "factor_source_factors=" << factors << '\n';
    std::cout << "factor_retained_variables=" << retained.size() << '\n';
    std::cout << "factor_route=" << qubit::exact_factor_route_name(plan.route()) << '\n';
    std::cout << "factor_compile_ms=" << compile_ms << '\n';
    std::cout << "factor_prepared_query_ms=" << prepared_query_ms << '\n';
    std::cout << "factor_unique_workspace_query_ms=" << unique_query_ms << '\n';
    std::cout << "factor_reuse_over_unique_workspace_speed=" << unique_query_ms / prepared_query_ms << '\n';
    std::cout << "factor_generic_query_ms=" << generic_query_ms << '\n';
    std::cout << "factor_compiled_over_generic_speed=" << generic_query_ms / prepared_query_ms << '\n';
    std::cout << "factor_one_shot_ms=" << one_shot_ms << '\n';
    std::cout << "factor_prepared_over_one_shot_speed=" << one_shot_ms / prepared_query_ms << '\n';
    std::cout << "factor_specialized_control_ms=" << control_ms << '\n';
    std::cout << "factor_prepared_over_specialized_control_ratio=" << control_ms / prepared_query_ms << '\n';
    std::cout << "factor_generic_over_specialized_control_ratio=" << control_ms / generic_query_ms << '\n';
    std::cout << "factor_peak_union_variables=" << plan.stats().peak_union_variables << '\n';
    std::cout << "factor_peak_entries=" << plan.stats().peak_factor_entries << '\n';
    std::cout << "factor_compiled_index_entries=" << plan.stats().compiled_index_entries << '\n';
    std::cout << "factor_generic_compiled_index_entries=" << generic_plan.stats().compiled_index_entries << '\n';
    std::cout << "factor_workspace_slots=" << plan.stats().workspace_slots << '\n';
    std::cout << "factor_unique_workspace_slots=" << unique_plan.stats().workspace_slots << '\n';
    std::cout << "factor_elimination_steps=" << plan.step_count() << '\n';
    std::cout << "factor_plan_bytes=" << plan.estimated_bytes() << '\n';
    std::cout << "factor_unique_plan_bytes=" << unique_plan.estimated_bytes() << '\n';
    std::cout << "factor_generic_plan_bytes=" << generic_plan.estimated_bytes() << '\n';
    std::cout << "factor_compiled_index_memory_bytes="
              << plan.estimated_bytes() - generic_plan.estimated_bytes() << '\n';
    std::cout << "factor_workspace_bytes=" << workspace.estimated_bytes() << '\n';
    std::cout << "factor_unique_workspace_bytes=" << unique_workspace.estimated_bytes() << '\n';
    std::cout << "factor_workspace_memory_reduction="
              << static_cast<double>(unique_workspace.estimated_bytes()) /
                     static_cast<double>(workspace.estimated_bytes()) << '\n';
    std::cout << "factor_initial_error=" << initial_error << '\n';
    std::cout << "factor_unique_workspace_error=" << unique_error << '\n';
    std::cout << "factor_reuse_unique_error=" << reuse_unique_error << '\n';
    std::cout << "factor_generic_error=" << generic_error << '\n';
    std::cout << "factor_compiled_generic_error=" << compiled_generic_error << '\n';
    std::cout << "factor_rebind_points=" << points << '\n';
    std::cout << "factor_targeted_rebind_total_ms=" << targeted_total_ms << '\n';
    std::cout << "factor_full_rebind_total_ms=" << full_rebind_total_ms << '\n';
    std::cout << "factor_targeted_over_full_rebind_speed=" << full_rebind_total_ms / targeted_total_ms << '\n';
    std::cout << "factor_rebind_total_ms=" << full_rebind_total_ms << '\n';
    std::cout << "factor_recompile_total_ms=" << one_shot_total_ms << '\n';
    std::cout << "factor_targeted_over_recompile_speed=" << one_shot_total_ms / targeted_total_ms << '\n';
    std::cout << "factor_rebind_over_recompile_speed=" << one_shot_total_ms / full_rebind_total_ms << '\n';
    std::cout << "factor_targeted_rebind_error=" << targeted_error << '\n';
    std::cout << "factor_rebind_error=" << full_rebind_error << '\n';
    std::cout << "factor_recompile_error=" << one_shot_point_error << '\n';
    std::cout << "factor_targeted_rebind_count=" << targeted_plan.rebind_count() << '\n';
    std::cout << "factor_rebind_count=" << full_plan.rebind_count() << '\n';
    return 0;
}

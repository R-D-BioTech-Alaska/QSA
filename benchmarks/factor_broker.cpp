#include "qubit/qfactor_broker.hpp"

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
using qubit::ExactFactorBrokerPlan;
using qubit::ExactFactorBrokerRoute;
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

    ExactFactorChainPlan direct_chain(graph, retained);
    auto direct_chain_workspace = direct_chain.workspace();
    std::vector<QComplex> direct_chain_values;
    const double direct_chain_query_ms = median_ms([&] {
        direct_chain_values = direct_chain.evaluate(direct_chain_workspace);
    }, 101);

    ExactFactorBrokerPlan broker(graph, retained);
    if (broker.route() != ExactFactorBrokerRoute::ChainTransfer ||
        !broker.chain_rejection().empty()) {
        std::cerr << "exact factor broker did not select ChainTransfer\n";
        return 1;
    }
    auto broker_workspace = broker.workspace();
    std::vector<QComplex> broker_values;
    const double broker_query_ms = median_ms([&] {
        broker_values = broker.evaluate(broker_workspace);
    }, 101);
    const double broker_compile_ms = median_ms([&] {
        ExactFactorBrokerPlan candidate(graph, retained);
        if (candidate.route() != ExactFactorBrokerRoute::ChainTransfer) {
            throw std::runtime_error("broker compile lost ChainTransfer eligibility");
        }
    }, 11);
    const double direct_chain_compile_ms = median_ms([&] {
        ExactFactorChainPlan candidate(graph, retained);
        if (candidate.stats().factor_count != factors) {
            throw std::runtime_error("direct ChainTransfer compile lost source factors");
        }
    }, 11);

    std::array<QComplex, 4> control{};
    const double specialized_ms = median_ms([&] {
        control = specialized_chain(tables);
    }, 101);

    const double broker_control_error = max_error(broker_values, control);
    const double broker_generic_error = max_error(broker_values, generic_values);
    const double broker_direct_chain_error = max_error(broker_values, direct_chain_values);
    if (broker_control_error > 2e-9 || broker_generic_error > 2e-9 ||
        broker_direct_chain_error > 2e-9) {
        std::cerr << "exact factor broker benchmark mismatch\n";
        return 1;
    }

    constexpr std::size_t points = 16U;
    const std::size_t rebound_factor = factors / 2U;
    ExactFactorBrokerPlan broker_updates(graph, retained);
    ExactFactorPlan generic_updates = graph.compile(retained);
    auto broker_update_workspace = broker_updates.workspace();
    auto generic_update_workspace = generic_updates.workspace();
    auto updated_tables = tables;
    double broker_update_error = 0.0;
    const auto broker_update_start = Clock::now();
    for (std::size_t point = 1U; point <= points; ++point) {
        updated_tables[rebound_factor] = nonlinear_table(rebound_factor, point);
        broker_updates.rebind_dense_factor(
            factor_ids[rebound_factor], updated_tables[rebound_factor]);
        broker_values = broker_updates.evaluate(broker_update_workspace);
        control = specialized_chain(updated_tables);
        broker_update_error =
            std::max(broker_update_error, max_error(broker_values, control));
    }
    const auto broker_update_stop = Clock::now();
    const double broker_update_ms =
        std::chrono::duration<double, std::milli>(
            broker_update_stop - broker_update_start).count();

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
        std::chrono::duration<double, std::milli>(
            generic_update_stop - generic_update_start).count();

    if (broker_update_error > 2e-9 || generic_update_error > 2e-9) {
        std::cerr << "exact factor broker update benchmark mismatch\n";
        return 1;
    }

    ExactFactorGraph fallback_graph;
    const FactorVariableId a = fallback_graph.add_variable(2U);
    const FactorVariableId b = fallback_graph.add_variable(2U);
    const FactorVariableId c = fallback_graph.add_variable(2U);
    const std::array<QComplex, 4> fallback_values{{
        QComplex{1.0}, QComplex{0.7}, QComplex{0.4}, QComplex{1.2},
    }};
    const std::array<FactorVariableId, 2> first{{a, b}};
    const std::array<FactorVariableId, 2> second{{a, c}};
    static_cast<void>(fallback_graph.add_dense_factor(first, fallback_values));
    static_cast<void>(fallback_graph.add_dense_factor(second, fallback_values));
    const std::array<FactorVariableId, 1> fallback_retained{{c}};

    ExactFactorBrokerPlan fallback(fallback_graph, fallback_retained);
    if (fallback.route() != ExactFactorBrokerRoute::VariableElimination ||
        fallback.chain_rejection().empty()) {
        std::cerr << "exact factor broker did not fail closed to variable elimination\n";
        return 1;
    }
    ExactFactorPlan fallback_generic(fallback_graph, fallback_retained);
    auto fallback_workspace = fallback.workspace();
    auto fallback_generic_workspace = fallback_generic.workspace();
    std::vector<QComplex> fallback_values_broker;
    std::vector<QComplex> fallback_values_generic;
    const double fallback_query_ms = median_ms([&] {
        fallback_values_broker = fallback.evaluate(fallback_workspace);
    }, 101);
    const double fallback_generic_query_ms = median_ms([&] {
        fallback_values_generic = fallback_generic.evaluate(fallback_generic_workspace);
    }, 101);
    const double fallback_error = max_error(
        fallback_values_broker, fallback_values_generic);
    if (fallback_error > 2e-9) {
        std::cerr << "exact factor broker fallback mismatch\n";
        return 1;
    }
    const double fallback_compile_ms = median_ms([&] {
        ExactFactorBrokerPlan candidate(fallback_graph, fallback_retained);
        if (candidate.route() != ExactFactorBrokerRoute::VariableElimination) {
            throw std::runtime_error("fallback broker compile selected wrong route");
        }
    }, 101);
    const double fallback_generic_compile_ms = median_ms([&] {
        ExactFactorPlan candidate(fallback_graph, fallback_retained);
        if (candidate.output_entries() != 2U) {
            throw std::runtime_error("direct fallback compile produced wrong output size");
        }
    }, 101);

    std::cout << "broker_variables=" << variables << '\n';
    std::cout << "broker_factors=" << factors << '\n';
    std::cout << "broker_width=3\n";
    std::cout << "broker_route=" << broker.route_name() << '\n';
    std::cout << "broker_chain_rejection_empty=" << broker.chain_rejection().empty() << '\n';
    std::cout << "broker_compile_ms=" << broker_compile_ms << '\n';
    std::cout << "broker_direct_chain_compile_ms=" << direct_chain_compile_ms << '\n';
    std::cout << "broker_compile_over_direct_chain=" << broker_compile_ms / direct_chain_compile_ms << '\n';
    std::cout << "broker_query_ms=" << broker_query_ms << '\n';
    std::cout << "broker_direct_chain_query_ms=" << direct_chain_query_ms << '\n';
    std::cout << "broker_query_over_direct_chain=" << broker_query_ms / direct_chain_query_ms << '\n';
    std::cout << "broker_generic_query_ms=" << generic_query_ms << '\n';
    std::cout << "broker_over_generic_speed=" << generic_query_ms / broker_query_ms << '\n';
    std::cout << "broker_specialized_control_ms=" << specialized_ms << '\n';
    std::cout << "broker_specialized_over_broker_ratio=" << specialized_ms / broker_query_ms << '\n';
    std::cout << "broker_control_error=" << broker_control_error << '\n';
    std::cout << "broker_generic_error=" << broker_generic_error << '\n';
    std::cout << "broker_direct_chain_error=" << broker_direct_chain_error << '\n';
    std::cout << "broker_plan_bytes=" << broker.estimated_bytes() << '\n';
    std::cout << "broker_workspace_bytes=" << broker_workspace.estimated_bytes() << '\n';
    std::cout << "broker_generic_plan_bytes=" << generic.estimated_bytes() << '\n';
    std::cout << "broker_generic_workspace_bytes=" << generic_workspace.estimated_bytes() << '\n';
    std::cout << "broker_update_points=" << points << '\n';
    std::cout << "broker_update_total_ms=" << broker_update_ms << '\n';
    std::cout << "broker_generic_update_total_ms=" << generic_update_ms << '\n';
    std::cout << "broker_update_over_generic_speed=" << generic_update_ms / broker_update_ms << '\n';
    std::cout << "broker_update_error=" << broker_update_error << '\n';
    std::cout << "broker_generic_update_error=" << generic_update_error << '\n';
    std::cout << "broker_fallback_route=" << fallback.route_name() << '\n';
    std::cout << "broker_fallback_reason_visible=" << !fallback.chain_rejection().empty() << '\n';
    std::cout << "broker_fallback_query_ms=" << fallback_query_ms << '\n';
    std::cout << "broker_fallback_generic_query_ms=" << fallback_generic_query_ms << '\n';
    std::cout << "broker_fallback_query_over_generic=" << fallback_query_ms / fallback_generic_query_ms << '\n';
    std::cout << "broker_fallback_compile_ms=" << fallback_compile_ms << '\n';
    std::cout << "broker_fallback_generic_compile_ms=" << fallback_generic_compile_ms << '\n';
    std::cout << "broker_fallback_compile_over_generic=" << fallback_compile_ms / fallback_generic_compile_ms << '\n';
    std::cout << "broker_fallback_error=" << fallback_error << '\n';
    return 0;
}

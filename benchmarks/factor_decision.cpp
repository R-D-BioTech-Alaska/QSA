#include "qubit/qfactor_decision.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::ExactFactorConfig;
using qubit::ExactFactorDecisionConfig;
using qubit::ExactFactorDecisionPlan;
using qubit::ExactFactorGraph;
using qubit::ExactFactorPlan;
using qubit::FactorVariableId;
using qubit::QComplex;
using qubit::QStateError;

constexpr std::array<QComplex, 2> weights{{
    {0.41, 0.07},
    {0.73, -0.03},
}};

[[nodiscard]] ExactFactorGraph equality_clique(
    std::size_t variables,
    std::size_t max_factor_entries) {
    ExactFactorConfig config;
    config.max_variables = variables + 1U;
    config.max_factors = variables * variables + 2U;
    config.max_factor_entries = max_factor_entries;
    ExactFactorGraph graph(config);
    for (std::size_t variable = 0U; variable < variables; ++variable) {
        static_cast<void>(graph.add_variable(2U));
    }

    const std::array<QComplex, 4> equality{{
        {1.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        {1.0, 0.0},
    }};
    for (std::size_t first = 0U; first < variables; ++first) {
        for (std::size_t second = first + 1U; second < variables; ++second) {
            const std::array<FactorVariableId, 2> scope{{
                static_cast<FactorVariableId>(first),
                static_cast<FactorVariableId>(second),
            }};
            static_cast<void>(graph.add_dense_factor(scope, equality));
        }
    }
    const std::array<FactorVariableId, 1> unary_scope{{0U}};
    static_cast<void>(graph.add_dense_factor(unary_scope, weights));
    return graph;
}

template <class Function>
[[nodiscard]] double median_ms(
    Function&& function,
    std::size_t repeats,
    std::size_t iterations = 1U) {
    std::vector<double> samples;
    samples.reserve(repeats);
    for (std::size_t repeat = 0U; repeat < repeats; ++repeat) {
        const auto start = Clock::now();
        for (std::size_t iteration = 0U; iteration < iterations; ++iteration) {
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

void require_close(QComplex actual, QComplex expected, const char* message) {
    if (std::abs(actual.re - expected.re) > 2e-11 ||
        std::abs(actual.im - expected.im) > 2e-11) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);
    const std::array<FactorVariableId, 1> retained{{0U}};

    constexpr std::size_t matched_variables = 16U;
    ExactFactorGraph matched = equality_clique(matched_variables, 1U << 18U);
    ExactFactorDecisionConfig decision_config;
    decision_config.max_nodes = 1U << 18U;
    decision_config.max_apply_pairs = 1U << 20U;
    decision_config.max_variables = 128U;

    std::optional<ExactFactorDecisionPlan> decision;
    const double decision_compile_ms = median_ms([&] {
        decision.emplace(matched, retained, decision_config);
    }, 7U);
    std::optional<ExactFactorPlan> generic;
    const double generic_compile_ms = median_ms([&] {
        generic.emplace(matched, retained);
    }, 5U);

    auto decision_workspace = decision->workspace();
    auto generic_workspace = generic->workspace();
    std::array<QComplex, 2> decision_output{};
    std::array<QComplex, 2> generic_output{};
    const double decision_query_ms = median_ms([&] {
        decision->evaluate(decision_output, decision_workspace);
    }, 11U, 1000U);
    const double generic_query_ms = median_ms([&] {
        generic->evaluate(generic_output, generic_workspace);
    }, 7U, 3U);

    require_close(decision_output[0], weights[0],
                  "decision equality zero marginal differs from specialized control");
    require_close(decision_output[1], weights[1],
                  "decision equality one marginal differs from specialized control");
    require_close(generic_output[0], weights[0],
                  "generic equality zero marginal differs from specialized control");
    require_close(generic_output[1], weights[1],
                  "generic equality one marginal differs from specialized control");
    const double decision_generic_error = std::max(
        std::hypot(
            decision_output[0].re - generic_output[0].re,
            decision_output[0].im - generic_output[0].im),
        std::hypot(
            decision_output[1].re - generic_output[1].re,
            decision_output[1].im - generic_output[1].im));

    volatile double control_sink = 0.0;
    const double specialized_control_ms = median_ms([&] {
        control_sink = weights[0].re + weights[1].re;
    }, 11U, 100000U);
    static_cast<void>(control_sink);

    std::cout << "factor_decision_matched_variables=" << matched_variables << '\n';
    std::cout << "factor_decision_matched_factors=" << matched.factor_count() << '\n';
    std::cout << "factor_decision_matched_logical_assignments="
              << (std::size_t{1U} << matched_variables) << '\n';
    std::cout << "factor_decision_compiled_nodes=" << decision->stats().compiled_nodes << '\n';
    std::cout << "factor_decision_peak_compile_nodes=" << decision->stats().peak_compile_nodes << '\n';
    std::cout << "factor_decision_apply_pairs_peak=" << decision->stats().apply_pairs_peak << '\n';
    std::cout << "factor_decision_compile_ms=" << decision_compile_ms << '\n';
    std::cout << "factor_decision_generic_compile_ms=" << generic_compile_ms << '\n';
    std::cout << "factor_decision_compile_speedup="
              << generic_compile_ms / decision_compile_ms << '\n';
    std::cout << "factor_decision_query_ms=" << decision_query_ms << '\n';
    std::cout << "factor_decision_generic_query_ms=" << generic_query_ms << '\n';
    std::cout << "factor_decision_query_speedup="
              << generic_query_ms / decision_query_ms << '\n';
    std::cout << "factor_decision_specialized_control_ms=" << specialized_control_ms << '\n';
    std::cout << "factor_decision_specialized_over_decision_ratio="
              << specialized_control_ms / decision_query_ms << '\n';
    std::cout << "factor_decision_plan_bytes=" << decision->estimated_bytes() << '\n';
    std::cout << "factor_decision_workspace_bytes=" << decision_workspace.estimated_bytes() << '\n';
    std::cout << "factor_decision_generic_plan_bytes=" << generic->estimated_bytes() << '\n';
    std::cout << "factor_decision_generic_workspace_bytes=" << generic_workspace.estimated_bytes() << '\n';
    std::cout << "factor_decision_matched_error=" << decision_generic_error << '\n';

    constexpr std::size_t capability_variables = 48U;
    ExactFactorGraph capability = equality_clique(capability_variables, 1U << 20U);
    std::optional<ExactFactorDecisionPlan> capability_decision;
    const double capability_compile_ms = median_ms([&] {
        capability_decision.emplace(capability, retained, decision_config);
    }, 5U);
    auto capability_workspace = capability_decision->workspace();
    std::array<QComplex, 2> capability_output{};
    const double capability_query_ms = median_ms([&] {
        capability_decision->evaluate(capability_output, capability_workspace);
    }, 11U, 1000U);
    require_close(capability_output[0], weights[0],
                  "large decision zero marginal differs from specialized control");
    require_close(capability_output[1], weights[1],
                  "large decision one marginal differs from specialized control");

    bool generic_rejected = false;
    double generic_reject_ms = 0.0;
    const auto generic_start = Clock::now();
    try {
        ExactFactorPlan capability_generic(capability, retained);
        static_cast<void>(capability_generic);
    } catch (const QStateError&) {
        generic_rejected = true;
    }
    const auto generic_finish = Clock::now();
    generic_reject_ms = std::chrono::duration<double, std::milli>(
        generic_finish - generic_start).count();
    if (!generic_rejected) {
        std::cerr << "generic VE unexpectedly accepted the high-treewidth capability case\n";
        return 2;
    }

    std::cout << "factor_decision_capability_variables=" << capability_variables << '\n';
    std::cout << "factor_decision_capability_factors=" << capability.factor_count() << '\n';
    std::cout << "factor_decision_capability_logical_assignments=281474976710656\n";
    std::cout << "factor_decision_capability_compiled_nodes="
              << capability_decision->stats().compiled_nodes << '\n';
    std::cout << "factor_decision_capability_peak_compile_nodes="
              << capability_decision->stats().peak_compile_nodes << '\n';
    std::cout << "factor_decision_capability_compile_ms=" << capability_compile_ms << '\n';
    std::cout << "factor_decision_capability_query_ms=" << capability_query_ms << '\n';
    std::cout << "factor_decision_capability_plan_bytes="
              << capability_decision->estimated_bytes() << '\n';
    std::cout << "factor_decision_capability_workspace_bytes="
              << capability_workspace.estimated_bytes() << '\n';
    std::cout << "factor_decision_capability_generic_rejected=1\n";
    std::cout << "factor_decision_capability_generic_reject_ms=" << generic_reject_ms << '\n';
    std::cout << "factor_decision_capability_error=0\n";
    return 0;
}

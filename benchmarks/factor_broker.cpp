#include "qubit/qfactor_broker.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::ExactFactorAffinePlan;
using qubit::ExactFactorBrokerConfig;
using qubit::ExactFactorBrokerPlan;
using qubit::ExactFactorBrokerRoute;
using qubit::ExactFactorBrokerWorkspace;
using qubit::ExactFactorChainPlan;
using qubit::ExactFactorConfig;
using qubit::ExactFactorDecisionConfig;
using qubit::ExactFactorDecisionPlan;
using qubit::ExactFactorGraph;
using qubit::ExactFactorPlan;
using qubit::FactorId;
using qubit::FactorSparseEntry;
using qubit::FactorVariableId;
using qubit::QComplex;
using qubit::QStateError;

#if defined(_MSC_VER)
#define QSA_BROKER_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define QSA_BROKER_NOINLINE __attribute__((noinline))
#else
#define QSA_BROKER_NOINLINE
#endif

volatile double affine_query_sink = 0.0;

constexpr std::array<QComplex, 2> equality_weights{{
    {0.41, 0.07},
    {0.73, -0.03},
}};

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
        const auto stop = Clock::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count() /
            static_cast<double>(iterations));
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

[[nodiscard]] double max_error(
    std::span<const QComplex> actual,
    std::span<const QComplex> expected) {
    double error = 0.0;
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        error = std::max(error, (actual[index] - expected[index]).magnitude());
    }
    return error;
}

[[nodiscard]] std::array<QComplex, 8> nonlinear_table(
    std::size_t factor,
    std::size_t point = 0U) {
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

[[nodiscard]] std::array<QComplex, 4> specialized_chain(
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
                    next[b + 2U * c] +=
                        prefix * table[a + 2U * b + 4U * c];
                }
            }
        }
        current = next;
    }
    return current;
}

struct EqualityGraph {
    ExactFactorGraph graph{};
    FactorId unary{0U};
};

[[nodiscard]] EqualityGraph equality_clique(
    std::size_t variables,
    std::size_t max_factor_entries = 1U << 20U) {
    ExactFactorConfig config;
    config.max_variables = variables + 1U;
    config.max_factors = variables * variables + 2U;
    config.max_factor_entries = max_factor_entries;
    EqualityGraph result{ExactFactorGraph(config), 0U};
    for (std::size_t variable = 0U; variable < variables; ++variable) {
        static_cast<void>(result.graph.add_variable(2U));
    }
    const std::array<QComplex, 4> equality{{
        QComplex{1.0}, QComplex{}, QComplex{}, QComplex{1.0},
    }};
    for (std::size_t first = 0U; first < variables; ++first) {
        for (std::size_t second = first + 1U; second < variables; ++second) {
            const std::array<FactorVariableId, 2> scope{{
                static_cast<FactorVariableId>(first),
                static_cast<FactorVariableId>(second),
            }};
            static_cast<void>(result.graph.add_dense_factor(scope, equality));
        }
    }
    const std::array<FactorVariableId, 1> unary_scope{{0U}};
    result.unary = result.graph.add_dense_factor(unary_scope, equality_weights);
    return result;
}

[[nodiscard]] std::array<QComplex, 2> specialized_equality(
    std::span<const QComplex, 2> weights,
    std::size_t variables) {
    std::array<QComplex, 2> result{{weights[0], weights[1]}};
    volatile std::size_t witness = variables;
    if (witness == 0U) {
        result = {};
    }
    return result;
}

struct XorEquation {
    std::array<FactorVariableId, 3> variables{};
    bool rhs{false};
};

struct XorGraph {
    ExactFactorGraph graph{};
    std::vector<FactorId> factors{};
    std::array<FactorVariableId, 1> retained{{0U}};
};

[[nodiscard]] std::array<QComplex, 8> parity_table(
    bool rhs,
    QComplex coefficient = {1.0, 0.0}) {
    std::array<QComplex, 8> table{};
    for (std::size_t index = 0U; index < table.size(); ++index) {
        if (((std::popcount(index) & 1U) != 0U) == rhs) {
            table[index] = coefficient;
        }
    }
    return table;
}

[[nodiscard]] XorGraph xor_graph(
    std::size_t variables,
    std::size_t equations,
    std::uint64_t seed) {
    ExactFactorConfig config;
    config.max_variables = variables + 1U;
    config.max_factors = equations + 1U;
    XorGraph result{ExactFactorGraph(config), {}, {{0U}}};
    for (std::size_t variable = 0U; variable < variables; ++variable) {
        static_cast<void>(result.graph.add_variable(2U));
    }

    std::mt19937_64 generator(seed);
    std::vector<std::uint8_t> witness(variables);
    for (std::uint8_t& bit : witness) {
        bit = static_cast<std::uint8_t>(generator() & 1U);
    }
    result.factors.reserve(equations);
    while (result.factors.size() < equations) {
        XorEquation equation;
        for (std::size_t position = 0U; position < equation.variables.size();) {
            const FactorVariableId candidate =
                static_cast<FactorVariableId>(generator() % variables);
            bool duplicate = false;
            for (std::size_t previous = 0U; previous < position; ++previous) {
                duplicate = duplicate || equation.variables[previous] == candidate;
            }
            if (!duplicate) {
                equation.variables[position++] = candidate;
            }
        }
        for (const FactorVariableId variable : equation.variables) {
            equation.rhs = equation.rhs != (witness[variable] != 0U);
        }
        const std::array<QComplex, 8> table = parity_table(equation.rhs);
        if ((result.factors.size() & 1U) == 0U) {
            result.factors.push_back(
                result.graph.add_dense_factor(equation.variables, table));
        } else {
            std::vector<FactorSparseEntry> entries;
            entries.reserve(4U);
            for (std::size_t index = 0U; index < table.size(); ++index) {
                if (table[index].re != 0.0 || table[index].im != 0.0) {
                    entries.push_back({index, table[index]});
                }
            }
            result.factors.push_back(
                result.graph.add_sparse_factor(equation.variables, entries));
        }
    }
    return result;
}

QSA_BROKER_NOINLINE void run_affine_broker_query(
    const ExactFactorBrokerPlan& plan,
    ExactFactorBrokerWorkspace& workspace,
    std::array<QComplex, 2>& output) {
    plan.evaluate(output, workspace);
    affine_query_sink = output[0].re + output[1].re;
}

QSA_BROKER_NOINLINE void run_affine_direct_query(
    const ExactFactorAffinePlan& plan,
    std::array<QComplex, 2>& output) {
    plan.evaluate(output);
    affine_query_sink = output[0].re + output[1].re;
}

void chain_evidence() {
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

    ExactFactorPlan generic(graph, retained);
    ExactFactorChainPlan direct_chain(graph, retained);
    ExactFactorBrokerPlan broker(graph, retained);
    if (broker.route() != ExactFactorBrokerRoute::ChainTransfer ||
        !broker.chain_rejection().empty() ||
        !broker.affine_rejection().empty() ||
        !broker.decision_rejection().empty()) {
        throw std::runtime_error("broker lost ChainTransfer precedence");
    }
    auto generic_workspace = generic.workspace();
    auto direct_workspace = direct_chain.workspace();
    auto broker_workspace = broker.workspace();
    std::array<QComplex, 4> generic_output{};
    std::array<QComplex, 4> direct_output{};
    std::array<QComplex, 4> broker_output{};

    const double generic_query_ms = median_ms([&] {
        generic.evaluate(generic_output, generic_workspace);
    }, 31U);
    const double direct_query_ms = median_ms([&] {
        direct_chain.evaluate(direct_output, direct_workspace);
    }, 51U);
    const double broker_query_ms = median_ms([&] {
        broker.evaluate(broker_output, broker_workspace);
    }, 51U);
    const double broker_compile_ms = median_ms([&] {
        ExactFactorBrokerPlan candidate(graph, retained);
        if (candidate.route() != ExactFactorBrokerRoute::ChainTransfer) {
            throw std::runtime_error("chain broker compile selected wrong route");
        }
    }, 9U);

    std::array<QComplex, 4> control{};
    const double specialized_ms = median_ms([&] {
        control = specialized_chain(tables);
    }, 51U);
    const double error = std::max({
        max_error(broker_output, generic_output),
        max_error(broker_output, direct_output),
        max_error(broker_output, control),
    });
    if (error > 2e-9) {
        throw std::runtime_error("chain broker evidence mismatch");
    }

    constexpr std::size_t points = 12U;
    const std::size_t rebound = factors / 2U;
    ExactFactorBrokerPlan update_broker(graph, retained);
    auto update_workspace = update_broker.workspace();
    auto updated_tables = tables;
    double update_error = 0.0;
    const auto update_start = Clock::now();
    for (std::size_t point = 1U; point <= points; ++point) {
        updated_tables[rebound] = nonlinear_table(rebound, point);
        update_broker.rebind_dense_factor(factor_ids[rebound], updated_tables[rebound]);
        update_broker.evaluate(broker_output, update_workspace);
        control = specialized_chain(updated_tables);
        update_error = std::max(update_error, max_error(broker_output, control));
    }
    const double update_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - update_start).count();
    if (update_error > 2e-9) {
        throw std::runtime_error("chain broker rebind evidence mismatch");
    }

    std::cout << "broker_chain_variables=" << variables << '\n';
    std::cout << "broker_chain_factors=" << factors << '\n';
    std::cout << "broker_chain_route=" << broker.route_name() << '\n';
    std::cout << "broker_chain_compile_ms=" << broker_compile_ms << '\n';
    std::cout << "broker_chain_query_ms=" << broker_query_ms << '\n';
    std::cout << "broker_chain_direct_query_ms=" << direct_query_ms << '\n';
    std::cout << "broker_chain_query_over_direct=" << broker_query_ms / direct_query_ms << '\n';
    std::cout << "broker_chain_generic_query_ms=" << generic_query_ms << '\n';
    std::cout << "broker_chain_over_generic_speed=" << generic_query_ms / broker_query_ms << '\n';
    std::cout << "broker_chain_specialized_ms=" << specialized_ms << '\n';
    std::cout << "broker_chain_error=" << error << '\n';
    std::cout << "broker_chain_update_points=" << points << '\n';
    std::cout << "broker_chain_update_total_ms=" << update_ms << '\n';
    std::cout << "broker_chain_update_error=" << update_error << '\n';
}

void affine_evidence() {
    constexpr std::size_t variables = 16U;
    constexpr std::size_t equations = 40U;
    XorGraph source = xor_graph(variables, equations, 0xa6611eULL);

    std::optional<ExactFactorBrokerPlan> broker;
    const double broker_compile_ms = median_ms([&] {
        broker.emplace(source.graph, source.retained);
        if (broker->route() != ExactFactorBrokerRoute::AffineXOR) {
            throw std::runtime_error("broker did not select AffineXOR");
        }
    }, 11U);
    std::optional<ExactFactorAffinePlan> direct;
    const double direct_compile_ms = median_ms([&] {
        direct.emplace(source.graph, source.retained);
    }, 11U);
    std::optional<ExactFactorDecisionPlan> decision;
    const double decision_compile_ms = median_ms([&] {
        decision.emplace(source.graph, source.retained);
    }, 7U);
    std::optional<ExactFactorPlan> generic;
    const double generic_compile_ms = median_ms([&] {
        generic.emplace(source.graph, source.retained);
    }, 5U);

    auto broker_workspace = broker->workspace();
    auto decision_workspace = decision->workspace();
    auto generic_workspace = generic->workspace();
    std::array<QComplex, 2> broker_output{};
    std::array<QComplex, 2> direct_output{};
    std::array<QComplex, 2> decision_output{};
    std::array<QComplex, 2> generic_output{};
    const double broker_query_ms = median_ms([&] {
        run_affine_broker_query(*broker, broker_workspace, broker_output);
    }, 11U, 100000U);
    const double direct_query_ms = median_ms([&] {
        run_affine_direct_query(*direct, direct_output);
    }, 11U, 100000U);
    const double decision_query_ms = median_ms([&] {
        decision->evaluate(decision_output, decision_workspace);
    }, 11U, 1000U);
    const double generic_query_ms = median_ms([&] {
        generic->evaluate(generic_output, generic_workspace);
    }, 7U, 3U);
    const double error = std::max({
        max_error(broker_output, direct_output),
        max_error(broker_output, decision_output),
        max_error(broker_output, generic_output),
    });
    if (error > 2e-11 || broker->chain_rejection().empty() ||
        !broker->affine_rejection().empty() || !broker->decision_rejection().empty()) {
        throw std::runtime_error("affine broker matched evidence failed");
    }

    constexpr std::size_t points = 12U;
    ExactFactorBrokerPlan update_broker(source.graph, source.retained);
    ExactFactorPlan update_generic(source.graph, source.retained);
    auto update_broker_workspace = update_broker.workspace();
    auto update_generic_workspace = update_generic.workspace();
    std::array<QComplex, 2> updated_broker{};
    std::array<QComplex, 2> updated_generic{};
    double update_error = 0.0;
    const auto broker_update_start = Clock::now();
    for (std::size_t point = 1U; point <= points; ++point) {
        const std::array<QComplex, 8> replacement = parity_table(
            false, {1.0 + 0.001 * static_cast<double>(point), 0.0});
        update_broker.rebind_dense_factor(source.factors.front(), replacement);
        run_affine_broker_query(update_broker, update_broker_workspace, updated_broker);
    }
    const double broker_update_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - broker_update_start).count();

    const auto generic_update_start = Clock::now();
    for (std::size_t point = 1U; point <= points; ++point) {
        const std::array<QComplex, 8> replacement = parity_table(
            false, {1.0 + 0.001 * static_cast<double>(point), 0.0});
        update_generic.rebind_dense_factor(source.factors.front(), replacement);
        update_generic.evaluate(updated_generic, update_generic_workspace);
    }
    const double generic_update_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - generic_update_start).count();
    update_error = max_error(updated_broker, updated_generic);
    if (update_error > 2e-11) {
        throw std::runtime_error("affine broker update evidence failed");
    }

    std::cout << "broker_affine_variables=" << variables << '\n';
    std::cout << "broker_affine_equations=" << equations << '\n';
    std::cout << "broker_affine_route=" << broker->route_name() << '\n';
    std::cout << "broker_affine_chain_rejection_visible="
              << !broker->chain_rejection().empty() << '\n';
    std::cout << "broker_affine_compile_ms=" << broker_compile_ms << '\n';
    std::cout << "broker_affine_direct_compile_ms=" << direct_compile_ms << '\n';
    std::cout << "broker_affine_decision_compile_ms=" << decision_compile_ms << '\n';
    std::cout << "broker_affine_generic_compile_ms=" << generic_compile_ms << '\n';
    std::cout << "broker_affine_compile_over_direct=" << broker_compile_ms / direct_compile_ms << '\n';
    std::cout << "broker_affine_compile_speedup_vs_decision=" << decision_compile_ms / broker_compile_ms << '\n';
    std::cout << "broker_affine_compile_speedup_vs_generic=" << generic_compile_ms / broker_compile_ms << '\n';
    std::cout << "broker_affine_query_ms=" << broker_query_ms << '\n';
    std::cout << "broker_affine_direct_query_ms=" << direct_query_ms << '\n';
    std::cout << "broker_affine_query_over_direct=" << broker_query_ms / direct_query_ms << '\n';
    std::cout << "broker_affine_decision_query_ms=" << decision_query_ms << '\n';
    std::cout << "broker_affine_generic_query_ms=" << generic_query_ms << '\n';
    std::cout << "broker_affine_query_speedup_vs_decision=" << decision_query_ms / broker_query_ms << '\n';
    std::cout << "broker_affine_query_speedup_vs_generic=" << generic_query_ms / broker_query_ms << '\n';
    std::cout << "broker_affine_plan_bytes=" << broker->estimated_bytes() << '\n';
    std::cout << "broker_affine_workspace_bytes=" << broker_workspace.estimated_bytes() << '\n';
    std::cout << "broker_affine_direct_plan_bytes=" << direct->estimated_bytes() << '\n';
    std::cout << "broker_affine_decision_plan_bytes=" << decision->estimated_bytes() << '\n';
    std::cout << "broker_affine_generic_plan_bytes=" << generic->estimated_bytes() << '\n';
    std::cout << "broker_affine_error=" << error << '\n';
    std::cout << "broker_affine_update_points=" << points << '\n';
    std::cout << "broker_affine_update_total_ms=" << broker_update_ms << '\n';
    std::cout << "broker_affine_generic_update_total_ms=" << generic_update_ms << '\n';
    std::cout << "broker_affine_update_speedup_vs_generic=" << generic_update_ms / broker_update_ms << '\n';
    std::cout << "broker_affine_update_error=" << update_error << '\n';
}

void decision_evidence() {
    constexpr std::size_t variables = 16U;
    EqualityGraph source = equality_clique(variables, 1U << 18U);
    const std::array<FactorVariableId, 1> retained{{0U}};

    std::optional<ExactFactorBrokerPlan> broker;
    const double broker_compile_ms = median_ms([&] {
        broker.emplace(source.graph, retained);
        if (broker->route() != ExactFactorBrokerRoute::DecisionDiagram) {
            throw std::runtime_error("broker did not select DecisionDiagram");
        }
    }, 7U);
    std::optional<ExactFactorDecisionPlan> direct;
    const double direct_compile_ms = median_ms([&] {
        direct.emplace(source.graph, retained);
    }, 7U);
    std::optional<ExactFactorPlan> generic;
    const double generic_compile_ms = median_ms([&] {
        generic.emplace(source.graph, retained);
    }, 5U);

    auto broker_workspace = broker->workspace();
    auto direct_workspace = direct->workspace();
    auto generic_workspace = generic->workspace();
    std::array<QComplex, 2> broker_output{};
    std::array<QComplex, 2> direct_output{};
    std::array<QComplex, 2> generic_output{};
    std::array<QComplex, 2> control{};
    const double broker_query_ms = median_ms([&] {
        broker->evaluate(broker_output, broker_workspace);
    }, 11U, 1000U);
    const double direct_query_ms = median_ms([&] {
        direct->evaluate(direct_output, direct_workspace);
    }, 11U, 1000U);
    const double generic_query_ms = median_ms([&] {
        generic->evaluate(generic_output, generic_workspace);
    }, 7U, 3U);
    const double specialized_ms = median_ms([&] {
        control = specialized_equality(equality_weights, variables);
    }, 11U, 100000U);

    const double error = std::max({
        max_error(broker_output, direct_output),
        max_error(broker_output, generic_output),
        max_error(broker_output, control),
    });
    if (error > 2e-11 || broker->chain_rejection().empty() ||
        broker->affine_rejection().empty() || !broker->decision_rejection().empty()) {
        throw std::runtime_error("decision broker matched evidence failed");
    }

    constexpr std::size_t points = 12U;
    ExactFactorBrokerPlan update_broker(source.graph, retained);
    ExactFactorPlan update_generic(source.graph, retained);
    auto update_broker_workspace = update_broker.workspace();
    auto update_generic_workspace = update_generic.workspace();
    std::array<QComplex, 2> updated_broker{};
    std::array<QComplex, 2> updated_generic{};
    double broker_update_error = 0.0;
    const auto broker_update_start = Clock::now();
    for (std::size_t point = 1U; point <= points; ++point) {
        const std::array<QComplex, 2> replacement{{
            QComplex{0.31 + 0.005 * static_cast<double>(point), 0.01},
            QComplex{0.67 - 0.003 * static_cast<double>(point), -0.02},
        }};
        update_broker.rebind_dense_factor(source.unary, replacement);
        update_broker.evaluate(updated_broker, update_broker_workspace);
        broker_update_error = std::max(
            broker_update_error, max_error(updated_broker, replacement));
    }
    const double broker_update_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - broker_update_start).count();

    double generic_update_error = 0.0;
    const auto generic_update_start = Clock::now();
    for (std::size_t point = 1U; point <= points; ++point) {
        const std::array<QComplex, 2> replacement{{
            QComplex{0.31 + 0.005 * static_cast<double>(point), 0.01},
            QComplex{0.67 - 0.003 * static_cast<double>(point), -0.02},
        }};
        update_generic.rebind_dense_factor(source.unary, replacement);
        update_generic.evaluate(updated_generic, update_generic_workspace);
        generic_update_error = std::max(
            generic_update_error, max_error(updated_generic, replacement));
    }
    const double generic_update_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - generic_update_start).count();
    if (broker_update_error > 2e-11 || generic_update_error > 2e-11) {
        throw std::runtime_error("decision broker rebind evidence failed");
    }

    std::cout << "broker_decision_variables=" << variables << '\n';
    std::cout << "broker_decision_factors=" << source.graph.factor_count() << '\n';
    std::cout << "broker_decision_route=" << broker->route_name() << '\n';
    std::cout << "broker_decision_chain_rejection_visible="
              << !broker->chain_rejection().empty() << '\n';
    std::cout << "broker_decision_affine_rejection_visible="
              << !broker->affine_rejection().empty() << '\n';
    std::cout << "broker_decision_compile_ms=" << broker_compile_ms << '\n';
    std::cout << "broker_decision_direct_compile_ms=" << direct_compile_ms << '\n';
    std::cout << "broker_decision_generic_compile_ms=" << generic_compile_ms << '\n';
    std::cout << "broker_decision_compile_over_direct=" << broker_compile_ms / direct_compile_ms << '\n';
    std::cout << "broker_decision_affine_preflight_extra_ms=" << broker_compile_ms - direct_compile_ms << '\n';
    std::cout << "broker_decision_compile_speedup_vs_generic=" << generic_compile_ms / broker_compile_ms << '\n';
    std::cout << "broker_decision_query_ms=" << broker_query_ms << '\n';
    std::cout << "broker_decision_direct_query_ms=" << direct_query_ms << '\n';
    std::cout << "broker_decision_query_over_direct=" << broker_query_ms / direct_query_ms << '\n';
    std::cout << "broker_decision_generic_query_ms=" << generic_query_ms << '\n';
    std::cout << "broker_decision_query_speedup_vs_generic=" << generic_query_ms / broker_query_ms << '\n';
    std::cout << "broker_decision_specialized_ms=" << specialized_ms << '\n';
    std::cout << "broker_decision_specialized_over_broker=" << specialized_ms / broker_query_ms << '\n';
    std::cout << "broker_decision_plan_bytes=" << broker->estimated_bytes() << '\n';
    std::cout << "broker_decision_workspace_bytes=" << broker_workspace.estimated_bytes() << '\n';
    std::cout << "broker_decision_direct_plan_bytes=" << direct->estimated_bytes() << '\n';
    std::cout << "broker_decision_generic_plan_bytes=" << generic->estimated_bytes() << '\n';
    std::cout << "broker_decision_error=" << error << '\n';
    std::cout << "broker_decision_update_points=" << points << '\n';
    std::cout << "broker_decision_update_total_ms=" << broker_update_ms << '\n';
    std::cout << "broker_decision_generic_update_total_ms=" << generic_update_ms << '\n';
    std::cout << "broker_decision_update_speedup_vs_generic=" << generic_update_ms / broker_update_ms << '\n';
    std::cout << "broker_decision_update_error=" << broker_update_error << '\n';
}

void affine_capability_evidence() {
    constexpr std::size_t variables = 512U;
    constexpr std::size_t equations = 1536U;
    XorGraph source = xor_graph(variables, equations, 0x3a6611eULL);

    std::optional<ExactFactorBrokerPlan> broker;
    const double broker_compile_ms = median_ms([&] {
        broker.emplace(source.graph, source.retained);
        if (broker->route() != ExactFactorBrokerRoute::AffineXOR) {
            throw std::runtime_error("large parity graph lost AffineXOR route");
        }
    }, 3U);
    std::optional<ExactFactorAffinePlan> direct;
    const double direct_compile_ms = median_ms([&] {
        direct.emplace(source.graph, source.retained);
    }, 3U);
    auto workspace = broker->workspace();
    std::array<QComplex, 2> broker_output{};
    std::array<QComplex, 2> direct_output{};
    const double broker_query_ms = median_ms([&] {
        run_affine_broker_query(*broker, workspace, broker_output);
    }, 11U, 100000U);
    const double direct_query_ms = median_ms([&] {
        run_affine_direct_query(*direct, direct_output);
    }, 11U, 100000U);
    const double error = max_error(broker_output, direct_output);
    if (error > 2e-11) {
        throw std::runtime_error("large AffineXOR broker result mismatch");
    }

    ExactFactorDecisionConfig decision_config;
    decision_config.max_variables = variables;
    decision_config.max_nodes = 65'536U;
    decision_config.max_apply_pairs = 262'144U;
    bool decision_rejected = false;
    const auto decision_start = Clock::now();
    try {
        ExactFactorDecisionPlan decision(source.graph, source.retained, decision_config);
        static_cast<void>(decision);
    } catch (const QStateError&) {
        decision_rejected = true;
    }
    const double decision_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - decision_start).count();

    bool generic_rejected = false;
    const auto generic_start = Clock::now();
    try {
        ExactFactorPlan generic(source.graph, source.retained);
        static_cast<void>(generic);
    } catch (const QStateError&) {
        generic_rejected = true;
    }
    const double generic_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - generic_start).count();
    if (!decision_rejected || !generic_rejected) {
        throw std::runtime_error("large parity capability controls unexpectedly accepted");
    }

    std::cout << "broker_affine_capability_variables=" << variables << '\n';
    std::cout << "broker_affine_capability_equations=" << equations << '\n';
    std::cout << "broker_affine_capability_route=" << broker->route_name() << '\n';
    std::cout << "broker_affine_capability_compile_ms=" << broker_compile_ms << '\n';
    std::cout << "broker_affine_capability_direct_compile_ms=" << direct_compile_ms << '\n';
    std::cout << "broker_affine_capability_compile_over_direct=" << broker_compile_ms / direct_compile_ms << '\n';
    std::cout << "broker_affine_capability_query_ms=" << broker_query_ms << '\n';
    std::cout << "broker_affine_capability_direct_query_ms=" << direct_query_ms << '\n';
    std::cout << "broker_affine_capability_query_over_direct=" << broker_query_ms / direct_query_ms << '\n';
    std::cout << "broker_affine_capability_plan_bytes=" << broker->estimated_bytes() << '\n';
    std::cout << "broker_affine_capability_workspace_bytes=" << workspace.estimated_bytes() << '\n';
    std::cout << "broker_affine_capability_decision_rejected=1\n";
    std::cout << "broker_affine_capability_decision_reject_ms=" << decision_ms << '\n';
    std::cout << "broker_affine_capability_generic_rejected=1\n";
    std::cout << "broker_affine_capability_generic_reject_ms=" << generic_ms << '\n';
    std::cout << "broker_affine_capability_error=" << error << '\n';
}

void decision_capability_evidence() {
    constexpr std::size_t variables = 48U;
    EqualityGraph source = equality_clique(variables);
    const std::array<FactorVariableId, 1> retained{{0U}};
    std::optional<ExactFactorBrokerPlan> broker;
    const double compile_ms = median_ms([&] {
        broker.emplace(source.graph, retained);
        if (broker->route() != ExactFactorBrokerRoute::DecisionDiagram) {
            throw std::runtime_error("large repeated graph lost DecisionDiagram route");
        }
    }, 5U);
    auto workspace = broker->workspace();
    std::array<QComplex, 2> output{};
    const double query_ms = median_ms([&] {
        broker->evaluate(output, workspace);
    }, 11U, 1000U);
    const double error = max_error(output, equality_weights);
    if (error > 2e-11 || broker->affine_rejection().empty()) {
        throw std::runtime_error("large DecisionDiagram broker result mismatch");
    }

    bool generic_rejected = false;
    const auto generic_start = Clock::now();
    try {
        ExactFactorPlan generic(source.graph, retained);
        static_cast<void>(generic);
    } catch (const QStateError&) {
        generic_rejected = true;
    }
    const double generic_reject_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - generic_start).count();
    if (!generic_rejected) {
        throw std::runtime_error("generic VE unexpectedly accepted capability graph");
    }

    std::cout << "broker_decision_capability_variables=" << variables << '\n';
    std::cout << "broker_decision_capability_factors=" << source.graph.factor_count() << '\n';
    std::cout << "broker_decision_capability_logical_assignments=281474976710656\n";
    std::cout << "broker_decision_capability_route=" << broker->route_name() << '\n';
    std::cout << "broker_decision_capability_affine_rejection_visible="
              << !broker->affine_rejection().empty() << '\n';
    std::cout << "broker_decision_capability_compile_ms=" << compile_ms << '\n';
    std::cout << "broker_decision_capability_query_ms=" << query_ms << '\n';
    std::cout << "broker_decision_capability_plan_bytes=" << broker->estimated_bytes() << '\n';
    std::cout << "broker_decision_capability_workspace_bytes=" << workspace.estimated_bytes() << '\n';
    std::cout << "broker_decision_capability_generic_rejected=1\n";
    std::cout << "broker_decision_capability_generic_reject_ms=" << generic_reject_ms << '\n';
    std::cout << "broker_decision_capability_error=" << error << '\n';
}

void fallback_evidence() {
    ExactFactorGraph graph;
    const FactorVariableId variable = graph.add_variable(3U);
    const std::array<FactorVariableId, 1> scope{{variable}};
    const std::array<QComplex, 3> values{{
        QComplex{0.2}, QComplex{0.3}, QComplex{0.5},
    }};
    static_cast<void>(graph.add_dense_factor(scope, values));

    ExactFactorBrokerPlan broker(graph);
    if (broker.route() != ExactFactorBrokerRoute::VariableElimination ||
        broker.chain_rejection().empty() || broker.affine_rejection().empty() ||
        broker.decision_rejection().empty()) {
        throw std::runtime_error("broker did not fail closed to generic VE");
    }
    ExactFactorPlan generic(graph);
    auto broker_workspace = broker.workspace();
    auto generic_workspace = generic.workspace();
    std::array<QComplex, 1> broker_output{};
    std::array<QComplex, 1> generic_output{};
    const double broker_query_ms = median_ms([&] {
        broker.evaluate(broker_output, broker_workspace);
    }, 101U, 100U);
    const double generic_query_ms = median_ms([&] {
        generic.evaluate(generic_output, generic_workspace);
    }, 101U, 100U);
    const double error = max_error(broker_output, generic_output);
    if (error > 2e-11) {
        throw std::runtime_error("generic fallback broker mismatch");
    }
    const double broker_compile_ms = median_ms([&] {
        ExactFactorBrokerPlan candidate(graph);
        if (candidate.route() != ExactFactorBrokerRoute::VariableElimination) {
            throw std::runtime_error("fallback compile selected wrong route");
        }
    }, 101U);
    const double generic_compile_ms = median_ms([&] {
        ExactFactorPlan candidate(graph);
        static_cast<void>(candidate);
    }, 101U);

    XorGraph parity = xor_graph(16U, 40U, 0xa6611eULL);
    ExactFactorBrokerConfig affine_capped;
    affine_capped.affine.max_equations = 1U;
    ExactFactorBrokerPlan affine_to_decision(
        parity.graph, parity.retained, affine_capped);
    if (affine_to_decision.route() != ExactFactorBrokerRoute::DecisionDiagram ||
        affine_to_decision.affine_rejection().empty() ||
        !affine_to_decision.decision_rejection().empty()) {
        throw std::runtime_error("Affine resource cap did not fall through to Decision");
    }

    ExactFactorBrokerConfig both_capped = affine_capped;
    both_capped.decision.max_nodes = 2U;
    ExactFactorBrokerPlan both_to_generic(
        parity.graph, parity.retained, both_capped);
    if (both_to_generic.route() != ExactFactorBrokerRoute::VariableElimination ||
        both_to_generic.affine_rejection().empty() ||
        both_to_generic.decision_rejection().empty()) {
        throw std::runtime_error("combined resource caps did not fall through to generic VE");
    }

    std::cout << "broker_fallback_route=" << broker.route_name() << '\n';
    std::cout << "broker_fallback_chain_rejection_visible=" << !broker.chain_rejection().empty() << '\n';
    std::cout << "broker_fallback_affine_rejection_visible=" << !broker.affine_rejection().empty() << '\n';
    std::cout << "broker_fallback_decision_rejection_visible=" << !broker.decision_rejection().empty() << '\n';
    std::cout << "broker_fallback_query_ms=" << broker_query_ms << '\n';
    std::cout << "broker_fallback_generic_query_ms=" << generic_query_ms << '\n';
    std::cout << "broker_fallback_query_over_generic=" << broker_query_ms / generic_query_ms << '\n';
    std::cout << "broker_fallback_compile_ms=" << broker_compile_ms << '\n';
    std::cout << "broker_fallback_generic_compile_ms=" << generic_compile_ms << '\n';
    std::cout << "broker_fallback_compile_over_generic=" << broker_compile_ms / generic_compile_ms << '\n';
    std::cout << "broker_fallback_extra_compile_ms=" << broker_compile_ms - generic_compile_ms << '\n';
    std::cout << "broker_fallback_error=" << error << '\n';
    std::cout << "broker_affine_resource_fallback_route=" << affine_to_decision.route_name() << '\n';
    std::cout << "broker_affine_resource_rejection_visible="
              << !affine_to_decision.affine_rejection().empty() << '\n';
    std::cout << "broker_combined_resource_fallback_route=" << both_to_generic.route_name() << '\n';
    std::cout << "broker_combined_resource_affine_rejection_visible="
              << !both_to_generic.affine_rejection().empty() << '\n';
    std::cout << "broker_combined_resource_decision_rejection_visible="
              << !both_to_generic.decision_rejection().empty() << '\n';
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);
    chain_evidence();
    affine_evidence();
    decision_evidence();
    affine_capability_evidence();
    decision_capability_evidence();
    fallback_evidence();
    return 0;
}

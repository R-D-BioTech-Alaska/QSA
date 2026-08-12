#include "qubit/qfactor_decision.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using qubit::ExactFactorDecisionConfig;
using qubit::ExactFactorDecisionPlan;
using qubit::ExactFactorDecisionWorkspace;
using qubit::ExactFactorGraph;
using qubit::ExactFactorPlan;
using qubit::FactorSparseEntry;
using qubit::FactorVariableId;
using qubit::QComplex;
using qubit::QStateError;

struct ReferenceFactor {
    std::vector<FactorVariableId> variables{};
    std::vector<QComplex> values{};
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(QComplex actual, QComplex expected, const std::string& message) {
    require(std::abs(actual.re - expected.re) <= 3e-11 &&
            std::abs(actual.im - expected.im) <= 3e-11,
            message);
}

void require_vector_close(
    std::span<const QComplex> actual,
    std::span<const QComplex> expected,
    const std::string& message) {
    require(actual.size() == expected.size(), message + " size");
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        require_close(actual[index], expected[index], message);
    }
}

[[nodiscard]] std::vector<QComplex> brute(
    std::size_t variable_count,
    std::span<const ReferenceFactor> factors,
    std::span<const FactorVariableId> retained = {}) {
    require(retained.size() <= 1U, "brute decision control supports one retained variable");
    const std::size_t assignments = std::size_t{1U} << variable_count;
    std::vector<QComplex> output(retained.empty() ? 1U : 2U);
    for (std::size_t assignment = 0U; assignment < assignments; ++assignment) {
        QComplex value{1.0, 0.0};
        for (const ReferenceFactor& factor : factors) {
            std::size_t index = 0U;
            for (std::size_t position = 0U; position < factor.variables.size(); ++position) {
                index |= ((assignment >> factor.variables[position]) & 1U) << position;
            }
            value *= factor.values[index];
        }
        const std::size_t output_index = retained.empty()
            ? 0U
            : ((assignment >> retained.front()) & 1U);
        output[output_index] += value;
    }
    return output;
}

[[nodiscard]] QComplex generated_value(std::uint64_t raw) {
    const double re = 0.2 + 0.03 * static_cast<double>(raw % 13U);
    const double im = 0.004 * static_cast<double>((raw / 13U) % 9U);
    return {re, im};
}

void fixed_cyclic_case() {
    ExactFactorGraph graph;
    for (std::size_t variable = 0U; variable < 5U; ++variable) {
        static_cast<void>(graph.add_variable(2U));
    }

    std::vector<ReferenceFactor> reference;
    const auto add_dense = [&](std::vector<FactorVariableId> scope, std::vector<QComplex> values) {
        static_cast<void>(graph.add_dense_factor(scope, values));
        reference.push_back({std::move(scope), std::move(values)});
    };
    const auto add_sparse = [&](std::vector<FactorVariableId> scope, std::vector<QComplex> values) {
        std::vector<FactorSparseEntry> sparse;
        for (std::size_t index = 0U; index < values.size(); ++index) {
            if (values[index].norm2() != 0.0) {
                sparse.push_back({index, values[index]});
            }
        }
        static_cast<void>(graph.add_sparse_factor(scope, sparse));
        reference.push_back({std::move(scope), std::move(values)});
    };

    add_dense({2U, 0U}, {{0.8, 0.0}, {0.2, 0.01}, {0.3, -0.02}, {0.7, 0.0}});
    add_sparse({1U, 2U}, {{0.9, 0.0}, {}, {}, {0.6, 0.03}});
    add_dense({3U, 1U}, {{0.4, 0.0}, {0.6, 0.0}, {0.7, 0.0}, {0.3, 0.0}});
    add_dense({0U, 3U}, {{0.5, 0.01}, {0.5, 0.0}, {0.1, 0.0}, {0.9, -0.01}});
    add_dense({4U, 2U, 1U}, {
        {0.7, 0.0}, {0.3, 0.0}, {0.2, 0.0}, {0.8, 0.0},
        {0.6, 0.01}, {0.4, 0.0}, {0.9, 0.0}, {0.1, -0.01},
    });
    add_dense({}, {{0.75, 0.02}});

    ExactFactorDecisionPlan partition_plan(graph);
    ExactFactorPlan generic_partition(graph);
    const auto expected_partition = brute(5U, reference);
    require(partition_plan.route_name() == std::string("ExactFactorDecisionDiagram"),
            "decision route name changed");
    require_vector_close(partition_plan.evaluate(), expected_partition,
                         "decision cyclic partition differs from brute force");
    require_vector_close(generic_partition.evaluate(), expected_partition,
                         "generic cyclic partition differs from brute force");

    const std::array<FactorVariableId, 1> retained{{2U}};
    ExactFactorDecisionPlan marginal_plan(graph, retained);
    ExactFactorPlan generic_marginal(graph, retained);
    const auto expected_marginal = brute(5U, reference, retained);
    require_vector_close(marginal_plan.evaluate(), expected_marginal,
                         "decision cyclic marginal differs from brute force");
    require_vector_close(generic_marginal.evaluate(), expected_marginal,
                         "generic cyclic marginal differs from brute force");
    require_close(marginal_plan.partition(), generic_marginal.partition(),
                  "decision marginal partition differs from generic");

    const auto normalized = marginal_plan.normalized_marginal();
    require_close(normalized[0] + normalized[1], {1.0, 0.0},
                  "decision normalized marginal does not sum to one");

    ExactFactorDecisionWorkspace bad_workspace;
    std::array<QComplex, 2> output{};
    bool rejected = false;
    try {
        marginal_plan.evaluate(output, bad_workspace);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "decision workspace mismatch was not rejected");
}

void randomized_binary_graphs() {
    std::mt19937_64 generator(0xd1a6a9ULL);
    for (std::size_t trial = 0U; trial < 40U; ++trial) {
        const std::size_t variables = 3U + static_cast<std::size_t>(generator() % 5U);
        ExactFactorGraph graph;
        for (std::size_t variable = 0U; variable < variables; ++variable) {
            static_cast<void>(graph.add_variable(2U));
        }

        std::vector<ReferenceFactor> reference;
        const std::size_t factor_count = 3U + static_cast<std::size_t>(generator() % 6U);
        for (std::size_t factor_index = 0U; factor_index < factor_count; ++factor_index) {
            const std::size_t width = 1U + static_cast<std::size_t>(generator() % std::min<std::size_t>(3U, variables));
            std::vector<FactorVariableId> scope;
            while (scope.size() < width) {
                const FactorVariableId variable = static_cast<FactorVariableId>(generator() % variables);
                if (std::find(scope.begin(), scope.end(), variable) == scope.end()) {
                    scope.push_back(variable);
                }
            }
            std::shuffle(scope.begin(), scope.end(), generator);
            std::vector<QComplex> dense(std::size_t{1U} << width);
            for (std::size_t index = 0U; index < dense.size(); ++index) {
                dense[index] = generated_value(generator());
            }
            if ((trial + factor_index) % 2U == 0U) {
                static_cast<void>(graph.add_dense_factor(scope, dense));
            } else {
                std::vector<FactorSparseEntry> sparse;
                for (std::size_t index = 0U; index < dense.size(); ++index) {
                    if ((generator() & 3U) != 0U) {
                        sparse.push_back({index, dense[index]});
                    } else {
                        dense[index] = {};
                    }
                }
                static_cast<void>(graph.add_sparse_factor(scope, sparse));
            }
            reference.push_back({scope, dense});
        }

        if (trial % 5U == 0U) {
            const QComplex scalar = generated_value(generator());
            static_cast<void>(graph.add_dense_factor(
                std::vector<FactorVariableId>{}, std::vector<QComplex>{scalar}));
            reference.push_back({{}, {scalar}});
        }

        ExactFactorDecisionPlan partition_plan(graph);
        ExactFactorPlan generic_partition(graph);
        const auto expected_partition = brute(variables, reference);
        require_vector_close(partition_plan.evaluate(), expected_partition,
                             "random decision partition differs from brute force");
        require_vector_close(generic_partition.evaluate(), expected_partition,
                             "random generic partition differs from brute force");

        const std::array<FactorVariableId, 1> retained{{
            static_cast<FactorVariableId>(generator() % variables),
        }};
        ExactFactorDecisionPlan marginal_plan(graph, retained);
        ExactFactorPlan generic_marginal(graph, retained);
        const auto expected_marginal = brute(variables, reference, retained);
        require_vector_close(marginal_plan.evaluate(), expected_marginal,
                             "random decision marginal differs from brute force");
        require_vector_close(generic_marginal.evaluate(), expected_marginal,
                             "random generic marginal differs from brute force");
    }
}

void rejection_cases() {
    ExactFactorGraph nonbinary;
    static_cast<void>(nonbinary.add_variable(3U));
    static_cast<void>(nonbinary.add_dense_factor(
        std::vector<FactorVariableId>{0U},
        std::vector<QComplex>{{0.2, 0.0}, {0.3, 0.0}, {0.5, 0.0}}));
    bool rejected = false;
    try {
        ExactFactorDecisionPlan plan(nonbinary);
        static_cast<void>(plan);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "nonbinary decision graph was not rejected");

    ExactFactorGraph simple;
    static_cast<void>(simple.add_variable(2U));
    static_cast<void>(simple.add_variable(2U));
    static_cast<void>(simple.add_dense_factor(
        std::vector<FactorVariableId>{0U, 1U},
        std::vector<QComplex>{{0.1, 0.0}, {0.2, 0.0}, {0.3, 0.0}, {0.4, 0.0}}));
    const std::array<FactorVariableId, 2> two_retained{{0U, 1U}};
    rejected = false;
    try {
        ExactFactorDecisionPlan plan(simple, two_retained);
        static_cast<void>(plan);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "multi-retained decision query was not rejected");

    ExactFactorDecisionConfig node_cap;
    node_cap.max_nodes = 2U;
    rejected = false;
    try {
        ExactFactorDecisionPlan plan(simple, {}, node_cap);
        static_cast<void>(plan);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "decision node cap was not enforced");

    ExactFactorDecisionConfig variable_cap;
    variable_cap.max_variables = 1U;
    rejected = false;
    try {
        ExactFactorDecisionPlan plan(simple, {}, variable_cap);
        static_cast<void>(plan);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "decision variable recursion cap was not enforced");

    ExactFactorGraph zero;
    static_cast<void>(zero.add_variable(2U));
    static_cast<void>(zero.add_dense_factor(
        std::vector<FactorVariableId>{0U},
        std::vector<QComplex>{{0.0, 0.0}, {0.0, 0.0}}));
    const std::array<FactorVariableId, 1> retained{{0U}};
    ExactFactorDecisionPlan zero_plan(zero, retained);
    rejected = false;
    try {
        static_cast<void>(zero_plan.normalized_marginal());
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "zero decision partition normalization was not rejected");
}

}  // namespace

int main() {
    fixed_cyclic_case();
    randomized_binary_graphs();
    rejection_cases();
    return 0;
}

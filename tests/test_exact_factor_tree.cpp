#include "qubit/qfactor_tree.hpp"

#include <algorithm>
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

using qubit::ExactFactorGraph;
using qubit::ExactFactorPlan;
using qubit::ExactFactorTreePlan;
using qubit::ExactFactorTreeWorkspace;
using qubit::FactorId;
using qubit::FactorSparseEntry;
using qubit::FactorVariableId;
using qubit::QComplex;
using qubit::QStateError;

struct ReferenceFactor {
    std::vector<FactorVariableId> variables{};
    std::vector<QComplex> dense{};
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(QComplex actual, QComplex expected, const std::string& message) {
    require(std::abs(actual.re - expected.re) <= 2e-11 &&
            std::abs(actual.im - expected.im) <= 2e-11,
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
    std::span<const std::size_t> dimensions,
    std::span<const ReferenceFactor> factors,
    std::span<const FactorVariableId> retained = {}) {
    require(retained.size() <= 1U, "brute control supports at most one retained variable");
    std::size_t total = 1U;
    std::vector<std::size_t> global_strides(dimensions.size(), 1U);
    for (std::size_t variable = 0U; variable < dimensions.size(); ++variable) {
        global_strides[variable] = total;
        total *= dimensions[variable];
    }

    std::vector<QComplex> output(
        retained.empty() ? 1U : dimensions[retained.front()]);
    std::vector<std::size_t> coordinates(dimensions.size(), 0U);
    for (std::size_t assignment = 0U; assignment < total; ++assignment) {
        for (std::size_t variable = 0U; variable < dimensions.size(); ++variable) {
            coordinates[variable] =
                (assignment / global_strides[variable]) % dimensions[variable];
        }
        QComplex value{1.0, 0.0};
        for (const ReferenceFactor& factor : factors) {
            std::size_t factor_index = 0U;
            std::size_t stride = 1U;
            for (const FactorVariableId variable : factor.variables) {
                factor_index += coordinates[variable] * stride;
                stride *= dimensions[variable];
            }
            value *= factor.dense[factor_index];
        }
        const std::size_t output_index = retained.empty()
            ? 0U
            : coordinates[retained.front()];
        output[output_index] += value;
    }
    return output;
}

[[nodiscard]] QComplex value_for(std::size_t seed, std::size_t index) {
    const double re = 0.17 + 0.013 * static_cast<double>((seed + 3U * index) % 19U);
    const double im = 0.002 * static_cast<double>((2U * seed + index) % 7U);
    return {re, im};
}

void fixed_branching_tree() {
    const std::vector<std::size_t> dimensions{2U, 3U, 2U, 2U, 3U};
    ExactFactorGraph graph;
    for (const std::size_t dimension : dimensions) {
        static_cast<void>(graph.add_variable(dimension));
    }

    std::vector<ReferenceFactor> reference;
    const auto add_dense = [&](std::vector<FactorVariableId> variables, std::size_t seed) {
        std::size_t entries = 1U;
        for (const FactorVariableId variable : variables) {
            entries *= dimensions[variable];
        }
        std::vector<QComplex> values(entries);
        for (std::size_t index = 0U; index < entries; ++index) {
            values[index] = value_for(seed, index);
        }
        static_cast<void>(graph.add_dense_factor(variables, values));
        reference.push_back({std::move(variables), std::move(values)});
    };
    const auto add_sparse = [&](std::vector<FactorVariableId> variables, std::size_t seed) {
        std::size_t entries = 1U;
        for (const FactorVariableId variable : variables) {
            entries *= dimensions[variable];
        }
        std::vector<QComplex> dense(entries);
        std::vector<FactorSparseEntry> sparse;
        for (std::size_t index = 0U; index < entries; ++index) {
            if ((index + seed) % 3U != 0U) {
                dense[index] = value_for(seed, index);
                sparse.push_back({index, dense[index]});
            }
        }
        static_cast<void>(graph.add_sparse_factor(variables, sparse));
        reference.push_back({std::move(variables), std::move(dense)});
    };

    add_dense({0U, 1U, 2U}, 1U);
    add_sparse({1U, 3U}, 2U);
    add_dense({2U, 4U}, 3U);
    add_dense({3U}, 4U);
    add_sparse({4U}, 5U);

    ExactFactorTreePlan partition_plan(graph);
    ExactFactorPlan generic_partition(graph);
    const auto expected_partition = brute(dimensions, reference);
    require(partition_plan.route_name() == std::string("ExactFactorTreeTransfer"),
            "tree route name changed");
    require(partition_plan.stats().incidence_edges == 9U,
            "branching tree incidence certificate changed");
    require(partition_plan.output_entries() == 1U,
            "tree partition output size changed");
    require_vector_close(partition_plan.evaluate(), expected_partition,
                         "tree partition differs from brute force");
    require_vector_close(generic_partition.evaluate(), expected_partition,
                         "generic partition differs from brute force");

    const std::vector<FactorVariableId> retained{3U};
    ExactFactorTreePlan marginal_plan(graph, retained);
    ExactFactorPlan generic_marginal(graph, retained);
    const auto expected_marginal = brute(dimensions, reference, retained);
    require_vector_close(marginal_plan.evaluate(), expected_marginal,
                         "tree marginal differs from brute force");
    require_vector_close(generic_marginal.evaluate(), expected_marginal,
                         "generic marginal differs from brute force");
    require_close(marginal_plan.partition(), generic_marginal.partition(),
                  "tree partition from marginal plan differs from generic");
    const auto normalized = marginal_plan.normalized_marginal();
    QComplex normalization{};
    for (const QComplex& value : normalized) {
        normalization += value;
    }
    require_close(normalization, {1.0, 0.0},
                  "tree normalized marginal does not sum to one");

    ExactFactorTreeWorkspace bad_workspace;
    std::vector<QComplex> output(marginal_plan.output_entries());
    bool rejected = false;
    try {
        marginal_plan.evaluate(output, bad_workspace);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "tree workspace mismatch was not rejected");

    const FactorId dense_factor = 0U;
    std::vector<QComplex> replacement = reference[0].dense;
    replacement[1U] = {0.91, -0.03};
    marginal_plan.rebind_dense_factor(dense_factor, replacement);
    generic_marginal.rebind_dense_factor(dense_factor, replacement);
    require_vector_close(marginal_plan.evaluate(), generic_marginal.evaluate(),
                         "tree dense rebind differs from generic");
    require(marginal_plan.rebind_count() == 1U,
            "tree dense rebind count changed");

    const auto before_invalid = marginal_plan.evaluate();
    replacement[2U] = {std::numeric_limits<double>::infinity(), 0.0};
    rejected = false;
    try {
        marginal_plan.rebind_dense_factor(dense_factor, replacement);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "tree non-finite dense rebind was not rejected");
    require_vector_close(marginal_plan.evaluate(), before_invalid,
                         "failed tree dense rebind changed state");

    std::vector<FactorSparseEntry> sparse{{0U, {0.4, 0.0}}, {3U, {0.8, 0.01}}};
    const FactorId sparse_factor = 1U;
    marginal_plan.rebind_sparse_factor(sparse_factor, sparse);
    generic_marginal.rebind_sparse_factor(sparse_factor, sparse);
    require_vector_close(marginal_plan.evaluate(), generic_marginal.evaluate(),
                         "tree sparse rebind differs from generic");
}

void randomized_trees() {
    std::mt19937_64 generator(0x71eeU);
    for (std::size_t trial = 0U; trial < 32U; ++trial) {
        const std::size_t variable_count = 3U + static_cast<std::size_t>(generator() % 5U);
        std::vector<std::size_t> dimensions(variable_count);
        ExactFactorGraph graph;
        for (std::size_t variable = 0U; variable < variable_count; ++variable) {
            dimensions[variable] = 2U + static_cast<std::size_t>(generator() % 3U);
            static_cast<void>(graph.add_variable(dimensions[variable]));
        }

        std::vector<ReferenceFactor> reference;
        for (std::size_t child = 1U; child < variable_count; ++child) {
            const FactorVariableId parent = static_cast<FactorVariableId>(generator() % child);
            const FactorVariableId child_id = static_cast<FactorVariableId>(child);
            const std::vector<FactorVariableId> scope{parent, child_id};
            const std::size_t entries = dimensions[parent] * dimensions[child];
            std::vector<QComplex> dense(entries);
            for (std::size_t index = 0U; index < entries; ++index) {
                dense[index] = value_for(trial + child, index);
            }
            if ((trial + child) % 2U == 0U) {
                static_cast<void>(graph.add_dense_factor(scope, dense));
            } else {
                std::vector<FactorSparseEntry> sparse;
                for (std::size_t index = 0U; index < entries; ++index) {
                    if ((index + child) % 4U != 0U) {
                        sparse.push_back({index, dense[index]});
                    } else {
                        dense[index] = {};
                    }
                }
                static_cast<void>(graph.add_sparse_factor(scope, sparse));
            }
            reference.push_back({scope, dense});
        }

        const auto expected_partition = brute(dimensions, reference);
        ExactFactorTreePlan tree_partition(graph);
        ExactFactorPlan generic_partition(graph);
        require_vector_close(tree_partition.evaluate(), expected_partition,
                             "random tree partition differs from brute force");
        require_vector_close(generic_partition.evaluate(), expected_partition,
                             "random generic partition differs from brute force");

        const FactorVariableId retained =
            static_cast<FactorVariableId>(generator() % variable_count);
        const std::vector<FactorVariableId> retained_variables{retained};
        const auto expected_marginal = brute(dimensions, reference, retained_variables);
        ExactFactorTreePlan tree_marginal(graph, retained_variables);
        ExactFactorPlan generic_marginal(graph, retained_variables);
        require_vector_close(tree_marginal.evaluate(), expected_marginal,
                             "random tree marginal differs from brute force");
        require_vector_close(generic_marginal.evaluate(), expected_marginal,
                             "random generic marginal differs from brute force");
    }
}

void rejection_cases() {
    ExactFactorGraph cycle;
    for (std::size_t index = 0U; index < 3U; ++index) {
        static_cast<void>(cycle.add_variable(2U));
    }
    const std::vector<QComplex> pair(4U, {1.0, 0.0});
    static_cast<void>(cycle.add_dense_factor(std::vector<FactorVariableId>{0U, 1U}, pair));
    static_cast<void>(cycle.add_dense_factor(std::vector<FactorVariableId>{1U, 2U}, pair));
    static_cast<void>(cycle.add_dense_factor(std::vector<FactorVariableId>{2U, 0U}, pair));
    bool rejected = false;
    try {
        ExactFactorTreePlan plan(cycle);
        static_cast<void>(plan);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "cyclic incidence graph was not rejected");

    ExactFactorGraph disconnected;
    for (std::size_t index = 0U; index < 4U; ++index) {
        static_cast<void>(disconnected.add_variable(2U));
    }
    static_cast<void>(disconnected.add_dense_factor(
        std::vector<FactorVariableId>{0U, 1U}, pair));
    static_cast<void>(disconnected.add_dense_factor(
        std::vector<FactorVariableId>{2U, 3U}, pair));
    rejected = false;
    try {
        ExactFactorTreePlan plan(disconnected);
        static_cast<void>(plan);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "disconnected incidence graph was not rejected");

    ExactFactorGraph scalar;
    static_cast<void>(scalar.add_variable(2U));
    static_cast<void>(scalar.add_dense_factor(
        std::vector<FactorVariableId>{}, std::vector<QComplex>{{1.0, 0.0}}));
    rejected = false;
    try {
        ExactFactorTreePlan plan(scalar);
        static_cast<void>(plan);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "scalar factor was not rejected from tree topology");

    ExactFactorGraph simple;
    static_cast<void>(simple.add_variable(2U));
    static_cast<void>(simple.add_variable(2U));
    static_cast<void>(simple.add_dense_factor(
        std::vector<FactorVariableId>{0U, 1U}, pair));
    const std::vector<FactorVariableId> two_retained{0U, 1U};
    rejected = false;
    try {
        ExactFactorTreePlan plan(simple, two_retained);
        static_cast<void>(plan);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "multi-retained tree query was not rejected");

    ExactFactorGraph zero;
    static_cast<void>(zero.add_variable(2U));
    static_cast<void>(zero.add_dense_factor(
        std::vector<FactorVariableId>{0U}, std::vector<QComplex>{{0.0, 0.0}, {0.0, 0.0}}));
    ExactFactorTreePlan zero_plan(zero, std::vector<FactorVariableId>{0U});
    rejected = false;
    try {
        static_cast<void>(zero_plan.normalized_marginal());
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "zero-partition tree normalization was not rejected");
}

}  // namespace

int main() {
    fixed_branching_tree();
    randomized_trees();
    rejection_cases();
    return 0;
}

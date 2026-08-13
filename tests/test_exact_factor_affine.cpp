#include "qubit/qfactor_affine.hpp"
#include "qubit/qfactor_decision.hpp"

#include <algorithm>
#include <array>
#include <bit>
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

using qubit::ExactFactorAffineConfig;
using qubit::ExactFactorAffinePlan;
using qubit::ExactFactorDecisionPlan;
using qubit::ExactFactorGraph;
using qubit::ExactFactorPlan;
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
    require(qubit::almost_equal(actual, expected, 4e-11), message);
}

void compare(
    std::span<const QComplex> actual,
    std::span<const QComplex> expected,
    const std::string& message) {
    require(actual.size() == expected.size(), message + " size");
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        require_close(actual[index], expected[index], message + " value");
    }
}

template <class Function>
void require_reject(Function&& function, const std::string& message) {
    bool rejected = false;
    try {
        function();
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, message);
}

[[nodiscard]] std::vector<QComplex> parity_dense(
    std::size_t width,
    bool rhs,
    QComplex coefficient) {
    const std::size_t entries = std::size_t{1U} << width;
    std::vector<QComplex> dense(entries);
    for (std::size_t index = 0U; index < entries; ++index) {
        if (((std::popcount(index) & 1U) != 0U) == rhs) {
            dense[index] = coefficient;
        }
    }
    return dense;
}

[[nodiscard]] std::vector<FactorSparseEntry> sparse_of(
    std::span<const QComplex> dense,
    bool include_zero = false) {
    std::vector<FactorSparseEntry> sparse;
    for (std::size_t index = 0U; index < dense.size(); ++index) {
        if (dense[index].re != 0.0 || dense[index].im != 0.0 ||
            (include_zero && index == 0U)) {
            sparse.push_back({index, dense[index]});
        }
    }
    return sparse;
}

[[nodiscard]] std::vector<QComplex> brute(
    std::size_t variable_count,
    std::span<const ReferenceFactor> factors,
    std::span<const FactorVariableId> retained = {}) {
    require(retained.size() <= 1U, "brute affine control supports one retained variable");
    const std::size_t assignments = std::size_t{1U} << variable_count;
    std::vector<QComplex> output(retained.empty() ? 1U : 2U);
    for (std::size_t assignment = 0U; assignment < assignments; ++assignment) {
        QComplex value{1.0, 0.0};
        for (const ReferenceFactor& factor : factors) {
            std::size_t index = 0U;
            for (std::size_t position = 0U; position < factor.variables.size(); ++position) {
                index |= ((assignment >> factor.variables[position]) & 1U) << position;
            }
            value *= factor.dense[index];
        }
        const std::size_t output_index = retained.empty()
            ? 0U
            : ((assignment >> retained.front()) & 1U);
        output[output_index] += value;
    }
    return output;
}

void fixed_dense_sparse_complex() {
    ExactFactorGraph graph;
    for (std::size_t variable = 0U; variable < 5U; ++variable) {
        static_cast<void>(graph.add_variable(2U));
    }
    std::vector<ReferenceFactor> reference;

    const auto add_dense = [&](std::vector<FactorVariableId> scope, bool rhs, QComplex coefficient) {
        std::vector<QComplex> values = parity_dense(scope.size(), rhs, coefficient);
        static_cast<void>(graph.add_dense_factor(scope, values));
        reference.push_back({std::move(scope), std::move(values)});
    };
    const auto add_sparse = [&](std::vector<FactorVariableId> scope, bool rhs, QComplex coefficient) {
        std::vector<QComplex> values = parity_dense(scope.size(), rhs, coefficient);
        const std::vector<FactorSparseEntry> entries = sparse_of(values, true);
        static_cast<void>(graph.add_sparse_factor(scope, entries));
        reference.push_back({std::move(scope), std::move(values)});
    };

    add_dense({2U, 0U, 1U}, true, {0.75, 0.05});
    add_sparse({3U, 1U}, false, {0.6, -0.02});
    add_dense({4U}, true, {1.1, 0.03});
    const QComplex scalar{0.5, 0.1};
    static_cast<void>(graph.add_dense_factor(
        std::vector<FactorVariableId>{}, std::vector<QComplex>{scalar}));
    reference.push_back({{}, {scalar}});

    ExactFactorAffinePlan partition(graph);
    ExactFactorPlan generic_partition(graph);
    ExactFactorDecisionPlan decision_partition(graph);
    const std::vector<QComplex> expected_partition = brute(5U, reference);
    require(std::string(partition.route_name()) == "ExactFactorAffineXOR",
            "affine route name changed");
    require(partition.stats().rank == 3U, "fixed affine rank changed");
    require(partition.stats().free_variables == 2U, "fixed affine nullity changed");
    compare(partition.evaluate(), expected_partition,
            "fixed affine partition differs from brute force");
    compare(generic_partition.evaluate(), expected_partition,
            "fixed generic partition differs from brute force");
    compare(decision_partition.evaluate(), expected_partition,
            "fixed decision partition differs from brute force");

    const std::array<FactorVariableId, 1> retained{{4U}};
    ExactFactorAffinePlan marginal(graph, retained);
    ExactFactorPlan generic_marginal(graph, retained);
    ExactFactorDecisionPlan decision_marginal(graph, retained);
    const std::vector<QComplex> expected_marginal = brute(5U, reference, retained);
    compare(marginal.evaluate(), expected_marginal,
            "fixed affine marginal differs from brute force");
    compare(generic_marginal.evaluate(), expected_marginal,
            "fixed generic marginal differs from brute force");
    compare(decision_marginal.evaluate(), expected_marginal,
            "fixed decision marginal differs from brute force");
    require_close(marginal.partition(), partition.partition(),
                  "affine marginal partition changed");
    const std::vector<QComplex> normalized = marginal.normalized_marginal();
    require_close(normalized[0] + normalized[1], {1.0, 0.0},
                  "affine normalized marginal does not sum to one");
}

void free_and_fixed_coordinates() {
    ExactFactorGraph free_graph;
    for (std::size_t variable = 0U; variable < 3U; ++variable) {
        static_cast<void>(free_graph.add_variable(2U));
    }
    const std::vector<QComplex> equality = parity_dense(2U, false, {1.0, 0.0});
    static_cast<void>(free_graph.add_dense_factor(
        std::array<FactorVariableId, 2>{{0U, 1U}}, equality));
    const std::array<FactorVariableId, 1> free_retained{{2U}};
    ExactFactorAffinePlan free_plan(free_graph, free_retained);
    const std::vector<QComplex> free_values = free_plan.evaluate();
    require_close(free_values[0], {2.0, 0.0}, "free affine zero bin changed");
    require_close(free_values[1], {2.0, 0.0}, "free affine one bin changed");
    compare(free_values, ExactFactorPlan(free_graph, free_retained).evaluate(),
            "free affine differs from generic VE");
    compare(free_values, ExactFactorDecisionPlan(free_graph, free_retained).evaluate(),
            "free affine differs from DecisionDiagram");

    ExactFactorGraph fixed_graph;
    static_cast<void>(fixed_graph.add_variable(2U));
    static_cast<void>(fixed_graph.add_variable(2U));
    const std::vector<QComplex> one = parity_dense(1U, true, {0.8, 0.0});
    static_cast<void>(fixed_graph.add_dense_factor(
        std::array<FactorVariableId, 1>{{0U}}, one));
    const std::array<FactorVariableId, 1> fixed_retained{{0U}};
    ExactFactorAffinePlan fixed_plan(fixed_graph, fixed_retained);
    const std::vector<QComplex> fixed_values = fixed_plan.evaluate();
    require_close(fixed_values[0], {}, "fixed affine wrong zero bin");
    require_close(fixed_values[1], {1.6, 0.0}, "fixed affine wrong one bin");
}

void inconsistent_and_zero() {
    ExactFactorGraph inconsistent;
    static_cast<void>(inconsistent.add_variable(2U));
    static_cast<void>(inconsistent.add_dense_factor(
        std::array<FactorVariableId, 1>{{0U}}, parity_dense(1U, false, {1.0, 0.0})));
    static_cast<void>(inconsistent.add_dense_factor(
        std::array<FactorVariableId, 1>{{0U}}, parity_dense(1U, true, {1.0, 0.0})));
    const std::array<FactorVariableId, 1> retained{{0U}};
    ExactFactorAffinePlan inconsistent_plan(inconsistent, retained);
    require(!inconsistent_plan.consistent(), "inconsistent affine system reported consistent");
    compare(inconsistent_plan.evaluate(), ExactFactorPlan(inconsistent, retained).evaluate(),
            "inconsistent affine differs from generic VE");
    compare(inconsistent_plan.evaluate(), ExactFactorDecisionPlan(inconsistent, retained).evaluate(),
            "inconsistent affine differs from DecisionDiagram");
    require_close(inconsistent_plan.partition(), {}, "inconsistent affine partition is nonzero");
    require_reject([&] {
        static_cast<void>(inconsistent_plan.normalized_marginal());
    }, "inconsistent affine normalization was not rejected");

    ExactFactorGraph zero_graph;
    static_cast<void>(zero_graph.add_variable(2U));
    static_cast<void>(zero_graph.add_variable(2U));
    static_cast<void>(zero_graph.add_dense_factor(
        std::array<FactorVariableId, 2>{{0U, 1U}},
        std::vector<QComplex>(4U)));
    static_cast<void>(zero_graph.add_dense_factor(
        std::array<FactorVariableId, 2>{{1U, 0U}},
        parity_dense(2U, true, {1.0, 0.0})));
    ExactFactorAffinePlan zero_plan(zero_graph, retained);
    require(zero_plan.stats().zero_factors == 1U, "affine zero factor count changed");
    compare(zero_plan.evaluate(), ExactFactorPlan(zero_graph, retained).evaluate(),
            "zero affine differs from generic VE");
    compare(zero_plan.evaluate(), ExactFactorDecisionPlan(zero_graph, retained).evaluate(),
            "zero affine differs from DecisionDiagram");
}

void randomized_small_systems() {
    std::mt19937_64 generator(0xaff1e123ULL);
    for (std::size_t trial = 0U; trial < 40U; ++trial) {
        const std::size_t variables = 4U + static_cast<std::size_t>(generator() % 5U);
        ExactFactorGraph graph;
        for (std::size_t variable = 0U; variable < variables; ++variable) {
            static_cast<void>(graph.add_variable(2U));
        }
        std::vector<std::uint8_t> witness(variables);
        for (std::uint8_t& bit : witness) {
            bit = static_cast<std::uint8_t>(generator() & 1U);
        }
        std::vector<ReferenceFactor> reference;

        const auto add_constraint = [&](std::vector<FactorVariableId> scope, bool sparse_storage) {
            bool rhs = false;
            for (const FactorVariableId variable : scope) {
                rhs = rhs != (witness[variable] != 0U);
            }
            std::shuffle(scope.begin(), scope.end(), generator);
            const QComplex coefficient{
                0.4 + 0.03 * static_cast<double>(generator() % 7U),
                0.002 * static_cast<double>(generator() % 5U),
            };
            std::vector<QComplex> values = parity_dense(scope.size(), rhs, coefficient);
            if (sparse_storage) {
                static_cast<void>(graph.add_sparse_factor(scope, sparse_of(values)));
            } else {
                static_cast<void>(graph.add_dense_factor(scope, values));
            }
            reference.push_back({std::move(scope), std::move(values)});
        };

        for (std::size_t child = 2U; child < variables; ++child) {
            add_constraint({
                static_cast<FactorVariableId>(child - 2U),
                static_cast<FactorVariableId>(child - 1U),
                static_cast<FactorVariableId>(child),
            }, ((trial + child) & 1U) != 0U);
        }
        const std::size_t extras = 2U + static_cast<std::size_t>(generator() % 4U);
        for (std::size_t factor = 0U; factor < extras; ++factor) {
            const std::size_t width = 1U + static_cast<std::size_t>(generator() % std::min<std::size_t>(4U, variables));
            std::vector<FactorVariableId> scope;
            while (scope.size() < width) {
                const FactorVariableId variable = static_cast<FactorVariableId>(generator() % variables);
                if (std::find(scope.begin(), scope.end(), variable) == scope.end()) {
                    scope.push_back(variable);
                }
            }
            add_constraint(std::move(scope), ((trial + factor) & 1U) == 0U);
        }
        if (trial % 4U == 0U) {
            const QComplex scalar{0.75, 0.01};
            static_cast<void>(graph.add_dense_factor(
                std::vector<FactorVariableId>{}, std::vector<QComplex>{scalar}));
            reference.push_back({{}, {scalar}});
        }

        const std::vector<QComplex> expected_partition = brute(variables, reference);
        ExactFactorAffinePlan affine_partition(graph);
        compare(affine_partition.evaluate(), expected_partition,
                "random affine partition differs from brute force");
        compare(ExactFactorPlan(graph).evaluate(), expected_partition,
                "random generic partition differs from brute force");
        compare(ExactFactorDecisionPlan(graph).evaluate(), expected_partition,
                "random Decision partition differs from brute force");

        const std::array<FactorVariableId, 1> retained{{
            static_cast<FactorVariableId>(generator() % variables),
        }};
        const std::vector<QComplex> expected_marginal = brute(variables, reference, retained);
        ExactFactorAffinePlan affine_marginal(graph, retained);
        compare(affine_marginal.evaluate(), expected_marginal,
                "random affine marginal differs from brute force");
        compare(ExactFactorPlan(graph, retained).evaluate(), expected_marginal,
                "random generic marginal differs from brute force");
        compare(ExactFactorDecisionPlan(graph, retained).evaluate(), expected_marginal,
                "random Decision marginal differs from brute force");
    }
}

void rejection_and_resource_cases() {
    ExactFactorGraph nonbinary;
    static_cast<void>(nonbinary.add_variable(3U));
    static_cast<void>(nonbinary.add_dense_factor(
        std::array<FactorVariableId, 1>{{0U}},
        std::array<QComplex, 3>{{QComplex{1.0}, QComplex{}, QComplex{}}}));
    require_reject([&] {
        ExactFactorAffinePlan plan(nonbinary);
        static_cast<void>(plan);
    }, "nonbinary affine graph was not rejected");

    ExactFactorGraph simple;
    static_cast<void>(simple.add_variable(2U));
    static_cast<void>(simple.add_variable(2U));
    static_cast<void>(simple.add_dense_factor(
        std::array<FactorVariableId, 2>{{0U, 1U}},
        parity_dense(2U, false, {1.0, 0.0})));
    const std::array<FactorVariableId, 2> two_retained{{0U, 1U}};
    require_reject([&] {
        ExactFactorAffinePlan plan(simple, two_retained);
        static_cast<void>(plan);
    }, "multi-retained affine query was not rejected");

    ExactFactorGraph weighted;
    static_cast<void>(weighted.add_variable(2U));
    static_cast<void>(weighted.add_dense_factor(
        std::array<FactorVariableId, 1>{{0U}},
        std::array<QComplex, 2>{{QComplex{1.0}, QComplex{2.0}}}));
    require_reject([&] {
        ExactFactorAffinePlan plan(weighted);
        static_cast<void>(plan);
    }, "non-affine weighted factor was not rejected");

    ExactFactorGraph uneven;
    static_cast<void>(uneven.add_variable(2U));
    static_cast<void>(uneven.add_variable(2U));
    static_cast<void>(uneven.add_dense_factor(
        std::array<FactorVariableId, 2>{{0U, 1U}},
        std::array<QComplex, 4>{{QComplex{1.0}, QComplex{}, QComplex{}, QComplex{2.0}}}));
    require_reject([&] {
        ExactFactorAffinePlan plan(uneven);
        static_cast<void>(plan);
    }, "nonuniform parity coefficient was not rejected");

    ExactFactorGraph incomplete;
    static_cast<void>(incomplete.add_variable(2U));
    static_cast<void>(incomplete.add_variable(2U));
    const std::array<FactorSparseEntry, 1> one_entry{{
        FactorSparseEntry{0U, QComplex{1.0}},
    }};
    static_cast<void>(incomplete.add_sparse_factor(
        std::array<FactorVariableId, 2>{{0U, 1U}}, one_entry));
    require_reject([&] {
        ExactFactorAffinePlan plan(incomplete);
        static_cast<void>(plan);
    }, "incomplete parity support was not rejected");

    ExactFactorGraph caps;
    for (std::size_t variable = 0U; variable < 4U; ++variable) {
        static_cast<void>(caps.add_variable(2U));
    }
    static_cast<void>(caps.add_dense_factor(
        std::array<FactorVariableId, 2>{{0U, 1U}},
        parity_dense(2U, false, {1.0, 0.0})));
    static_cast<void>(caps.add_dense_factor(
        std::array<FactorVariableId, 2>{{2U, 3U}},
        parity_dense(2U, true, {1.0, 0.0})));

    ExactFactorAffineConfig equation_cap;
    equation_cap.max_equations = 1U;
    require_reject([&] {
        ExactFactorAffinePlan plan(caps, {}, equation_cap);
        static_cast<void>(plan);
    }, "affine equation cap was not enforced");

    ExactFactorAffineConfig variable_cap;
    variable_cap.max_variables = 3U;
    require_reject([&] {
        ExactFactorAffinePlan plan(caps, {}, variable_cap);
        static_cast<void>(plan);
    }, "affine variable cap was not enforced");

    ExactFactorGraph wide;
    for (std::size_t variable = 0U; variable < 3U; ++variable) {
        static_cast<void>(wide.add_variable(2U));
    }
    static_cast<void>(wide.add_dense_factor(
        std::array<FactorVariableId, 3>{{0U, 1U, 2U}},
        parity_dense(3U, false, {1.0, 0.0})));
    ExactFactorAffineConfig row_cap;
    row_cap.max_row_terms = 2U;
    require_reject([&] {
        ExactFactorAffinePlan plan(wide, {}, row_cap);
        static_cast<void>(plan);
    }, "affine row cap was not enforced");

    ExactFactorAffineConfig basis_cap;
    basis_cap.max_basis_terms = 3U;
    require_reject([&] {
        ExactFactorAffinePlan plan(caps, {}, basis_cap);
        static_cast<void>(plan);
    }, "affine basis term cap was not enforced");
}

}  // namespace

int main() {
    fixed_dense_sparse_complex();
    free_and_fixed_coordinates();
    inconsistent_and_zero();
    randomized_small_systems();
    rejection_and_resource_cases();
    return 0;
}

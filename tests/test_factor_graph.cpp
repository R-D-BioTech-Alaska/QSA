#include "qubit/qfactor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using qubit::ExactFactorConfig;
using qubit::ExactFactorGraph;
using qubit::ExactFactorPlan;
using qubit::FactorId;
using qubit::FactorSparseEntry;
using qubit::FactorVariableId;
using qubit::QComplex;
using qubit::QStateError;

struct ReferenceFactor {
    std::vector<FactorVariableId> variables{};
    std::vector<QComplex> values{};
};

struct ReferenceGraph {
    std::vector<std::size_t> dimensions{};
    std::vector<ReferenceFactor> factors{};
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    const QComplex& actual,
    const QComplex& expected,
    const std::string& message,
    double tolerance = 2e-11) {
    require(qubit::almost_equal(actual, expected, tolerance), message);
}

std::size_t entries(
    std::span<const FactorVariableId> variables,
    std::span<const std::size_t> dimensions) {
    std::size_t count = 1U;
    for (const FactorVariableId variable : variables) {
        count *= dimensions[variable];
    }
    return count;
}

std::vector<std::size_t> decode(
    std::size_t index,
    std::span<const std::size_t> dimensions) {
    std::vector<std::size_t> coordinates(dimensions.size(), 0U);
    for (std::size_t variable = 0U; variable < dimensions.size(); ++variable) {
        coordinates[variable] = index % dimensions[variable];
        index /= dimensions[variable];
    }
    return coordinates;
}

std::size_t factor_index(
    const ReferenceFactor& factor,
    std::span<const std::size_t> coordinates,
    std::span<const std::size_t> dimensions) {
    std::size_t index = 0U;
    std::size_t stride = 1U;
    for (const FactorVariableId variable : factor.variables) {
        index += coordinates[variable] * stride;
        stride *= dimensions[variable];
    }
    return index;
}

QComplex reference_weight(
    const ReferenceGraph& graph,
    std::span<const std::size_t> coordinates) {
    QComplex result{1.0, 0.0};
    for (const ReferenceFactor& factor : graph.factors) {
        result *= factor.values[factor_index(factor, coordinates, graph.dimensions)];
    }
    return result;
}

std::vector<QComplex> reference_marginal(
    const ReferenceGraph& graph,
    std::span<const FactorVariableId> retained) {
    std::size_t output_entries = 1U;
    for (const FactorVariableId variable : retained) {
        output_entries *= graph.dimensions[variable];
    }
    std::vector<QComplex> result(output_entries);

    std::size_t global_entries = 1U;
    for (const std::size_t dimension : graph.dimensions) {
        global_entries *= dimension;
    }
    for (std::size_t assignment = 0U; assignment < global_entries; ++assignment) {
        const std::vector<std::size_t> coordinates = decode(assignment, graph.dimensions);
        std::size_t output = 0U;
        std::size_t stride = 1U;
        for (const FactorVariableId variable : retained) {
            output += coordinates[variable] * stride;
            stride *= graph.dimensions[variable];
        }
        result[output] += reference_weight(graph, coordinates);
    }
    return result;
}

QComplex sum(std::span<const QComplex> values) {
    QComplex result{};
    for (const QComplex& value : values) {
        result += value;
    }
    return result;
}

void compare(
    const std::vector<QComplex>& actual,
    const std::vector<QComplex>& expected,
    const std::string& message) {
    require(actual.size() == expected.size(), message + " size");
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        require_close(actual[index], expected[index], message + " value");
    }
}

bool differs(
    const std::vector<QComplex>& first,
    const std::vector<QComplex>& second) {
    if (first.size() != second.size()) {
        return true;
    }
    for (std::size_t index = 0U; index < first.size(); ++index) {
        if (!qubit::almost_equal(first[index], second[index], 1e-13)) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main() {
    {
        ExactFactorGraph graph;
        const FactorVariableId a = graph.add_variable(2U);
        const FactorVariableId b = graph.add_variable(2U);
        const std::array<FactorVariableId, 2> scope{{a, b}};
        const std::array<QComplex, 4> equality{{
            QComplex{1.0}, QComplex{}, QComplex{}, QComplex{1.0},
        }};
        static_cast<void>(graph.add_dense_factor(scope, equality));
        require_close(graph.partition(), QComplex{2.0}, "binary partition is wrong");
        const std::array<FactorVariableId, 1> retained{{a}};
        const auto marginal = graph.marginal(retained);
        require(marginal.size() == 2U, "binary marginal size is wrong");
        require_close(marginal[0], QComplex{1.0}, "binary marginal zero is wrong");
        require_close(marginal[1], QComplex{1.0}, "binary marginal one is wrong");
        const auto normalized = graph.normalized_marginal(retained);
        require_close(normalized[0], QComplex{0.5}, "binary normalized zero is wrong");
        require_close(normalized[1], QComplex{0.5}, "binary normalized one is wrong");
    }

    {
        ExactFactorGraph graph;
        const FactorVariableId a = graph.add_variable(2U);
        const FactorVariableId b = graph.add_variable(3U);
        const FactorVariableId c = graph.add_variable(4U);
        const std::array<FactorVariableId, 2> ab{{a, b}};
        const std::array<QComplex, 6> ab_values{{
            QComplex{1.0}, QComplex{2.0}, QComplex{3.0},
            QComplex{4.0}, QComplex{5.0}, QComplex{6.0},
        }};
        static_cast<void>(graph.add_dense_factor(ab, ab_values));
        const std::array<FactorVariableId, 2> bc{{b, c}};
        const std::array<FactorSparseEntry, 5> bc_values{{
            FactorSparseEntry{0U, QComplex{1.0}},
            FactorSparseEntry{2U, QComplex{2.0}},
            FactorSparseEntry{5U, QComplex{3.0}},
            FactorSparseEntry{7U, QComplex{4.0}},
            FactorSparseEntry{11U, QComplex{5.0}},
        }};
        static_cast<void>(graph.add_sparse_factor(bc, bc_values));
        const std::array<FactorVariableId, 1> c_scope{{c}};
        const std::array<QComplex, 4> c_values{{
            QComplex{0.5}, QComplex{1.0}, QComplex{1.5}, QComplex{2.0},
        }};
        static_cast<void>(graph.add_dense_factor(c_scope, c_values));

        ReferenceGraph reference;
        reference.dimensions = {2U, 3U, 4U};
        reference.factors.push_back({std::vector<FactorVariableId>(ab.begin(), ab.end()),
                                     std::vector<QComplex>(ab_values.begin(), ab_values.end())});
        std::vector<QComplex> bc_dense(12U);
        for (const FactorSparseEntry& entry : bc_values) {
            bc_dense[entry.index] = entry.value;
        }
        reference.factors.push_back({std::vector<FactorVariableId>(bc.begin(), bc.end()), bc_dense});
        reference.factors.push_back({std::vector<FactorVariableId>(c_scope.begin(), c_scope.end()),
                                     std::vector<QComplex>(c_values.begin(), c_values.end())});

        const std::array<FactorVariableId, 2> retained{{c, a}};
        compare(graph.marginal(retained), reference_marginal(reference, retained),
                "mixed-dimension marginal");
        const std::span<const FactorVariableId> none{};
        require_close(graph.partition(), sum(reference_marginal(reference, none)),
                      "mixed-dimension partition is wrong");
        const auto normalized = graph.normalized_marginal(retained);
        require_close(sum(normalized), QComplex{1.0},
                      "mixed-dimension normalized marginal does not sum to one");
    }

    {
        ExactFactorGraph graph;
        const FactorVariableId isolated = graph.add_variable(3U);
        const std::array<QComplex, 1> constant{{QComplex{2.0, 0.5}}};
        static_cast<void>(graph.add_dense_factor({}, constant));
        require_close(graph.partition(), QComplex{6.0, 1.5},
                      "isolated variable did not contribute its dimension");
        const std::array<FactorVariableId, 1> retained{{isolated}};
        const auto marginal = graph.marginal(retained);
        require(marginal.size() == 3U, "isolated marginal has wrong size");
        for (const QComplex& value : marginal) {
            require_close(value, constant[0], "isolated marginal is not constant");
        }
        const auto normalized = graph.normalized_marginal(retained);
        for (const QComplex& value : normalized) {
            require_close(value, QComplex{1.0 / 3.0}, "isolated normalized marginal is not uniform");
        }
    }

    {
        ExactFactorGraph graph;
        const FactorVariableId a = graph.add_variable(2U);
        const std::array<FactorVariableId, 1> scope{{a}};
        const std::array<QComplex, 2> zero{{QComplex{}, QComplex{}}};
        static_cast<void>(graph.add_dense_factor(scope, zero));
        bool rejected = false;
        try {
            static_cast<void>(graph.normalized_marginal(scope));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "zero partition normalized successfully");
    }

    {
        ExactFactorConfig config;
        config.max_factor_entries = 8U;
        ExactFactorGraph graph(config);
        const FactorVariableId a = graph.add_variable(2U);
        const FactorVariableId b = graph.add_variable(2U);
        const FactorVariableId c = graph.add_variable(2U);
        const std::array<QComplex, 4> ones{{QComplex{1.0}, QComplex{1.0}, QComplex{1.0}, QComplex{1.0}}};
        const std::array<FactorVariableId, 2> ab{{a, b}};
        const std::array<FactorVariableId, 2> bc{{b, c}};
        const std::array<FactorVariableId, 2> ac{{a, c}};
        static_cast<void>(graph.add_dense_factor(ab, ones));
        static_cast<void>(graph.add_dense_factor(bc, ones));
        static_cast<void>(graph.add_dense_factor(ac, ones));
        const ExactFactorPlan plan = graph.compile();
        require(plan.stats().peak_factor_entries == 8U,
                "exact factor boundary did not reach eight entries");

        ExactFactorConfig tight = config;
        tight.max_factor_entries = 4U;
        ExactFactorGraph rejected_graph(tight);
        const FactorVariableId ta = rejected_graph.add_variable(2U);
        const FactorVariableId tb = rejected_graph.add_variable(2U);
        const FactorVariableId tc = rejected_graph.add_variable(2U);
        const std::array<FactorVariableId, 2> tab{{ta, tb}};
        const std::array<FactorVariableId, 2> tbc{{tb, tc}};
        const std::array<FactorVariableId, 2> tac{{ta, tc}};
        static_cast<void>(rejected_graph.add_dense_factor(tab, ones));
        static_cast<void>(rejected_graph.add_dense_factor(tbc, ones));
        static_cast<void>(rejected_graph.add_dense_factor(tac, ones));
        bool rejected = false;
        try {
            static_cast<void>(rejected_graph.compile());
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "exact factor elimination ignored max_factor_entries");
        require(rejected_graph.validate(), "failed factor compilation damaged its graph");
    }

    {
        ExactFactorGraph graph;
        const FactorVariableId a = graph.add_variable(2U);
        const FactorVariableId b = graph.add_variable(3U);
        const std::array<FactorVariableId, 2> scope{{a, b}};
        const std::array<QComplex, 6> first{{
            QComplex{1.0}, QComplex{2.0}, QComplex{3.0},
            QComplex{4.0}, QComplex{5.0}, QComplex{6.0},
        }};
        const FactorId factor = graph.add_dense_factor(scope, first);
        const std::array<FactorVariableId, 1> retained{{b}};
        ExactFactorPlan plan = graph.compile(retained);
        auto workspace = plan.workspace();
        const auto before = plan.evaluate(workspace);

        const std::array<FactorSparseEntry, 3> second{{
            FactorSparseEntry{0U, QComplex{2.0}},
            FactorSparseEntry{3U, QComplex{7.0}},
            FactorSparseEntry{5U, QComplex{11.0}},
        }};
        graph.set_sparse_factor(factor, second);
        compare(plan.evaluate(workspace), before,
                "compiled factor plan did not snapshot source values");
        plan.rebind(graph);
        require(plan.rebind_count() == 1U, "factor rebind count is wrong");
        require(plan.stats().source_sparse_factors == 1U &&
                    plan.stats().source_dense_factors == 0U,
                "factor rebind storage stats are wrong");
        const auto after = plan.evaluate(workspace);
        require(differs(after, before), "factor rebind did not change the evaluation");
        compare(after, graph.marginal(retained), "factor sparse rebind");

        const std::array<QComplex, 6> third{{
            QComplex{6.0}, QComplex{5.0}, QComplex{4.0},
            QComplex{3.0}, QComplex{2.0}, QComplex{1.0},
        }};
        graph.set_dense_factor(factor, third);
        plan.rebind(graph);
        require(plan.rebind_count() == 2U &&
                    plan.stats().source_dense_factors == 1U &&
                    plan.stats().source_sparse_factors == 0U,
                "factor dense rebind stats are wrong");
        const auto dense_after = plan.evaluate(workspace);
        compare(dense_after, graph.marginal(retained), "factor dense rebind");

        ExactFactorGraph mismatch;
        const FactorVariableId ma = mismatch.add_variable(2U);
        const FactorVariableId mb = mismatch.add_variable(3U);
        const std::array<FactorVariableId, 1> changed_scope{{ma}};
        const std::array<QComplex, 2> changed_values{{QComplex{1.0}, QComplex{1.0}}};
        static_cast<void>(mismatch.add_dense_factor(changed_scope, changed_values));
        static_cast<void>(mb);
        bool rejected = false;
        try {
            plan.rebind(mismatch);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "factor rebind accepted changed topology");
        compare(plan.evaluate(workspace), dense_after,
                "failed factor rebind was not transactional");
    }

    {
        ExactFactorGraph first;
        const FactorVariableId a = first.add_variable(2U);
        const FactorVariableId b = first.add_variable(2U);
        const std::array<FactorVariableId, 2> scope{{a, b}};
        const std::array<QComplex, 4> values{{QComplex{1.0}, QComplex{2.0}, QComplex{3.0}, QComplex{4.0}}};
        static_cast<void>(first.add_dense_factor(scope, values));
        const ExactFactorPlan first_plan = first.compile();

        ExactFactorGraph second;
        const FactorVariableId c = second.add_variable(3U);
        const std::array<FactorVariableId, 1> c_scope{{c}};
        const std::array<QComplex, 3> c_values{{QComplex{1.0}, QComplex{1.0}, QComplex{1.0}}};
        static_cast<void>(second.add_dense_factor(c_scope, c_values));
        const ExactFactorPlan second_plan = second.compile(c_scope);
        auto wrong_workspace = second_plan.workspace();
        std::array<QComplex, 1> output{};
        bool rejected = false;
        try {
            first_plan.evaluate(output, wrong_workspace);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "factor plan accepted a workspace from another plan");
    }

    {
        ExactFactorGraph graph;
        bool rejected = false;
        try {
            static_cast<void>(graph.add_variable(0U));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "zero-dimension factor variable was accepted");

        const FactorVariableId a = graph.add_variable(2U);
        const std::array<FactorVariableId, 2> duplicate{{a, a}};
        const std::array<QComplex, 4> values{{QComplex{1.0}, QComplex{1.0}, QComplex{1.0}, QComplex{1.0}}};
        rejected = false;
        try {
            static_cast<void>(graph.add_dense_factor(duplicate, values));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "duplicate factor scope variable was accepted");

        const std::array<FactorVariableId, 1> scope{{a}};
        rejected = false;
        try {
            static_cast<void>(graph.add_dense_factor(scope, values));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "wrong dense factor size was accepted");

        const std::array<FactorSparseEntry, 2> duplicate_sparse{{
            FactorSparseEntry{0U, QComplex{1.0}},
            FactorSparseEntry{0U, QComplex{2.0}},
        }};
        rejected = false;
        try {
            static_cast<void>(graph.add_sparse_factor(scope, duplicate_sparse));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "duplicate sparse factor index was accepted");

        const std::array<FactorSparseEntry, 1> out_of_range{{
            FactorSparseEntry{2U, QComplex{1.0}},
        }};
        rejected = false;
        try {
            static_cast<void>(graph.add_sparse_factor(scope, out_of_range));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "out-of-range sparse factor index was accepted");

        const std::array<QComplex, 2> nonfinite{{
            QComplex{1.0}, QComplex{std::numeric_limits<double>::infinity()},
        }};
        rejected = false;
        try {
            static_cast<void>(graph.add_dense_factor(scope, nonfinite));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "non-finite factor value was accepted");
    }

    {
        std::mt19937_64 generator(0x464143544F525341ULL);
        for (std::size_t test_case = 0U; test_case < 48U; ++test_case) {
            ExactFactorConfig config;
            config.max_factor_entries = 4096U;
            ExactFactorGraph graph(config);
            ReferenceGraph reference;
            constexpr std::size_t variable_count = 5U;
            for (std::size_t variable = 0U; variable < variable_count; ++variable) {
                const std::size_t dimension = 2U + static_cast<std::size_t>(generator() % 3U);
                static_cast<void>(graph.add_variable(dimension));
                reference.dimensions.push_back(dimension);
            }

            for (std::size_t factor_number = 0U; factor_number < 7U; ++factor_number) {
                std::array<FactorVariableId, variable_count> permutation{{0U, 1U, 2U, 3U, 4U}};
                std::shuffle(permutation.begin(), permutation.end(), generator);
                const std::size_t rank = 1U + static_cast<std::size_t>(generator() % 3U);
                std::vector<FactorVariableId> scope(permutation.begin(), permutation.begin() + rank);
                const std::size_t count = entries(scope, reference.dimensions);
                std::vector<QComplex> dense(count);
                std::vector<FactorSparseEntry> sparse;
                for (std::size_t index = 0U; index < count; ++index) {
                    if ((generator() % 5U) == 0U) {
                        dense[index] = {};
                        continue;
                    }
                    const double value = 0.05 +
                        static_cast<double>(1U + generator() % 200U) / 100.0;
                    dense[index] = {value, 0.0};
                    sparse.push_back({index, dense[index]});
                }
                if ((generator() & 1U) == 0U) {
                    static_cast<void>(graph.add_dense_factor(scope, dense));
                } else {
                    static_cast<void>(graph.add_sparse_factor(scope, sparse));
                }
                reference.factors.push_back({scope, dense});
            }

            std::array<FactorVariableId, variable_count> retained_order{{0U, 1U, 2U, 3U, 4U}};
            std::shuffle(retained_order.begin(), retained_order.end(), generator);
            const std::size_t retained_count = 1U + static_cast<std::size_t>(generator() % 3U);
            const std::span<const FactorVariableId> retained(
                retained_order.data(), retained_count);

            const ExactFactorPlan first = graph.compile(retained);
            const ExactFactorPlan second = graph.compile(retained);
            require(first.step_count() == second.step_count() &&
                        first.stats().peak_union_variables == second.stats().peak_union_variables &&
                        first.stats().peak_factor_entries == second.stats().peak_factor_entries &&
                        first.stats().output_entries == second.stats().output_entries,
                    "factor compilation is not deterministic");
            const auto expected = reference_marginal(reference, retained);
            compare(first.evaluate(), expected, "random factor marginal");
            const QComplex expected_partition = sum(expected);
            require_close(first.partition(), expected_partition,
                          "random factor partition is wrong");
            if (expected_partition.norm2() > std::numeric_limits<double>::min()) {
                const auto normalized = first.normalized_marginal();
                require_close(sum(normalized), QComplex{1.0},
                              "random normalized marginal does not sum to one");
            } else {
                bool rejected = false;
                try {
                    static_cast<void>(first.normalized_marginal());
                } catch (const QStateError&) {
                    rejected = true;
                }
                require(rejected, "zero random partition normalized successfully");
            }
        }
    }

    return 0;
}

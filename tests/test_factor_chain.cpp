#include "qubit/qfactor_chain.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using qubit::ExactFactorChainPlan;
using qubit::ExactFactorConfig;
using qubit::ExactFactorGraph;
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
    double tolerance = 3e-11) {
    require(qubit::almost_equal(actual, expected, tolerance), message);
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

QComplex sum(std::span<const QComplex> values) {
    QComplex result{};
    for (const QComplex& value : values) {
        result += value;
    }
    return result;
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

std::vector<QComplex> brute_marginal(
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
        QComplex weight{1.0, 0.0};
        for (const ReferenceFactor& factor : graph.factors) {
            weight *= factor.values[factor_index(factor, coordinates, graph.dimensions)];
        }
        std::size_t output = 0U;
        std::size_t stride = 1U;
        for (const FactorVariableId variable : retained) {
            output += coordinates[variable] * stride;
            stride *= graph.dimensions[variable];
        }
        result[output] += weight;
    }
    return result;
}

std::vector<FactorVariableId> final_separator(
    std::size_t factor_count,
    std::size_t width) {
    std::vector<FactorVariableId> retained(width - 1U);
    for (std::size_t index = 0U; index < retained.size(); ++index) {
        retained[index] = static_cast<FactorVariableId>(factor_count + index);
    }
    return retained;
}

std::vector<QComplex> dense_values(
    std::size_t entries,
    std::mt19937_64& rng,
    bool sparse_shape) {
    std::uniform_real_distribution<double> value(0.05, 1.25);
    std::vector<QComplex> result(entries);
    for (std::size_t index = 0U; index < entries; ++index) {
        if (sparse_shape && (rng() % 4U) != 0U) {
            continue;
        }
        result[index] = QComplex{value(rng), 0.02 * value(rng)};
    }
    if (std::none_of(result.begin(), result.end(), [](const QComplex& item) {
            return item.re != 0.0 || item.im != 0.0;
        })) {
        result[0] = QComplex{1.0, 0.0};
    }
    return result;
}

std::vector<FactorSparseEntry> sparse_entries(std::span<const QComplex> dense) {
    std::vector<FactorSparseEntry> entries;
    for (std::size_t index = 0U; index < dense.size(); ++index) {
        if (dense[index].re != 0.0 || dense[index].im != 0.0) {
            entries.push_back({index, dense[index]});
        }
    }
    return entries;
}

ReferenceGraph build_chain(
    ExactFactorGraph& graph,
    std::span<const std::size_t> dimensions,
    std::size_t width,
    std::mt19937_64& rng,
    bool mix_sparse) {
    ReferenceGraph reference;
    reference.dimensions.assign(dimensions.begin(), dimensions.end());
    for (const std::size_t dimension : dimensions) {
        static_cast<void>(graph.add_variable(dimension));
    }
    const std::size_t factor_count = dimensions.size() - width + 1U;
    for (std::size_t factor = 0U; factor < factor_count; ++factor) {
        std::vector<FactorVariableId> scope(width);
        std::size_t table_entries = 1U;
        for (std::size_t position = 0U; position < width; ++position) {
            scope[position] = static_cast<FactorVariableId>(factor + position);
            table_entries *= dimensions[factor + position];
        }
        const bool sparse = mix_sparse && (factor % 2U == 1U);
        std::vector<QComplex> values = dense_values(table_entries, rng, sparse);
        if (sparse) {
            const std::vector<FactorSparseEntry> entries = sparse_entries(values);
            static_cast<void>(graph.add_sparse_factor(scope, entries));
        } else {
            static_cast<void>(graph.add_dense_factor(scope, values));
        }
        reference.factors.push_back({scope, std::move(values)});
    }
    return reference;
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

}  // namespace

int main() {
    std::mt19937_64 rng(0x5a17c0deULL);

    for (const auto& dimensions : {
             std::vector<std::size_t>{2U, 3U, 4U},
             std::vector<std::size_t>{2U, 3U, 4U, 2U},
             std::vector<std::size_t>{2U, 2U, 3U, 2U, 2U}}) {
        const std::size_t width = dimensions.size() == 3U ? 2U : dimensions.size() - 2U;
        ExactFactorGraph graph;
        const ReferenceGraph reference = build_chain(graph, dimensions, width, rng, true);
        const std::size_t factor_count = dimensions.size() - width + 1U;
        const std::vector<FactorVariableId> retained = final_separator(factor_count, width);
        ExactFactorChainPlan chain(graph, retained);
        auto workspace = chain.workspace();
        const std::vector<QComplex> expected = brute_marginal(reference, retained);
        const std::vector<QComplex> actual = chain.evaluate(workspace);
        compare(actual, expected, "deterministic chain brute force");
        compare(actual, graph.marginal(retained), "deterministic chain generic VE");
        require_close(chain.partition(workspace), sum(expected), "deterministic chain partition");
        const std::vector<QComplex> normalized = chain.normalized_marginal(workspace);
        require_close(sum(normalized), QComplex{1.0}, "deterministic chain normalization");
        require(std::string(chain.route_name()) == "ExactFactorChainTransfer",
                "exact factor chain route name is wrong");
        require(chain.stats().factor_width == width &&
                    chain.stats().factor_count == factor_count &&
                    chain.stats().retained_variables == width - 1U,
                "exact factor chain stats are wrong");
    }

    for (std::size_t trial = 0U; trial < 48U; ++trial) {
        const std::size_t width = 2U + static_cast<std::size_t>(rng() % 3U);
        const std::size_t factor_count = 1U + static_cast<std::size_t>(rng() % 4U);
        const std::size_t variable_count = factor_count + width - 1U;
        std::vector<std::size_t> dimensions(variable_count);
        for (std::size_t& dimension : dimensions) {
            dimension = 2U + static_cast<std::size_t>(rng() % 3U);
        }
        ExactFactorConfig config;
        config.max_factor_entries = 1U << 16U;
        ExactFactorGraph graph(config);
        const ReferenceGraph reference = build_chain(graph, dimensions, width, rng, true);
        const std::vector<FactorVariableId> retained = final_separator(factor_count, width);
        ExactFactorChainPlan chain(graph, retained);
        const std::vector<QComplex> expected = brute_marginal(reference, retained);
        const std::vector<QComplex> actual = chain.evaluate();
        compare(actual, expected, "random chain brute force");
        compare(actual, graph.marginal(retained), "random chain generic VE");
        require_close(chain.partition(), sum(expected), "random chain partition");
    }

    {
        const std::array<std::size_t, 5> dimensions{{2U, 3U, 2U, 2U, 3U}};
        ExactFactorGraph graph;
        const ReferenceGraph reference = build_chain(graph, dimensions, 3U, rng, false);
        static_cast<void>(reference);
        const std::vector<FactorVariableId> retained = final_separator(3U, 3U);
        ExactFactorChainPlan chain(graph, retained);
        auto workspace = chain.workspace();
        const std::vector<QComplex> before = chain.evaluate(workspace);

        std::vector<QComplex> replacement(12U);
        replacement[0] = QComplex{2.0, 0.1};
        replacement[5] = QComplex{0.7, -0.2};
        replacement[11] = QComplex{1.3, 0.0};
        const std::vector<FactorSparseEntry> sparse = sparse_entries(replacement);
        graph.set_sparse_factor(1U, sparse);
        compare(chain.evaluate(workspace), before, "chain snapshot isolation");
        chain.rebind_sparse_factor(1U, sparse);
        compare(chain.evaluate(workspace), graph.marginal(retained), "chain sparse rebind");
        require(chain.rebind_count() == 1U &&
                    chain.stats().source_sparse_factors == 1U,
                "chain sparse rebind stats are wrong");

        std::vector<QComplex> dense(12U, QComplex{0.25, 0.0});
        dense[3] = QComplex{1.5, 0.2};
        graph.set_dense_factor(1U, dense);
        chain.rebind_dense_factor(1U, dense);
        compare(chain.evaluate(workspace), graph.marginal(retained), "chain dense rebind");
        require(chain.rebind_count() == 2U &&
                    chain.stats().source_sparse_factors == 0U,
                "chain dense rebind stats are wrong");

        const std::vector<QComplex> stable = chain.evaluate(workspace);
        const std::size_t stable_count = chain.rebind_count();
        require_reject([&] {
            const std::array<QComplex, 2> wrong{{QComplex{1.0}, QComplex{2.0}}};
            chain.rebind_dense_factor(1U, wrong);
        }, "chain dense rebind accepted a changed size");
        require_reject([&] {
            const std::array<FactorSparseEntry, 2> duplicate{{
                FactorSparseEntry{1U, QComplex{1.0}},
                FactorSparseEntry{1U, QComplex{2.0}},
            }};
            chain.rebind_sparse_factor(1U, duplicate);
        }, "chain sparse rebind accepted a duplicate index");
        require_reject([&] {
            const std::array<FactorSparseEntry, 1> out_of_range{{
                FactorSparseEntry{12U, QComplex{1.0}},
            }};
            chain.rebind_sparse_factor(1U, out_of_range);
        }, "chain sparse rebind accepted an out-of-range index");
        require_reject([&] {
            std::vector<QComplex> nonfinite(12U, QComplex{1.0});
            nonfinite[0] = QComplex{std::numeric_limits<double>::infinity()};
            chain.rebind_dense_factor(1U, nonfinite);
        }, "chain dense rebind accepted a non-finite value");
        require_reject([&] {
            chain.rebind_dense_factor(3U, dense);
        }, "chain rebind accepted an out-of-range factor id");
        require(chain.rebind_count() == stable_count,
                "failed chain rebind changed the rebind count");
        compare(chain.evaluate(workspace), stable, "failed chain rebind was not transactional");
    }

    {
        ExactFactorGraph zero_graph;
        const FactorVariableId a = zero_graph.add_variable(2U);
        const FactorVariableId b = zero_graph.add_variable(2U);
        const std::array<FactorVariableId, 2> scope{{a, b}};
        const std::array<QComplex, 4> zeros{};
        static_cast<void>(zero_graph.add_dense_factor(scope, zeros));
        const std::array<FactorVariableId, 1> retained{{b}};
        ExactFactorChainPlan zero_chain(zero_graph, retained);
        require_reject([&] {
            static_cast<void>(zero_chain.normalized_marginal());
        }, "zero chain partition normalized successfully");
    }

    {
        ExactFactorGraph graph;
        static_cast<void>(graph.add_variable(2U));
        const std::array<FactorVariableId, 1> retained{{0U}};
        require_reject([&] {
            static_cast<void>(ExactFactorChainPlan(graph, retained));
        }, "chain accepted a graph without factors");
    }

    {
        ExactFactorGraph graph;
        const FactorVariableId a = graph.add_variable(2U);
        const std::array<FactorVariableId, 1> scope{{a}};
        const std::array<QComplex, 2> values{{QComplex{1.0}, QComplex{1.0}}};
        static_cast<void>(graph.add_dense_factor(scope, values));
        const std::span<const FactorVariableId> none{};
        require_reject([&] {
            static_cast<void>(ExactFactorChainPlan(graph, none));
        }, "chain accepted factor width one");
    }

    {
        ExactFactorGraph graph;
        const FactorVariableId a = graph.add_variable(2U);
        const FactorVariableId b = graph.add_variable(2U);
        static_cast<void>(graph.add_variable(2U));
        const std::array<FactorVariableId, 2> scope{{a, b}};
        const std::array<QComplex, 4> values{{QComplex{1.0}, QComplex{1.0}, QComplex{1.0}, QComplex{1.0}}};
        static_cast<void>(graph.add_dense_factor(scope, values));
        const std::array<FactorVariableId, 1> retained{{b}};
        require_reject([&] {
            static_cast<void>(ExactFactorChainPlan(graph, retained));
        }, "chain accepted an unspanned variable");
    }

    {
        ExactFactorGraph graph;
        const FactorVariableId a = graph.add_variable(2U);
        const FactorVariableId b = graph.add_variable(2U);
        const FactorVariableId c = graph.add_variable(2U);
        const std::array<QComplex, 4> values{{QComplex{1.0}, QComplex{1.0}, QComplex{1.0}, QComplex{1.0}}};
        const std::array<FactorVariableId, 2> first{{a, b}};
        const std::array<FactorVariableId, 2> permuted{{c, b}};
        static_cast<void>(graph.add_dense_factor(first, values));
        static_cast<void>(graph.add_dense_factor(permuted, values));
        const std::array<FactorVariableId, 1> retained{{c}};
        require_reject([&] {
            static_cast<void>(ExactFactorChainPlan(graph, retained));
        }, "chain accepted a permuted scope");
    }

    {
        ExactFactorGraph graph;
        const FactorVariableId a = graph.add_variable(2U);
        const FactorVariableId b = graph.add_variable(2U);
        const FactorVariableId c = graph.add_variable(2U);
        const std::array<QComplex, 4> pair{{QComplex{1.0}, QComplex{1.0}, QComplex{1.0}, QComplex{1.0}}};
        const std::array<QComplex, 2> unary{{QComplex{1.0}, QComplex{1.0}}};
        const std::array<FactorVariableId, 2> first{{a, b}};
        const std::array<FactorVariableId, 1> second{{c}};
        static_cast<void>(graph.add_dense_factor(first, pair));
        static_cast<void>(graph.add_dense_factor(second, unary));
        const std::array<FactorVariableId, 1> retained{{c}};
        require_reject([&] {
            static_cast<void>(ExactFactorChainPlan(graph, retained));
        }, "chain accepted a factor width mismatch");
    }

    {
        ExactFactorGraph graph;
        const FactorVariableId a = graph.add_variable(2U);
        const FactorVariableId b = graph.add_variable(2U);
        const FactorVariableId c = graph.add_variable(2U);
        const std::array<QComplex, 4> values{{QComplex{1.0}, QComplex{1.0}, QComplex{1.0}, QComplex{1.0}}};
        const std::array<FactorVariableId, 2> first{{a, b}};
        const std::array<FactorVariableId, 2> second{{b, c}};
        static_cast<void>(graph.add_dense_factor(first, values));
        static_cast<void>(graph.add_dense_factor(second, values));
        const std::array<FactorVariableId, 1> wrong{{b}};
        require_reject([&] {
            static_cast<void>(ExactFactorChainPlan(graph, wrong));
        }, "chain accepted the wrong retained variable");
        const std::array<FactorVariableId, 2> too_many{{b, c}};
        require_reject([&] {
            static_cast<void>(ExactFactorChainPlan(graph, too_many));
        }, "chain accepted the wrong retained width");
    }

    {
        ExactFactorGraph small;
        const FactorVariableId a = small.add_variable(2U);
        const FactorVariableId b = small.add_variable(2U);
        const std::array<FactorVariableId, 2> scope{{a, b}};
        const std::array<QComplex, 4> values{{QComplex{1.0}, QComplex{2.0}, QComplex{3.0}, QComplex{4.0}}};
        static_cast<void>(small.add_dense_factor(scope, values));
        const std::array<FactorVariableId, 1> retained{{b}};
        ExactFactorChainPlan small_chain(small, retained);
        auto wrong_workspace = small_chain.workspace();

        ExactFactorGraph larger;
        const FactorVariableId x = larger.add_variable(3U);
        const FactorVariableId y = larger.add_variable(3U);
        const std::array<FactorVariableId, 2> large_scope{{x, y}};
        std::vector<QComplex> large_values(9U, QComplex{1.0});
        static_cast<void>(larger.add_dense_factor(large_scope, large_values));
        const std::array<FactorVariableId, 1> large_retained{{y}};
        ExactFactorChainPlan large_chain(larger, large_retained);
        require_reject([&] {
            static_cast<void>(large_chain.evaluate(wrong_workspace));
        }, "chain accepted a mismatched workspace");
    }

    {
        ExactFactorConfig tight;
        tight.max_factor_entries = 4U;
        ExactFactorGraph graph(tight);
        const FactorVariableId a = graph.add_variable(2U);
        const FactorVariableId b = graph.add_variable(2U);
        const FactorVariableId c = graph.add_variable(2U);
        const std::array<FactorVariableId, 3> scope{{a, b, c}};
        std::vector<QComplex> values(8U, QComplex{1.0});
        require_reject([&] {
            static_cast<void>(graph.add_dense_factor(scope, values));
        }, "factor graph admitted a chain factor beyond the exact resource cap");
    }

    return 0;
}

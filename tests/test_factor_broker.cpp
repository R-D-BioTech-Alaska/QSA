#include "qubit/qfactor_broker.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using qubit::ExactFactorBrokerPlan;
using qubit::ExactFactorBrokerRoute;
using qubit::ExactFactorConfig;
using qubit::ExactFactorGraph;
using qubit::FactorId;
using qubit::FactorSparseEntry;
using qubit::FactorVariableId;
using qubit::QComplex;
using qubit::QStateError;

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

std::vector<FactorVariableId> final_separator(
    std::size_t factor_count,
    std::size_t width) {
    std::vector<FactorVariableId> retained(width - 1U);
    for (std::size_t index = 0U; index < retained.size(); ++index) {
        retained[index] = static_cast<FactorVariableId>(factor_count + index);
    }
    return retained;
}

std::vector<QComplex> values(
    std::size_t entries,
    std::mt19937_64& rng,
    bool sparse_shape) {
    std::uniform_real_distribution<double> weight(0.05, 1.2);
    std::vector<QComplex> result(entries);
    for (std::size_t index = 0U; index < entries; ++index) {
        if (sparse_shape && (rng() % 3U) != 0U) {
            continue;
        }
        result[index] = QComplex{weight(rng), 0.01 * weight(rng)};
    }
    if (std::none_of(result.begin(), result.end(), [](const QComplex& value) {
            return value.re != 0.0 || value.im != 0.0;
        })) {
        result[0] = QComplex{1.0, 0.0};
    }
    return result;
}

std::vector<FactorSparseEntry> sparse(std::span<const QComplex> dense) {
    std::vector<FactorSparseEntry> result;
    for (std::size_t index = 0U; index < dense.size(); ++index) {
        if (dense[index].re != 0.0 || dense[index].im != 0.0) {
            result.push_back({index, dense[index]});
        }
    }
    return result;
}

std::vector<FactorId> build_chain(
    ExactFactorGraph& graph,
    std::span<const std::size_t> dimensions,
    std::size_t width,
    std::mt19937_64& rng,
    bool mix_sparse) {
    for (const std::size_t dimension : dimensions) {
        static_cast<void>(graph.add_variable(dimension));
    }
    const std::size_t factor_count = dimensions.size() - width + 1U;
    std::vector<FactorId> ids;
    ids.reserve(factor_count);
    for (std::size_t factor = 0U; factor < factor_count; ++factor) {
        std::vector<FactorVariableId> scope(width);
        std::size_t entries = 1U;
        for (std::size_t position = 0U; position < width; ++position) {
            scope[position] = static_cast<FactorVariableId>(factor + position);
            entries *= dimensions[factor + position];
        }
        const bool use_sparse = mix_sparse && (factor % 2U == 1U);
        const std::vector<QComplex> dense = values(entries, rng, use_sparse);
        if (use_sparse) {
            const std::vector<FactorSparseEntry> entries_sparse = sparse(dense);
            ids.push_back(graph.add_sparse_factor(scope, entries_sparse));
        } else {
            ids.push_back(graph.add_dense_factor(scope, dense));
        }
    }
    return ids;
}

}  // namespace

int main() {
    std::mt19937_64 rng(0xb40c3e17ULL);

    ExactFactorConfig config;
    config.max_factor_entries = 1U << 16U;
    ExactFactorGraph graph(config);
    const std::array<std::size_t, 5> dimensions{{2U, 3U, 2U, 2U, 3U}};
    const std::vector<FactorId> factor_ids = build_chain(graph, dimensions, 3U, rng, true);
    const std::vector<FactorVariableId> retained = final_separator(3U, 3U);

    ExactFactorBrokerPlan broker(graph, retained);
    require(broker.route() == ExactFactorBrokerRoute::ChainTransfer,
            "broker did not select ChainTransfer for an eligible marginal");
    require(std::string(broker.route_name()) == "ExactFactorChainTransfer",
            "broker ChainTransfer route name is wrong");
    require(broker.chain_rejection().empty(),
            "eligible ChainTransfer retained a rejection reason");
    auto broker_workspace = broker.workspace();
    compare(broker.evaluate(broker_workspace), graph.marginal(retained),
            "broker ChainTransfer marginal");
    require_close(broker.partition(broker_workspace), graph.partition(),
                  "broker ChainTransfer partition");
    const std::vector<QComplex> normalized = broker.normalized_marginal(broker_workspace);
    QComplex normalized_sum{};
    for (const QComplex& value : normalized) {
        normalized_sum += value;
    }
    require_close(normalized_sum, QComplex{1.0}, "broker ChainTransfer normalization");

    ExactFactorBrokerPlan partition_broker(graph);
    require(partition_broker.route() == ExactFactorBrokerRoute::ChainTransfer,
            "broker did not select ChainTransfer for an eligible partition");
    auto partition_workspace = partition_broker.workspace();
    const std::vector<QComplex> partition_value = partition_broker.evaluate(partition_workspace);
    require(partition_value.size() == 1U, "broker partition output is not scalar");
    require_close(partition_value[0], graph.partition(), "broker scalar partition");
    const std::vector<QComplex> normalized_partition =
        partition_broker.normalized_marginal(partition_workspace);
    require(normalized_partition.size() == 1U,
            "broker normalized scalar partition size is wrong");
    require_close(normalized_partition[0], QComplex{1.0},
                  "broker normalized scalar partition is wrong");

    std::vector<QComplex> replacement(12U, QComplex{0.2, 0.0});
    replacement[0] = QComplex{1.7, 0.1};
    replacement[7] = QComplex{0.9, -0.1};
    graph.set_dense_factor(factor_ids[1], replacement);
    broker.rebind_dense_factor(factor_ids[1], replacement);
    compare(broker.evaluate(broker_workspace), graph.marginal(retained),
            "broker dense targeted rebind");
    require(broker.rebind_count() == 1U, "broker dense rebind count is wrong");

    replacement.assign(12U, QComplex{});
    replacement[1] = QComplex{0.8, 0.0};
    replacement[10] = QComplex{1.3, 0.2};
    const std::vector<FactorSparseEntry> sparse_replacement = sparse(replacement);
    graph.set_sparse_factor(factor_ids[1], sparse_replacement);
    broker.rebind_sparse_factor(factor_ids[1], sparse_replacement);
    compare(broker.evaluate(broker_workspace), graph.marginal(retained),
            "broker sparse targeted rebind");
    require(broker.rebind_count() == 2U, "broker sparse rebind count is wrong");

    const std::array<FactorVariableId, 1> nonfinal{{2U}};
    ExactFactorBrokerPlan nonfinal_broker(graph, nonfinal);
    require(nonfinal_broker.route() == ExactFactorBrokerRoute::VariableElimination,
            "broker selected ChainTransfer for a non-final marginal");
    require(!nonfinal_broker.chain_rejection().empty(),
            "broker fallback omitted its ChainTransfer rejection reason");
    compare(nonfinal_broker.evaluate(), graph.marginal(nonfinal),
            "broker non-final marginal fallback");

    ExactFactorGraph nonchain;
    const FactorVariableId a = nonchain.add_variable(2U);
    const FactorVariableId b = nonchain.add_variable(2U);
    const FactorVariableId c = nonchain.add_variable(2U);
    const std::array<QComplex, 4> pair{{
        QComplex{1.0}, QComplex{0.4}, QComplex{0.7}, QComplex{1.1},
    }};
    const std::array<FactorVariableId, 2> first{{a, b}};
    const std::array<FactorVariableId, 2> second{{a, c}};
    static_cast<void>(nonchain.add_dense_factor(first, pair));
    static_cast<void>(nonchain.add_dense_factor(second, pair));
    const std::array<FactorVariableId, 1> keep_c{{c}};
    ExactFactorBrokerPlan nonchain_broker(nonchain, keep_c);
    require(nonchain_broker.route() == ExactFactorBrokerRoute::VariableElimination,
            "broker selected ChainTransfer for a non-chain topology");
    require(!nonchain_broker.chain_rejection().empty(),
            "non-chain fallback omitted its rejection reason");
    compare(nonchain_broker.evaluate(), nonchain.marginal(keep_c),
            "broker non-chain fallback");

    ExactFactorGraph unary;
    const FactorVariableId u = unary.add_variable(3U);
    const std::array<FactorVariableId, 1> unary_scope{{u}};
    const std::array<QComplex, 3> unary_values{{
        QComplex{0.2}, QComplex{0.3}, QComplex{0.5},
    }};
    static_cast<void>(unary.add_dense_factor(unary_scope, unary_values));
    ExactFactorBrokerPlan unary_partition(unary);
    require(unary_partition.route() == ExactFactorBrokerRoute::VariableElimination,
            "broker selected ChainTransfer for unary factors");
    require(!unary_partition.chain_rejection().empty(),
            "unary fallback omitted its rejection reason");
    require_close(unary_partition.partition(), unary.partition(),
                  "broker unary partition fallback");

    auto wrong_workspace = nonfinal_broker.workspace();
    require_reject([&] {
        static_cast<void>(broker.evaluate(wrong_workspace));
    }, "broker accepted a workspace from another route");

    ExactFactorGraph zero;
    const FactorVariableId z0 = zero.add_variable(2U);
    const FactorVariableId z1 = zero.add_variable(2U);
    const std::array<FactorVariableId, 2> zero_scope{{z0, z1}};
    const std::array<QComplex, 4> zeros{};
    static_cast<void>(zero.add_dense_factor(zero_scope, zeros));
    ExactFactorBrokerPlan zero_partition(zero);
    require(zero_partition.route() == ExactFactorBrokerRoute::ChainTransfer,
            "zero chain did not retain ChainTransfer eligibility");
    require_reject([&] {
        static_cast<void>(zero_partition.normalized_marginal());
    }, "zero broker partition normalized successfully");

    for (std::size_t trial = 0U; trial < 48U; ++trial) {
        const std::size_t width = 2U + static_cast<std::size_t>(rng() % 3U);
        const std::size_t factor_count = 1U + static_cast<std::size_t>(rng() % 4U);
        const std::size_t variable_count = factor_count + width - 1U;
        std::vector<std::size_t> random_dimensions(variable_count);
        for (std::size_t& dimension : random_dimensions) {
            dimension = 2U + static_cast<std::size_t>(rng() % 3U);
        }
        ExactFactorGraph random_graph(config);
        static_cast<void>(build_chain(
            random_graph, random_dimensions, width, rng, true));
        const std::vector<FactorVariableId> random_retained =
            final_separator(factor_count, width);
        ExactFactorBrokerPlan random_broker(random_graph, random_retained);
        require(random_broker.route() == ExactFactorBrokerRoute::ChainTransfer,
                "random certified chain did not select ChainTransfer");
        compare(random_broker.evaluate(), random_graph.marginal(random_retained),
                "random broker ChainTransfer marginal");
    }

    return 0;
}

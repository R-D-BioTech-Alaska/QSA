#include "qubit/qfactor_broker.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using qubit::ExactFactorAffinePlan;
using qubit::ExactFactorBrokerConfig;
using qubit::ExactFactorBrokerPlan;
using qubit::ExactFactorBrokerRoute;
using qubit::ExactFactorChainPlan;
using qubit::ExactFactorDecisionPlan;
using qubit::ExactFactorGraph;
using qubit::ExactFactorPlan;
using qubit::FactorId;
using qubit::FactorSparseEntry;
using qubit::FactorVariableId;
using qubit::QComplex;
using qubit::QStateError;

static_assert(static_cast<std::uint8_t>(ExactFactorBrokerRoute::ChainTransfer) == 0U);
static_assert(static_cast<std::uint8_t>(ExactFactorBrokerRoute::DecisionDiagram) == 1U);
static_assert(static_cast<std::uint8_t>(ExactFactorBrokerRoute::VariableElimination) == 2U);
static_assert(static_cast<std::uint8_t>(ExactFactorBrokerRoute::AffineXOR) == 3U);

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

std::vector<FactorSparseEntry> sparse(std::span<const QComplex> dense) {
    std::vector<FactorSparseEntry> result;
    for (std::size_t index = 0U; index < dense.size(); ++index) {
        if (dense[index].re != 0.0 || dense[index].im != 0.0) {
            result.push_back({index, dense[index]});
        }
    }
    return result;
}

std::vector<QComplex> parity_dense(
    std::size_t width,
    bool rhs,
    QComplex coefficient = {1.0, 0.0}) {
    std::vector<QComplex> values(std::size_t{1U} << width);
    for (std::size_t index = 0U; index < values.size(); ++index) {
        if (((std::popcount(index) & 1U) != 0U) == rhs) {
            values[index] = coefficient;
        }
    }
    return values;
}

void chain_precedence_and_rebind() {
    ExactFactorGraph graph;
    const FactorVariableId a = graph.add_variable(2U);
    const FactorVariableId b = graph.add_variable(3U);
    const FactorVariableId c = graph.add_variable(2U);
    const std::array<FactorVariableId, 2> first_scope{{a, b}};
    const std::array<FactorVariableId, 2> second_scope{{b, c}};
    const std::array<QComplex, 6> first{{
        QComplex{0.8, 0.01}, QComplex{0.2, 0.0},
        QComplex{0.4, -0.01}, QComplex{0.6, 0.0},
        QComplex{0.7, 0.02}, QComplex{0.3, 0.0},
    }};
    const std::array<QComplex, 6> second{{
        QComplex{0.9, 0.0}, QComplex{0.3, 0.01}, QComplex{0.5, 0.0},
        QComplex{0.1, -0.01}, QComplex{0.7, 0.0}, QComplex{0.6, 0.02},
    }};
    static_cast<void>(graph.add_dense_factor(first_scope, first));
    const FactorId second_id = graph.add_dense_factor(second_scope, second);
    const std::array<FactorVariableId, 1> retained{{c}};

    ExactFactorBrokerPlan broker(graph, retained);
    require(broker.route() == ExactFactorBrokerRoute::ChainTransfer,
            "eligible chain did not retain first-route authority");
    require(std::string(broker.route_name()) == "ExactFactorChainTransfer",
            "chain route name changed");
    require(broker.chain_rejection().empty(),
            "eligible chain retained a chain rejection");
    require(broker.affine_rejection().empty(),
            "eligible chain evaluated a later affine route");
    require(broker.decision_rejection().empty(),
            "eligible chain evaluated a later decision route");
    compare(broker.evaluate(), ExactFactorChainPlan(graph, retained).evaluate(),
            "chain broker differs from direct ChainTransfer");
    compare(broker.evaluate(), ExactFactorPlan(graph, retained).evaluate(),
            "chain broker differs from generic exact VE");

    auto workspace = broker.workspace();
    const auto before = broker.evaluate(workspace);
    std::array<QComplex, 6> replacement = second;
    replacement[1U] = QComplex{1.2, -0.04};
    graph.set_dense_factor(second_id, replacement);
    compare(broker.evaluate(workspace), before,
            "chain broker snapshot changed before targeted rebind");
    broker.rebind_dense_factor(second_id, replacement);
    compare(broker.evaluate(workspace), ExactFactorPlan(graph, retained).evaluate(),
            "chain broker dense rebind differs from generic VE");
    require(broker.rebind_count() == 1U, "chain broker rebind count changed");

    ExactFactorBrokerPlan partition(graph);
    require(partition.route() == ExactFactorBrokerRoute::ChainTransfer,
            "eligible chain partition did not retain ChainTransfer");
    require_close(partition.partition(), graph.partition(),
                  "chain broker partition differs from graph partition");
}

struct AffineFixture {
    ExactFactorGraph graph{};
    FactorId rebound{0U};
    std::array<FactorVariableId, 1> retained{{0U}};
};

AffineFixture affine_fixture() {
    AffineFixture fixture;
    for (std::size_t variable = 0U; variable < 4U; ++variable) {
        static_cast<void>(fixture.graph.add_variable(2U));
    }
    const std::array<FactorVariableId, 3> first{{0U, 1U, 2U}};
    const std::array<FactorVariableId, 3> second{{0U, 2U, 3U}};
    const std::array<FactorVariableId, 3> third{{0U, 1U, 3U}};
    static_cast<void>(fixture.graph.add_dense_factor(
        first, parity_dense(first.size(), false, {0.8, 0.01})));
    fixture.rebound = fixture.graph.add_dense_factor(
        second, parity_dense(second.size(), true, {0.7, -0.02}));
    const std::vector<QComplex> third_dense =
        parity_dense(third.size(), false, {0.9, 0.03});
    static_cast<void>(fixture.graph.add_sparse_factor(third, sparse(third_dense)));
    return fixture;
}

void affine_route_and_transactional_rebind() {
    AffineFixture fixture = affine_fixture();
    ExactFactorBrokerPlan broker(fixture.graph, fixture.retained);
    require(broker.route() == ExactFactorBrokerRoute::AffineXOR,
            "certified parity graph did not select AffineXOR");
    require(std::string(broker.route_name()) == "ExactFactorAffineXOR",
            "affine route name changed");
    require(!broker.chain_rejection().empty(),
            "affine route omitted ChainTransfer rejection evidence");
    require(broker.affine_rejection().empty(),
            "accepted affine route retained an affine rejection");
    require(broker.decision_rejection().empty(),
            "accepted affine route evaluated a later decision route");
    compare(broker.evaluate(), ExactFactorAffinePlan(fixture.graph, fixture.retained).evaluate(),
            "affine broker differs from direct AffineXOR");
    compare(broker.evaluate(), ExactFactorPlan(fixture.graph, fixture.retained).evaluate(),
            "affine broker differs from generic VE");

    auto workspace = broker.workspace();
    const std::vector<QComplex> initial = broker.evaluate(workspace);
    const std::vector<QComplex> dense_replacement =
        parity_dense(3U, false, {0.55, 0.04});
    fixture.graph.set_dense_factor(fixture.rebound, dense_replacement);
    compare(broker.evaluate(workspace), initial,
            "affine broker snapshot changed before targeted rebind");
    broker.rebind_dense_factor(fixture.rebound, dense_replacement);
    compare(broker.evaluate(workspace), ExactFactorAffinePlan(fixture.graph, fixture.retained).evaluate(),
            "affine broker dense rebind differs from direct AffineXOR");
    require(broker.rebind_count() == 1U, "affine dense rebind count changed");

    const std::vector<QComplex> sparse_dense =
        parity_dense(3U, true, {0.61, -0.01});
    const std::vector<FactorSparseEntry> sparse_replacement = sparse(sparse_dense);
    fixture.graph.set_sparse_factor(fixture.rebound, sparse_replacement);
    broker.rebind_sparse_factor(fixture.rebound, sparse_replacement);
    compare(broker.evaluate(workspace), ExactFactorAffinePlan(fixture.graph, fixture.retained).evaluate(),
            "affine broker sparse rebind differs from direct AffineXOR");
    require(broker.rebind_count() == 2U, "affine sparse rebind count changed");

    const std::vector<QComplex> before_failed = broker.evaluate(workspace);
    const std::array<QComplex, 8> non_affine{{
        QComplex{1.0}, QComplex{}, QComplex{}, QComplex{1.0},
        QComplex{}, QComplex{2.0}, QComplex{1.0}, QComplex{},
    }};
    require_reject([&] {
        broker.rebind_dense_factor(fixture.rebound, non_affine);
    }, "affine broker accepted a non-affine rebind");
    compare(broker.evaluate(workspace), before_failed,
            "failed affine rebind changed broker state");
    require(broker.rebind_count() == 2U,
            "failed affine rebind changed successful rebind count");
}

struct DecisionFixture {
    ExactFactorGraph graph{};
    FactorId unary{0U};
    std::array<FactorVariableId, 1> retained{{0U}};
};

DecisionFixture decision_fixture() {
    DecisionFixture fixture;
    std::array<FactorVariableId, 4> variables{};
    for (std::size_t index = 0U; index < variables.size(); ++index) {
        variables[index] = fixture.graph.add_variable(2U);
    }
    const std::array<QComplex, 4> equality{{
        QComplex{1.0}, QComplex{}, QComplex{}, QComplex{1.0},
    }};
    for (std::size_t first = 0U; first < variables.size(); ++first) {
        for (std::size_t second = first + 1U; second < variables.size(); ++second) {
            const std::array<FactorVariableId, 2> scope{{variables[first], variables[second]}};
            static_cast<void>(fixture.graph.add_dense_factor(scope, equality));
        }
    }
    const std::array<QComplex, 2> weights{{
        QComplex{0.41, 0.07}, QComplex{0.73, -0.03},
    }};
    const std::array<FactorVariableId, 1> unary_scope{{variables[0]}};
    fixture.unary = fixture.graph.add_dense_factor(unary_scope, weights);
    fixture.retained[0] = variables[0];
    return fixture;
}

void decision_route_and_transactional_rebind() {
    DecisionFixture fixture = decision_fixture();
    ExactFactorBrokerPlan broker(fixture.graph, fixture.retained);
    require(broker.route() == ExactFactorBrokerRoute::DecisionDiagram,
            "cyclic weighted binary graph did not select DecisionDiagram");
    require(std::string(broker.route_name()) == "ExactFactorDecisionDiagram",
            "decision route name changed");
    require(!broker.chain_rejection().empty(),
            "decision route omitted ChainTransfer rejection evidence");
    require(!broker.affine_rejection().empty(),
            "decision route omitted AffineXOR rejection evidence");
    require(broker.decision_rejection().empty(),
            "accepted decision route retained a decision rejection");
    compare(broker.evaluate(), ExactFactorDecisionPlan(fixture.graph, fixture.retained).evaluate(),
            "decision broker differs from direct decision plan");
    compare(broker.evaluate(), ExactFactorPlan(fixture.graph, fixture.retained).evaluate(),
            "decision broker differs from generic VE");

    auto workspace = broker.workspace();
    const std::vector<QComplex> initial = broker.evaluate(workspace);
    const std::array<QComplex, 2> dense_replacement{{
        QComplex{0.22, 0.01}, QComplex{0.91, -0.02},
    }};
    fixture.graph.set_dense_factor(fixture.unary, dense_replacement);
    compare(broker.evaluate(workspace), initial,
            "decision broker snapshot changed before targeted rebind");
    broker.rebind_dense_factor(fixture.unary, dense_replacement);
    compare(broker.evaluate(workspace), ExactFactorPlan(fixture.graph, fixture.retained).evaluate(),
            "decision broker dense rebind differs from generic VE");
    require(broker.rebind_count() == 1U, "decision dense rebind count changed");

    const std::array<QComplex, 2> sparse_dense{{
        QComplex{}, QComplex{1.3, 0.04},
    }};
    const std::vector<FactorSparseEntry> sparse_replacement = sparse(sparse_dense);
    fixture.graph.set_sparse_factor(fixture.unary, sparse_replacement);
    broker.rebind_sparse_factor(fixture.unary, sparse_replacement);
    compare(broker.evaluate(workspace), ExactFactorPlan(fixture.graph, fixture.retained).evaluate(),
            "decision broker sparse rebind differs from generic VE");
    require(broker.rebind_count() == 2U, "decision sparse rebind count changed");

    const std::vector<QComplex> before_failed = broker.evaluate(workspace);
    const std::array<QComplex, 1> wrong_size{{QComplex{1.0}}};
    require_reject([&] {
        broker.rebind_dense_factor(fixture.unary, wrong_size);
    }, "decision broker accepted a shape-changing rebind");
    compare(broker.evaluate(workspace), before_failed,
            "failed decision rebind changed broker state");
    require(broker.rebind_count() == 2U,
            "failed decision rebind changed successful rebind count");

    const std::array<QComplex, 2> nonfinite{{
        QComplex{0.2}, QComplex{std::numeric_limits<double>::infinity()},
    }};
    require_reject([&] {
        broker.rebind_dense_factor(fixture.unary, nonfinite);
    }, "decision broker accepted a non-finite rebind");
    compare(broker.evaluate(workspace), before_failed,
            "non-finite decision rebind changed broker state");
}

void fallback_and_resource_routes() {
    ExactFactorGraph ternary;
    const FactorVariableId variable = ternary.add_variable(3U);
    const std::array<FactorVariableId, 1> scope{{variable}};
    const std::array<QComplex, 3> values{{
        QComplex{0.2}, QComplex{0.3}, QComplex{0.5},
    }};
    static_cast<void>(ternary.add_dense_factor(scope, values));
    ExactFactorBrokerPlan generic(ternary);
    require(generic.route() == ExactFactorBrokerRoute::VariableElimination,
            "nonbinary graph did not fail closed to generic VE");
    require(!generic.chain_rejection().empty(),
            "generic fallback omitted ChainTransfer rejection");
    require(!generic.affine_rejection().empty(),
            "generic fallback omitted AffineXOR rejection");
    require(!generic.decision_rejection().empty(),
            "generic fallback omitted DecisionDiagram rejection");
    require_close(generic.partition(), ternary.partition(),
                  "generic broker fallback partition changed");

    AffineFixture affine = affine_fixture();
    ExactFactorBrokerConfig affine_capped;
    affine_capped.affine.max_equations = 1U;
    ExactFactorBrokerPlan to_decision(affine.graph, affine.retained, affine_capped);
    require(to_decision.route() == ExactFactorBrokerRoute::DecisionDiagram,
            "Affine resource cap did not fail through to DecisionDiagram");
    require(!to_decision.affine_rejection().empty(),
            "Affine resource fallback omitted AffineXOR rejection");
    require(to_decision.decision_rejection().empty(),
            "accepted Decision fallback retained a Decision rejection");
    compare(to_decision.evaluate(), ExactFactorDecisionPlan(affine.graph, affine.retained).evaluate(),
            "Affine resource fallback differs from direct DecisionDiagram");

    ExactFactorBrokerConfig both_capped = affine_capped;
    both_capped.decision.max_nodes = 2U;
    ExactFactorBrokerPlan to_generic(affine.graph, affine.retained, both_capped);
    require(to_generic.route() == ExactFactorBrokerRoute::VariableElimination,
            "combined Affine/Decision caps did not fail through to generic VE");
    require(!to_generic.affine_rejection().empty(),
            "combined fallback omitted AffineXOR rejection");
    require(!to_generic.decision_rejection().empty(),
            "combined fallback omitted DecisionDiagram rejection");
    compare(to_generic.evaluate(), ExactFactorPlan(affine.graph, affine.retained).evaluate(),
            "combined resource fallback differs from generic VE");

    DecisionFixture decision = decision_fixture();
    ExactFactorBrokerConfig decision_capped;
    decision_capped.decision.max_nodes = 2U;
    ExactFactorBrokerPlan weighted_generic(
        decision.graph, decision.retained, decision_capped);
    require(weighted_generic.route() == ExactFactorBrokerRoute::VariableElimination,
            "weighted Decision cap did not fail through to generic VE");
    require(!weighted_generic.affine_rejection().empty(),
            "weighted generic fallback omitted AffineXOR rejection");
    require(!weighted_generic.decision_rejection().empty(),
            "weighted generic fallback omitted DecisionDiagram rejection");
    compare(weighted_generic.evaluate(), ExactFactorPlan(decision.graph, decision.retained).evaluate(),
            "weighted resource fallback differs from generic VE");

    const std::array<FactorVariableId, 2> two_retained{{0U, 1U}};
    ExactFactorBrokerPlan multi(decision.graph, two_retained);
    require(multi.route() == ExactFactorBrokerRoute::VariableElimination,
            "multi-retained query did not fail closed to generic VE");
    require(!multi.affine_rejection().empty(),
            "multi-retained fallback omitted AffineXOR rejection");
    require(!multi.decision_rejection().empty(),
            "multi-retained fallback omitted DecisionDiagram rejection");
    compare(multi.evaluate(), ExactFactorPlan(decision.graph, two_retained).evaluate(),
            "multi-retained broker differs from generic VE");
}

void workspace_route_rejection() {
    AffineFixture affine_fixture_value = affine_fixture();
    ExactFactorBrokerPlan affine(
        affine_fixture_value.graph, affine_fixture_value.retained);
    require(affine.route() == ExactFactorBrokerRoute::AffineXOR,
            "workspace affine control lost AffineXOR");

    DecisionFixture decision_fixture_value = decision_fixture();
    ExactFactorBrokerPlan decision(
        decision_fixture_value.graph, decision_fixture_value.retained);
    require(decision.route() == ExactFactorBrokerRoute::DecisionDiagram,
            "workspace decision control lost DecisionDiagram");

    ExactFactorGraph chain_graph;
    const FactorVariableId a = chain_graph.add_variable(2U);
    const FactorVariableId b = chain_graph.add_variable(2U);
    const FactorVariableId c = chain_graph.add_variable(2U);
    const std::array<QComplex, 4> pair{{
        QComplex{0.8}, QComplex{0.2}, QComplex{0.3}, QComplex{0.7},
    }};
    const std::array<FactorVariableId, 2> first{{a, b}};
    const std::array<FactorVariableId, 2> second{{b, c}};
    static_cast<void>(chain_graph.add_dense_factor(first, pair));
    static_cast<void>(chain_graph.add_dense_factor(second, pair));
    const std::array<FactorVariableId, 1> retained{{c}};
    ExactFactorBrokerPlan chain(chain_graph, retained);
    require(chain.route() == ExactFactorBrokerRoute::ChainTransfer,
            "workspace chain control lost ChainTransfer");

    ExactFactorGraph ternary;
    static_cast<void>(ternary.add_variable(3U));
    const std::array<FactorVariableId, 1> ternary_scope{{0U}};
    const std::array<QComplex, 3> ternary_values{{
        QComplex{0.2}, QComplex{0.3}, QComplex{0.5},
    }};
    static_cast<void>(ternary.add_dense_factor(ternary_scope, ternary_values));
    ExactFactorBrokerPlan generic(ternary);
    require(generic.route() == ExactFactorBrokerRoute::VariableElimination,
            "workspace generic control lost VE");

    auto chain_workspace = chain.workspace();
    require_reject([&] { static_cast<void>(affine.evaluate(chain_workspace)); },
                   "AffineXOR accepted ChainTransfer workspace");
    auto affine_workspace = affine.workspace();
    require_reject([&] { static_cast<void>(decision.evaluate(affine_workspace)); },
                   "DecisionDiagram accepted AffineXOR workspace");
    auto decision_workspace = decision.workspace();
    require_reject([&] { static_cast<void>(generic.evaluate(decision_workspace)); },
                   "generic VE accepted DecisionDiagram workspace");
    auto generic_workspace = generic.workspace();
    require_reject([&] { static_cast<void>(chain.evaluate(generic_workspace)); },
                   "ChainTransfer accepted generic VE workspace");
}

void randomized_chain_precedence() {
    std::mt19937_64 generator(0xb40c3e17ULL);
    for (std::size_t trial = 0U; trial < 32U; ++trial) {
        const std::size_t variables = 4U + static_cast<std::size_t>(generator() % 5U);
        ExactFactorGraph graph;
        std::vector<FactorVariableId> ids(variables);
        for (std::size_t index = 0U; index < variables; ++index) {
            ids[index] = graph.add_variable(2U + static_cast<std::size_t>(generator() % 2U));
        }
        for (std::size_t factor = 0U; factor + 1U < variables; ++factor) {
            const std::array<FactorVariableId, 2> scope{{ids[factor], ids[factor + 1U]}};
            const std::size_t entries =
                graph.dimension(ids[factor]) * graph.dimension(ids[factor + 1U]);
            std::vector<QComplex> dense(entries);
            for (std::size_t entry = 0U; entry < entries; ++entry) {
                dense[entry] = QComplex{
                    0.1 + 0.01 * static_cast<double>((entry + trial) % 31U),
                    0.001 * static_cast<double>((entry + factor) % 7U),
                };
            }
            static_cast<void>(graph.add_dense_factor(scope, dense));
        }
        const std::array<FactorVariableId, 1> retained{{ids.back()}};
        ExactFactorBrokerPlan broker(graph, retained);
        require(broker.route() == ExactFactorBrokerRoute::ChainTransfer,
                "random certified chain lost ChainTransfer precedence");
        require(broker.affine_rejection().empty(),
                "random chain evaluated later AffineXOR route");
        compare(broker.evaluate(), ExactFactorPlan(graph, retained).evaluate(),
                "random chain broker differs from generic VE");
    }
}

}  // namespace

int main() {
    chain_precedence_and_rebind();
    affine_route_and_transactional_rebind();
    decision_route_and_transactional_rebind();
    fallback_and_resource_routes();
    workspace_route_rejection();
    randomized_chain_precedence();
    return 0;
}

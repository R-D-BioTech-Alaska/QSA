#include "qubit/qfactor.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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
    const std::vector<QComplex>& actual,
    const std::vector<QComplex>& expected,
    const std::string& message) {
    require(actual.size() == expected.size(), message + " size");
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        require(qubit::almost_equal(actual[index], expected[index], 2e-12),
                message + " value");
    }
}

ExactFactorGraph make_chain(ExactFactorConfig config) {
    ExactFactorGraph graph(config);
    constexpr std::size_t variables = 8U;
    for (std::size_t variable = 0U; variable < variables; ++variable) {
        static_cast<void>(graph.add_variable(2U));
    }
    const std::array<QComplex, 8> values{{
        QComplex{1.0}, QComplex{0.2}, QComplex{0.3}, QComplex{1.1},
        QComplex{0.4}, QComplex{1.2}, QComplex{1.3}, QComplex{0.5},
    }};
    for (std::size_t factor = 0U; factor + 2U < variables; ++factor) {
        const std::array<FactorVariableId, 3> scope{{
            static_cast<FactorVariableId>(factor),
            static_cast<FactorVariableId>(factor + 1U),
            static_cast<FactorVariableId>(factor + 2U),
        }};
        static_cast<void>(graph.add_dense_factor(scope, values));
    }
    return graph;
}

}  // namespace

int main() {
    ExactFactorGraph graph;
    const FactorVariableId a = graph.add_variable(2U);
    const FactorVariableId b = graph.add_variable(3U);
    const std::array<FactorVariableId, 2> scope{{a, b}};
    const std::array<QComplex, 6> initial{{
        QComplex{1.0}, QComplex{2.0}, QComplex{3.0},
        QComplex{4.0}, QComplex{5.0}, QComplex{6.0},
    }};
    const FactorId factor = graph.add_dense_factor(scope, initial);
    const std::array<FactorVariableId, 1> retained{{b}};
    auto plan = graph.compile(retained);
    auto workspace = plan.workspace();
    require(plan.stats().compiled_index_entries == 6U,
            "bounded factor map did not compile the eligible step");
    require(plan.stats().workspace_slots == 1U,
            "single factor step should require one workspace slot");

    ExactFactorConfig generic_config;
    generic_config.max_compiled_index_entries = 0U;
    ExactFactorGraph generic_graph(generic_config);
    const FactorVariableId ga = generic_graph.add_variable(2U);
    const FactorVariableId gb = generic_graph.add_variable(3U);
    const std::array<FactorVariableId, 2> generic_scope{{ga, gb}};
    static_cast<void>(generic_graph.add_dense_factor(generic_scope, initial));
    const std::array<FactorVariableId, 1> generic_retained{{gb}};
    auto generic_plan = generic_graph.compile(generic_retained);
    require(generic_plan.stats().compiled_index_entries == 0U,
            "zero map cap did not preserve the generic evaluator");
    require_close(plan.evaluate(workspace), generic_plan.evaluate(),
                  "compiled and generic factor evaluators disagree");

    ExactFactorConfig bounded_config;
    bounded_config.max_compiled_index_entries = 5U;
    ExactFactorGraph bounded_graph(bounded_config);
    const FactorVariableId ba = bounded_graph.add_variable(2U);
    const FactorVariableId bb = bounded_graph.add_variable(3U);
    const std::array<FactorVariableId, 2> bounded_scope{{ba, bb}};
    static_cast<void>(bounded_graph.add_dense_factor(bounded_scope, initial));
    const std::array<FactorVariableId, 1> bounded_retained{{bb}};
    auto bounded_plan = bounded_graph.compile(bounded_retained);
    require(bounded_plan.stats().compiled_index_entries == 0U,
            "factor map exceeded its hard plan cap");
    require_close(bounded_plan.evaluate(), generic_plan.evaluate(),
                  "hard map cap changed exact evaluation");

    ExactFactorConfig reused_config;
    reused_config.max_factor_entries = 64U;
    reused_config.reuse_workspace_slots = true;
    ExactFactorGraph reused_chain = make_chain(reused_config);
    const std::array<FactorVariableId, 2> chain_retained{{6U, 7U}};
    auto reused_plan = reused_chain.compile(chain_retained);
    auto reused_workspace = reused_plan.workspace();

    ExactFactorConfig unique_config = reused_config;
    unique_config.reuse_workspace_slots = false;
    ExactFactorGraph unique_chain = make_chain(unique_config);
    auto unique_plan = unique_chain.compile(chain_retained);
    auto unique_workspace = unique_plan.workspace();

    require(reused_plan.step_count() > 2U,
            "workspace liveness chain did not create enough elimination steps");
    require(unique_plan.stats().workspace_slots == unique_plan.step_count(),
            "unique workspace mode did not retain one slot per step");
    require(reused_plan.stats().workspace_slots < reused_plan.step_count(),
            "workspace liveness reuse did not release dead intermediates");
    require(reused_workspace.estimated_bytes() < unique_workspace.estimated_bytes(),
            "workspace liveness reuse did not reduce retained bytes");
    require_close(reused_plan.evaluate(reused_workspace),
                  unique_plan.evaluate(unique_workspace),
                  "workspace liveness reuse changed exact evaluation");

    bool rejected = false;
    try {
        unique_plan.evaluate(reused_workspace);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "unique plan accepted a reused workspace shape");

    rejected = false;
    try {
        reused_plan.evaluate(unique_workspace);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "reused plan accepted a unique workspace shape");

    const std::array<FactorSparseEntry, 3> sparse{{
        FactorSparseEntry{0U, QComplex{2.0}},
        FactorSparseEntry{3U, QComplex{7.0}},
        FactorSparseEntry{5U, QComplex{11.0}},
    }};
    plan.rebind_sparse_factor(factor, sparse);
    graph.set_sparse_factor(factor, sparse);
    require_close(plan.evaluate(workspace), graph.marginal(retained),
                  "targeted sparse rebind");
    require(plan.rebind_count() == 1U &&
                plan.stats().source_dense_factors == 0U &&
                plan.stats().source_sparse_factors == 1U,
            "targeted sparse rebind stats are wrong");

    const std::array<QComplex, 6> dense{{
        QComplex{6.0}, QComplex{5.0}, QComplex{4.0},
        QComplex{3.0}, QComplex{2.0}, QComplex{1.0},
    }};
    plan.rebind_dense_factor(factor, dense);
    graph.set_dense_factor(factor, dense);
    require_close(plan.evaluate(workspace), graph.marginal(retained),
                  "targeted dense rebind");
    require(plan.rebind_count() == 2U &&
                plan.stats().source_dense_factors == 1U &&
                plan.stats().source_sparse_factors == 0U,
            "targeted dense rebind stats are wrong");

    const auto stable = plan.evaluate(workspace);
    const std::size_t stable_count = plan.rebind_count();

    rejected = false;
    const std::array<QComplex, 2> wrong_size{{QComplex{1.0}, QComplex{2.0}}};
    try {
        plan.rebind_dense_factor(factor, wrong_size);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "targeted dense rebind accepted a changed size");

    rejected = false;
    const std::array<FactorSparseEntry, 2> duplicate{{
        FactorSparseEntry{1U, QComplex{1.0}},
        FactorSparseEntry{1U, QComplex{2.0}},
    }};
    try {
        plan.rebind_sparse_factor(factor, duplicate);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "targeted sparse rebind accepted a duplicate index");

    rejected = false;
    const std::array<FactorSparseEntry, 1> out_of_range{{
        FactorSparseEntry{6U, QComplex{1.0}},
    }};
    try {
        plan.rebind_sparse_factor(factor, out_of_range);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "targeted sparse rebind accepted an out-of-range index");

    rejected = false;
    const std::array<QComplex, 6> nonfinite{{
        QComplex{1.0}, QComplex{2.0}, QComplex{3.0},
        QComplex{4.0}, QComplex{5.0},
        QComplex{std::numeric_limits<double>::infinity()},
    }};
    try {
        plan.rebind_dense_factor(factor, nonfinite);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "targeted dense rebind accepted a non-finite value");

    rejected = false;
    try {
        plan.rebind_dense_factor(static_cast<FactorId>(factor + 1U), dense);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "targeted rebind accepted an out-of-range factor id");

    require(plan.rebind_count() == stable_count,
            "failed targeted rebind changed the rebind count");
    require_close(plan.evaluate(workspace), stable,
                  "failed targeted rebind was not transactional");
    return 0;
}

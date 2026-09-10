#include "qubit/qfactor.hpp"
#include "qubit/qfactor_cache.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_exact(
    const qubit::QComplex& actual,
    const qubit::QComplex& expected,
    const char* message) {
    require(actual.re == expected.re && actual.im == expected.im, message);
}

qubit::ExactFactorGraph graph() {
    qubit::ExactFactorConfig config;
    config.max_variables = 16U;
    config.max_factors = 32U;
    config.max_factor_entries = 64U;
    config.max_compiled_index_entries = 4096U;
    config.reuse_workspace_slots = true;
    qubit::ExactFactorGraph result(config);
    const auto a = result.add_variable(2U);
    const auto b = result.add_variable(2U);
    const auto c = result.add_variable(2U);
    const std::array<qubit::FactorVariableId, 1> a_scope{a};
    const std::array<qubit::FactorVariableId, 2> ab_scope{a, b};
    const std::array<qubit::FactorVariableId, 2> bc_scope{b, c};
    const std::array<qubit::QComplex, 2> a_values{
        qubit::QComplex{1.0, 0.0}, qubit::QComplex{1.0, 0.0}};
    const std::array<qubit::QComplex, 4> ab_values{
        qubit::QComplex{1.0, 0.0}, qubit::QComplex{0.25, 0.0},
        qubit::QComplex{0.5, 0.0}, qubit::QComplex{2.0, 0.0}};
    const std::array<qubit::QComplex, 4> bc_values{
        qubit::QComplex{1.0, 0.0}, qubit::QComplex{0.75, 0.0},
        qubit::QComplex{1.25, 0.0}, qubit::QComplex{0.5, 0.0}};
    (void)result.add_dense_factor(a_scope, a_values);
    (void)result.add_dense_factor(ab_scope, ab_values);
    (void)result.add_dense_factor(bc_scope, bc_values);
    return result;
}

struct ChainCarrier {
    qubit::ExactFactorGraph graph;
    std::vector<qubit::FactorVariableId> variables;
    std::vector<qubit::FactorId> mutable_factors;
};

ChainCarrier chain(std::size_t variables, std::size_t cache_cap = 4096U) {
    qubit::ExactFactorConfig config;
    config.max_variables = variables;
    config.max_factors = 2U * variables;
    config.max_factor_entries = 16U;
    config.max_compiled_index_entries = cache_cap;
    config.reuse_workspace_slots = true;

    ChainCarrier result{qubit::ExactFactorGraph(config), {}, {}};
    result.variables.reserve(variables);
    result.mutable_factors.reserve(variables);
    for (std::size_t index = 0U; index < variables; ++index) {
        result.variables.push_back(result.graph.add_variable(2U));
    }
    const std::array<qubit::QComplex, 2> unary{
        qubit::QComplex{1.0, 0.0}, qubit::QComplex{1.0, 0.0}};
    for (const qubit::FactorVariableId variable : result.variables) {
        const std::array<qubit::FactorVariableId, 1> scope{variable};
        result.mutable_factors.push_back(result.graph.add_dense_factor(scope, unary));
    }
    const std::array<qubit::QComplex, 4> pair{
        qubit::QComplex{0.6, 0.0}, qubit::QComplex{0.4, 0.0},
        qubit::QComplex{0.4, 0.0}, qubit::QComplex{0.6, 0.0}};
    for (std::size_t index = 0U; index + 1U < variables; ++index) {
        const std::array<qubit::FactorVariableId, 2> scope{
            result.variables[index], result.variables[index + 1U]};
        (void)result.graph.add_dense_factor(scope, pair);
    }
    return result;
}

qubit::QComplex direct_partition(
    const std::array<qubit::QComplex, 2>& values) {
    auto direct_graph = graph();
    direct_graph.set_dense_factor(0U, values);
    return direct_graph.partition();
}

void independent_bound_workspaces_match_direct_graphs() {
    auto source = graph();
    const qubit::ExactFactorPlan plan(source);
    const std::array<qubit::FactorId, 1> bound{0U};
    auto first = plan.workspace(bound);
    auto second = plan.workspace(bound);

    const std::array<qubit::QComplex, 2> first_values{
        qubit::QComplex{2.0, 0.25}, qubit::QComplex{-0.5, 0.75}};
    const std::array<qubit::QComplex, 2> second_values{
        qubit::QComplex{-1.0, 0.5}, qubit::QComplex{0.25, -0.125}};
    plan.bind_dense_factor(first, 0U, first_values);
    plan.bind_dense_factor(second, 0U, second_values);

    require(qubit::almost_equal(
                plan.bound_partition(first), direct_partition(first_values), 2e-12),
        "first bound workspace differs from direct graph recompile");
    require(qubit::almost_equal(
                plan.bound_partition(second), direct_partition(second_values), 2e-12),
        "second bound workspace differs from direct graph recompile");
    require(qubit::almost_equal(
                plan.bound_partition(first), direct_partition(first_values), 2e-12),
        "second workspace contaminated first bound workspace");
    require(first.bound_rebind_count() == 1U && second.bound_rebind_count() == 1U,
        "bound workspace rebind counters are not workspace-local");
    require(plan.rebind_count() == 0U,
        "workspace binding mutated the compiled factor plan");
    require(first.binding_estimated_bytes() > 0U,
        "bound workspace did not account for binding storage");
}

void same_plan_parallel_workspaces_are_isolated() {
    auto source = graph();
    const qubit::ExactFactorPlan plan(source);
    const std::array<qubit::FactorId, 1> bound{0U};
    constexpr std::size_t workers = 8U;
    constexpr std::size_t iterations = 200U;
    std::array<std::thread, workers> threads;
    std::atomic<bool> failed{false};

    for (std::size_t worker = 0U; worker < workers; ++worker) {
        threads[worker] = std::thread([&, worker]() {
            try {
                auto workspace = plan.workspace(bound);
                for (std::size_t iteration = 0U; iteration < iterations; ++iteration) {
                    const double scale =
                        0.125 * static_cast<double>(1U + worker) +
                        0.001 * static_cast<double>(iteration);
                    const std::array<qubit::QComplex, 2> values{
                        qubit::QComplex{1.0 + scale, -0.5 * scale},
                        qubit::QComplex{-0.25 * scale, 0.75 + scale}};
                    plan.bind_dense_factor(workspace, 0U, values);
                    const qubit::QComplex observed = plan.bound_partition(workspace);
                    const qubit::QComplex expected = direct_partition(values);
                    if (!qubit::almost_equal(observed, expected, 3e-12)) {
                        failed.store(true, std::memory_order_relaxed);
                        return;
                    }
                }
                if (workspace.bound_rebind_count() != iterations) {
                    failed.store(true, std::memory_order_relaxed);
                }
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    require(!failed.load(std::memory_order_relaxed),
        "parallel bound workspaces shared mutable source state");
    require(plan.rebind_count() == 0U,
        "parallel bound workspaces mutated the compiled plan");
}

void binding_contract_fails_closed() {
    auto source = graph();
    const qubit::ExactFactorPlan plan(source);
    const std::array<qubit::FactorId, 1> bound{0U};
    auto workspace = plan.workspace(bound);
    bool rejected = false;
    try {
        const std::array<qubit::QComplex, 2> values{
            qubit::QComplex{1.0, 0.0}, qubit::QComplex{1.0, 0.0}};
        plan.bind_dense_factor(workspace, 1U, values);
    } catch (const qubit::QStateError&) {
        rejected = true;
    }
    require(rejected, "bound workspace accepted an undeclared factor");

    rejected = false;
    try {
        const std::array<qubit::QComplex, 1> short_values{qubit::QComplex{1.0, 0.0}};
        plan.bind_dense_factor(workspace, 0U, short_values);
    } catch (const qubit::QStateError&) {
        rejected = true;
    }
    require(rejected, "bound workspace accepted a changed dense factor shape");

    rejected = false;
    try {
        const std::array<qubit::QComplex, 2> bad_values{
            qubit::QComplex{1.0, 0.0},
            qubit::QComplex{std::numeric_limits<double>::quiet_NaN(), 0.0}};
        plan.bind_dense_factor(workspace, 0U, bad_values);
    } catch (const qubit::QStateError&) {
        rejected = true;
    }
    require(rejected, "bound workspace accepted non-finite source values");

    rejected = false;
    try {
        const std::array<qubit::FactorId, 2> duplicate{0U, 0U};
        (void)plan.workspace(duplicate);
    } catch (const qubit::QStateError&) {
        rejected = true;
    }
    require(rejected, "bound workspace accepted duplicate source declarations");

    rejected = false;
    try {
        auto sparse_graph = graph();
        const std::array<qubit::FactorSparseEntry, 1> entries{
            qubit::FactorSparseEntry{0U, qubit::QComplex{1.0, 0.0}}};
        sparse_graph.set_sparse_factor(0U, entries);
        const qubit::ExactFactorPlan sparse_plan(sparse_graph);
        (void)sparse_plan.workspace(bound);
    } catch (const qubit::QStateError&) {
        rejected = true;
    }
    require(rejected, "bound workspace silently converted a sparse source factor");
}

void persistent_messages_match_full_recompute() {
    ChainCarrier carrier = chain(24U, 16384U);
    qubit::ExactFactorPlan plan(carrier.graph);
    auto workspace = plan.workspace(carrier.mutable_factors);
    qubit::ExactFactorMessageCache cache(plan, carrier.mutable_factors);

    require_exact(cache.partition(), plan.bound_partition(workspace),
        "cold message cache differs from full factor evaluation");
    require(cache.stats().last_recomputed_steps == plan.step_count() &&
            cache.stats().last_reused_steps == 0U,
        "cold message cache did not materialize every elimination message");

    const std::array<qubit::QComplex, 2> last_values{
        qubit::QComplex{1.25, 0.0}, qubit::QComplex{0.75, 0.0}};
    const qubit::FactorId last = carrier.mutable_factors.back();
    plan.bind_dense_factor(workspace, last, last_values);
    cache.bind_dense_factor(last, last_values);
    require_exact(cache.partition(), plan.bound_partition(workspace),
        "local message-cache update differs from full factor evaluation");
    require(cache.stats().last_recomputed_steps == 1U &&
            cache.stats().last_reused_steps + 1U == plan.step_count(),
        "terminal chain update did not reuse unaffected elimination messages");

    require_exact(cache.partition(), plan.bound_partition(workspace),
        "unchanged message cache differs from full factor evaluation");
    require(cache.stats().last_recomputed_steps == 0U &&
            cache.stats().last_reused_steps == plan.step_count(),
        "unchanged message cache recomputed clean messages");

    const qubit::QComplex accepted = cache.partition();
    bool rejected = false;
    try {
        const std::array<qubit::QComplex, 2> bad{
            qubit::QComplex{1.0, 0.0},
            qubit::QComplex{std::numeric_limits<double>::quiet_NaN(), 0.0}};
        cache.bind_dense_factor(last, bad);
    } catch (const qubit::QStateError&) {
        rejected = true;
    }
    require(rejected, "message cache accepted non-finite bound values");
    require_exact(cache.partition(), accepted,
        "failed message-cache bind changed accepted cached state");
    require(cache.stats().last_recomputed_steps == 0U,
        "failed message-cache bind dirtied accepted messages");
}

void persistent_messages_handle_retained_terminals() {
    ChainCarrier carrier = chain(12U, 8192U);
    const std::array<qubit::FactorVariableId, 1> retained{carrier.variables.back()};
    qubit::ExactFactorPlan plan(carrier.graph, retained);
    auto workspace = plan.workspace(carrier.mutable_factors);
    qubit::ExactFactorMessageCache cache(plan, carrier.mutable_factors);
    (void)cache.evaluate();

    const std::array<qubit::QComplex, 2> values{
        qubit::QComplex{1.3, 0.0}, qubit::QComplex{0.7, 0.0}};
    const qubit::FactorId last = carrier.mutable_factors.back();
    plan.bind_dense_factor(workspace, last, values);
    cache.bind_dense_factor(last, values);
    const auto cached = cache.evaluate();
    const auto full = plan.bound_evaluate(workspace);
    require(cached.size() == full.size(), "retained message-cache output size changed");
    for (std::size_t index = 0U; index < full.size(); ++index) {
        require_exact(cached[index], full[index],
            "retained terminal message-cache update differs from full evaluation");
    }
    require(cache.stats().last_recomputed_steps == 0U &&
            cache.stats().last_reused_steps == plan.step_count(),
        "retained terminal source invalidated unrelated elimination messages");
}

void repeated_local_messages_match_full_recompute() {
    ChainCarrier carrier = chain(16U, 8192U);
    qubit::ExactFactorPlan plan(carrier.graph);
    auto workspace = plan.workspace(carrier.mutable_factors);
    qubit::ExactFactorMessageCache cache(plan, carrier.mutable_factors);
    (void)cache.partition();
    bool reused = false;

    for (std::size_t iteration = 0U; iteration < 64U; ++iteration) {
        const std::size_t variable = (7U * iteration + 3U) % carrier.mutable_factors.size();
        const double delta = 1e-4 * static_cast<double>(1U + iteration);
        const std::array<qubit::QComplex, 2> values{
            qubit::QComplex{1.0 + delta, 0.0}, qubit::QComplex{1.0 - delta, 0.0}};
        const qubit::FactorId factor = carrier.mutable_factors[variable];
        plan.bind_dense_factor(workspace, factor, values);
        cache.bind_dense_factor(factor, values);
        require_exact(cache.partition(), plan.bound_partition(workspace),
            "repeated local message update differs from full recompute");
        reused = reused || cache.stats().last_reused_steps != 0U;
    }
    require(reused, "repeated local message updates never reused a separator message");
    require(cache.stats().rebind_count == 64U,
        "message-cache rebind accounting is wrong");
}

void message_cache_fails_closed() {
    {
        ChainCarrier carrier = chain(4U, 4U);
        const qubit::ExactFactorPlan plan(carrier.graph);
        bool rejected = false;
        try {
            (void)qubit::ExactFactorMessageCache(plan, carrier.mutable_factors);
        } catch (const qubit::QStateError&) {
            rejected = true;
        }
        require(rejected, "message cache ignored its compiled-state resource cap");
        require(carrier.graph.validate(), "failed message-cache compile damaged its graph");
    }

    {
        ChainCarrier carrier = chain(6U, 4096U);
        qubit::ExactFactorPlan plan(carrier.graph);
        qubit::ExactFactorMessageCache cache(plan, carrier.mutable_factors);
        (void)cache.partition();
        const std::array<qubit::QComplex, 2> values{
            qubit::QComplex{1.1, 0.0}, qubit::QComplex{0.9, 0.0}};
        plan.rebind_dense_factor(carrier.mutable_factors.front(), values);
        bool rejected = false;
        try {
            (void)cache.partition();
        } catch (const qubit::QStateError&) {
            rejected = true;
        }
        require(rejected, "message cache reused messages after plan source mutation");
    }

    {
        auto source = graph();
        const qubit::ExactFactorPlan plan(source);
        bool rejected = false;
        try {
            const std::span<const qubit::FactorId> none{};
            (void)qubit::ExactFactorMessageCache(plan, none);
        } catch (const qubit::QStateError&) {
            rejected = true;
        }
        require(rejected, "message cache accepted an empty binding contract");
    }
}

}  // namespace

int main() {
    independent_bound_workspaces_match_direct_graphs();
    same_plan_parallel_workspaces_are_isolated();
    binding_contract_fails_closed();
    persistent_messages_match_full_recompute();
    persistent_messages_handle_retained_terminals();
    repeated_local_messages_match_full_recompute();
    message_cache_fails_closed();
    return 0;
}

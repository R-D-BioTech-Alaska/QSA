#include "qubit/qfactor.hpp"

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

}  // namespace

int main() {
    independent_bound_workspaces_match_direct_graphs();
    same_plan_parallel_workspaces_are_isolated();
    binding_contract_fails_closed();
    return 0;
}

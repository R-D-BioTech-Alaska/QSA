#include "qubit/qfactor.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

namespace {

struct Carrier {
    qubit::ExactFactorGraph graph;
    std::vector<qubit::FactorId> mutable_factors;
};

Carrier make_carrier() {
    constexpr std::size_t variables = 4096U;
    qubit::ExactFactorConfig config;
    config.max_variables = variables;
    config.max_factors = 2U * variables;
    config.max_factor_entries = 16U;
    config.max_compiled_index_entries = 2U * 1024U * 1024U;
    config.reuse_workspace_slots = true;

    Carrier carrier{qubit::ExactFactorGraph(config), {}};
    std::vector<qubit::FactorVariableId> ids;
    ids.reserve(variables);
    carrier.mutable_factors.reserve(variables);
    for (std::size_t index = 0U; index < variables; ++index) {
        ids.push_back(carrier.graph.add_variable(2U));
    }
    for (std::size_t index = 0U; index < variables; ++index) {
        const std::array<qubit::FactorVariableId, 1> scope{ids[index]};
        const std::array<qubit::QComplex, 2> values{
            qubit::QComplex{0.5, 0.0}, qubit::QComplex{0.5, 0.0}};
        carrier.mutable_factors.push_back(carrier.graph.add_dense_factor(scope, values));
    }
    for (std::size_t index = 0U; index + 1U < variables; ++index) {
        const std::array<qubit::FactorVariableId, 2> scope{ids[index], ids[index + 1U]};
        const std::array<qubit::QComplex, 4> values{
            qubit::QComplex{1.0, 0.0}, qubit::QComplex{0.75, 0.0},
            qubit::QComplex{0.75, 0.0}, qubit::QComplex{1.0, 0.0}};
        (void)carrier.graph.add_dense_factor(scope, values);
    }
    return carrier;
}

std::array<qubit::QComplex, 2> values_for(
    std::size_t worker,
    std::size_t variable) {
    const double delta =
        1e-5 * static_cast<double>(1U + ((worker + 3U * variable) % 23U));
    return {
        qubit::QComplex{0.5 + delta, 0.0},
        qubit::QComplex{0.5 - delta, 0.0},
    };
}

}  // namespace

int main() {
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t workers = 16U;

    Carrier carrier = make_carrier();
    const auto compile_begin = Clock::now();
    const qubit::ExactFactorPlan plan(carrier.graph);
    const auto compile_end = Clock::now();

    const auto workspace_begin = Clock::now();
    std::vector<qubit::ExactFactorWorkspace> workspaces;
    workspaces.reserve(workers);
    for (std::size_t worker = 0U; worker < workers; ++worker) {
        workspaces.push_back(plan.workspace(carrier.mutable_factors));
        for (std::size_t variable = 0U;
             variable < carrier.mutable_factors.size();
             ++variable) {
            const auto values = values_for(worker, variable);
            plan.bind_dense_factor(
                workspaces.back(), carrier.mutable_factors[variable], values);
        }
    }
    const auto workspace_end = Clock::now();

    std::vector<qubit::QComplex> serial(workers);
    const auto serial_begin = Clock::now();
    for (std::size_t worker = 0U; worker < workers; ++worker) {
        serial[worker] = plan.bound_partition(workspaces[worker]);
    }
    const auto serial_end = Clock::now();

    std::vector<qubit::QComplex> parallel(workers);
    std::array<std::thread, workers> threads;
    std::atomic<bool> failed{false};
    const auto parallel_begin = Clock::now();
    for (std::size_t worker = 0U; worker < workers; ++worker) {
        threads[worker] = std::thread([&, worker]() {
            try {
                parallel[worker] = plan.bound_partition(workspaces[worker]);
            } catch (...) {
                failed.store(true, std::memory_order_relaxed);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    const auto parallel_end = Clock::now();

    double max_parallel_error = 0.0;
    double guard = 0.0;
    std::size_t total_workspace_bytes = 0U;
    std::size_t total_binding_bytes = 0U;
    std::size_t total_rebinds = 0U;
    for (std::size_t worker = 0U; worker < workers; ++worker) {
        max_parallel_error = std::max(
            max_parallel_error,
            std::hypot(
                serial[worker].re - parallel[worker].re,
                serial[worker].im - parallel[worker].im));
        guard += parallel[worker].norm2();
        total_workspace_bytes += workspaces[worker].estimated_bytes();
        total_binding_bytes += workspaces[worker].binding_estimated_bytes();
        total_rebinds += workspaces[worker].bound_rebind_count();
    }

    const std::size_t plan_bytes = plan.estimated_bytes();
    const std::size_t per_workspace_bytes =
        workspaces.front().estimated_bytes() + workspaces.front().binding_estimated_bytes();
    const std::size_t plan_copy_equivalent_batch =
        workers * (plan_bytes + workspaces.front().estimated_bytes());
    const std::size_t shared_plan_batch =
        plan_bytes + total_workspace_bytes + total_binding_bytes;
    const double memory_reduction_ratio =
        static_cast<double>(plan_copy_equivalent_batch) /
        static_cast<double>(shared_plan_batch);

    const double compile_ms =
        std::chrono::duration<double, std::milli>(compile_end - compile_begin).count();
    const double workspace_ms =
        std::chrono::duration<double, std::milli>(workspace_end - workspace_begin).count();
    const double serial_ms =
        std::chrono::duration<double, std::milli>(serial_end - serial_begin).count();
    const double parallel_ms =
        std::chrono::duration<double, std::milli>(parallel_end - parallel_begin).count();
    const double parallel_ratio = serial_ms / parallel_ms;

    const auto stats = plan.stats();
    std::cout << std::setprecision(17)
              << "binding_variables=" << stats.variable_count << '\n'
              << "binding_source_factors=" << stats.source_factors << '\n'
              << "binding_mutable_factors=" << carrier.mutable_factors.size() << '\n'
              << "binding_peak_union_variables=" << stats.peak_union_variables << '\n'
              << "binding_peak_factor_entries=" << stats.peak_factor_entries << '\n'
              << "binding_plan_bytes=" << plan_bytes << '\n'
              << "binding_worker_count=" << workers << '\n'
              << "binding_per_workspace_bytes=" << per_workspace_bytes << '\n'
              << "binding_total_workspace_bytes=" << total_workspace_bytes << '\n'
              << "binding_total_override_bytes=" << total_binding_bytes << '\n'
              << "binding_shared_plan_batch_bytes=" << shared_plan_batch << '\n'
              << "binding_plan_copy_equivalent_batch_bytes=" << plan_copy_equivalent_batch << '\n'
              << "binding_memory_reduction_ratio=" << memory_reduction_ratio << '\n'
              << "binding_total_rebinds=" << total_rebinds << '\n'
              << "binding_plan_rebind_count=" << plan.rebind_count() << '\n'
              << "binding_compile_ms=" << compile_ms << '\n'
              << "binding_workspace_prepare_ms=" << workspace_ms << '\n'
              << "binding_serial_ms=" << serial_ms << '\n'
              << "binding_parallel_ms=" << parallel_ms << '\n'
              << "binding_parallel_ratio=" << parallel_ratio << '\n'
              << "binding_max_parallel_error=" << max_parallel_error << '\n'
              << "binding_guard=" << guard << '\n'
              << "same_plan_parallel_execution=1\n"
              << "plan_source_mutation=0\n"
              << "dense_global_state_materialized=0\n";

    return !failed.load(std::memory_order_relaxed) &&
            stats.variable_count == 4096U &&
            stats.source_factors == 8191U &&
            carrier.mutable_factors.size() == 4096U &&
            stats.peak_union_variables <= 3U &&
            stats.peak_factor_entries <= 8U &&
            plan.rebind_count() == 0U &&
            total_rebinds == workers * 4096U &&
            per_workspace_bytes < 4U * 1024U * 1024U &&
            shared_plan_batch < plan_copy_equivalent_batch &&
            memory_reduction_ratio > 2.0 &&
            max_parallel_error <= 1e-12 &&
            std::isfinite(guard) && guard > 0.0 &&
            compile_ms > 0.0 && compile_ms < 10000.0 &&
            workspace_ms > 0.0 && workspace_ms < 10000.0 &&
            serial_ms > 0.0 && serial_ms < 10000.0 &&
            parallel_ms > 0.0 && parallel_ms < 10000.0
        ? 0
        : 1;
}

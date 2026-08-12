#include "qubit/qfactor_tree.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::ExactFactorGraph;
using qubit::ExactFactorPlan;
using qubit::ExactFactorTreePlan;
using qubit::FactorVariableId;
using qubit::QComplex;

constexpr std::size_t variable_count = 16'383U;

[[nodiscard]] std::array<QComplex, 4> factor_values(std::size_t child) {
    const double offset = 0.0002 * static_cast<double>(child % 17U);
    return {{
        {0.55 + offset, 0.0},
        {0.45 + offset, 0.0},
        {0.45 - offset, 0.0},
        {0.55 - offset, 0.0},
    }};
}

[[nodiscard]] ExactFactorGraph make_graph() {
    qubit::ExactFactorConfig config;
    config.max_variables = variable_count + 1U;
    config.max_factors = variable_count + 1U;
    config.max_factor_entries = 16U;
    ExactFactorGraph graph(config);
    for (std::size_t variable = 0U; variable < variable_count; ++variable) {
        static_cast<void>(graph.add_variable(2U));
    }
    for (std::size_t child = 1U; child < variable_count; ++child) {
        const FactorVariableId parent = static_cast<FactorVariableId>((child - 1U) / 2U);
        const FactorVariableId child_id = static_cast<FactorVariableId>(child);
        const std::array<FactorVariableId, 2> scope{{parent, child_id}};
        const auto values = factor_values(child);
        static_cast<void>(graph.add_dense_factor(scope, values));
    }
    return graph;
}

class BinaryTreeControl {
public:
    BinaryTreeControl() : messages_(variable_count) {}

    [[nodiscard]] std::array<double, 2> query() {
        for (std::size_t child = variable_count - 1U; child > 0U; --child) {
            const auto values = factor_values(child);
            const std::size_t left = 2U * child + 1U;
            const std::size_t right = left + 1U;
            const double child_zero =
                (left < variable_count ? messages_[left][0] : 1.0) *
                (right < variable_count ? messages_[right][0] : 1.0);
            const double child_one =
                (left < variable_count ? messages_[left][1] : 1.0) *
                (right < variable_count ? messages_[right][1] : 1.0);
            messages_[child][0] = values[0].re * child_zero + values[2].re * child_one;
            messages_[child][1] = values[1].re * child_zero + values[3].re * child_one;
        }
        const double left_zero = messages_[1U][0];
        const double left_one = messages_[1U][1];
        const double right_zero = messages_[2U][0];
        const double right_one = messages_[2U][1];
        return {{left_zero * right_zero, left_one * right_one}};
    }

private:
    std::vector<std::array<double, 2>> messages_{};
};

template <class Function>
[[nodiscard]] double median_ms(Function&& function, std::size_t repeats) {
    std::vector<double> samples;
    samples.reserve(repeats);
    for (std::size_t repeat = 0U; repeat < repeats; ++repeat) {
        const auto start = Clock::now();
        function();
        const auto finish = Clock::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(finish - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);
    ExactFactorGraph graph = make_graph();
    const std::array<FactorVariableId, 1> retained{{0U}};

    std::optional<ExactFactorTreePlan> tree;
    const double tree_compile_ms = median_ms([&] {
        tree.emplace(graph, retained);
    }, 5U);
    std::optional<ExactFactorPlan> generic;
    const double generic_compile_ms = median_ms([&] {
        generic.emplace(graph, retained);
    }, 5U);

    auto tree_workspace = tree->workspace();
    auto generic_workspace = generic->workspace();
    std::array<QComplex, 2> tree_output{};
    std::array<QComplex, 2> generic_output{};
    BinaryTreeControl control;
    std::array<double, 2> control_output{};

    const double tree_query_ms = median_ms([&] {
        tree->evaluate(tree_output, tree_workspace);
    }, 11U);
    const double generic_query_ms = median_ms([&] {
        generic->evaluate(generic_output, generic_workspace);
    }, 11U);
    const double control_query_ms = median_ms([&] {
        control_output = control.query();
    }, 11U);

    const double tree_generic_error = std::max(
        std::abs(tree_output[0].re - generic_output[0].re),
        std::abs(tree_output[1].re - generic_output[1].re));
    const double tree_control_error = std::max(
        std::abs(tree_output[0].re - control_output[0]),
        std::abs(tree_output[1].re - control_output[1]));
    const double generic_control_error = std::max(
        std::abs(generic_output[0].re - control_output[0]),
        std::abs(generic_output[1].re - control_output[1]));
    const double imaginary_error = std::max(
        std::abs(tree_output[0].im),
        std::abs(tree_output[1].im));
    if (tree_generic_error > 2e-11 ||
        tree_control_error > 2e-11 ||
        generic_control_error > 2e-11 ||
        imaginary_error > 2e-11) {
        std::cerr << "factor tree matched-control equality failed\n";
        return 1;
    }

    const QComplex tree_partition = tree_output[0] + tree_output[1];
    const QComplex generic_partition = generic_output[0] + generic_output[1];
    if (std::abs(tree_partition.re - generic_partition.re) > 2e-11 ||
        std::abs(tree_partition.im - generic_partition.im) > 2e-11) {
        std::cerr << "factor tree partition equality failed\n";
        return 2;
    }

    std::cout << "factor_tree_variables=" << variable_count << '\n';
    std::cout << "factor_tree_factors=" << graph.factor_count() << '\n';
    std::cout << "factor_tree_incidence_edges=" << tree->stats().incidence_edges << '\n';
    std::cout << "factor_tree_messages=" << tree->stats().message_count << '\n';
    std::cout << "factor_tree_message_entries=" << tree->stats().message_entries << '\n';
    std::cout << "factor_tree_compile_ms=" << tree_compile_ms << '\n';
    std::cout << "factor_tree_generic_compile_ms=" << generic_compile_ms << '\n';
    std::cout << "factor_tree_compile_speedup=" << generic_compile_ms / tree_compile_ms << '\n';
    std::cout << "factor_tree_query_ms=" << tree_query_ms << '\n';
    std::cout << "factor_tree_generic_query_ms=" << generic_query_ms << '\n';
    std::cout << "factor_tree_query_speedup=" << generic_query_ms / tree_query_ms << '\n';
    std::cout << "factor_tree_specialized_control_ms=" << control_query_ms << '\n';
    std::cout << "factor_tree_specialized_over_tree_ratio=" << control_query_ms / tree_query_ms << '\n';
    std::cout << "factor_tree_plan_bytes=" << tree->estimated_bytes() << '\n';
    std::cout << "factor_tree_workspace_bytes=" << tree_workspace.estimated_bytes() << '\n';
    std::cout << "factor_tree_generic_plan_bytes=" << generic->estimated_bytes() << '\n';
    std::cout << "factor_tree_generic_workspace_bytes=" << generic_workspace.estimated_bytes() << '\n';
    std::cout << "factor_tree_tree_generic_error=" << tree_generic_error << '\n';
    std::cout << "factor_tree_tree_control_error=" << tree_control_error << '\n';
    std::cout << "factor_tree_generic_control_error=" << generic_control_error << '\n';
    std::cout << "factor_tree_partition_re=" << tree_partition.re << '\n';
    return 0;
}

#include "qubit/qqtt_operator.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace qubit;

int main() {
    constexpr std::size_t logical_bits = 4096U;
    std::vector<double> weights(logical_bits);
    for (std::size_t index = 0U; index < logical_bits; ++index) {
        weights[index] = 1.0 + 0.125 * static_cast<double>(index % 7U);
    }

    QTTConfig state_config;
    state_config.max_rank = 16U;
    state_config.max_core_scalars = 1024U;
    state_config.max_total_scalars = 1U << 20U;

    QTTOperatorConfig operator_config;
    operator_config.max_rank = 16U;
    operator_config.max_core_scalars = 1024U;
    operator_config.max_total_scalars = 1U << 20U;
    operator_config.max_apply_rank = 16U;
    operator_config.max_apply_core_scalars = 4096U;
    operator_config.max_apply_total_scalars = 1U << 20U;

    const ExactQTTFunction state = ExactQTTFunction::hamming_weight(logical_bits, state_config);

    const auto build_start = std::chrono::steady_clock::now();
    const ExactQTTOperator laplacian =
        ExactQTTOperator::weighted_hypercube_laplacian(weights, operator_config);
    const auto build_end = std::chrono::steady_clock::now();

    const auto apply_start = std::chrono::steady_clock::now();
    const ExactQTTFunction result = laplacian.apply(state);
    const auto apply_end = std::chrono::steady_clock::now();

    std::vector<std::uint8_t> selected(logical_bits);
    for (std::size_t index = 0U; index < logical_bits; ++index) {
        selected[index] = static_cast<std::uint8_t>(((index * 17U + 3U) % 11U) < 5U ? 1U : 0U);
    }

    const auto query_start = std::chrono::steady_clock::now();
    const QComplex observed = result.value_bits(selected);
    const auto query_end = std::chrono::steady_clock::now();

    const auto direct_start = std::chrono::steady_clock::now();
    double expected = 0.0;
    for (std::size_t index = 0U; index < logical_bits; ++index) {
        expected += weights[index] * (selected[index] == 0U ? 1.0 : -1.0);
    }
    const auto direct_end = std::chrono::steady_clock::now();

    const double error = (observed - QComplex{expected}).magnitude();
    const auto milliseconds = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    std::cout << std::setprecision(17);
    std::cout << "qtt_operator_bits=" << logical_bits << '\n';
    std::cout << "qtt_operator_logical_domain_exponent=" << logical_bits << '\n';
    std::cout << "qtt_operator_rank=" << laplacian.stats().maximum_rank << '\n';
    std::cout << "qtt_operator_descriptor_scalars=" << laplacian.stats().descriptor_scalars << '\n';
    std::cout << "qtt_operator_output_rank=" << result.stats().maximum_rank << '\n';
    std::cout << "qtt_operator_output_descriptor_scalars=" << result.stats().descriptor_scalars << '\n';
    std::cout << "qtt_operator_build_ms=" << milliseconds(build_start, build_end) << '\n';
    std::cout << "qtt_operator_apply_ms=" << milliseconds(apply_start, apply_end) << '\n';
    std::cout << "qtt_operator_query_ms=" << milliseconds(query_start, query_end) << '\n';
    std::cout << "qtt_operator_direct_closed_form_query_ms="
              << milliseconds(direct_start, direct_end) << '\n';
    std::cout << "qtt_operator_selected_error=" << error << '\n';
    return error <= 1e-9 ? 0 : 1;
}

#include "qubit/qqtt_split_step.hpp"

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
    constexpr double mu = 0.8;
    constexpr double alpha = 1.25;
    constexpr double kinetic = 0.3;
    constexpr double nonlinear = 0.15;
    constexpr double background = 0.6;
    constexpr double dt = 1.0e-3;

    std::vector<double> weights(logical_bits);
    std::vector<std::size_t> support;
    for (std::size_t i = 0U; i < logical_bits; ++i) {
        weights[i] = 0.25 + 0.03125 * static_cast<double>(i % 7U);
        if ((i % 17U) == 3U) {
            support.push_back(i);
        }
    }

    ComplexWalshConfig config;
    config.max_terms = 16U;
    config.max_support_entries = 1U << 18U;
    config.max_products = 4096U;
    const ExactComplexWalshField wave = ExactComplexWalshField::from_terms(
        logical_bits,
        {
            ComplexWalshTerm{QComplex{0.9, 0.05}, {}},
            ComplexWalshTerm{QComplex{0.1, -0.03}, support},
        },
        config);

    WalshFieldConfig real_config;
    real_config.max_terms = 16U;
    real_config.max_support_entries = 1U << 18U;
    const ExactQTTSchrodingerPoissonSplitStep integrator(
        weights, mu, alpha, kinetic, nonlinear, config);

    const auto forward_start = std::chrono::steady_clock::now();
    const QTTSplitStepResult forward = integrator.step(wave, dt, background, real_config);
    const auto forward_end = std::chrono::steady_clock::now();

    const auto reverse_start = std::chrono::steady_clock::now();
    const QTTSplitStepResult backward = integrator.step(
        forward.wave, -dt, background, real_config);
    const auto reverse_end = std::chrono::steady_clock::now();
    const ExactComplexWalshField reversal_error = backward.wave.add(wave.scaled(QComplex{-1.0}));
    const double reversal_l2 = reversal_error.mean_inner_product(reversal_error).re;

    QTTConfig qtt_config;
    qtt_config.max_rank = 16U;
    qtt_config.max_core_scalars = 1024U;
    qtt_config.max_total_scalars = 1U << 20U;
    const auto compile_start = std::chrono::steady_clock::now();
    const ExactQTTFunction qtt_wave = forward.wave.to_qtt(qtt_config);
    const auto compile_end = std::chrono::steady_clock::now();

    std::vector<std::uint8_t> bits(logical_bits);
    for (std::size_t i = 0U; i < logical_bits; ++i) {
        bits[i] = static_cast<std::uint8_t>(((i * 31U + 9U) % 37U) < 18U ? 1U : 0U);
    }
    const auto query_start = std::chrono::steady_clock::now();
    const QComplex exact_value = forward.wave.value_bits(bits);
    const QComplex qtt_value = qtt_wave.value_bits(bits);
    const auto query_end = std::chrono::steady_clock::now();
    const double selected_error = (exact_value - qtt_value).magnitude();

    const auto milliseconds = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    std::cout << std::setprecision(17);
    std::cout << "qtt_split_bits=" << logical_bits << '\n';
    std::cout << "qtt_split_logical_domain_exponent=" << logical_bits << '\n';
    std::cout << "qtt_split_input_terms=" << forward.stats.input_terms << '\n';
    std::cout << "qtt_split_density_terms=" << forward.stats.density_terms << '\n';
    std::cout << "qtt_split_potential_terms=" << forward.stats.potential_terms << '\n';
    std::cout << "qtt_split_phase_terms=" << forward.stats.phase_terms << '\n';
    std::cout << "qtt_split_output_terms=" << forward.stats.output_terms << '\n';
    std::cout << "qtt_split_output_qtt_rank=" << qtt_wave.stats().maximum_rank << '\n';
    std::cout << "qtt_split_output_qtt_descriptor_scalars=" << qtt_wave.stats().descriptor_scalars << '\n';
    std::cout << "qtt_split_norm_error=" << forward.stats.norm_error << '\n';
    std::cout << "qtt_split_reverse_norm_error=" << backward.stats.norm_error << '\n';
    std::cout << "qtt_split_reversal_l2=" << reversal_l2 << '\n';
    std::cout << "qtt_split_selected_error=" << selected_error << '\n';
    std::cout << "qtt_split_forward_ms=" << milliseconds(forward_start, forward_end) << '\n';
    std::cout << "qtt_split_reverse_ms=" << milliseconds(reverse_start, reverse_end) << '\n';
    std::cout << "qtt_split_compile_ms=" << milliseconds(compile_start, compile_end) << '\n';
    std::cout << "qtt_split_query_ms=" << milliseconds(query_start, query_end) << '\n';

    return forward.stats.norm_error <= 1e-11 && backward.stats.norm_error <= 1e-11 &&
                   reversal_l2 <= 1e-18 && selected_error <= 1e-9
               ? 0
               : 1;
}

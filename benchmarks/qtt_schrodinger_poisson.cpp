#include "qubit/qqtt_sp_invariants.hpp"

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
    constexpr double background = 0.4;

    std::vector<double> weights(logical_bits);
    for (std::size_t i = 0U; i < logical_bits; ++i) {
        weights[i] = 0.25 + 0.03125 * static_cast<double>(i % 7U);
    }

    std::vector<std::size_t> support_a;
    std::vector<std::size_t> support_b;
    for (std::size_t i = 0U; i < logical_bits; ++i) {
        if ((i % 11U) == 0U) {
            support_a.push_back(i);
        }
        if ((i % 17U) == 3U) {
            support_b.push_back(i);
        }
    }

    ComplexWalshConfig wave_config;
    wave_config.max_terms = 16U;
    wave_config.max_support_entries = 1U << 18U;
    wave_config.max_products = 4096U;
    const ExactComplexWalshField wave = ExactComplexWalshField::from_terms(
        logical_bits,
        {
            ComplexWalshTerm{QComplex{0.9, 0.05}, {}},
            ComplexWalshTerm{QComplex{0.12, -0.04}, support_a},
            ComplexWalshTerm{QComplex{-0.08, 0.06}, support_b},
        },
        wave_config);

    WalshFieldConfig real_config;
    real_config.max_terms = 16U;
    real_config.max_support_entries = 1U << 18U;
    const ExactQTTSchrodingerPoisson solver(weights, mu, alpha, kinetic, nonlinear);

    const auto solve_start = std::chrono::steady_clock::now();
    const QTTSchrodingerPoissonResult result = solver.evaluate(wave, background, real_config);
    const auto solve_end = std::chrono::steady_clock::now();

    const ExactWalshField residual = solver.poisson().residual(result.source, result.potential);
    const QTTSchrodingerPoissonDiagnostics diagnostics =
        diagnose_schrodinger_poisson(wave, result);

    QTTConfig qtt_config;
    qtt_config.max_rank = 16U;
    qtt_config.max_core_scalars = 1024U;
    qtt_config.max_total_scalars = 1U << 20U;
    const auto compile_start = std::chrono::steady_clock::now();
    const ExactQTTFunction rhs_qtt = result.rhs.to_qtt(qtt_config);
    const auto compile_end = std::chrono::steady_clock::now();

    std::vector<std::uint8_t> bits(logical_bits);
    for (std::size_t i = 0U; i < logical_bits; ++i) {
        bits[i] = static_cast<std::uint8_t>(((i * 29U + 5U) % 31U) < 15U ? 1U : 0U);
    }
    const auto query_start = std::chrono::steady_clock::now();
    const QComplex spectral_rhs = result.rhs.value_bits(bits);
    const QComplex qtt_rhs = rhs_qtt.value_bits(bits);
    const auto query_end = std::chrono::steady_clock::now();
    const double error = (spectral_rhs - qtt_rhs).magnitude();

    const auto milliseconds = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    std::cout << std::setprecision(17);
    std::cout << "qtt_sp_bits=" << logical_bits << '\n';
    std::cout << "qtt_sp_logical_domain_exponent=" << logical_bits << '\n';
    std::cout << "qtt_sp_wave_terms=" << result.stats.wave_terms << '\n';
    std::cout << "qtt_sp_density_terms=" << result.stats.density_terms << '\n';
    std::cout << "qtt_sp_potential_terms=" << result.stats.potential_terms << '\n';
    std::cout << "qtt_sp_hamiltonian_terms=" << result.stats.hamiltonian_terms << '\n';
    std::cout << "qtt_sp_rhs_qtt_rank=" << rhs_qtt.stats().maximum_rank << '\n';
    std::cout << "qtt_sp_rhs_qtt_descriptor_scalars=" << rhs_qtt.stats().descriptor_scalars << '\n';
    std::cout << "qtt_sp_inverse_poisson_norm_bound=" << result.stats.inverse_poisson_norm_bound << '\n';
    std::cout << "qtt_sp_poisson_condition_bound=" << result.stats.poisson_condition_bound << '\n';
    std::cout << "qtt_sp_wave_mean_norm_squared=" << diagnostics.wave_mean_norm_squared << '\n';
    std::cout << "qtt_sp_hamiltonian_expectation_imag=" << diagnostics.hamiltonian_expectation.im << '\n';
    std::cout << "qtt_sp_norm_rate=" << diagnostics.norm_rate << '\n';
    std::cout << "qtt_sp_poisson_residual_max=" << residual.maximum_absolute_coefficient() << '\n';
    std::cout << "qtt_sp_selected_error=" << error << '\n';
    std::cout << "qtt_sp_solve_ms=" << milliseconds(solve_start, solve_end) << '\n';
    std::cout << "qtt_sp_compile_ms=" << milliseconds(compile_start, compile_end) << '\n';
    std::cout << "qtt_sp_query_ms=" << milliseconds(query_start, query_end) << '\n';

    return residual.maximum_absolute_coefficient() <= 1e-11 && error <= 1e-9 &&
                   diagnostics.wave_mean_norm_squared > 0.0 &&
                   std::abs(diagnostics.hamiltonian_expectation.im) <= 1e-10 &&
                   std::abs(diagnostics.norm_rate) <= 1e-10
               ? 0
               : 1;
}

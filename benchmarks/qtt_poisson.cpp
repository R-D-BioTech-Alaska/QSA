#include "qubit/qqtt_poisson.hpp"

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
    constexpr std::size_t mode_count = 8U;
    constexpr double mu = 0.75;
    constexpr double alpha = 1.5;

    std::vector<double> weights(logical_bits);
    for (std::size_t i = 0U; i < logical_bits; ++i) {
        weights[i] = 0.25 + 0.03125 * static_cast<double>(i % 9U);
    }

    std::vector<WalshFieldTerm> terms;
    terms.reserve(mode_count);
    for (std::size_t mode = 0U; mode < mode_count; ++mode) {
        WalshFieldTerm term;
        term.coefficient = (mode % 2U == 0U ? 1.0 : -1.0) / static_cast<double>(mode + 1U);
        const std::size_t stride = 11U + 2U * mode;
        for (std::size_t bit = mode; bit < logical_bits; bit += stride) {
            term.active_bits.push_back(bit);
        }
        terms.push_back(std::move(term));
    }

    WalshFieldConfig field_config;
    field_config.max_terms = 16U;
    field_config.max_support_entries = 1U << 18U;
    const ExactWalshField source = ExactWalshField::from_terms(
        logical_bits, std::move(terms), field_config);
    const ExactQTTRegularizedPoisson poisson(weights, mu, alpha);

    const auto solve_start = std::chrono::steady_clock::now();
    const ExactWalshField potential = poisson.solve(source);
    const auto solve_end = std::chrono::steady_clock::now();

    const ExactWalshField residual = poisson.residual(source, potential);

    QTTConfig qtt_config;
    qtt_config.max_rank = 16U;
    qtt_config.max_core_scalars = 1024U;
    qtt_config.max_total_scalars = 1U << 20U;
    const auto compile_start = std::chrono::steady_clock::now();
    const ExactQTTFunction qtt_potential = potential.to_qtt(qtt_config);
    const auto compile_end = std::chrono::steady_clock::now();

    std::vector<std::uint8_t> bits(logical_bits);
    for (std::size_t i = 0U; i < logical_bits; ++i) {
        bits[i] = static_cast<std::uint8_t>(((i * 19U + 7U) % 23U) < 11U ? 1U : 0U);
    }
    const auto query_start = std::chrono::steady_clock::now();
    const double spectral_value = potential.value_bits(bits);
    const QComplex qtt_value = qtt_potential.value_bits(bits);
    const auto query_end = std::chrono::steady_clock::now();

    const double query_error = std::abs(qtt_value.re - spectral_value) + std::abs(qtt_value.im);
    const auto stats = poisson.stats(source);
    const auto milliseconds = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    std::cout << std::setprecision(17);
    std::cout << "qtt_poisson_bits=" << logical_bits << '\n';
    std::cout << "qtt_poisson_logical_domain_exponent=" << logical_bits << '\n';
    std::cout << "qtt_poisson_source_terms=" << source.term_count() << '\n';
    std::cout << "qtt_poisson_source_support_entries=" << source.support_entries() << '\n';
    std::cout << "qtt_poisson_qtt_rank=" << qtt_potential.stats().maximum_rank << '\n';
    std::cout << "qtt_poisson_qtt_descriptor_scalars=" << qtt_potential.stats().descriptor_scalars << '\n';
    std::cout << "qtt_poisson_inverse_norm_bound=" << stats.inverse_norm_bound << '\n';
    std::cout << "qtt_poisson_condition_bound=" << stats.condition_bound << '\n';
    std::cout << "qtt_poisson_residual_max=" << residual.maximum_absolute_coefficient() << '\n';
    std::cout << "qtt_poisson_selected_error=" << query_error << '\n';
    std::cout << "qtt_poisson_solve_ms=" << milliseconds(solve_start, solve_end) << '\n';
    std::cout << "qtt_poisson_compile_ms=" << milliseconds(compile_start, compile_end) << '\n';
    std::cout << "qtt_poisson_query_ms=" << milliseconds(query_start, query_end) << '\n';

    return residual.maximum_absolute_coefficient() <= 1e-11 && query_error <= 1e-9 ? 0 : 1;
}

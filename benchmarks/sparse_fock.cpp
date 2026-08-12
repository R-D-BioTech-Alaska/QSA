#include "qubit/qfock.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace qubit;

int main() {
    constexpr std::size_t modes = 4096U;
    constexpr std::size_t particles = 4096U;
    constexpr std::size_t support_terms = 8U;

    SparseFockConfig config;
    config.max_modes = 8192U;
    config.max_particles = 8192U;
    config.max_terms = 1024U;
    config.max_occupied_entries = 1U << 18U;
    config.max_branch_products = 1U << 18U;

    std::vector<SparseFockTerm> terms;
    terms.reserve(support_terms);
    for (std::size_t i = 0U; i < support_terms; ++i) {
        const std::size_t base = 8U * i;
        terms.push_back(SparseFockTerm{
            {{base, 2048U}, {base + 1U, 1024U}, {base + 2U, 1024U}},
            QComplex{1.0 / std::sqrt(static_cast<double>(support_terms))},
        });
    }
    const ExactSparseFockState state = ExactSparseFockState::from_terms(
        modes, std::move(terms), config);

    std::vector<FockHoppingTerm> hopping;
    for (std::size_t i = 0U; i < support_terms; ++i) {
        const std::size_t base = 8U * i;
        hopping.push_back(FockHoppingTerm{base + 3U, base, QComplex{-0.125}});
        hopping.push_back(FockHoppingTerm{base + 4U, base + 1U, QComplex{0.0625}});
    }

    const auto apply_start = std::chrono::steady_clock::now();
    const ExactSparseFockState output = state.apply_hopping(hopping);
    const auto apply_end = std::chrono::steady_clock::now();
    const auto query_start = std::chrono::steady_clock::now();
    const double mean0 = state.mean_number(0U);
    const double norm = output.norm_squared();
    const auto query_end = std::chrono::steady_clock::now();

    const double sector_log2 = state.sector_log2_dimension();
    const auto milliseconds = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    std::cout << std::setprecision(17);
    std::cout << "fock_modes=" << modes << '\n';
    std::cout << "fock_particles=" << particles << '\n';
    std::cout << "fock_sector_log2_dimension=" << sector_log2 << '\n';
    std::cout << "fock_input_terms=" << state.stats().terms << '\n';
    std::cout << "fock_input_occupied_entries=" << state.stats().occupied_entries << '\n';
    std::cout << "fock_output_terms=" << output.stats().terms << '\n';
    std::cout << "fock_output_occupied_entries=" << output.stats().occupied_entries << '\n';
    std::cout << "fock_output_fixed_particles=" << output.fixed_particle_number().value_or(0U) << '\n';
    std::cout << "fock_mean_number_mode0=" << mean0 << '\n';
    std::cout << "fock_output_norm_squared=" << norm << '\n';
    std::cout << "fock_apply_ms=" << milliseconds(apply_start, apply_end) << '\n';
    std::cout << "fock_query_ms=" << milliseconds(query_start, query_end) << '\n';

    return sector_log2 > 8000.0 && output.fixed_particle_number().value_or(0U) == particles &&
                   output.stats().terms <= support_terms * hopping.size() &&
                   output.stats().occupied_entries < 4096U && std::isfinite(norm) && norm > 0.0
               ? 0
               : 1;
}

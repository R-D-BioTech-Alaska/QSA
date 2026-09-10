#include "qubit/qcoherent.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace qubit;

int main() {
    constexpr std::size_t modes = 10000U;
    constexpr std::size_t cutoff = 4U;
    std::vector<QComplex> amplitudes(modes);
    for (std::size_t mode = 0U; mode < modes; ++mode) {
        amplitudes[mode] = QComplex::from_polar(
            0.01,
            0.013 * static_cast<double>(mode % 17U));
    }

    CoherentSuperpositionConfig config;
    config.max_modes = 20000U;
    config.max_terms = 8U;
    config.max_complex_entries = 1U << 20U;

    const auto setup_begin = std::chrono::steady_clock::now();
    BoundedCoherentSuperposition state = BoundedCoherentSuperposition::even_cat(amplitudes, config);
    state.phase_shift(31U, 0.37);
    state.beam_splitter(31U, 32U, 0.29, -0.11);
    const auto setup_end = std::chrono::steady_clock::now();

    const auto query_begin = std::chrono::steady_clock::now();
    const double norm = state.norm_squared();
    const double selected_number = state.mean_number(4321U);
    const double selected_x = state.mean_x(4321U);
    const auto query_end = std::chrono::steady_clock::now();

    const double explicit_fock_log2_entries =
        static_cast<double>(modes) * std::log2(static_cast<double>(cutoff));
    const double structured_log2_entries =
        std::log2(static_cast<double>(state.stats().stored_complex_entries));

    const auto milliseconds = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    std::cout << std::setprecision(17);
    std::cout << "coherent_modes=" << modes << '\n';
    std::cout << "coherent_terms=" << state.stats().terms << '\n';
    std::cout << "coherent_stored_complex_entries=" << state.stats().stored_complex_entries << '\n';
    std::cout << "coherent_comparison_fock_cutoff=" << cutoff << '\n';
    std::cout << "coherent_explicit_fock_log2_entries=" << explicit_fock_log2_entries << '\n';
    std::cout << "coherent_structured_log2_entries=" << structured_log2_entries << '\n';
    std::cout << "coherent_log2_entry_gap=" << explicit_fock_log2_entries - structured_log2_entries << '\n';
    std::cout << "coherent_norm_squared=" << norm << '\n';
    std::cout << "coherent_selected_mean_number=" << selected_number << '\n';
    std::cout << "coherent_selected_mean_x=" << selected_x << '\n';
    std::cout << "coherent_setup_ms=" << milliseconds(setup_begin, setup_end) << '\n';
    std::cout << "coherent_query_ms=" << milliseconds(query_begin, query_end) << '\n';

    return state.stats().terms == 2U &&
                   state.stats().stored_complex_entries == 20002U &&
                   explicit_fock_log2_entries == 20000.0 &&
                   explicit_fock_log2_entries - structured_log2_entries > 19985.0 &&
                   std::abs(norm - 1.0) < 1e-9 &&
                   std::isfinite(selected_number) && std::isfinite(selected_x)
               ? 0
               : 1;
}

#include "qubit/qcoherent.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

using namespace qubit;

namespace {

bool close(double left, double right, double tolerance = 2e-11) {
    return std::abs(left - right) <= tolerance * (1.0 + std::max(std::abs(left), std::abs(right)));
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw QStateError(message);
    }
}

void coherent_control() {
    const QComplex alpha{0.37, -0.21};
    BoundedCoherentSuperposition state = BoundedCoherentSuperposition::from_terms(
        1U,
        {CoherentTerm{QComplex{1.0}, {alpha}}});
    require(close(state.norm_squared(), 1.0), "single coherent norm mismatch");
    require(close(state.mean_number(0U), alpha.norm2()), "single coherent number mismatch");
    require(close(state.mean_x(0U), std::sqrt(2.0) * alpha.re), "single coherent mean-x mismatch");
}

void even_cat_control() {
    constexpr double alpha = 0.73;
    const std::vector<QComplex> amplitudes{QComplex{alpha}};
    BoundedCoherentSuperposition cat = BoundedCoherentSuperposition::even_cat(amplitudes);
    const double expected_number = alpha * alpha * std::tanh(alpha * alpha);
    require(cat.term_count() == 2U, "even cat rank mismatch");
    require(close(cat.norm_squared(), 1.0), "even cat normalization mismatch");
    require(close(cat.mean_number(0U), expected_number), "even cat number mismatch");
    require(close(cat.mean_x(0U), 0.0), "even cat parity mean-x mismatch");
}

void passive_transform_control() {
    BoundedCoherentSuperposition state = BoundedCoherentSuperposition::from_terms(
        2U,
        {CoherentTerm{QComplex{1.0}, {QComplex{0.4, 0.2}, QComplex{-0.1, 0.3}}}});
    const double before = state.mean_number(0U) + state.mean_number(1U);
    state.phase_shift(0U, 0.71);
    state.beam_splitter(0U, 1U, 0.43, -0.17);
    const double after = state.mean_number(0U) + state.mean_number(1U);
    require(close(before, after), "passive coherent transforms did not preserve total number");
    require(close(state.norm_squared(), 1.0), "passive coherent transforms did not preserve norm");
}

void caps_and_rejection() {
    CoherentSuperpositionConfig config;
    config.max_modes = 16U;
    config.max_terms = 1U;
    config.max_complex_entries = 32U;

    bool rejected = false;
    try {
        const std::vector<QComplex> amplitude{QComplex{0.2}};
        (void)BoundedCoherentSuperposition::even_cat(amplitude, config);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "coherent rank cap did not reject rank-two cat");

    rejected = false;
    try {
        BoundedCoherentSuperposition state = BoundedCoherentSuperposition::from_terms(
            1U,
            {CoherentTerm{QComplex{1.0}, {QComplex{0.2}}}});
        state.kerr(0U, 0.3);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "coherent passive contract accepted Kerr evolution");
}

void large_rank_two_structure() {
    constexpr std::size_t modes = 10000U;
    std::vector<QComplex> amplitudes(modes);
    for (std::size_t mode = 0U; mode < modes; ++mode) {
        amplitudes[mode] = QComplex::from_polar(0.01, 0.013 * static_cast<double>(mode % 17U));
    }
    CoherentSuperpositionConfig config;
    config.max_modes = 20000U;
    config.max_terms = 8U;
    config.max_complex_entries = 1U << 20U;
    BoundedCoherentSuperposition cat = BoundedCoherentSuperposition::even_cat(amplitudes, config);
    require(cat.stats().modes == modes, "large coherent mode count mismatch");
    require(cat.stats().terms == 2U, "large coherent rank mismatch");
    require(cat.stats().stored_complex_entries == 20002U, "large coherent storage mismatch");
    require(close(cat.norm_squared(), 1.0, 2e-10), "large coherent norm mismatch");
    require(std::isfinite(cat.mean_number(4321U)), "large coherent number became non-finite");
}

}  // namespace

int main() {
    coherent_control();
    even_cat_control();
    passive_transform_control();
    caps_and_rejection();
    large_rank_two_structure();
    std::cout << "bounded coherent tests passed\n";
    return 0;
}

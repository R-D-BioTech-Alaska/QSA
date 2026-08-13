#include "qubit/qsector.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace qubit;

namespace {

void dense_phase(std::vector<QComplex>& state, std::size_t mode, double angle) {
    const std::size_t bit = std::size_t{1U} << mode;
    const QComplex phase = QComplex::from_polar(1.0, angle);
    for (std::size_t basis = 0U; basis < state.size(); ++basis) {
        if ((basis & bit) != 0U) {
            state[basis] *= phase;
        }
    }
}

void dense_givens(
    std::vector<QComplex>& state,
    std::size_t left,
    std::size_t right,
    double angle,
    double phase_angle) {
    const std::size_t left_bit = std::size_t{1U} << left;
    const std::size_t right_bit = std::size_t{1U} << right;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const QComplex phase = QComplex::from_polar(1.0, phase_angle);
    const QComplex phase_conjugate = phase.conjugate();
    for (std::size_t basis = 0U; basis < state.size(); ++basis) {
        if ((basis & left_bit) != 0U && (basis & right_bit) == 0U) {
            const std::size_t partner = (basis ^ left_bit) | right_bit;
            const QComplex a = state[basis];
            const QComplex b = state[partner];
            state[basis] = a * cosine - phase_conjugate * b * sine;
            state[partner] = phase * a * sine + b * cosine;
        }
    }
}

double dense_occupation(const std::vector<QComplex>& state, std::size_t mode) {
    const std::size_t bit = std::size_t{1U} << mode;
    double total = 0.0;
    for (std::size_t basis = 0U; basis < state.size(); ++basis) {
        if ((basis & bit) != 0U) {
            total += state[basis].norm2();
        }
    }
    return total;
}

struct Result {
    double sector_seconds{0.0};
    double dense_seconds{0.0};
    double error{0.0};
    std::size_t sector_dimension{0U};
};

Result measure(std::size_t modes) {
    constexpr std::size_t repeats = 3U;
    double sector_best = 1e100;
    double dense_best = 1e100;
    double sector_probability = 0.0;
    double dense_probability = 0.0;
    std::size_t dimension = 0U;

    for (std::size_t repeat = 0U; repeat < repeats; ++repeat) {
        const std::vector<QubitId> occupied{0U, 2U};
        const auto started = std::chrono::steady_clock::now();
        ExactFixedExcitationState state = ExactFixedExcitationState::basis(modes, occupied);
        state.apply_mode_phase(2U, 0.23);
        state.apply_givens(0U, 1U, 0.37, 0.11);
        state.apply_givens(2U, 3U, -0.29, -0.07);
        state.apply_givens(1U, modes - 1U, 0.19, 0.05);
        sector_probability = state.occupation_probability(modes - 1U);
        const auto finished = std::chrono::steady_clock::now();
        sector_best = std::min(
            sector_best, std::chrono::duration<double>(finished - started).count());
        dimension = state.sector_dimension();
    }

    for (std::size_t repeat = 0U; repeat < repeats; ++repeat) {
        const auto started = std::chrono::steady_clock::now();
        std::vector<QComplex> state(std::size_t{1U} << modes, QComplex{});
        state[(std::size_t{1U} << 0U) | (std::size_t{1U} << 2U)] = QComplex{1.0};
        dense_phase(state, 2U, 0.23);
        dense_givens(state, 0U, 1U, 0.37, 0.11);
        dense_givens(state, 2U, 3U, -0.29, -0.07);
        dense_givens(state, 1U, modes - 1U, 0.19, 0.05);
        dense_probability = dense_occupation(state, modes - 1U);
        const auto finished = std::chrono::steady_clock::now();
        dense_best = std::min(
            dense_best, std::chrono::duration<double>(finished - started).count());
    }

    return Result{
        sector_best,
        dense_best,
        std::abs(sector_probability - dense_probability),
        dimension,
    };
}

}  // namespace

int main() {
    std::cout << std::setprecision(17);
    double maximum_error = 0.0;
    double n18_speedup = 0.0;
    for (const std::size_t modes : std::vector<std::size_t>{6U, 8U, 10U, 12U, 14U, 16U, 18U}) {
        const Result result = measure(modes);
        maximum_error = std::max(maximum_error, result.error);
        const double speedup = result.dense_seconds / result.sector_seconds;
        if (modes == 18U) {
            n18_speedup = speedup;
        }
        std::cout << "case_" << modes << "_sector_dimension=" << result.sector_dimension << '\n';
        std::cout << "case_" << modes << "_dense_amplitudes=" << (std::size_t{1U} << modes) << '\n';
        std::cout << "case_" << modes << "_sector_seconds=" << result.sector_seconds << '\n';
        std::cout << "case_" << modes << "_dense_seconds=" << result.dense_seconds << '\n';
        std::cout << "case_" << modes << "_speedup=" << speedup << '\n';
        std::cout << "case_" << modes << "_absolute_error=" << result.error << '\n';
    }

    const std::vector<QubitId> occupied{0U, 1U};
    const auto large_start = std::chrono::steady_clock::now();
    ExactFixedExcitationState large = ExactFixedExcitationState::basis(1024U, occupied);
    for (std::size_t mode = 0U; mode < 16U; ++mode) {
        large.apply_mode_phase(mode, 0.003 * static_cast<double>(mode + 1U));
    }
    for (std::size_t mode = 0U; mode < 16U; ++mode) {
        large.apply_givens(mode, mode + 1U, 0.011 * static_cast<double>(mode + 1U), 0.007);
    }
    const double p0 = large.occupation_probability(0U);
    const double p1 = large.occupation_probability(1U);
    const double p2 = large.occupation_probability(2U);
    const auto large_end = std::chrono::steady_clock::now();

    bool x_rejected = false;
    try {
        large.apply_x(0U);
    } catch (const QStateError&) {
        x_rejected = true;
    }

    std::cout << "maximum_dense_error=" << maximum_error << '\n';
    std::cout << "n18_speedup=" << n18_speedup << '\n';
    std::cout << "large_modes=" << large.mode_count() << '\n';
    std::cout << "large_excitations=" << large.excitation_count() << '\n';
    std::cout << "large_sector_dimension=" << large.sector_dimension() << '\n';
    std::cout << "large_dense_to_sector_ratio_log2=" << large.dense_to_sector_log2_ratio() << '\n';
    std::cout << "large_estimated_bytes=" << large.stats().estimated_bytes << '\n';
    std::cout << "large_scalar_operations=" << large.stats().scalar_operations << '\n';
    std::cout << "large_norm=" << large.norm_squared() << '\n';
    std::cout << "large_p0=" << p0 << '\n';
    std::cout << "large_p1=" << p1 << '\n';
    std::cout << "large_p2=" << p2 << '\n';
    std::cout << "large_seconds="
              << std::chrono::duration<double>(large_end - large_start).count() << '\n';
    std::cout << "large_dense_state_materialized=0\n";
    std::cout << "number_changing_x_rejected=" << (x_rejected ? 1 : 0) << '\n';
    return 0;
}

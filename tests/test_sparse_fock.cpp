#include "qubit/qfock.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

using namespace qubit;

namespace {

bool close(QComplex left, QComplex right, double tolerance = 1e-11) {
    return almost_equal(left, right, tolerance);
}

bool close(double left, double right, double tolerance = 1e-11) {
    return std::abs(left - right) <= tolerance * (1.0 + std::max(std::abs(left), std::abs(right)));
}

template <class Fn>
bool throws_qstate(Fn&& fn) {
    try {
        fn();
    } catch (const QStateError&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    SparseFockConfig config;
    config.max_modes = 32U;
    config.max_particles = 32U;
    config.max_terms = 64U;
    config.max_occupied_entries = 512U;
    config.max_branch_products = 512U;

    const ExactSparseFockState state = ExactSparseFockState::from_terms(
        4U,
        {
            SparseFockTerm{{{0U, 2U}, {2U, 1U}}, QComplex{0.6, 0.2}},
            SparseFockTerm{{{1U, 3U}}, QComplex{-0.1, 0.7}},
        },
        config);
    assert(state.stats().fixed_particle_number);
    assert(state.fixed_particle_number().value() == 3U);
    assert(close(state.sector_log2_dimension(), std::log2(20.0)));
    assert(close(state.norm_squared(), 0.9));
    assert(close(state.mean_number(0U), (0.4 * 2.0) / 0.9));

    const std::vector<FockHoppingTerm> hopping{
        FockHoppingTerm{1U, 0U, QComplex{1.25}},
        FockHoppingTerm{3U, 1U, QComplex{-0.5, 0.25}},
        FockHoppingTerm{2U, 2U, QComplex{0.75}},
    };
    const ExactSparseFockState hopped = state.apply_hopping(hopping);

    const QComplex a = QComplex{0.6, 0.2};
    const QComplex b = QComplex{-0.1, 0.7};
    assert(close(
        hopped.amplitude({{0U, 1U}, {1U, 1U}, {2U, 1U}}),
        a * 1.25 * std::sqrt(2.0)));
    assert(close(
        hopped.amplitude({{1U, 2U}, {3U, 1U}}),
        b * QComplex{-0.5, 0.25} * std::sqrt(3.0)));
    assert(close(hopped.amplitude({{0U, 2U}, {2U, 1U}}), a * 0.75));
    assert(hopped.fixed_particle_number().value() == 3U);

    const std::vector<double> onsite{0.2, -0.1, 0.4, 0.3};
    const std::vector<double> interaction{0.5, 0.25, -0.2, 0.0};
    const ExactSparseFockState hstate = state.apply_bose_hubbard(onsite, interaction, hopping);
    const double energy_a = 2.0 * 0.2 + 0.4 + 0.5 * 0.5 * 2.0 * 1.0;
    const double energy_b = 3.0 * -0.1 + 0.5 * 0.25 * 3.0 * 2.0;
    assert(close(
        hstate.amplitude({{0U, 2U}, {2U, 1U}}),
        a * (energy_a + 0.75)));
    assert(close(hstate.amplitude({{1U, 3U}}), b * energy_b));

    assert(throws_qstate([&] {
        (void)ExactSparseFockState::basis(4U, {{4U, 1U}}, config);
    }));
    assert(throws_qstate([&] {
        (void)ExactSparseFockState::basis(4U, {{1U, 1U}, {1U, 2U}}, config);
    }));
    assert(throws_qstate([&] {
        SparseFockConfig tight = config;
        tight.max_branch_products = 2U;
        const ExactSparseFockState bounded = ExactSparseFockState::from_terms(
            4U,
            {
                SparseFockTerm{{{0U, 3U}}, QComplex{1.0}},
                SparseFockTerm{{{1U, 3U}}, QComplex{1.0}},
            },
            tight);
        (void)bounded.apply_hopping(hopping);
    }));
    assert(throws_qstate([&] {
        const ExactSparseFockState mixed = ExactSparseFockState::from_terms(
            4U,
            {
                SparseFockTerm{{{0U, 1U}}, QComplex{1.0}},
                SparseFockTerm{{{1U, 2U}}, QComplex{1.0}},
            },
            config);
        (void)mixed.sector_log2_dimension();
    }));
    assert(throws_qstate([&] {
        (void)state.scaled(QComplex{std::numeric_limits<double>::infinity()});
    }));

    return 0;
}

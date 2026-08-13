#include "qubit/qbosonic_islands.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(double observed, double expected, double tolerance, const char* message) {
    if (std::abs(observed - expected) > tolerance * (1.0 + std::abs(expected))) {
        throw std::runtime_error(message);
    }
}

void product_queries() {
    using qubit::BosonicIslandKind;
    using qubit::ExactBosonicIslandProduct;
    using qubit::ExactSparseFockState;
    using qubit::StructuredGaussianState;

    const std::array<double, 2> thermal{0.25, 0.5};
    auto gaussian = StructuredGaussianState::thermal(thermal);
    auto fock = ExactSparseFockState::basis(3U, {{0U, 2U}, {2U, 1U}});
    auto product = ExactBosonicIslandProduct::from_product(
        std::move(gaussian), std::move(fock));

    require(product.mode_count() == 5U, "bosonic island total mode count mismatch");
    require(product.gaussian_mode_count() == 2U, "bosonic island Gaussian mode count mismatch");
    require(product.fock_mode_count() == 3U, "bosonic island Fock mode count mismatch");
    require(product.fock_global_offset() == 2U, "bosonic island Fock offset mismatch");
    require(product.island_of(1U) == BosonicIslandKind::Gaussian,
        "bosonic Gaussian island classification mismatch");
    require(product.island_of(4U) == BosonicIslandKind::SparseFock,
        "bosonic Fock island classification mismatch");

    require_close(product.mean_number(0U), 0.25, 1e-12, "Gaussian mean occupation mismatch");
    require_close(product.mean_number(1U), 0.5, 1e-12, "Gaussian mean occupation mismatch");
    require_close(product.mean_number(2U), 2.0, 1e-12, "Fock mean occupation mismatch");
    require_close(product.mean_number(3U), 0.0, 1e-12, "Fock empty mode mismatch");
    require_close(product.mean_number(4U), 1.0, 1e-12, "Fock mean occupation mismatch");
    require_close(product.total_mean_number(), 3.75, 1e-12, "bosonic total mean number mismatch");
    require_close(product.cross_island_number_product(1U, 2U), 1.0, 1e-12,
        "bosonic cross-island number product mismatch");
    require_close(product.cross_island_number_covariance(1U, 2U), 0.0, 1e-12,
        "bosonic product-state covariance mismatch");
}

void independent_island_evolution() {
    using qubit::ExactBosonicIslandProduct;
    using qubit::ExactSparseFockState;
    using qubit::GlobalFockHoppingTerm;
    using qubit::StructuredGaussianState;

    auto product = ExactBosonicIslandProduct::from_product(
        StructuredGaussianState::vacuum(3U),
        ExactSparseFockState::basis(3U, {{0U, 2U}}));

    product.gaussian_displace(0U, 1.0, -0.5);
    product.gaussian_squeeze(1U, 0.3);
    product.gaussian_two_mode_squeeze(1U, 2U, 0.25);
    product.gaussian_beam_splitter(0U, 1U, 0.5);
    product.gaussian_loss(2U, 0.7, 0.2);
    require(product.gaussian().stats().components == 1U,
        "bosonic Gaussian component merge mismatch");
    require(product.gaussian().mean_occupation(1U) > 0.0,
        "bosonic two-mode squeezing produced no occupation");
    require(product.fock().stats().terms == 1U,
        "Gaussian evolution changed sparse-Fock terms");

    const std::array<GlobalFockHoppingTerm, 1> hopping{{
        {4U, 3U, {1.0, 0.0}},
    }};
    product.apply_fock_hopping(hopping);
    require(product.fock().fixed_particle_number().has_value(),
        "Fock hopping lost fixed particle sector");
    require(*product.fock().fixed_particle_number() == 2U,
        "Fock hopping changed particle number");
    require_close(product.mean_number(4U), 1.0, 1e-12,
        "Fock hopping target mean mismatch");
    require_close(product.mean_number(3U), 1.0, 1e-12,
        "Fock hopping source mean mismatch");
}

void fail_closed_boundaries() {
    using qubit::BosonicIslandConfig;
    using qubit::ExactBosonicIslandProduct;
    using qubit::ExactSparseFockState;
    using qubit::GlobalFockHoppingTerm;
    using qubit::QStateError;
    using qubit::StructuredGaussianState;

    auto product = ExactBosonicIslandProduct::from_product(
        StructuredGaussianState::vacuum(2U),
        ExactSparseFockState::basis(2U, {{0U, 1U}}));

    bool rejected = false;
    try {
        product.gaussian_beam_splitter(0U, 2U, 0.5);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "cross-island beam splitter did not reject");

    rejected = false;
    try {
        product.gaussian_two_mode_squeeze(0U, 2U, 0.3);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "cross-island two-mode squeezing did not reject");

    rejected = false;
    try {
        const std::array<GlobalFockHoppingTerm, 1> hopping{{
            {2U, 0U, {1.0, 0.0}},
        }};
        product.apply_fock_hopping(hopping);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "cross-island Fock hopping did not reject");

    rejected = false;
    try {
        (void)product.cross_island_number_covariance(0U, 1U);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "same-island factorization query did not reject");

    rejected = false;
    try {
        (void)product.mean_number(4U);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "out-of-range bosonic mode did not reject");

    rejected = false;
    try {
        BosonicIslandConfig config;
        config.max_total_modes = 3U;
        (void)ExactBosonicIslandProduct::from_product(
            StructuredGaussianState::vacuum(2U),
            ExactSparseFockState::basis(2U, {{0U, 1U}}),
            config);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "bosonic island total mode cap did not reject");
}

}  // namespace

int main() {
    product_queries();
    independent_island_evolution();
    fail_closed_boundaries();
    return 0;
}

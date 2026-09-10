#include "qubit/qbosonic_islands.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>

int main() {
    using Clock = std::chrono::steady_clock;
    using qubit::ExactBosonicIslandProduct;
    using qubit::ExactSparseFockState;
    using qubit::FockOccupation;
    using qubit::GlobalFockHoppingTerm;
    using qubit::QStateError;
    using qubit::StructuredGaussianState;

    constexpr std::size_t gaussian_modes = 4096U;
    constexpr std::size_t fock_modes = 4096U;
    FockOccupation occupation;
    for (std::size_t mode = 0U; mode < 16U; mode += 2U) {
        occupation.push_back({mode, 512U});
    }

    auto gaussian = StructuredGaussianState::vacuum(gaussian_modes);
    auto fock = ExactSparseFockState::basis(fock_modes, occupation);
    const double sector_log2 = fock.sector_log2_dimension();
    auto product = ExactBosonicIslandProduct::from_product(
        std::move(gaussian), std::move(fock));

    const auto transform_begin = Clock::now();
    product.gaussian_two_mode_squeeze(0U, 1U, 0.35);
    product.gaussian_displace(2U, 0.75, -0.25);
    for (std::size_t pair = 0U; pair < 1024U; ++pair) {
        product.gaussian_beam_splitter(2U * pair, 2U * pair + 1U, 0.5);
    }

    std::array<GlobalFockHoppingTerm, 8> hopping{};
    for (std::size_t index = 0U; index < hopping.size(); ++index) {
        hopping[index] = GlobalFockHoppingTerm{
            gaussian_modes + 2U * index + 1U,
            gaussian_modes + 2U * index,
            {1.0, 0.0},
        };
    }
    product.apply_fock_hopping(hopping);
    const auto transform_end = Clock::now();

    const auto query_begin = Clock::now();
    const double gaussian_mean = product.mean_number(0U);
    const double fock_mean = product.mean_number(gaussian_modes);
    const double total_mean = product.total_mean_number();
    const double cross_product = product.cross_island_number_product(0U, gaussian_modes);
    const double cross_covariance = product.cross_island_number_covariance(0U, gaussian_modes);
    const auto query_end = Clock::now();

    bool cross_beam_splitter_rejected = false;
    try {
        product.gaussian_beam_splitter(0U, gaussian_modes, 0.5);
    } catch (const QStateError&) {
        cross_beam_splitter_rejected = true;
    }

    bool cross_hopping_rejected = false;
    try {
        const std::array<GlobalFockHoppingTerm, 1> cross{{
            {gaussian_modes, 0U, {1.0, 0.0}},
        }};
        product.apply_fock_hopping(cross);
    } catch (const QStateError&) {
        cross_hopping_rejected = true;
    }

    const auto stats = product.stats();
    const auto fixed_particles = product.fock().fixed_particle_number();
    const std::size_t dense_gaussian_scalars =
        (2U * gaussian_modes) * (2U * gaussian_modes) + 2U * gaussian_modes;
    const double gaussian_descriptor_ratio =
        static_cast<double>(dense_gaussian_scalars) /
        static_cast<double>(stats.gaussian_descriptor_scalars);
    const double transform_ms =
        std::chrono::duration<double, std::milli>(transform_end - transform_begin).count();
    const double query_ms =
        std::chrono::duration<double, std::milli>(query_end - query_begin).count();

    std::cout << std::setprecision(17)
              << "island_total_modes=" << stats.total_modes << '\n'
              << "island_gaussian_modes=" << stats.gaussian_modes << '\n'
              << "island_fock_modes=" << stats.fock_modes << '\n'
              << "island_gaussian_components=" << stats.gaussian_components << '\n'
              << "island_gaussian_descriptor_scalars=" << stats.gaussian_descriptor_scalars << '\n'
              << "island_gaussian_dense_descriptor_scalars=" << dense_gaussian_scalars << '\n'
              << "island_gaussian_descriptor_ratio=" << gaussian_descriptor_ratio << '\n'
              << "island_fock_terms=" << stats.fock_terms << '\n'
              << "island_fock_occupied_entries=" << stats.fock_occupied_entries << '\n'
              << "island_fock_fixed_particles="
              << (fixed_particles.has_value() ? *fixed_particles : 0U) << '\n'
              << "island_fock_sector_log2_dimension=" << sector_log2 << '\n'
              << "island_gaussian_mean_mode0=" << gaussian_mean << '\n'
              << "island_fock_mean_first_mode=" << fock_mean << '\n'
              << "island_total_mean_number=" << total_mean << '\n'
              << "island_cross_number_product=" << cross_product << '\n'
              << "island_cross_number_covariance=" << cross_covariance << '\n'
              << "island_cross_beam_splitter_rejected="
              << (cross_beam_splitter_rejected ? 1 : 0) << '\n'
              << "island_cross_hopping_rejected="
              << (cross_hopping_rejected ? 1 : 0) << '\n'
              << "island_two_mode_squeezing_exercised=1\n"
              << "island_transform_ms=" << transform_ms << '\n'
              << "island_query_ms=" << query_ms << '\n'
              << "gaussian_to_fock_conversion_performed=0\n"
              << "fock_to_gaussian_conversion_performed=0\n";
    return 0;
}

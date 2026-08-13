#include "qubit/qgaussian.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace qubit;

int main() {
    constexpr std::size_t modes = 4096U;
    GaussianConfig config;
    config.max_modes = 8192U;
    config.max_component_modes = 4U;
    config.max_component_scalars = 128U;
    config.max_total_scalars = 1U << 20U;
    config.max_abs_squeeze = 4.0;

    std::vector<double> occupations(modes, 0.0);
    StructuredGaussianState state = StructuredGaussianState::thermal(occupations, config);

    const auto transform_start = std::chrono::steady_clock::now();
    for (std::size_t pair = 0U; pair < 1024U; ++pair) {
        const std::size_t first = 2U * pair;
        const std::size_t second = first + 1U;
        state.squeeze(first, 0.05 * static_cast<double>((pair % 5U) + 1U));
        state.rotate(second, 0.01 * static_cast<double>((pair % 11U) + 1U));
        state.beam_splitter(first, second, 0.5);
    }
    for (std::size_t mode = 2048U; mode < modes; mode += 257U) {
        state.displace(mode, 0.05, -0.025);
        state.loss(mode, 0.95, 0.1);
    }
    const auto transform_end = std::chrono::steady_clock::now();

    const auto query_start = std::chrono::steady_clock::now();
    const double total_occupation = state.total_mean_occupation();
    const auto cross = state.covariance_block(0U, 1U);
    const auto query_end = std::chrono::steady_clock::now();

    const std::size_t dense_dimension = 2U * modes;
    const std::size_t dense_covariance_scalars = dense_dimension * dense_dimension;
    const std::size_t dense_descriptor_scalars = dense_covariance_scalars + dense_dimension;
    const double compression_ratio = static_cast<double>(dense_descriptor_scalars) /
                                     static_cast<double>(state.stats().descriptor_scalars);
    const auto milliseconds = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    std::cout << std::setprecision(17);
    std::cout << "gaussian_modes=" << modes << '\n';
    std::cout << "gaussian_components=" << state.stats().components << '\n';
    std::cout << "gaussian_largest_component_modes=" << state.stats().largest_component_modes << '\n';
    std::cout << "gaussian_descriptor_scalars=" << state.stats().descriptor_scalars << '\n';
    std::cout << "gaussian_dense_descriptor_scalars=" << dense_descriptor_scalars << '\n';
    std::cout << "gaussian_dense_to_structured_scalar_ratio=" << compression_ratio << '\n';
    std::cout << "gaussian_total_mean_occupation=" << total_occupation << '\n';
    std::cout << "gaussian_cross_qq_0_1=" << cross[0] << '\n';
    std::cout << "gaussian_transform_ms=" << milliseconds(transform_start, transform_end) << '\n';
    std::cout << "gaussian_query_ms=" << milliseconds(query_start, query_end) << '\n';

    return state.stats().components == 3072U &&
                   state.stats().largest_component_modes == 2U &&
                   state.stats().descriptor_scalars == 32768U &&
                   compression_ratio > 2000.0 &&
                   std::isfinite(total_occupation) && total_occupation > 0.0 &&
                   std::isfinite(cross[0])
               ? 0
               : 1;
}

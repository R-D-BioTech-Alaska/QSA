#include "qubit/qgaussian.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

using namespace qubit;

namespace {

bool close(double left, double right, double tolerance = 2e-11) {
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
    GaussianConfig config;
    config.max_modes = 32U;
    config.max_component_modes = 8U;
    config.max_component_scalars = 512U;
    config.max_total_scalars = 4096U;
    config.max_abs_squeeze = 4.0;

    StructuredGaussianState vacuum = StructuredGaussianState::vacuum(4U, config);
    assert(vacuum.stats().components == 4U);
    assert(vacuum.stats().largest_component_modes == 1U);
    assert(vacuum.stats().descriptor_scalars == 24U);
    for (std::size_t mode = 0U; mode < 4U; ++mode) {
        assert(close(vacuum.mean_occupation(mode), 0.0));
        const auto block = vacuum.covariance_block(mode, mode);
        assert(close(block[0], 0.5));
        assert(close(block[1], 0.0));
        assert(close(block[2], 0.0));
        assert(close(block[3], 0.5));
    }

    vacuum.displace(0U, 0.4, -0.3);
    assert(close(vacuum.mean_occupation(0U), 0.125));
    vacuum.rotate(0U, 0.37);
    assert(close(vacuum.mean_occupation(0U), 0.125));

    StructuredGaussianState squeezed = StructuredGaussianState::vacuum(1U, config);
    constexpr double squeeze = 0.6;
    squeezed.squeeze(0U, squeeze);
    const auto squeezed_block = squeezed.covariance_block(0U, 0U);
    assert(close(squeezed_block[0], 0.5 * std::exp(-2.0 * squeeze)));
    assert(close(squeezed_block[3], 0.5 * std::exp(2.0 * squeeze)));
    assert(close(squeezed.mean_occupation(0U), std::sinh(squeeze) * std::sinh(squeeze)));

    const std::vector<double> occupations{1.0, 0.0};
    StructuredGaussianState mixed = StructuredGaussianState::thermal(occupations, config);
    mixed.beam_splitter(0U, 1U, 0.5);
    assert(mixed.stats().components == 1U);
    assert(mixed.stats().largest_component_modes == 2U);
    assert(mixed.stats().descriptor_scalars == 20U);
    assert(close(mixed.mean_occupation(0U), 0.5));
    assert(close(mixed.mean_occupation(1U), 0.5));
    assert(close(mixed.total_mean_occupation(), 1.0));
    const auto cross = mixed.covariance_block(0U, 1U);
    assert(close(cross[0], -0.5));
    assert(close(cross[1], 0.0));
    assert(close(cross[2], 0.0));
    assert(close(cross[3], -0.5));

    mixed.loss(0U, 0.4, 0.25);
    assert(close(mixed.mean_occupation(0U), 0.4 * 0.5 + 0.6 * 0.25));
    assert(close(mixed.mean_occupation(1U), 0.5));

    GaussianConfig component_cap = config;
    component_cap.max_component_modes = 1U;
    StructuredGaussianState separated = StructuredGaussianState::vacuum(2U, component_cap);
    assert(throws_qstate([&] { separated.beam_splitter(0U, 1U, 0.5); }));
    assert(throws_qstate([&] { squeezed.squeeze(0U, 5.0); }));
    assert(throws_qstate([&] { mixed.loss(0U, 1.1); }));
    assert(throws_qstate([&] { mixed.beam_splitter(0U, 0U, 0.5); }));

    return 0;
}

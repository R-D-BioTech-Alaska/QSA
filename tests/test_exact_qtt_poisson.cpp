#include "qubit/qqtt_poisson.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

using namespace qubit;

namespace {

bool close(double left, double right, double tolerance = 1e-11) {
    return std::abs(left - right) <= tolerance;
}

std::vector<std::uint8_t> bits_for(std::size_t logical_bits, BasisIndex index) {
    std::vector<std::uint8_t> bits(logical_bits);
    for (std::size_t position = 0U; position < logical_bits; ++position) {
        const std::size_t shift = logical_bits - 1U - position;
        bits[position] = static_cast<std::uint8_t>((index >> shift) & BasisIndex{1});
    }
    return bits;
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
    constexpr std::size_t logical_bits = 6U;
    const std::vector<double> weights{0.5, 1.0, 1.5, 0.75, 1.25, 2.0};
    const ExactWalshField source = ExactWalshField::from_terms(
        logical_bits,
        {
            WalshFieldTerm{0.4, {}},
            WalshFieldTerm{1.25, {0U}},
            WalshFieldTerm{-0.75, {1U, 3U}},
            WalshFieldTerm{0.5, {2U, 4U, 5U}},
            WalshFieldTerm{0.25, {3U, 1U}},
        });
    assert(source.term_count() == 4U);

    WalshFieldConfig merge_config;
    merge_config.max_terms = 2U;
    const ExactWalshField merge_left = ExactWalshField::from_terms(
        logical_bits,
        {WalshFieldTerm{1.0, {0U}}, WalshFieldTerm{2.0, {1U}}},
        merge_config);
    const ExactWalshField merge_right = ExactWalshField::from_terms(
        logical_bits,
        {WalshFieldTerm{-1.0, {0U}}, WalshFieldTerm{3.0, {1U}}},
        merge_config);
    const ExactWalshField merged = merge_left.add(merge_right);
    assert(merged.term_count() == 1U);
    assert(close(merged.terms().front().coefficient, 5.0));

    constexpr double mu = 0.625;
    constexpr double alpha = 1.75;
    const ExactQTTRegularizedPoisson poisson(weights, mu, alpha);
    const ExactWalshField potential = poisson.solve(source);
    const ExactWalshField residual = poisson.residual(source, potential);
    assert(residual.maximum_absolute_coefficient() <= 1e-12);

    const auto stats = poisson.stats(source);
    assert(stats.logical_bits == logical_bits);
    assert(stats.source_terms == 4U);
    assert(close(stats.minimum_eigenvalue, mu));
    assert(close(stats.inverse_norm_bound, 1.0 / mu));
    assert(close(stats.maximum_eigenvalue, mu + 2.0 * 7.0));

    QTTConfig qtt_config;
    qtt_config.max_rank = 8U;
    qtt_config.max_core_scalars = 256U;
    qtt_config.max_total_scalars = 4096U;
    const ExactQTTFunction qtt_potential = potential.to_qtt(qtt_config);
    assert(qtt_potential.stats().maximum_rank == potential.term_count());

    for (BasisIndex index = 0U; index < (BasisIndex{1} << logical_bits); ++index) {
        const std::vector<std::uint8_t> bits = bits_for(logical_bits, index);
        const double center = potential.value_bits(bits);
        double laplacian = 0.0;
        for (std::size_t position = 0U; position < logical_bits; ++position) {
            std::vector<std::uint8_t> neighbor = bits;
            neighbor[position] ^= 1U;
            laplacian += weights[position] * (center - potential.value_bits(neighbor));
        }
        const double direct_residual = mu * center + laplacian - alpha * source.value_bits(bits);
        assert(std::abs(direct_residual) <= 2e-11);
        assert(close(qtt_potential.value_bits(bits).re, center, 2e-11));
        assert(std::abs(qtt_potential.value_bits(bits).im) <= 2e-11);
    }

    assert(throws_qstate([&] {
        (void)ExactQTTRegularizedPoisson(weights, 0.0);
    }));
    assert(throws_qstate([&] {
        std::vector<double> invalid = weights;
        invalid[2] = -1.0;
        (void)ExactQTTRegularizedPoisson(invalid, mu);
    }));
    assert(throws_qstate([&] {
        WalshFieldConfig config;
        config.max_terms = 2U;
        (void)ExactWalshField::from_terms(
            logical_bits,
            {WalshFieldTerm{1.0, {}}, WalshFieldTerm{1.0, {0U}}, WalshFieldTerm{1.0, {1U}}},
            config);
    }));
    assert(throws_qstate([&] {
        QTTConfig tight;
        tight.max_rank = 2U;
        tight.max_core_scalars = 1024U;
        tight.max_total_scalars = 4096U;
        (void)potential.to_qtt(tight);
    }));
    assert(throws_qstate([&] {
        (void)ExactWalshField::from_terms(
            logical_bits,
            {WalshFieldTerm{std::numeric_limits<double>::infinity(), {0U}}});
    }));
    assert(throws_qstate([&] {
        (void)ExactWalshField::from_terms(
            logical_bits,
            {WalshFieldTerm{1.0, {1U, 1U}}});
    }));

    return 0;
}

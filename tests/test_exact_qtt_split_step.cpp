#include "qubit/qqtt_split_step.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

using namespace qubit;

namespace {

bool close(QComplex left, QComplex right, double tolerance = 2e-10) {
    return almost_equal(left, right, tolerance);
}

bool close(double left, double right, double tolerance = 2e-10) {
    return std::abs(left - right) <= tolerance * (1.0 + std::max(std::abs(left), std::abs(right)));
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
    constexpr std::size_t logical_bits = 5U;
    const std::vector<double> weights{0.5, 0.75, 1.0, 1.25, 1.5};
    ComplexWalshConfig wave_config;
    wave_config.max_terms = 64U;
    wave_config.max_support_entries = 4096U;
    wave_config.max_products = 1U << 16U;
    const ExactComplexWalshField wave = ExactComplexWalshField::from_terms(
        logical_bits,
        {
            ComplexWalshTerm{QComplex{0.85, 0.05}, {}},
            ComplexWalshTerm{QComplex{0.15, -0.08}, {0U, 2U}},
        },
        wave_config);

    ComplexWalshConfig phase_config = wave_config;
    phase_config.max_terms = 64U;
    constexpr double mu = 0.7;
    constexpr double alpha = 1.2;
    constexpr double kinetic = 0.35;
    constexpr double nonlinear = 0.18;
    constexpr double background = 0.25;
    constexpr double dt = 0.01;
    const ExactQTTSchrodingerPoissonSplitStep integrator(
        weights, mu, alpha, kinetic, nonlinear, phase_config);

    const QTTSplitStepResult forward = integrator.step(wave, dt, background);
    assert(forward.stats.norm_error <= 2e-12);
    assert(close(forward.stats.input_mean_norm_squared, forward.stats.output_mean_norm_squared, 2e-12));

    for (BasisIndex index = 0U; index < (BasisIndex{1} << logical_bits); ++index) {
        const std::vector<std::uint8_t> bits = bits_for(logical_bits, index);
        const QComplex phase = forward.local_phase.value_bits(bits);
        assert(close(phase.norm2(), 1.0, 2e-12));
    }

    const QTTSplitStepResult backward = integrator.step(forward.wave, -dt, background);
    const ExactComplexWalshField reversal_error = backward.wave.add(wave.scaled(QComplex{-1.0}));
    const double reversal_l2 = reversal_error.mean_inner_product(reversal_error).re;
    assert(reversal_l2 <= 2e-20);
    assert(backward.stats.norm_error <= 2e-12);

    QTTConfig qtt_config;
    qtt_config.max_rank = 64U;
    qtt_config.max_core_scalars = 8192U;
    qtt_config.max_total_scalars = 1U << 18U;
    const ExactQTTFunction output_qtt = forward.wave.to_qtt(qtt_config);
    for (BasisIndex index = 0U; index < (BasisIndex{1} << logical_bits); ++index) {
        const std::vector<std::uint8_t> bits = bits_for(logical_bits, index);
        assert(close(output_qtt.value_bits(bits), forward.wave.value_bits(bits), 2e-10));
    }

    ComplexWalshConfig phase_tight = wave_config;
    phase_tight.max_terms = 1U;
    const ExactQTTSchrodingerPoissonSplitStep capped(
        weights, mu, alpha, kinetic, nonlinear, phase_tight);
    assert(throws_qstate([&] { (void)capped.step(wave, dt, background); }));

    assert(throws_qstate([&] {
        (void)integrator.step(wave, std::numeric_limits<double>::infinity(), background);
    }));

    return 0;
}

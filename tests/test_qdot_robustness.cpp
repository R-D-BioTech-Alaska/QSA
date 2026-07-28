#include "test_qdot_support.hpp"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <random>
#include <vector>

using namespace qubit;
using namespace qubit::qdot;

namespace {

double extreme_finite(std::mt19937_64& rng) {
    std::uniform_int_distribution<int> exponent(-1020, 1020);
    std::uniform_real_distribution<double> mantissa(0.5, 1.0);
    std::bernoulli_distribution sign(0.5);
    const double value = std::ldexp(mantissa(rng), exponent(rng));
    return sign(rng) ? value : -value;
}

bool same_probabilities(
    const std::vector<double>& left,
    const std::vector<double>& right) {
    return left == right;
}

}  // namespace

int main() {
    std::mt19937_64 rng(0x51444F54524F4255ULL);
    std::uniform_real_distribution<double> ordinary(-4.0, 4.0);
    std::uniform_real_distribution<double> strength(0.0, 1.0);
    std::uniform_int_distribution<int> dots_distribution(1, 7);

    std::size_t accepted_extreme = 0;
    std::size_t rejected_extreme = 0;
    constexpr std::size_t extreme_cases = 3000;
    for (std::size_t test_case = 0; test_case < extreme_cases; ++test_case) {
        PocketConfig config;
        config.dot_count = static_cast<std::size_t>(dots_distribution(rng));
        config.topology = (config.dot_count % 2U == 0U)
                              ? Topology::PairBlocks
                              : Topology::PairPlusContext;
        config.dt = std::abs(extreme_finite(rng));
        if (config.dt == 0.0) {
            config.dt = std::numeric_limits<double>::min();
        }
        config.trotter_steps = 1U + (test_case % 4U);
        double* dot_values[] = {
            &config.dot.e0,
            &config.dot.eup,
            &config.dot.edown,
            &config.dot.ex,
            &config.dot.detuning_gain,
            &config.dot.exciton_gain,
            &config.dot.charge_drive,
            &config.dot.spin_drive,
            &config.dot.intra_dot_bias,
        };
        for (double* value : dot_values) {
            *value = extreme_finite(rng);
        }
        double* coupling_values[] = {
            &config.coupling.capacitive,
            &config.coupling.spin_exchange,
            &config.coupling.charge_tunneling,
            &config.coupling.scale,
        };
        for (double* value : coupling_values) {
            *value = extreme_finite(rng);
        }

        QuantumDotPocket pocket(config);
        std::vector<DotInput> inputs(config.dot_count);
        for (DotInput& input : inputs) {
            input = {extreme_finite(rng), extreme_finite(rng), strength(rng)};
        }
        const auto before = pocket.probabilities_one();
        try {
            pocket.step(inputs);
            ++accepted_extreme;
            require(pocket.validate(), "accepted extreme finite case must remain valid");
            for (double value : pocket.probabilities_one()) {
                require(std::isfinite(value) && value >= 0.0 && value <= 1.0,
                        "accepted extreme case returned an invalid probability");
            }
        } catch (const QStateError&) {
            ++rejected_extreme;
            require(same_probabilities(pocket.probabilities_one(), before),
                    "rejected extreme case partially mutated the pocket");
            require(pocket.validate(), "rejected extreme case left an invalid pocket");
        }
    }

    constexpr std::size_t invalid_cases = 1200;
    for (std::size_t test_case = 0; test_case < invalid_cases; ++test_case) {
        PocketConfig config;
        config.dot_count = 3U + 2U * (test_case % 3U);
        config.topology = Topology::PairPlusContext;
        QuantumDotPocket pocket(config);
        std::vector<DotInput> inputs(config.dot_count);
        for (DotInput& input : inputs) {
            input = {ordinary(rng), ordinary(rng), strength(rng)};
        }
        DotInput& bad = inputs[test_case % inputs.size()];
        switch (test_case % 6U) {
            case 0U:
                bad.theta = std::numeric_limits<double>::quiet_NaN();
                break;
            case 1U:
                bad.theta = std::numeric_limits<double>::infinity();
                break;
            case 2U:
                bad.phi = -std::numeric_limits<double>::infinity();
                break;
            case 3U:
                bad.strength = std::numeric_limits<double>::quiet_NaN();
                break;
            case 4U:
                bad.strength = -0.001;
                break;
            default:
                bad.strength = 1.001;
                break;
        }
        const auto before = pocket.probabilities_one();
        bool rejected = false;
        try {
            pocket.step(inputs);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "invalid randomized input was accepted");
        require(same_probabilities(pocket.probabilities_one(), before),
                "invalid randomized input partially mutated the pocket");
        require(pocket.validate(), "invalid randomized input left an invalid pocket");
    }

    {
        PocketConfig config;
        config.dot_count = 3;
        config.topology = Topology::PairPlusContext;
        config.dt = std::numeric_limits<double>::max();
        config.dot.charge_drive = std::numeric_limits<double>::max();
        QRegister reference(6);
        const auto before = reference.materialize(12);
        std::vector<DotInput> inputs(3, DotInput{1.0, 0.0, 1.0});
        bool rejected = false;
        try {
            apply_reference_step(reference, config, inputs);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "reference path accepted overflowing derived angles");
        require(reference.materialize(12) == before,
                "reference path partially mutated before rejecting overflow");
        require(reference.validate(),
                "reference path became invalid after overflow rejection");
    }

    require(accepted_extreme + rejected_extreme == extreme_cases,
            "extreme robustness accounting mismatch");
    std::cout << "robustness_extreme_cases=" << extreme_cases
              << " accepted=" << accepted_extreme
              << " rejected=" << rejected_extreme
              << " invalid_cases=" << invalid_cases << '\n';
    return 0;
}

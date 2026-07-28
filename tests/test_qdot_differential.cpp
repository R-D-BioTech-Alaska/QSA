#include "test_qdot_support.hpp"

#include <iomanip>
#include <random>

using namespace qubit;
using namespace qubit::qdot;

int main() {
    std::mt19937_64 rng(0x51444F5453454544ULL);
    std::uniform_real_distribution<double> angle(-3.0, 3.0);
    std::uniform_real_distribution<double> strength(0.0, 1.0);
    std::uniform_real_distribution<double> energy(-0.4, 2.6);
    std::uniform_real_distribution<double> gain(-0.5, 1.4);
    std::uniform_real_distribution<double> coupling(-0.22, 0.25);
    std::uniform_real_distribution<double> dt(0.015, 0.25);

    double minimum_fidelity = 1.0;
    double maximum_probability_error = 0.0;
    double maximum_norm_error = 0.0;
    constexpr std::size_t cases = 72;

    for (std::size_t test_case = 0; test_case < cases; ++test_case) {
        PocketConfig config;
        config.dot_count = 1U + (test_case % 6U);
        config.topology = (config.dot_count % 2U == 0U)
                              ? Topology::PairBlocks
                              : Topology::PairPlusContext;
        config.dt = dt(rng);
        config.trotter_steps = 1U + (test_case % 4U);
        config.dot.e0 = energy(rng);
        config.dot.eup = energy(rng);
        config.dot.edown = energy(rng);
        config.dot.ex = energy(rng);
        config.dot.detuning_gain = gain(rng);
        config.dot.exciton_gain = gain(rng);
        config.dot.charge_drive = gain(rng);
        config.dot.spin_drive = gain(rng);
        config.dot.intra_dot_bias = gain(rng);
        config.coupling.capacitive = coupling(rng);
        config.coupling.spin_exchange = coupling(rng);
        config.coupling.charge_tunneling = coupling(rng);
        config.coupling.scale = gain(rng);

        QuantumDotPocket specialized(config);
        QRegister reference(config.dot_count * 2U);
        const std::size_t steps = 1U + (test_case % 9U);
        std::vector<DotInput> inputs(config.dot_count);
        for (std::size_t step = 0; step < steps; ++step) {
            for (DotInput& input : inputs) {
                input = DotInput{angle(rng), angle(rng), strength(rng)};
            }
            specialized.step(inputs);
            apply_reference_step(reference, config, inputs);
            maximum_norm_error = std::max(
                maximum_norm_error, specialized.current_max_norm_error());
            require(specialized.validate(),
                    "specialized state failed validation during differential test");
            require(reference.validate(),
                    "QSA reference failed validation during differential test");
        }

        const auto actual = specialized.materialize(16);
        const auto expected = reference.materialize(16);
        const double fidelity = state_fidelity(actual, expected);
        const double probability_error = max_probability_error(
            specialized.probabilities_one(), reference.probabilities_one());
        minimum_fidelity = std::min(minimum_fidelity, fidelity);
        maximum_probability_error = std::max(maximum_probability_error, probability_error);
        require(fidelity > 1.0 - 1.5e-7,
                "specialized state diverged from generic QSA reference");
        require(probability_error < 8e-5,
                "specialized probabilities diverged from QSA reference");
    }

    for (unsigned mask = 0; mask < 8; ++mask) {
        PocketConfig config;
        config.dot_count = 3;
        config.topology = Topology::PairPlusContext;
        config.coupling.capacitive = (mask & 1U) ? 0.19 : 0.0;
        config.coupling.spin_exchange = (mask & 2U) ? -0.11 : 0.0;
        config.coupling.charge_tunneling = (mask & 4U) ? 0.07 : 0.0;
        QuantumDotPocket specialized(config);
        QRegister reference(6);
        std::vector<DotInput> inputs(3, DotInput{0.9, -1.1, 0.73});
        for (int step = 0; step < 16; ++step) {
            inputs[1].phi += 0.037;
            specialized.step(inputs);
            apply_reference_step(reference, config, inputs);
        }
        require(
            state_fidelity(specialized.materialize(12), reference.materialize(12)) >
                1.0 - 2e-8,
            "interaction-family ablation diverged from reference");
    }

    std::cout << std::setprecision(17)
              << "differential_cases=" << cases
              << " minimum_fidelity=" << minimum_fidelity
              << " maximum_probability_error=" << maximum_probability_error
              << " maximum_norm_error=" << maximum_norm_error << '\n';
    return 0;
}

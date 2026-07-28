#include "test_qdot_support.hpp"

#include <iomanip>
#include <vector>

using namespace qubit::qdot;

std::vector<DotInput> make_inputs(std::size_t count, std::size_t step) {
    std::vector<DotInput> values(count);
    for (std::size_t dot = 0; dot < count; ++dot) {
        values[dot] = DotInput{
            0.1 + 0.0021 * static_cast<double>(step) + 0.017 * static_cast<double>(dot),
            -0.3 + 0.0013 * static_cast<double>(step) - 0.011 * static_cast<double>(dot),
            0.5 + 0.49 * std::sin(0.007 * static_cast<double>(step + 3U * dot)),
        };
    }
    return values;
}

void run_stress(std::size_t dots, std::size_t steps) {
    PocketConfig config;
    config.dot_count = dots;
    config.topology = (dots % 2U == 0U) ? Topology::PairBlocks : Topology::PairPlusContext;
    QuantumDotPocket pocket(config);
    double peak = 0.0;
    for (std::size_t step = 0; step < steps; ++step) {
        const auto inputs = make_inputs(dots, step);
        pocket.step(inputs);
        peak = std::max(peak, pocket.current_max_norm_error());
        if ((step % 97U) == 0U) {
            require(pocket.validate(), "stress pocket failed periodic validation");
        }
    }
    require(pocket.validate(), "stress pocket failed final validation");
    require(peak < 2e-9, "stress norm drift exceeded gate");
    const std::size_t expected_components = (dots + 1U) / 2U;
    require(pocket.component_count() == expected_components,
            "stress component count changed");
    require(pocket.max_component_qubits() <= 4U,
            "stress component width exceeded four qubits");
    const double bytes_per_dot =
        static_cast<double>(pocket.estimated_bytes()) / static_cast<double>(dots);
    require(bytes_per_dot < 400.0, "stress memory scaling exceeded linear bound");
    std::cout << std::setprecision(12)
              << "stress dots=" << dots
              << " steps=" << steps
              << " peak_norm_error=" << peak
              << " bytes=" << pocket.estimated_bytes()
              << " bytes_per_dot=" << bytes_per_dot << '\n';
}

int main() {
    run_stress(3, 50'000);
    run_stress(25, 20'000);
    run_stress(256, 2'000);
    run_stress(1025, 200);
    return 0;
}

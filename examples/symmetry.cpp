#include "qubit/qsymmetry.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
    qubit::SymmetryState state = qubit::SymmetryState::hamming_weight(60);
    std::vector<double> phases(state.class_count());
    for (std::size_t weight = 0; weight < phases.size(); ++weight) {
        phases[weight] = -0.02 * static_cast<double>(weight);
    }
    state.apply_class_phases(phases);
    state.apply_weighted_reflection();

    std::cout << state.describe();
    std::cout << "P(weight=30): " << state.class_probability(30) << '\n';
    return 0;
}

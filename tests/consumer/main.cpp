#include "qubit/qstate.hpp"

#include <cmath>

int main() {
    qubit::QRegister state(2);
    state.apply_h(0);
    state.apply_cnot(0, 1);
    const double expected = 1.0 / std::sqrt(2.0);
    return std::abs(state.amplitude(0).re - expected) < 1e-12 &&
                   std::abs(state.amplitude(3).re - expected) < 1e-12
               ? 0
               : 1;
}

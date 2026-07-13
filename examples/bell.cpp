#include "qubit/qstate.hpp"

#include <iostream>

int main() {
    qubit::QRegister state(2);
    state.apply_h(0);
    state.apply_cnot(0, 1);

    std::cout << state.describe() << '\n';
    const auto amplitudes = state.materialize();
    for (std::size_t basis = 0; basis < amplitudes.size(); ++basis) {
        std::cout << basis << ": " << amplitudes[basis].re
                  << (amplitudes[basis].im < 0.0 ? "" : "+")
                  << amplitudes[basis].im << "i\n";
    }
    return 0;
}

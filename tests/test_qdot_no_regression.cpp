#include "test_qdot_support.hpp"

#include <cmath>

using namespace qubit;

int main() {
    QRegister bell(2);
    bell.apply_h(0);
    bell.apply_cnot(0, 1);
    require(bell.validate(), "Bell state must validate with qdot linked");
    const double inv_sqrt_two = 1.0 / std::sqrt(2.0);
    require_near(bell.amplitude(0).magnitude(), inv_sqrt_two, 1e-12,
                 "Bell |00> amplitude");
    require_near(bell.amplitude(3).magnitude(), inv_sqrt_two, 1e-12,
                 "Bell |11> amplitude");

    QRegister independent(4096);
    for (QubitId qubit = 0; qubit < 4096; ++qubit) {
        independent.apply_ry(qubit, 0.001 * static_cast<double>(qubit % 17));
        independent.apply_rz(qubit, -0.002 * static_cast<double>(qubit % 13));
    }
    require(independent.component_count() == 4096,
            "qdot link must not merge ordinary independent qubits");
    require(independent.validate(), "ordinary independent register must validate");

    QRegister compact(2);
    compact.apply_x(0);
    compact.apply_cnot(0, 1);
    require(compact.component_count() == 2,
            "basis-state CNOT must still compact to cells");
    require_near(compact.probability_one(0), 1.0, 1e-12,
                 "ordinary q0 probability");
    require_near(compact.probability_one(1), 1.0, 1e-12,
                 "ordinary q1 probability");

    std::cout << "ordinary QSA no-regression tests passed\n";
    return 0;
}

#include "qubit/qpauli.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using qubit::Operation;
using qubit::OperationCode;
using qubit::OperationPlan;
using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
using qubit::PauliPropagationConfig;
using qubit::QComplex;
using qubit::QRegister;
using qubit::QStateError;
using qubit::QubitId;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(const QComplex& actual, const QComplex& expected, double tolerance, const char* message) {
    if (!qubit::almost_equal(actual, expected, tolerance)) {
        throw std::runtime_error(message);
    }
}

PauliObservable single_term(
    std::size_t qubits,
    std::initializer_list<PauliFactor> factors,
    QComplex coefficient = {1.0, 0.0},
    PauliPropagationConfig config = {}) {
    PauliObservable observable(qubits, config);
    const std::vector<PauliFactor> values(factors);
    observable.add_term(coefficient, values);
    return observable;
}

}  // namespace

int main() {
    {
        QRegister state(2);
        state.apply_h(0);
        state.apply_cnot(0, 1);

        const auto xx = single_term(2, {{0, PauliAxis::X}, {1, PauliAxis::X}});
        const auto yy = single_term(2, {{0, PauliAxis::Y}, {1, PauliAxis::Y}});
        const auto zz = single_term(2, {{0, PauliAxis::Z}, {1, PauliAxis::Z}});
        require_close(xx.expectation(state), {1.0, 0.0}, 2e-12, "Bell XX expectation mismatch");
        require_close(yy.expectation(state), {-1.0, 0.0}, 2e-12, "Bell YY expectation mismatch");
        require_close(zz.expectation(state), {1.0, 0.0}, 2e-12, "Bell ZZ expectation mismatch");
    }

    {
        QRegister state(4);
        state.apply_h(0);
        state.apply_cnot(0, 1);
        state.apply_h(2);
        state.apply_cnot(2, 3);
        const auto observable = single_term(
            4,
            {{0, PauliAxis::X},
             {1, PauliAxis::X},
             {2, PauliAxis::Z},
             {3, PauliAxis::Z}});
        require_close(
            observable.expectation(state),
            {1.0, 0.0},
            2e-12,
            "factorized Bell-pair expectation mismatch");
    }

    {
        QRegister state(10'000);
        state.apply_h(0);
        state.apply_h(9'999);
        const auto observable = single_term(
            10'000,
            {{0, PauliAxis::X}, {9'999, PauliAxis::X}});
        require_close(
            observable.expectation(state),
            {1.0, 0.0},
            2e-12,
            "large factorized Pauli expectation mismatch");
        require(state.component_count() == 10'000U,
                "Pauli readout changed the factorized component layout");
    }

    {
        QRegister initial(4);
        const std::vector<Operation> operations{
            {OperationCode::H, 0, 0, 0.0, 0.0},
            {OperationCode::Ry, 1, 0, 0.37, 0.0},
            {OperationCode::Cnot, 0, 1, 0.0, 0.0},
            {OperationCode::T, 1, 0, 0.0, 0.0},
            {OperationCode::Cz, 1, 2, 0.0, 0.0},
            {OperationCode::Rx, 2, 0, -0.21, 0.0},
            {OperationCode::Swap, 2, 3, 0.0, 0.0},
            {OperationCode::S, 3, 0, 0.0, 0.0},
        };
        QRegister evolved = initial;
        OperationPlan plan(operations, false);
        plan.execute(evolved);

        const auto observable = single_term(
            4,
            {{0, PauliAxis::X}, {1, PauliAxis::Y}, {3, PauliAxis::Z}});
        const QComplex direct = observable.expectation(evolved);
        const PauliObservable backward = observable.propagated_backward(operations);
        const QComplex propagated = backward.expectation(initial);
        require_close(
            propagated,
            direct,
            2e-11,
            "backward Pauli propagation does not match direct QRegister evolution");
        require(backward.term_count() > 1U,
                "non-Clifford propagation did not produce the expected sparse expansion");
    }

    {
        const auto observable = single_term(
            1,
            {{0, PauliAxis::X}},
            {1.0, 0.0},
            PauliPropagationConfig{1U});
        bool rejected = false;
        try {
            const std::array<Operation, 1> operations{
                Operation{OperationCode::Rz, 0, 0, 0.37, 0.0}};
            (void)observable.propagated_backward(operations);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "Pauli propagation did not fail closed at max_terms");
    }

    {
        auto observable = single_term(1, {{0, PauliAxis::Z}});
        bool rejected = false;
        try {
            const std::array<Operation, 1> operations{
                Operation{OperationCode::BitFlipTrajectory, 0, 0, 0.1, 0.5}};
            observable.propagate_backward(operations);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "Pauli propagation accepted a trajectory-noise operation");
    }

    {
        PauliObservable observable(2);
        const std::array<PauliFactor, 2> factors{
            PauliFactor{0, PauliAxis::X},
            PauliFactor{0, PauliAxis::Y},
        };
        observable.add_term({1.0, 0.0}, factors);
        QRegister state(2);
        require_close(
            observable.expectation(state),
            {0.0, 1.0},
            2e-12,
            "Pauli factor canonicalization lost its multiplication phase");
        std::string reason;
        require(observable.validate(&reason), "canonical Pauli observable failed validation");
        require(reason.empty(), "valid Pauli observable reported a validation reason");
    }

    std::cout << "pauli propagation tests passed\n";
    return 0;
}

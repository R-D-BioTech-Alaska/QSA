#include "qubit/qrouter.hpp"

#include <array>
#include <iostream>
#include <stdexcept>

namespace {

using qubit::Operation;
using qubit::OperationCode;
using qubit::QRegister;
using qubit::RepresentationAdvisor;
using qubit::RepresentationKind;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    RepresentationAdvisor advisor;
    QRegister state(64);

    {
        const std::array<Operation, 6> operations{{
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::Cnot, 0U, 1U, 0.0, 0.0},
            {OperationCode::S, 2U, 0U, 0.0, 0.0},
            {OperationCode::Cz, 1U, 2U, 0.0, 0.0},
            {OperationCode::X, 3U, 0U, 0.0, 0.0},
            {OperationCode::Swap, 3U, 4U, 0.0, 0.0},
        }};
        const auto features = RepresentationAdvisor::inspect_operations(
            state, operations, 100'000U);
        require(features.clifford_only,
                "Clifford operation list was not certified");
        require(features.stabilizer_input_certified,
                "computational-basis product input was not certified as stabilizer");
        require(!features.uniform_phase_graph,
                "operation certification invented phase-graph eligibility");
        require(advisor.recommend(features).kind == RepresentationKind::Stabilizer,
                "certified Clifford workload did not select stabilizer route");
    }

    {
        QRegister pauli_product(64);
        pauli_product.apply_h(0U);
        pauli_product.apply_h(1U);
        pauli_product.apply_s(1U);
        const std::array<Operation, 2> operations{{
            {OperationCode::H, 2U, 0U, 0.0, 0.0},
            {OperationCode::Cnot, 2U, 3U, 0.0, 0.0},
        }};
        const auto features = RepresentationAdvisor::inspect_operations(
            pauli_product, operations, 100'000U);
        require(features.clifford_only,
                "Pauli-eigenstate product workload lost Clifford certification");
        require(features.stabilizer_input_certified,
                "Pauli-eigenstate product input was not certified");
        require(advisor.recommend(features).kind == RepresentationKind::Stabilizer,
                "certified Pauli-eigenstate product did not select stabilizer route");
    }

    {
        QRegister magic(64);
        magic.apply_h(0U);
        magic.apply_t(0U);
        const std::array<Operation, 2> operations{{
            {OperationCode::H, 1U, 0U, 0.0, 0.0},
            {OperationCode::Cnot, 1U, 2U, 0.0, 0.0},
        }};
        const auto features = RepresentationAdvisor::inspect_operations(
            magic, operations, 100'000U);
        require(features.clifford_only,
                "future Clifford circuit was not recognized");
        require(!features.stabilizer_input_certified,
                "magic input was incorrectly certified as a stabilizer state");
        require(advisor.recommend(features).kind == RepresentationKind::Register,
                "magic input escaped the exact QRegister fallback");
    }

    {
        QRegister entangled(64);
        entangled.apply_h(0U);
        entangled.apply_cnot(0U, 1U);
        const std::array<Operation, 1> operations{{
            {OperationCode::H, 2U, 0U, 0.0, 0.0},
        }};
        const auto features = RepresentationAdvisor::inspect_operations(
            entangled, operations, 100'000U);
        require(features.clifford_only,
                "future Clifford circuit was not recognized for entangled input");
        require(!features.stabilizer_input_certified,
                "entangled input was certified without an exact stabilizer-state proof");
        require(advisor.recommend(features).kind == RepresentationKind::Register,
                "uncertified entangled input escaped the exact QRegister fallback");
    }

    {
        const std::array<Operation, 2> operations{{
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::T, 0U, 0U, 0.0, 0.0},
        }};
        const auto features = RepresentationAdvisor::inspect_operations(
            state, operations, 100'000U);
        require(!features.clifford_only,
                "T gate was incorrectly certified as Clifford");
        require(!features.stabilizer_input_certified,
                "non-Clifford workload retained stabilizer eligibility");
        require(advisor.recommend(features).kind == RepresentationKind::Register,
                "non-Clifford workload escaped the exact general fallback");
    }

    {
        const std::array<Operation, 1> operations{{
            {OperationCode::AmplitudeDampingTrajectory, 0U, 0U, 0.2, 0.5},
        }};
        const auto features = RepresentationAdvisor::inspect_operations(
            state, operations, 100'000U);
        require(!features.clifford_only,
                "trajectory noise was incorrectly certified as Clifford");
        require(!features.stabilizer_input_certified,
                "trajectory workload retained stabilizer eligibility");
        require(advisor.recommend(features).kind == RepresentationKind::Register,
                "trajectory workload escaped the exact general fallback");
    }

    {
        const std::array<Operation, 0> operations{};
        const auto features = RepresentationAdvisor::inspect_operations(
            state, operations, 1U);
        require(features.clifford_only,
                "empty operation list should be Clifford compatible");
        require(features.stabilizer_input_certified,
                "computational-basis product was not certified for an empty circuit");
    }

    std::cout << "representation certification tests passed\n";
    return 0;
}

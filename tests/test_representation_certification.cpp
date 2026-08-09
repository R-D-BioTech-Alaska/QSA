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
        require(!features.uniform_phase_graph,
                "operation certification invented phase-graph eligibility");
        require(advisor.recommend(features).kind == RepresentationKind::Stabilizer,
                "certified Clifford workload did not select stabilizer route");
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
        require(advisor.recommend(features).kind == RepresentationKind::Register,
                "trajectory workload escaped the exact general fallback");
    }

    {
        const std::array<Operation, 0> operations{};
        const auto features = RepresentationAdvisor::inspect_operations(
            state, operations, 1U);
        require(features.clifford_only,
                "empty operation list should be Clifford compatible");
    }

    std::cout << "representation certification tests passed\n";
    return 0;
}

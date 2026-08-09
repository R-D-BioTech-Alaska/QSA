#include "qubit/qbroker.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using qubit::BasisIndex;
using qubit::ExactExecutionBroker;
using qubit::ExactExecutionBrokerConfig;
using qubit::ExactExecutionRoute;
using qubit::Operation;
using qubit::OperationCode;
using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
using qubit::PauliPropagationConfig;
using qubit::QComplex;
using qubit::QRegister;
using qubit::QStateError;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

QRegister evolve(QRegister state, std::span<const Operation> operations) {
    qubit::OperationPlan plan(operations);
    plan.execute(state);
    return state;
}

void require_close(const QComplex& actual, const QComplex& expected, const std::string& message) {
    require(qubit::almost_equal(actual, expected, 2e-11), message);
}

}  // namespace

int main() {
    {
        QRegister input(128U);
        input.apply_h(0U);
        input.apply_cnot(0U, 1U);
        std::vector<Operation> operations;
        operations.reserve(2'002U);
        for (std::size_t index = 0; index < 2'000U; ++index) {
            operations.push_back({OperationCode::Rz, 64U, 0U, 0.0001 * static_cast<double>(index + 1U), 0.0});
        }
        operations.push_back({OperationCode::H, 0U, 0U, 0.0, 0.0});
        operations.push_back({OperationCode::Cnot, 0U, 1U, 0.0, 0.0});

        PauliObservable observable(128U);
        const std::array<PauliFactor, 1> factors{{{1U, PauliAxis::Z}}};
        observable.add_term({1.0, 0.0}, factors);

        ExactExecutionBroker broker;
        const auto result = broker.expectation(input, operations, observable);
        const QRegister direct = evolve(input, operations);
        require(result.route == ExactExecutionRoute::CausalPauli,
                "bounded observable did not use causal Pauli route");
        require(result.pauli_stats.visited_operations < operations.size(),
                "causal Pauli route did not prune irrelevant operations");
        require_close(result.value, observable.expectation(direct),
                      "causal Pauli broker value differs from direct execution");
        require(result.fallback_reason.empty(),
                "successful causal Pauli route reported a fallback reason");
    }

    {
        QRegister input(3U);
        input.apply_h(0U);
        PauliPropagationConfig pauli_config;
        pauli_config.max_terms = 1U;
        PauliObservable observable(3U, pauli_config);
        const std::array<PauliFactor, 1> factors{{{0U, PauliAxis::X}}};
        observable.add_term({1.0, 0.0}, factors);
        const std::array<Operation, 1> operations{{
            {OperationCode::Rz, 0U, 0U, 0.37, 0.0},
        }};

        ExactExecutionBroker broker;
        const auto result = broker.expectation(input, operations, observable);
        const QRegister direct = evolve(input, operations);
        require(result.route == ExactExecutionRoute::Register,
                "Pauli term-cap collapse did not fall back to QRegister");
        require(!result.fallback_reason.empty(),
                "term-cap fallback did not record its reason");
        require_close(result.value, observable.expectation(direct),
                      "term-cap fallback changed the observable value");
    }

    {
        QRegister input(2U);
        input.apply_h(0U);
        PauliObservable observable(2U);
        const std::array<PauliFactor, 1> factors{{{0U, PauliAxis::Z}}};
        observable.add_term({1.0, 0.0}, factors);
        const std::array<Operation, 1> operations{{
            {OperationCode::AmplitudeDampingTrajectory, 0U, 0U, 0.25, 0.1},
        }};

        ExactExecutionBroker broker;
        const auto result = broker.expectation(input, operations, observable);
        const QRegister direct = evolve(input, operations);
        require(result.route == ExactExecutionRoute::Register,
                "trajectory observable did not fall back to QRegister");
        require(!result.fallback_reason.empty(),
                "trajectory fallback did not record its reason");
        require_close(result.value, observable.expectation(direct),
                      "trajectory fallback changed the observable value");
    }

    {
        const std::array<Operation, 5> operations{{
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::Cnot, 0U, 1U, 0.0, 0.0},
            {OperationCode::Ry, 2U, 0U, 0.19, 0.0},
            {OperationCode::Cz, 1U, 2U, 0.0, 0.0},
            {OperationCode::Rz, 0U, 0U, -0.23, 0.0},
        }};
        const std::array<std::uint8_t, 3> bits{{1U, 0U, 1U}};

        ExactExecutionBroker broker;
        const auto result = broker.amplitude_from_zero(3U, operations, bits);
        const QRegister direct = evolve(QRegister(3U), operations);
        require(result.route == ExactExecutionRoute::TensorNetwork,
                "bounded unitary amplitude did not use tensor route");
        require_close(result.value, direct.amplitude_bits(bits),
                      "tensor broker amplitude differs from QRegister");
        require(result.fallback_reason.empty(),
                "successful tensor route reported a fallback reason");
    }

    {
        ExactExecutionBrokerConfig config;
        config.tensor.max_contraction_entries = 8U;
        ExactExecutionBroker broker(config);
        const std::array<Operation, 2> operations{{
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::Cnot, 0U, 1U, 0.0, 0.0},
        }};
        const auto result = broker.amplitude_from_zero(2U, operations, BasisIndex{0});
        const QRegister direct = evolve(QRegister(2U), operations);
        require(result.route == ExactExecutionRoute::Register,
                "tensor width collapse did not fall back to QRegister");
        require(!result.fallback_reason.empty(),
                "tensor width fallback did not record its reason");
        require_close(result.value, direct.amplitude(0U),
                      "tensor width fallback changed the amplitude");
    }

    {
        ExactExecutionBroker broker;
        const std::array<Operation, 2> operations{{
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::BitFlipTrajectory, 0U, 0U, 1.0, 0.0},
        }};
        const auto result = broker.amplitude_from_zero(1U, operations, BasisIndex{1});
        const QRegister direct = evolve(QRegister(1U), operations);
        require(result.route == ExactExecutionRoute::Register,
                "trajectory amplitude did not fall back to QRegister");
        require(!result.fallback_reason.empty(),
                "trajectory amplitude fallback did not record its reason");
        require_close(result.value, direct.amplitude(1U),
                      "trajectory amplitude fallback changed the amplitude");
    }

    {
        constexpr std::size_t qubits = 80U;
        std::vector<Operation> operations;
        operations.reserve(160U);
        for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
            operations.push_back({
                OperationCode::Ry,
                static_cast<qubit::QubitId>(qubit),
                0U,
                0.001 * static_cast<double>(qubit + 1U),
                0.0,
            });
        }
        for (std::size_t qubit = 0; qubit + 1U < qubits; qubit += 2U) {
            operations.push_back({
                OperationCode::Cnot,
                static_cast<qubit::QubitId>(qubit),
                static_cast<qubit::QubitId>(qubit + 1U),
                0.0,
                0.0,
            });
        }
        std::vector<std::uint8_t> bits(qubits, 0U);
        ExactExecutionBroker broker;
        const auto result = broker.amplitude_from_zero(qubits, operations, bits);
        long double expected = 1.0L;
        for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
            expected *= std::cos(0.0005 * static_cast<double>(qubit + 1U));
        }
        require(result.route == ExactExecutionRoute::TensorNetwork,
                "wide bounded circuit did not stay on tensor route");
        require(std::abs(result.value.re - static_cast<double>(expected)) <= 2e-11 &&
                    std::abs(result.value.im) <= 2e-11,
                "wide tensor broker amplitude differs from analytic result");
    }

    {
        ExactExecutionBroker broker;
        bool rejected = false;
        try {
            const std::array<std::uint8_t, 2> bad_bits{{0U, 2U}};
            static_cast<void>(broker.amplitude_from_zero(2U, {}, bad_bits));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "broker accepted an invalid basis bit");
    }

    std::cout << "execution broker tests passed\n";
    return 0;
}

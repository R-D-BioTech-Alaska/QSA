#include "qubit/qbroker.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

using qubit::BasisIndex;
using qubit::ExactExecutionBroker;
using qubit::ExactExecutionBrokerConfig;
using qubit::ExactExecutionRoute;
using qubit::ExactPreparedProbabilityPlan;
using qubit::Operation;
using qubit::OperationCode;
using qubit::QStateError;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(double actual, double expected, const std::string& message) {
    require(std::abs(actual - expected) <= 2e-11, message);
}

void require_matches_one_shot(
    const ExactExecutionBroker& broker,
    const ExactPreparedProbabilityPlan& prepared,
    std::span<const Operation> operations,
    std::span<const std::uint8_t> bits,
    const std::string& message) {
    const auto one_shot = broker.basis_probability_from_zero(
        prepared.qubit_count(), operations, bits);
    const auto compiled = prepared.probability(bits);
    require(compiled.route == one_shot.route, message + " route mismatch");
    require_close(compiled.value, one_shot.value, message + " value mismatch");
}

}  // namespace

int main() {
    {
        const std::array<Operation, 12> operations{{
            {OperationCode::X, 0U, 0U, 0.0, 0.0},
            {OperationCode::Cnot, 1U, 2U, 0.0, 0.0},
            {OperationCode::Y, 1U, 0U, 0.0, 0.0},
            {OperationCode::Cnot, 1U, 2U, 0.0, 0.0},
            {OperationCode::Z, 2U, 0U, 0.0, 0.0},
            {OperationCode::S, 3U, 0U, 0.0, 0.0},
            {OperationCode::Sdg, 3U, 0U, 0.0, 0.0},
            {OperationCode::T, 1U, 0U, 0.0, 0.0},
            {OperationCode::Tdg, 2U, 0U, 0.0, 0.0},
            {OperationCode::Rz, 0U, 0U, 0.37, 0.0},
            {OperationCode::Cz, 1U, 2U, 0.0, 0.0},
            {OperationCode::Swap, 0U, 3U, 0.0, 0.0},
        }};
        const std::array<std::uint8_t, 4> hit{{0U, 1U, 1U, 1U}};
        const std::array<std::uint8_t, 4> miss{{1U, 1U, 1U, 1U}};

        ExactExecutionBroker broker;
        ExactPreparedProbabilityPlan prepared(4U, operations);
        require(prepared.prepared_route() == ExactExecutionRoute::BasisPermutation,
                "prepared monomial circuit did not select BasisPermutation");
        require_matches_one_shot(broker, prepared, operations, hit,
                                 "prepared BasisPermutation hit");
        require_matches_one_shot(broker, prepared, operations, miss,
                                 "prepared BasisPermutation miss");
        require(prepared.probability(hit).value == 1.0 &&
                    prepared.probability(miss).value == 0.0,
                "prepared BasisPermutation did not preserve deterministic probabilities");
        require(prepared.estimated_bytes() > 0U,
                "prepared BasisPermutation did not report retained memory");
    }

    {
        const std::array<Operation, 9> operations{{
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::H, 1U, 0U, 0.0, 0.0},
            {OperationCode::H, 2U, 0U, 0.0, 0.0},
            {OperationCode::H, 3U, 0U, 0.0, 0.0},
            {OperationCode::Cz, 0U, 3U, 0.0, 0.0},
            {OperationCode::Rz, 1U, 0U, 0.37, 0.0},
            {OperationCode::T, 2U, 0U, 0.0, 0.0},
            {OperationCode::Sdg, 3U, 0U, 0.0, 0.0},
            {OperationCode::Swap, 0U, 2U, 0.0, 0.0},
        }};
        const std::array<std::uint8_t, 4> bits{{1U, 0U, 1U, 1U}};

        ExactExecutionBroker broker;
        ExactPreparedProbabilityPlan prepared(4U, operations);
        require(prepared.prepared_route() == ExactExecutionRoute::UniformMagnitude,
                "prepared uniform circuit did not select UniformMagnitude");
        require_matches_one_shot(broker, prepared, operations, bits,
                                 "prepared UniformMagnitude");
        require(prepared.probability(bits).fallback_reason.empty(),
                "prepared UniformMagnitude reported a fallback reason");
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
        ExactPreparedProbabilityPlan prepared(3U, operations);
        require(prepared.prepared_route() == ExactExecutionRoute::TensorNetwork,
                "prepared bounded circuit did not select TensorNetwork");
        require_matches_one_shot(broker, prepared, operations, bits,
                                 "prepared TensorNetwork");
        require(prepared.probability(bits).fallback_reason.empty(),
                "prepared TensorNetwork success reported a fallback reason");
    }

    {
        ExactExecutionBrokerConfig config;
        config.tensor.max_contraction_entries = 8U;
        const std::array<Operation, 2> operations{{
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::Cnot, 0U, 1U, 0.0, 0.0},
        }};
        const std::array<std::uint8_t, 2> bits{{0U, 0U}};

        ExactExecutionBroker broker(config);
        ExactPreparedProbabilityPlan prepared(2U, operations, config);
        require(prepared.prepared_route() == ExactExecutionRoute::PersistentMPS,
                "prepared tensor-collapse circuit did not select PersistentMPS");
        require_matches_one_shot(broker, prepared, operations, bits,
                                 "prepared PersistentMPS");
        require(prepared.probability(bits).fallback_reason.find("tensor:") != std::string::npos,
                "prepared PersistentMPS did not retain tensor rejection reason");
    }

    {
        ExactExecutionBrokerConfig config;
        config.tensor.max_contraction_entries = 2U;
        config.mps.max_bond_dimension = 1U;
        const std::array<Operation, 2> operations{{
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::Cnot, 0U, 1U, 0.0, 0.0},
        }};
        const std::array<std::uint8_t, 2> bits{{0U, 0U}};

        ExactExecutionBroker broker(config);
        ExactPreparedProbabilityPlan prepared(2U, operations, config);
        require(prepared.prepared_route() == ExactExecutionRoute::Register,
                "prepared exhausted circuit did not select QRegister");
        require_matches_one_shot(broker, prepared, operations, bits,
                                 "prepared QRegister");
        const auto result = prepared.probability(BasisIndex{0});
        require(result.route == ExactExecutionRoute::Register,
                "prepared BasisIndex query changed the retained route");
        require_close(result.value, broker.basis_probability_from_zero(
                                      2U, operations, BasisIndex{0}).value,
                      "prepared BasisIndex query differs from one-shot broker");
        require(result.fallback_reason.find("phase_graph:") != std::string::npos,
                "prepared QRegister did not retain specialized rejection reasons");
    }

    {
        ExactPreparedProbabilityPlan prepared(2U, {});
        bool rejected = false;
        try {
            const std::array<std::uint8_t, 2> bad_bits{{0U, 2U}};
            static_cast<void>(prepared.probability(bad_bits));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "prepared probability plan accepted an invalid basis bit");
    }

    return 0;
}

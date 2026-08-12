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
using qubit::QubitId;

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

        const std::array<QubitId, 2> marginal_qubits{{1U, 3U}};
        const std::array<std::uint8_t, 2> marginal_hit{{1U, 1U}};
        const std::array<std::uint8_t, 2> marginal_miss{{0U, 1U}};
        const auto one_shot_marginal = broker.marginal_probability_from_zero(
            4U, operations, marginal_qubits, marginal_hit);
        const auto prepared_marginal = prepared.marginal_probability(
            marginal_qubits, marginal_hit);
        require(one_shot_marginal.route == ExactExecutionRoute::BasisPermutation &&
                    prepared_marginal.route == ExactExecutionRoute::BasisPermutation,
                "BasisPermutation marginal query changed route");
        require(one_shot_marginal.value == 1.0 && prepared_marginal.value == 1.0,
                "BasisPermutation marginal hit was not exact");
        require(prepared.marginal_probability(marginal_qubits, marginal_miss).value == 0.0,
                "BasisPermutation marginal miss was not exact");
        const std::array<QubitId, 0> empty_qubits{};
        const std::array<std::uint8_t, 0> empty_bits{};
        require(prepared.marginal_probability(empty_qubits, empty_bits).value == 1.0,
                "BasisPermutation empty marginal was not one");
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

        const std::array<QubitId, 2> marginal_qubits{{0U, 2U}};
        const std::array<std::uint8_t, 2> marginal_bits{{1U, 0U}};
        const auto one_shot_marginal = broker.marginal_probability_from_zero(
            4U, operations, marginal_qubits, marginal_bits);
        const auto prepared_marginal = prepared.marginal_probability(
            marginal_qubits, marginal_bits);
        require(one_shot_marginal.route == ExactExecutionRoute::UniformMagnitude &&
                    prepared_marginal.route == ExactExecutionRoute::UniformMagnitude,
                "UniformMagnitude marginal query changed route");
        require_close(one_shot_marginal.value, 0.25,
                      "UniformMagnitude one-shot marginal is not 2^-k");
        require_close(prepared_marginal.value, 0.25,
                      "UniformMagnitude prepared marginal is not 2^-k");
        require_close(prepared.probability(bits).value, 0.0625,
                      "UniformMagnitude full basis probability changed");
    }

    {
        const std::array<Operation, 3> operations{{
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::Cnot, 0U, 1U, 0.0, 0.0},
            {OperationCode::Cnot, 1U, 2U, 0.0, 0.0},
        }};
        const std::array<std::uint8_t, 3> zero{{0U, 0U, 0U}};
        const std::array<std::uint8_t, 3> one{{1U, 1U, 1U}};
        const std::array<std::uint8_t, 3> miss{{0U, 1U, 0U}};

        ExactExecutionBroker broker;
        ExactPreparedProbabilityPlan prepared(3U, operations);
        require(prepared.prepared_route() == ExactExecutionRoute::Stabilizer,
                "prepared GHZ circuit did not select Stabilizer");
        require_matches_one_shot(broker, prepared, operations, zero,
                                 "prepared Stabilizer zero branch");
        require_matches_one_shot(broker, prepared, operations, one,
                                 "prepared Stabilizer one branch");
        require_matches_one_shot(broker, prepared, operations, miss,
                                 "prepared Stabilizer impossible branch");
        require_close(prepared.probability(zero).value, 0.5,
                      "prepared Stabilizer P(000) is wrong");
        require_close(prepared.probability(one).value, 0.5,
                      "prepared Stabilizer P(111) is wrong");
        require(prepared.probability(miss).value == 0.0,
                "prepared Stabilizer impossible basis state was nonzero");

        const std::array<QubitId, 1> q0{{0U}};
        const std::array<std::uint8_t, 1> q0_zero{{0U}};
        const std::array<std::uint8_t, 1> q0_one{{1U}};
        require_close(prepared.marginal_probability(q0, q0_zero).value, 0.5,
                      "prepared Stabilizer marginal P(q0=0) is wrong");
        require_close(prepared.marginal_probability(q0, q0_one).value, 0.5,
                      "prepared Stabilizer marginal P(q0=1) is wrong");

        const std::array<QubitId, 2> pair{{0U, 2U}};
        const std::array<std::uint8_t, 2> pair_hit{{1U, 1U}};
        const std::array<std::uint8_t, 2> pair_miss{{1U, 0U}};
        const auto one_shot_pair = broker.marginal_probability_from_zero(
            3U, operations, pair, pair_hit);
        require(one_shot_pair.route == ExactExecutionRoute::Stabilizer,
                "one-shot GHZ marginal did not select Stabilizer");
        require_close(one_shot_pair.value, 0.5,
                      "one-shot Stabilizer pair marginal is wrong");
        require_close(prepared.marginal_probability(pair, pair_hit).value, 0.5,
                      "prepared Stabilizer pair marginal is wrong");
        require(prepared.marginal_probability(pair, pair_miss).value == 0.0,
                "prepared Stabilizer inconsistent marginal was nonzero");
        require(prepared.estimated_bytes() > 0U,
                "prepared Stabilizer did not report retained memory");
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

        bool marginal_rejected = false;
        try {
            const std::array<QubitId, 1> marginal_qubits{{0U}};
            const std::array<std::uint8_t, 1> marginal_bits{{1U}};
            static_cast<void>(prepared.marginal_probability(marginal_qubits, marginal_bits));
        } catch (const QStateError&) {
            marginal_rejected = true;
        }
        require(marginal_rejected,
                "prepared nonanalytic route accepted an unsupported marginal query");
    }

    {
        ExactExecutionBrokerConfig config;
        config.tensor.max_contraction_entries = 2U;
        const std::array<Operation, 3> operations{{
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::Cnot, 0U, 1U, 0.0, 0.0},
            {OperationCode::Ry, 1U, 0U, 0.19, 0.0},
        }};
        const std::array<std::uint8_t, 2> bits{{0U, 0U}};

        ExactExecutionBroker broker(config);
        ExactPreparedProbabilityPlan prepared(2U, operations, config);
        require(prepared.prepared_route() == ExactExecutionRoute::PersistentMPS,
                "prepared tensor-collapse circuit did not select PersistentMPS");
        require_matches_one_shot(broker, prepared, operations, bits,
                                 "prepared PersistentMPS");
        require(prepared.probability(bits).fallback_reason.find("stabilizer:") != std::string::npos &&
                    prepared.probability(bits).fallback_reason.find("tensor:") != std::string::npos,
                "prepared PersistentMPS did not retain upstream rejection reasons");

        const std::array<QubitId, 1> q0{{0U}};
        const std::array<std::uint8_t, 1> q0_zero{{0U}};
        const auto one_shot_marginal = broker.marginal_probability_from_zero(
            2U, operations, q0, q0_zero);
        const auto prepared_marginal = prepared.marginal_probability(q0, q0_zero);
        require(one_shot_marginal.route == ExactExecutionRoute::PersistentMPS &&
                    prepared_marginal.route == ExactExecutionRoute::PersistentMPS,
                "PersistentMPS marginal query changed route");
        require_close(one_shot_marginal.value, 0.5,
                      "PersistentMPS one-shot marginal is wrong");
        require_close(prepared_marginal.value, 0.5,
                      "PersistentMPS prepared marginal is wrong");

        const std::array<QubitId, 2> pair{{0U, 1U}};
        const std::array<std::uint8_t, 2> pair_zero{{0U, 0U}};
        const double expected_pair = 0.5 * std::pow(std::cos(0.19 / 2.0), 2.0);
        require_close(prepared.marginal_probability(pair, pair_zero).value, expected_pair,
                      "PersistentMPS joint marginal is wrong");
    }

    {
        ExactExecutionBrokerConfig config;
        config.tensor.max_contraction_entries = 2U;
        config.mps.max_bond_dimension = 1U;
        const std::array<Operation, 3> operations{{
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::Cnot, 0U, 1U, 0.0, 0.0},
            {OperationCode::Ry, 0U, 0U, 0.19, 0.0},
        }};
        const std::array<std::uint8_t, 2> bits{{0U, 0U}};

        ExactExecutionBroker broker(config);
        ExactPreparedProbabilityPlan prepared(2U, operations, config);
        require(prepared.prepared_route() == ExactExecutionRoute::Register,
                "prepared exhausted non-Clifford circuit did not select QRegister");
        require_matches_one_shot(broker, prepared, operations, bits,
                                 "prepared QRegister");
        const auto result = prepared.probability(BasisIndex{0});
        require(result.route == ExactExecutionRoute::Register,
                "prepared BasisIndex query changed the retained route");
        require_close(result.value, broker.basis_probability_from_zero(
                                      2U, operations, BasisIndex{0}).value,
                      "prepared BasisIndex query differs from one-shot broker");
        require(result.fallback_reason.find("stabilizer:") != std::string::npos &&
                    result.fallback_reason.find("phase_graph:") != std::string::npos,
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

        const std::array<QubitId, 2> duplicate_qubits{{0U, 0U}};
        const std::array<std::uint8_t, 2> duplicate_bits{{0U, 0U}};
        rejected = false;
        try {
            static_cast<void>(prepared.marginal_probability(duplicate_qubits, duplicate_bits));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "prepared marginal query accepted duplicate qubits");

        const std::array<QubitId, 1> out_of_range_qubit{{2U}};
        const std::array<std::uint8_t, 1> valid_bit{{0U}};
        rejected = false;
        try {
            static_cast<void>(prepared.marginal_probability(out_of_range_qubit, valid_bit));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "prepared marginal query accepted an out-of-range qubit");

        const std::array<QubitId, 1> one_qubit{{0U}};
        const std::array<std::uint8_t, 2> two_bits{{0U, 0U}};
        rejected = false;
        try {
            static_cast<void>(prepared.marginal_probability(one_qubit, two_bits));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "prepared marginal query accepted mismatched query widths");
    }

    return 0;
}

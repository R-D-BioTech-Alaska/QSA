#include "qubit/qbroker.hpp"

#include <array>
#include <cmath>
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
using qubit::QRegister;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(double actual, double expected, const std::string& message) {
    require(std::abs(actual - expected) <= 2e-11, message);
}

[[nodiscard]] QRegister evolve(
    std::size_t qubits,
    std::span<const Operation> operations) {
    QRegister state(qubits);
    qubit::OperationPlan plan(operations);
    plan.execute(state);
    return state;
}

}  // namespace

int main() {
    const std::array<Operation, 3> eligible{{
        {OperationCode::H, 0U, 0U, 0.0, 0.0},
        {OperationCode::Cnot, 0U, 1U, 0.0, 0.0},
        {OperationCode::Ry, 1U, 0U, 0.19, 0.0},
    }};
    const std::array<std::uint8_t, 2> zero{{0U, 0U}};
    const double expected = evolve(2U, eligible).amplitude_bits(zero).norm2();

    ExactExecutionBrokerConfig baseline_config;
    baseline_config.tensor_planning_defer_variables = 0U;
    ExactExecutionBroker baseline_broker(baseline_config);
    const auto baseline = baseline_broker.basis_probability_from_zero(2U, eligible, zero);
    require(baseline.route == ExactExecutionRoute::TensorNetwork,
            "disabled Tensor planning deferral changed the exact route");
    require(baseline.fallback_reason.empty(),
            "disabled Tensor planning deferral reported a fallback reason");
    require_close(baseline.value, expected,
                  "disabled Tensor planning deferral changed probability");

    ExactPreparedProbabilityPlan baseline_prepared(2U, eligible, baseline_config);
    require(baseline_prepared.prepared_route() == ExactExecutionRoute::TensorNetwork,
            "disabled prepared Tensor planning deferral changed the route");
    require_close(baseline_prepared.probability(zero).value, expected,
                  "disabled prepared Tensor planning deferral changed probability");

    ExactExecutionBrokerConfig deferred_config = baseline_config;
    deferred_config.tensor_planning_defer_variables = 1U;
    ExactExecutionBroker deferred_broker(deferred_config);
    const auto deferred = deferred_broker.basis_probability_from_zero(2U, eligible, zero);
    require(deferred.route == ExactExecutionRoute::PersistentMPS,
            "certified Tensor planning deferral did not select PersistentMPS");
    require(deferred.fallback_reason.find("tensor: planning deferred to certified MPS") !=
                std::string::npos,
            "Tensor planning deferral reason is not visible");
    require_close(deferred.value, expected,
                  "deferred one-shot MPS probability differs from QRegister");

    ExactPreparedProbabilityPlan deferred_prepared(2U, eligible, deferred_config);
    require(deferred_prepared.prepared_route() == ExactExecutionRoute::PersistentMPS,
            "certified prepared Tensor planning deferral did not select PersistentMPS");
    const auto deferred_prepared_result = deferred_prepared.probability(zero);
    require(deferred_prepared_result.fallback_reason.find(
                "tensor: planning deferred to certified MPS") != std::string::npos,
            "prepared Tensor planning deferral reason is not visible");
    require_close(deferred_prepared_result.value, expected,
                  "deferred prepared MPS probability differs from QRegister");

    ExactExecutionBrokerConfig exhausted_config = deferred_config;
    exhausted_config.mps.max_bond_dimension = 1U;
    ExactExecutionBroker exhausted_broker(exhausted_config);
    const auto exhausted = exhausted_broker.basis_probability_from_zero(2U, eligible, zero);
    require(exhausted.route == ExactExecutionRoute::TensorNetwork,
            "MPS resource certificate did not keep TensorNetwork first");
    require(exhausted.fallback_reason.empty(),
            "resource-ineligible MPS was attempted before TensorNetwork");
    require_close(exhausted.value, expected,
                  "resource-certified TensorNetwork probability differs from QRegister");

    ExactPreparedProbabilityPlan exhausted_prepared(2U, eligible, exhausted_config);
    require(exhausted_prepared.prepared_route() == ExactExecutionRoute::TensorNetwork,
            "prepared MPS resource certificate did not keep TensorNetwork first");
    const auto exhausted_prepared_result = exhausted_prepared.probability(zero);
    require(exhausted_prepared_result.fallback_reason.empty(),
            "prepared resource-ineligible MPS was attempted before TensorNetwork");
    require_close(exhausted_prepared_result.value, expected,
                  "prepared resource-certified TensorNetwork probability differs from QRegister");

    const std::array<Operation, 5> repeated_cut{{
        {OperationCode::H, 0U, 0U, 0.0, 0.0},
        {OperationCode::Cnot, 0U, 1U, 0.0, 0.0},
        {OperationCode::Cz, 0U, 1U, 0.0, 0.0},
        {OperationCode::Cnot, 1U, 0U, 0.0, 0.0},
        {OperationCode::Ry, 1U, 0U, 0.17, 0.0},
    }};
    const double repeated_expected = evolve(2U, repeated_cut).amplitude_bits(zero).norm2();
    ExactExecutionBrokerConfig repeated_config = deferred_config;
    repeated_config.mps.max_bond_dimension = 4U;
    const auto repeated = ExactExecutionBroker(repeated_config).basis_probability_from_zero(
        2U, repeated_cut, zero);
    require(repeated.route == ExactExecutionRoute::TensorNetwork,
            "repeated-cut MPS bond growth was not rejected before TensorNetwork");
    require(repeated.fallback_reason.empty(),
            "repeated-cut resource rejection incorrectly reported Tensor deferral");
    require_close(repeated.value, repeated_expected,
                  "repeated-cut TensorNetwork probability differs from QRegister");

    ExactExecutionBrokerConfig scalar_config = deferred_config;
    scalar_config.mps.max_scalars = 7U;
    const auto scalar_limited = ExactExecutionBroker(scalar_config).basis_probability_from_zero(
        2U, eligible, zero);
    require(scalar_limited.route == ExactExecutionRoute::TensorNetwork,
            "MPS scalar certificate did not keep TensorNetwork first");
    require(scalar_limited.fallback_reason.empty(),
            "scalar-ineligible MPS incorrectly reported Tensor deferral");
    require_close(scalar_limited.value, expected,
                  "scalar-certified TensorNetwork probability differs from QRegister");

    const std::array<Operation, 3> non_mps{{
        {OperationCode::H, 0U, 0U, 0.0, 0.0},
        {OperationCode::Cnot, 0U, 2U, 0.0, 0.0},
        {OperationCode::Ry, 2U, 0U, 0.23, 0.0},
    }};
    const double non_mps_expected = evolve(3U, non_mps).amplitude(BasisIndex{0}).norm2();
    const auto non_mps_result = deferred_broker.basis_probability_from_zero(
        3U, non_mps, BasisIndex{0});
    require(non_mps_result.route == ExactExecutionRoute::TensorNetwork,
            "non-MPS topology was incorrectly deferred away from TensorNetwork");
    require(non_mps_result.fallback_reason.empty(),
            "non-MPS TensorNetwork route reported a deferral reason");
    require_close(non_mps_result.value, non_mps_expected,
                  "non-MPS TensorNetwork probability differs from QRegister");

    ExactPreparedProbabilityPlan non_mps_prepared(3U, non_mps, deferred_config);
    require(non_mps_prepared.prepared_route() == ExactExecutionRoute::TensorNetwork,
            "prepared non-MPS topology was incorrectly deferred from TensorNetwork");
    require_close(non_mps_prepared.probability(BasisIndex{0}).value, non_mps_expected,
                  "prepared non-MPS TensorNetwork probability differs from QRegister");

    return 0;
}

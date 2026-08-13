#include "qubit/qbroker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

using qubit::ExactExecutionBroker;
using qubit::ExactExecutionBrokerConfig;
using qubit::ExactExecutionRoute;
using qubit::ExactPreparedProbabilityPlan;
using qubit::Operation;
using qubit::OperationCode;
using qubit::QRegister;
using qubit::QStateError;
using qubit::QubitId;

static_assert(static_cast<std::uint8_t>(ExactExecutionRoute::Register) == 0U);
static_assert(static_cast<std::uint8_t>(ExactExecutionRoute::CausalPauli) == 1U);
static_assert(static_cast<std::uint8_t>(ExactExecutionRoute::TensorNetwork) == 2U);
static_assert(static_cast<std::uint8_t>(ExactExecutionRoute::PersistentMPS) == 3U);
static_assert(static_cast<std::uint8_t>(ExactExecutionRoute::PhaseGraph) == 4U);
static_assert(static_cast<std::uint8_t>(ExactExecutionRoute::UniformMagnitude) == 5U);
static_assert(static_cast<std::uint8_t>(ExactExecutionRoute::BasisPermutation) == 6U);
static_assert(static_cast<std::uint8_t>(ExactExecutionRoute::Stabilizer) == 7U);
static_assert(static_cast<std::uint8_t>(ExactExecutionRoute::TreeTensor) == 8U);

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(double actual, double expected, const std::string& message) {
    const double scale = 1.0 + std::max(std::abs(actual), std::abs(expected));
    require(std::abs(actual - expected) <= 5e-10 * scale, message);
}

QRegister evolve(std::size_t qubits, std::span<const Operation> operations) {
    QRegister state(qubits);
    qubit::OperationPlan plan(operations);
    plan.execute(state);
    return state;
}

void route_precedence() {
    require(std::string(qubit::exact_execution_route_name(ExactExecutionRoute::TreeTensor)) ==
                "TreeTensor",
            "TreeTensor route name changed");

    {
        const std::array<Operation, 1> operations{{
            {OperationCode::X, 0U, 0U, 0.0, 0.0},
        }};
        const auto plan = ExactPreparedProbabilityPlan::for_marginals(2U, operations);
        require(plan.prepared_route() == ExactExecutionRoute::BasisPermutation,
                "TreeTensor displaced BasisPermutation marginal authority");
    }
    {
        const std::array<Operation, 3> operations{{
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::H, 1U, 0U, 0.0, 0.0},
            {OperationCode::Cz, 0U, 1U, 0.0, 0.0},
        }};
        const auto plan = ExactPreparedProbabilityPlan::for_marginals(2U, operations);
        require(plan.prepared_route() == ExactExecutionRoute::UniformMagnitude,
                "TreeTensor displaced UniformMagnitude marginal authority");
    }
    {
        const std::array<Operation, 3> operations{{
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::Cnot, 0U, 2U, 0.0, 0.0},
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
        }};
        const auto plan = ExactPreparedProbabilityPlan::for_marginals(3U, operations);
        require(plan.prepared_route() == ExactExecutionRoute::Stabilizer,
                "TreeTensor displaced Stabilizer marginal authority");
    }
    {
        const std::array<Operation, 3> operations{{
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::Cnot, 0U, 1U, 0.0, 0.0},
            {OperationCode::Ry, 0U, 0U, 0.19, 0.0},
        }};
        const auto plan = ExactPreparedProbabilityPlan::for_marginals(2U, operations);
        require(plan.prepared_route() == ExactExecutionRoute::PersistentMPS,
                "TreeTensor displaced PersistentMPS marginal authority");
    }
}

std::array<Operation, 4> ttn_carrier() {
    return {{
        {OperationCode::Ry, 0U, 0U, 0.37, 0.0},
        {OperationCode::H, 1U, 0U, 0.0, 0.0},
        {OperationCode::Cnot, 0U, 3U, 0.0, 0.0},
        {OperationCode::Rz, 2U, 0U, -0.19, 0.0},
    }};
}

void tree_tensor_selection() {
    const auto operations = ttn_carrier();
    const std::array<QubitId, 2> selected{{0U, 3U}};
    const std::array<std::uint8_t, 2> bits{{1U, 1U}};

    ExactPreparedProbabilityPlan prepared =
        ExactPreparedProbabilityPlan::for_marginals(4U, operations);
    require(prepared.prepared_route() == ExactExecutionRoute::TreeTensor,
            "nonadjacent non-Clifford marginal did not select TreeTensor");
    require(prepared.estimated_bytes() > sizeof(ExactPreparedProbabilityPlan),
            "TreeTensor prepared memory was not retained in accounting");

    const QRegister direct = evolve(4U, operations);
    const double expected = direct.marginal_probability(selected, bits);
    const auto prepared_result = prepared.marginal_probability(selected, bits);
    require(prepared_result.route == ExactExecutionRoute::TreeTensor,
            "prepared marginal result lost TreeTensor route");
    require(prepared_result.fallback_reason.find("mps:") != std::string::npos,
            "TreeTensor success did not preserve MPS rejection reason");
    require(prepared_result.fallback_reason.find("ttn:") == std::string::npos,
            "TreeTensor success reported a TreeTensor failure");
    require_close(prepared_result.value, expected,
                  "prepared TreeTensor marginal differs from QRegister");

    ExactExecutionBroker broker;
    const auto one_shot = broker.marginal_probability_from_zero(
        4U, operations, selected, bits);
    require(one_shot.route == ExactExecutionRoute::TreeTensor,
            "one-shot marginal did not select TreeTensor");
    require_close(one_shot.value, prepared_result.value,
                  "one-shot and prepared TreeTensor marginals differ");

    bool full_query_rejected = false;
    try {
        const std::array<std::uint8_t, 4> basis{{0U, 0U, 0U, 0U}};
        static_cast<void>(prepared.probability(basis));
    } catch (const QStateError&) {
        full_query_rejected = true;
    }
    require(full_query_rejected,
            "marginal-prepared TreeTensor plan accepted a full-basis query");
}

void resource_fallback() {
    const auto operations = ttn_carrier();
    const std::array<QubitId, 2> selected{{0U, 3U}};
    const std::array<std::uint8_t, 2> bits{{1U, 1U}};

    ExactExecutionBrokerConfig config;
    config.tree_tensor.max_bond_dimension = 1U;
    ExactPreparedProbabilityPlan prepared =
        ExactPreparedProbabilityPlan::for_marginals(4U, operations, config);
    require(prepared.prepared_route() == ExactExecutionRoute::Register,
            "TreeTensor bond rejection did not fall through to QRegister");
    const auto result = prepared.marginal_probability(selected, bits);
    require(result.fallback_reason.find("ttn:") != std::string::npos,
            "TreeTensor rejection reason was not retained");
    require_close(
        result.value,
        evolve(4U, operations).marginal_probability(selected, bits),
        "TreeTensor bond fallback changed the marginal");
}

void unsupported_fallback() {
    const std::array<Operation, 2> operations{{
        {OperationCode::H, 0U, 0U, 0.0, 0.0},
        {OperationCode::BitFlipTrajectory, 0U, 0U, 1.0, 0.0},
    }};
    const std::array<QubitId, 1> selected{{0U}};
    const std::array<std::uint8_t, 1> bits{{0U}};

    ExactExecutionBroker broker;
    const auto result = broker.marginal_probability_from_zero(
        2U, operations, selected, bits);
    require(result.route == ExactExecutionRoute::Register,
            "unsupported TTN operation did not fall through to QRegister");
    require(result.fallback_reason.find("ttn:") != std::string::npos,
            "unsupported TTN fallback did not retain TTN rejection");
    require_close(
        result.value,
        evolve(2U, operations).marginal_probability(selected, bits),
        "unsupported TTN fallback changed the marginal");
}

void full_basis_exclusion() {
    const auto operations = ttn_carrier();
    ExactPreparedProbabilityPlan prepared(4U, operations);
    require(prepared.prepared_route() != ExactExecutionRoute::TreeTensor,
            "full-basis preparation selected marginal-only TreeTensor route");

    const std::array<std::uint8_t, 4> basis{{0U, 0U, 0U, 0U}};
    const auto result = prepared.probability(basis);
    require(result.route != ExactExecutionRoute::TreeTensor,
            "full-basis result reported marginal-only TreeTensor route");
    require_close(
        result.value,
        evolve(4U, operations).amplitude_bits(basis).norm2(),
        "full-basis non-TTN route changed the probability");
}

}  // namespace

int main() {
    route_precedence();
    tree_tensor_selection();
    resource_fallback();
    unsupported_fallback();
    full_basis_exclusion();
    return 0;
}

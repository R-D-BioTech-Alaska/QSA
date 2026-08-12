#include "qubit/qbroker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using qubit::ExactExecutionBroker;
using qubit::ExactExecutionBrokerConfig;
using qubit::ExactExecutionRoute;
using qubit::ExactPreparedProbabilityPlan;
using qubit::Operation;
using qubit::OperationCode;
using qubit::QComplex;
using qubit::QRegister;
using qubit::QStateError;
using qubit::QubitId;
using qubit::StorageMode;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(double actual, double expected, const std::string& message) {
    require(std::abs(actual - expected) <= 2e-11, message);
}

QRegister evolve(std::size_t qubits, std::span<const Operation> operations) {
    QRegister state(qubits);
    qubit::OperationPlan plan(operations);
    plan.execute(state);
    return state;
}

double dense_reference(
    const QRegister& state,
    std::span<const QubitId> qubits,
    std::span<const std::uint8_t> bits) {
    if (qubits.size() != bits.size()) {
        throw std::runtime_error("dense reference received mismatched marginal widths");
    }
    const std::vector<QComplex> amplitudes = state.materialize(12U);
    long double probability = 0.0L;
    for (std::size_t basis = 0U; basis < amplitudes.size(); ++basis) {
        bool match = true;
        for (std::size_t index = 0U; index < qubits.size(); ++index) {
            const std::uint8_t bit = static_cast<std::uint8_t>(
                (basis >> static_cast<std::size_t>(qubits[index])) & 1U);
            if (bit != bits[index]) {
                match = false;
                break;
            }
        }
        if (match) {
            probability += static_cast<long double>(amplitudes[basis].norm2());
        }
    }
    return static_cast<double>(probability);
}

Operation random_operation(std::mt19937_64& generator, std::size_t qubits) {
    Operation operation;
    operation.first = static_cast<QubitId>(generator() % qubits);
    operation.second = static_cast<QubitId>(generator() % qubits);
    if (operation.second == operation.first) {
        operation.second = static_cast<QubitId>((operation.second + 1U) % qubits);
    }
    operation.parameter = (static_cast<double>(generator() % 20001U) / 10000.0) - 1.0;
    switch (generator() % 14U) {
        case 0U: operation.code = OperationCode::X; break;
        case 1U: operation.code = OperationCode::Y; break;
        case 2U: operation.code = OperationCode::Z; break;
        case 3U: operation.code = OperationCode::H; break;
        case 4U: operation.code = OperationCode::S; break;
        case 5U: operation.code = OperationCode::Sdg; break;
        case 6U: operation.code = OperationCode::T; break;
        case 7U: operation.code = OperationCode::Tdg; break;
        case 8U: operation.code = OperationCode::Rx; break;
        case 9U: operation.code = OperationCode::Ry; break;
        case 10U: operation.code = OperationCode::Rz; break;
        case 11U: operation.code = OperationCode::Cnot; break;
        case 12U: operation.code = OperationCode::Cz; break;
        default: operation.code = OperationCode::Swap; break;
    }
    return operation;
}

}  // namespace

int main() {
    {
        QRegister state(4U);
        state.apply_h(0U);
        state.apply_cnot(0U, 1U);
        state.apply_ry(2U, 0.37);
        state.apply_x(3U);

        const std::array<QubitId, 0> no_qubits{};
        const std::array<std::uint8_t, 0> no_bits{};
        require(state.marginal_probability(no_qubits, no_bits) == 1.0,
                "empty QRegister marginal was not one");

        const std::array<QubitId, 3> ordered_qubits{{0U, 2U, 3U}};
        const std::array<std::uint8_t, 3> ordered_bits{{1U, 0U, 1U}};
        const std::array<QubitId, 3> reversed_qubits{{3U, 2U, 0U}};
        const std::array<std::uint8_t, 3> reversed_bits{{1U, 0U, 1U}};
        const double ordered = state.marginal_probability(ordered_qubits, ordered_bits);
        const double reversed = state.marginal_probability(reversed_qubits, reversed_bits);
        require_close(ordered, reversed, "QRegister marginal depends on query order");
        require_close(ordered, dense_reference(state, ordered_qubits, ordered_bits),
                      "QRegister disconnected-component marginal differs from dense reference");

        const std::array<QubitId, 1> q0{{0U}};
        const std::array<std::uint8_t, 1> zero{{0U}};
        const std::array<std::uint8_t, 1> one{{1U}};
        require_close(
            state.marginal_probability(q0, zero) + state.marginal_probability(q0, one),
            1.0,
            "QRegister single-qubit marginals do not sum to one");

        const std::array<QubitId, 1> q2{{2U}};
        const std::array<std::uint8_t, 1> q2_zero{{0U}};
        const std::array<QubitId, 2> q2q3{{2U, 3U}};
        const std::array<std::uint8_t, 2> q2q3_zero{{0U, 0U}};
        const std::array<std::uint8_t, 2> q2q3_one{{0U, 1U}};
        require_close(
            state.marginal_probability(q2, q2_zero),
            state.marginal_probability(q2q3, q2q3_zero) +
                state.marginal_probability(q2q3, q2q3_one),
            "QRegister marginal failed exact refinement consistency");
    }

    {
        QRegister state(3U);
        state.apply_h(0U);
        state.apply_cnot(0U, 1U);
        state.apply_cnot(1U, 2U);
        const std::array<QubitId, 2> pair{{0U, 2U}};
        const std::array<std::uint8_t, 2> hit{{1U, 1U}};
        const std::array<std::uint8_t, 2> miss{{1U, 0U}};
        require_close(state.marginal_probability(pair, hit), 0.5,
                      "sparse entangled QRegister marginal is wrong");
        require(state.marginal_probability(pair, miss) == 0.0,
                "sparse entangled QRegister impossible marginal was nonzero");
    }

    {
        std::vector<QComplex> amplitudes{
            {1.0, 0.0}, {2.0, -0.5}, {3.0, 0.25}, {4.0, 0.75},
            {5.0, -0.25}, {6.0, 0.5}, {7.0, -0.75}, {8.0, 0.125},
        };
        QRegister state = QRegister::from_amplitudes(std::move(amplitudes));
        require(state.component_storage_mode(0U) == StorageMode::Dense,
                "dense marginal gate did not retain dense storage");
        const std::array<QubitId, 2> qubits{{0U, 2U}};
        const std::array<std::uint8_t, 2> bits{{1U, 0U}};
        require_close(state.marginal_probability(qubits, bits),
                      dense_reference(state, qubits, bits),
                      "dense QRegister marginal differs from materialized reference");

        const std::array<QubitId, 3> full_qubits{{0U, 1U, 2U}};
        const std::array<std::uint8_t, 3> full_bits{{1U, 0U, 1U}};
        require_close(state.marginal_probability(full_qubits, full_bits),
                      state.amplitude_bits(full_bits).norm2(),
                      "full QRegister marginal differs from basis amplitude probability");
    }

    {
        std::mt19937_64 generator(0x4D415247494E414CULL);
        constexpr std::size_t qubits = 5U;
        for (std::size_t test_case = 0U; test_case < 32U; ++test_case) {
            std::vector<Operation> operations;
            operations.reserve(36U);
            for (std::size_t gate = 0U; gate < 36U; ++gate) {
                operations.push_back(random_operation(generator, qubits));
            }
            const QRegister state = evolve(qubits, operations);
            std::array<QubitId, qubits> permutation{{0U, 1U, 2U, 3U, 4U}};
            std::shuffle(permutation.begin(), permutation.end(), generator);
            const std::size_t selected = 1U + static_cast<std::size_t>(generator() % qubits);
            std::vector<QubitId> query_qubits(permutation.begin(), permutation.begin() + selected);
            std::vector<std::uint8_t> query_bits(selected);
            for (std::uint8_t& bit : query_bits) {
                bit = static_cast<std::uint8_t>(generator() & 1U);
            }
            require_close(state.marginal_probability(query_qubits, query_bits),
                          dense_reference(state, query_qubits, query_bits),
                          "random QRegister marginal differs from materialized reference");
        }
    }

    {
        QRegister state(2U);
        const std::array<QubitId, 2> duplicate_qubits{{0U, 0U}};
        const std::array<std::uint8_t, 2> duplicate_bits{{0U, 1U}};
        bool rejected = false;
        try {
            static_cast<void>(state.marginal_probability(duplicate_qubits, duplicate_bits));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "QRegister marginal accepted duplicate qubits");

        const std::array<QubitId, 1> one_qubit{{0U}};
        const std::array<std::uint8_t, 2> two_bits{{0U, 1U}};
        rejected = false;
        try {
            static_cast<void>(state.marginal_probability(one_qubit, two_bits));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "QRegister marginal accepted mismatched widths");

        const std::array<QubitId, 1> out_of_range{{2U}};
        const std::array<std::uint8_t, 1> valid_bit{{0U}};
        rejected = false;
        try {
            static_cast<void>(state.marginal_probability(out_of_range, valid_bit));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "QRegister marginal accepted out-of-range qubit");

        const std::array<QubitId, 1> valid_qubit{{1U}};
        const std::array<std::uint8_t, 1> bad_bit{{2U}};
        rejected = false;
        try {
            static_cast<void>(state.marginal_probability(valid_qubit, bad_bit));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "QRegister marginal accepted non-binary bit");
    }

    {
        const std::array<Operation, 5> operations{{
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::Cnot, 0U, 1U, 0.0, 0.0},
            {OperationCode::Ry, 2U, 0U, 0.19, 0.0},
            {OperationCode::Cz, 1U, 2U, 0.0, 0.0},
            {OperationCode::Rz, 0U, 0U, -0.23, 0.0},
        }};
        ExactExecutionBroker broker;
        ExactPreparedProbabilityPlan full(3U, operations);
        ExactPreparedProbabilityPlan marginal =
            ExactPreparedProbabilityPlan::for_marginals(3U, operations);
        require(full.prepared_route() == ExactExecutionRoute::TensorNetwork,
                "full-basis plan no longer preserves TensorNetwork priority");
        require(marginal.prepared_route() == ExactExecutionRoute::PersistentMPS,
                "marginal-aware plan did not skip unsupported TensorNetwork route");

        const std::array<QubitId, 2> query_qubits{{0U, 2U}};
        const std::array<std::uint8_t, 2> query_bits{{1U, 1U}};
        const QRegister direct = evolve(3U, operations);
        const double expected = dense_reference(direct, query_qubits, query_bits);
        const auto one_shot = broker.marginal_probability_from_zero(
            3U, operations, query_qubits, query_bits);
        const auto prepared = marginal.marginal_probability(query_qubits, query_bits);
        require(one_shot.route == ExactExecutionRoute::PersistentMPS &&
                    prepared.route == ExactExecutionRoute::PersistentMPS,
                "query-aware broker did not preserve marginal-capable MPS route");
        require_close(one_shot.value, expected,
                      "query-aware MPS one-shot marginal differs from dense reference");
        require_close(prepared.value, expected,
                      "query-aware MPS prepared marginal differs from dense reference");
        require(prepared.fallback_reason.find(
                    "tensor: route does not support marginal probability") != std::string::npos,
                "query-aware MPS route did not record TensorNetwork capability rejection");

        bool rejected = false;
        try {
            static_cast<void>(full.marginal_probability(query_qubits, query_bits));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected,
                "ordinary full-basis TensorNetwork plan silently changed marginal semantics");
    }

    {
        const std::array<Operation, 3> operations{{
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::Cnot, 0U, 2U, 0.0, 0.0},
            {OperationCode::Ry, 0U, 0U, 0.19, 0.0},
        }};
        ExactExecutionBroker broker;
        ExactPreparedProbabilityPlan full(3U, operations);
        ExactPreparedProbabilityPlan marginal =
            ExactPreparedProbabilityPlan::for_marginals(3U, operations);
        require(full.prepared_route() == ExactExecutionRoute::TensorNetwork,
                "full-basis nonadjacent circuit did not preserve TensorNetwork priority");
        require(marginal.prepared_route() == ExactExecutionRoute::Register,
                "marginal-aware nonadjacent circuit did not reach exact QRegister fallback");

        const std::array<QubitId, 2> query_qubits{{0U, 2U}};
        const std::array<std::uint8_t, 2> query_bits{{1U, 1U}};
        const QRegister direct = evolve(3U, operations);
        const double expected = dense_reference(direct, query_qubits, query_bits);
        const auto one_shot = broker.marginal_probability_from_zero(
            3U, operations, query_qubits, query_bits);
        const auto prepared = marginal.marginal_probability(query_qubits, query_bits);
        require(one_shot.route == ExactExecutionRoute::Register &&
                    prepared.route == ExactExecutionRoute::Register,
                "generic marginal fallback did not report QRegister route");
        require_close(one_shot.value, expected,
                      "one-shot QRegister marginal fallback differs from dense reference");
        require_close(prepared.value, expected,
                      "prepared QRegister marginal fallback differs from dense reference");
        require(prepared.fallback_reason.find(
                    "tensor: route does not support marginal probability") != std::string::npos &&
                    prepared.fallback_reason.find("mps:") != std::string::npos &&
                    prepared.fallback_reason.find(
                        "phase_graph: route does not support marginal probability") != std::string::npos,
                "QRegister marginal fallback did not retain capability/rejection reasons");
    }

    {
        const std::array<Operation, 2> operations{{
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::AmplitudeDampingTrajectory, 0U, 0U, 0.25, 0.1},
        }};
        ExactExecutionBroker broker;
        const std::array<QubitId, 1> query_qubits{{0U}};
        const std::array<std::uint8_t, 1> query_bits{{0U}};
        const QRegister direct = evolve(2U, operations);
        const auto result = broker.marginal_probability_from_zero(
            2U, operations, query_qubits, query_bits);
        require(result.route == ExactExecutionRoute::Register,
                "trajectory marginal did not reach exact QRegister fallback");
        require_close(result.value, dense_reference(direct, query_qubits, query_bits),
                      "trajectory QRegister marginal differs from dense reference");
    }

    return 0;
}

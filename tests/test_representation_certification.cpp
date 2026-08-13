#include "qubit/qrepresentation_compiler.hpp"
#include "qubit/qrouter.hpp"
#include "qubit/qsemantic_tripair.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using qubit::Operation;
using qubit::OperationCode;
using qubit::QRegister;
using qubit::QubitId;
using qubit::RepresentationAdvisor;
using qubit::RepresentationKind;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(double left, double right, const char* message) {
    require(std::abs(left - right) <= 2e-11, message);
}

void append(
    qubit::ExactRepresentationFabric& fabric,
    std::vector<Operation>& operations,
    const Operation& operation) {
    fabric.append(operation);
    operations.push_back(operation);
}

void exact_representation_fabric() {
    qubit::ExactRepresentationFabric fabric(12U);
    std::vector<Operation> operations;
    append(fabric, operations, {OperationCode::H, 0U});
    append(fabric, operations, {OperationCode::Cnot, 0U, 1U});
    append(fabric, operations, {OperationCode::Ry, 2U, 0U, 0.37});
    append(fabric, operations, {OperationCode::Cnot, 2U, 3U});
    append(fabric, operations, {OperationCode::H, 4U});

    const std::vector<QubitId> query{0U, 2U, 4U};
    const std::vector<std::uint8_t> zero(3U, 0U);
    const auto first = fabric.marginal_probability(query, zero);
    const auto control = qubit::ExactPreparedProbabilityPlan::for_marginals(12U, operations);
    require_close(
        first.value,
        control.marginal_probability(query, zero).value,
        "representation fabric differs from exact global control");
    require(fabric.stats().active_components == 3U,
            "representation fabric did not preserve independent islands");
    require(first.receipt.fabric_components == 3U,
            "representation fabric queried excess islands");

    const auto warm = fabric.marginal_probability(query, zero);
    require(warm.receipt.cache_hits == 3U && warm.receipt.cache_misses == 0U,
            "representation fabric did not reuse unchanged island programs");

    append(fabric, operations, {OperationCode::Cnot, 1U, 2U});
    require(fabric.stats().active_components == 2U &&
            fabric.stats().component_merges == 1U,
            "representation fabric did not merge only the crossed closure");
    const auto merged = fabric.marginal_probability(query, zero);
    const auto merged_control = qubit::ExactPreparedProbabilityPlan::for_marginals(12U, operations);
    require_close(
        merged.value,
        merged_control.marginal_probability(query, zero).value,
        "representation fabric cross-island merge changed the exact result");
    require(merged.receipt.cache_hits >= 1U && merged.receipt.cache_misses >= 1U,
            "representation fabric failed to preserve the untouched island cache");

    const std::array<QubitId, 3> declared{0U, 4U, 5U};
    fabric.declare_dependency(declared);
    const std::vector<QubitId> declared_query{0U, 2U, 4U, 5U};
    const std::vector<std::uint8_t> declared_zero(4U, 0U);
    const auto declared_result = fabric.marginal_probability(declared_query, declared_zero);
    require_close(
        declared_result.value,
        merged_control.marginal_probability(declared_query, declared_zero).value,
        "structural hyperedge fabricated numerical coupling");
    require(fabric.stats().active_components == 1U &&
            fabric.stats().declared_dependencies == 1U,
            "declared hyperedge did not form one conservative dependency island");

    qubit::ExactRepresentationFabricConfig bounded_config;
    bounded_config.max_component_qubits = 2U;
    qubit::ExactRepresentationFabric bounded(4U, bounded_config);
    bounded.append({OperationCode::H, 0U});
    bounded.append({OperationCode::Cnot, 0U, 1U});
    bool rejected = false;
    try {
        bounded.append({OperationCode::Cnot, 1U, 2U});
    } catch (const qubit::QStateError&) {
        rejected = true;
    }
    require(rejected && bounded.stats().operations == 2U,
            "representation fabric did not fail closed on component growth");

    fabric.reset();
    const std::array<QubitId, 1> untouched{0U};
    const std::array<std::uint8_t, 1> one{1U};
    require(fabric.marginal_probability(untouched, one).value == 0.0,
            "representation fabric reset did not restore exact zero state");
}

void semantic_tripair_fabric_contract() {
    qubit::ExactRepresentationFabric fabric(6U);
    std::vector<Operation> operations;
    const qubit::SemanticTripairInput first{{0.41, 0.73, 0.29}, {0.17, -0.31, 0.43}};
    const qubit::SemanticTripairInput second{{0.52, 0.21, 0.66}, {-0.27, 0.38, 0.11}};
    const qubit::SemanticTripairProgram semantic;
    const auto first_state = semantic.evaluate(first);
    const auto second_state = semantic.evaluate(second);

    const auto add_tripair = [&](QubitId base, const qubit::SemanticTripairInput& input) {
        for (std::size_t local = 0U; local < 3U; ++local) {
            append(fabric, operations, {
                OperationCode::Ry,
                static_cast<QubitId>(base + local),
                0U,
                input.theta[local],
            });
            append(fabric, operations, {
                OperationCode::Rz,
                static_cast<QubitId>(base + local),
                0U,
                input.phase[local],
            });
        }
        append(fabric, operations, {OperationCode::Cnot, base, static_cast<QubitId>(base + 1U)});
        append(fabric, operations, {
            OperationCode::Cnot,
            static_cast<QubitId>(base + 1U),
            static_cast<QubitId>(base + 2U),
        });
        append(fabric, operations, {OperationCode::Cnot, static_cast<QubitId>(base + 2U), base});
    };

    add_tripair(0U, first);
    add_tripair(3U, second);
    const std::vector<QubitId> all{0U, 1U, 2U, 3U, 4U, 5U};
    const std::vector<std::uint8_t> zero(6U, 0U);
    const auto independent = fabric.marginal_probability(all, zero);
    require_close(
        independent.value,
        first_state.amplitudes[0].norm2() * second_state.amplitudes[0].norm2(),
        "representation fabric did not preserve independent Semantic Tripair states");
    require(independent.receipt.fabric_components == 2U,
            "semantic consumer contract did not remain component local");

    append(fabric, operations, {OperationCode::Cnot, 2U, 3U});
    const auto coupled = fabric.marginal_probability(all, zero);
    const auto global = qubit::ExactPreparedProbabilityPlan::for_marginals(6U, operations);
    require_close(
        coupled.value,
        global.marginal_probability(all, zero).value,
        "semantic consumer cross-island operation changed exact state semantics");
    require(fabric.stats().active_components == 1U,
            "semantic consumer cross-island operation did not merge its exact closure");
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
        const auto features = RepresentationAdvisor::inspect_operations(state, operations, 100'000U);
        require(features.clifford_only, "Clifford operation list was not certified");
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
        const auto features = RepresentationAdvisor::inspect_operations(magic, operations, 100'000U);
        require(features.clifford_only, "future Clifford circuit was not recognized");
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
        const auto features = RepresentationAdvisor::inspect_operations(state, operations, 100'000U);
        require(!features.clifford_only, "T gate was incorrectly certified as Clifford");
        require(!features.stabilizer_input_certified,
                "non-Clifford workload retained stabilizer eligibility");
        require(advisor.recommend(features).kind == RepresentationKind::Register,
                "non-Clifford workload escaped the exact general fallback");
    }

    {
        const std::array<Operation, 1> operations{{
            {OperationCode::AmplitudeDampingTrajectory, 0U, 0U, 0.2, 0.5},
        }};
        const auto features = RepresentationAdvisor::inspect_operations(state, operations, 100'000U);
        require(!features.clifford_only,
                "trajectory noise was incorrectly certified as Clifford");
        require(!features.stabilizer_input_certified,
                "trajectory workload retained stabilizer eligibility");
        require(advisor.recommend(features).kind == RepresentationKind::Register,
                "trajectory workload escaped the exact general fallback");
    }

    {
        const std::array<Operation, 0> operations{};
        const auto features = RepresentationAdvisor::inspect_operations(state, operations, 1U);
        require(features.clifford_only,
                "empty operation list should be Clifford compatible");
        require(features.stabilizer_input_certified,
                "computational-basis product was not certified for an empty circuit");
    }

    exact_representation_fabric();
    semantic_tripair_fabric_contract();
    std::cout << "representation certification tests passed\n";
    return 0;
}

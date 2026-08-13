#include "qubit/qmagic.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(double observed, double expected, double tolerance, const char* message) {
    if (std::abs(observed - expected) > tolerance * (1.0 + std::abs(expected))) {
        throw std::runtime_error(
            std::string(message) + ": observed=" + std::to_string(observed) +
            " expected=" + std::to_string(expected));
    }
}

std::vector<qubit::Operation> circuit(std::size_t qubits) {
    using qubit::Operation;
    using qubit::OperationCode;
    std::vector<Operation> operations{
        {OperationCode::H, 0U},
        {OperationCode::H, 1U},
        {OperationCode::T, 0U},
        {OperationCode::Cnot, 0U, 1U},
        {OperationCode::S, 1U},
        {OperationCode::Tdg, 1U},
        {OperationCode::H, 0U},
        {OperationCode::Y, 1U},
        {OperationCode::T, 1U},
        {OperationCode::Cz, 0U, 1U},
        {OperationCode::Sdg, 0U},
    };
    if (qubits > 2U) {
        operations.push_back({OperationCode::H, 2U});
        operations.push_back({OperationCode::Cnot, 1U, 2U});
        operations.push_back({OperationCode::T, 2U});
        operations.push_back({OperationCode::Swap, 0U, 2U});
        operations.push_back({OperationCode::Z, 1U});
    }
    if (qubits > 3U) {
        operations.push_back({OperationCode::H, 3U});
        operations.push_back({OperationCode::Cz, 2U, 3U});
        operations.push_back({OperationCode::X, 3U});
    }
    return operations;
}

void dense_equivalence() {
    using qubit::ExactLowMagicConfig;
    using qubit::ExactLowMagicStabilizerSum;
    using qubit::OperationPlan;
    using qubit::PauliSupportPlan;
    using qubit::PauliSupportTerm;
    using qubit::QRegister;

    for (std::size_t qubits = 2U; qubits <= 6U; ++qubits) {
        const auto operations = circuit(qubits);
        QRegister dense(qubits);
        OperationPlan dense_plan(std::span<const qubit::Operation>(operations), false);
        dense_plan.execute(dense);

        ExactLowMagicConfig config;
        config.max_live_branches = 256U;
        config.max_intermediate_branches = 512U;
        config.max_expectation_word_ops = 20'000'000U;
        ExactLowMagicStabilizerSum compact(qubits, config);
        compact.apply_many(std::span<const qubit::Operation>(operations));

        require_close(compact.norm_squared(), 1.0, 2e-11, "low-magic norm mismatch");
        for (std::size_t qubit_index = 0U; qubit_index < qubits; ++qubit_index) {
            const auto qubit_id = static_cast<qubit::QubitId>(qubit_index);
            require_close(
                compact.probability_one(qubit_id),
                dense.probability_one(qubit_id),
                3e-11,
                "low-magic marginal mismatch");
        }

        std::vector<std::vector<PauliSupportTerm>> observables{
            {{0U, 'X'}},
            {{0U, 'Z'}},
            {{1U, 'Y'}},
            {{0U, 'Z'}, {1U, 'Z'}},
            {{0U, 'X'}, {1U, 'Y'}},
        };
        if (qubits > 2U) {
            observables.push_back({{0U, 'X'}, {2U, 'Z'}});
        }
        PauliSupportPlan dense_observables(
            qubits, std::span<const std::vector<PauliSupportTerm>>(observables));
        const auto dense_values = dense_observables.execute(dense);
        for (std::size_t index = 0U; index < observables.size(); ++index) {
            require_close(
                compact.expectation_pauli(
                    std::span<const PauliSupportTerm>(observables[index])),
                dense_values[index],
                4e-11,
                "low-magic Pauli expectation mismatch");
        }
    }
}

void exact_reconvergence() {
    using qubit::ExactLowMagicStabilizerSum;
    using qubit::OperationCode;

    ExactLowMagicStabilizerSum state(2U);
    state.apply({OperationCode::H, 0U});
    state.apply({OperationCode::T, 0U});
    require(state.branch_count() == 2U, "single T did not create two exact branches");
    state.apply({OperationCode::Tdg, 0U});
    require(state.branch_count() == 1U, "T/Tdg did not reconverge exactly");
    require(state.stats().t_defects == 2U, "T/Tdg defect count mismatch");
    require(
        state.stats().extracted_power_of_two_bits >= 2U,
        "T/Tdg common denominator was not cancelled exactly");
    require(state.stats().scale_power_two == 0, "T/Tdg did not restore common scale");
    require_close(state.norm_squared(), 1.0, 1e-12, "T/Tdg reconverged norm mismatch");
}

void fail_closed_gates() {
    using qubit::ExactLowMagicConfig;
    using qubit::ExactLowMagicStabilizerSum;
    using qubit::OperationCode;
    using qubit::PauliSupportTerm;
    using qubit::QStateError;

    ExactLowMagicConfig config;
    config.max_live_branches = 4U;
    config.max_intermediate_branches = 8U;
    ExactLowMagicStabilizerSum state(3U, config);
    state.apply({OperationCode::H, 0U});
    state.apply({OperationCode::H, 1U});
    state.apply({OperationCode::H, 2U});
    state.apply({OperationCode::T, 0U});
    state.apply({OperationCode::T, 1U});
    require(state.branch_count() == 4U, "branch-cap setup did not reach four branches");
    const auto before = state.stats();
    bool rejected = false;
    try {
        state.apply({OperationCode::T, 2U});
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "low-magic branch cap did not reject");
    require(state.branch_count() == before.live_branches, "branch-cap rejection mutated branch state");
    require(state.stats().t_defects == before.t_defects, "branch-cap rejection mutated counters");

    rejected = false;
    try {
        state.apply({OperationCode::Rz, 0U, 0U, 0.125});
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "floating Rz did not reject from low-magic exact route");

    const std::vector<PauliSupportTerm> duplicate{{0U, 'X'}, {0U, 'Z'}};
    rejected = false;
    try {
        (void)state.expectation_pauli(std::span<const PauliSupportTerm>(duplicate));
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "duplicate Pauli support did not reject");

    ExactLowMagicConfig work_config;
    work_config.max_expectation_word_ops = 1U;
    ExactLowMagicStabilizerSum bounded(4U, work_config);
    rejected = false;
    try {
        (void)bounded.norm_squared();
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "expectation work cap did not reject");
}

}  // namespace

int main() {
    try {
        dense_equivalence();
        exact_reconvergence();
        fail_closed_gates();
        std::cout << "exact low-magic stabilizer sum tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

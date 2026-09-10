#include "qubit/qtensor.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using qubit::Operation;
using qubit::OperationCode;
using qubit::QComplex;
using qubit::QStateError;
using qubit::QubitId;
using qubit::TensorContractionPlan;
using qubit::TensorContractionStats;
using qubit::TensorContractionWorkspace;
using qubit::TensorNetworkCircuit;
using qubit::TensorNetworkConfig;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

[[nodiscard]] bool same_certificate(
    const TensorContractionStats& first,
    const TensorContractionStats& second) noexcept {
    return first.source_operations == second.source_operations &&
           first.source_factors == second.source_factors &&
           first.eliminated_variables == second.eliminated_variables &&
           first.peak_union_variables == second.peak_union_variables &&
           first.peak_contraction_entries == second.peak_contraction_entries;
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
        TensorNetworkCircuit tensor(2U);
        tensor.apply({OperationCode::H, 0U, 0U, 0.0, 0.0});
        tensor.apply({OperationCode::Cnot, 0U, 1U, 0.0, 0.0});
        const TensorContractionPlan plan = tensor.compile();
        TensorContractionWorkspace workspace = plan.workspace();

        for (std::uint64_t basis = 0U; basis < 4U; ++basis) {
            TensorContractionStats direct_stats;
            TensorContractionStats compiled_stats;
            const QComplex direct = tensor.amplitude(basis, &direct_stats);
            const QComplex compiled = plan.amplitude(basis, workspace, &compiled_stats);
            require(qubit::almost_equal(direct, compiled, 1e-13),
                    "compiled Bell contraction disagrees with direct contraction");
            require(same_certificate(direct_stats, compiled_stats),
                    "compiled Bell contraction changed its resource certificate");
        }
        require(plan.step_count() == plan.stats().eliminated_variables,
                "compiled Bell step count is inconsistent");
        require(plan.estimated_bytes() > 0U && workspace.estimated_bytes() > 0U,
                "compiled Bell plan reports no storage");
    }

    {
        std::mt19937_64 generator(0x434F4E5452414354ULL);
        for (std::size_t test_case = 0; test_case < 32U; ++test_case) {
            constexpr std::size_t qubits = 6U;
            TensorNetworkCircuit tensor(qubits);
            for (std::size_t gate = 0; gate < 42U; ++gate) {
                tensor.apply(random_operation(generator, qubits));
            }
            const TensorContractionPlan plan = tensor.compile();
            const TensorContractionPlan repeated_plan = tensor.compile();
            require(same_certificate(plan.stats(), repeated_plan.stats()) &&
                        plan.step_count() == repeated_plan.step_count(),
                    "incremental contraction planning is not deterministic");
            TensorContractionWorkspace workspace = plan.workspace();
            for (std::size_t query = 0; query < 16U; ++query) {
                const std::uint64_t basis = generator() & 63U;
                TensorContractionStats direct_stats;
                TensorContractionStats compiled_stats;
                const QComplex direct = tensor.amplitude(basis, &direct_stats);
                const QComplex compiled = plan.amplitude(basis, workspace, &compiled_stats);
                require(qubit::almost_equal(direct, compiled, 2e-12),
                        "compiled random contraction disagrees with direct contraction");
                require(same_certificate(direct_stats, compiled_stats),
                        "incremental planner changed the greedy resource certificate");
            }
        }
    }

    {
        constexpr std::size_t qubits = 100U;
        TensorNetworkConfig config;
        config.max_contraction_entries = 1U << 14U;
        TensorNetworkCircuit tensor(qubits, config);
        long double expected_real = 1.0L;
        for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
            const double angle = 0.002 * static_cast<double>(1U + qubit % 17U);
            tensor.apply({OperationCode::Ry, static_cast<QubitId>(qubit), 0U, angle, 0.0});
            expected_real *= std::cos(angle * 0.5);
        }
        for (std::size_t layer = 0; layer < 4U; ++layer) {
            const std::size_t offset = layer & 1U;
            for (std::size_t qubit = offset; qubit + 1U < qubits; qubit += 2U) {
                tensor.apply({
                    OperationCode::Cnot,
                    static_cast<QubitId>(qubit),
                    static_cast<QubitId>(qubit + 1U),
                    0.0,
                    0.0,
                });
            }
        }

        std::vector<std::uint8_t> zero_bits(qubits, 0U);
        TensorContractionStats direct_stats;
        const QComplex direct = tensor.amplitude(zero_bits, &direct_stats);
        const TensorContractionPlan plan = tensor.compile();
        TensorContractionWorkspace workspace = plan.workspace();
        TensorContractionStats compiled_stats;
        const QComplex compiled = plan.amplitude(zero_bits, workspace, &compiled_stats);
        require(std::abs(compiled.re - static_cast<double>(expected_real)) <= 2e-11,
                "compiled 100-qubit amplitude disagrees with analytic result");
        require(std::abs(compiled.im) <= 2e-11,
                "compiled 100-qubit zero amplitude acquired an imaginary part");
        require(qubit::almost_equal(direct, compiled, 1e-13),
                "compiled 100-qubit contraction disagrees with direct contraction");
        require(same_certificate(direct_stats, compiled_stats),
                "compiled 100-qubit contraction changed its resource certificate");
    }

    {
        TensorNetworkConfig config;
        config.max_contraction_entries = 16U;
        TensorNetworkCircuit tensor(2U, config);
        tensor.apply({OperationCode::H, 0U, 0U, 0.0, 0.0});
        tensor.apply({OperationCode::Cnot, 0U, 1U, 0.0, 0.0});
        const TensorContractionPlan plan = tensor.compile();
        require(plan.stats().peak_contraction_entries == 16U,
                "compiled contraction changed its exact resource boundary");
    }

    {
        TensorNetworkConfig config;
        config.max_contraction_entries = 8U;
        TensorNetworkCircuit tensor(2U, config);
        tensor.apply({OperationCode::H, 0U, 0U, 0.0, 0.0});
        tensor.apply({OperationCode::Cnot, 0U, 1U, 0.0, 0.0});
        bool rejected = false;
        try {
            static_cast<void>(tensor.compile());
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "compiled contraction ignored max_contraction_entries");
        require(tensor.validate(), "failed plan compilation damaged the source circuit");
    }

    {
        TensorNetworkCircuit tensor(1U);
        tensor.apply({OperationCode::H, 0U, 0U, 0.0, 0.0});
        const TensorContractionPlan plan = tensor.compile();
        TensorContractionWorkspace workspace = plan.workspace();
        const QComplex frozen = plan.amplitude(1U, workspace);
        tensor.apply({OperationCode::Z, 0U, 0U, 0.0, 0.0});
        require(!qubit::almost_equal(frozen, tensor.amplitude(1U), 1e-13),
                "compiled plan changed after the source circuit mutated");
        require(qubit::almost_equal(frozen, QComplex{std::sqrt(0.5), 0.0}, 1e-13),
                "compiled plan did not preserve its source snapshot");
    }

    {
        TensorNetworkCircuit first(2U);
        first.apply({OperationCode::H, 0U, 0U, 0.0, 0.0});
        TensorNetworkCircuit second(3U);
        second.apply({OperationCode::H, 0U, 0U, 0.0, 0.0});
        const TensorContractionPlan first_plan = first.compile();
        const TensorContractionPlan second_plan = second.compile();
        TensorContractionWorkspace wrong_workspace = second_plan.workspace();
        bool rejected = false;
        try {
            static_cast<void>(first_plan.amplitude(0U, wrong_workspace));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "compiled contraction accepted a workspace from another plan");
    }

    std::cout << "tensor contraction plan tests passed\n";
    return 0;
}

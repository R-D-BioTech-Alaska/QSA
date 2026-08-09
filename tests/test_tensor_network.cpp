#include "qubit/qtensor.hpp"

#include <array>
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
using qubit::QRegister;
using qubit::QStateError;
using qubit::QubitId;
using qubit::TensorContractionStats;
using qubit::TensorNetworkCircuit;
using qubit::TensorNetworkConfig;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void apply_reference(QRegister& state, const Operation& operation) {
    switch (operation.code) {
        case OperationCode::X:
            state.apply_x(operation.first);
            break;
        case OperationCode::Y:
            state.apply_y(operation.first);
            break;
        case OperationCode::Z:
            state.apply_z(operation.first);
            break;
        case OperationCode::H:
            state.apply_h(operation.first);
            break;
        case OperationCode::S:
            state.apply_s(operation.first);
            break;
        case OperationCode::Sdg:
            state.apply_sdg(operation.first);
            break;
        case OperationCode::T:
            state.apply_t(operation.first);
            break;
        case OperationCode::Tdg:
            state.apply_tdg(operation.first);
            break;
        case OperationCode::Rx:
            state.apply_rx(operation.first, operation.parameter);
            break;
        case OperationCode::Ry:
            state.apply_ry(operation.first, operation.parameter);
            break;
        case OperationCode::Rz:
            state.apply_rz(operation.first, operation.parameter);
            break;
        case OperationCode::Cnot:
            state.apply_cnot(operation.first, operation.second);
            break;
        case OperationCode::Cz:
            state.apply_cz(operation.first, operation.second);
            break;
        case OperationCode::Swap:
            state.apply_swap(operation.first, operation.second);
            break;
        case OperationCode::BitFlipTrajectory:
        case OperationCode::PhaseFlipTrajectory:
        case OperationCode::DepolarizingTrajectory:
        case OperationCode::AmplitudeDampingTrajectory:
            throw std::runtime_error("noise is not part of the tensor differential reference");
    }
}

void compare(
    const TensorNetworkCircuit& tensor,
    const QRegister& reference,
    double tolerance,
    const std::string& label) {
    const std::vector<QComplex> actual = tensor.materialize(10U);
    const std::vector<QComplex> expected = reference.materialize(10U);
    require(actual.size() == expected.size(), label + ": dimensions differ");

    QComplex phase{1.0, 0.0};
    bool found = false;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual[index].magnitude() > tolerance && expected[index].magnitude() > tolerance) {
            phase = actual[index] / expected[index];
            phase /= phase.magnitude();
            found = true;
            break;
        }
    }
    require(found, label + ": no nonzero amplitude found");

    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!qubit::almost_equal(actual[index], phase * expected[index], tolerance)) {
            throw std::runtime_error(label + ": amplitude mismatch at " + std::to_string(index));
        }
    }
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
        case 0U:
            operation.code = OperationCode::X;
            break;
        case 1U:
            operation.code = OperationCode::Y;
            break;
        case 2U:
            operation.code = OperationCode::Z;
            break;
        case 3U:
            operation.code = OperationCode::H;
            break;
        case 4U:
            operation.code = OperationCode::S;
            break;
        case 5U:
            operation.code = OperationCode::Sdg;
            break;
        case 6U:
            operation.code = OperationCode::T;
            break;
        case 7U:
            operation.code = OperationCode::Tdg;
            break;
        case 8U:
            operation.code = OperationCode::Rx;
            break;
        case 9U:
            operation.code = OperationCode::Ry;
            break;
        case 10U:
            operation.code = OperationCode::Rz;
            break;
        case 11U:
            operation.code = OperationCode::Cnot;
            break;
        case 12U:
            operation.code = OperationCode::Cz;
            break;
        default:
            operation.code = OperationCode::Swap;
            break;
    }
    return operation;
}

}  // namespace

int main() {
    {
        const std::array<Operation, 2> bell{{
            {OperationCode::H, 0U, 0U, 0.0, 0.0},
            {OperationCode::Cnot, 0U, 1U, 0.0, 0.0},
        }};
        TensorNetworkCircuit tensor(2U, bell);
        QRegister reference(2U);
        for (const Operation& operation : bell) {
            apply_reference(reference, operation);
        }
        compare(tensor, reference, 1e-12, "Bell circuit");
        require(tensor.validate(), "Bell tensor network failed validation");
    }

    {
        std::mt19937_64 generator(0x54454E534F524E45ULL);
        for (std::size_t test_case = 0; test_case < 48U; ++test_case) {
            constexpr std::size_t qubits = 5U;
            TensorNetworkCircuit tensor(qubits);
            QRegister reference(qubits);
            for (std::size_t gate = 0; gate < 36U; ++gate) {
                const Operation operation = random_operation(generator, qubits);
                tensor.apply(operation);
                apply_reference(reference, operation);
            }
            compare(tensor, reference, 2e-10, "random tensor differential");
            require(tensor.validate(), "random tensor network failed validation");
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
        TensorContractionStats first_stats;
        TensorContractionStats second_stats;
        const QComplex first = tensor.amplitude(zero_bits, &first_stats);
        const QComplex second = tensor.amplitude(zero_bits, &second_stats);
        require(std::abs(first.re - static_cast<double>(expected_real)) <= 2e-11,
                "100-qubit tensor amplitude disagrees with analytic zero amplitude");
        require(std::abs(first.im) <= 2e-11,
                "100-qubit tensor zero amplitude acquired an imaginary part");
        require(qubit::almost_equal(first, second, 1e-13),
                "repeated tensor contraction is not deterministic");
        require(first_stats.peak_contraction_entries == second_stats.peak_contraction_entries &&
                    first_stats.peak_union_variables == second_stats.peak_union_variables &&
                    first_stats.eliminated_variables == second_stats.eliminated_variables,
                "repeated tensor contraction stats are not deterministic");
        require(first_stats.peak_contraction_entries <= config.max_contraction_entries,
                "structured tensor query exceeded its contraction certificate");
        require(tensor.validate(), "100-qubit tensor network failed validation");
    }

    {
        TensorNetworkConfig config;
        config.max_contraction_entries = 8U;
        TensorNetworkCircuit tensor(2U, config);
        tensor.apply({OperationCode::H, 0U, 0U, 0.0, 0.0});
        tensor.apply({OperationCode::Cnot, 0U, 1U, 0.0, 0.0});
        bool rejected = false;
        try {
            static_cast<void>(tensor.amplitude(0U));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "tensor contraction limit did not fail closed");
        require(tensor.validate(), "failed contraction damaged the tensor network");
    }

    {
        TensorNetworkCircuit tensor(3U);
        bool rejected = false;
        try {
            tensor.apply({OperationCode::AmplitudeDampingTrajectory, 0U, 0U, 0.2, 0.5});
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "tensor network accepted trajectory noise");
        require(tensor.operation_count() == 0U,
                "rejected trajectory noise changed operation count");
        require(tensor.validate(), "noise rejection damaged the tensor network");
    }

    {
        TensorNetworkCircuit tensor(21U);
        bool rejected = false;
        try {
            static_cast<void>(tensor.materialize(20U));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "tensor network ignored materialization width limit");
    }

    std::cout << "tensor network tests passed\n";
    return 0;
}

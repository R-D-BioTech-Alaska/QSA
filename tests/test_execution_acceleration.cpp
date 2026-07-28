#include "qubit/qexecution.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using qubit::DependencyOperationPlan;
using qubit::IndependentComponentPlan;
using qubit::Operation;
using qubit::OperationCode;
using qubit::QComplex;
using qubit::QRegister;
using qubit::QStateError;
using qubit::QubitId;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

[[nodiscard]] double fidelity(
    const std::vector<QComplex>& first,
    const std::vector<QComplex>& second) {
    require(first.size() == second.size(), "execution state dimensions differ");
    QComplex overlap{};
    for (std::size_t index = 0; index < first.size(); ++index) {
        overlap += second[index].conjugate() * first[index];
    }
    return overlap.norm2();
}

void execute_direct(QRegister& state, const Operation& operation) {
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
            state.apply_cnot_structured(operation.first, operation.second);
            break;
        case OperationCode::Cz:
            state.apply_cz_structured(operation.first, operation.second);
            break;
        case OperationCode::Swap:
            state.apply_swap_structured(operation.first, operation.second);
            break;
        default:
            throw std::runtime_error("test received unsupported operation");
    }
}

void execute_direct(QRegister& state, const std::vector<Operation>& operations) {
    for (const Operation& operation : operations) {
        execute_direct(state, operation);
    }
}

[[nodiscard]] QRegister make_components(
    std::size_t component_count,
    std::size_t width) {
    QRegister state(component_count * width);
    for (std::size_t component = 0; component < component_count; ++component) {
        const std::size_t offset = component * width;
        for (std::size_t local = 0; local < width; ++local) {
            const QubitId qubit = static_cast<QubitId>(offset + local);
            state.apply_ry(qubit, 0.17 + 0.013 * static_cast<double>(local));
            state.apply_rz(qubit, -0.11 + 0.009 * static_cast<double>(component));
        }
        for (std::size_t local = 1; local < width; ++local) {
            state.apply_cz(
                static_cast<QubitId>(offset + local - 1U),
                static_cast<QubitId>(offset + local));
        }
    }
    return state;
}

void compare_factorized_states(const QRegister& first, const QRegister& second) {
    require(first.qubit_count() == second.qubit_count(), "qubit counts differ");
    const auto first_probabilities = first.probabilities_one();
    const auto second_probabilities = second.probabilities_one();
    for (std::size_t qubit = 0; qubit < first.qubit_count(); ++qubit) {
        require(std::abs(first_probabilities[qubit] - second_probabilities[qubit]) < 2e-11,
                "parallel component probability differs");
    }
    std::mt19937_64 generator(0x455845435554494FULL);
    std::vector<std::uint8_t> bits(first.qubit_count());
    for (std::size_t sample = 0; sample < 64U; ++sample) {
        for (std::uint8_t& bit : bits) {
            bit = static_cast<std::uint8_t>(generator() & 1U);
        }
        require(qubit::almost_equal(
                    first.amplitude_bits(bits), second.amplitude_bits(bits), 3e-11),
                "parallel component amplitude differs");
    }
}

}  // namespace

int main() {
    {
        const std::vector<Operation> operations{
            Operation{OperationCode::H, 0},
            Operation{OperationCode::X, 1},
            Operation{OperationCode::H, 0},
            Operation{OperationCode::X, 1},
        };
        DependencyOperationPlan plan(operations);
        require(plan.source_operation_count() == 4U, "dependency source count is wrong");
        require(plan.optimized_operation_count() == 0U,
                "dependency plan did not cancel commuting involutions");
        QRegister state(2);
        plan.execute(state);
        require(state.probability_one(0) == 0.0 && state.probability_one(1) == 0.0,
                "empty dependency plan changed the state");
    }

    {
        const std::vector<Operation> operations{
            Operation{OperationCode::Rx, 0, 0, 0.31},
            Operation{OperationCode::H, 1},
            Operation{OperationCode::Rx, 0, 0, -0.11},
            Operation{OperationCode::H, 1},
            Operation{OperationCode::Rx, 0, 0, -0.20},
            Operation{OperationCode::Cnot, 2, 3},
            Operation{OperationCode::Z, 4},
            Operation{OperationCode::Cnot, 2, 3},
            Operation{OperationCode::Z, 4},
        };
        DependencyOperationPlan plan(operations);
        require(plan.optimized_operation_count() == 0U,
                "dependency plan did not cancel rotations and two-qubit gates");
    }

    {
        std::mt19937_64 generator(0x444550454E44454EULL);
        std::vector<Operation> operations;
        for (std::size_t block = 0; block < 300U; ++block) {
            const QubitId qubit = static_cast<QubitId>(generator() % 8U);
            const QubitId disjoint = static_cast<QubitId>((qubit + 3U) % 8U);
            const OperationCode code = (generator() & 1U) == 0U
                ? OperationCode::H
                : OperationCode::X;
            operations.push_back(Operation{code, qubit});
            operations.push_back(Operation{OperationCode::Rz, disjoint, 0, 0.013});
            operations.push_back(Operation{code, qubit});
        }
        QRegister direct(8);
        execute_direct(direct, operations);
        QRegister optimized(8);
        DependencyOperationPlan plan(operations);
        plan.execute(optimized);
        require(
            fidelity(direct.materialize(12), optimized.materialize(12)) > 1.0 - 2e-11,
            "dependency optimization changed the quantum state");
        require(plan.optimized_operation_count() < operations.size() / 2U,
                "dependency optimization did not reduce the test circuit");
    }

    {
        constexpr std::size_t components = 12U;
        constexpr std::size_t width = 5U;
        QRegister sequential = make_components(components, width);
        QRegister parallel = sequential;
        std::vector<Operation> operations;
        for (std::size_t component = 0; component < components; ++component) {
            const std::size_t offset = component * width;
            for (std::size_t round = 0; round < 20U; ++round) {
                operations.push_back(Operation{
                    OperationCode::Ry,
                    static_cast<QubitId>(offset + round % width),
                    0,
                    0.007 * static_cast<double>(round + 1U),
                });
                operations.push_back(Operation{
                    OperationCode::Rz,
                    static_cast<QubitId>(offset + (round + 2U) % width),
                    0,
                    -0.005 * static_cast<double>(round + 1U),
                });
            }
        }
        execute_direct(sequential, operations);
        IndependentComponentPlan plan(operations);
        plan.execute(parallel, 4U);
        require(sequential.validate() && parallel.validate(),
                "independent component execution produced an invalid state");
        require(sequential.component_count() == parallel.component_count(),
                "independent component execution changed component count");
        compare_factorized_states(sequential, parallel);
    }

    {
        bool rejected = false;
        try {
            const std::vector<Operation> operations{
                Operation{OperationCode::Cnot, 0, 1},
            };
            IndependentComponentPlan plan(operations);
            (void)plan;
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "independent component plan accepted a two-qubit gate");
    }

    {
        bool rejected = false;
        try {
            const std::vector<Operation> operations{
                Operation{
                    OperationCode::Ry,
                    0,
                    0,
                    std::numeric_limits<double>::infinity(),
                },
            };
            DependencyOperationPlan plan(operations);
            (void)plan;
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "dependency plan accepted a non-finite rotation");
    }

    std::cout << "execution acceleration tests passed\n";
    return 0;
}

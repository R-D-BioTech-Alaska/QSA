#include "qubit/qstabilizer.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <exception>
#include <thread>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using qubit::QRegister;
using qubit::QStateError;
using qubit::QubitId;
using qubit::StabilizerConfig;
using qubit::StabilizerOperation;
using qubit::StabilizerOperationCode;
using qubit::StabilizerState;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

[[nodiscard]] StabilizerOperation random_operation(
    std::mt19937_64& generator,
    std::size_t qubits) {
    const QubitId first = static_cast<QubitId>(generator() % qubits);
    QubitId second = static_cast<QubitId>(generator() % qubits);
    if (second == first) {
        second = static_cast<QubitId>((second + 1U) % qubits);
    }
    return StabilizerOperation{
        static_cast<StabilizerOperationCode>(generator() % 9U),
        first,
        second,
    };
}

void apply_scalar(StabilizerState& state, const StabilizerOperation& operation) {
    switch (operation.code) {
        case StabilizerOperationCode::X:
            state.apply_x(operation.first);
            break;
        case StabilizerOperationCode::Y:
            state.apply_y(operation.first);
            break;
        case StabilizerOperationCode::Z:
            state.apply_z(operation.first);
            break;
        case StabilizerOperationCode::H:
            state.apply_h(operation.first);
            break;
        case StabilizerOperationCode::S:
            state.apply_s(operation.first);
            break;
        case StabilizerOperationCode::Sdg:
            state.apply_sdg(operation.first);
            break;
        case StabilizerOperationCode::Cnot:
            state.apply_cnot(operation.first, operation.second);
            break;
        case StabilizerOperationCode::Cz:
            state.apply_cz(operation.first, operation.second);
            break;
        case StabilizerOperationCode::Swap:
            state.apply_swap(operation.first, operation.second);
            break;
    }
}

void apply_reference(QRegister& state, const StabilizerOperation& operation) {
    switch (operation.code) {
        case StabilizerOperationCode::X:
            state.apply_x(operation.first);
            break;
        case StabilizerOperationCode::Y:
            state.apply_y(operation.first);
            break;
        case StabilizerOperationCode::Z:
            state.apply_z(operation.first);
            break;
        case StabilizerOperationCode::H:
            state.apply_h(operation.first);
            break;
        case StabilizerOperationCode::S:
            state.apply_s(operation.first);
            break;
        case StabilizerOperationCode::Sdg:
            state.apply_sdg(operation.first);
            break;
        case StabilizerOperationCode::Cnot:
            state.apply_cnot(operation.first, operation.second);
            break;
        case StabilizerOperationCode::Cz:
            state.apply_cz(operation.first, operation.second);
            break;
        case StabilizerOperationCode::Swap:
            state.apply_swap(operation.first, operation.second);
            break;
    }
}

void compare_reference_probabilities(
    const StabilizerState& stabilizer,
    const QRegister& reference) {
    for (std::size_t qubit = 0; qubit < stabilizer.qubit_count(); ++qubit) {
        require(std::abs(
                    stabilizer.probability_one(static_cast<QubitId>(qubit)) -
                    reference.probability_one(static_cast<QubitId>(qubit))) < 2e-10,
                "batch probability differs from QRegister");
    }
}

void compare_probabilities(
    const StabilizerState& first,
    const StabilizerState& second) {
    require(first.qubit_count() == second.qubit_count(), "batch state width differs");
    for (std::size_t qubit = 0; qubit < first.qubit_count(); ++qubit) {
        require(std::abs(
                    first.probability_one(static_cast<QubitId>(qubit)) -
                    second.probability_one(static_cast<QubitId>(qubit))) < 1e-12,
                "batch probability differs from scalar execution");
    }
}

}  // namespace

int main() {
    std::mt19937_64 generator(0x5452414E53504F53ULL);
    for (std::size_t trial = 0; trial < 120U; ++trial) {
        const std::size_t qubits = 2U + trial % 9U;
        StabilizerState batched(qubits);
        QRegister reference(qubits);
        const std::size_t operation_count = 1U + (trial * 29U) % 320U;
        std::vector<StabilizerOperation> operations;
        operations.reserve(operation_count);
        for (std::size_t index = 0; index < operation_count; ++index) {
            operations.push_back(random_operation(generator, qubits));
            apply_reference(reference, operations.back());
        }
        batched.apply_batch(operations);
        require(batched.validate_full(),
                "direct-reference batch failed full validation");
        compare_reference_probabilities(batched, reference);
        for (std::size_t qubit = 0; qubit < qubits; ++qubit) {
            const double sample = static_cast<double>(generator() >> 11U) *
                                  (1.0 / 9007199254740992.0);
            require(batched.measure_z(static_cast<QubitId>(qubit), sample) ==
                        reference.measure(static_cast<QubitId>(qubit), sample),
                    "batch measurement differs from QRegister");
            compare_reference_probabilities(batched, reference);
        }
    }
    for (std::size_t trial = 0; trial < 180U; ++trial) {
        const std::size_t qubits = 2U + trial % 66U;
        StabilizerState scalar(qubits);
        StabilizerState batched(qubits);

        const std::size_t prefix_count = trial % 47U;
        for (std::size_t index = 0; index < prefix_count; ++index) {
            const StabilizerOperation operation = random_operation(generator, qubits);
            apply_scalar(scalar, operation);
            apply_scalar(batched, operation);
        }

        if ((trial % 3U) == 0U) {
            const QubitId measured_qubit = static_cast<QubitId>(generator() % qubits);
            const double sample = static_cast<double>(generator() >> 11U) *
                                  (1.0 / 9007199254740992.0);
            require(scalar.measure_z(measured_qubit, sample) ==
                        batched.measure_z(measured_qubit, sample),
                    "pre-batch stabilizer measurement differs");
        }

        const std::size_t operation_count = 1U + (trial * 37U) % 700U;
        std::vector<StabilizerOperation> operations;
        operations.reserve(operation_count);
        for (std::size_t index = 0; index < operation_count; ++index) {
            operations.push_back(random_operation(generator, qubits));
            apply_scalar(scalar, operations.back());
        }
        batched.apply_batch(operations);

        require(scalar.validate_full(), "scalar stabilizer state failed validation");
        require(batched.validate_full(), "batched stabilizer state failed validation");
        compare_probabilities(scalar, batched);
        const std::uint64_t seed = generator();
        require(scalar.measure_all(seed) == batched.measure_all(seed),
                "batched stabilizer measurement differs from scalar execution");
    }

    {
        StabilizerState state(8U);
        state.apply_batch({});
        require(state.validate_full(), "empty stabilizer batch changed the state");
    }

    {
        StabilizerConfig config;
        config.max_batch_scratch_bytes = 64U;
        StabilizerState state(64U, config);
        const std::vector<StabilizerOperation> operations{
            {StabilizerOperationCode::H, 0U, 0U},
        };
        bool rejected = false;
        try {
            state.apply_batch(operations);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "stabilizer batch scratch limit was not enforced");
        require(state.validate_full(), "scratch rejection changed the stabilizer state");
    }

    {
        for (std::size_t qubits : {31U, 32U, 33U, 63U, 64U, 65U}) {
            StabilizerState scalar(qubits);
            StabilizerState batched(qubits);
            std::vector<StabilizerOperation> operations;
            for (std::size_t index = 0; index < 600U; ++index) {
                operations.push_back(random_operation(generator, qubits));
                apply_scalar(scalar, operations.back());
            }
            batched.apply_batch(operations);
            require(batched.validate_full(),
                    "row-boundary batch failed full validation");
            require(scalar.measure_all(0xB017DA7AULL + qubits) ==
                        batched.measure_all(0xB017DA7AULL + qubits),
                    "row-boundary batch measurement differs");
        }
    }

    {
        StabilizerState state(4U);
        const std::vector<StabilizerOperation> invalid{
            {StabilizerOperationCode::H, 0U, 0U},
            {StabilizerOperationCode::Cnot, 1U, 1U},
        };
        bool rejected = false;
        try {
            state.apply_batch(invalid);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "stabilizer batch accepted a repeated two-qubit operand");
        require(state.probability_one(0U) == 0.0,
                "invalid batch partially changed the stabilizer state");
        require(state.validate_full(), "invalid batch changed the stabilizer state");
    }

    {
        StabilizerState state(4U);
        const std::vector<StabilizerOperation> invalid{
            {StabilizerOperationCode::X, 0U, 0U},
            {static_cast<StabilizerOperationCode>(255U), 1U, 0U},
        };
        bool rejected = false;
        try {
            state.apply_batch(invalid);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "stabilizer batch accepted an unknown operation");
        require(state.probability_one(0U) == 0.0,
                "unknown operation rejection partially changed the state");
    }

    {
        constexpr std::size_t qubits = 257U;
        constexpr std::size_t active_qubits = 8U;
        StabilizerState scalar(qubits);
        StabilizerState batched(qubits);
        std::vector<StabilizerOperation> operations;
        operations.reserve(2'000U);
        for (std::size_t index = 0; index < 2'000U; ++index) {
            operations.push_back(random_operation(generator, active_qubits));
            apply_scalar(scalar, operations.back());
        }
        batched.apply_batch(operations);
        require(batched.validate_full(),
                "localized active-column batch failed full validation");
        compare_probabilities(batched, scalar);
        require(batched.measure_all(0x10CA11A7ULL) ==
                    scalar.measure_all(0x10CA11A7ULL),
                "localized active-column batch differs from scalar execution");
    }

    {
        constexpr std::size_t qubits = 256U;
        constexpr std::size_t active_qubits = 200U;
        std::vector<StabilizerOperation> operations;
        operations.reserve(active_qubits);
        StabilizerState scalar(qubits);
        for (std::size_t qubit = 0; qubit < active_qubits; ++qubit) {
            operations.push_back(StabilizerOperation{
                StabilizerOperationCode::H,
                static_cast<QubitId>(qubit),
                0U,
            });
            apply_scalar(scalar, operations.back());
        }

        StabilizerState full_path(qubits);
        full_path.apply_batch(operations);
        compare_probabilities(full_path, scalar);

        StabilizerConfig constrained_config;
        constrained_config.max_batch_scratch_bytes = 32U * 1024U;
        StabilizerState fallback_path(qubits, constrained_config);
        fallback_path.apply_batch(operations);
        require(fallback_path.validate_full(),
                "scratch-limited active-column fallback failed validation");
        compare_probabilities(fallback_path, scalar);
        require(full_path.measure_all(0xFA11BAC7ULL) ==
                    fallback_path.measure_all(0xFA11BAC7ULL),
                "full and scratch-limited batch strategies differ");
    }

    {
        constexpr std::size_t workers = 8U;
        constexpr std::size_t qubits = 257U;
        std::vector<StabilizerOperation> operations;
        operations.reserve(5'000U);
        for (std::size_t index = 0; index < 5'000U; ++index) {
            operations.push_back(random_operation(generator, qubits));
        }
        StabilizerState expected(qubits);
        for (const StabilizerOperation& operation : operations) {
            apply_scalar(expected, operation);
        }
        std::vector<StabilizerState> states;
        states.reserve(workers);
        for (std::size_t index = 0; index < workers; ++index) {
            states.emplace_back(qubits);
        }
        std::vector<std::exception_ptr> errors(workers);
        std::vector<std::thread> threads;
        threads.reserve(workers);
        for (std::size_t index = 0; index < workers; ++index) {
            threads.emplace_back([&, index] {
                try {
                    states[index].apply_batch(operations);
                } catch (...) {
                    errors[index] = std::current_exception();
                }
            });
        }
        for (std::thread& thread : threads) {
            thread.join();
        }
        for (std::size_t index = 0; index < workers; ++index) {
            require(errors[index] == nullptr,
                    "independent concurrent batch execution threw");
            require(states[index].validate_full(),
                    "independent concurrent batch state failed validation");
            compare_probabilities(states[index], expected);
            StabilizerState measured_expected = expected;
            require(states[index].measure_all(0xC0A7C0A7ULL) ==
                        measured_expected.measure_all(0xC0A7C0A7ULL),
                    "independent concurrent batch result differs");
        }
    }

    std::cout << "stabilizer transposed-batch tests passed\n";
    return 0;
}

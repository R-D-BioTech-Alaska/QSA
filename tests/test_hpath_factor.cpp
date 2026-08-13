#include "qubit/qhpath_factor.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    const qubit::QComplex& observed,
    const qubit::QComplex& expected,
    double tolerance,
    const char* message) {
    if (!qubit::almost_equal(observed, expected, tolerance)) {
        throw std::runtime_error(
            std::string(message) + ": observed=(" +
            std::to_string(observed.re) + "," + std::to_string(observed.im) +
            ") expected=(" + std::to_string(expected.re) + "," +
            std::to_string(expected.im) + ")");
    }
}

struct DenseReference {
    std::size_t qubits{0U};
    std::vector<qubit::QComplex> amplitudes{};

    explicit DenseReference(std::size_t count) : qubits(count) {
        const std::size_t dimension = std::size_t{1U} << qubits;
        const double amplitude = std::exp2(-0.5 * static_cast<double>(qubits));
        amplitudes.assign(dimension, qubit::QComplex{amplitude, 0.0});
    }

    void apply_single(qubit::QubitId qubit, const qubit::QMatrix2& matrix) {
        const std::size_t mask = std::size_t{1U} << qubit;
        for (std::size_t basis = 0U; basis < amplitudes.size(); ++basis) {
            if ((basis & mask) != 0U) {
                continue;
            }
            const std::size_t one = basis | mask;
            const qubit::QComplex zero_value = amplitudes[basis];
            const qubit::QComplex one_value = amplitudes[one];
            amplitudes[basis] = matrix(0U, 0U) * zero_value + matrix(0U, 1U) * one_value;
            amplitudes[one] = matrix(1U, 0U) * zero_value + matrix(1U, 1U) * one_value;
        }
    }

    void apply_cz(qubit::QubitId first, qubit::QubitId second) {
        const std::size_t mask =
            (std::size_t{1U} << first) | (std::size_t{1U} << second);
        for (std::size_t basis = 0U; basis < amplitudes.size(); ++basis) {
            if ((basis & mask) == mask) {
                amplitudes[basis] = -amplitudes[basis];
            }
        }
    }

    void apply(const qubit::Operation& operation) {
        using qubit::OperationCode;
        switch (operation.code) {
            case OperationCode::H:
                apply_single(operation.first, qubit::gates::h());
                return;
            case OperationCode::Z:
                apply_single(operation.first, qubit::gates::z());
                return;
            case OperationCode::S:
                apply_single(operation.first, qubit::gates::s());
                return;
            case OperationCode::Sdg:
                apply_single(operation.first, qubit::gates::sdg());
                return;
            case OperationCode::T:
                apply_single(operation.first, qubit::gates::t());
                return;
            case OperationCode::Tdg:
                apply_single(operation.first, qubit::gates::tdg());
                return;
            case OperationCode::Rz:
                apply_single(operation.first, qubit::gates::rz(operation.parameter));
                return;
            case OperationCode::Cz:
                apply_cz(operation.first, operation.second);
                return;
            default:
                throw std::runtime_error("dense single-H reference received unsupported operation");
        }
    }
};

std::vector<qubit::Operation> structured_case(std::size_t qubits) {
    using qubit::Operation;
    using qubit::OperationCode;
    std::vector<Operation> operations;
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({
            OperationCode::Rz,
            static_cast<qubit::QubitId>(qubit),
            0U,
            0.137 * static_cast<double>(qubit + 1U),
        });
        operations.push_back({
            (qubit & 1U) == 0U ? OperationCode::T : OperationCode::Sdg,
            static_cast<qubit::QubitId>(qubit),
        });
    }
    for (std::size_t qubit = 0U; qubit + 1U < qubits; ++qubit) {
        operations.push_back({
            OperationCode::Cz,
            static_cast<qubit::QubitId>(qubit),
            static_cast<qubit::QubitId>(qubit + 1U),
        });
    }
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({OperationCode::H, static_cast<qubit::QubitId>(qubit)});
        operations.push_back({
            OperationCode::Rz,
            static_cast<qubit::QubitId>(qubit),
            0U,
            -0.091 * static_cast<double>(qubit + 1U),
        });
    }
    for (std::size_t qubit = 0U; qubit + 2U < qubits; qubit += 2U) {
        operations.push_back({
            OperationCode::Cz,
            static_cast<qubit::QubitId>(qubit),
            static_cast<qubit::QubitId>(qubit + 2U),
        });
    }
    return operations;
}

void dense_equivalence() {
    using qubit::ExactSingleHPathAmplitudePlan;

    for (std::size_t qubits = 2U; qubits <= 6U; ++qubits) {
        const auto operations = structured_case(qubits);
        ExactSingleHPathAmplitudePlan plan(qubits, operations);
        DenseReference dense(qubits);
        for (const auto& operation : operations) {
            dense.apply(operation);
        }

        const std::size_t dimension = std::size_t{1U} << qubits;
        std::vector<std::uint8_t> bits(qubits, 0U);
        for (std::size_t basis = 0U; basis < dimension; ++basis) {
            for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
                bits[qubit] = static_cast<std::uint8_t>((basis >> qubit) & 1U);
            }
            const auto observed = plan.scaled_amplitude_bits(bits);
            require_close(
                observed.amplitude(), dense.amplitudes[basis], 8e-12,
                "single-H path amplitude mismatch");
            require(observed.factor_stats.peak_factor_entries <= 8U,
                "single-H chain differential exceeded bounded factor width");
        }
    }
}

void fixed_qubits_and_scale() {
    using qubit::ExactSingleHPathAmplitudePlan;
    using qubit::Operation;
    using qubit::OperationCode;

    const std::vector<Operation> operations{
        {OperationCode::T, 0U},
        {OperationCode::H, 1U},
        {OperationCode::S, 0U},
        {OperationCode::Rz, 2U, 0U, 0.43},
    };
    ExactSingleHPathAmplitudePlan plan(3U, operations);
    const std::array<std::uint8_t, 3> bits{1U, 0U, 1U};
    const auto result = plan.scaled_amplitude_bits(bits);
    require(result.h_defects == 1U, "single-H mixed fixed-variable defect count mismatch");
    require(std::abs(result.log2_scale + 1.0) < 1e-15,
        "single-H fixed-qubit scale accounting mismatch");
}

void fail_closed() {
    using qubit::ExactSingleHPathAmplitudePlan;
    using qubit::Operation;
    using qubit::OperationCode;
    using qubit::QStateError;

    bool rejected = false;
    try {
        const std::vector<Operation> repeated{
            {OperationCode::H, 0U},
            {OperationCode::H, 0U},
        };
        (void)ExactSingleHPathAmplitudePlan(2U, repeated);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "single-H compiler accepted a repeated Hadamard target");

    rejected = false;
    try {
        const std::vector<Operation> x_gate{{OperationCode::X, 0U}};
        (void)ExactSingleHPathAmplitudePlan(2U, x_gate);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "single-H compiler accepted an unsupported X gate");

    rejected = false;
    try {
        const std::vector<Operation> bad_rz{{
            OperationCode::Rz,
            0U,
            0U,
            std::numeric_limits<double>::quiet_NaN(),
        }};
        (void)ExactSingleHPathAmplitudePlan(2U, bad_rz);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "single-H compiler accepted a non-finite Rz angle");

    const std::vector<Operation> valid{{OperationCode::H, 0U}};
    ExactSingleHPathAmplitudePlan plan(2U, valid);
    const std::array<std::uint8_t, 1> short_bits{0U};
    rejected = false;
    try {
        (void)plan.scaled_amplitude_bits(short_bits);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "single-H compiler accepted the wrong output bit count");
}

}  // namespace

int main() {
    dense_equivalence();
    fixed_qubits_and_scale();
    fail_closed();
    return 0;
}

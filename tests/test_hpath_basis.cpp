#include "qubit/qhpath_basis.hpp"
#include "qubit/qhpath_prepared.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct DenseReference {
    std::size_t qubits{0U};
    std::vector<qubit::QComplex> amplitudes{};

    DenseReference(std::span<const std::uint8_t> input_bits)
        : qubits(input_bits.size()) {
        const std::size_t dimension = std::size_t{1U} << qubits;
        amplitudes.assign(dimension, {});
        std::size_t basis = 0U;
        for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
            basis |= static_cast<std::size_t>(input_bits[qubit]) << qubit;
        }
        amplitudes[basis] = {1.0, 0.0};
    }

    void single(qubit::QubitId qubit, const qubit::QMatrix2& matrix) {
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

    void cz(qubit::QubitId first, qubit::QubitId second) {
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
            case OperationCode::H: single(operation.first, qubit::gates::h()); return;
            case OperationCode::Z: single(operation.first, qubit::gates::z()); return;
            case OperationCode::S: single(operation.first, qubit::gates::s()); return;
            case OperationCode::Sdg: single(operation.first, qubit::gates::sdg()); return;
            case OperationCode::T: single(operation.first, qubit::gates::t()); return;
            case OperationCode::Tdg: single(operation.first, qubit::gates::tdg()); return;
            case OperationCode::Rz:
                single(operation.first, qubit::gates::rz(operation.parameter));
                return;
            case OperationCode::Cz: cz(operation.first, operation.second); return;
            default:
                throw std::runtime_error("dense basis Hpath reference received unsupported operation");
        }
    }
};

std::vector<qubit::Operation> mixed_case() {
    using qubit::Operation;
    using qubit::OperationCode;
    return {
        {OperationCode::Rz, 3U, 0U, 0.19},
        {OperationCode::T, 0U},
        {OperationCode::Cz, 0U, 3U},
        {OperationCode::H, 0U},
        {OperationCode::Rz, 0U, 0U, -0.27},
        {OperationCode::H, 1U},
        {OperationCode::Cz, 0U, 1U},
        {OperationCode::H, 2U},
        {OperationCode::Cz, 0U, 2U},
        {OperationCode::S, 2U},
        {OperationCode::H, 0U},
        {OperationCode::Tdg, 0U},
        {OperationCode::Cz, 0U, 1U},
        {OperationCode::Rz, 1U, 0U, 0.33},
        {OperationCode::H, 1U},
        {OperationCode::Sdg, 1U},
        {OperationCode::Cz, 0U, 1U},
        {OperationCode::Cz, 1U, 3U},
    };
}

std::vector<qubit::Operation> plus_emulation(
    std::span<const std::uint8_t> input_bits,
    std::span<const qubit::Operation> operations) {
    using qubit::Operation;
    using qubit::OperationCode;
    std::vector<Operation> result;
    result.reserve(input_bits.size() * 2U + operations.size());
    for (std::size_t qubit = 0U; qubit < input_bits.size(); ++qubit) {
        if (input_bits[qubit] != 0U) {
            result.push_back({OperationCode::Z, static_cast<qubit::QubitId>(qubit)});
        }
        result.push_back({OperationCode::H, static_cast<qubit::QubitId>(qubit)});
    }
    result.insert(result.end(), operations.begin(), operations.end());
    return result;
}

qubit::ExactHadamardPathConfig config() {
    qubit::ExactHadamardPathConfig result;
    result.factor.max_variables = 4096U;
    result.factor.max_factors = 20000U;
    result.factor.max_factor_entries = 4096U;
    result.factor.max_compiled_index_entries = 1U << 20U;
    result.factor.reuse_workspace_slots = true;
    result.max_qubits = 4096U;
    result.max_operations = 20000U;
    result.max_h_events = 4096U;
    result.max_metadata_bytes = 64U * 1024U * 1024U;
    return result;
}

void dense_and_emulated_equivalence() {
    using qubit::ExactPreparedBasisHadamardPathPlan;
    using qubit::ExactPreparedHadamardPathPlan;

    const auto operations = mixed_case();
    const std::array<std::array<std::uint8_t, 4>, 4> inputs{
        std::array<std::uint8_t, 4>{0U, 0U, 0U, 0U},
        std::array<std::uint8_t, 4>{1U, 0U, 1U, 0U},
        std::array<std::uint8_t, 4>{0U, 1U, 0U, 1U},
        std::array<std::uint8_t, 4>{1U, 1U, 1U, 1U},
    };

    for (const auto& input : inputs) {
        ExactPreparedBasisHadamardPathPlan native(input, operations, config());
        const auto emulated_operations = plus_emulation(input, operations);
        ExactPreparedHadamardPathPlan emulated(4U, emulated_operations, config());
        auto native_workspace = native.workspace();
        auto emulated_workspace = emulated.workspace();

        require(native.stats().h_events == 5U, "basis Hpath H-event count mismatch");
        require(native.stats().h_active_qubits == 3U, "basis Hpath active-qubit count mismatch");
        require(native.stats().removed_first_h_variables == 3U,
            "basis Hpath did not remove one first-H variable per active qubit");
        require(native.stats().factor_variables == 2U,
            "basis Hpath hidden-variable count mismatch");
        require(native.stats().fixed_support_qubits == 1U,
            "basis Hpath fixed-support qubit count mismatch");
        require(emulated.stats().factor_variables == 9U,
            "plus-state basis emulation variable count changed unexpectedly");

        DenseReference dense(input);
        for (const auto& operation : operations) {
            dense.apply(operation);
        }

        std::array<std::uint8_t, 4> output{};
        for (std::size_t basis = 0U; basis < 16U; ++basis) {
            for (std::size_t qubit = 0U; qubit < output.size(); ++qubit) {
                output[qubit] = static_cast<std::uint8_t>((basis >> qubit) & 1U);
            }
            const qubit::QComplex native_value =
                native.amplitude_bits(output, native_workspace);
            const qubit::QComplex emulated_value =
                emulated.amplitude_bits(output, emulated_workspace);
            require(qubit::almost_equal(native_value, dense.amplitudes[basis], 3e-11),
                "basis-native Hpath amplitude differs from dense reference");
            require(qubit::almost_equal(native_value, emulated_value, 3e-11),
                "basis-native Hpath amplitude differs from plus-state preparation emulation");
            if (output[3] != input[3]) {
                require(native_value.norm2() == 0.0,
                    "basis-native Hpath did not reject impossible no-H output support");
            }
        }
    }
}

void single_h_and_no_h_are_exact() {
    using qubit::ExactPreparedBasisHadamardPathPlan;
    using qubit::Operation;
    using qubit::OperationCode;

    const std::array<std::uint8_t, 3> input{1U, 0U, 1U};
    const std::vector<Operation> operations{
        {OperationCode::T, 0U},
        {OperationCode::Cz, 0U, 2U},
        {OperationCode::H, 1U},
        {OperationCode::Rz, 1U, 0U, 0.41},
    };
    ExactPreparedBasisHadamardPathPlan plan(input, operations, config());
    auto workspace = plan.workspace();
    require(plan.stats().factor_variables == 0U,
        "single-H basis case allocated a hidden factor variable");
    require(workspace.rebind_count() == 0U,
        "single-H basis workspace started with rebinds");

    std::array<std::uint8_t, 3> output{1U, 0U, 1U};
    const auto zero_output = plan.amplitude_bits(output, workspace);
    output[1] = 1U;
    const auto one_output = plan.amplitude_bits(output, workspace);
    require(std::abs(zero_output.norm2() - 0.5) <= 2e-12,
        "single-H zero-output probability mismatch");
    require(std::abs(one_output.norm2() - 0.5) <= 2e-12,
        "single-H one-output probability mismatch");
    output[0] = 0U;
    require(plan.amplitude_bits(output, workspace).norm2() == 0.0,
        "no-H support mismatch did not return exact zero");
}

void workspace_identity_and_caps_are_enforced() {
    using qubit::ExactPreparedBasisHadamardPathPlan;
    using qubit::Operation;
    using qubit::OperationCode;
    using qubit::QStateError;

    const auto operations = mixed_case();
    const std::array<std::uint8_t, 4> first_input{0U, 0U, 0U, 0U};
    const std::array<std::uint8_t, 4> second_input{1U, 0U, 0U, 0U};
    ExactPreparedBasisHadamardPathPlan first(first_input, operations, config());
    ExactPreparedBasisHadamardPathPlan second(second_input, operations, config());
    auto wrong_workspace = second.workspace();
    const std::array<std::uint8_t, 4> output{0U, 0U, 0U, 0U};
    bool rejected = false;
    try {
        (void)first.amplitude_bits(output, wrong_workspace);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "basis Hpath accepted a workspace from another plan");

    auto h_limited = config();
    h_limited.max_h_events = 4U;
    rejected = false;
    try {
        (void)ExactPreparedBasisHadamardPathPlan(first_input, operations, h_limited);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "basis Hpath H-event cap did not reject");

    auto variable_limited = config();
    variable_limited.factor.max_variables = 1U;
    rejected = false;
    try {
        (void)ExactPreparedBasisHadamardPathPlan(first_input, operations, variable_limited);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "basis Hpath hidden-variable cap did not reject");

    rejected = false;
    try {
        const std::vector<Operation> unsupported{{OperationCode::X, 0U}};
        (void)ExactPreparedBasisHadamardPathPlan(first_input, unsupported, config());
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "basis Hpath accepted unsupported X");

    rejected = false;
    try {
        const std::array<std::uint8_t, 4> bad_input{0U, 0U, 2U, 0U};
        (void)ExactPreparedBasisHadamardPathPlan(bad_input, operations, config());
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "basis Hpath accepted non-binary input bits");

    rejected = false;
    try {
        const std::vector<Operation> bad_rz{{
            OperationCode::Rz, 0U, 0U, std::numeric_limits<double>::quiet_NaN()}};
        (void)ExactPreparedBasisHadamardPathPlan(first_input, bad_rz, config());
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "basis Hpath accepted non-finite Rz");
}

}  // namespace

int main() {
    dense_and_emulated_equivalence();
    single_h_and_no_h_are_exact();
    workspace_identity_and_caps_are_enforced();
    return 0;
}

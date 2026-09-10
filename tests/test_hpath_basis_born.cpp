#include "qubit/qhpath_basis_born.hpp"

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
        amplitudes.assign(dimension, qubit::QComplex{});
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
                throw std::runtime_error("dense basis Born reference received unsupported operation");
        }
    }

    std::vector<qubit::QComplex> marginal(std::span<const std::size_t> retained) const {
        const std::size_t entries = std::size_t{1U} << retained.size();
        std::vector<qubit::QComplex> result(entries, qubit::QComplex{});
        for (std::size_t basis = 0U; basis < amplitudes.size(); ++basis) {
            std::size_t index = 0U;
            for (std::size_t position = 0U; position < retained.size(); ++position) {
                index |= ((basis >> retained[position]) & 1U) << position;
            }
            result[index].re += amplitudes[basis].norm2();
        }
        return result;
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

qubit::ExactHadamardBornConfig config() {
    qubit::ExactHadamardBornConfig result;
    result.factor.max_variables = 4096U;
    result.factor.max_factors = 20000U;
    result.factor.max_factor_entries = 4096U;
    result.factor.max_compiled_index_entries = 1U << 20U;
    result.factor.reuse_workspace_slots = true;
    result.max_qubits = 4096U;
    result.max_operations = 20000U;
    result.max_h_events = 4096U;
    result.max_retained_qubits = 8U;
    return result;
}

void compare_vector(
    std::span<const qubit::QComplex> observed,
    std::span<const qubit::QComplex> expected,
    double tolerance,
    const char* message) {
    require(observed.size() == expected.size(), message);
    for (std::size_t index = 0U; index < observed.size(); ++index) {
        require(qubit::almost_equal(observed[index], expected[index], tolerance), message);
    }
}

void dense_and_emulated_marginals_match() {
    using qubit::ExactBasisHadamardBornMarginalPlan;
    using qubit::ExactHadamardBornMarginalPlan;

    const auto operations = mixed_case();
    const std::array<std::uint8_t, 4> input{1U, 0U, 1U, 1U};
    DenseReference dense(input);
    for (const auto& operation : operations) {
        dense.apply(operation);
    }
    const auto emulated_operations = plus_emulation(input, operations);

    const std::array<std::vector<std::size_t>, 4> retained_sets{
        std::vector<std::size_t>{},
        std::vector<std::size_t>{0U},
        std::vector<std::size_t>{3U},
        std::vector<std::size_t>{0U, 2U, 3U},
    };

    for (const auto& retained : retained_sets) {
        ExactBasisHadamardBornMarginalPlan native(input, operations, retained, config());
        ExactHadamardBornMarginalPlan emulated(
            input.size(), emulated_operations, retained, config());
        auto native_workspace = native.workspace();
        auto emulated_workspace = emulated.workspace();

        const auto native_values = native.marginal(native_workspace);
        const auto emulated_values = emulated.marginal(emulated_workspace);
        const auto dense_values = dense.marginal(retained);
        compare_vector(native_values, dense_values, 5e-11,
            "basis-native Born marginal differs from dense reference");
        compare_vector(native_values, emulated_values, 5e-11,
            "basis-native Born marginal differs from plus-state basis emulation");

        require(native.stats().h_events == 5U,
            "basis-native Born H-event count mismatch");
        require(native.stats().h_active_qubits == 3U,
            "basis-native Born active-qubit count mismatch");
        require(native.stats().hidden_variables_per_side == 2U,
            "basis-native Born hidden-variable count mismatch");
        require(native.stats().removed_first_h_variables_per_side == 3U,
            "basis-native Born first-H removal count mismatch");
    }

    const std::vector<std::size_t> retained{0U, 2U, 3U};
    ExactBasisHadamardBornMarginalPlan native(input, operations, retained, config());
    ExactHadamardBornMarginalPlan emulated(
        input.size(), emulated_operations, retained, config());
    require(native.stats().physical_output_variables == 4U,
        "basis-native Born physical-output variable count mismatch");
    require(native.stats().retained_fixed_support_qubits == 1U,
        "basis-native Born retained fixed-support count mismatch");
    require(native.stats().born_variables == 8U,
        "basis-native Born total variable count mismatch");
    require(emulated.stats().born_variables == 22U,
        "plus-state basis Born emulation variable count changed unexpectedly");
}

void deterministic_no_h_support_is_preserved() {
    using qubit::ExactBasisHadamardBornMarginalPlan;
    using qubit::Operation;
    using qubit::OperationCode;

    const std::array<std::uint8_t, 3> input{1U, 0U, 1U};
    const std::vector<Operation> operations{
        {OperationCode::T, 0U},
        {OperationCode::Cz, 0U, 2U},
        {OperationCode::Rz, 2U, 0U, 0.41},
    };
    const std::array<std::size_t, 3> retained{0U, 1U, 2U};
    ExactBasisHadamardBornMarginalPlan plan(input, operations, retained, config());
    auto workspace = plan.workspace();
    const auto values = plan.marginal(workspace);
    require(values.size() == 8U, "no-H basis Born output size mismatch");
    for (std::size_t index = 0U; index < values.size(); ++index) {
        const double expected = index == 5U ? 1.0 : 0.0;
        require(std::abs(values[index].re - expected) <= 1e-12,
            "no-H basis Born deterministic support mismatch");
        require(std::abs(values[index].im) <= 1e-12,
            "no-H basis Born deterministic marginal became complex");
    }
    require(plan.stats().born_variables == 3U,
        "no-H basis Born allocated variables beyond retained deterministic outputs");
}

void workspace_and_limits_fail_closed() {
    using qubit::ExactBasisHadamardBornMarginalPlan;
    using qubit::Operation;
    using qubit::OperationCode;
    using qubit::QStateError;

    const auto operations = mixed_case();
    const std::array<std::uint8_t, 4> first_input{0U, 0U, 0U, 0U};
    const std::array<std::uint8_t, 4> second_input{1U, 0U, 0U, 0U};
    const std::array<std::size_t, 1> retained{0U};
    ExactBasisHadamardBornMarginalPlan first(first_input, operations, retained, config());
    ExactBasisHadamardBornMarginalPlan second(second_input, operations, retained, config());
    auto wrong_workspace = second.workspace();
    bool rejected = false;
    try {
        (void)first.marginal(wrong_workspace);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "basis Born accepted a workspace from another plan");

    auto h_limited = config();
    h_limited.max_h_events = 4U;
    rejected = false;
    try {
        (void)ExactBasisHadamardBornMarginalPlan(
            first_input, operations, retained, h_limited);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "basis Born H-event cap did not reject");

    auto variable_limited = config();
    variable_limited.factor.max_variables = 4U;
    rejected = false;
    try {
        const std::array<std::size_t, 3> wide_retained{0U, 2U, 3U};
        (void)ExactBasisHadamardBornMarginalPlan(
            first_input, operations, wide_retained, variable_limited);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "basis Born variable cap did not reject");

    rejected = false;
    try {
        const std::vector<Operation> unsupported{{OperationCode::X, 0U}};
        (void)ExactBasisHadamardBornMarginalPlan(
            first_input, unsupported, retained, config());
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "basis Born accepted unsupported X");

    rejected = false;
    try {
        const std::array<std::uint8_t, 4> bad_input{0U, 2U, 0U, 0U};
        (void)ExactBasisHadamardBornMarginalPlan(
            bad_input, operations, retained, config());
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "basis Born accepted non-binary input");

    rejected = false;
    try {
        const std::vector<Operation> bad_rz{{
            OperationCode::Rz, 0U, 0U, std::numeric_limits<double>::quiet_NaN()}};
        (void)ExactBasisHadamardBornMarginalPlan(
            first_input, bad_rz, retained, config());
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "basis Born accepted non-finite Rz");
}

}  // namespace

int main() {
    dense_and_emulated_marginals_match();
    deterministic_no_h_support_is_preserved();
    workspace_and_limits_fail_closed();
    return 0;
}

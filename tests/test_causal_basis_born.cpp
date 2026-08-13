#include "qubit/qcausal_basis_born.hpp"

#include <algorithm>
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

    explicit DenseReference(std::span<const std::uint8_t> input_bits)
        : qubits(input_bits.size()) {
        amplitudes.assign(std::size_t{1U} << qubits, qubit::QComplex{});
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
            const auto zero_value = amplitudes[basis];
            const auto one_value = amplitudes[one];
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
                throw std::runtime_error("dense causal basis Born reference received unsupported operation");
        }
    }

    std::vector<qubit::QComplex> marginal(std::span<const std::size_t> retained) const {
        std::vector<qubit::QComplex> result(
            std::size_t{1U} << retained.size(), qubit::QComplex{});
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

qubit::ExactCausalBasisHadamardBornConfig config() {
    qubit::ExactCausalBasisHadamardBornConfig result;
    result.light_cone.max_qubits = 10000U;
    result.light_cone.max_operations = 100000U;
    result.light_cone.max_active_qubits = 1024U;
    result.light_cone.max_active_operations = 10000U;
    result.born.factor.max_variables = 4096U;
    result.born.factor.max_factors = 20000U;
    result.born.factor.max_factor_entries = 4096U;
    result.born.factor.max_compiled_index_entries = 1U << 20U;
    result.born.factor.reuse_workspace_slots = true;
    result.born.max_qubits = 4096U;
    result.born.max_operations = 20000U;
    result.born.max_h_events = 4096U;
    result.born.max_retained_qubits = 8U;
    return result;
}

std::vector<qubit::Operation> small_operations() {
    using qubit::Operation;
    using qubit::OperationCode;
    return {
        {OperationCode::H, 3U},
        {OperationCode::H, 5U},
        {OperationCode::Rz, 5U, 0U, 0.17},
        {OperationCode::H, 2U},
        {OperationCode::Cz, 2U, 3U},
        {OperationCode::H, 4U},
        {OperationCode::Cz, 4U, 5U},
        {OperationCode::H, 1U},
        {OperationCode::Cz, 1U, 2U},
        {OperationCode::T, 6U},
        {OperationCode::H, 0U},
        {OperationCode::Cz, 0U, 1U},
        {OperationCode::Rz, 0U, 0U, -0.31},
        {OperationCode::Sdg, 2U},
        {OperationCode::Cz, 5U, 6U},
        {OperationCode::H, 0U},
        {OperationCode::Tdg, 0U},
    };
}

void causal_marginal_matches_dense_and_full_exact() {
    using qubit::ExactBasisHadamardBornMarginalPlan;
    using qubit::ExactCausalBasisHadamardBornPlan;

    const std::array<std::uint8_t, 7> input{1U, 0U, 1U, 0U, 1U, 0U, 1U};
    const auto operations = small_operations();
    const std::array<std::size_t, 2> retained{0U, 2U};

    DenseReference dense(input);
    for (const auto& operation : operations) {
        dense.apply(operation);
    }
    const auto expected = dense.marginal(retained);

    ExactCausalBasisHadamardBornPlan causal(input, operations, retained, config());
    auto causal_workspace = causal.workspace();
    const auto observed = causal.marginal(causal_workspace);

    ExactBasisHadamardBornMarginalPlan full(input, operations, retained, config().born);
    auto full_workspace = full.workspace();
    const auto full_values = full.marginal(full_workspace);

    require(observed.size() == expected.size(), "causal basis Born output size mismatch");
    for (std::size_t index = 0U; index < observed.size(); ++index) {
        require(qubit::almost_equal(observed[index], expected[index], 5e-11),
            "causal basis Born differs from dense full-circuit marginal");
        require(qubit::almost_equal(observed[index], full_values[index], 5e-11),
            "causal basis Born differs from full exact Born compiler");
    }

    const auto& stats = causal.stats().light_cone;
    require(stats.full_qubits == 7U && stats.active_qubits == 4U,
        "causal basis Born active-qubit cone mismatch");
    require(stats.pruned_qubits == 3U,
        "causal basis Born did not prune unrelated qubits");
    require(stats.active_operations < stats.full_operations,
        "causal basis Born did not prune unrelated operations");
    const std::array<std::size_t, 4> expected_active{0U, 1U, 2U, 3U};
    require(std::equal(
                causal.light_cone().active_qubits().begin(),
                causal.light_cone().active_qubits().end(),
                expected_active.begin(), expected_active.end()),
        "causal basis Born active support changed");
    require(causal.light_cone().retained_local_qubits()[0] == 0U &&
            causal.light_cone().retained_local_qubits()[1] == 2U,
        "causal basis Born retained remap mismatch");
}

void deterministic_isolated_output_stays_local() {
    using qubit::ExactCausalBasisHadamardBornPlan;

    const std::array<std::uint8_t, 7> input{1U, 0U, 1U, 0U, 1U, 0U, 1U};
    const auto operations = small_operations();
    const std::array<std::size_t, 1> retained{4U};
    ExactCausalBasisHadamardBornPlan causal(input, operations, retained, config());
    require(causal.stats().light_cone.active_qubits <= 3U,
        "isolated causal output pulled the unrelated main component into its cone");
    auto workspace = causal.workspace();
    const auto values = causal.marginal(workspace);
    require(values.size() == 2U, "isolated causal marginal output size mismatch");
    require(std::abs(values[0].re + values[1].re - 1.0) <= 1e-10,
        "isolated causal marginal lost probability mass");
}

void limits_and_nonunitary_inputs_fail_closed() {
    using qubit::ExactCausalBasisHadamardBornPlan;
    using qubit::Operation;
    using qubit::OperationCode;
    using qubit::QStateError;

    const std::array<std::uint8_t, 7> input{1U, 0U, 1U, 0U, 1U, 0U, 1U};
    const auto operations = small_operations();
    const std::array<std::size_t, 1> retained{0U};

    auto limited = config();
    limited.light_cone.max_active_qubits = 2U;
    bool rejected = false;
    try {
        (void)ExactCausalBasisHadamardBornPlan(input, operations, retained, limited);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "causal basis Born active-qubit cap did not reject");

    limited = config();
    limited.light_cone.max_active_operations = 3U;
    rejected = false;
    try {
        (void)ExactCausalBasisHadamardBornPlan(input, operations, retained, limited);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "causal basis Born active-operation cap did not reject");

    rejected = false;
    try {
        auto noisy = operations;
        noisy.push_back({OperationCode::BitFlipTrajectory, 6U, 0U, 0.1, 0.5});
        (void)ExactCausalBasisHadamardBornPlan(input, noisy, retained, config());
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "causal basis Born accepted trajectory noise as a unitary light cone");

    rejected = false;
    try {
        const std::array<std::uint8_t, 7> bad_input{1U, 0U, 1U, 2U, 1U, 0U, 1U};
        (void)ExactCausalBasisHadamardBornPlan(bad_input, operations, retained, config());
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "causal basis Born accepted non-binary input bits");

    rejected = false;
    try {
        const std::array<std::size_t, 2> duplicate{0U, 0U};
        (void)ExactCausalBasisHadamardBornPlan(input, operations, duplicate, config());
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "causal basis Born accepted duplicate retained qubits");
}

}  // namespace

int main() {
    causal_marginal_matches_dense_and_full_exact();
    deterministic_isolated_output_stays_local();
    limits_and_nonunitary_inputs_fail_closed();
    return 0;
}

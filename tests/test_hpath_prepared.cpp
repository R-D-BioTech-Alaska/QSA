#include "qubit/qhpath_prepared.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
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

void prepared_matches_direct_all_outputs() {
    using qubit::ExactHadamardPathAmplitudePlan;
    using qubit::ExactPreparedHadamardPathPlan;
    using qubit::Operation;
    using qubit::OperationCode;

    const std::vector<Operation> operations{
        {OperationCode::H, 0U},
        {OperationCode::H, 2U},
        {OperationCode::Rz, 0U, 0U, 0.37},
        {OperationCode::T, 2U},
        {OperationCode::Cz, 0U, 2U},
        {OperationCode::Cz, 0U, 1U},
        {OperationCode::Sdg, 1U},
        {OperationCode::H, 0U},
        {OperationCode::S, 0U},
        {OperationCode::Cz, 0U, 3U},
        {OperationCode::Rz, 2U, 0U, -0.21},
        {OperationCode::H, 2U},
        {OperationCode::Tdg, 2U},
        {OperationCode::Z, 3U},
        {OperationCode::Cz, 1U, 3U},
    };

    ExactPreparedHadamardPathPlan prepared(4U, operations, config());
    ExactHadamardPathAmplitudePlan direct(4U, operations, config());
    auto workspace = prepared.workspace();
    require(prepared.stats().output_bindings >= 3U,
        "prepared plan did not retain expected output bindings");
    require(prepared.stats().fixed_cz_pairs >= 2U,
        "prepared plan did not retain fixed CZ phase pairs");

    std::array<std::uint8_t, 4> bits{};
    for (std::size_t basis = 0U; basis < 16U; ++basis) {
        for (std::size_t qubit = 0U; qubit < bits.size(); ++qubit) {
            bits[qubit] = static_cast<std::uint8_t>((basis >> qubit) & 1U);
        }
        const auto prepared_value = prepared.scaled_amplitude_bits(bits, workspace);
        const auto direct_value = direct.scaled_amplitude_bits(bits);
        require(qubit::almost_equal(prepared_value.mantissa, direct_value.mantissa, 3e-12),
            "prepared Hpath mantissa differs from direct Hpath");
        require(std::abs(prepared_value.log2_scale - direct_value.log2_scale) <= 1e-15,
            "prepared Hpath scale differs from direct Hpath");
        const double prepared_logp = prepared_value.log2_probability();
        const double direct_logp = direct_value.log2_probability();
        require(
            (std::isinf(prepared_logp) && std::isinf(direct_logp)) ||
                std::abs(prepared_logp - direct_logp) <= 2e-11,
            "prepared Hpath log probability differs from direct Hpath");
    }
    require(workspace.rebind_count() == prepared.stats().output_bindings * 16U,
        "prepared Hpath targeted rebind count mismatch");

    auto second_workspace = prepared.workspace();
    bits = {1U, 0U, 1U, 1U};
    const auto second_value = prepared.scaled_amplitude_bits(bits, second_workspace);
    const auto second_direct = direct.scaled_amplitude_bits(bits);
    require(qubit::almost_equal(second_value.mantissa, second_direct.mantissa, 3e-12),
        "second compact workspace inherited stale prepared bindings");
    require(second_workspace.rebind_count() == prepared.stats().output_bindings,
        "second compact workspace rebind count is not independent");
    require(workspace.rebind_count() == prepared.stats().output_bindings * 16U,
        "first compact workspace rebind count changed after second workspace query");
}

void no_h_path_matches_direct() {
    using qubit::ExactHadamardPathAmplitudePlan;
    using qubit::ExactPreparedHadamardPathPlan;
    using qubit::Operation;
    using qubit::OperationCode;

    const std::vector<Operation> operations{
        {OperationCode::Rz, 0U, 0U, 0.17},
        {OperationCode::T, 1U},
        {OperationCode::Sdg, 2U},
        {OperationCode::Cz, 0U, 1U},
        {OperationCode::Cz, 1U, 2U},
    };
    ExactPreparedHadamardPathPlan prepared(3U, operations, config());
    ExactHadamardPathAmplitudePlan direct(3U, operations, config());
    auto workspace = prepared.workspace();
    require(prepared.stats().h_events == 0U, "no-H prepared plan created H events");
    require(prepared.stats().factor_count == 0U, "no-H prepared plan created factor storage");

    std::array<std::uint8_t, 3> bits{};
    for (std::size_t basis = 0U; basis < 8U; ++basis) {
        for (std::size_t qubit = 0U; qubit < bits.size(); ++qubit) {
            bits[qubit] = static_cast<std::uint8_t>((basis >> qubit) & 1U);
        }
        require(qubit::almost_equal(
                    prepared.scaled_amplitude_bits(bits, workspace).mantissa,
                    direct.scaled_amplitude_bits(bits).mantissa,
                    2e-12),
            "no-H prepared amplitude differs from direct Hpath");
    }
    require(workspace.rebind_count() == 0U,
        "no-H compact workspace reported source rebinds");
}

void workspace_identity_is_enforced() {
    using qubit::ExactPreparedHadamardPathPlan;
    using qubit::Operation;
    using qubit::OperationCode;
    using qubit::QStateError;

    const std::vector<Operation> first_ops{{OperationCode::H, 0U}};
    const std::vector<Operation> second_ops{{OperationCode::H, 1U}};
    ExactPreparedHadamardPathPlan first(2U, first_ops, config());
    ExactPreparedHadamardPathPlan second(2U, second_ops, config());
    auto wrong_workspace = second.workspace();
    const std::array<std::uint8_t, 2> bits{0U, 0U};
    bool rejected = false;
    try {
        (void)first.log2_probability_bits(bits, wrong_workspace);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "prepared Hpath accepted a workspace from another plan");
}

void broad_width_rejects() {
    using qubit::ExactPreparedHadamardPathPlan;
    using qubit::Operation;
    using qubit::OperationCode;
    using qubit::QStateError;

    constexpr std::size_t qubits = 16U;
    std::vector<Operation> operations;
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({OperationCode::H, static_cast<qubit::QubitId>(qubit)});
    }
    for (std::size_t first = 0U; first < qubits; ++first) {
        for (std::size_t second = first + 1U; second < qubits; ++second) {
            operations.push_back({
                OperationCode::Cz,
                static_cast<qubit::QubitId>(first),
                static_cast<qubit::QubitId>(second),
            });
        }
    }
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({OperationCode::H, static_cast<qubit::QubitId>(qubit)});
    }

    auto limited = config();
    limited.factor.max_factor_entries = 1024U;
    bool rejected = false;
    try {
        (void)ExactPreparedHadamardPathPlan(qubits, operations, limited);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "prepared Hpath did not reject broad induced width");
}

}  // namespace

int main() {
    prepared_matches_direct_all_outputs();
    no_h_path_matches_direct();
    workspace_identity_is_enforced();
    broad_width_rejects();
    return 0;
}

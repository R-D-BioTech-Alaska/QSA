#include "qubit/c_api.h"
#include "qubit/qbehavior.hpp"
#include "qubit/qbehavior_c_api.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "C API TEST FAILURE: " << message;
        const char* error = qstate_last_error();
        if (error != nullptr && std::strlen(error) != 0U) {
            std::cerr << " (native error: " << error << ')';
        }
        std::cerr << '\n';
        std::exit(1);
    }
}

void require_near(double actual, double expected, double tolerance, const std::string& message) {
    require(std::abs(actual - expected) <= tolerance, message);
}

struct BehaviorControlState {
    qubit::BehaviorFoldPlan plan{};
    std::size_t last_index{0U};
};

std::uint64_t behavior_join(
    std::uint64_t left,
    std::uint64_t right,
    qubit::BehaviorJoin join) {
    if (join == qubit::BehaviorJoin::And) return left & right;
    if (join == qubit::BehaviorJoin::Or) return left | right;
    return left ^ right;
}

bool behavior_plan_less(
    const qubit::BehaviorFoldPlan& left,
    const qubit::BehaviorFoldPlan& right) {
    if (left.leaf_count() != right.leaf_count()) return left.leaf_count() < right.leaf_count();
    if (left.branch_indices != right.branch_indices) return left.branch_indices < right.branch_indices;
    return left.joins < right.joins;
}

std::vector<std::map<std::uint64_t, BehaviorControlState>> behavior_control(
    const std::vector<std::uint64_t>& masks,
    std::size_t max_leaves) {
    const std::array<qubit::BehaviorJoin, 3> joins{
        qubit::BehaviorJoin::And,
        qubit::BehaviorJoin::Or,
        qubit::BehaviorJoin::Xor,
    };
    std::vector<std::map<std::uint64_t, BehaviorControlState>> levels(max_leaves);
    for (std::size_t index = 0U; index < masks.size(); ++index) {
        qubit::BehaviorFoldPlan plan{{static_cast<std::uint32_t>(index)}, {}};
        const auto found = levels[0].find(masks[index]);
        if (found == levels[0].end() || index < found->second.last_index ||
            (index == found->second.last_index && behavior_plan_less(plan, found->second.plan))) {
            levels[0][masks[index]] = BehaviorControlState{std::move(plan), index};
        }
    }
    for (std::size_t leaf_count = 2U; leaf_count <= max_leaves; ++leaf_count) {
        for (const auto& [behavior, state] : levels[leaf_count - 2U]) {
            for (std::size_t branch = state.last_index + 1U; branch < masks.size(); ++branch) {
                for (const qubit::BehaviorJoin join : joins) {
                    const std::uint64_t output = behavior_join(behavior, masks[branch], join);
                    qubit::BehaviorFoldPlan plan = state.plan;
                    plan.branch_indices.push_back(static_cast<std::uint32_t>(branch));
                    plan.joins.push_back(join);
                    const auto found = levels[leaf_count - 1U].find(output);
                    if (found == levels[leaf_count - 1U].end() || branch < found->second.last_index ||
                        (branch == found->second.last_index && behavior_plan_less(plan, found->second.plan))) {
                        levels[leaf_count - 1U][output] = BehaviorControlState{std::move(plan), branch};
                    }
                }
            }
        }
    }
    return levels;
}

std::optional<qubit::BehaviorFoldPlan> behavior_canonical(
    const std::vector<std::map<std::uint64_t, BehaviorControlState>>& levels,
    std::uint64_t behavior) {
    for (const auto& level : levels) {
        const auto found = level.find(behavior);
        if (found != level.end()) return found->second.plan;
    }
    return std::nullopt;
}

}  // namespace

int main() {
    require(qstate_abi_version_major() == 1U, "ABI major must remain 1");
    require(qstate_abi_version_minor() >= 5U, "compiled-plan ABI must be available");
    require(std::string(qstate_version_string()) == "0.2.0", "native version string");

    qstate_handle state = qstate_create(2);
    require(state != nullptr, "qstate_create");
    require(qstate_qubit_count(state) == 2U, "qubit count");
    require(qstate_component_count(state) == 2U, "initial component count");

    const qstate_operation bell_operations[] = {
        {QSTATE_OP_H, 0U, 0U, 0U, 0.0, 0.0},
        {QSTATE_OP_CNOT, 0U, 1U, 0U, 0.0, 0.0},
    };
    size_t completed = 0U;
    require(qstate_apply_operations(state, bell_operations, 2U, &completed) == 0,
            "batched Bell preparation");
    require(completed == 2U, "batch completion count");

    double real = 0.0;
    double imag = 0.0;
    require(qstate_amplitude(state, 0, &real, &imag) == 0, "integer amplitude query");
    require_near(real, std::sqrt(0.5), 1e-12, "Bell |00> amplitude");
    require_near(imag, 0.0, 1e-12, "Bell |00> imaginary amplitude");

    const std::uint8_t bits[] = {1U, 1U};
    require(qstate_amplitude_bits(state, bits, 2U, &real, &imag) == 0, "bit-vector amplitude query");
    require_near(real, std::sqrt(0.5), 1e-12, "Bell |11> amplitude");

    int kind = -1;
    require(qstate_component_kind(state, 0, &kind) == 0, "component kind query");
    require(kind == 1, "Bell component must be sparse");
    require(qstate_validate(state) == 0, "native validation");

    const size_t qsc_size = qstate_qsc_size(state);
    require(qsc_size > 0U, "QSC size");
    std::vector<std::uint8_t> qsc(qsc_size);
    require(qstate_qsc_write(state, qsc.data(), qsc.size()) == 0, "QSC write");
    qstate_handle restored = qstate_qsc_read(qsc.data(), qsc.size());
    require(restored != nullptr, "QSC read");
    require(qstate_component_count(restored) == 1U, "QSC component partition");

    std::uint8_t measurements[2]{};
    require(qstate_measure_all(restored, 7U, measurements, 2U) == 0, "measure all");
    require(measurements[0] == measurements[1], "Bell measurements must correlate");

    qstate_destroy(restored);
    qstate_destroy(state);

    qstate_handle phase = qstate_create(1);
    require(phase != nullptr, "phase register creation");
    require(qstate_apply_x(phase, 0) == 0, "prepare |1>");
    require(qstate_apply_s(phase, 0) == 0, "S gate");
    require(qstate_apply_sdg(phase, 0) == 0, "S dagger gate");
    require(qstate_apply_t(phase, 0) == 0, "T gate");
    require(qstate_apply_tdg(phase, 0) == 0, "T dagger gate");
    require(qstate_apply_amplitude_damping_trajectory(phase, 0, 1.0, 0.0) == 0,
            "amplitude damping trajectory");
    double probability = 1.0;
    require(qstate_probability_one(phase, 0, &probability) == 0, "probability query");
    require_near(probability, 0.0, 1e-12, "full damping must produce |0>");
    qstate_destroy(phase);

    const qstate_operation fused_operations[] = {
        {QSTATE_OP_H, 0U, 0U, 0U, 0.0, 0.0},
        {QSTATE_OP_H, 0U, 0U, 0U, 0.0, 0.0},
        {QSTATE_OP_RY, 0U, 0U, 0U, 0.31, 0.0},
        {QSTATE_OP_RZ, 0U, 0U, 0U, -0.22, 0.0},
    };
    qstate_plan_handle plan = qstate_plan_create(
        fused_operations, 4U, static_cast<uint32_t>(QSTATE_PLAN_OPTIMIZE));
    require(plan != nullptr, "compiled plan creation");
    require(qstate_plan_source_operation_count(plan) == 4U, "compiled plan source count");
    require(qstate_plan_compiled_step_count(plan) == 2U, "single-qubit plan fusion");

    std::vector<qstate_handle> ensemble;
    for (std::size_t index = 0; index < 32U; ++index) {
        ensemble.push_back(qstate_create(1));
        require(ensemble.back() != nullptr, "ensemble register creation");
    }
    size_t completed_registers = 0U;
    require(qstate_plan_execute_many(
                plan,
                ensemble.data(),
                ensemble.size(),
                4U,
                &completed_registers) == 0,
            "parallel plan execution");
    require(completed_registers == ensemble.size(), "parallel plan completion count");
    for (qstate_handle member : ensemble) {
        double all_probabilities[1]{};
        require(qstate_probabilities_one(member, all_probabilities, 1U) == 0,
                "bulk probability query");
        require_near(all_probabilities[0], std::sin(0.31 / 2.0) * std::sin(0.31 / 2.0),
                     1e-12, "compiled plan probability");
        qstate_destroy(member);
    }
    qstate_plan_destroy(plan);

    const qstate_parameterized_operation parameterized_operations[] = {
        {QSTATE_OP_RY, 0U, 0U, 0U, 0.0, 0.0, 0, -1},
        {QSTATE_OP_RZ, 0U, 0U, 0U, 0.0, 0.0, 1, -1},
        {QSTATE_OP_CNOT, 0U, 1U, 0U, 0.0, 0.0, -1, -1},
    };
    qstate_parameterized_plan_handle parameterized = qstate_parameterized_plan_create(
        parameterized_operations, 3U, static_cast<uint32_t>(QSTATE_PLAN_OPTIMIZE));
    require(parameterized != nullptr, "parameterized plan creation");
    require(qstate_parameterized_plan_parameter_count(parameterized) == 2U,
            "parameterized plan slot count");
    qstate_handle parameterized_state = qstate_create(2);
    require(parameterized_state != nullptr, "parameterized state creation");
    const double values[] = {0.31, -0.22};
    completed = 0U;
    require(qstate_parameterized_plan_execute(
                parameterized_state, parameterized, values, 2U, &completed) == 0,
            "parameterized plan execution");
    require(completed == 3U, "parameterized plan completion count");
    require(qstate_probability_one(parameterized_state, 0U, &probability) == 0,
            "parameterized probability query");
    require_near(probability, std::sin(0.31 / 2.0) * std::sin(0.31 / 2.0),
                 1e-12, "parameterized plan probability");
    qstate_destroy(parameterized_state);
    qstate_parameterized_plan_destroy(parameterized);

    const uint64_t grover_marked[] = {5U};
    qstate_grover_handle grover = qstate_grover_create(3U, grover_marked, 1U);
    require(grover != nullptr, "compressed Grover creation");
    require(qstate_grover_space_size(grover) == 8U, "compressed Grover space size");
    require(qstate_grover_optimal_iterations(grover) == 2U, "compressed Grover optimum");
    require(qstate_grover_run_optimal(grover) == 0, "compressed Grover execution");
    require(qstate_grover_success_probability(grover, &probability) == 0,
            "compressed Grover probability");
    require_near(probability, 0.9453125, 1e-12, "compressed Grover amplification");
    uint64_t sampled_basis = 0U;
    require(qstate_grover_sample_basis(grover, 0.0, 0.5, &sampled_basis) == 0,
            "compressed Grover sampling");
    require(sampled_basis == 5U, "compressed Grover marked sample");
    require(qstate_grover_validate(grover) == 0, "compressed Grover validation");
    qstate_grover_destroy(grover);

    qstate_handle exact_grover = qstate_create(3U);
    require(exact_grover != nullptr, "exact Grover register creation");
    for (uint32_t qubit = 0; qubit < 3U; ++qubit) {
        require(qstate_apply_h(exact_grover, qubit) == 0, "exact Grover uniform state");
    }
    require(qstate_apply_grover_iterations(exact_grover, grover_marked, 1U, 2U) == 0,
            "exact Grover iterations");
    require(qstate_amplitude(exact_grover, 5U, &real, &imag) == 0,
            "exact Grover marked amplitude");
    require_near(real * real + imag * imag, 0.9453125, 1e-12,
                 "exact Grover amplification");
    qstate_destroy(exact_grover);

    const uint64_t symmetry_counts[] = {2U, 3U, 3U};
    qstate_symmetry_handle symmetry = qstate_symmetry_create_ordered(
        3U, symmetry_counts, 3U);
    require(symmetry != nullptr, "symmetry state creation");
    require(qstate_symmetry_class_count(symmetry) == 3U, "symmetry class count");
    require(qstate_symmetry_membership_mode(symmetry) == 1, "symmetry membership mode");
    require(qstate_symmetry_class_probability(symmetry, 0U, &probability) == 0,
            "symmetry probability query");
    require_near(probability, 0.25, 1e-12, "symmetry uniform probability");
    require(qstate_symmetry_apply_class_phase(symmetry, 0U, std::acos(-1.0)) == 0,
            "symmetry class phase");
    require(qstate_symmetry_apply_reflection(symmetry) == 0,
            "symmetry weighted reflection");
    require(qstate_symmetry_validate(symmetry) == 0, "symmetry validation");
    qstate_handle fallback = qstate_symmetry_to_register(symmetry, 3U);
    require(fallback != nullptr, "symmetry QRegister fallback");
    for (uint64_t basis = 0; basis < 8U; ++basis) {
        double symmetry_real = 0.0;
        double symmetry_imag = 0.0;
        require(qstate_symmetry_amplitude(
                    symmetry, basis, &symmetry_real, &symmetry_imag) == 0,
                "symmetry amplitude query");
        require(qstate_amplitude(fallback, basis, &real, &imag) == 0,
                "symmetry fallback amplitude query");
        require_near(real, symmetry_real, 1e-12, "symmetry fallback real amplitude");
        require_near(imag, symmetry_imag, 1e-12, "symmetry fallback imaginary amplitude");
    }
    qstate_destroy(fallback);
    qstate_symmetry_destroy(symmetry);

    qstate_handle uniform_source = qstate_create(6U);
    require(uniform_source != nullptr, "symmetry discovery source creation");
    for (uint32_t qubit = 0; qubit < 6U; ++qubit) {
        require(qstate_apply_h(uniform_source, qubit) == 0,
                "symmetry discovery uniform preparation");
    }
    qstate_symmetry_handle discovered = qstate_symmetry_discover(
        uniform_source, 6U, 1e-12, 16U);
    require(discovered != nullptr, "symmetry discovery");
    require(qstate_symmetry_class_count(discovered) == 1U,
            "symmetry discovery class count");
    double discovery_error = 1.0;
    require(qstate_symmetry_discovery_error(discovered, &discovery_error) == 0,
            "symmetry discovery error query");
    require(discovery_error < 1e-12, "symmetry discovery bounded error");
    qstate_symmetry_destroy(discovered);
    qstate_destroy(uniform_source);

    qstate_symmetry_handle hamming = qstate_symmetry_create_hamming_weight(60U);
    require(hamming != nullptr, "Hamming-weight symmetry creation");
    require(qstate_symmetry_class_count(hamming) == 61U,
            "Hamming-weight symmetry class count");
    require(qstate_symmetry_membership_mode(hamming) == 3,
            "Hamming-weight symmetry membership mode");
    require(qstate_symmetry_class_size(hamming, 30U) == 118264581564861424ULL,
            "Hamming-weight central binomial class size");
    qstate_symmetry_destroy(hamming);

    const std::vector<std::uint64_t> behavior_masks{
        0b0011U,
        0b0101U,
        0b1110U,
        0b1001U,
        0b0110U,
        0b1100U,
    };
    const auto control = behavior_control(behavior_masks, 4U);
    const std::array<qubit::BehaviorJoin, 3> behavior_joins{
        qubit::BehaviorJoin::And,
        qubit::BehaviorJoin::Or,
        qubit::BehaviorJoin::Xor,
    };
    qubit::ExactBehaviorFoldUniverse behavior_universe(behavior_masks, 4U, 4U, behavior_joins);
    const auto& behavior_stats = behavior_universe.stats();
    require(behavior_stats.depth_state_counts == std::vector<std::size_t>({
                control[0].size(), control[1].size(), control[2].size(), control[3].size()}),
            "qbehavior optimized depth counts must match exhaustive control");
    require(behavior_stats.structural_count_by_leaf ==
                std::vector<std::uint64_t>({6U, 45U, 180U, 405U}),
            "qbehavior structural counts");

    std::size_t expected_behavior_count = 0U;
    std::size_t expected_deep_count = 0U;
    std::optional<std::uint64_t> behavior_target;
    std::optional<qubit::BehaviorFoldPlan> behavior_target_plan;
    for (std::uint64_t behavior = 0U; behavior < 16U; ++behavior) {
        const auto expected = behavior_canonical(control, behavior);
        const auto actual = behavior_universe.canonical_plan(behavior);
        require(actual == expected, "qbehavior canonical plan must match exhaustive control");
        if (expected.has_value()) {
            ++expected_behavior_count;
            if (!behavior_target.has_value()) {
                behavior_target = behavior;
                behavior_target_plan = expected;
            }
        }
        if (control[3].contains(behavior) && !control[0].contains(behavior) &&
            !control[1].contains(behavior) && !control[2].contains(behavior)) {
            ++expected_deep_count;
        }
        for (std::size_t depth = 1U; depth <= 4U; ++depth) {
            require(
                behavior_universe.contains_at_depth(depth, behavior) == control[depth - 1U].contains(behavior),
                "qbehavior depth membership must match exhaustive control");
        }
    }
    require(behavior_stats.behavior_class_count == expected_behavior_count &&
                behavior_stats.deep_behavior_count == expected_deep_count,
            "qbehavior canonical population metrics");
    require(behavior_target.has_value() && behavior_target_plan.has_value(),
            "qbehavior exhaustive control must contain a target");

    const std::array<std::size_t, 4> behavior_positions{0U, 1U, 2U, 3U};
    for (std::uint64_t behavior = 0U; behavior < 16U; ++behavior) {
        const auto expected = behavior_canonical(control, behavior);
        if (!expected.has_value()) continue;
        const std::array<std::uint8_t, 4> observed{
            static_cast<std::uint8_t>((behavior >> 0U) & 1U),
            static_cast<std::uint8_t>((behavior >> 1U) & 1U),
            static_cast<std::uint8_t>((behavior >> 2U) & 1U),
            static_cast<std::uint8_t>((behavior >> 3U) & 1U),
        };
        const qubit::BehaviorFoldUniqueMatch match =
            behavior_universe.conditioned_unique(behavior_positions, observed);
        require(match.count == 1U && match.behavior == behavior && match.plan == expected,
                "qbehavior full observation must recover canonical behavior");
    }

    qubit::detail::BehaviorSha256 behavior_sha;
    const std::string abc = "abc";
    behavior_sha.update(abc.data(), abc.size());
    require(
        behavior_sha.finish() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "qbehavior portable SHA-256");
    for (std::size_t depth = 1U; depth <= 4U; ++depth) {
        const std::string digest = behavior_universe.state_digest(depth);
        require(digest.size() == 64U && digest == behavior_universe.state_digest(depth),
                "qbehavior depth digest determinism");
    }

    require(qbehavior_api_version_major() == 0U && qbehavior_api_version_minor() == 1U,
            "qbehavior experimental API version");
    const std::array<std::uint8_t, 3> behavior_join_codes{
        QBEHAVIOR_JOIN_AND,
        QBEHAVIOR_JOIN_OR,
        QBEHAVIOR_JOIN_XOR,
    };
    qbehavior_fold_handle behavior_handle = qbehavior_fold_create(
        behavior_masks.data(),
        behavior_masks.size(),
        4U,
        4U,
        behavior_join_codes.data(),
        behavior_join_codes.size(),
        nullptr);
    require(behavior_handle != nullptr, "qbehavior C bridge creation");
    qbehavior_fold_stats behavior_c_stats{};
    require(qbehavior_fold_stats_read(behavior_handle, &behavior_c_stats) == 0 &&
                behavior_c_stats.behavior_class_count == behavior_stats.behavior_class_count &&
                behavior_c_stats.deep_behavior_count == behavior_stats.deep_behavior_count,
            "qbehavior C bridge stats");

    std::array<std::uint32_t, 4> behavior_branches{};
    std::array<std::uint8_t, 3> behavior_plan_joins{};
    std::size_t behavior_leaf_count = 0U;
    int behavior_found = 0;
    require(qbehavior_fold_canonical_plan(
                behavior_handle,
                *behavior_target,
                behavior_branches.data(),
                behavior_branches.size(),
                behavior_plan_joins.data(),
                behavior_plan_joins.size(),
                &behavior_leaf_count,
                &behavior_found) == 0 &&
                behavior_found == 1 &&
                behavior_leaf_count == behavior_target_plan->leaf_count(),
            "qbehavior C bridge canonical plan");
    for (std::size_t i = 0U; i < behavior_leaf_count; ++i) {
        require(behavior_branches[i] == behavior_target_plan->branch_indices[i],
                "qbehavior C bridge branch index");
    }
    for (std::size_t i = 0U; i + 1U < behavior_leaf_count; ++i) {
        require(behavior_plan_joins[i] == static_cast<std::uint8_t>(behavior_target_plan->joins[i]),
                "qbehavior C bridge join code");
    }

    std::array<char, 65> behavior_digest{};
    require(qbehavior_fold_canonical_digest(
                behavior_handle, behavior_digest.data(), behavior_digest.size()) == 0 &&
                std::string(behavior_digest.data()) == behavior_universe.canonical_digest(),
            "qbehavior C bridge canonical digest");
    const std::array<std::size_t, 2> repeated_positions{0U, 0U};
    const std::array<std::uint8_t, 2> repeated_bits{1U, 1U};
    std::size_t repeated_count = 0U;
    require(qbehavior_fold_conditioned_count(
                behavior_handle,
                repeated_positions.data(),
                repeated_bits.data(),
                repeated_positions.size(),
                &repeated_count) != 0,
            "qbehavior repeated observation must fail closed");
    qbehavior_fold_destroy(behavior_handle);

    std::cout << "QSA C ABI compatibility tests passed.\n";
    return 0;
}

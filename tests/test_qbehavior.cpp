#include "qubit/qbehavior.hpp"
#include "qubit/qbehavior_c_api.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace qubit;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct ControlState {
    BehaviorFoldPlan plan{};
    std::size_t last_index{0U};
};

std::uint64_t apply_join(std::uint64_t left, std::uint64_t right, BehaviorJoin join) {
    if (join == BehaviorJoin::And) return left & right;
    if (join == BehaviorJoin::Or) return left | right;
    return left ^ right;
}

bool less_plan(const BehaviorFoldPlan& left, const BehaviorFoldPlan& right) {
    if (left.leaf_count() != right.leaf_count()) return left.leaf_count() < right.leaf_count();
    if (left.branch_indices != right.branch_indices) return left.branch_indices < right.branch_indices;
    return left.joins < right.joins;
}

std::vector<std::map<std::uint64_t, ControlState>> build_control(
    const std::vector<std::uint64_t>& masks,
    std::size_t max_leaves) {
    const std::array<BehaviorJoin, 3> joins{BehaviorJoin::And, BehaviorJoin::Or, BehaviorJoin::Xor};
    std::vector<std::map<std::uint64_t, ControlState>> levels(max_leaves);
    for (std::size_t index = 0U; index < masks.size(); ++index) {
        BehaviorFoldPlan plan{{static_cast<std::uint32_t>(index)}, {}};
        const auto found = levels[0].find(masks[index]);
        if (found == levels[0].end() || index < found->second.last_index ||
            (index == found->second.last_index && less_plan(plan, found->second.plan))) {
            levels[0][masks[index]] = ControlState{std::move(plan), index};
        }
    }
    for (std::size_t leaf_count = 2U; leaf_count <= max_leaves; ++leaf_count) {
        for (const auto& [behavior, state] : levels[leaf_count - 2U]) {
            for (std::size_t branch = state.last_index + 1U; branch < masks.size(); ++branch) {
                for (const BehaviorJoin join : joins) {
                    const std::uint64_t output = apply_join(behavior, masks[branch], join);
                    BehaviorFoldPlan plan = state.plan;
                    plan.branch_indices.push_back(static_cast<std::uint32_t>(branch));
                    plan.joins.push_back(join);
                    const auto found = levels[leaf_count - 1U].find(output);
                    if (found == levels[leaf_count - 1U].end() || branch < found->second.last_index ||
                        (branch == found->second.last_index && less_plan(plan, found->second.plan))) {
                        levels[leaf_count - 1U][output] = ControlState{std::move(plan), branch};
                    }
                }
            }
        }
    }
    return levels;
}

std::optional<BehaviorFoldPlan> control_canonical(
    const std::vector<std::map<std::uint64_t, ControlState>>& levels,
    std::uint64_t behavior) {
    for (const auto& level : levels) {
        const auto found = level.find(behavior);
        if (found != level.end()) return found->second.plan;
    }
    return std::nullopt;
}

}  // namespace

int main() {
    const std::vector<std::uint64_t> masks{
        0b0011U,
        0b0101U,
        0b1110U,
        0b1001U,
        0b0110U,
        0b1100U,
    };
    const std::array<BehaviorJoin, 3> joins{BehaviorJoin::And, BehaviorJoin::Or, BehaviorJoin::Xor};
    const auto control = build_control(masks, 4U);
    ExactBehaviorFoldUniverse universe(masks, 4U, 4U, joins);

    const auto& stats = universe.stats();
    require(stats.branch_count == masks.size() && stats.probe_bit_count == 4U && stats.max_leaves == 4U,
            "behavior-fold metadata changed");
    require(stats.depth_state_counts == std::vector<std::size_t>({
                control[0].size(), control[1].size(), control[2].size(), control[3].size()}),
            "optimized behavior-fold state counts differ from exhaustive control");
    require(stats.structural_count_by_leaf == std::vector<std::uint64_t>({6U, 45U, 180U, 405U}),
            "behavior-fold structural count is wrong");

    std::size_t canonical_count = 0U;
    std::size_t deep_count = 0U;
    for (std::uint64_t behavior = 0U; behavior < 16U; ++behavior) {
        const auto expected = control_canonical(control, behavior);
        const auto actual = universe.canonical_plan(behavior);
        require(actual == expected, "optimized behavior-fold canonical plan differs from exhaustive control");
        if (expected.has_value()) ++canonical_count;
        if (control[3].contains(behavior) && !control[0].contains(behavior) &&
            !control[1].contains(behavior) && !control[2].contains(behavior)) {
            ++deep_count;
        }
        for (std::size_t depth = 1U; depth <= 4U; ++depth) {
            require(universe.contains_at_depth(depth, behavior) == control[depth - 1U].contains(behavior),
                    "optimized behavior-fold depth membership differs from exhaustive control");
        }
    }
    require(stats.behavior_class_count == canonical_count && stats.deep_behavior_count == deep_count,
            "behavior-fold canonical or deep population changed");

    const std::array<std::size_t, 4> all_positions{0U, 1U, 2U, 3U};
    for (std::uint64_t behavior = 0U; behavior < 16U; ++behavior) {
        const auto expected = control_canonical(control, behavior);
        if (!expected.has_value()) continue;
        const std::array<std::uint8_t, 4> bits{
            static_cast<std::uint8_t>((behavior >> 0U) & 1U),
            static_cast<std::uint8_t>((behavior >> 1U) & 1U),
            static_cast<std::uint8_t>((behavior >> 2U) & 1U),
            static_cast<std::uint8_t>((behavior >> 3U) & 1U),
        };
        const BehaviorFoldUniqueMatch match = universe.conditioned_unique(all_positions, bits);
        require(match.count == 1U && match.behavior == behavior && match.plan == expected,
                "behavior-fold full observation did not recover the exact canonical behavior");
    }

    const std::array<std::size_t, 1> one_position{0U};
    const std::array<std::uint8_t, 1> one_bit{1U};
    std::size_t expected_count = 0U;
    for (std::uint64_t behavior = 0U; behavior < 16U; ++behavior) {
        if (control_canonical(control, behavior).has_value() && (behavior & 1U) != 0U) ++expected_count;
    }
    require(universe.conditioned_count(one_position, one_bit) == expected_count,
            "behavior-fold conditioned count differs from exhaustive control");

    detail::BehaviorSha256 sha;
    const std::string abc = "abc";
    sha.update(abc.data(), abc.size());
    require(sha.finish() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "portable behavior-fold SHA-256 is wrong");
    for (std::size_t depth = 1U; depth <= 4U; ++depth) {
        const std::string digest = universe.state_digest(depth);
        require(digest.size() == 64U && digest == universe.state_digest(depth),
                "behavior-fold state digest is not deterministic");
    }
    require(universe.canonical_digest().size() == 64U &&
            universe.canonical_digest() == universe.canonical_digest(),
            "behavior-fold canonical digest is not deterministic");

    const std::array<std::uint8_t, 3> join_codes{
        QBEHAVIOR_JOIN_AND,
        QBEHAVIOR_JOIN_OR,
        QBEHAVIOR_JOIN_XOR,
    };
    qbehavior_fold_handle handle = qbehavior_fold_create(
        masks.data(), masks.size(), 4U, 4U, join_codes.data(), join_codes.size(), nullptr);
    require(handle != nullptr, "qbehavior C bridge did not construct");
    qbehavior_fold_stats c_stats{};
    require(qbehavior_fold_stats_read(handle, &c_stats) == 0 &&
            c_stats.behavior_class_count == stats.behavior_class_count &&
            c_stats.deep_behavior_count == stats.deep_behavior_count,
            "qbehavior C bridge stats differ from C++ universe");

    const std::uint64_t target = *universe.conditioned_unique(all_positions, std::array<std::uint8_t, 4>{1U, 0U, 0U, 0U}).behavior;
    std::array<std::uint32_t, 4> branches{};
    std::array<std::uint8_t, 3> plan_joins{};
    std::size_t leaf_count = 0U;
    int found = 0;
    require(qbehavior_fold_canonical_plan(
                handle, target, branches.data(), branches.size(), plan_joins.data(), plan_joins.size(),
                &leaf_count, &found) == 0 && found == 1,
            "qbehavior C bridge did not return canonical plan");
    const auto target_plan = universe.canonical_plan(target);
    require(target_plan.has_value() && leaf_count == target_plan->leaf_count(),
            "qbehavior C bridge plan width changed");
    for (std::size_t i = 0U; i < leaf_count; ++i) require(branches[i] == target_plan->branch_indices[i], "qbehavior branch index changed");
    for (std::size_t i = 0U; i + 1U < leaf_count; ++i) require(plan_joins[i] == static_cast<std::uint8_t>(target_plan->joins[i]), "qbehavior join code changed");

    std::array<char, 65> digest{};
    require(qbehavior_fold_canonical_digest(handle, digest.data(), digest.size()) == 0 &&
            std::string(digest.data()) == universe.canonical_digest(),
            "qbehavior C bridge canonical digest differs from C++ universe");

    const std::array<std::size_t, 2> invalid_positions{0U, 0U};
    const std::array<std::uint8_t, 2> invalid_bits{1U, 1U};
    std::size_t invalid_count = 0U;
    require(qbehavior_fold_conditioned_count(
                handle, invalid_positions.data(), invalid_bits.data(), invalid_positions.size(), &invalid_count) != 0,
            "qbehavior C bridge accepted repeated observation positions");
    qbehavior_fold_destroy(handle);

    bool rejected = false;
    try {
        (void)ExactBehaviorFoldUniverse(
            masks,
            4U,
            4U,
            joins,
            BehaviorFoldConfig{2U, 1024U, 1000U});
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "behavior-fold state cap did not fail closed");

    std::cout << "QBehavior tests passed\n";
}

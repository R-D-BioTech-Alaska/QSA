#define QSTATE_BUILD_SHARED
#include "qubit/detail/qbehavior_c_api_impl.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct ControlState {
    qubit::BehaviorFoldPlan plan{};
    std::size_t last_index{0U};
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::uint64_t join(
    std::uint64_t left,
    std::uint64_t right,
    qubit::BehaviorJoin operation) {
    if (operation == qubit::BehaviorJoin::And) return left & right;
    if (operation == qubit::BehaviorJoin::Or) return left | right;
    return left ^ right;
}

bool plan_less(
    const qubit::BehaviorFoldPlan& left,
    const qubit::BehaviorFoldPlan& right) {
    if (left.leaf_count() != right.leaf_count()) return left.leaf_count() < right.leaf_count();
    if (left.branch_indices != right.branch_indices) return left.branch_indices < right.branch_indices;
    return left.joins < right.joins;
}

std::vector<std::map<std::uint64_t, ControlState>> control(
    const std::vector<std::uint64_t>& masks,
    std::size_t max_leaves) {
    const std::array<qubit::BehaviorJoin, 3> joins{
        qubit::BehaviorJoin::And,
        qubit::BehaviorJoin::Or,
        qubit::BehaviorJoin::Xor,
    };
    std::vector<std::map<std::uint64_t, ControlState>> levels(max_leaves);
    for (std::size_t index = 0U; index < masks.size(); ++index) {
        qubit::BehaviorFoldPlan plan{{static_cast<std::uint32_t>(index)}, {}};
        const auto found = levels[0].find(masks[index]);
        if (found == levels[0].end() || index < found->second.last_index ||
            (index == found->second.last_index && plan_less(plan, found->second.plan))) {
            levels[0][masks[index]] = ControlState{std::move(plan), index};
        }
    }
    for (std::size_t leaf_count = 2U; leaf_count <= max_leaves; ++leaf_count) {
        for (const auto& [behavior, state] : levels[leaf_count - 2U]) {
            for (std::size_t branch = state.last_index + 1U; branch < masks.size(); ++branch) {
                for (const auto operation : joins) {
                    const std::uint64_t output = join(behavior, masks[branch], operation);
                    qubit::BehaviorFoldPlan plan = state.plan;
                    plan.branch_indices.push_back(static_cast<std::uint32_t>(branch));
                    plan.joins.push_back(operation);
                    const auto found = levels[leaf_count - 1U].find(output);
                    if (found == levels[leaf_count - 1U].end() || branch < found->second.last_index ||
                        (branch == found->second.last_index && plan_less(plan, found->second.plan))) {
                        levels[leaf_count - 1U][output] = ControlState{std::move(plan), branch};
                    }
                }
            }
        }
    }
    return levels;
}

std::optional<qubit::BehaviorFoldPlan> canonical(
    const std::vector<std::map<std::uint64_t, ControlState>>& levels,
    std::uint64_t behavior) {
    for (const auto& level : levels) {
        const auto found = level.find(behavior);
        if (found != level.end()) return found->second.plan;
    }
    return std::nullopt;
}

void expect_rejected(const qubit::BehaviorFoldConfig& config, const char* message) {
    const std::array<std::uint64_t, 6> masks{3U, 5U, 14U, 9U, 6U, 12U};
    bool rejected = false;
    try {
        qubit::ExactBehaviorFoldUniverse universe(masks, 4U, 4U, {}, config);
        (void)universe;
    } catch (const qubit::QStateError&) {
        rejected = true;
    }
    require(rejected, message);
}

}  // namespace

int main() {
    const std::vector<std::uint64_t> masks{3U, 5U, 14U, 9U, 6U, 12U};
    const auto expected = control(masks, 4U);
    const std::array<qubit::BehaviorJoin, 3> joins{
        qubit::BehaviorJoin::And,
        qubit::BehaviorJoin::Or,
        qubit::BehaviorJoin::Xor,
    };
    qubit::ExactBehaviorFoldUniverse universe(masks, 4U, 4U, joins);
    const auto& stats = universe.stats();
    require(stats.depth_state_counts == std::vector<std::size_t>({
                expected[0].size(), expected[1].size(), expected[2].size(), expected[3].size()}),
            "behavior-fold depth counts differ from exhaustive control");
    require(stats.structural_count_by_leaf ==
                std::vector<std::uint64_t>({6U, 45U, 180U, 405U}),
            "behavior-fold structural counts are wrong");

    std::size_t class_count = 0U;
    std::size_t deep_count = 0U;
    for (std::uint64_t behavior = 0U; behavior < 16U; ++behavior) {
        const auto plan = canonical(expected, behavior);
        require(universe.canonical_plan(behavior) == plan,
                "behavior-fold canonical plan differs from exhaustive control");
        if (plan.has_value()) ++class_count;
        if (expected[3].contains(behavior) && !expected[0].contains(behavior) &&
            !expected[1].contains(behavior) && !expected[2].contains(behavior)) {
            ++deep_count;
        }
        for (std::size_t depth = 1U; depth <= 4U; ++depth) {
            require(universe.contains_at_depth(depth, behavior) == expected[depth - 1U].contains(behavior),
                    "behavior-fold depth membership differs from exhaustive control");
        }
    }
    require(stats.behavior_class_count == class_count && stats.deep_behavior_count == deep_count,
            "behavior-fold population accounting is wrong");

    const std::array<std::size_t, 4> positions{0U, 1U, 2U, 3U};
    for (std::uint64_t behavior = 0U; behavior < 16U; ++behavior) {
        const auto plan = canonical(expected, behavior);
        if (!plan.has_value()) continue;
        const std::array<std::uint8_t, 4> bits{
            static_cast<std::uint8_t>((behavior >> 0U) & 1U),
            static_cast<std::uint8_t>((behavior >> 1U) & 1U),
            static_cast<std::uint8_t>((behavior >> 2U) & 1U),
            static_cast<std::uint8_t>((behavior >> 3U) & 1U),
        };
        const auto match = universe.conditioned_unique(positions, bits);
        require(match.count == 1U && match.behavior == behavior && match.plan == plan,
                "full behavior observation did not recover the canonical plan");
    }

    qubit::detail::BehaviorSha256 sha;
    const std::string abc = "abc";
    sha.update(abc.data(), abc.size());
    require(sha.finish() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "behavior-fold SHA-256 vector is wrong");
    for (std::size_t depth = 1U; depth <= 4U; ++depth) {
        const std::string digest = universe.state_digest(depth);
        require(digest.size() == 64U && digest == universe.state_digest(depth),
                "behavior-fold state digest is not deterministic");
    }

    const std::array<std::uint8_t, 3> join_codes{
        QBEHAVIOR_JOIN_AND,
        QBEHAVIOR_JOIN_OR,
        QBEHAVIOR_JOIN_XOR,
    };
    qbehavior_fold_handle handle = qbehavior_fold_create(
        masks.data(), masks.size(), 4U, 4U, join_codes.data(), join_codes.size(), nullptr);
    require(handle != nullptr, "behavior-fold C bridge creation failed");
    qbehavior_fold_stats c_stats{};
    require(qbehavior_fold_stats_read(handle, &c_stats) == 0,
            "behavior-fold C bridge stats failed");
    require(c_stats.behavior_class_count == stats.behavior_class_count &&
            c_stats.deep_behavior_count == stats.deep_behavior_count,
            "behavior-fold C bridge stats differ from C++ runtime");

    std::array<char, 65> digest{};
    require(qbehavior_fold_canonical_digest(handle, digest.data(), digest.size()) == 0 &&
            std::string(digest.data()) == universe.canonical_digest(),
            "behavior-fold C bridge digest differs from C++ runtime");
    const std::array<std::size_t, 2> repeated_positions{0U, 0U};
    const std::array<std::uint8_t, 2> repeated_bits{1U, 1U};
    std::size_t count = 0U;
    require(qbehavior_fold_conditioned_count(
                handle, repeated_positions.data(), repeated_bits.data(),
                repeated_positions.size(), &count) != 0,
            "repeated behavior observation did not fail closed");
    qbehavior_fold_destroy(handle);

    qubit::BehaviorFoldConfig state_cap;
    state_cap.max_states_per_level = 1U;
    expect_rejected(state_cap, "behavior-fold state cap was not enforced");

    qubit::BehaviorFoldConfig index_cap;
    index_cap.max_index_bytes = 1U;
    expect_rejected(index_cap, "behavior-fold index cap was not enforced");

    qubit::BehaviorFoldConfig lookup_cap;
    lookup_cap.max_destination_lookups = 1U;
    expect_rejected(lookup_cap, "behavior-fold lookup cap was not enforced");

    std::cout << "behavior-fold tests passed\n";
    return 0;
}

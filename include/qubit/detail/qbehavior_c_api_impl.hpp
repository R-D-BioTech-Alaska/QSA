#pragma once

#include "qubit/qbehavior_c_api.h"

#include "qubit/qbehavior.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

thread_local std::string behavior_last_error;

struct BehaviorFoldHandle {
    BehaviorFoldHandle(
        std::span<const std::uint64_t> branch_masks,
        std::size_t probe_bit_count,
        std::size_t max_leaves,
        std::span<const qubit::BehaviorJoin> joins,
        qubit::BehaviorFoldConfig config)
        : universe(branch_masks, probe_bit_count, max_leaves, joins, config) {}

    qubit::ExactBehaviorFoldUniverse universe;
};

BehaviorFoldHandle* as_behavior(qbehavior_fold_handle handle) {
    if (handle == nullptr) throw qubit::QStateError("QSA behavior-fold handle is null");
    return static_cast<BehaviorFoldHandle*>(handle);
}

template <class Function>
int guarded_behavior(Function&& function) {
    try {
        function();
        behavior_last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        behavior_last_error = error.what();
        return -1;
    } catch (...) {
        behavior_last_error = "Unknown QSA behavior-fold runtime error";
        return -1;
    }
}

std::vector<qubit::BehaviorJoin> decode_behavior_joins(
    const std::uint8_t* join_codes,
    std::size_t join_count) {
    if (join_count == 0U) {
        const auto defaults = qubit::ExactBehaviorFoldUniverse::default_joins();
        return std::vector<qubit::BehaviorJoin>(defaults.begin(), defaults.end());
    }
    if (join_codes == nullptr) throw qubit::QStateError("Behavior join-code buffer is null");
    std::vector<qubit::BehaviorJoin> joins;
    joins.reserve(join_count);
    for (std::size_t i = 0U; i < join_count; ++i) {
        if (join_codes[i] > QBEHAVIOR_JOIN_XOR) throw qubit::QStateError("Behavior join code is invalid");
        joins.push_back(static_cast<qubit::BehaviorJoin>(join_codes[i]));
    }
    return joins;
}

qubit::BehaviorFoldConfig decode_behavior_config(const qbehavior_fold_config* config) {
    if (config == nullptr) return {};
    return qubit::BehaviorFoldConfig{
        config->max_states_per_level,
        config->max_index_bytes,
        config->max_destination_lookups,
    };
}

void write_behavior_plan(
    const qubit::BehaviorFoldPlan& plan,
    std::uint32_t* branch_indices,
    std::size_t branch_capacity,
    std::uint8_t* join_codes,
    std::size_t join_capacity,
    std::size_t* leaf_count) {
    if (leaf_count == nullptr) throw qubit::QStateError("Behavior plan leaf-count output is null");
    *leaf_count = plan.leaf_count();
    if (branch_capacity < plan.branch_indices.size() ||
        (plan.branch_indices.size() != 0U && branch_indices == nullptr)) {
        throw qubit::QStateError("Behavior plan branch buffer is too small");
    }
    if (join_capacity < plan.joins.size() || (plan.joins.size() != 0U && join_codes == nullptr)) {
        throw qubit::QStateError("Behavior plan join buffer is too small");
    }
    std::copy(plan.branch_indices.begin(), plan.branch_indices.end(), branch_indices);
    for (std::size_t i = 0U; i < plan.joins.size(); ++i) {
        join_codes[i] = static_cast<std::uint8_t>(plan.joins[i]);
    }
}

void write_behavior_digest(const std::string& digest, char* output, std::size_t output_size) {
    if (output == nullptr || output_size < digest.size() + 1U) {
        throw qubit::QStateError("Behavior digest output buffer is too small");
    }
    std::memcpy(output, digest.data(), digest.size());
    output[digest.size()] = '\0';
}

}  // namespace

extern "C" {

uint32_t qbehavior_api_version_major(void) {
    return 0U;
}

uint32_t qbehavior_api_version_minor(void) {
    return 1U;
}

uint32_t qbehavior_api_version_patch(void) {
    return 0U;
}

const char* qbehavior_last_error(void) {
    return behavior_last_error.c_str();
}

qbehavior_fold_handle qbehavior_fold_create(
    const uint64_t* branch_masks,
    size_t branch_count,
    size_t probe_bit_count,
    size_t max_leaves,
    const uint8_t* join_codes,
    size_t join_count,
    const qbehavior_fold_config* config) {
    try {
        if (branch_masks == nullptr || branch_count == 0U) {
            throw qubit::QStateError("Behavior branch-mask buffer is null or empty");
        }
        const std::vector<qubit::BehaviorJoin> joins = decode_behavior_joins(join_codes, join_count);
        auto* handle = new BehaviorFoldHandle(
            std::span<const std::uint64_t>(branch_masks, branch_count),
            probe_bit_count,
            max_leaves,
            joins,
            decode_behavior_config(config));
        behavior_last_error.clear();
        return handle;
    } catch (const std::exception& error) {
        behavior_last_error = error.what();
        return nullptr;
    } catch (...) {
        behavior_last_error = "Unknown QSA behavior-fold runtime error";
        return nullptr;
    }
}

void qbehavior_fold_destroy(qbehavior_fold_handle handle) {
    delete static_cast<BehaviorFoldHandle*>(handle);
}

int qbehavior_fold_stats_read(qbehavior_fold_handle handle, qbehavior_fold_stats* output) {
    return guarded_behavior([&] {
        if (output == nullptr) throw qubit::QStateError("Behavior stats output is null");
        const qubit::BehaviorFoldStats& stats = as_behavior(handle)->universe.stats();
        *output = qbehavior_fold_stats{
            stats.branch_count,
            stats.probe_bit_count,
            stats.max_leaves,
            stats.behavior_class_count,
            stats.deep_behavior_count,
            stats.estimated_bytes,
            stats.eligible_pairs,
            stats.destination_lookups,
            stats.fold_seconds,
            stats.index_seconds,
            stats.build_seconds,
        };
    });
}

int qbehavior_fold_depth_state_count(
    qbehavior_fold_handle handle,
    size_t leaf_count,
    size_t* output) {
    return guarded_behavior([&] {
        if (output == nullptr) throw qubit::QStateError("Behavior depth-count output is null");
        const auto& stats = as_behavior(handle)->universe.stats();
        if (leaf_count == 0U || leaf_count > stats.depth_state_counts.size()) {
            throw qubit::QStateError("Behavior leaf count is outside the constructed universe");
        }
        *output = stats.depth_state_counts[leaf_count - 1U];
    });
}

int qbehavior_fold_structural_count(
    qbehavior_fold_handle handle,
    size_t leaf_count,
    uint64_t* output) {
    return guarded_behavior([&] {
        if (output == nullptr) throw qubit::QStateError("Behavior structural-count output is null");
        const auto& stats = as_behavior(handle)->universe.stats();
        if (leaf_count == 0U || leaf_count > stats.structural_count_by_leaf.size()) {
            throw qubit::QStateError("Behavior leaf count is outside the constructed universe");
        }
        *output = stats.structural_count_by_leaf[leaf_count - 1U];
    });
}

int qbehavior_fold_contains_at_depth(
    qbehavior_fold_handle handle,
    size_t leaf_count,
    uint64_t behavior,
    int* output) {
    return guarded_behavior([&] {
        if (output == nullptr) throw qubit::QStateError("Behavior membership output is null");
        *output = as_behavior(handle)->universe.contains_at_depth(leaf_count, behavior) ? 1 : 0;
    });
}

int qbehavior_fold_canonical_plan(
    qbehavior_fold_handle handle,
    uint64_t behavior,
    uint32_t* branch_indices,
    size_t branch_capacity,
    uint8_t* join_codes,
    size_t join_capacity,
    size_t* leaf_count,
    int* found) {
    return guarded_behavior([&] {
        if (found == nullptr || leaf_count == nullptr) {
            throw qubit::QStateError("Behavior plan status output is null");
        }
        const auto plan = as_behavior(handle)->universe.canonical_plan(behavior);
        if (!plan.has_value()) {
            *found = 0;
            *leaf_count = 0U;
            return;
        }
        *found = 1;
        write_behavior_plan(*plan, branch_indices, branch_capacity, join_codes, join_capacity, leaf_count);
    });
}

int qbehavior_fold_conditioned_count(
    qbehavior_fold_handle handle,
    const size_t* positions,
    const uint8_t* bits,
    size_t observation_count,
    size_t* output) {
    return guarded_behavior([&] {
        if (output == nullptr || (observation_count != 0U && (positions == nullptr || bits == nullptr))) {
            throw qubit::QStateError("Behavior conditioned-count buffers are invalid");
        }
        *output = as_behavior(handle)->universe.conditioned_count(
            std::span<const std::size_t>(positions, observation_count),
            std::span<const std::uint8_t>(bits, observation_count));
    });
}

int qbehavior_fold_conditioned_unique(
    qbehavior_fold_handle handle,
    const size_t* positions,
    const uint8_t* bits,
    size_t observation_count,
    size_t* count,
    int* unique,
    uint64_t* behavior,
    uint32_t* branch_indices,
    size_t branch_capacity,
    uint8_t* join_codes,
    size_t join_capacity,
    size_t* leaf_count) {
    return guarded_behavior([&] {
        if (count == nullptr || unique == nullptr || behavior == nullptr || leaf_count == nullptr ||
            (observation_count != 0U && (positions == nullptr || bits == nullptr))) {
            throw qubit::QStateError("Behavior conditioned-unique buffers are invalid");
        }
        const qubit::BehaviorFoldUniqueMatch result = as_behavior(handle)->universe.conditioned_unique(
            std::span<const std::size_t>(positions, observation_count),
            std::span<const std::uint8_t>(bits, observation_count));
        *count = result.count;
        if (!result.behavior.has_value() || !result.plan.has_value()) {
            *unique = 0;
            *behavior = 0U;
            *leaf_count = 0U;
            return;
        }
        *unique = 1;
        *behavior = *result.behavior;
        write_behavior_plan(
            *result.plan,
            branch_indices,
            branch_capacity,
            join_codes,
            join_capacity,
            leaf_count);
    });
}

int qbehavior_fold_state_digest(
    qbehavior_fold_handle handle,
    size_t leaf_count,
    char* output,
    size_t output_size) {
    return guarded_behavior([&] {
        write_behavior_digest(as_behavior(handle)->universe.state_digest(leaf_count), output, output_size);
    });
}

int qbehavior_fold_canonical_digest(
    qbehavior_fold_handle handle,
    char* output,
    size_t output_size) {
    return guarded_behavior([&] {
        write_behavior_digest(as_behavior(handle)->universe.canonical_digest(), output, output_size);
    });
}

}  // extern "C"

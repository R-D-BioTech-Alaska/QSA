#pragma once

#include "qubit/qtropical_qtt.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace qubit {

struct PersistentTropicalQTTConfig {
    std::size_t max_tree_nodes{1U << 20U};
    std::size_t max_total_transfer_entries{1U << 24U};
    std::size_t max_update_transfer_entries{1U << 20U};
};

struct PersistentTropicalQTTStats {
    std::size_t logical_bits{0U};
    std::size_t maximum_rank{0U};
    std::size_t tree_nodes{0U};
    std::size_t transfer_entries{0U};
    std::size_t tree_depth{0U};
};

struct PersistentTropicalQTTUpdate {
    std::size_t core_index{0U};
    std::int64_t old_minimum_energy{0};
    std::int64_t new_minimum_energy{0};
    std::size_t nodes_created{0U};
    std::size_t transfer_entries_created{0U};
};

class PersistentTropicalQTT {
public:
    [[nodiscard]] static PersistentTropicalQTT build(
        const ExactTropicalQTT& state,
        PersistentTropicalQTTConfig config = {}) {
        validate_config(config);
        const std::size_t logical_bits = state.logical_bits();
        if (logical_bits == 0U) {
            throw QStateError("Persistent Tropical QTT requires at least one core");
        }
        const std::size_t node_count = checked_node_count(logical_bits);
        if (node_count > config.max_tree_nodes) {
            throw QStateError("Persistent Tropical QTT exceeds configured tree-node cap");
        }

        BuildBudget budget{config.max_total_transfer_entries, 0U, 0U};
        std::shared_ptr<const Node> root = build_range(
            state.cores(), 0U, logical_bits, budget);
        if (!root->transfer.values.front().supported) {
            throw QStateError("Persistent Tropical QTT has no globally supported sector");
        }
        return PersistentTropicalQTT(
            std::move(root),
            logical_bits,
            state.stats().maximum_rank,
            PersistentTropicalQTTStats{
                logical_bits,
                state.stats().maximum_rank,
                budget.nodes,
                budget.entries,
                depth_for(logical_bits),
            },
            config);
    }

    [[nodiscard]] std::size_t logical_bits() const noexcept { return logical_bits_; }
    [[nodiscard]] std::size_t maximum_rank() const noexcept { return maximum_rank_; }
    [[nodiscard]] const PersistentTropicalQTTStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const PersistentTropicalQTTConfig& config() const noexcept { return config_; }

    [[nodiscard]] std::int64_t minimum_energy() const {
        const TropicalQTTEntry& value = root_->transfer.values.front();
        if (!value.supported) {
            throw QStateError("Persistent Tropical QTT has no globally supported sector");
        }
        return value.cost;
    }

    [[nodiscard]] std::uint8_t selected_bit(std::size_t index) const {
        if (index >= logical_bits_) {
            throw QStateError("Persistent Tropical QTT selected-bit index is outside logical shape");
        }
        const Node* node = root_.get();
        std::size_t start_state = 0U;
        std::size_t end_state = 0U;
        while (!node->leaf) {
            const std::size_t middle = argmin_middle(
                node->left->transfer,
                node->right->transfer,
                start_state,
                end_state);
            if (index < node->left->end) {
                node = node->left.get();
                end_state = middle;
            } else {
                node = node->right.get();
                start_state = middle;
            }
        }
        return leaf_bit(node->core, start_state, end_state);
    }

    [[nodiscard]] std::vector<std::uint8_t> minimum_bits() const {
        std::vector<std::uint8_t> bits(logical_bits_);
        descend_bits(*root_, 0U, 0U, bits);
        return bits;
    }

    [[nodiscard]] ExactTropicalQTT to_state() const {
        std::vector<TropicalQTTCore> cores(logical_bits_);
        collect_cores(*root_, cores);
        TropicalQTTConfig state_config;
        state_config.max_rank = std::max(state_config.max_rank, maximum_rank_);
        const std::size_t rank_square = checked_size_product(
            maximum_rank_, maximum_rank_, "Persistent Tropical QTT state rank-square overflowed");
        const std::size_t core_entries = checked_size_product(
            rank_square, 2U, "Persistent Tropical QTT state core-entry cap overflowed");
        state_config.max_core_entries = std::max(state_config.max_core_entries, core_entries);
        state_config.max_total_entries = std::max(
            state_config.max_total_entries,
            checked_size_product(
                stats_.transfer_entries,
                2U,
                "Persistent Tropical QTT state total-entry cap overflowed"));
        return ExactTropicalQTT::from_certified_cores(std::move(cores), state_config);
    }

    [[nodiscard]] std::pair<PersistentTropicalQTT, PersistentTropicalQTTUpdate> update_core(
        std::size_t index,
        TropicalQTTCore core) const {
        if (index >= logical_bits_) {
            throw QStateError("Persistent Tropical QTT update index is outside logical shape");
        }
        const Node& old_leaf = leaf_node(index);
        if (core.left_rank != old_leaf.core.left_rank ||
            core.right_rank != old_leaf.core.right_rank) {
            throw QStateError("Persistent Tropical QTT update cannot change local bond dimensions");
        }
        validate_core_storage(core);

        UpdateBudget budget{config_.max_update_transfer_entries, 0U, 0U};
        std::shared_ptr<const Node> root = update_node(root_, index, std::move(core), budget);
        if (!root->transfer.values.front().supported) {
            throw QStateError("Persistent Tropical QTT update removed all globally supported sectors");
        }

        const std::int64_t old_energy = minimum_energy();
        const std::int64_t new_energy = root->transfer.values.front().cost;
        PersistentTropicalQTT next(
            std::move(root), logical_bits_, maximum_rank_, stats_, config_);
        PersistentTropicalQTTUpdate receipt{
            index,
            old_energy,
            new_energy,
            budget.nodes,
            budget.entries,
        };
        return {std::move(next), receipt};
    }

private:
    struct MinPlusMatrix {
        std::size_t rows{0U};
        std::size_t columns{0U};
        std::vector<TropicalQTTEntry> values{};
    };

    struct Node {
        std::size_t start{0U};
        std::size_t end{0U};
        MinPlusMatrix transfer{};
        std::shared_ptr<const Node> left{};
        std::shared_ptr<const Node> right{};
        TropicalQTTCore core{};
        bool leaf{false};
    };

    struct BuildBudget {
        std::size_t cap{0U};
        std::size_t entries{0U};
        std::size_t nodes{0U};
    };

    struct UpdateBudget {
        std::size_t cap{0U};
        std::size_t entries{0U};
        std::size_t nodes{0U};
    };

    std::shared_ptr<const Node> root_{};
    std::size_t logical_bits_{0U};
    std::size_t maximum_rank_{0U};
    PersistentTropicalQTTStats stats_{};
    PersistentTropicalQTTConfig config_{};

    PersistentTropicalQTT(
        std::shared_ptr<const Node> root,
        std::size_t logical_bits,
        std::size_t maximum_rank,
        PersistentTropicalQTTStats stats,
        PersistentTropicalQTTConfig config)
        : root_(std::move(root)),
          logical_bits_(logical_bits),
          maximum_rank_(maximum_rank),
          stats_(stats),
          config_(config) {}

    [[nodiscard]] static std::size_t checked_size_product(
        std::size_t left,
        std::size_t right,
        const char* message) {
        if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
            throw QStateError(message);
        }
        return left * right;
    }

    [[nodiscard]] static std::size_t checked_size_sum(
        std::size_t left,
        std::size_t right,
        const char* message) {
        if (right > std::numeric_limits<std::size_t>::max() - left) {
            throw QStateError(message);
        }
        return left + right;
    }

    [[nodiscard]] static std::int64_t checked_cost_sum(
        std::int64_t left,
        std::int64_t right) {
        if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
            (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right)) {
            throw QStateError("Persistent Tropical QTT integer cost overflowed");
        }
        return left + right;
    }

    static void validate_config(const PersistentTropicalQTTConfig& config) {
        if (config.max_tree_nodes == 0U || config.max_total_transfer_entries == 0U ||
            config.max_update_transfer_entries == 0U) {
            throw QStateError("Persistent Tropical QTT configuration contains a zero resource cap");
        }
    }

    [[nodiscard]] static std::size_t checked_node_count(std::size_t logical_bits) {
        return checked_size_sum(
            logical_bits,
            logical_bits - 1U,
            "Persistent Tropical QTT tree-node count overflowed");
    }

    [[nodiscard]] static std::size_t depth_for(std::size_t logical_bits) noexcept {
        return logical_bits <= 1U ? 0U : static_cast<std::size_t>(std::bit_width(logical_bits - 1U));
    }

    static void validate_core_storage(const TropicalQTTCore& core) {
        if (core.left_rank == 0U || core.right_rank == 0U) {
            throw QStateError("Persistent Tropical QTT update core ranks must be positive");
        }
        const std::size_t matrix_size = checked_size_product(
            core.left_rank,
            core.right_rank,
            "Persistent Tropical QTT update core size overflowed");
        if (core.zero.size() != matrix_size || core.one.size() != matrix_size) {
            throw QStateError("Persistent Tropical QTT update core slices do not match declared ranks");
        }
    }

    static void reserve_entry_budget(
        std::size_t count,
        std::size_t cap,
        std::size_t& used,
        const char* message) {
        used = checked_size_sum(used, count, message);
        if (used > cap) {
            throw QStateError(message);
        }
    }

    [[nodiscard]] static std::size_t core_transfer_entries(const TropicalQTTCore& core) {
        validate_core_storage(core);
        return checked_size_product(
            core.left_rank,
            core.right_rank,
            "Persistent Tropical QTT core transfer size overflowed");
    }

    [[nodiscard]] static std::size_t composed_transfer_entries(
        const MinPlusMatrix& left,
        const MinPlusMatrix& right) {
        if (left.columns != right.rows) {
            throw QStateError("Persistent Tropical QTT transfer dimensions do not align");
        }
        return checked_size_product(
            left.rows,
            right.columns,
            "Persistent Tropical QTT composed transfer size overflowed");
    }

    [[nodiscard]] static MinPlusMatrix core_transfer(const TropicalQTTCore& core) {
        const std::size_t count = core_transfer_entries(core);
        MinPlusMatrix matrix{core.left_rank, core.right_rank, std::vector<TropicalQTTEntry>(count)};
        for (std::size_t index = 0U; index < count; ++index) {
            const TropicalQTTEntry& zero = core.zero[index];
            const TropicalQTTEntry& one = core.one[index];
            if (zero.supported && one.supported) {
                matrix.values[index] = zero.cost <= one.cost ? zero : one;
            } else if (zero.supported) {
                matrix.values[index] = zero;
            } else if (one.supported) {
                matrix.values[index] = one;
            }
        }
        return matrix;
    }

    [[nodiscard]] static MinPlusMatrix compose(
        const MinPlusMatrix& left,
        const MinPlusMatrix& right) {
        const std::size_t count = composed_transfer_entries(left, right);
        MinPlusMatrix output{left.rows, right.columns, std::vector<TropicalQTTEntry>(count)};
        for (std::size_t row = 0U; row < left.rows; ++row) {
            for (std::size_t column = 0U; column < right.columns; ++column) {
                TropicalQTTEntry best{};
                for (std::size_t middle = 0U; middle < left.columns; ++middle) {
                    const TropicalQTTEntry& left_value =
                        left.values[row * left.columns + middle];
                    const TropicalQTTEntry& right_value =
                        right.values[middle * right.columns + column];
                    if (!left_value.supported || !right_value.supported) {
                        continue;
                    }
                    const std::int64_t candidate =
                        checked_cost_sum(left_value.cost, right_value.cost);
                    if (!best.supported || candidate < best.cost) {
                        best = TropicalQTTEntry::edge(candidate);
                    }
                }
                output.values[row * right.columns + column] = best;
            }
        }
        return output;
    }

    [[nodiscard]] static std::shared_ptr<const Node> build_range(
        const std::vector<TropicalQTTCore>& cores,
        std::size_t start,
        std::size_t end,
        BuildBudget& budget) {
        if (end - start == 1U) {
            const std::size_t entry_count = core_transfer_entries(cores[start]);
            reserve_entry_budget(
                entry_count,
                budget.cap,
                budget.entries,
                "Persistent Tropical QTT build exceeds transfer-entry cap");
            MinPlusMatrix transfer = core_transfer(cores[start]);
            budget.nodes = checked_size_sum(
                budget.nodes, 1U, "Persistent Tropical QTT build node count overflowed");
            return std::make_shared<Node>(Node{
                start, end, std::move(transfer), {}, {}, cores[start], true,
            });
        }
        const std::size_t middle = start + (end - start) / 2U;
        std::shared_ptr<const Node> left = build_range(cores, start, middle, budget);
        std::shared_ptr<const Node> right = build_range(cores, middle, end, budget);
        const std::size_t entry_count = composed_transfer_entries(left->transfer, right->transfer);
        reserve_entry_budget(
            entry_count,
            budget.cap,
            budget.entries,
            "Persistent Tropical QTT build exceeds transfer-entry cap");
        MinPlusMatrix transfer = compose(left->transfer, right->transfer);
        budget.nodes = checked_size_sum(
            budget.nodes, 1U, "Persistent Tropical QTT build node count overflowed");
        return std::make_shared<Node>(Node{
            start, end, std::move(transfer), std::move(left), std::move(right), {}, false,
        });
    }

    [[nodiscard]] const Node& leaf_node(std::size_t index) const {
        const Node* node = root_.get();
        while (!node->leaf) {
            node = index < node->left->end ? node->left.get() : node->right.get();
        }
        return *node;
    }

    [[nodiscard]] static std::shared_ptr<const Node> update_node(
        const std::shared_ptr<const Node>& node,
        std::size_t index,
        TropicalQTTCore core,
        UpdateBudget& budget) {
        if (node->leaf) {
            const std::size_t entry_count = core_transfer_entries(core);
            reserve_entry_budget(
                entry_count,
                budget.cap,
                budget.entries,
                "Persistent Tropical QTT update exceeds transfer-entry cap");
            MinPlusMatrix transfer = core_transfer(core);
            budget.nodes = checked_size_sum(
                budget.nodes, 1U, "Persistent Tropical QTT update node count overflowed");
            return std::make_shared<Node>(Node{
                node->start, node->end, std::move(transfer), {}, {}, std::move(core), true,
            });
        }

        std::shared_ptr<const Node> left = node->left;
        std::shared_ptr<const Node> right = node->right;
        if (index < left->end) {
            left = update_node(left, index, std::move(core), budget);
        } else {
            right = update_node(right, index, std::move(core), budget);
        }
        const std::size_t entry_count = composed_transfer_entries(left->transfer, right->transfer);
        reserve_entry_budget(
            entry_count,
            budget.cap,
            budget.entries,
            "Persistent Tropical QTT update exceeds transfer-entry cap");
        MinPlusMatrix transfer = compose(left->transfer, right->transfer);
        budget.nodes = checked_size_sum(
            budget.nodes, 1U, "Persistent Tropical QTT update node count overflowed");
        return std::make_shared<Node>(Node{
            node->start,
            node->end,
            std::move(transfer),
            std::move(left),
            std::move(right),
            {},
            false,
        });
    }

    [[nodiscard]] static std::size_t argmin_middle(
        const MinPlusMatrix& left,
        const MinPlusMatrix& right,
        std::size_t start_state,
        std::size_t end_state) {
        if (start_state >= left.rows || end_state >= right.columns ||
            left.columns != right.rows) {
            throw QStateError("Persistent Tropical QTT optimum boundary states are invalid");
        }
        bool have = false;
        std::size_t best_middle = 0U;
        std::int64_t best_cost = 0;
        for (std::size_t middle = 0U; middle < left.columns; ++middle) {
            const TropicalQTTEntry& left_value =
                left.values[start_state * left.columns + middle];
            const TropicalQTTEntry& right_value =
                right.values[middle * right.columns + end_state];
            if (!left_value.supported || !right_value.supported) {
                continue;
            }
            const std::int64_t candidate = checked_cost_sum(left_value.cost, right_value.cost);
            if (!have || candidate < best_cost ||
                (candidate == best_cost && middle < best_middle)) {
                have = true;
                best_middle = middle;
                best_cost = candidate;
            }
        }
        if (!have) {
            throw QStateError("Persistent Tropical QTT optimum path has no supported middle state");
        }
        return best_middle;
    }

    [[nodiscard]] static std::uint8_t leaf_bit(
        const TropicalQTTCore& core,
        std::size_t start_state,
        std::size_t end_state) {
        if (start_state >= core.left_rank || end_state >= core.right_rank) {
            throw QStateError("Persistent Tropical QTT leaf boundary state is invalid");
        }
        const std::size_t offset = start_state * core.right_rank + end_state;
        const TropicalQTTEntry& zero = core.zero[offset];
        const TropicalQTTEntry& one = core.one[offset];
        if (zero.supported && one.supported) {
            return zero.cost <= one.cost ? 0U : 1U;
        }
        if (zero.supported) {
            return 0U;
        }
        if (one.supported) {
            return 1U;
        }
        throw QStateError("Persistent Tropical QTT optimum path reached an unsupported leaf");
    }

    static void descend_bits(
        const Node& node,
        std::size_t start_state,
        std::size_t end_state,
        std::vector<std::uint8_t>& bits) {
        if (node.leaf) {
            bits[node.start] = leaf_bit(node.core, start_state, end_state);
            return;
        }
        const std::size_t middle = argmin_middle(
            node.left->transfer, node.right->transfer, start_state, end_state);
        descend_bits(*node.left, start_state, middle, bits);
        descend_bits(*node.right, middle, end_state, bits);
    }

    static void collect_cores(const Node& node, std::vector<TropicalQTTCore>& cores) {
        if (node.leaf) {
            cores[node.start] = node.core;
            return;
        }
        collect_cores(*node.left, cores);
        collect_cores(*node.right, cores);
    }
};

}  // namespace qubit

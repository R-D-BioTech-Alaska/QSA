#pragma once

#include "qubit/qstate.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace qubit {

using ExactSemanticNodeId = std::uint32_t;

enum class ExactSemanticDecisionStatus : std::uint8_t {
    Empty = 0,
    Unique = 1,
    Ambiguous = 2,
};

struct ExactSemanticForestConfig {
    std::size_t max_nodes{1U << 20U};
    std::size_t max_children_per_node{1U << 16U};
    std::size_t max_identity_bytes{1U << 20U};
    std::size_t max_attribute_bytes{1U << 20U};
};

struct ExactSemanticForestStats {
    std::size_t nodes{0U};
    std::size_t atoms{0U};
    std::size_t composites{0U};
    std::size_t child_references{0U};
    std::size_t max_children{0U};
    std::size_t ambiguity_short_circuits{0U};
    std::size_t estimated_bytes{0U};
};

struct ExactSemanticNodeView {
    std::string_view kind{};
    std::string_view attributes{};
    std::span<const ExactSemanticNodeId> children{};
    bool commutative{false};
};

class ExactSemanticDecisionForest;

class ExactSemanticDecision {
public:
    [[nodiscard]] ExactSemanticDecisionStatus status() const noexcept {
        if (size_ == 0U) {
            return ExactSemanticDecisionStatus::Empty;
        }
        if (size_ == 1U) {
            return ExactSemanticDecisionStatus::Unique;
        }
        return ExactSemanticDecisionStatus::Ambiguous;
    }

    [[nodiscard]] std::size_t witness_count() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0U; }
    [[nodiscard]] bool unique() const noexcept { return size_ == 1U; }
    [[nodiscard]] bool ambiguous() const noexcept { return size_ == 2U; }

    [[nodiscard]] ExactSemanticNodeId witness(std::size_t index) const {
        if (index >= size_) {
            throw QStateError("Semantic decision witness is out of range");
        }
        return witnesses_[index];
    }

private:
    const ExactSemanticDecisionForest* owner_{nullptr};
    std::array<ExactSemanticNodeId, 2> witnesses_{};
    std::uint8_t size_{0U};

    ExactSemanticDecision(
        const ExactSemanticDecisionForest* owner,
        ExactSemanticNodeId first)
        : owner_(owner), witnesses_{first, 0U}, size_(1U) {}

    ExactSemanticDecision(
        const ExactSemanticDecisionForest* owner,
        ExactSemanticNodeId first,
        ExactSemanticNodeId second)
        : owner_(owner), witnesses_{first, second}, size_(2U) {}

    explicit ExactSemanticDecision(const ExactSemanticDecisionForest* owner)
        : owner_(owner) {}

    friend class ExactSemanticDecisionForest;
};

class ExactSemanticDecisionForest {
public:
    explicit ExactSemanticDecisionForest(ExactSemanticForestConfig config = {})
        : config_(config) {
        if (config_.max_nodes == 0U ||
            config_.max_nodes > static_cast<std::size_t>(std::numeric_limits<ExactSemanticNodeId>::max()) ||
            config_.max_children_per_node == 0U ||
            config_.max_identity_bytes == 0U ||
            config_.max_attribute_bytes == 0U) {
            throw QStateError("Semantic decision forest configuration is invalid");
        }
    }

    [[nodiscard]] ExactSemanticDecision empty() const noexcept {
        return ExactSemanticDecision(this);
    }

    [[nodiscard]] ExactSemanticDecision atom(
        std::string_view semantic_identity,
        std::string_view domain = "semantic") {
        if (semantic_identity.empty() || semantic_identity.size() > config_.max_identity_bytes ||
            domain.empty() || domain.size() > config_.max_attribute_bytes) {
            throw QStateError("Semantic atom identity or domain is invalid");
        }
        const std::string attributes =
            std::string(domain) + "\n" + std::string(semantic_identity);
        const ExactSemanticNodeId id = intern("ATOM", attributes, {}, false);
        return ExactSemanticDecision(this, id);
    }

    [[nodiscard]] ExactSemanticDecision merge(
        std::span<const ExactSemanticDecision> decisions) const {
        std::array<ExactSemanticNodeId, 2> found{};
        std::size_t count = 0U;
        for (const ExactSemanticDecision& decision : decisions) {
            validate_decision(decision);
            for (std::size_t index = 0U; index < decision.size_; ++index) {
                const ExactSemanticNodeId candidate = decision.witnesses_[index];
                bool duplicate = false;
                for (std::size_t existing = 0U; existing < count; ++existing) {
                    duplicate = duplicate || found[existing] == candidate;
                }
                if (duplicate) {
                    continue;
                }
                found[count++] = candidate;
                if (count == 2U) {
                    return ExactSemanticDecision(this, found[0], found[1]);
                }
            }
        }
        if (count == 0U) {
            return empty();
        }
        return ExactSemanticDecision(this, found[0]);
    }

    [[nodiscard]] ExactSemanticDecision compose(
        std::string_view kind,
        std::span<const ExactSemanticDecision> children,
        std::string_view attributes = {},
        bool commutative = false) {
        if (kind.empty() || kind.size() > config_.max_attribute_bytes ||
            attributes.size() > config_.max_attribute_bytes ||
            children.empty() || children.size() > config_.max_children_per_node) {
            throw QStateError("Semantic composition dimensions are invalid");
        }

        std::vector<ExactSemanticNodeId> baseline;
        baseline.reserve(children.size());
        std::vector<std::size_t> ambiguous_positions;
        for (std::size_t index = 0U; index < children.size(); ++index) {
            const ExactSemanticDecision& child = children[index];
            validate_decision(child);
            if (child.empty()) {
                return empty();
            }
            baseline.push_back(child.witnesses_[0]);
            if (child.ambiguous()) {
                ambiguous_positions.push_back(index);
            }
        }

        const ExactSemanticNodeId first = intern(kind, attributes, baseline, commutative);
        for (const std::size_t position : ambiguous_positions) {
            std::vector<ExactSemanticNodeId> alternate = baseline;
            alternate[position] = children[position].witnesses_[1];
            const ExactSemanticNodeId second = intern(kind, attributes, alternate, commutative);
            if (second != first) {
                ++stats_.ambiguity_short_circuits;
                refresh_stats_bytes();
                return ExactSemanticDecision(this, first, second);
            }
        }
        refresh_stats_bytes();
        return ExactSemanticDecision(this, first);
    }

    [[nodiscard]] ExactSemanticNodeView node(ExactSemanticNodeId id) const {
        const std::size_t index = static_cast<std::size_t>(id);
        if (index >= nodes_.size()) {
            throw QStateError("Semantic node id is out of range");
        }
        const Node& value = nodes_[index];
        return ExactSemanticNodeView{
            value.kind,
            value.attributes,
            value.children,
            value.commutative,
        };
    }

    [[nodiscard]] const ExactSemanticForestConfig& config() const noexcept { return config_; }
    [[nodiscard]] const ExactSemanticForestStats& stats() const noexcept { return stats_; }

private:
    struct Node {
        std::string kind{};
        std::string attributes{};
        std::vector<ExactSemanticNodeId> children{};
        bool commutative{false};
    };

    struct Key {
        bool commutative{false};
        std::string kind{};
        std::string attributes{};
        std::vector<ExactSemanticNodeId> children{};

        [[nodiscard]] bool operator<(const Key& other) const noexcept {
            return std::tie(commutative, kind, attributes, children) <
                std::tie(other.commutative, other.kind, other.attributes, other.children);
        }
    };

    ExactSemanticForestConfig config_{};
    std::vector<Node> nodes_{};
    std::map<Key, ExactSemanticNodeId> index_{};
    ExactSemanticForestStats stats_{};

    [[nodiscard]] ExactSemanticNodeId intern(
        std::string_view kind,
        std::string_view attributes,
        std::span<const ExactSemanticNodeId> children,
        bool commutative) {
        if (children.size() > config_.max_children_per_node) {
            throw QStateError("Semantic node exceeds configured child cap");
        }
        Key key{
            commutative,
            std::string(kind),
            std::string(attributes),
            std::vector<ExactSemanticNodeId>(children.begin(), children.end()),
        };
        for (const ExactSemanticNodeId child : key.children) {
            if (static_cast<std::size_t>(child) >= nodes_.size()) {
                throw QStateError("Semantic node references an unknown child");
            }
        }
        if (commutative) {
            std::sort(key.children.begin(), key.children.end());
        }
        const auto existing = index_.find(key);
        if (existing != index_.end()) {
            return existing->second;
        }
        if (nodes_.size() >= config_.max_nodes) {
            throw QStateError("Semantic decision forest exceeds configured node cap");
        }
        const ExactSemanticNodeId id = static_cast<ExactSemanticNodeId>(nodes_.size());
        nodes_.push_back(Node{key.kind, key.attributes, key.children, key.commutative});
        index_.emplace(std::move(key), id);
        ++stats_.nodes;
        if (children.empty()) {
            ++stats_.atoms;
        } else {
            ++stats_.composites;
            stats_.child_references = checked_sum(stats_.child_references, children.size());
            stats_.max_children = std::max(stats_.max_children, children.size());
        }
        refresh_stats_bytes();
        return id;
    }

    void validate_decision(const ExactSemanticDecision& decision) const {
        if (decision.owner_ != this) {
            throw QStateError("Semantic decision belongs to a different forest");
        }
    }

    void refresh_stats_bytes() noexcept {
        std::size_t bytes = sizeof(*this);
        for (const Node& node : nodes_) {
            bytes = saturating_sum(bytes, sizeof(Node));
            bytes = saturating_sum(bytes, node.kind.capacity());
            bytes = saturating_sum(bytes, node.attributes.capacity());
            bytes = saturating_sum(
                bytes,
                saturating_product(node.children.capacity(), sizeof(ExactSemanticNodeId)));
        }
        stats_.estimated_bytes = bytes;
    }

    [[nodiscard]] static std::size_t checked_sum(std::size_t left, std::size_t right) {
        if (right > std::numeric_limits<std::size_t>::max() - left) {
            throw QStateError("Semantic decision forest counter overflowed");
        }
        return left + right;
    }

    [[nodiscard]] static std::size_t saturating_sum(
        std::size_t left,
        std::size_t right) noexcept {
        if (right > std::numeric_limits<std::size_t>::max() - left) {
            return std::numeric_limits<std::size_t>::max();
        }
        return left + right;
    }

    [[nodiscard]] static std::size_t saturating_product(
        std::size_t left,
        std::size_t right) noexcept {
        if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
            return std::numeric_limits<std::size_t>::max();
        }
        return left * right;
    }
};

[[nodiscard]] inline const char* exact_semantic_decision_status_name(
    ExactSemanticDecisionStatus status) noexcept {
    switch (status) {
        case ExactSemanticDecisionStatus::Empty: return "Empty";
        case ExactSemanticDecisionStatus::Unique: return "Unique";
        case ExactSemanticDecisionStatus::Ambiguous: return "Ambiguous";
    }
    return "Unknown";
}

}  // namespace qubit

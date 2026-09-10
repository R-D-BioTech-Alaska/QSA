#pragma once

#include "qubit/qmath.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qubit {

enum class QOrderRelation : std::uint8_t {
    Same = 0,
    Before = 1,
    After = 2,
    Unresolved = 3,
};

struct QStrictOrderConfig {
    std::size_t max_symbols{4'096U};
    std::size_t max_relations{1U << 20U};
    std::size_t max_closure_words{1U << 23U};
};

struct QStrictOrderReceipt {
    std::size_t symbols{0U};
    std::size_t direct_relations{0U};
    std::size_t closure_relations{0U};
    std::size_t inferred_relations{0U};
    std::size_t closure_words{0U};
    std::size_t minima{0U};
    std::size_t maxima{0U};
    std::string canonical{};
    bool exact{true};
    bool acyclic{true};
};

class QStrictOrder {
public:
    using SymbolId = std::size_t;

    explicit QStrictOrder(QStrictOrderConfig config = {}) : config_(config) {
        if (config_.max_symbols == 0U || config_.max_relations == 0U ||
            config_.max_closure_words == 0U) {
            throw QMathError("strict-order limits must be positive");
        }
    }

    [[nodiscard]] SymbolId add_symbol(std::string identity) {
        if (identity.empty()) throw QMathError("strict-order symbol identity is empty");
        const auto found = index_.find(identity);
        if (found != index_.end()) return found->second;
        if (symbols_.size() >= config_.max_symbols) {
            throw QMathError("strict-order symbol cap exceeded");
        }
        const SymbolId id = symbols_.size();
        symbols_.push_back(std::move(identity));
        index_.emplace(symbols_.back(), id);
        invalidate();
        return id;
    }

    [[nodiscard]] std::optional<SymbolId> find_symbol(std::string_view identity) const {
        const auto found = index_.find(std::string(identity));
        if (found == index_.end()) return std::nullopt;
        return found->second;
    }

    [[nodiscard]] bool add_before(SymbolId before, SymbolId after) {
        require_symbol(before);
        require_symbol(after);
        if (before == after) throw QMathError("strict order cannot contain a self relation");
        if (direct_path(before, after)) return false;
        if (direct_path(after, before)) {
            throw QMathError("strict-order relation would create a cycle");
        }
        if (relations_.size() >= config_.max_relations) {
            throw QMathError("strict-order relation cap exceeded");
        }
        relations_.emplace_back(before, after);
        invalidate();
        return true;
    }

    [[nodiscard]] bool add_before(std::string_view before, std::string_view after) {
        return add_before(require_symbol(before), require_symbol(after));
    }

    [[nodiscard]] QOrderRelation relation(SymbolId lhs, SymbolId rhs) const {
        require_symbol(lhs);
        require_symbol(rhs);
        if (lhs == rhs) return QOrderRelation::Same;
        ensure_closed();
        if (closed(lhs, rhs)) return QOrderRelation::Before;
        if (closed(rhs, lhs)) return QOrderRelation::After;
        return QOrderRelation::Unresolved;
    }

    [[nodiscard]] QOrderRelation relation(std::string_view lhs, std::string_view rhs) const {
        return relation(require_symbol(lhs), require_symbol(rhs));
    }

    [[nodiscard]] bool precedes(SymbolId lhs, SymbolId rhs) const {
        return relation(lhs, rhs) == QOrderRelation::Before;
    }

    [[nodiscard]] std::vector<SymbolId> minima() const {
        ensure_closed();
        std::vector<SymbolId> result;
        for (SymbolId candidate = 0U; candidate < symbols_.size(); ++candidate) {
            bool incoming = false;
            for (SymbolId other = 0U; other < symbols_.size(); ++other) {
                if (other != candidate && closed(other, candidate)) {
                    incoming = true;
                    break;
                }
            }
            if (!incoming) result.push_back(candidate);
        }
        return result;
    }

    [[nodiscard]] std::vector<SymbolId> maxima() const {
        ensure_closed();
        std::vector<SymbolId> result;
        for (SymbolId candidate = 0U; candidate < symbols_.size(); ++candidate) {
            bool outgoing = false;
            for (SymbolId other = 0U; other < symbols_.size(); ++other) {
                if (other != candidate && closed(candidate, other)) {
                    outgoing = true;
                    break;
                }
            }
            if (!outgoing) result.push_back(candidate);
        }
        return result;
    }

    [[nodiscard]] std::optional<SymbolId> unique_minimum() const {
        const std::vector<SymbolId> values = minima();
        if (values.size() != 1U) return std::nullopt;
        return values.front();
    }

    [[nodiscard]] std::optional<SymbolId> unique_maximum() const {
        const std::vector<SymbolId> values = maxima();
        if (values.size() != 1U) return std::nullopt;
        return values.front();
    }

    [[nodiscard]] const std::string& symbol(SymbolId id) const {
        require_symbol(id);
        return symbols_[id];
    }

    [[nodiscard]] std::span<const std::string> symbols() const noexcept {
        return symbols_;
    }

    [[nodiscard]] std::size_t symbol_count() const noexcept {
        return symbols_.size();
    }

    [[nodiscard]] std::size_t direct_relation_count() const noexcept {
        return relations_.size();
    }

    [[nodiscard]] std::string canonical() const {
        ensure_closed();
        return compiled_->canonical;
    }

    [[nodiscard]] QStrictOrderReceipt receipt() const {
        ensure_closed();
        QStrictOrderReceipt result;
        result.symbols = symbols_.size();
        result.direct_relations = relations_.size();
        result.closure_relations = compiled_->relation_count;
        result.inferred_relations = compiled_->relation_count - relations_.size();
        result.closure_words = compiled_->reach.size();
        result.minima = minima().size();
        result.maxima = maxima().size();
        result.canonical = compiled_->canonical;
        return result;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) +
                            symbols_.capacity() * sizeof(std::string) +
                            relations_.capacity() * sizeof(Relation);
        for (const std::string& value : symbols_) bytes += value.capacity();
        if (compiled_.has_value()) {
            bytes += compiled_->reach.capacity() * sizeof(std::uint64_t) +
                     compiled_->canonical.capacity();
        }
        return bytes;
    }

private:
    using Relation = std::pair<SymbolId, SymbolId>;

    struct Compiled {
        std::size_t words_per_row{0U};
        std::size_t relation_count{0U};
        std::vector<std::uint64_t> reach{};
        std::string canonical{};
    };

    QStrictOrderConfig config_{};
    std::vector<std::string> symbols_{};
    std::unordered_map<std::string, SymbolId> index_{};
    std::vector<Relation> relations_{};
    mutable std::optional<Compiled> compiled_{};

    void invalidate() noexcept {
        compiled_.reset();
    }

    void require_symbol(SymbolId id) const {
        if (id >= symbols_.size()) throw QMathError("strict-order symbol id is out of range");
    }

    [[nodiscard]] SymbolId require_symbol(std::string_view identity) const {
        const std::optional<SymbolId> id = find_symbol(identity);
        if (!id.has_value()) throw QMathError("strict-order symbol identity is unknown");
        return *id;
    }

    [[nodiscard]] bool direct_path(SymbolId source, SymbolId target) const {
        if (source == target) return true;
        std::vector<std::uint8_t> seen(symbols_.size(), 0U);
        std::vector<SymbolId> stack;
        stack.push_back(source);
        seen[source] = 1U;
        while (!stack.empty()) {
            const SymbolId current = stack.back();
            stack.pop_back();
            for (const Relation& relation_value : relations_) {
                if (relation_value.first != current) continue;
                const SymbolId next = relation_value.second;
                if (next == target) return true;
                if (seen[next] == 0U) {
                    seen[next] = 1U;
                    stack.push_back(next);
                }
            }
        }
        return false;
    }

    [[nodiscard]] bool closed(SymbolId lhs, SymbolId rhs) const noexcept {
        const Compiled& value = *compiled_;
        const std::size_t offset = lhs * value.words_per_row + rhs / 64U;
        return (value.reach[offset] & (std::uint64_t{1U} << (rhs % 64U))) != 0U;
    }

    void ensure_closed() const {
        if (compiled_.has_value()) return;
        Compiled result;
        const std::size_t count = symbols_.size();
        result.words_per_row = count == 0U ? 0U : (count + 63U) / 64U;
        if (result.words_per_row != 0U &&
            count > config_.max_closure_words / result.words_per_row) {
            throw QMathError("strict-order closure workspace cap exceeded");
        }
        const std::size_t words = count * result.words_per_row;
        if (words > config_.max_closure_words) {
            throw QMathError("strict-order closure workspace cap exceeded");
        }
        result.reach.assign(words, 0U);
        const auto set_bit = [&](SymbolId row, SymbolId column) {
            result.reach[row * result.words_per_row + column / 64U] |=
                std::uint64_t{1U} << (column % 64U);
        };
        const auto has_bit = [&](SymbolId row, SymbolId column) {
            return (result.reach[row * result.words_per_row + column / 64U] &
                    (std::uint64_t{1U} << (column % 64U))) != 0U;
        };
        for (const Relation& relation_value : relations_) {
            set_bit(relation_value.first, relation_value.second);
        }
        for (SymbolId pivot = 0U; pivot < count; ++pivot) {
            const std::size_t pivot_offset = pivot * result.words_per_row;
            for (SymbolId row = 0U; row < count; ++row) {
                if (!has_bit(row, pivot)) continue;
                const std::size_t row_offset = row * result.words_per_row;
                for (std::size_t word = 0U; word < result.words_per_row; ++word) {
                    result.reach[row_offset + word] |= result.reach[pivot_offset + word];
                }
            }
        }
        for (SymbolId id = 0U; id < count; ++id) {
            if (has_bit(id, id)) throw QMathError("strict-order closure contains a cycle");
        }
        for (const std::uint64_t word : result.reach) {
            result.relation_count += static_cast<std::size_t>(std::popcount(word));
        }
        result.canonical = canonical_closed(result);
        compiled_ = std::move(result);
    }

    [[nodiscard]] std::string canonical_closed(const Compiled& value) const {
        std::vector<SymbolId> order(symbols_.size());
        std::iota(order.begin(), order.end(), 0U);
        std::sort(order.begin(), order.end(), [&](SymbolId lhs, SymbolId rhs) {
            return symbols_[lhs] < symbols_[rhs];
        });
        std::string out = "strict-order:v1:";
        for (const SymbolId id : order) {
            out += std::to_string(symbols_[id].size());
            out += ':';
            out += symbols_[id];
            out += ';';
        }
        out += '|';
        for (std::size_t lhs = 0U; lhs < order.size(); ++lhs) {
            for (std::size_t rhs = 0U; rhs < order.size(); ++rhs) {
                const SymbolId left_id = order[lhs];
                const SymbolId right_id = order[rhs];
                const std::size_t offset = left_id * value.words_per_row + right_id / 64U;
                if ((value.reach[offset] & (std::uint64_t{1U} << (right_id % 64U))) == 0U) continue;
                out += std::to_string(lhs);
                out += '<';
                out += std::to_string(rhs);
                out += ';';
            }
        }
        return out;
    }
};

struct QAffineRelationConfig {
    std::size_t max_symbols{1U << 16U};
    std::size_t max_dimensions{64U};
    std::size_t max_independent_constraints{1U << 20U};
};

struct QAffineRelationReceipt {
    std::size_t dimensions{0U};
    std::size_t symbols{0U};
    std::size_t independent_constraints{0U};
    std::size_t components{0U};
    QPhysicalDimension coordinate_dimension{};
    std::string canonical{};
    bool exact{true};
    bool consistent{true};
};

class QAffineRelationSpace {
public:
    using SymbolId = std::size_t;
    using Displacement = std::vector<QRational>;

    explicit QAffineRelationSpace(
        std::size_t dimensions,
        QPhysicalDimension coordinate_dimension = {},
        QAffineRelationConfig config = {})
        : dimensions_(dimensions),
          coordinate_dimension_(std::move(coordinate_dimension)),
          config_(config) {
        if (dimensions_ == 0U || dimensions_ > config_.max_dimensions ||
            config_.max_symbols == 0U || config_.max_independent_constraints == 0U) {
            throw QMathError("affine-relation limits are invalid");
        }
    }

    [[nodiscard]] SymbolId add_symbol(std::string identity) {
        if (identity.empty()) throw QMathError("affine-relation symbol identity is empty");
        const auto found = index_.find(identity);
        if (found != index_.end()) return found->second;
        if (symbols_.size() >= config_.max_symbols) {
            throw QMathError("affine-relation symbol cap exceeded");
        }
        const SymbolId id = symbols_.size();
        symbols_.push_back(std::move(identity));
        index_.emplace(symbols_.back(), id);
        parent_.push_back(id);
        rank_.push_back(0U);
        potential_.insert(potential_.end(), dimensions_, QRational(0));
        return id;
    }

    [[nodiscard]] std::optional<SymbolId> find_symbol(std::string_view identity) const {
        const auto found = index_.find(std::string(identity));
        if (found == index_.end()) return std::nullopt;
        return found->second;
    }

    [[nodiscard]] bool add_difference(
        SymbolId lhs,
        SymbolId rhs,
        std::span<const QRational> lhs_minus_rhs) {
        require_symbol(lhs);
        require_symbol(rhs);
        require_displacement(lhs_minus_rhs);
        const Trace left = trace(lhs);
        const Trace right = trace(rhs);
        if (left.root == right.root) {
            if (subtract(left.offset, right.offset) != Displacement(lhs_minus_rhs.begin(), lhs_minus_rhs.end())) {
                throw QMathError("affine-relation constraint contradicts accepted state");
            }
            return false;
        }
        if (independent_constraints_ >= config_.max_independent_constraints) {
            throw QMathError("affine-relation independent-constraint cap exceeded");
        }
        const Displacement delta(lhs_minus_rhs.begin(), lhs_minus_rhs.end());
        Displacement root_delta = add(subtract(delta, left.offset), right.offset);
        if (rank_[left.root] <= rank_[right.root]) {
            parent_[left.root] = right.root;
            set_potential(left.root, root_delta);
            if (rank_[left.root] == rank_[right.root]) ++rank_[right.root];
        } else {
            parent_[right.root] = left.root;
            set_potential(right.root, negate(root_delta));
        }
        ++independent_constraints_;
        return true;
    }

    [[nodiscard]] bool add_difference(
        std::string_view lhs,
        std::string_view rhs,
        std::span<const QRational> lhs_minus_rhs) {
        return add_difference(require_symbol(lhs), require_symbol(rhs), lhs_minus_rhs);
    }

    [[nodiscard]] std::optional<Displacement> displacement(SymbolId lhs, SymbolId rhs) const {
        require_symbol(lhs);
        require_symbol(rhs);
        const Trace left = trace(lhs);
        const Trace right = trace(rhs);
        if (left.root != right.root) return std::nullopt;
        return subtract(left.offset, right.offset);
    }

    [[nodiscard]] std::optional<Displacement> displacement(
        std::string_view lhs,
        std::string_view rhs) const {
        return displacement(require_symbol(lhs), require_symbol(rhs));
    }

    [[nodiscard]] bool connected(SymbolId lhs, SymbolId rhs) const {
        require_symbol(lhs);
        require_symbol(rhs);
        return trace(lhs).root == trace(rhs).root;
    }

    [[nodiscard]] const std::string& symbol(SymbolId id) const {
        require_symbol(id);
        return symbols_[id];
    }

    [[nodiscard]] std::span<const std::string> symbols() const noexcept {
        return symbols_;
    }

    [[nodiscard]] std::size_t dimensions() const noexcept {
        return dimensions_;
    }

    [[nodiscard]] std::size_t symbol_count() const noexcept {
        return symbols_.size();
    }

    [[nodiscard]] std::size_t independent_constraint_count() const noexcept {
        return independent_constraints_;
    }

    [[nodiscard]] const QPhysicalDimension& coordinate_dimension() const noexcept {
        return coordinate_dimension_;
    }

    [[nodiscard]] QMathType coordinate_type() const {
        return QMathType{
            QMathScalar::Rational,
            QMathSpace::Vector,
            {dimensions_},
            coordinate_dimension_,
        };
    }

    [[nodiscard]] std::string canonical() const {
        return canonical_state().first;
    }

    [[nodiscard]] QAffineRelationReceipt receipt() const {
        const auto [identity, components] = canonical_state();
        return QAffineRelationReceipt{
            dimensions_,
            symbols_.size(),
            independent_constraints_,
            components,
            coordinate_dimension_,
            identity,
            true,
            true,
        };
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) +
                            symbols_.capacity() * sizeof(std::string) +
                            parent_.capacity() * sizeof(SymbolId) +
                            rank_.capacity() * sizeof(std::uint8_t) +
                            potential_.capacity() * sizeof(QRational);
        for (const std::string& value : symbols_) bytes += value.capacity();
        return bytes;
    }

private:
    struct Trace {
        SymbolId root{0U};
        Displacement offset{};
    };

    std::size_t dimensions_{0U};
    QPhysicalDimension coordinate_dimension_{};
    QAffineRelationConfig config_{};
    std::vector<std::string> symbols_{};
    std::unordered_map<std::string, SymbolId> index_{};
    std::vector<SymbolId> parent_{};
    std::vector<std::uint8_t> rank_{};
    std::vector<QRational> potential_{};
    std::size_t independent_constraints_{0U};

    void require_symbol(SymbolId id) const {
        if (id >= symbols_.size()) throw QMathError("affine-relation symbol id is out of range");
    }

    [[nodiscard]] SymbolId require_symbol(std::string_view identity) const {
        const std::optional<SymbolId> id = find_symbol(identity);
        if (!id.has_value()) throw QMathError("affine-relation symbol identity is unknown");
        return *id;
    }

    void require_displacement(std::span<const QRational> value) const {
        if (value.size() != dimensions_) {
            throw QMathError("affine-relation displacement dimension mismatch");
        }
    }

    [[nodiscard]] Trace trace(SymbolId id) const {
        Displacement offset(dimensions_, QRational(0));
        SymbolId current = id;
        std::size_t steps = 0U;
        while (parent_[current] != current) {
            for (std::size_t axis = 0U; axis < dimensions_; ++axis) {
                offset[axis] = offset[axis] + potential_[current * dimensions_ + axis];
            }
            current = parent_[current];
            if (++steps > symbols_.size()) {
                throw QMathError("affine-relation parent structure is cyclic");
            }
        }
        return Trace{current, std::move(offset)};
    }

    void set_potential(SymbolId id, std::span<const QRational> value) {
        require_displacement(value);
        for (std::size_t axis = 0U; axis < dimensions_; ++axis) {
            potential_[id * dimensions_ + axis] = value[axis];
        }
    }

    [[nodiscard]] static Displacement add(
        std::span<const QRational> lhs,
        std::span<const QRational> rhs) {
        Displacement result(lhs.size());
        for (std::size_t axis = 0U; axis < lhs.size(); ++axis) {
            result[axis] = lhs[axis] + rhs[axis];
        }
        return result;
    }

    [[nodiscard]] static Displacement subtract(
        std::span<const QRational> lhs,
        std::span<const QRational> rhs) {
        Displacement result(lhs.size());
        for (std::size_t axis = 0U; axis < lhs.size(); ++axis) {
            result[axis] = lhs[axis] - rhs[axis];
        }
        return result;
    }

    [[nodiscard]] static Displacement negate(std::span<const QRational> value) {
        Displacement result(value.size());
        for (std::size_t axis = 0U; axis < value.size(); ++axis) {
            result[axis] = QRational(0) - value[axis];
        }
        return result;
    }

    [[nodiscard]] std::pair<std::string, std::size_t> canonical_state() const {
        std::vector<SymbolId> order(symbols_.size());
        std::iota(order.begin(), order.end(), 0U);
        std::sort(order.begin(), order.end(), [&](SymbolId lhs, SymbolId rhs) {
            return symbols_[lhs] < symbols_[rhs];
        });

        const SymbolId none = std::numeric_limits<SymbolId>::max();
        std::vector<SymbolId> anchor_for_root(symbols_.size(), none);
        std::vector<Trace> traces;
        traces.reserve(symbols_.size());
        for (SymbolId id = 0U; id < symbols_.size(); ++id) traces.push_back(trace(id));
        for (const SymbolId id : order) {
            const SymbolId root = traces[id].root;
            if (anchor_for_root[root] == none) anchor_for_root[root] = id;
        }

        std::size_t components = 0U;
        for (const SymbolId anchor : anchor_for_root) {
            if (anchor != none) ++components;
        }

        std::string out = "affine-relation:v1:d=" + std::to_string(dimensions_) +
                          ":u=" + coordinate_dimension_.canonical() + ':';
        for (const SymbolId id : order) {
            const Trace& item = traces[id];
            const SymbolId anchor = anchor_for_root[item.root];
            const Displacement relative = subtract(item.offset, traces[anchor].offset);
            out += std::to_string(symbols_[id].size());
            out += ':';
            out += symbols_[id];
            out += '@';
            out += std::to_string(symbols_[anchor].size());
            out += ':';
            out += symbols_[anchor];
            out += '=';
            for (std::size_t axis = 0U; axis < dimensions_; ++axis) {
                if (axis != 0U) out += ',';
                out += relative[axis].canonical();
            }
            out += ';';
        }
        return {std::move(out), components};
    }
};

[[nodiscard]] inline const char* qorder_relation_name(QOrderRelation relation) noexcept {
    switch (relation) {
        case QOrderRelation::Same:
            return "Same";
        case QOrderRelation::Before:
            return "Before";
        case QOrderRelation::After:
            return "After";
        case QOrderRelation::Unresolved:
            return "Unresolved";
    }
    return "unknown";
}

}  // namespace qubit

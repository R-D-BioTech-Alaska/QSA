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

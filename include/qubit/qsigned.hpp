#pragma once

#include "qubit/qmath.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qubit {

enum class QPolarity : std::int8_t {
    Negative = -1,
    Neutral = 0,
    Positive = 1,
};

enum class QSignedDomain : std::uint8_t {
    Mathematical = 0,
    BrainState = 1,
    QuantumOperator = 2,
    Field = 3,
};

[[nodiscard]] constexpr QPolarity qpolarity_negated(QPolarity value) noexcept {
    if (value == QPolarity::Positive) return QPolarity::Negative;
    if (value == QPolarity::Negative) return QPolarity::Positive;
    return QPolarity::Neutral;
}

[[nodiscard]] constexpr QPolarity qpolarity_product(QPolarity lhs, QPolarity rhs) noexcept {
    if (lhs == QPolarity::Neutral || rhs == QPolarity::Neutral) return QPolarity::Neutral;
    return lhs == rhs ? QPolarity::Positive : QPolarity::Negative;
}

[[nodiscard]] inline QInteger qinteger_from_size(std::size_t value) {
    return QInteger::parse(std::to_string(value));
}

struct QSignedInteractionReceipt {
    std::size_t coordinates{0};
    std::size_t aligned{0};
    std::size_t opposed{0};
    std::size_t unresolved{0};

    [[nodiscard]] QInteger net_alignment() const {
        return qinteger_from_size(aligned) - qinteger_from_size(opposed);
    }

    [[nodiscard]] QInteger active_overlap() const {
        return qinteger_from_size(aligned) + qinteger_from_size(opposed);
    }

    [[nodiscard]] std::string canonical() const {
        return "qsigned-interaction1:" + std::to_string(coordinates) + ':' + std::to_string(aligned) + ':' +
               std::to_string(opposed) + ':' + std::to_string(unresolved);
    }
};

class QTernaryVector {
public:
    explicit QTernaryVector(std::size_t size = 0)
        : size_(size), positive_((size + 63U) / 64U, 0), negative_((size + 63U) / 64U, 0) {}

    QTernaryVector(std::initializer_list<QPolarity> values) : QTernaryVector(values.size()) {
        std::size_t index = 0;
        for (const QPolarity value : values) set(index++, value);
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    [[nodiscard]] QPolarity get(std::size_t index) const {
        check_index(index);
        const std::uint64_t mask = 1ULL << (index & 63U);
        const std::size_t word = index >> 6U;
        if ((positive_[word] & mask) != 0) return QPolarity::Positive;
        if ((negative_[word] & mask) != 0) return QPolarity::Negative;
        return QPolarity::Neutral;
    }

    void set(std::size_t index, QPolarity value) {
        check_index(index);
        const std::uint64_t mask = 1ULL << (index & 63U);
        const std::size_t word = index >> 6U;
        positive_[word] &= ~mask;
        negative_[word] &= ~mask;
        if (value == QPolarity::Positive) positive_[word] |= mask;
        else if (value == QPolarity::Negative) negative_[word] |= mask;
        else if (value != QPolarity::Neutral) throw QMathError("invalid ternary polarity");
    }

    [[nodiscard]] std::size_t positive_count() const noexcept {
        std::size_t count = 0;
        for (const std::uint64_t word : positive_) count += static_cast<std::size_t>(std::popcount(word));
        return count;
    }

    [[nodiscard]] std::size_t negative_count() const noexcept {
        std::size_t count = 0;
        for (const std::uint64_t word : negative_) count += static_cast<std::size_t>(std::popcount(word));
        return count;
    }

    [[nodiscard]] std::size_t neutral_count() const noexcept {
        return size_ - positive_count() - negative_count();
    }

    [[nodiscard]] QTernaryVector negated() const {
        QTernaryVector result(*this);
        result.positive_.swap(result.negative_);
        return result;
    }

    [[nodiscard]] QTernaryVector multiplied(const QTernaryVector& rhs) const {
        require_same_size(rhs);
        QTernaryVector result(size_);
        for (std::size_t i = 0; i < positive_.size(); ++i) {
            result.positive_[i] = (positive_[i] & rhs.positive_[i]) | (negative_[i] & rhs.negative_[i]);
            result.negative_[i] = (positive_[i] & rhs.negative_[i]) | (negative_[i] & rhs.positive_[i]);
        }
        return result;
    }

    [[nodiscard]] QSignedInteractionReceipt interaction(const QTernaryVector& rhs) const {
        require_same_size(rhs);
        QSignedInteractionReceipt receipt;
        receipt.coordinates = size_;
        for (std::size_t i = 0; i < positive_.size(); ++i) {
            const std::uint64_t aligned = (positive_[i] & rhs.positive_[i]) | (negative_[i] & rhs.negative_[i]);
            const std::uint64_t opposed = (positive_[i] & rhs.negative_[i]) | (negative_[i] & rhs.positive_[i]);
            receipt.aligned += static_cast<std::size_t>(std::popcount(aligned));
            receipt.opposed += static_cast<std::size_t>(std::popcount(opposed));
        }
        receipt.unresolved = size_ - receipt.aligned - receipt.opposed;
        return receipt;
    }

    [[nodiscard]] QInteger dot(const QTernaryVector& rhs) const {
        return interaction(rhs).net_alignment();
    }

    [[nodiscard]] std::vector<std::uint8_t> pack_base243() const {
        std::vector<std::uint8_t> packed((size_ + 4U) / 5U, 0U);
        for (std::size_t group = 0; group < packed.size(); ++group) {
            std::uint16_t value = 0;
            std::uint16_t scale = 1;
            for (std::size_t offset = 0; offset < 5; ++offset) {
                const std::size_t index = group * 5U + offset;
                std::uint16_t digit = 0;
                if (index < size_) {
                    const QPolarity polarity = get(index);
                    digit = polarity == QPolarity::Positive ? 1U : polarity == QPolarity::Negative ? 2U : 0U;
                }
                value = static_cast<std::uint16_t>(value + digit * scale);
                scale = static_cast<std::uint16_t>(scale * 3U);
            }
            packed[group] = static_cast<std::uint8_t>(value);
        }
        return packed;
    }

    [[nodiscard]] static QTernaryVector from_base243(std::span<const std::uint8_t> packed, std::size_t size) {
        if (packed.size() != (size + 4U) / 5U) throw QMathError("invalid base-243 ternary payload length");
        QTernaryVector result(size);
        for (std::size_t group = 0; group < packed.size(); ++group) {
            if (packed[group] >= 243U) throw QMathError("invalid base-243 ternary byte");
            std::uint16_t value = packed[group];
            for (std::size_t offset = 0; offset < 5; ++offset) {
                const std::uint16_t digit = static_cast<std::uint16_t>(value % 3U);
                value = static_cast<std::uint16_t>(value / 3U);
                const std::size_t index = group * 5U + offset;
                if (index >= size) {
                    if (digit != 0) throw QMathError("noncanonical base-243 ternary padding");
                    continue;
                }
                result.set(index, digit == 1U ? QPolarity::Positive : digit == 2U ? QPolarity::Negative : QPolarity::Neutral);
            }
        }
        return result;
    }

    [[nodiscard]] std::string canonical() const {
        static constexpr char hex[] = "0123456789abcdef";
        const std::vector<std::uint8_t> packed = pack_base243();
        std::string out = "qtri1:" + std::to_string(size_) + ':';
        out.reserve(out.size() + packed.size() * 2U);
        for (const std::uint8_t byte : packed) {
            out.push_back(hex[byte >> 4U]);
            out.push_back(hex[byte & 0x0fU]);
        }
        return out;
    }

    friend bool operator==(const QTernaryVector&, const QTernaryVector&) = default;

private:
    void check_index(std::size_t index) const {
        if (index >= size_) throw QMathError("ternary coordinate out of range");
    }

    void require_same_size(const QTernaryVector& rhs) const {
        if (size_ != rhs.size_) throw QMathError("ternary vectors have different dimensions");
    }

    std::size_t size_{0};
    std::vector<std::uint64_t> positive_{};
    std::vector<std::uint64_t> negative_{};
};

class QSignedChannels {
public:
    explicit QSignedChannels(QSignedDomain domain = QSignedDomain::Mathematical) : domain_(domain) {}

    [[nodiscard]] QSignedDomain domain() const noexcept { return domain_; }
    [[nodiscard]] const QRational& positive() const noexcept { return positive_; }
    [[nodiscard]] const QRational& neutral() const noexcept { return neutral_; }
    [[nodiscard]] const QRational& negative() const noexcept { return negative_; }

    void add(QPolarity polarity, QRational magnitude = QRational(1)) {
        require_nonnegative(magnitude);
        if (magnitude.is_zero()) return;
        if (polarity == QPolarity::Positive) positive_ = positive_ + magnitude;
        else if (polarity == QPolarity::Negative) negative_ = negative_ + magnitude;
        else if (polarity == QPolarity::Neutral) neutral_ = neutral_ + magnitude;
        else throw QMathError("invalid signed polarity");
    }

    void resolve_neutral(QPolarity polarity, QRational magnitude) {
        if (polarity == QPolarity::Neutral) throw QMathError("neutral resolution requires positive or negative polarity");
        require_nonnegative(magnitude);
        if (neutral_ < magnitude) throw QMathError("neutral resolution exceeds unresolved magnitude");
        neutral_ = neutral_ - magnitude;
        add(polarity, std::move(magnitude));
    }

    [[nodiscard]] QRational net() const { return positive_ - negative_; }
    [[nodiscard]] QRational active() const { return positive_ + negative_; }
    [[nodiscard]] QRational total() const { return positive_ + negative_ + neutral_; }
    [[nodiscard]] QRational canceled() const { return positive_ < negative_ ? positive_ : negative_; }

    [[nodiscard]] QSignedChannels negated() const {
        QSignedChannels result(domain_);
        result.positive_ = negative_;
        result.negative_ = positive_;
        result.neutral_ = neutral_;
        return result;
    }

    [[nodiscard]] std::string canonical() const {
        return "qchannels1:" + std::to_string(static_cast<unsigned>(domain_)) + ':' + positive_.canonical() + ':' +
               neutral_.canonical() + ':' + negative_.canonical();
    }

private:
    static void require_nonnegative(const QRational& value) {
        if (value < QRational(0)) throw QMathError("signed channel magnitude cannot be negative");
    }

    QSignedDomain domain_{QSignedDomain::Mathematical};
    QRational positive_{0};
    QRational neutral_{0};
    QRational negative_{0};
};

struct QSignedProjectionReceipt {
    QSignedDomain domain{QSignedDomain::Mathematical};
    std::size_t source_terms{0};
    std::size_t surviving_terms{0};
    std::size_t cancellation_sites{0};
    QRational positive_weight{0};
    QRational neutral_weight{0};
    QRational negative_weight{0};
    QRational canceled_weight{0};

    [[nodiscard]] QRational net_weight() const { return positive_weight - negative_weight; }
    [[nodiscard]] QRational active_weight() const { return positive_weight + negative_weight; }

    [[nodiscard]] std::string canonical() const {
        return "qprojection1:" + std::to_string(static_cast<unsigned>(domain)) + ':' + std::to_string(source_terms) + ':' +
               std::to_string(surviving_terms) + ':' + std::to_string(cancellation_sites) + ':' +
               positive_weight.canonical() + ':' + neutral_weight.canonical() + ':' + negative_weight.canonical() + ':' +
               canceled_weight.canonical();
    }
};

struct QSignedProjection {
    QMathExpr expression{};
    QSignedProjectionReceipt receipt{};
};

class QSignedExpression {
public:
    explicit QSignedExpression(QMathType type, QSignedDomain domain = QSignedDomain::Mathematical)
        : type_(std::move(type)), domain_(domain) {}

    [[nodiscard]] const QMathType& type() const noexcept { return type_; }
    [[nodiscard]] QSignedDomain domain() const noexcept { return domain_; }
    [[nodiscard]] std::size_t term_count() const noexcept { return terms_.size(); }
    [[nodiscard]] bool empty() const noexcept { return terms_.empty(); }

    void add(QPolarity polarity, const QMathExpr& basis, QRational magnitude = QRational(1)) {
        if (!basis) throw QMathError("null signed-expression basis");
        if (basis->type != type_) throw QMathError("signed-expression basis type mismatch");
        if (magnitude < QRational(0)) throw QMathError("signed-expression magnitude cannot be negative");
        if (magnitude.is_zero()) return;
        auto found = terms_.find(basis->canonical);
        if (found == terms_.end()) {
            QSignedExpressionTerm term;
            term.basis = basis;
            term.channels = QSignedChannels(domain_);
            found = terms_.emplace(basis->canonical, std::move(term)).first;
        }
        found->second.channels.add(polarity, std::move(magnitude));
    }

    [[nodiscard]] QSignedExpression negated() const {
        QSignedExpression result(type_, domain_);
        for (const auto& item : terms_) {
            const QSignedChannels& channels = item.second.channels;
            result.add(QPolarity::Positive, item.second.basis, channels.negative());
            result.add(QPolarity::Negative, item.second.basis, channels.positive());
            result.add(QPolarity::Neutral, item.second.basis, channels.neutral());
        }
        return result;
    }

    [[nodiscard]] QSignedProjection project(QMathArena& arena) const {
        QSignedProjection result;
        result.receipt.domain = domain_;
        result.receipt.source_terms = terms_.size();
        std::vector<QMathExpr> projected;
        projected.reserve(terms_.size());
        for (const auto& item : terms_) {
            const QSignedExpressionTerm& term = item.second;
            const QSignedChannels& channels = term.channels;
            result.receipt.positive_weight = result.receipt.positive_weight + channels.positive();
            result.receipt.neutral_weight = result.receipt.neutral_weight + channels.neutral();
            result.receipt.negative_weight = result.receipt.negative_weight + channels.negative();
            const QRational canceled = channels.canceled();
            result.receipt.canceled_weight = result.receipt.canceled_weight + canceled;
            if (!canceled.is_zero()) ++result.receipt.cancellation_sites;
            const QRational net = channels.net();
            if (net.is_zero()) continue;
            ++result.receipt.surviving_terms;
            if (net == QRational(1)) projected.push_back(term.basis);
            else if (net == QRational(-1)) projected.push_back(arena.negate(term.basis));
            else projected.push_back(arena.multiply({arena.rational(net), term.basis}));
        }
        result.expression = projected.empty() ? arena.zero(type_) : arena.add(projected);
        return result;
    }

    [[nodiscard]] std::string canonical() const {
        std::vector<std::string> keys;
        keys.reserve(terms_.size());
        for (const auto& item : terms_) keys.push_back(item.first);
        std::sort(keys.begin(), keys.end());
        std::string out = "qsigned-expr1:" + std::to_string(static_cast<unsigned>(domain_)) + ':' + type_.canonical() + '(';
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (i != 0) out += ';';
            const auto found = terms_.find(keys[i]);
            out += keys[i] + '=' + found->second.channels.canonical();
        }
        out += ')';
        return out;
    }

private:
    struct QSignedExpressionTerm {
        QMathExpr basis{};
        QSignedChannels channels{};
    };

    QMathType type_{};
    QSignedDomain domain_{QSignedDomain::Mathematical};
    std::unordered_map<std::string, QSignedExpressionTerm> terms_{};
};

struct QSignedBasis3 {
    QMathExpr negative{};
    QMathExpr neutral{};
    QMathExpr positive{};
};

class QSignedLocalSpace {
public:
    [[nodiscard]] static QMathType state_type(std::size_t local_dimension) {
        require_dimension(local_dimension);
        QMathType type;
        type.scalar = QMathScalar::Complex;
        type.space = QMathSpace::State;
        type.shape = {local_dimension};
        return type;
    }

    [[nodiscard]] static QMathType operator_type(std::size_t local_dimension) {
        require_dimension(local_dimension);
        QMathType type;
        type.scalar = QMathScalar::Complex;
        type.space = QMathSpace::Operator;
        type.shape = {local_dimension, local_dimension};
        return type;
    }

    [[nodiscard]] static QSignedBasis3 basis3(QMathArena& arena, std::string prefix) {
        if (prefix.empty()) throw QMathError("empty signed basis prefix");
        const QMathType type = state_type(3);
        return QSignedBasis3{
            arena.symbol(prefix + "[-1]", type, false),
            arena.symbol(prefix + "[0]", type, false),
            arena.symbol(prefix + "[+1]", type, false),
        };
    }

private:
    static void require_dimension(std::size_t local_dimension) {
        if (local_dimension < 2) throw QMathError("local quantum dimension must be at least two");
    }
};

class QSignedOperatorProgram {
public:
    explicit QSignedOperatorProgram(QMathType operator_type)
        : expression_(checked_operator_type(std::move(operator_type)), QSignedDomain::QuantumOperator) {}

    explicit QSignedOperatorProgram(std::size_t local_dimension)
        : QSignedOperatorProgram(QSignedLocalSpace::operator_type(local_dimension)) {}

    void add_generator(QPolarity polarity, const QMathExpr& generator, QRational magnitude = QRational(1)) {
        expression_.add(polarity, generator, std::move(magnitude));
    }

    [[nodiscard]] QSignedProjection compile(QMathArena& arena) const {
        return expression_.project(arena);
    }

    [[nodiscard]] QMathExpr apply(QMathArena& arena, const QMathExpr& state) const {
        if (!state || state->type.space != QMathSpace::State) throw QMathError("signed operator application requires state rhs");
        const QMathType& op_type = expression_.type();
        if (op_type.shape.size() == 2 && state->type.shape.size() == 1 &&
            (op_type.shape[0] != op_type.shape[1] || op_type.shape[1] != state->type.shape[0])) {
            throw QMathError("signed operator/state local dimension mismatch");
        }
        return arena.apply(compile(arena).expression, state);
    }

    [[nodiscard]] std::string canonical() const { return expression_.canonical(); }

private:
    static QMathType checked_operator_type(QMathType type) {
        if (type.space != QMathSpace::Operator) throw QMathError("signed operator program requires operator type");
        return type;
    }

    QSignedExpression expression_;
};

}  // namespace qubit

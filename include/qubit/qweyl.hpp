#pragma once

#include "qubit/qplan.hpp"
#include "qubit/qsigned.hpp"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace qubit {

class QWeylSpace {
public:
    explicit QWeylSpace(std::vector<std::uint32_t> dimensions) : dimensions_(std::move(dimensions)) {
        if (dimensions_.empty()) throw QMathError("Weyl space requires at least one local site");
        for (const std::uint32_t dimension : dimensions_) {
            if (dimension < 2U) throw QMathError("Weyl local dimension must be at least two");
        }
    }

    QWeylSpace(std::initializer_list<std::uint32_t> dimensions)
        : QWeylSpace(std::vector<std::uint32_t>(dimensions)) {}

    [[nodiscard]] std::size_t site_count() const noexcept { return dimensions_.size(); }
    [[nodiscard]] std::uint32_t dimension(std::size_t site) const {
        if (site >= dimensions_.size()) throw QMathError("Weyl site out of range");
        return dimensions_[site];
    }
    [[nodiscard]] const std::vector<std::uint32_t>& dimensions() const noexcept { return dimensions_; }

    [[nodiscard]] std::size_t dense_dimension() const {
        std::size_t result = 1U;
        for (const std::uint32_t dimension : dimensions_) {
            if (result > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dimension)) {
                throw QMathError("Weyl dense dimension exceeds size_t");
            }
            result *= static_cast<std::size_t>(dimension);
        }
        return result;
    }

    [[nodiscard]] std::string canonical() const {
        std::string out = "qweyl-space1:";
        for (std::size_t i = 0; i < dimensions_.size(); ++i) {
            if (i != 0U) out += 'x';
            out += std::to_string(dimensions_[i]);
        }
        return out;
    }

    friend bool operator==(const QWeylSpace&, const QWeylSpace&) = default;

private:
    std::vector<std::uint32_t> dimensions_{};
};

struct QWeylExponent {
    std::uint32_t shift{0U};
    std::uint32_t clock{0U};

    [[nodiscard]] bool identity() const noexcept { return shift == 0U && clock == 0U; }
    friend bool operator==(const QWeylExponent&, const QWeylExponent&) = default;
};

class QWeylOperator {
public:
    explicit QWeylOperator(QWeylSpace space)
        : space_(std::move(space)), exponents_(space_.site_count()) {}

    QWeylOperator(QWeylSpace space, QRational phase_turns, std::vector<QWeylExponent> exponents)
        : space_(std::move(space)), phase_turns_(mod_one(std::move(phase_turns))), exponents_(std::move(exponents)) {
        if (exponents_.size() != space_.site_count()) throw QMathError("Weyl exponent vector does not match local space");
        for (std::size_t site = 0; site < exponents_.size(); ++site) {
            const std::uint32_t dimension = space_.dimension(site);
            exponents_[site].shift %= dimension;
            exponents_[site].clock %= dimension;
        }
    }

    [[nodiscard]] static QWeylOperator identity(const QWeylSpace& space) { return QWeylOperator(space); }

    [[nodiscard]] static QWeylOperator local(
        const QWeylSpace& space,
        std::size_t site,
        std::int64_t shift,
        std::int64_t clock,
        QRational phase_turns = QRational(0)) {
        if (site >= space.site_count()) throw QMathError("Weyl site out of range");
        std::vector<QWeylExponent> exponents(space.site_count());
        const std::uint32_t dimension = space.dimension(site);
        exponents[site].shift = reduce_exponent(shift, dimension);
        exponents[site].clock = reduce_exponent(clock, dimension);
        return QWeylOperator(space, std::move(phase_turns), std::move(exponents));
    }

    [[nodiscard]] static QWeylOperator from_ternary(const QTernaryVector& shift, const QTernaryVector& clock) {
        if (shift.size() != clock.size() || shift.size() == 0U) {
            throw QMathError("ternary Weyl exponent vectors must be nonempty and equal length");
        }
        QWeylSpace space(std::vector<std::uint32_t>(shift.size(), 3U));
        std::vector<QWeylExponent> exponents(shift.size());
        for (std::size_t site = 0; site < shift.size(); ++site) {
            exponents[site].shift = ternary_exponent(shift.get(site));
            exponents[site].clock = ternary_exponent(clock.get(site));
        }
        return QWeylOperator(std::move(space), QRational(0), std::move(exponents));
    }

    [[nodiscard]] const QWeylSpace& space() const noexcept { return space_; }
    [[nodiscard]] const QRational& phase_turns() const noexcept { return phase_turns_; }
    [[nodiscard]] const std::vector<QWeylExponent>& exponents() const noexcept { return exponents_; }

    [[nodiscard]] std::size_t support_size() const noexcept {
        std::size_t count = 0U;
        for (const QWeylExponent& exponent : exponents_) if (!exponent.identity()) ++count;
        return count;
    }

    [[nodiscard]] bool identity_up_to_phase() const noexcept { return support_size() == 0U; }
    [[nodiscard]] bool identity_exact() const noexcept { return identity_up_to_phase() && phase_turns_.is_zero(); }

    [[nodiscard]] QWeylOperator multiplied(const QWeylOperator& rhs) const {
        require_same_space(rhs);
        QRational phase = phase_turns_ + rhs.phase_turns_;
        std::vector<QWeylExponent> result(exponents_.size());
        for (std::size_t site = 0; site < exponents_.size(); ++site) {
            const std::uint32_t dimension = space_.dimension(site);
            phase = phase + root_turns(exponents_[site].clock, rhs.exponents_[site].shift, dimension);
            result[site].shift = add_mod(exponents_[site].shift, rhs.exponents_[site].shift, dimension);
            result[site].clock = add_mod(exponents_[site].clock, rhs.exponents_[site].clock, dimension);
        }
        return QWeylOperator(space_, std::move(phase), std::move(result));
    }

    [[nodiscard]] QWeylOperator inverse() const {
        QRational phase = -phase_turns_;
        std::vector<QWeylExponent> result(exponents_.size());
        for (std::size_t site = 0; site < exponents_.size(); ++site) {
            const std::uint32_t dimension = space_.dimension(site);
            phase = phase + root_turns(exponents_[site].shift, exponents_[site].clock, dimension);
            result[site].shift = negate_mod(exponents_[site].shift, dimension);
            result[site].clock = negate_mod(exponents_[site].clock, dimension);
        }
        return QWeylOperator(space_, std::move(phase), std::move(result));
    }

    [[nodiscard]] QWeylOperator power(std::int64_t exponent) const {
        if (exponent == 0) return identity(space_);
        const bool invert = exponent < 0;
        std::uint64_t count = magnitude(exponent);
        QWeylOperator factor = invert ? inverse() : *this;
        QWeylOperator result = identity(space_);
        while (count != 0U) {
            if ((count & 1U) != 0U) result = result.multiplied(factor);
            count >>= 1U;
            if (count != 0U) factor = factor.multiplied(factor);
        }
        return result;
    }

    [[nodiscard]] QRational commutation_turns(const QWeylOperator& rhs) const {
        require_same_space(rhs);
        QRational phase(0);
        for (std::size_t site = 0; site < exponents_.size(); ++site) {
            const QInteger left = QInteger(static_cast<std::int64_t>(exponents_[site].clock)) *
                                  QInteger(static_cast<std::int64_t>(rhs.exponents_[site].shift));
            const QInteger right = QInteger(static_cast<std::int64_t>(rhs.exponents_[site].clock)) *
                                   QInteger(static_cast<std::int64_t>(exponents_[site].shift));
            phase = phase + QRational(left - right, QInteger(static_cast<std::int64_t>(space_.dimension(site))));
        }
        return mod_one(std::move(phase));
    }

    [[nodiscard]] bool commutes_with(const QWeylOperator& rhs) const { return commutation_turns(rhs).is_zero(); }

    [[nodiscard]] bool equivalent_up_to_global_phase(const QWeylOperator& rhs) const {
        require_same_space(rhs);
        return exponents_ == rhs.exponents_;
    }

    [[nodiscard]] QTernaryVector ternary_shift() const {
        require_ternary_space();
        QTernaryVector result(exponents_.size());
        for (std::size_t site = 0; site < exponents_.size(); ++site) result.set(site, ternary_polarity(exponents_[site].shift));
        return result;
    }

    [[nodiscard]] QTernaryVector ternary_clock() const {
        require_ternary_space();
        QTernaryVector result(exponents_.size());
        for (std::size_t site = 0; site < exponents_.size(); ++site) result.set(site, ternary_polarity(exponents_[site].clock));
        return result;
    }

    [[nodiscard]] std::string canonical() const {
        std::string out = "qweyl1:" + space_.canonical() + ':' + phase_turns_.canonical() + ':';
        for (std::size_t site = 0; site < exponents_.size(); ++site) {
            if (site != 0U) out += ';';
            out += std::to_string(exponents_[site].shift) + ',' + std::to_string(exponents_[site].clock);
        }
        return out;
    }

    friend bool operator==(const QWeylOperator&, const QWeylOperator&) = default;

    [[nodiscard]] static QRational normalize_turns(QRational value) { return mod_one(std::move(value)); }

private:
    static QRational mod_one(QRational value) {
        QInteger remainder = value.numerator() % value.denominator();
        if (remainder.is_negative()) remainder = remainder + value.denominator();
        return QRational(std::move(remainder), value.denominator());
    }

    static std::uint32_t reduce_exponent(std::int64_t value, std::uint32_t dimension) noexcept {
        const std::int64_t modulus = static_cast<std::int64_t>(dimension);
        std::int64_t reduced = value % modulus;
        if (reduced < 0) reduced += modulus;
        return static_cast<std::uint32_t>(reduced);
    }

    static std::uint32_t add_mod(std::uint32_t lhs, std::uint32_t rhs, std::uint32_t dimension) noexcept {
        return static_cast<std::uint32_t>((static_cast<std::uint64_t>(lhs) + rhs) % dimension);
    }

    static std::uint32_t negate_mod(std::uint32_t value, std::uint32_t dimension) noexcept {
        return value == 0U ? 0U : dimension - value;
    }

    static QRational root_turns(std::uint32_t left, std::uint32_t right, std::uint32_t dimension) {
        const QInteger numerator = QInteger(static_cast<std::int64_t>(left)) * QInteger(static_cast<std::int64_t>(right));
        return QRational(numerator, QInteger(static_cast<std::int64_t>(dimension)));
    }

    static std::uint64_t magnitude(std::int64_t value) noexcept {
        if (value >= 0) return static_cast<std::uint64_t>(value);
        return static_cast<std::uint64_t>(-(value + 1)) + 1U;
    }

    static std::uint32_t ternary_exponent(QPolarity polarity) {
        if (polarity == QPolarity::Positive) return 1U;
        if (polarity == QPolarity::Negative) return 2U;
        if (polarity == QPolarity::Neutral) return 0U;
        throw QMathError("invalid ternary Weyl polarity");
    }

    static QPolarity ternary_polarity(std::uint32_t exponent) {
        if (exponent == 0U) return QPolarity::Neutral;
        if (exponent == 1U) return QPolarity::Positive;
        if (exponent == 2U) return QPolarity::Negative;
        throw QMathError("non-ternary Weyl exponent cannot map to polarity");
    }

    void require_same_space(const QWeylOperator& rhs) const {
        if (space_ != rhs.space_) throw QMathError("Weyl operators belong to different local spaces");
    }

    void require_ternary_space() const {
        for (const std::uint32_t dimension : space_.dimensions()) {
            if (dimension != 3U) throw QMathError("Weyl operator is not entirely ternary");
        }
    }

    QWeylSpace space_;
    QRational phase_turns_{0};
    std::vector<QWeylExponent> exponents_{};
};

struct QWeylQubitLowering {
    std::vector<Operation> operations{};
    QRational global_phase_turns{0};

    [[nodiscard]] bool phase_free() const noexcept { return global_phase_turns.is_zero(); }
};

[[nodiscard]] inline QWeylQubitLowering lower_weyl_qubits(const QWeylOperator& value) {
    for (const std::uint32_t dimension : value.space().dimensions()) {
        if (dimension != 2U) throw QMathError("Weyl qubit lowering requires local dimension two on every site");
    }
    if (value.space().site_count() > static_cast<std::size_t>(std::numeric_limits<QubitId>::max()) + 1ULL) {
        throw QMathError("Weyl qubit lowering exceeds QubitId range");
    }

    QWeylQubitLowering result;
    result.global_phase_turns = value.phase_turns();
    result.operations.reserve(value.support_size());
    for (std::size_t site = 0; site < value.exponents().size(); ++site) {
        const QWeylExponent exponent = value.exponents()[site];
        if (exponent.identity()) continue;
        Operation operation;
        operation.first = static_cast<QubitId>(site);
        if (exponent.shift == 1U && exponent.clock == 0U) operation.code = OperationCode::X;
        else if (exponent.shift == 0U && exponent.clock == 1U) operation.code = OperationCode::Z;
        else if (exponent.shift == 1U && exponent.clock == 1U) {
            operation.code = OperationCode::Y;
            result.global_phase_turns = QWeylOperator::normalize_turns(result.global_phase_turns - QRational(1, 4));
        } else {
            throw QMathError("invalid qubit Weyl exponent");
        }
        result.operations.push_back(operation);
    }
    return result;
}

struct QWeylMathProjection {
    QMathExpr expression{};
    QRational global_phase_turns{0};
};

[[nodiscard]] inline QWeylMathProjection project_weyl_qmath(const QWeylOperator& value, QMathArena& arena) {
    QMathExpr global;
    for (std::size_t site = 0; site < value.space().site_count(); ++site) {
        const std::uint32_t dimension = value.space().dimension(site);
        const QMathType type = QSignedLocalSpace::operator_type(dimension);
        const QWeylExponent exponent = value.exponents()[site];
        const std::string suffix = "[" + std::to_string(site) + "," + std::to_string(dimension) + "]";
        QMathExpr local;
        if (exponent.identity()) {
            local = arena.symbol("I" + suffix, type, false);
        } else {
            std::vector<QMathExpr> factors;
            if (exponent.shift != 0U) {
                const QMathExpr x = arena.symbol("X" + suffix, type, false);
                factors.push_back(exponent.shift == 1U ? x : arena.power(x, QRational(static_cast<std::int64_t>(exponent.shift))));
            }
            if (exponent.clock != 0U) {
                const QMathExpr z = arena.symbol("Z" + suffix, type, false);
                factors.push_back(exponent.clock == 1U ? z : arena.power(z, QRational(static_cast<std::int64_t>(exponent.clock))));
            }
            local = factors.size() == 1U ? factors.front() : arena.multiply(factors);
        }
        global = global ? arena.tensor_product(global, local) : local;
    }
    return QWeylMathProjection{global, value.phase_turns()};
}

struct QWeylCircuitReceipt {
    std::size_t source_steps{0U};
    std::size_t positive_steps{0U};
    std::size_t negative_steps{0U};
    std::size_t unresolved_steps{0U};
    std::size_t source_support_terms{0U};
    std::size_t reduced_support_terms{0U};
    std::size_t local_cancellations{0U};
    std::size_t local_fusions{0U};
    bool ready{false};
    QRational global_phase_turns{0};
};

struct QWeylCircuitCompilation {
    std::optional<QWeylOperator> value{};
    QWeylCircuitReceipt receipt{};
};

class QSignedWeylCircuit {
public:
    explicit QSignedWeylCircuit(QWeylSpace space) : space_(std::move(space)) {}

    [[nodiscard]] const QWeylSpace& space() const noexcept { return space_; }
    [[nodiscard]] std::size_t step_count() const noexcept { return steps_.size(); }

    void append(QPolarity polarity, QWeylOperator value) {
        if (value.space() != space_) throw QMathError("signed Weyl circuit step belongs to a different local space");
        steps_.push_back(Step{polarity, std::move(value)});
    }

    void resolve(std::size_t index, QPolarity polarity) {
        if (index >= steps_.size()) throw QMathError("signed Weyl circuit step out of range");
        if (steps_[index].polarity != QPolarity::Neutral) throw QMathError("only unresolved Weyl steps may be resolved");
        if (polarity == QPolarity::Neutral) throw QMathError("Weyl resolution requires positive or negative polarity");
        steps_[index].polarity = polarity;
    }

    [[nodiscard]] QWeylCircuitCompilation compile() const {
        QWeylCircuitCompilation result;
        result.receipt.source_steps = steps_.size();
        for (const Step& step : steps_) {
            result.receipt.source_support_terms += step.value.support_size();
            if (step.polarity == QPolarity::Positive) ++result.receipt.positive_steps;
            else if (step.polarity == QPolarity::Negative) ++result.receipt.negative_steps;
            else if (step.polarity == QPolarity::Neutral) ++result.receipt.unresolved_steps;
            else throw QMathError("invalid signed Weyl circuit polarity");
        }
        if (result.receipt.unresolved_steps != 0U) return result;

        QWeylOperator accumulated = QWeylOperator::identity(space_);
        for (const Step& step : steps_) {
            const QWeylOperator oriented = step.polarity == QPolarity::Positive ? step.value : step.value.inverse();
            const std::vector<QWeylExponent> before = accumulated.exponents();
            QWeylOperator next = accumulated.multiplied(oriented);
            for (std::size_t site = 0; site < before.size(); ++site) {
                const bool had = !before[site].identity();
                const bool incoming = !oriented.exponents()[site].identity();
                if (!had || !incoming) continue;
                if (next.exponents()[site].identity()) ++result.receipt.local_cancellations;
                else ++result.receipt.local_fusions;
            }
            accumulated = std::move(next);
        }
        result.receipt.ready = true;
        result.receipt.reduced_support_terms = accumulated.support_size();
        result.receipt.global_phase_turns = accumulated.phase_turns();
        result.value = std::move(accumulated);
        return result;
    }

    [[nodiscard]] std::string canonical() const {
        std::string out = "qweyl-circuit1:" + space_.canonical() + '(';
        for (std::size_t i = 0; i < steps_.size(); ++i) {
            if (i != 0U) out += ';';
            out += std::to_string(static_cast<int>(steps_[i].polarity)) + ':' + steps_[i].value.canonical();
        }
        out += ')';
        return out;
    }

private:
    struct Step {
        QPolarity polarity{QPolarity::Neutral};
        QWeylOperator value;
    };

    QWeylSpace space_;
    std::vector<Step> steps_{};
};

}  // namespace qubit

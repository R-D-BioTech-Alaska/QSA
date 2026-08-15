#pragma once

#include "qubit/qexact.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace qubit {

struct QPolynomialConfig {
    std::size_t max_degree{128U};
    std::size_t max_steps{512U};
};

struct QPolynomialReceipt {
    std::string variable{};
    std::size_t degree{0U};
    std::size_t coefficients{0U};
    bool zero{true};
    bool exact{true};
    std::string canonical{};
};

struct QPolynomialRootCountReceipt {
    std::string variable{};
    QRational left{};
    QRational right{};
    std::size_t degree{0U};
    std::size_t square_free_degree{0U};
    std::size_t sturm_terms{0U};
    std::size_t distinct_roots{0U};
    bool exact{true};
    std::string polynomial_canonical{};
};

class QPolynomial {
public:
    explicit QPolynomial(
        std::string variable = "x",
        std::vector<QRational> coefficients = {},
        QPolynomialConfig config = {})
        : variable_(std::move(variable)), coefficients_(std::move(coefficients)), config_(config) {
        if (variable_.empty() || config_.max_steps == 0U) {
            throw QMathError("polynomial configuration or variable identity is invalid");
        }
        normalize();
        enforce_degree();
    }

    [[nodiscard]] const std::string& variable() const noexcept { return variable_; }
    [[nodiscard]] const std::vector<QRational>& coefficients() const noexcept { return coefficients_; }
    [[nodiscard]] const QPolynomialConfig& config() const noexcept { return config_; }
    [[nodiscard]] bool is_zero() const noexcept { return coefficients_.empty(); }
    [[nodiscard]] bool is_constant() const noexcept { return coefficients_.size() <= 1U; }
    [[nodiscard]] std::size_t degree() const noexcept {
        return coefficients_.empty() ? 0U : coefficients_.size() - 1U;
    }

    [[nodiscard]] QRational coefficient(std::size_t power) const {
        return power < coefficients_.size() ? coefficients_[power] : QRational(0);
    }

    [[nodiscard]] const QRational& leading() const {
        if (is_zero()) throw QMathError("zero polynomial has no leading coefficient");
        return coefficients_.back();
    }

    [[nodiscard]] std::string canonical() const {
        std::string result = "poly:" + variable_ + ":[";
        if (coefficients_.empty()) {
            result += "0";
        } else {
            for (std::size_t index = 0U; index < coefficients_.size(); ++index) {
                if (index != 0U) result += ',';
                result += coefficients_[index].canonical();
            }
        }
        result += ']';
        return result;
    }

    [[nodiscard]] QPolynomialReceipt receipt() const {
        return QPolynomialReceipt{
            variable_,
            degree(),
            coefficients_.size(),
            is_zero(),
            true,
            canonical(),
        };
    }

    [[nodiscard]] QRational evaluate(const QRational& value) const {
        QRational result(0);
        for (std::size_t index = coefficients_.size(); index-- > 0U;) {
            result = result * value + coefficients_[index];
        }
        return result;
    }

    [[nodiscard]] QPolynomial derivative() const {
        if (coefficients_.size() <= 1U) return QPolynomial(variable_, {}, config_);
        std::vector<QRational> result(coefficients_.size() - 1U, QRational(0));
        for (std::size_t power = 1U; power < coefficients_.size(); ++power) {
            if (power > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
                throw QMathError("polynomial derivative power exceeds exact integer scale");
            }
            result[power - 1U] = coefficients_[power] *
                QRational(static_cast<std::int64_t>(power));
        }
        return QPolynomial(variable_, std::move(result), config_);
    }

    [[nodiscard]] QPolynomial negated() const {
        std::vector<QRational> result = coefficients_;
        for (QRational& value : result) value = -value;
        return QPolynomial(variable_, std::move(result), config_);
    }

    [[nodiscard]] QPolynomial monic() const {
        if (is_zero()) return *this;
        const QRational scale = leading();
        std::vector<QRational> result = coefficients_;
        for (QRational& value : result) value = value / scale;
        return QPolynomial(variable_, std::move(result), config_);
    }

    [[nodiscard]] std::pair<QPolynomial, QPolynomial> divmod(const QPolynomial& divisor) const {
        require_same_variable(divisor);
        if (divisor.is_zero()) throw QMathError("polynomial division by zero");
        const QPolynomialConfig bounded = restrictive(config_, divisor.config_);
        if (is_zero() || degree() < divisor.degree()) {
            return {QPolynomial(variable_, {}, bounded), QPolynomial(variable_, coefficients_, bounded)};
        }

        std::vector<QRational> quotient(degree() - divisor.degree() + 1U, QRational(0));
        QPolynomial remainder(variable_, coefficients_, bounded);
        std::size_t steps = 0U;
        while (!remainder.is_zero() && remainder.degree() >= divisor.degree()) {
            if (steps++ >= bounded.max_steps) {
                throw QMathError("polynomial division exceeded configured step cap");
            }
            const std::size_t power = remainder.degree() - divisor.degree();
            const QRational scale = remainder.leading() / divisor.leading();
            quotient[power] = quotient[power] + scale;
            std::vector<QRational> next = remainder.coefficients_;
            for (std::size_t index = 0U; index < divisor.coefficients_.size(); ++index) {
                next[index + power] = next[index + power] - scale * divisor.coefficients_[index];
            }
            remainder = QPolynomial(variable_, std::move(next), bounded);
        }
        return {
            QPolynomial(variable_, std::move(quotient), bounded),
            std::move(remainder),
        };
    }

    [[nodiscard]] QPolynomial square_free() const {
        if (is_zero() || is_constant()) return *this;
        const QPolynomial common = gcd(*this, derivative());
        const auto division = divmod(common);
        if (!division.second.is_zero()) {
            throw QMathError("polynomial square-free division was not exact");
        }
        return division.first.monic();
    }

    [[nodiscard]] std::vector<QPolynomial> sturm_sequence() const {
        if (is_zero()) throw QMathError("zero polynomial has infinitely many roots");
        const QPolynomial square = square_free().monic();
        std::vector<QPolynomial> sequence;
        sequence.push_back(square);
        if (square.is_constant()) return sequence;
        sequence.push_back(square.derivative());
        std::size_t steps = 0U;
        while (!sequence.back().is_zero()) {
            if (steps++ >= config_.max_steps) {
                throw QMathError("Sturm sequence exceeded configured step cap");
            }
            const std::size_t count = sequence.size();
            const auto division = sequence[count - 2U].divmod(sequence[count - 1U]);
            if (division.second.is_zero()) break;
            sequence.push_back(division.second.negated());
        }
        return sequence;
    }

    [[nodiscard]] QPolynomialRootCountReceipt count_distinct_real_roots(
        const QRational& left,
        const QRational& right) const {
        if (!(left < right)) throw QMathError("polynomial root interval must be strictly increasing");
        if (is_zero()) throw QMathError("zero polynomial has infinitely many roots");
        if (evaluate(left).is_zero() || evaluate(right).is_zero()) {
            throw QMathError("polynomial root interval endpoint is an exact root");
        }
        const QPolynomial square = square_free().monic();
        if (square.is_constant()) {
            return QPolynomialRootCountReceipt{
                variable_, left, right, degree(), square.degree(), 1U, 0U, true, canonical(),
            };
        }
        const std::vector<QPolynomial> sequence = square.sturm_sequence();
        const std::size_t before = sign_variations(sequence, left);
        const std::size_t after = sign_variations(sequence, right);
        if (after > before) throw QMathError("Sturm root count violated exact variation ordering");
        return QPolynomialRootCountReceipt{
            variable_,
            left,
            right,
            degree(),
            square.degree(),
            sequence.size(),
            before - after,
            true,
            canonical(),
        };
    }

    [[nodiscard]] static QPolynomial gcd(QPolynomial left, QPolynomial right) {
        left.require_same_variable(right);
        const QPolynomialConfig bounded = restrictive(left.config_, right.config_);
        left.config_ = bounded;
        right.config_ = bounded;
        if (left.is_zero()) return right.monic();
        if (right.is_zero()) return left.monic();
        std::size_t steps = 0U;
        while (!right.is_zero()) {
            if (steps++ >= bounded.max_steps) {
                throw QMathError("polynomial gcd exceeded configured step cap");
            }
            auto division = left.divmod(right);
            left = std::move(right);
            right = std::move(division.second);
        }
        return left.monic();
    }

    friend bool operator==(const QPolynomial& left, const QPolynomial& right) noexcept {
        return left.variable_ == right.variable_ && left.coefficients_ == right.coefficients_;
    }

    friend QPolynomial operator+(const QPolynomial& left, const QPolynomial& right) {
        left.require_same_variable(right);
        const QPolynomialConfig bounded = restrictive(left.config_, right.config_);
        std::vector<QRational> result(
            std::max(left.coefficients_.size(), right.coefficients_.size()), QRational(0));
        for (std::size_t index = 0U; index < result.size(); ++index) {
            result[index] = left.coefficient(index) + right.coefficient(index);
        }
        return QPolynomial(left.variable_, std::move(result), bounded);
    }

    friend QPolynomial operator-(const QPolynomial& left, const QPolynomial& right) {
        return left + right.negated();
    }

    friend QPolynomial operator*(const QPolynomial& left, const QPolynomial& right) {
        left.require_same_variable(right);
        const QPolynomialConfig bounded = restrictive(left.config_, right.config_);
        if (left.is_zero() || right.is_zero()) return QPolynomial(left.variable_, {}, bounded);
        if (left.degree() > bounded.max_degree || right.degree() > bounded.max_degree ||
            left.degree() > bounded.max_degree - right.degree()) {
            throw QMathError("polynomial multiplication exceeds configured degree cap");
        }
        std::vector<QRational> result(
            left.coefficients_.size() + right.coefficients_.size() - 1U, QRational(0));
        for (std::size_t i = 0U; i < left.coefficients_.size(); ++i) {
            for (std::size_t j = 0U; j < right.coefficients_.size(); ++j) {
                result[i + j] = result[i + j] + left.coefficients_[i] * right.coefficients_[j];
            }
        }
        return QPolynomial(left.variable_, std::move(result), bounded);
    }

private:
    void normalize() {
        while (!coefficients_.empty() && coefficients_.back().is_zero()) coefficients_.pop_back();
    }

    void enforce_degree() const {
        if (!is_zero() && degree() > config_.max_degree) {
            throw QMathError("polynomial degree exceeds configured cap");
        }
    }

    void require_same_variable(const QPolynomial& other) const {
        if (variable_ != other.variable_) {
            throw QMathError("polynomial variable identities differ");
        }
    }

    [[nodiscard]] static QPolynomialConfig restrictive(
        const QPolynomialConfig& left,
        const QPolynomialConfig& right) noexcept {
        return {
            std::min(left.max_degree, right.max_degree),
            std::min(left.max_steps, right.max_steps),
        };
    }

    [[nodiscard]] static std::size_t sign_variations(
        const std::vector<QPolynomial>& sequence,
        const QRational& point) {
        int previous = 0;
        std::size_t variations = 0U;
        for (const QPolynomial& polynomial : sequence) {
            const QRational value = polynomial.evaluate(point);
            if (value.is_zero()) continue;
            const int sign = value < QRational(0) ? -1 : 1;
            if (previous != 0 && sign != previous) ++variations;
            previous = sign;
        }
        return variations;
    }

    std::string variable_;
    std::vector<QRational> coefficients_;
    QPolynomialConfig config_{};
};

}  // namespace qubit

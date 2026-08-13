#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qubit {

class QMathError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class QInteger {
public:
    QInteger(std::int64_t value = 0) noexcept : small_(value) {}

    [[nodiscard]] static QInteger parse(std::string_view text) {
        if (text.empty()) throw QMathError("empty integer literal");
        bool negative = false;
        std::size_t position = 0;
        if (text.front() == '+' || text.front() == '-') {
            negative = text.front() == '-';
            position = 1;
        }
        if (position == text.size()) throw QMathError("invalid integer literal");
        while (position < text.size() && text[position] == '0') ++position;
        if (position == text.size()) return QInteger{};
        for (std::size_t i = position; i < text.size(); ++i) {
            if (!std::isdigit(static_cast<unsigned char>(text[i]))) throw QMathError("invalid integer literal");
        }

        QInteger result;
        Big value;
        value.negative = negative;
        for (std::size_t end = text.size(); end > position;) {
            const std::size_t begin = end >= position + 9 ? end - 9 : position;
            std::uint32_t limb = 0;
            for (std::size_t i = begin; i < end; ++i) {
                limb = static_cast<std::uint32_t>(limb * 10U + static_cast<unsigned>(text[i] - '0'));
            }
            value.limbs.push_back(limb);
            end = begin;
        }
        normalize(value);
        result.assign_big(std::move(value));
        return result;
    }

    [[nodiscard]] bool is_small() const noexcept { return limbs_.empty(); }
    [[nodiscard]] bool is_zero() const noexcept { return is_small() ? small_ == 0 : false; }
    [[nodiscard]] bool is_one() const noexcept { return is_small() ? small_ == 1 : limbs_.size() == 1 && !negative_ && limbs_[0] == 1; }
    [[nodiscard]] bool is_negative() const noexcept { return is_small() ? small_ < 0 : negative_; }

    [[nodiscard]] bool fits_int64() const noexcept {
        return is_small();
    }

    [[nodiscard]] std::int64_t to_int64() const {
        if (!is_small()) throw QMathError("integer does not fit int64");
        return small_;
    }

    [[nodiscard]] QInteger abs() const {
        if (is_small()) {
            if (small_ == std::numeric_limits<std::int64_t>::min()) {
                QInteger result;
                result.negative_ = false;
                result.limbs_ = magnitude_limbs(small_);
                return result;
            }
            return QInteger(small_ < 0 ? -small_ : small_);
        }
        QInteger result = *this;
        result.negative_ = false;
        return result;
    }

    [[nodiscard]] std::string canonical() const {
        if (is_small()) return std::to_string(small_);
        std::string out = negative_ ? "-" : "";
        out += std::to_string(limbs_.back());
        for (std::size_t i = limbs_.size() - 1; i-- > 0;) {
            const std::string limb = std::to_string(limbs_[i]);
            out.append(9 - limb.size(), '0');
            out += limb;
        }
        return out;
    }

    friend bool operator==(const QInteger& lhs, const QInteger& rhs) noexcept {
        if (lhs.is_small() && rhs.is_small()) return lhs.small_ == rhs.small_;
        const Big left = lhs.as_big();
        const Big right = rhs.as_big();
        return left.negative == right.negative && left.limbs == right.limbs;
    }

    friend bool operator<(const QInteger& lhs, const QInteger& rhs) noexcept {
        if (lhs.is_small() && rhs.is_small()) return lhs.small_ < rhs.small_;
        const Big left = lhs.as_big();
        const Big right = rhs.as_big();
        if (left.negative != right.negative) return left.negative;
        const int cmp = compare_magnitude(left.limbs, right.limbs);
        return left.negative ? cmp > 0 : cmp < 0;
    }

    friend QInteger operator-(const QInteger& value) {
        if (value.is_small()) {
            if (value.small_ != std::numeric_limits<std::int64_t>::min()) return QInteger(-value.small_);
            QInteger result;
            result.negative_ = false;
            result.limbs_ = magnitude_limbs(value.small_);
            return result;
        }
        QInteger result = value;
        result.negative_ = !result.negative_;
        return result;
    }

    friend QInteger operator+(const QInteger& lhs, const QInteger& rhs) {
        if (lhs.is_small() && rhs.is_small()) {
            std::int64_t value = 0;
            if (checked_add(lhs.small_, rhs.small_, value)) return QInteger(value);
        }
        Big left = lhs.as_big();
        Big right = rhs.as_big();
        Big result;
        if (left.negative == right.negative) {
            result.negative = left.negative;
            result.limbs = add_magnitude(left.limbs, right.limbs);
        } else {
            const int cmp = compare_magnitude(left.limbs, right.limbs);
            if (cmp == 0) return QInteger{};
            if (cmp > 0) {
                result.negative = left.negative;
                result.limbs = subtract_magnitude(left.limbs, right.limbs);
            } else {
                result.negative = right.negative;
                result.limbs = subtract_magnitude(right.limbs, left.limbs);
            }
        }
        QInteger value;
        value.assign_big(std::move(result));
        return value;
    }

    friend QInteger operator-(const QInteger& lhs, const QInteger& rhs) {
        return lhs + (-rhs);
    }

    friend QInteger operator*(const QInteger& lhs, const QInteger& rhs) {
        if (lhs.is_zero() || rhs.is_zero()) return QInteger{};
        if (lhs.is_small() && rhs.is_small()) {
            std::int64_t value = 0;
            if (checked_multiply(lhs.small_, rhs.small_, value)) return QInteger(value);
        }
        const Big left = lhs.as_big();
        const Big right = rhs.as_big();
        Big result;
        result.negative = left.negative != right.negative;
        result.limbs.assign(left.limbs.size() + right.limbs.size(), 0U);
        for (std::size_t i = 0; i < left.limbs.size(); ++i) {
            std::uint64_t carry = 0;
            for (std::size_t j = 0; j < right.limbs.size() || carry != 0; ++j) {
                const std::uint64_t current = result.limbs[i + j] + carry +
                    (j < right.limbs.size() ? static_cast<std::uint64_t>(left.limbs[i]) * right.limbs[j] : 0ULL);
                result.limbs[i + j] = static_cast<std::uint32_t>(current % Base);
                carry = current / Base;
            }
        }
        normalize(result);
        QInteger value;
        value.assign_big(std::move(result));
        return value;
    }

    friend QInteger operator/(const QInteger& lhs, const QInteger& rhs) {
        return divmod(lhs, rhs).first;
    }

    friend QInteger operator%(const QInteger& lhs, const QInteger& rhs) {
        return divmod(lhs, rhs).second;
    }

    [[nodiscard]] static QInteger gcd(QInteger lhs, QInteger rhs) {
        lhs = lhs.abs();
        rhs = rhs.abs();
        while (!rhs.is_zero()) {
            QInteger remainder = lhs % rhs;
            lhs = std::move(rhs);
            rhs = std::move(remainder);
        }
        return lhs;
    }

private:
    static constexpr std::uint32_t Base = 1'000'000'000U;

    struct Big {
        bool negative{false};
        std::vector<std::uint32_t> limbs{};
    };

    static std::uint64_t magnitude(std::int64_t value) noexcept {
        if (value >= 0) return static_cast<std::uint64_t>(value);
        return static_cast<std::uint64_t>(-(value + 1)) + 1ULL;
    }

    static std::vector<std::uint32_t> magnitude_limbs(std::int64_t value) {
        std::uint64_t current = magnitude(value);
        std::vector<std::uint32_t> result;
        while (current != 0) {
            result.push_back(static_cast<std::uint32_t>(current % Base));
            current /= Base;
        }
        return result;
    }

    [[nodiscard]] Big as_big() const noexcept {
        if (!is_small()) return Big{negative_, limbs_};
        Big result;
        result.negative = small_ < 0;
        result.limbs = magnitude_limbs(small_);
        return result;
    }

    static void normalize(Big& value) noexcept {
        while (!value.limbs.empty() && value.limbs.back() == 0U) value.limbs.pop_back();
        if (value.limbs.empty()) value.negative = false;
    }

    void assign_big(Big value) {
        normalize(value);
        if (value.limbs.empty()) {
            small_ = 0;
            negative_ = false;
            limbs_.clear();
            return;
        }
        std::uint64_t magnitude_value = 0;
        bool fits = value.limbs.size() <= 3;
        if (fits) {
            for (std::size_t i = value.limbs.size(); i-- > 0;) {
                if (magnitude_value > (std::numeric_limits<std::uint64_t>::max() - value.limbs[i]) / Base) {
                    fits = false;
                    break;
                }
                magnitude_value = magnitude_value * Base + value.limbs[i];
            }
        }
        if (fits) {
            const std::uint64_t positive_limit = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
            const std::uint64_t negative_limit = positive_limit + 1ULL;
            if ((!value.negative && magnitude_value <= positive_limit) || (value.negative && magnitude_value <= negative_limit)) {
                if (value.negative) {
                    if (magnitude_value == negative_limit) small_ = std::numeric_limits<std::int64_t>::min();
                    else small_ = -static_cast<std::int64_t>(magnitude_value);
                } else {
                    small_ = static_cast<std::int64_t>(magnitude_value);
                }
                negative_ = false;
                limbs_.clear();
                return;
            }
        }
        small_ = 0;
        negative_ = value.negative;
        limbs_ = std::move(value.limbs);
    }

    static bool checked_add(std::int64_t lhs, std::int64_t rhs, std::int64_t& out) noexcept {
        const auto max = std::numeric_limits<std::int64_t>::max();
        const auto min = std::numeric_limits<std::int64_t>::min();
        if ((rhs > 0 && lhs > max - rhs) || (rhs < 0 && lhs < min - rhs)) return false;
        out = lhs + rhs;
        return true;
    }

    static bool checked_multiply(std::int64_t lhs, std::int64_t rhs, std::int64_t& out) noexcept {
        const auto max = std::numeric_limits<std::int64_t>::max();
        const auto min = std::numeric_limits<std::int64_t>::min();
        if (lhs == 0 || rhs == 0) {
            out = 0;
            return true;
        }
        if (lhs == -1) {
            if (rhs == min) return false;
            out = -rhs;
            return true;
        }
        if (rhs == -1) {
            if (lhs == min) return false;
            out = -lhs;
            return true;
        }
        if (lhs > 0) {
            if ((rhs > 0 && lhs > max / rhs) || (rhs < 0 && rhs < min / lhs)) return false;
        } else {
            if ((rhs > 0 && lhs < min / rhs) || (rhs < 0 && lhs < max / rhs)) return false;
        }
        out = lhs * rhs;
        return true;
    }

    static int compare_magnitude(const std::vector<std::uint32_t>& lhs, const std::vector<std::uint32_t>& rhs) noexcept {
        if (lhs.size() != rhs.size()) return lhs.size() < rhs.size() ? -1 : 1;
        for (std::size_t i = lhs.size(); i-- > 0;) {
            if (lhs[i] != rhs[i]) return lhs[i] < rhs[i] ? -1 : 1;
        }
        return 0;
    }

    static std::vector<std::uint32_t> add_magnitude(
        const std::vector<std::uint32_t>& lhs,
        const std::vector<std::uint32_t>& rhs) {
        const std::size_t size = std::max(lhs.size(), rhs.size());
        std::vector<std::uint32_t> result(size + 1, 0U);
        std::uint64_t carry = 0;
        for (std::size_t i = 0; i < size; ++i) {
            const std::uint64_t current = carry + (i < lhs.size() ? lhs[i] : 0U) + (i < rhs.size() ? rhs[i] : 0U);
            result[i] = static_cast<std::uint32_t>(current % Base);
            carry = current / Base;
        }
        if (carry != 0) result[size] = static_cast<std::uint32_t>(carry);
        else result.pop_back();
        return result;
    }

    static std::vector<std::uint32_t> subtract_magnitude(
        const std::vector<std::uint32_t>& lhs,
        const std::vector<std::uint32_t>& rhs) {
        std::vector<std::uint32_t> result(lhs.size(), 0U);
        std::int64_t borrow = 0;
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            std::int64_t current = static_cast<std::int64_t>(lhs[i]) - borrow -
                (i < rhs.size() ? static_cast<std::int64_t>(rhs[i]) : 0LL);
            if (current < 0) {
                current += Base;
                borrow = 1;
            } else {
                borrow = 0;
            }
            result[i] = static_cast<std::uint32_t>(current);
        }
        while (!result.empty() && result.back() == 0U) result.pop_back();
        return result;
    }

    static Big multiply_small(Big value, std::uint32_t multiplier) {
        if (multiplier == 0 || value.limbs.empty()) return {};
        std::uint64_t carry = 0;
        for (std::uint32_t& limb : value.limbs) {
            const std::uint64_t current = static_cast<std::uint64_t>(limb) * multiplier + carry;
            limb = static_cast<std::uint32_t>(current % Base);
            carry = current / Base;
        }
        if (carry != 0) value.limbs.push_back(static_cast<std::uint32_t>(carry));
        return value;
    }

    static std::uint32_t divide_small(Big& value, std::uint32_t divisor) {
        if (divisor == 0) throw QMathError("integer division by zero");
        std::uint64_t remainder = 0;
        for (std::size_t i = value.limbs.size(); i-- > 0;) {
            const std::uint64_t current = value.limbs[i] + remainder * Base;
            value.limbs[i] = static_cast<std::uint32_t>(current / divisor);
            remainder = current % divisor;
        }
        normalize(value);
        return static_cast<std::uint32_t>(remainder);
    }

    static Big shift_base_add(Big value, std::uint32_t limb) {
        if (value.limbs.empty()) {
            if (limb != 0) value.limbs.push_back(limb);
            return value;
        }
        value.limbs.insert(value.limbs.begin(), limb);
        return value;
    }

    static Big subtract_big(const Big& lhs, const Big& rhs) {
        if (compare_magnitude(lhs.limbs, rhs.limbs) < 0) throw QMathError("internal negative magnitude subtraction");
        Big result;
        result.limbs = subtract_magnitude(lhs.limbs, rhs.limbs);
        return result;
    }

    static std::pair<QInteger, QInteger> divmod(const QInteger& lhs, const QInteger& rhs) {
        if (rhs.is_zero()) throw QMathError("integer division by zero");
        if (lhs.is_zero()) return {QInteger{}, QInteger{}};
        Big a = lhs.abs().as_big();
        Big b = rhs.abs().as_big();
        const int cmp = compare_magnitude(a.limbs, b.limbs);
        if (cmp < 0) return {QInteger{}, lhs};
        if (cmp == 0) return {QInteger(lhs.is_negative() != rhs.is_negative() ? -1 : 1), QInteger{}};

        const std::uint32_t norm = static_cast<std::uint32_t>(Base / (static_cast<std::uint64_t>(b.limbs.back()) + 1ULL));
        a = multiply_small(std::move(a), norm);
        b = multiply_small(std::move(b), norm);

        Big quotient;
        quotient.limbs.assign(a.limbs.size(), 0U);
        Big remainder;
        for (std::size_t index = a.limbs.size(); index-- > 0;) {
            remainder = shift_base_add(std::move(remainder), a.limbs[index]);
            const std::size_t n = b.limbs.size();
            const std::uint64_t high = remainder.limbs.size() > n ? remainder.limbs[n] : 0ULL;
            const std::uint64_t next = remainder.limbs.size() >= n ? remainder.limbs[n - 1] : 0ULL;
            std::uint64_t digit = (Base * high + next) / b.limbs.back();
            if (digit >= Base) digit = Base - 1;
            Big product = multiply_small(b, static_cast<std::uint32_t>(digit));
            while (compare_magnitude(remainder.limbs, product.limbs) < 0) {
                --digit;
                product = subtract_big(product, b);
            }
            remainder = subtract_big(remainder, product);
            quotient.limbs[index] = static_cast<std::uint32_t>(digit);
        }
        normalize(quotient);
        (void)divide_small(remainder, norm);
        quotient.negative = lhs.is_negative() != rhs.is_negative() && !quotient.limbs.empty();
        remainder.negative = lhs.is_negative() && !remainder.limbs.empty();
        QInteger q;
        QInteger r;
        q.assign_big(std::move(quotient));
        r.assign_big(std::move(remainder));
        return {std::move(q), std::move(r)};
    }

    std::int64_t small_{0};
    bool negative_{false};
    std::vector<std::uint32_t> limbs_{};
};

class QRational {
public:
    QRational(std::int64_t numerator = 0, std::int64_t denominator = 1)
        : QRational(QInteger(numerator), QInteger(denominator)) {}

    QRational(QInteger numerator, QInteger denominator = QInteger(1)) {
        assign(std::move(numerator), std::move(denominator));
    }

    [[nodiscard]] static QRational parse(std::string_view text) {
        const std::size_t slash = text.find('/');
        if (slash == std::string_view::npos) return QRational(QInteger::parse(text));
        if (text.find('/', slash + 1) != std::string_view::npos) throw QMathError("invalid rational literal");
        return QRational(QInteger::parse(text.substr(0, slash)), QInteger::parse(text.substr(slash + 1)));
    }

    [[nodiscard]] const QInteger& numerator() const noexcept { return numerator_; }
    [[nodiscard]] const QInteger& denominator() const noexcept { return denominator_; }
    [[nodiscard]] bool is_zero() const noexcept { return numerator_.is_zero(); }
    [[nodiscard]] bool is_one() const noexcept { return numerator_ == denominator_; }
    [[nodiscard]] bool is_integer() const noexcept { return denominator_.is_one(); }

    [[nodiscard]] bool integer_int64(std::int64_t& value) const noexcept {
        if (!is_integer() || !numerator_.fits_int64()) return false;
        value = numerator_.to_int64();
        return true;
    }

    [[nodiscard]] std::string canonical() const {
        if (denominator_.is_one()) return numerator_.canonical();
        return numerator_.canonical() + "/" + denominator_.canonical();
    }

    friend bool operator==(const QRational&, const QRational&) = default;
    friend bool operator<(const QRational& lhs, const QRational& rhs) {
        return lhs.numerator_ * rhs.denominator_ < rhs.numerator_ * lhs.denominator_;
    }

    friend QRational operator-(const QRational& value) {
        return QRational(-value.numerator_, value.denominator_);
    }

    friend QRational operator+(const QRational& lhs, const QRational& rhs) {
        const QInteger common = QInteger::gcd(lhs.denominator_, rhs.denominator_);
        const QInteger lhs_scale = rhs.denominator_ / common;
        const QInteger rhs_scale = lhs.denominator_ / common;
        return QRational(
            lhs.numerator_ * lhs_scale + rhs.numerator_ * rhs_scale,
            lhs.denominator_ * lhs_scale);
    }

    friend QRational operator-(const QRational& lhs, const QRational& rhs) {
        return lhs + (-rhs);
    }

    friend QRational operator*(const QRational& lhs, const QRational& rhs) {
        if (lhs.is_zero() || rhs.is_zero()) return QRational{};
        const QInteger g1 = QInteger::gcd(lhs.numerator_.abs(), rhs.denominator_);
        const QInteger g2 = QInteger::gcd(rhs.numerator_.abs(), lhs.denominator_);
        return QRational(
            (lhs.numerator_ / g1) * (rhs.numerator_ / g2),
            (lhs.denominator_ / g2) * (rhs.denominator_ / g1));
    }

    friend QRational operator/(const QRational& lhs, const QRational& rhs) {
        if (rhs.is_zero()) throw QMathError("division by zero rational");
        return lhs * QRational(rhs.denominator_, rhs.numerator_);
    }

private:
    void assign(QInteger numerator, QInteger denominator) {
        if (denominator.is_zero()) throw QMathError("zero rational denominator");
        if (denominator.is_negative()) {
            numerator = -numerator;
            denominator = -denominator;
        }
        if (numerator.is_zero()) {
            numerator_ = QInteger{};
            denominator_ = QInteger(1);
            return;
        }
        const QInteger divisor = QInteger::gcd(numerator.abs(), denominator);
        numerator_ = numerator / divisor;
        denominator_ = denominator / divisor;
    }

    QInteger numerator_{};
    QInteger denominator_{1};
};

}  // namespace qubit

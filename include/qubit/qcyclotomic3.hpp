#pragma once

#include "qubit/qweyl_algebra.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace qubit {

class QCyclotomic3 {
public:
    QCyclotomic3(QRational real = QRational(0), QRational omega = QRational(0))
        : real_(std::move(real)), omega_(std::move(omega)) {}

    [[nodiscard]] static QCyclotomic3 root(std::int64_t exponent) {
        std::int64_t reduced = exponent % 3;
        if (reduced < 0) reduced += 3;
        if (reduced == 0) return QCyclotomic3(QRational(1), QRational(0));
        if (reduced == 1) return QCyclotomic3(QRational(0), QRational(1));
        return QCyclotomic3(QRational(-1), QRational(-1));
    }

    [[nodiscard]] static QCyclotomic3 from_turns(QRational turns) {
        turns = QWeylOperator::normalize_turns(std::move(turns));
        const QRational thirds = turns * QRational(3);
        std::int64_t exponent = 0;
        if (!thirds.integer_int64(exponent)) throw QMathError("qutrit phase must be an exact third of a turn");
        return root(exponent);
    }

    [[nodiscard]] const QRational& real() const noexcept { return real_; }
    [[nodiscard]] const QRational& omega() const noexcept { return omega_; }
    [[nodiscard]] bool is_zero() const noexcept { return real_.is_zero() && omega_.is_zero(); }
    [[nodiscard]] bool is_one() const noexcept { return real_.is_one() && omega_.is_zero(); }

    [[nodiscard]] QCyclotomic3 conjugate() const { return QCyclotomic3(real_ - omega_, -omega_); }
    [[nodiscard]] QRational norm() const { return real_ * real_ - real_ * omega_ + omega_ * omega_; }

    [[nodiscard]] QCyclotomic3 inverse() const {
        const QRational magnitude = norm();
        if (magnitude.is_zero()) throw QMathError("division by zero qutrit cyclotomic value");
        const QCyclotomic3 conjugated = conjugate();
        return QCyclotomic3(conjugated.real_ / magnitude, conjugated.omega_ / magnitude);
    }

    [[nodiscard]] QCyclotomic3 scaled(const QRational& value) const {
        return QCyclotomic3(real_ * value, omega_ * value);
    }

    [[nodiscard]] std::string canonical() const {
        return "qcyclotomic3:1," + real_.canonical() + ',' + omega_.canonical();
    }

    friend bool operator==(const QCyclotomic3&, const QCyclotomic3&) = default;
    friend QCyclotomic3 operator-(const QCyclotomic3& value) { return QCyclotomic3(-value.real_, -value.omega_); }
    friend QCyclotomic3 operator+(const QCyclotomic3& lhs, const QCyclotomic3& rhs) {
        return QCyclotomic3(lhs.real_ + rhs.real_, lhs.omega_ + rhs.omega_);
    }
    friend QCyclotomic3 operator-(const QCyclotomic3& lhs, const QCyclotomic3& rhs) { return lhs + (-rhs); }
    friend QCyclotomic3 operator*(const QCyclotomic3& lhs, const QCyclotomic3& rhs) {
        const QRational omega_product = lhs.omega_ * rhs.omega_;
        return QCyclotomic3(
            lhs.real_ * rhs.real_ - omega_product,
            lhs.real_ * rhs.omega_ + lhs.omega_ * rhs.real_ - omega_product);
    }
    friend QCyclotomic3 operator/(const QCyclotomic3& lhs, const QCyclotomic3& rhs) { return lhs * rhs.inverse(); }

private:
    QRational real_{0};
    QRational omega_{0};
};

struct QWeyl3Term {
    QWeylOperator basis;
    QCyclotomic3 coefficient{};
};

class QWeyl3Algebra {
public:
    explicit QWeyl3Algebra(QWeylSpace space, QWeylAlgebraConfig config = {})
        : space_(validated_space(std::move(space))), config_(config) {
        if (config_.max_basis_terms == 0U || config_.max_pair_products == 0U) {
            throw QMathError("qutrit Weyl algebra resource caps must be positive");
        }
    }

    [[nodiscard]] static QWeyl3Algebra from_projection(
        const QWeylSpace& space,
        const QWeylAlgebraProjection& projection,
        QWeylAlgebraConfig config = {}) {
        if (!projection.receipt.ready) throw QMathError("unresolved signed Weyl algebra cannot enter qutrit cyclotomic form");
        QWeyl3Algebra result(space, config);
        for (const QWeylLinearTerm& term : projection.terms) result.add(term.basis, QCyclotomic3(term.coefficient));
        return result;
    }

    [[nodiscard]] const QWeylSpace& space() const noexcept { return space_; }
    [[nodiscard]] std::size_t term_count() const noexcept { return terms_.size(); }
    [[nodiscard]] bool empty() const noexcept { return terms_.empty(); }

    void add(QWeylOperator basis, QCyclotomic3 coefficient = QCyclotomic3(QRational(1))) {
        require_space(basis);
        if (coefficient.is_zero()) return;
        coefficient = coefficient * QCyclotomic3::from_turns(basis.phase_turns());
        basis = phase_free(std::move(basis));
        const std::string key = basis.canonical();
        const auto found = terms_.find(key);
        if (found == terms_.end()) {
            if (terms_.size() >= config_.max_basis_terms) throw QMathError("qutrit Weyl algebra basis-term cap exceeded");
            terms_.emplace(key, Term{std::move(basis), std::move(coefficient)});
            return;
        }
        found->second.coefficient = found->second.coefficient + coefficient;
        if (found->second.coefficient.is_zero()) terms_.erase(found);
    }

    [[nodiscard]] QWeyl3Algebra added(const QWeyl3Algebra& rhs) const {
        require_space(rhs);
        QWeyl3Algebra result(space_, conservative_config(rhs));
        result.merge_from(*this);
        result.merge_from(rhs);
        return result;
    }

    [[nodiscard]] QWeyl3Algebra negated() const {
        QWeyl3Algebra result(space_, config_);
        for (const auto& item : terms_) result.add(item.second.basis, -item.second.coefficient);
        return result;
    }

    [[nodiscard]] QWeyl3Algebra subtracted(const QWeyl3Algebra& rhs) const { return added(rhs.negated()); }

    [[nodiscard]] QWeyl3Algebra multiplied(const QWeyl3Algebra& rhs) const {
        require_space(rhs);
        const QWeylAlgebraConfig output_config = conservative_config(rhs);
        if (!rhs.terms_.empty() && terms_.size() > output_config.max_pair_products / rhs.terms_.size()) {
            throw QMathError("qutrit Weyl algebra pair-product cap exceeded");
        }
        QWeyl3Algebra result(space_, output_config);
        for (const auto& left : terms_) {
            for (const auto& right : rhs.terms_) {
                result.add(left.second.basis.multiplied(right.second.basis), left.second.coefficient * right.second.coefficient);
            }
        }
        return result;
    }

    [[nodiscard]] QWeyl3Algebra commutator(const QWeyl3Algebra& rhs) const {
        return multiplied(rhs).subtracted(rhs.multiplied(*this));
    }

    [[nodiscard]] QWeyl3Algebra anticommutator(const QWeyl3Algebra& rhs) const {
        return multiplied(rhs).added(rhs.multiplied(*this));
    }

    [[nodiscard]] QCyclotomic3 coefficient(const QWeylOperator& phase_free_basis) const {
        require_space(phase_free_basis);
        if (!phase_free_basis.phase_turns().is_zero()) throw QMathError("qutrit Weyl coefficient lookup requires phase-free basis");
        const auto found = terms_.find(phase_free_basis.canonical());
        return found == terms_.end() ? QCyclotomic3{} : found->second.coefficient;
    }

    [[nodiscard]] std::vector<QWeyl3Term> terms() const {
        std::vector<QWeyl3Term> result;
        result.reserve(terms_.size());
        for (const auto& item : terms_) result.push_back(QWeyl3Term{item.second.basis, item.second.coefficient});
        return result;
    }

    [[nodiscard]] std::string canonical() const {
        std::string out = "qweyl3-algebra1:" + space_.canonical() + '(';
        bool first = true;
        for (const auto& item : terms_) {
            if (!first) out += ';';
            first = false;
            out += item.first + '=' + item.second.coefficient.canonical();
        }
        out += ')';
        return out;
    }

private:
    struct Term {
        QWeylOperator basis;
        QCyclotomic3 coefficient;
    };

    [[nodiscard]] static QWeylSpace validated_space(QWeylSpace space) {
        for (const std::uint32_t dimension : space.dimensions()) {
            if (dimension != 3U) throw QMathError("qutrit cyclotomic Weyl algebra requires local dimension three");
        }
        return space;
    }

    [[nodiscard]] static QWeylOperator phase_free(QWeylOperator basis) {
        return QWeylOperator(basis.space(), QRational(0), basis.exponents());
    }

    void require_space(const QWeylOperator& basis) const {
        if (basis.space() != space_) throw QMathError("qutrit Weyl basis belongs to a different local space");
    }

    void require_space(const QWeyl3Algebra& rhs) const {
        if (rhs.space_ != space_) throw QMathError("qutrit Weyl algebras belong to different local spaces");
    }

    [[nodiscard]] QWeylAlgebraConfig conservative_config(const QWeyl3Algebra& rhs) const noexcept {
        return QWeylAlgebraConfig{
            std::min(config_.max_basis_terms, rhs.config_.max_basis_terms),
            std::min(config_.max_pair_products, rhs.config_.max_pair_products),
        };
    }

    void merge_from(const QWeyl3Algebra& source) {
        for (const auto& item : source.terms_) add(item.second.basis, item.second.coefficient);
    }

    QWeylSpace space_;
    QWeylAlgebraConfig config_{};
    std::map<std::string, Term> terms_{};
};

}  // namespace qubit

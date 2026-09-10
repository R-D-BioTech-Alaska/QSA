#pragma once

#include "qubit/qweyl.hpp"

#include <algorithm>
#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace qubit {

struct QWeylAlgebraConfig {
    std::size_t max_basis_terms{1U << 20U};
    std::size_t max_pair_products{1U << 24U};
};

struct QWeylLinearTerm {
    QWeylOperator basis;
    QRational coefficient{0};
};

struct QWeylAlgebraProjectionReceipt {
    std::size_t source_basis_terms{0U};
    std::size_t surviving_basis_terms{0U};
    std::size_t unresolved_basis_terms{0U};
    std::size_t cancellation_sites{0U};
    QRational positive_weight{0};
    QRational neutral_weight{0};
    QRational negative_weight{0};
    QRational canceled_weight{0};
    bool ready{false};
};

struct QWeylAlgebraProjection {
    std::vector<QWeylLinearTerm> terms{};
    QWeylAlgebraProjectionReceipt receipt{};
};

struct QWeylCommutingGroup {
    std::vector<QWeylOperator> terms{};
};

class QWeylAlgebra {
public:
    explicit QWeylAlgebra(QWeylSpace space, QWeylAlgebraConfig config = {})
        : space_(std::move(space)), config_(config) {
        if (config_.max_basis_terms == 0U || config_.max_pair_products == 0U) {
            throw QMathError("Weyl algebra resource caps must be positive");
        }
    }

    [[nodiscard]] const QWeylSpace& space() const noexcept { return space_; }
    [[nodiscard]] const QWeylAlgebraConfig& config() const noexcept { return config_; }
    [[nodiscard]] std::size_t basis_term_count() const noexcept { return terms_.size(); }
    [[nodiscard]] bool empty() const noexcept { return terms_.empty(); }

    void add(QPolarity polarity, QWeylOperator basis, QRational magnitude = QRational(1)) {
        require_space(basis);
        if (magnitude < QRational(0)) throw QMathError("Weyl algebra magnitude cannot be negative");
        if (magnitude.is_zero()) return;
        const std::string key = basis.canonical();
        auto found = terms_.find(key);
        if (found == terms_.end()) {
            if (terms_.size() >= config_.max_basis_terms) throw QMathError("Weyl algebra basis-term cap exceeded");
            Term term{std::move(basis), QSignedChannels(QSignedDomain::QuantumOperator)};
            found = terms_.emplace(key, std::move(term)).first;
        }
        found->second.channels.add(polarity, std::move(magnitude));
    }

    [[nodiscard]] QWeylAlgebra added(const QWeylAlgebra& rhs) const {
        require_space(rhs);
        QWeylAlgebra result(space_, conservative_config(rhs));
        result.merge_from(*this);
        result.merge_from(rhs);
        return result;
    }

    [[nodiscard]] QWeylAlgebra negated() const {
        QWeylAlgebra result(space_, config_);
        for (const auto& item : terms_) {
            const Term& term = item.second;
            result.add(QPolarity::Positive, term.basis, term.channels.negative());
            result.add(QPolarity::Negative, term.basis, term.channels.positive());
            result.add(QPolarity::Neutral, term.basis, term.channels.neutral());
        }
        return result;
    }

    [[nodiscard]] QWeylAlgebra subtracted(const QWeylAlgebra& rhs) const {
        return added(rhs.negated());
    }

    [[nodiscard]] QWeylAlgebra multiplied(const QWeylAlgebra& rhs) const {
        require_space(rhs);
        const QWeylAlgebraConfig output_config = conservative_config(rhs);
        if (!rhs.terms_.empty() && terms_.size() > output_config.max_pair_products / rhs.terms_.size()) {
            throw QMathError("Weyl algebra pair-product cap exceeded");
        }
        QWeylAlgebra result(space_, output_config);
        for (const auto& left_item : terms_) {
            const Term& left = left_item.second;
            for (const auto& right_item : rhs.terms_) {
                const Term& right = right_item.second;
                const QWeylOperator basis = left.basis.multiplied(right.basis);

                const QRational positive = left.channels.positive() * right.channels.positive() +
                                           left.channels.negative() * right.channels.negative();
                const QRational negative = left.channels.positive() * right.channels.negative() +
                                           left.channels.negative() * right.channels.positive();
                const QRational unresolved = left.channels.neutral() * right.channels.total() +
                                             left.channels.active() * right.channels.neutral();

                result.add(QPolarity::Positive, basis, positive);
                result.add(QPolarity::Negative, basis, negative);
                result.add(QPolarity::Neutral, basis, unresolved);
            }
        }
        return result;
    }

    [[nodiscard]] QWeylAlgebra commutator(const QWeylAlgebra& rhs) const {
        return multiplied(rhs).subtracted(rhs.multiplied(*this));
    }

    [[nodiscard]] QWeylAlgebra anticommutator(const QWeylAlgebra& rhs) const {
        return multiplied(rhs).added(rhs.multiplied(*this));
    }

    [[nodiscard]] QWeylAlgebraProjection project() const {
        QWeylAlgebraProjection result;
        result.receipt.source_basis_terms = terms_.size();
        for (const auto& item : terms_) {
            const QSignedChannels& channels = item.second.channels;
            result.receipt.positive_weight = result.receipt.positive_weight + channels.positive();
            result.receipt.neutral_weight = result.receipt.neutral_weight + channels.neutral();
            result.receipt.negative_weight = result.receipt.negative_weight + channels.negative();
            const QRational canceled = channels.canceled();
            result.receipt.canceled_weight = result.receipt.canceled_weight + canceled;
            if (!canceled.is_zero()) ++result.receipt.cancellation_sites;
            if (!channels.neutral().is_zero()) ++result.receipt.unresolved_basis_terms;
        }
        if (result.receipt.unresolved_basis_terms != 0U) return result;

        result.terms.reserve(terms_.size());
        for (const auto& item : terms_) {
            const QRational coefficient = item.second.channels.net();
            if (coefficient.is_zero()) continue;
            result.terms.push_back(QWeylLinearTerm{item.second.basis, coefficient});
        }
        result.receipt.surviving_basis_terms = result.terms.size();
        result.receipt.ready = true;
        return result;
    }

    [[nodiscard]] std::vector<QWeylCommutingGroup> commuting_groups() const {
        std::vector<QWeylCommutingGroup> groups;
        for (const auto& item : terms_) {
            if (item.second.channels.total().is_zero()) continue;
            bool placed = false;
            for (QWeylCommutingGroup& group : groups) {
                const bool compatible = std::all_of(
                    group.terms.begin(), group.terms.end(),
                    [&](const QWeylOperator& existing) { return existing.commutes_with(item.second.basis); });
                if (!compatible) continue;
                group.terms.push_back(item.second.basis);
                placed = true;
                break;
            }
            if (!placed) groups.push_back(QWeylCommutingGroup{{item.second.basis}});
        }
        return groups;
    }

    [[nodiscard]] bool all_commuting() const {
        const auto groups = commuting_groups();
        return groups.size() <= 1U;
    }

    [[nodiscard]] std::string canonical() const {
        std::string out = "qweyl-algebra1:" + space_.canonical() + '(';
        bool first = true;
        for (const auto& item : terms_) {
            if (!first) out += ';';
            first = false;
            out += item.first + '=' + item.second.channels.canonical();
        }
        out += ')';
        return out;
    }

private:
    struct Term {
        QWeylOperator basis;
        QSignedChannels channels;
    };

    void require_space(const QWeylOperator& basis) const {
        if (basis.space() != space_) throw QMathError("Weyl algebra basis belongs to a different local space");
    }

    void require_space(const QWeylAlgebra& rhs) const {
        if (rhs.space_ != space_) throw QMathError("Weyl algebras belong to different local spaces");
    }

    [[nodiscard]] QWeylAlgebraConfig conservative_config(const QWeylAlgebra& rhs) const noexcept {
        return QWeylAlgebraConfig{
            std::min(config_.max_basis_terms, rhs.config_.max_basis_terms),
            std::min(config_.max_pair_products, rhs.config_.max_pair_products),
        };
    }

    void merge_from(const QWeylAlgebra& source) {
        for (const auto& item : source.terms_) {
            add(QPolarity::Positive, item.second.basis, item.second.channels.positive());
            add(QPolarity::Negative, item.second.basis, item.second.channels.negative());
            add(QPolarity::Neutral, item.second.basis, item.second.channels.neutral());
        }
    }

    QWeylSpace space_;
    QWeylAlgebraConfig config_{};
    std::map<std::string, Term> terms_{};
};

}  // namespace qubit

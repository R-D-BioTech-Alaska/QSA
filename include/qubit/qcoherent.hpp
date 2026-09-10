#pragma once

#include "qubit/qstate.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

namespace qubit {

struct CoherentSuperpositionConfig {
    std::size_t max_modes{1U << 20U};
    std::size_t max_terms{1024U};
    std::size_t max_complex_entries{1U << 24U};
};

struct CoherentSuperpositionStats {
    std::size_t modes{0U};
    std::size_t terms{0U};
    std::size_t stored_complex_entries{0U};
};

struct CoherentTerm {
    QComplex coefficient{};
    std::vector<QComplex> amplitudes{};
};

class BoundedCoherentSuperposition {
public:
    [[nodiscard]] static BoundedCoherentSuperposition from_terms(
        std::size_t modes,
        std::vector<CoherentTerm> terms,
        CoherentSuperpositionConfig config = {}) {
        return BoundedCoherentSuperposition(modes, std::move(terms), config);
    }

    [[nodiscard]] static BoundedCoherentSuperposition even_cat(
        std::span<const QComplex> amplitudes,
        CoherentSuperpositionConfig config = {}) {
        if (amplitudes.empty()) {
            throw QStateError("Coherent cat requires at least one mode");
        }
        std::vector<QComplex> positive(amplitudes.begin(), amplitudes.end());
        std::vector<QComplex> negative = positive;
        for (QComplex& value : negative) {
            value = -value;
        }
        BoundedCoherentSuperposition state = from_terms(
            amplitudes.size(),
            {
                CoherentTerm{QComplex{1.0}, std::move(positive)},
                CoherentTerm{QComplex{1.0}, std::move(negative)},
            },
            config);
        state.normalize();
        return state;
    }

    [[nodiscard]] std::size_t mode_count() const noexcept { return modes_; }
    [[nodiscard]] std::size_t term_count() const noexcept { return terms_.size(); }
    [[nodiscard]] const CoherentSuperpositionStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const CoherentSuperpositionConfig& config() const noexcept { return config_; }
    [[nodiscard]] const std::vector<CoherentTerm>& terms() const noexcept { return terms_; }

    [[nodiscard]] QComplex overlap(std::size_t left, std::size_t right) const {
        if (left >= terms_.size() || right >= terms_.size()) {
            throw QStateError("Coherent overlap term index is out of range");
        }
        double left_norm = 0.0;
        double right_norm = 0.0;
        QComplex inner{};
        for (std::size_t mode = 0U; mode < modes_; ++mode) {
            left_norm += terms_[left].amplitudes[mode].norm2();
            right_norm += terms_[right].amplitudes[mode].norm2();
            inner += terms_[left].amplitudes[mode].conjugate() * terms_[right].amplitudes[mode];
        }
        const QComplex exponent{
            -0.5 * left_norm - 0.5 * right_norm + inner.re,
            inner.im,
        };
        return exp_complex(exponent);
    }

    [[nodiscard]] double norm_squared() const {
        QComplex total{};
        for (std::size_t left = 0U; left < terms_.size(); ++left) {
            for (std::size_t right = 0U; right < terms_.size(); ++right) {
                total += terms_[left].coefficient.conjugate() * terms_[right].coefficient *
                         overlap(left, right);
            }
        }
        return require_real_positive(total, "Coherent superposition norm is invalid");
    }

    void normalize() {
        const double norm = std::sqrt(norm_squared());
        if (!std::isfinite(norm) || norm <= std::numeric_limits<double>::min()) {
            throw QStateError("Coherent superposition cannot normalize a zero/non-finite norm");
        }
        for (CoherentTerm& term : terms_) {
            term.coefficient /= norm;
        }
    }

    [[nodiscard]] double mean_number(std::size_t mode) const {
        validate_mode(mode);
        const double norm = norm_squared();
        QComplex total{};
        for (std::size_t left = 0U; left < terms_.size(); ++left) {
            for (std::size_t right = 0U; right < terms_.size(); ++right) {
                const QComplex matrix_element =
                    terms_[left].amplitudes[mode].conjugate() * terms_[right].amplitudes[mode] *
                    overlap(left, right);
                total += terms_[left].coefficient.conjugate() * terms_[right].coefficient *
                         matrix_element;
            }
        }
        return require_real(total / norm, "Coherent mean number became complex/non-finite");
    }

    [[nodiscard]] double mean_x(std::size_t mode) const {
        validate_mode(mode);
        const double norm = norm_squared();
        constexpr double inverse_sqrt_two = std::numbers::sqrt2_v<double> / 2.0;
        QComplex total{};
        for (std::size_t left = 0U; left < terms_.size(); ++left) {
            for (std::size_t right = 0U; right < terms_.size(); ++right) {
                const QComplex linear =
                    (terms_[left].amplitudes[mode].conjugate() +
                     terms_[right].amplitudes[mode]) * inverse_sqrt_two;
                total += terms_[left].coefficient.conjugate() * terms_[right].coefficient *
                         linear * overlap(left, right);
            }
        }
        return require_real(total / norm, "Coherent mean-x became complex/non-finite");
    }

    void phase_shift(std::size_t mode, double angle) {
        validate_mode(mode);
        if (!std::isfinite(angle)) {
            throw QStateError("Coherent phase shift must be finite");
        }
        const QComplex phase = QComplex::from_polar(1.0, angle);
        for (CoherentTerm& term : terms_) {
            term.amplitudes[mode] *= phase;
        }
        require_finite();
    }

    void beam_splitter(
        std::size_t first,
        std::size_t second,
        double angle,
        double phase = 0.0) {
        validate_mode(first);
        validate_mode(second);
        if (first == second) {
            throw QStateError("Coherent beam splitter requires distinct modes");
        }
        if (!std::isfinite(angle) || !std::isfinite(phase)) {
            throw QStateError("Coherent beam-splitter parameters must be finite");
        }
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        const QComplex p = QComplex::from_polar(1.0, phase);
        const QComplex p_conjugate = p.conjugate();
        for (CoherentTerm& term : terms_) {
            const QComplex a = term.amplitudes[first];
            const QComplex b = term.amplitudes[second];
            term.amplitudes[first] = a * c - p_conjugate * b * s;
            term.amplitudes[second] = p * a * s + b * c;
        }
        require_finite();
    }

    void kerr(std::size_t mode, double strength) {
        validate_mode(mode);
        if (!std::isfinite(strength)) {
            throw QStateError("Coherent Kerr strength must be finite");
        }
        throw QStateError(
            "Kerr evolution is outside the bounded passive coherent-superposition contract");
    }

private:
    std::size_t modes_{0U};
    CoherentSuperpositionConfig config_{};
    std::vector<CoherentTerm> terms_{};
    CoherentSuperpositionStats stats_{};

    BoundedCoherentSuperposition(
        std::size_t modes,
        std::vector<CoherentTerm> terms,
        CoherentSuperpositionConfig config)
        : modes_(modes), config_(config), terms_(std::move(terms)) {
        if (modes_ == 0U || modes_ > config_.max_modes || config_.max_terms == 0U ||
            config_.max_complex_entries == 0U) {
            throw QStateError("Coherent superposition configuration or mode count is invalid");
        }
        if (terms_.empty() || terms_.size() > config_.max_terms) {
            throw QStateError("Coherent superposition term count is zero or exceeds configured cap");
        }
        const std::size_t per_term = checked_sum(
            modes_, 1U, "Coherent superposition per-term storage overflowed");
        const std::size_t entries = checked_product(
            terms_.size(), per_term, "Coherent superposition storage overflowed");
        if (entries > config_.max_complex_entries) {
            throw QStateError("Coherent superposition exceeds configured complex-entry cap");
        }
        for (const CoherentTerm& term : terms_) {
            if (term.amplitudes.size() != modes_) {
                throw QStateError("Coherent term amplitude vector has the wrong mode count");
            }
        }
        require_finite();
        stats_ = CoherentSuperpositionStats{modes_, terms_.size(), entries};
    }

    void validate_mode(std::size_t mode) const {
        if (mode >= modes_) {
            throw QStateError("Coherent superposition mode is out of range");
        }
    }

    [[nodiscard]] static bool finite_complex(const QComplex& value) noexcept {
        return std::isfinite(value.re) && std::isfinite(value.im);
    }

    void require_finite() const {
        for (const CoherentTerm& term : terms_) {
            if (!finite_complex(term.coefficient)) {
                throw QStateError("Coherent superposition coefficient is non-finite");
            }
            for (const QComplex& amplitude : term.amplitudes) {
                if (!finite_complex(amplitude)) {
                    throw QStateError("Coherent superposition amplitude is non-finite");
                }
            }
        }
    }

    [[nodiscard]] static QComplex exp_complex(const QComplex& value) {
        const double magnitude = std::exp(value.re);
        if (!std::isfinite(magnitude)) {
            throw QStateError("Coherent overlap exponential became non-finite");
        }
        return QComplex::from_polar(magnitude, value.im);
    }

    [[nodiscard]] static double require_real(const QComplex& value, const char* message) {
        if (!finite_complex(value)) {
            throw QStateError(message);
        }
        const double scale = 1.0 + std::abs(value.re);
        if (std::abs(value.im) > 1e-10 * scale) {
            throw QStateError(message);
        }
        return value.re;
    }

    [[nodiscard]] static double require_real_positive(const QComplex& value, const char* message) {
        const double result = require_real(value, message);
        if (result <= 0.0) {
            throw QStateError(message);
        }
        return result;
    }

    [[nodiscard]] static std::size_t checked_product(
        std::size_t left, std::size_t right, const char* message) {
        if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
            throw QStateError(message);
        }
        return left * right;
    }

    [[nodiscard]] static std::size_t checked_sum(
        std::size_t left, std::size_t right, const char* message) {
        if (right > std::numeric_limits<std::size_t>::max() - left) {
            throw QStateError(message);
        }
        return left + right;
    }
};

}  // namespace qubit

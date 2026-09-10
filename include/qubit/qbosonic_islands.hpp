#pragma once

#include "qubit/qfock.hpp"
#include "qubit/qgaussian.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace qubit {

enum class BosonicIslandKind : std::uint8_t {
    Gaussian = 0,
    SparseFock = 1,
};

struct BosonicIslandConfig {
    std::size_t max_total_modes{1U << 20U};
    std::size_t max_hopping_terms{1U << 20U};
};

struct GlobalFockHoppingTerm {
    std::size_t create_mode{0U};
    std::size_t annihilate_mode{0U};
    QComplex coefficient{};
};

struct BosonicIslandStats {
    std::size_t total_modes{0U};
    std::size_t gaussian_modes{0U};
    std::size_t fock_modes{0U};
    std::size_t gaussian_components{0U};
    std::size_t gaussian_descriptor_scalars{0U};
    std::size_t fock_terms{0U};
    std::size_t fock_occupied_entries{0U};
};

class ExactBosonicIslandProduct {
public:
    [[nodiscard]] static ExactBosonicIslandProduct from_product(
        StructuredGaussianState gaussian,
        ExactSparseFockState fock,
        BosonicIslandConfig config = {}) {
        return ExactBosonicIslandProduct(std::move(gaussian), std::move(fock), config);
    }

    [[nodiscard]] std::size_t mode_count() const noexcept { return stats_.total_modes; }
    [[nodiscard]] std::size_t gaussian_mode_count() const noexcept {
        return stats_.gaussian_modes;
    }
    [[nodiscard]] std::size_t fock_mode_count() const noexcept { return stats_.fock_modes; }
    [[nodiscard]] std::size_t fock_global_offset() const noexcept {
        return stats_.gaussian_modes;
    }
    [[nodiscard]] const BosonicIslandConfig& config() const noexcept { return config_; }
    [[nodiscard]] const BosonicIslandStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const StructuredGaussianState& gaussian() const noexcept { return gaussian_; }
    [[nodiscard]] const ExactSparseFockState& fock() const noexcept { return fock_; }

    [[nodiscard]] BosonicIslandKind island_of(std::size_t global_mode) const {
        validate_global_mode(global_mode);
        return global_mode < stats_.gaussian_modes
            ? BosonicIslandKind::Gaussian
            : BosonicIslandKind::SparseFock;
    }

    [[nodiscard]] std::size_t local_mode(std::size_t global_mode) const {
        validate_global_mode(global_mode);
        return global_mode < stats_.gaussian_modes
            ? global_mode
            : global_mode - stats_.gaussian_modes;
    }

    [[nodiscard]] double mean_number(std::size_t global_mode) const {
        return island_of(global_mode) == BosonicIslandKind::Gaussian
            ? gaussian_.mean_occupation(global_mode)
            : fock_.mean_number(local_mode(global_mode));
    }

    [[nodiscard]] double total_mean_number() const {
        const double value = gaussian_.total_mean_occupation() + fock_total_mean_number();
        if (!std::isfinite(value)) {
            throw QStateError("Bosonic island total mean number became non-finite");
        }
        return value;
    }

    [[nodiscard]] double cross_island_number_product(
        std::size_t first,
        std::size_t second) const {
        require_cross_island(first, second);
        const double value = mean_number(first) * mean_number(second);
        if (!std::isfinite(value)) {
            throw QStateError("Bosonic cross-island number product became non-finite");
        }
        return value;
    }

    [[nodiscard]] double cross_island_number_covariance(
        std::size_t first,
        std::size_t second) const {
        require_cross_island(first, second);
        return 0.0;
    }

    void gaussian_displace(std::size_t global_mode, double dq, double dp) {
        gaussian_.displace(require_gaussian(global_mode), dq, dp);
        refresh_stats();
    }

    void gaussian_rotate(std::size_t global_mode, double angle) {
        gaussian_.rotate(require_gaussian(global_mode), angle);
        refresh_stats();
    }

    void gaussian_squeeze(std::size_t global_mode, double parameter) {
        gaussian_.squeeze(require_gaussian(global_mode), parameter);
        refresh_stats();
    }

    void gaussian_two_mode_squeeze(
        std::size_t first,
        std::size_t second,
        double parameter) {
        const std::size_t local_first = require_gaussian(first);
        const std::size_t local_second = require_gaussian(second);
        gaussian_.two_mode_squeeze(local_first, local_second, parameter);
        refresh_stats();
    }

    void gaussian_loss(
        std::size_t global_mode,
        double transmissivity,
        double environment_occupation = 0.0) {
        gaussian_.loss(
            require_gaussian(global_mode), transmissivity, environment_occupation);
        refresh_stats();
    }

    void gaussian_beam_splitter(
        std::size_t first,
        std::size_t second,
        double transmissivity) {
        const std::size_t local_first = require_gaussian(first);
        const std::size_t local_second = require_gaussian(second);
        gaussian_.beam_splitter(local_first, local_second, transmissivity);
        refresh_stats();
    }

    void apply_fock_hopping(std::span<const GlobalFockHoppingTerm> hopping) {
        if (hopping.size() > config_.max_hopping_terms) {
            throw QStateError("Bosonic island hopping term count exceeds configured cap");
        }
        std::vector<FockHoppingTerm> local;
        local.reserve(hopping.size());
        for (const GlobalFockHoppingTerm& term : hopping) {
            local.push_back(FockHoppingTerm{
                require_fock(term.create_mode),
                require_fock(term.annihilate_mode),
                term.coefficient,
            });
        }
        fock_ = fock_.apply_hopping(local);
        refresh_stats();
    }

    void apply_fock_bose_hubbard(
        std::span<const double> onsite,
        std::span<const double> interaction,
        std::span<const GlobalFockHoppingTerm> hopping) {
        if (onsite.size() != stats_.fock_modes || interaction.size() != stats_.fock_modes) {
            throw QStateError("Bosonic island Bose-Hubbard arrays must match Fock island size");
        }
        if (hopping.size() > config_.max_hopping_terms) {
            throw QStateError("Bosonic island hopping term count exceeds configured cap");
        }
        std::vector<FockHoppingTerm> local;
        local.reserve(hopping.size());
        for (const GlobalFockHoppingTerm& term : hopping) {
            local.push_back(FockHoppingTerm{
                require_fock(term.create_mode),
                require_fock(term.annihilate_mode),
                term.coefficient,
            });
        }
        fock_ = fock_.apply_bose_hubbard(onsite, interaction, local);
        refresh_stats();
    }

private:
    StructuredGaussianState gaussian_;
    ExactSparseFockState fock_;
    BosonicIslandConfig config_{};
    BosonicIslandStats stats_{};

    ExactBosonicIslandProduct(
        StructuredGaussianState gaussian,
        ExactSparseFockState fock,
        BosonicIslandConfig config)
        : gaussian_(std::move(gaussian)), fock_(std::move(fock)), config_(config) {
        if (config_.max_total_modes == 0U || config_.max_hopping_terms == 0U) {
            throw QStateError("Bosonic island configuration contains a zero cap");
        }
        const std::size_t total = checked_sum(
            gaussian_.mode_count(), fock_.mode_count(),
            "Bosonic island total mode count overflowed");
        if (total > config_.max_total_modes) {
            throw QStateError("Bosonic island total mode count exceeds configured cap");
        }
        refresh_stats();
    }

    void refresh_stats() noexcept {
        stats_ = BosonicIslandStats{
            gaussian_.mode_count() + fock_.mode_count(),
            gaussian_.mode_count(),
            fock_.mode_count(),
            gaussian_.stats().components,
            gaussian_.stats().descriptor_scalars,
            fock_.stats().terms,
            fock_.stats().occupied_entries,
        };
    }

    void validate_global_mode(std::size_t global_mode) const {
        if (global_mode >= stats_.total_modes) {
            throw QStateError("Bosonic island mode lies outside global shape");
        }
    }

    [[nodiscard]] std::size_t require_gaussian(std::size_t global_mode) const {
        if (island_of(global_mode) != BosonicIslandKind::Gaussian) {
            throw QStateError("Operation crosses or targets outside the Gaussian island");
        }
        return global_mode;
    }

    [[nodiscard]] std::size_t require_fock(std::size_t global_mode) const {
        if (island_of(global_mode) != BosonicIslandKind::SparseFock) {
            throw QStateError("Operation crosses or targets outside the sparse-Fock island");
        }
        return global_mode - stats_.gaussian_modes;
    }

    void require_cross_island(std::size_t first, std::size_t second) const {
        if (island_of(first) == island_of(second)) {
            throw QStateError(
                "Cross-island factorization query requires modes from different islands");
        }
    }

    [[nodiscard]] double fock_total_mean_number() const {
        long double norm = 0.0L;
        long double numerator = 0.0L;
        for (const SparseFockTerm& term : fock_.terms()) {
            const long double weight = static_cast<long double>(term.amplitude.norm2());
            std::size_t particles = 0U;
            for (const auto& [mode, count] : term.occupation) {
                (void)mode;
                particles = checked_sum(
                    particles, count, "Bosonic island Fock particle count overflowed");
            }
            norm += weight;
            numerator += weight * static_cast<long double>(particles);
        }
        if (!std::isfinite(static_cast<double>(norm)) || norm <= 0.0L) {
            throw QStateError("Bosonic island Fock total mean requires positive finite norm");
        }
        const long double value = numerator / norm;
        if (!std::isfinite(static_cast<double>(value))) {
            throw QStateError("Bosonic island Fock total mean became non-finite");
        }
        return static_cast<double>(value);
    }

    [[nodiscard]] static std::size_t checked_sum(
        std::size_t left,
        std::size_t right,
        const char* message) {
        if (right > std::numeric_limits<std::size_t>::max() - left) {
            throw QStateError(message);
        }
        return left + right;
    }
};

}  // namespace qubit

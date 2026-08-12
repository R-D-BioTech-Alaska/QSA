#pragma once

#include "qubit/qstate.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace qubit {

using FockOccupation = std::vector<std::pair<std::size_t, std::size_t>>;

struct SparseFockTerm {
    FockOccupation occupation{};
    QComplex amplitude{};
};

struct FockHoppingTerm {
    std::size_t create_mode{0U};
    std::size_t annihilate_mode{0U};
    QComplex coefficient{};
};

struct SparseFockConfig {
    std::size_t max_modes{1U << 20U};
    std::size_t max_particles{1U << 30U};
    std::size_t max_terms{1U << 20U};
    std::size_t max_occupied_entries{1U << 24U};
    std::size_t max_branch_products{1U << 24U};
};

struct SparseFockStats {
    std::size_t modes{0U};
    std::size_t terms{0U};
    std::size_t occupied_entries{0U};
    std::size_t maximum_particles{0U};
    bool fixed_particle_number{false};
    std::size_t particles{0U};
};

class ExactSparseFockState {
public:
    [[nodiscard]] static ExactSparseFockState from_terms(
        std::size_t mode_count, std::vector<SparseFockTerm> terms, SparseFockConfig config = {});
    [[nodiscard]] static ExactSparseFockState basis(
        std::size_t mode_count, FockOccupation occupation, SparseFockConfig config = {});

    [[nodiscard]] std::size_t mode_count() const noexcept { return mode_count_; }
    [[nodiscard]] const SparseFockConfig& config() const noexcept { return config_; }
    [[nodiscard]] const SparseFockStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const std::vector<SparseFockTerm>& terms() const noexcept { return terms_; }
    [[nodiscard]] std::optional<std::size_t> fixed_particle_number() const noexcept;
    [[nodiscard]] double norm_squared() const noexcept;
    [[nodiscard]] QComplex amplitude(const FockOccupation& occupation) const;
    [[nodiscard]] double mean_number(std::size_t mode) const;
    [[nodiscard]] double sector_log2_dimension() const;
    [[nodiscard]] ExactSparseFockState scaled(QComplex scalar) const;
    [[nodiscard]] ExactSparseFockState add(const ExactSparseFockState& other) const;
    [[nodiscard]] ExactSparseFockState apply_hopping(std::span<const FockHoppingTerm> hopping) const;
    [[nodiscard]] ExactSparseFockState apply_bose_hubbard(
        std::span<const double> onsite,
        std::span<const double> interaction,
        std::span<const FockHoppingTerm> hopping) const;

private:
    std::size_t mode_count_{0U};
    std::vector<SparseFockTerm> terms_{};
    SparseFockConfig config_{};
    SparseFockStats stats_{};

    ExactSparseFockState(
        std::size_t mode_count, std::vector<SparseFockTerm> terms, SparseFockConfig config);

    [[nodiscard]] static bool finite(QComplex value) noexcept;
    [[nodiscard]] static std::size_t checked_product(std::size_t left, std::size_t right, const char* message);
    [[nodiscard]] static std::size_t checked_sum(std::size_t left, std::size_t right, const char* message);
    [[nodiscard]] static std::size_t particle_count(const FockOccupation& occupation);
    [[nodiscard]] static FockOccupation canonical(
        FockOccupation occupation, std::size_t mode_count, std::size_t max_particles);
    [[nodiscard]] static std::size_t occupation_of(const FockOccupation& occupation, std::size_t mode) noexcept;
    static void decrement(FockOccupation& occupation, std::size_t mode);
    static void increment(FockOccupation& occupation, std::size_t mode);
    void validate_mode(std::size_t mode) const;
    void require_shape(const ExactSparseFockState& other, const char* label) const;
};

}  // namespace qubit

#include "qubit/detail/qfock_impl.hpp"

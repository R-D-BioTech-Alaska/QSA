#pragma once

#include "qubit/qstate.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace qubit {

class GaussianState {
public:
    explicit GaussianState(std::size_t modes);

    [[nodiscard]] std::size_t mode_count() const noexcept { return modes_; }
    [[nodiscard]] const std::vector<double>& mean() const noexcept { return mean_; }
    [[nodiscard]] const std::vector<double>& covariance() const noexcept { return covariance_; }
    [[nodiscard]] std::size_t estimated_bytes() const noexcept;

    void displace(std::size_t mode, double q, double p);
    void phase_shift(std::size_t mode, double theta);
    void squeeze(std::size_t mode, double r);
    void beam_splitter(std::size_t first, std::size_t second, double theta);
    void two_mode_squeeze(std::size_t first, std::size_t second, double r);

    [[nodiscard]] double mean_photon_number(std::size_t mode) const;
    [[nodiscard]] double total_mean_photon_number() const;
    [[nodiscard]] bool validate(double tolerance = 1e-10) const noexcept;

private:
    std::size_t modes_{0};
    std::vector<double> mean_{};
    std::vector<double> covariance_{};

    void validate_mode(std::size_t mode) const;
    void apply_local(std::span<const std::size_t> indices, std::span<const double> transform);
};

enum class FermionicGaussianEligibilityCode {
    eligible,
    zero_modes,
    non_gaussian_input,
    non_quadratic_hamiltonian,
    size_overflow,
    memory_budget_exceeded,
};

struct FermionicGaussianEligibility {
    FermionicGaussianEligibilityCode code{FermionicGaussianEligibilityCode::zero_modes};
    std::size_t required_bytes{0};

    [[nodiscard]] bool accepted() const noexcept {
        return code == FermionicGaussianEligibilityCode::eligible;
    }
};

[[nodiscard]] std::size_t fermionic_gaussian_required_bytes(std::size_t modes) noexcept;
[[nodiscard]] FermionicGaussianEligibility fermionic_gaussian_eligibility(
    std::size_t modes,
    bool gaussian_input,
    std::size_t maximum_majorana_degree,
    std::size_t memory_budget_bytes) noexcept;

class FermionicGaussianState {
public:
    explicit FermionicGaussianState(std::size_t modes);

    [[nodiscard]] std::size_t mode_count() const noexcept { return modes_; }
    [[nodiscard]] std::size_t majorana_count() const noexcept { return 2U * modes_; }
    [[nodiscard]] const std::vector<double>& covariance() const noexcept { return covariance_; }
    [[nodiscard]] std::size_t estimated_bytes() const noexcept;

    void rotate_majoranas(std::size_t first, std::size_t second, double theta);

    [[nodiscard]] double occupation(std::size_t mode) const;
    [[nodiscard]] double total_occupation() const;
    [[nodiscard]] bool validate(double tolerance = 1e-10) const noexcept;

private:
    std::size_t modes_{0};
    std::vector<double> covariance_{};

    void validate_majorana(std::size_t index) const;
    void validate_mode(std::size_t mode) const;
};

}  // namespace qubit

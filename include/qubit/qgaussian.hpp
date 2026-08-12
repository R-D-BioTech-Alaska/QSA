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

}  // namespace qubit

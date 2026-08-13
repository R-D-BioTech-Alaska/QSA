#include "qubit/qgaussian.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(double actual, double expected, double tolerance, const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    using qubit::FermionicGaussianEligibilityCode;
    using qubit::FermionicGaussianState;
    using qubit::GaussianState;
    using qubit::QStateError;
    using qubit::fermionic_gaussian_eligibility;
    using qubit::fermionic_gaussian_required_bytes;

    {
        GaussianState state(1U);
        require(state.validate(), "vacuum Gaussian state failed validation");
        require_close(state.mean_photon_number(0U), 0.0, 1e-13, "vacuum photon number changed");
    }

    {
        GaussianState state(1U);
        state.displace(0U, 0.6, -0.8);
        require_close(state.mean_photon_number(0U), 0.5, 1e-13, "coherent displacement photon number is wrong");
        require(state.validate(), "displaced Gaussian state failed validation");
    }

    {
        constexpr double r = 0.73;
        GaussianState state(1U);
        state.squeeze(0U, r);
        require_close(
            state.mean_photon_number(0U),
            std::sinh(r) * std::sinh(r),
            2e-13,
            "single-mode squeezed photon number is wrong");
        require(state.validate(), "single-mode squeezed state failed validation");
    }

    {
        constexpr double r = 0.41;
        GaussianState state(2U);
        state.two_mode_squeeze(0U, 1U, r);
        const double expected = std::sinh(r) * std::sinh(r);
        require_close(state.mean_photon_number(0U), expected, 2e-13, "first two-mode squeezed photon number is wrong");
        require_close(state.mean_photon_number(1U), expected, 2e-13, "second two-mode squeezed photon number is wrong");
        require_close(state.total_mean_photon_number(), 2.0 * expected, 4e-13, "two-mode squeeze total photon number is wrong");
        require(state.validate(), "two-mode squeezed state failed validation");
    }

    {
        GaussianState state(2U);
        state.displace(0U, 0.8, -0.2);
        state.displace(1U, -0.3, 0.5);
        const double before = state.total_mean_photon_number();
        state.phase_shift(0U, 0.37);
        require_close(state.total_mean_photon_number(), before, 2e-13, "phase shift changed total photon number");
        state.beam_splitter(0U, 1U, -0.29);
        require_close(state.total_mean_photon_number(), before, 4e-13, "beam splitter changed total photon number");
        require(state.validate(), "passive Gaussian evolution failed validation");
    }

    {
        GaussianState state(256U);
        for (std::size_t mode = 0U; mode < state.mode_count(); ++mode) {
            state.phase_shift(mode, 0.0005 * static_cast<double>(mode));
            if ((mode & 7U) == 0U) {
                state.squeeze(mode, 0.01);
            }
        }
        for (std::size_t mode = 0U; mode + 1U < state.mode_count(); mode += 2U) {
            state.beam_splitter(mode, mode + 1U, 0.03);
        }
        require(state.validate(1e-9), "large Gaussian state failed validation");
        require(state.estimated_bytes() < 3U * 1024U * 1024U, "large Gaussian state storage exceeded polynomial envelope");
    }

    {
        bool rejected = false;
        try {
            static_cast<void>(GaussianState(0U));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "zero-mode Gaussian state was accepted");
    }

    {
        GaussianState state(2U);
        bool rejected = false;
        try {
            state.beam_splitter(0U, 0U, 0.1);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "same-mode Gaussian beam splitter was accepted");
    }

    {
        GaussianState state(1U);
        bool rejected = false;
        try {
            state.squeeze(0U, std::numeric_limits<double>::infinity());
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "nonfinite Gaussian squeeze was accepted");
    }

    {
        GaussianState state(1U);
        bool rejected = false;
        try {
            state.displace(1U, 0.0, 0.0);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "out-of-range Gaussian mode was accepted");
    }

    {
        FermionicGaussianState state(1U);
        require(state.validate(), "fermionic vacuum failed validation");
        require_close(state.occupation(0U), 0.0, 1e-13, "fermionic vacuum occupation changed");
    }

    {
        const double theta = 0.63;
        FermionicGaussianState state(2U);
        state.rotate_majoranas(1U, 2U, theta);
        const double expected = 0.5 * (1.0 - std::cos(theta));
        require_close(state.occupation(0U), expected, 2e-13, "first fermionic occupation is wrong");
        require_close(state.occupation(1U), expected, 2e-13, "second fermionic occupation is wrong");
        require(state.validate(1e-12), "fermionic Majorana rotation broke pure Gaussian closure");
        state.rotate_majoranas(1U, 2U, -theta);
        require_close(state.total_occupation(), 0.0, 4e-13, "inverse fermionic rotation did not restore vacuum");
        require(state.validate(1e-12), "inverse fermionic rotation broke validation");
    }

    {
        FermionicGaussianState state(128U);
        for (std::size_t mode = 0U; mode + 1U < state.mode_count(); ++mode) {
            state.rotate_majoranas(2U * mode + 1U, 2U * (mode + 1U), 0.001 * static_cast<double>(mode + 1U));
        }
        require(state.validate(1e-9), "large fermionic Gaussian state failed validation");
        require(state.estimated_bytes() < 600U * 1024U, "fermionic covariance exceeded polynomial storage envelope");
    }

    {
        constexpr std::size_t one_gib = 1024U * 1024U * 1024U;
        require(
            fermionic_gaussian_required_bytes(4096U) == 536870912U,
            "4096-mode fermionic covariance byte count changed");
        const auto accepted = fermionic_gaussian_eligibility(4096U, true, 2U, one_gib);
        require(accepted.accepted(), "4096-mode quadratic fermionic Gaussian request was rejected");
        require(accepted.required_bytes == 536870912U, "fermionic eligibility byte receipt changed");
        require(
            fermionic_gaussian_eligibility(4096U, false, 2U, one_gib).code ==
                FermionicGaussianEligibilityCode::non_gaussian_input,
            "non-Gaussian fermionic input was accepted");
        require(
            fermionic_gaussian_eligibility(4096U, true, 4U, one_gib).code ==
                FermionicGaussianEligibilityCode::non_quadratic_hamiltonian,
            "quartic fermionic interaction was accepted");
        require(
            fermionic_gaussian_eligibility(8192U, true, 2U, one_gib).code ==
                FermionicGaussianEligibilityCode::memory_budget_exceeded,
            "fermionic request exceeding the memory budget was accepted");
        require(
            fermionic_gaussian_eligibility(0U, true, 2U, one_gib).code ==
                FermionicGaussianEligibilityCode::zero_modes,
            "zero-mode fermionic request was accepted");
    }

    {
        FermionicGaussianState state(2U);
        bool rejected = false;
        try {
            state.rotate_majoranas(0U, 0U, 0.1);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "same-index fermionic rotation was accepted");
        rejected = false;
        try {
            state.rotate_majoranas(0U, 4U, 0.1);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "out-of-range fermionic Majorana index was accepted");
        rejected = false;
        try {
            state.rotate_majoranas(0U, 1U, std::numeric_limits<double>::infinity());
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "nonfinite fermionic rotation was accepted");
    }

    return 0;
}

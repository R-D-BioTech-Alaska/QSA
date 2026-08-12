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
    using qubit::GaussianState;
    using qubit::QStateError;

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

    return 0;
}

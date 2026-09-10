#include "qubit/qfermionic_gaussian.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

using qubit::FermionicGaussianConfig;
using qubit::FermionicGaussianEligibilityCode;
using qubit::FermionicGaussianState;
using qubit::QStateError;
using qubit::fermionic_gaussian_eligibility;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(double actual, double expected, double tolerance, const char* message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    {
        const auto zero = fermionic_gaussian_eligibility(0U, true, 2U, 1024U);
        require(zero.code == FermionicGaussianEligibilityCode::zero_modes,
                "zero-mode eligibility was not rejected");

        const auto input = fermionic_gaussian_eligibility(4U, false, 2U, 1024U);
        require(input.code == FermionicGaussianEligibilityCode::non_gaussian_input,
                "non-Gaussian input was not rejected");

        const auto degree = fermionic_gaussian_eligibility(4U, true, 4U, 1024U);
        require(degree.code == FermionicGaussianEligibilityCode::non_quadratic_hamiltonian,
                "non-quadratic Hamiltonian was not rejected");

        FermionicGaussianConfig cap;
        cap.max_modes = 3U;
        const auto modes = fermionic_gaussian_eligibility(4U, true, 2U, 4096U, cap);
        require(modes.code == FermionicGaussianEligibilityCode::mode_cap_exceeded,
                "mode cap was not enforced");

        cap.max_modes = 16U;
        cap.max_covariance_scalars = 16U;
        const auto scalars = fermionic_gaussian_eligibility(4U, true, 2U, 4096U, cap);
        require(scalars.code == FermionicGaussianEligibilityCode::scalar_cap_exceeded,
                "covariance scalar cap was not enforced");

        cap.max_covariance_scalars = 1024U;
        const auto memory = fermionic_gaussian_eligibility(4U, true, 2U, 32U, cap);
        require(memory.code == FermionicGaussianEligibilityCode::memory_budget_exceeded,
                "memory budget was not enforced");
        require(memory.required_bytes == 512U,
                "fermionic covariance byte estimate is wrong");

        const auto accepted = fermionic_gaussian_eligibility(4U, true, 2U, 512U, cap);
        require(accepted.accepted(), "eligible fermionic Gaussian route was rejected");
        require(accepted.required_bytes == 512U,
                "eligible covariance byte estimate is wrong");
    }

    {
        FermionicGaussianState state(4U);
        require(state.mode_count() == 4U, "fermionic Gaussian mode count is wrong");
        require(state.majorana_count() == 8U, "fermionic Gaussian Majorana count is wrong");
        require(state.validate(), "fermionic vacuum failed structural validation");
        require(state.validate_pure(), "fermionic vacuum failed purity validation");
        require_close(state.total_occupation(), 0.0, 1e-12,
                      "fermionic vacuum has nonzero occupation");

        const auto original = state.covariance();
        state.rotate_majoranas(0U, 1U, 0.73);
        require(state.validate(), "same-mode Majorana rotation damaged covariance");
        require(state.validate_pure(), "same-mode Majorana rotation damaged purity");
        require_close(state.total_occupation(), 0.0, 1e-12,
                      "same-mode Majorana rotation changed vacuum occupation");
        for (std::size_t index = 0U; index < original.size(); ++index) {
            require_close(state.covariance()[index], original[index], 2e-12,
                          "same-mode Majorana rotation changed vacuum covariance");
        }
    }

    {
        FermionicGaussianState state(2U);
        const double angle = 0.61;
        state.rotate_majoranas(1U, 2U, angle);
        const double expected = 0.5 * (1.0 - std::cos(angle));
        require_close(state.occupation(0U), expected, 2e-12,
                      "first fermionic occupation is wrong after Majorana mixing");
        require_close(state.occupation(1U), expected, 2e-12,
                      "second fermionic occupation is wrong after Majorana mixing");
        require(state.validate(), "mixed fermionic covariance failed structural validation");
        require(state.validate_pure(), "mixed fermionic covariance failed purity validation");

        state.rotate_majoranas(1U, 2U, -angle);
        require_close(state.total_occupation(), 0.0, 2e-12,
                      "inverse Majorana rotation did not restore the vacuum");
        require(state.validate_pure(), "inverse Majorana rotation damaged purity");
    }

    {
        FermionicGaussianState state(6U);
        for (std::size_t step = 0U; step < 80U; ++step) {
            const std::size_t first = (3U * step + 1U) % state.majorana_count();
            std::size_t second = (5U * step + 4U) % state.majorana_count();
            if (second == first) {
                second = (second + 1U) % state.majorana_count();
            }
            state.rotate_majoranas(first, second, 0.013 * static_cast<double>(step + 1U));
            require(state.validate(2e-10), "fermionic covariance lost antisymmetry");
        }
        require(state.validate_pure(2e-9), "fermionic rotation sequence lost purity");
        for (std::size_t mode = 0U; mode < state.mode_count(); ++mode) {
            const double value = state.occupation(mode);
            require(value >= -2e-10 && value <= 1.0 + 2e-10,
                    "fermionic occupation left the physical interval");
        }
    }

    {
        FermionicGaussianState state(3U);
        const auto before = state.covariance();
        bool rejected = false;
        try {
            state.rotate_majoranas(0U, 1U, std::numeric_limits<double>::infinity());
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "non-finite Majorana rotation was accepted");
        require(state.covariance() == before,
                "failed Majorana rotation changed the covariance");

        rejected = false;
        try {
            state.rotate_majoranas(2U, 2U, 0.1);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "same-index Majorana rotation was accepted");

        rejected = false;
        try {
            state.rotate_majoranas(0U, 99U, 0.1);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "out-of-range Majorana rotation was accepted");
    }

    {
        FermionicGaussianConfig config;
        config.max_modes = 8U;
        bool rejected = false;
        try {
            FermionicGaussianState state(9U, config);
            (void)state;
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "fermionic Gaussian constructor ignored the mode cap");
    }

    {
        FermionicGaussianState state(1024U);
        require(state.validate(), "large fermionic covariance failed structural validation");
        require(state.estimated_bytes() < 40ULL * 1024ULL * 1024ULL,
                "large fermionic covariance exceeded the resource gate");
        state.rotate_majoranas(1U, 2046U, 0.2);
        require(state.validate(2e-10), "large fermionic rotation failed structural validation");
    }

    std::cout << "fermionic Gaussian tests passed\n";
    return 0;
}

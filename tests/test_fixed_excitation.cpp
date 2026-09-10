#include "qubit/qsector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

using namespace qubit;

namespace {

bool close(double left, double right, double tolerance = 2e-11) {
    return std::abs(left - right) <= tolerance;
}

bool close(QComplex left, QComplex right, double tolerance = 2e-11) {
    return almost_equal(left, right, tolerance);
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw QStateError(message);
    }
}

std::size_t mask(std::span<const QubitId> occupied) {
    std::size_t result = 0U;
    for (const QubitId mode : occupied) {
        result |= std::size_t{1U} << static_cast<std::size_t>(mode);
    }
    return result;
}

void dense_phase(std::vector<QComplex>& state, std::size_t mode, double angle) {
    const std::size_t bit = std::size_t{1U} << mode;
    const QComplex phase = QComplex::from_polar(1.0, angle);
    for (std::size_t basis = 0U; basis < state.size(); ++basis) {
        if ((basis & bit) != 0U) {
            state[basis] *= phase;
        }
    }
}

void dense_givens(
    std::vector<QComplex>& state,
    std::size_t left,
    std::size_t right,
    double angle,
    double phase_angle) {
    const std::size_t left_bit = std::size_t{1U} << left;
    const std::size_t right_bit = std::size_t{1U} << right;
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const QComplex phase = QComplex::from_polar(1.0, phase_angle);
    const QComplex phase_conjugate = phase.conjugate();
    for (std::size_t basis = 0U; basis < state.size(); ++basis) {
        if ((basis & left_bit) != 0U && (basis & right_bit) == 0U) {
            const std::size_t partner = (basis ^ left_bit) | right_bit;
            const QComplex a = state[basis];
            const QComplex b = state[partner];
            state[basis] = a * cosine - phase_conjugate * b * sine;
            state[partner] = phase * a * sine + b * cosine;
        }
    }
}

double dense_occupation(const std::vector<QComplex>& state, std::size_t mode) {
    const std::size_t bit = std::size_t{1U} << mode;
    double total = 0.0;
    for (std::size_t basis = 0U; basis < state.size(); ++basis) {
        if ((basis & bit) != 0U) {
            total += state[basis].norm2();
        }
    }
    return total;
}

double dense_marginal(
    const std::vector<QComplex>& state,
    std::span<const QubitId> modes,
    std::span<const std::uint8_t> bits) {
    double total = 0.0;
    for (std::size_t basis = 0U; basis < state.size(); ++basis) {
        bool match = true;
        for (std::size_t query = 0U; query < modes.size(); ++query) {
            const bool present = (basis & (std::size_t{1U} << modes[query])) != 0U;
            if (present != (bits[query] != 0U)) {
                match = false;
                break;
            }
        }
        if (match) {
            total += state[basis].norm2();
        }
    }
    return total;
}

void dense_equivalence() {
    for (const std::size_t modes : std::vector<std::size_t>{6U, 8U, 10U, 12U}) {
        const std::vector<QubitId> occupied{0U, 2U};
        ExactFixedExcitationState sector = ExactFixedExcitationState::basis(modes, occupied);
        std::vector<QComplex> dense(std::size_t{1U} << modes, QComplex{});
        dense[mask(occupied)] = QComplex{1.0};

        sector.apply_mode_phase(2U, 0.23);
        dense_phase(dense, 2U, 0.23);
        sector.apply_givens(0U, 1U, 0.37, 0.11);
        dense_givens(dense, 0U, 1U, 0.37, 0.11);
        sector.apply_givens(2U, 3U, -0.29, -0.07);
        dense_givens(dense, 2U, 3U, -0.29, -0.07);
        sector.apply_givens(1U, modes - 1U, 0.19, 0.05);
        dense_givens(dense, 1U, modes - 1U, 0.19, 0.05);

        require(close(sector.norm_squared(), 1.0), "fixed-excitation norm drifted");
        for (std::size_t left = 0U; left < modes; ++left) {
            for (std::size_t right = left + 1U; right < modes; ++right) {
                const std::vector<QubitId> pair{
                    static_cast<QubitId>(left), static_cast<QubitId>(right),
                };
                require(close(sector.amplitude(pair), dense[mask(pair)]),
                        "fixed-excitation sector amplitude mismatch");
            }
            require(close(sector.occupation_probability(left), dense_occupation(dense, left)),
                    "fixed-excitation occupation probability mismatch");
        }

        const std::vector<QubitId> query{0U, 1U, static_cast<QubitId>(modes - 1U)};
        for (const std::vector<std::uint8_t> bits : std::vector<std::vector<std::uint8_t>>{
                 {0U, 0U, 0U}, {1U, 0U, 0U}, {0U, 1U, 1U}, {1U, 1U, 0U}}) {
            require(close(
                        sector.marginal_probability(query, bits),
                        dense_marginal(dense, query, bits)),
                    "fixed-excitation marginal probability mismatch");
        }
    }
}

void structural_boundaries() {
    const std::vector<QubitId> occupied{0U, 127U};
    ExactFixedExcitationState state = ExactFixedExcitationState::basis(128U, occupied);
    require(state.sector_dimension() == 8128U, "fixed-excitation 128q r2 dimension mismatch");
    require(state.dense_to_sector_log2_ratio() > 115.0,
            "fixed-excitation 128q r2 structural reduction too small");

    bool rejected = false;
    try {
        state.apply_x(0U);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "fixed-excitation accepted a number-changing X operation");

    rejected = false;
    try {
        const std::vector<QubitId> high_filling{0U, 1U, 2U, 3U, 4U};
        (void)ExactFixedExcitationState::basis(8U, high_filling);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "fixed-excitation direct route accepted the high-filling side");
}

void rejection_cases() {
    bool rejected = false;
    try {
        const std::vector<QubitId> duplicate{0U, 0U};
        (void)ExactFixedExcitationState::basis(8U, duplicate);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "fixed-excitation accepted duplicate occupied modes");

    rejected = false;
    try {
        const std::vector<QubitId> outside{0U, 8U};
        (void)ExactFixedExcitationState::basis(8U, outside);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "fixed-excitation accepted an out-of-range occupied mode");

    rejected = false;
    try {
        const std::vector<QubitId> occupied{0U, 1U};
        ExactFixedExcitationState state = ExactFixedExcitationState::basis(8U, occupied);
        state.apply_givens(0U, 1U, std::numeric_limits<double>::infinity());
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "fixed-excitation accepted non-finite Givens parameters");

    rejected = false;
    try {
        FixedExcitationConfig config;
        config.max_sector_amplitudes = 100U;
        const std::vector<QubitId> occupied{0U, 1U};
        (void)ExactFixedExcitationState::basis(32U, occupied, config);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "fixed-excitation ignored its sector dimension cap");

    rejected = false;
    try {
        const std::vector<QubitId> occupied{0U, 1U};
        ExactFixedExcitationState state = ExactFixedExcitationState::basis(8U, occupied);
        const std::vector<QubitId> duplicate_query{0U, 0U};
        const std::vector<std::uint8_t> bits{1U, 1U};
        (void)state.marginal_probability(duplicate_query, bits);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "fixed-excitation accepted duplicate marginal modes");
}

}  // namespace

int main() {
    dense_equivalence();
    structural_boundaries();
    rejection_cases();
    std::cout << "fixed-excitation tests passed\n";
    return 0;
}

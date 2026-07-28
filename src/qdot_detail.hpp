#pragma once

#include "qubit/qdot.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace qubit::qdot::detail {

[[nodiscard]] inline bool finite(double value) noexcept {
    return std::isfinite(value);
}

inline void validate_dot_config(const DotConfig& cfg) {
    const double values[] = {
        cfg.e0, cfg.eup, cfg.edown, cfg.ex, cfg.detuning_gain,
        cfg.exciton_gain, cfg.charge_drive, cfg.spin_drive, cfg.intra_dot_bias,
    };
    for (double value : values) {
        if (!finite(value)) {
            throw QStateError("Quantum-dot configuration values must be finite");
        }
    }
}

inline void validate_coupling_config(const CouplingConfig& cfg) {
    const double values[] = {
        cfg.capacitive, cfg.spin_exchange, cfg.charge_tunneling, cfg.scale,
    };
    for (double value : values) {
        if (!finite(value)) {
            throw QStateError("Quantum-dot coupling values must be finite");
        }
    }
}

inline void validate_input(const DotInput& input) {
    if (!finite(input.theta) || !finite(input.phi) || !finite(input.strength)) {
        throw QStateError("Quantum-dot inputs must be finite");
    }
    if (input.strength < 0.0 || input.strength > 1.0) {
        throw QStateError("Quantum-dot input strength must be in [0, 1]");
    }
}

inline void validate_pocket_config(const PocketConfig& config) {
    if (config.dot_count == 0U) {
        throw QStateError("QuantumDotPocket requires at least one dot");
    }
    if (config.dot_count > std::numeric_limits<std::size_t>::max() / 2U) {
        throw QStateError("QuantumDotPocket logical qubit count would overflow");
    }
    if (!finite(config.dt) || !(config.dt > 0.0)) {
        throw QStateError("QuantumDotPocket dt must be finite and positive");
    }
    if (config.trotter_steps == 0U || config.trotter_steps > 1'000'000U) {
        throw QStateError("QuantumDotPocket trotter_steps must be in [1, 1000000]");
    }
    validate_dot_config(config.dot);
    validate_coupling_config(config.coupling);
    if (config.topology == Topology::PairBlocks) {
        if (config.dot_count < 2U || (config.dot_count % 2U) != 0U) {
            throw QStateError("PairBlocks topology requires a positive even dot count");
        }
    } else if (config.topology == Topology::PairPlusContext) {
        if ((config.dot_count % 2U) == 0U) {
            throw QStateError("PairPlusContext topology requires an odd dot count");
        }
    } else if (config.topology != Topology::Chain) {
        throw QStateError("Unknown quantum-dot topology");
    }
}

struct Coordinates {
    double x;
    double y;
    double z;
};

[[nodiscard]] inline Coordinates coordinates(const DotInput& input) {
    const double sine = std::sin(input.theta);
    return {
        input.strength * sine * std::cos(input.phi),
        input.strength * sine * std::sin(input.phi),
        input.strength * std::cos(input.theta),
    };
}

struct DiagonalCoefficients {
    double occ;
    double spin;
    double zz;
};

[[nodiscard]] inline DiagonalCoefficients diagonal_coefficients(
    const DotConfig& cfg, const Coordinates& c) {
    const double split = cfg.detuning_gain * c.z;
    const double transverse = 0.15 * cfg.detuning_gain * (c.x - c.y);
    const double e0 = cfg.e0;
    const double eup = cfg.eup + split + transverse;
    const double edown = cfg.edown - split - transverse;
    const double ex = cfg.ex + cfg.exciton_gain * (c.x * c.x + c.y * c.y) +
                      cfg.intra_dot_bias * c.z;
    return {
        0.25 * (e0 - eup + edown - ex),
        0.25 * (e0 + eup - edown - ex),
        0.25 * (e0 - eup - edown + ex),
    };
}

inline void require_finite_derived(double value, const char* label) {
    if (!finite(value)) {
        throw QStateError(std::string("Quantum-dot derived ") + label + " is non-finite");
    }
}

inline void validate_derived_step(
    const PocketConfig& config,
    std::span<const DotInput> inputs) {
    const double sub_dt = config.dt / static_cast<double>(config.trotter_steps);
    const double half = 0.5 * sub_dt;
    require_finite_derived(sub_dt, "sub-step duration");
    require_finite_derived(half, "half-step duration");

    for (const DotInput& input : inputs) {
        const Coordinates c = coordinates(input);
        require_finite_derived(c.x, "input x coordinate");
        require_finite_derived(c.y, "input y coordinate");
        require_finite_derived(c.z, "input z coordinate");
        const DiagonalCoefficients diag = diagonal_coefficients(config.dot, c);
        const double angles[] = {
            2.0 * diag.occ * half,
            2.0 * diag.spin * half,
            2.0 * diag.zz * half,
            config.dot.charge_drive * c.x * half,
            config.dot.charge_drive * c.y * half,
            config.dot.spin_drive * c.y * half,
            config.dot.spin_drive * (0.65 * c.x + 0.35 * c.z) * half,
        };
        for (double angle : angles) {
            require_finite_derived(angle, "local evolution angle");
        }
    }

    const double coupling_angles[] = {
        2.0 * config.coupling.capacitive * config.coupling.scale * sub_dt,
        config.coupling.spin_exchange * config.coupling.scale * sub_dt,
        config.coupling.charge_tunneling * config.coupling.scale * sub_dt,
    };
    for (double angle : coupling_angles) {
        require_finite_derived(angle, "coupling angle");
    }
}

inline void apply_single_to_vector(
    std::vector<QComplex>& amplitudes,
    std::size_t bit,
    const QMatrix2& matrix) {
    const std::size_t mask = std::size_t{1} << bit;
    for (std::size_t base = 0; base < amplitudes.size(); ++base) {
        if ((base & mask) != 0U) {
            continue;
        }
        const std::size_t one_index = base | mask;
        const QComplex zero = amplitudes[base];
        const QComplex one = amplitudes[one_index];
        amplitudes[base] = matrix(0, 0) * zero + matrix(0, 1) * one;
        amplitudes[one_index] = matrix(1, 0) * zero + matrix(1, 1) * one;
    }
}

inline void apply_cnot_to_vector(
    std::vector<QComplex>& amplitudes,
    std::size_t control_bit,
    std::size_t target_bit) {
    const std::size_t control = std::size_t{1} << control_bit;
    const std::size_t target = std::size_t{1} << target_bit;
    for (std::size_t basis = 0; basis < amplitudes.size(); ++basis) {
        if ((basis & control) != 0U && (basis & target) == 0U) {
            std::swap(amplitudes[basis], amplitudes[basis | target]);
        }
    }
}

[[nodiscard]] inline double norm2(const std::vector<QComplex>& amplitudes) {
    long double total = 0.0L;
    for (const auto& value : amplitudes) {
        total += static_cast<long double>(value.norm2());
    }
    return static_cast<double>(total);
}

inline void reference_zz(QRegister& state, QubitId first, QubitId second, double angle) {
    state.apply_cnot(first, second);
    state.apply_rz(second, angle);
    state.apply_cnot(first, second);
}

inline void reference_xx(QRegister& state, QubitId first, QubitId second, double angle) {
    state.apply_h(first);
    state.apply_h(second);
    reference_zz(state, first, second, angle);
    state.apply_h(first);
    state.apply_h(second);
}

inline void reference_yy(QRegister& state, QubitId first, QubitId second, double angle) {
    state.apply_sdg(first);
    state.apply_h(first);
    state.apply_sdg(second);
    state.apply_h(second);
    reference_zz(state, first, second, angle);
    state.apply_h(first);
    state.apply_s(first);
    state.apply_h(second);
    state.apply_s(second);
}

inline void reference_local(
    QRegister& state,
    std::size_t dot,
    const DotConfig& cfg,
    const DotInput& input,
    double duration) {
    const QubitId occ = static_cast<QubitId>(2U * dot);
    const QubitId spin = static_cast<QubitId>(2U * dot + 1U);
    const Coordinates c = coordinates(input);
    const auto diag = diagonal_coefficients(cfg, c);
    state.apply_rz(occ, 2.0 * diag.occ * duration);
    state.apply_rz(spin, 2.0 * diag.spin * duration);
    reference_zz(state, occ, spin, 2.0 * diag.zz * duration);
    state.apply_rx(occ, cfg.charge_drive * c.x * duration);
    state.apply_ry(occ, cfg.charge_drive * c.y * duration);
    state.apply_rx(spin, cfg.spin_drive * c.y * duration);
    state.apply_ry(spin, cfg.spin_drive * (0.65 * c.x + 0.35 * c.z) * duration);
}

}  // namespace qubit::qdot::detail

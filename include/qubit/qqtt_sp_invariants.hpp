#pragma once

#include "qubit/qqtt_schrodinger_poisson.hpp"

#include <cmath>

namespace qubit {

struct QTTSchrodingerPoissonDiagnostics {
    double wave_mean_norm_squared{0.0};
    QComplex hamiltonian_expectation{};
    double norm_rate{0.0};
};

[[nodiscard]] inline QTTSchrodingerPoissonDiagnostics diagnose_schrodinger_poisson(
    const ExactComplexWalshField& wave,
    const QTTSchrodingerPoissonResult& result) {
    if (wave.logical_bits() != result.density.logical_bits() ||
        wave.logical_bits() != result.hamiltonian_wave.logical_bits() ||
        wave.logical_bits() != result.rhs.logical_bits()) {
        throw QStateError("Schrodinger-Poisson diagnostic requires matching logical shapes");
    }

    const QComplex norm = wave.mean_inner_product(wave);
    const QComplex hamiltonian_expectation =
        wave.mean_inner_product(result.hamiltonian_wave);
    const QComplex rhs_overlap = wave.mean_inner_product(result.rhs);
    const double scale = 1.0 + std::abs(norm.re);
    if (!std::isfinite(norm.re) || !std::isfinite(norm.im) ||
        !std::isfinite(hamiltonian_expectation.re) ||
        !std::isfinite(hamiltonian_expectation.im) ||
        !std::isfinite(rhs_overlap.re) || !std::isfinite(rhs_overlap.im)) {
        throw QStateError("Schrodinger-Poisson invariant diagnostic became non-finite");
    }
    if (std::abs(norm.im) > 1e-12 * scale ||
        (norm.re < 0.0 && std::abs(norm.re) > 1e-12 * scale)) {
        throw QStateError("Schrodinger-Poisson wave norm is not physically admissible");
    }

    return QTTSchrodingerPoissonDiagnostics{
        norm.re < 0.0 ? 0.0 : norm.re,
        hamiltonian_expectation,
        2.0 * rhs_overlap.re,
    };
}

}  // namespace qubit

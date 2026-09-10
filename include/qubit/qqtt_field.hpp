#pragma once

#include "qubit/qqtt_operator.hpp"

#include <cmath>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace qubit {

struct QTTFieldStats {
    std::size_t logical_bits{0U};
    std::size_t linear_operator_rank{0U};
    std::size_t linear_operator_scalars{0U};
};

class ExactQTTFieldHamiltonian {
public:
    [[nodiscard]] static ExactQTTFieldHamiltonian hypercube(
        std::span<const double> laplacian_weights,
        const ExactQTTFunction& potential,
        double kinetic_coupling,
        double nonlinear_coupling,
        QTTOperatorConfig operator_config = {}) {
        if (laplacian_weights.size() != potential.logical_bits()) {
            throw QStateError("QTT field Hamiltonian requires matching logical shapes");
        }
        if (!std::isfinite(kinetic_coupling) || !std::isfinite(nonlinear_coupling)) {
            throw QStateError("QTT field Hamiltonian couplings must be finite");
        }
        ExactQTTOperator linear = ExactQTTOperator::weighted_hypercube_laplacian(
            laplacian_weights, operator_config).scaled(QComplex{-kinetic_coupling});
        linear = linear.add(ExactQTTOperator::diagonal(potential, operator_config));
        return ExactQTTFieldHamiltonian(std::move(linear), nonlinear_coupling);
    }

    [[nodiscard]] std::size_t logical_bits() const noexcept { return linear_.logical_bits(); }
    [[nodiscard]] double nonlinear_coupling() const noexcept { return nonlinear_coupling_; }
    [[nodiscard]] const ExactQTTOperator& linear_operator() const noexcept { return linear_; }
    [[nodiscard]] const QTTFieldStats& stats() const noexcept { return stats_; }

    [[nodiscard]] static ExactQTTFunction conjugate(const ExactQTTFunction& field) {
        std::vector<QTTCore> cores = field.cores();
        for (QTTCore& core : cores) {
            for (QComplex& value : core.zero) {
                value = value.conjugate();
            }
            for (QComplex& value : core.one) {
                value = value.conjugate();
            }
        }
        return ExactQTTFunction::from_certified_cores(std::move(cores), field.config());
    }

    [[nodiscard]] static ExactQTTFunction density(const ExactQTTFunction& field) {
        return conjugate(field).hadamard(field);
    }

    [[nodiscard]] ExactQTTFunction apply(const ExactQTTFunction& field) const {
        require_shape(field);
        ExactQTTFunction result = linear_.apply(field);
        if (nonlinear_coupling_ == 0.0) {
            return result;
        }
        ExactQTTFunction nonlinear = density(field).hadamard(field).scaled(nonlinear_coupling_);
        return result.add(nonlinear);
    }

    [[nodiscard]] ExactQTTFunction rhs(const ExactQTTFunction& field) const {
        return apply(field).scaled(QComplex{0.0, -1.0});
    }

    [[nodiscard]] QComplex energy(const ExactQTTFunction& field) const {
        require_shape(field);
        const ExactQTTFunction linear = linear_.apply(field);
        QComplex result = field.inner_product(linear);
        if (nonlinear_coupling_ != 0.0) {
            const ExactQTTFunction rho = density(field);
            result += rho.inner_product(rho) * (0.5 * nonlinear_coupling_);
        }
        return result;
    }

private:
    ExactQTTOperator linear_;
    double nonlinear_coupling_{0.0};
    QTTFieldStats stats_{};

    ExactQTTFieldHamiltonian(ExactQTTOperator linear, double nonlinear_coupling)
        : linear_(std::move(linear)), nonlinear_coupling_(nonlinear_coupling) {
        stats_ = QTTFieldStats{
            linear_.logical_bits(),
            linear_.stats().maximum_rank,
            linear_.stats().descriptor_scalars,
        };
    }

    void require_shape(const ExactQTTFunction& field) const {
        if (field.logical_bits() != linear_.logical_bits()) {
            throw QStateError("QTT field Hamiltonian requires matching logical shapes");
        }
    }
};

}  // namespace qubit

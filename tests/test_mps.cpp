#include "qubit/qmps.hpp"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using qubit::MPSConfig;
using qubit::MPSPauliPlan;
using qubit::MPSSiteTensor;
using qubit::MatrixProductState;
using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
using qubit::QComplex;
using qubit::QRegister;
using qubit::QStateError;
using qubit::QubitId;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void compare_dense(
    const MatrixProductState& mps,
    const QRegister& reference,
    double tolerance,
    const std::string& label) {
    const std::vector<QComplex> actual = mps.materialize();
    const std::vector<QComplex> expected = reference.materialize(20U);
    require(actual.size() == expected.size(), label + ": dimensions differ");
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        if (!qubit::almost_equal(actual[index], expected[index], tolerance)) {
            throw std::runtime_error(label + ": amplitude mismatch at " + std::to_string(index));
        }
    }
}

PauliObservable single_term(
    std::size_t qubits,
    std::vector<PauliFactor> factors) {
    PauliObservable observable(qubits);
    observable.add_term({1.0, 0.0}, factors);
    return observable;
}

}  // namespace

int main() {
    {
        constexpr std::size_t qubits = 8U;
        const MatrixProductState mps = MatrixProductState::ghz(qubits);
        QRegister reference(qubits);
        reference.apply_h(0U);
        for (std::size_t target = 1U; target < qubits; ++target) {
            reference.apply_cnot(0U, static_cast<QubitId>(target));
        }
        compare_dense(mps, reference, 2e-12, "GHZ MPS");
        require(std::abs(mps.norm2() - 1.0) <= 2e-12, "GHZ MPS is not normalized");
        require(mps.max_bond_dimension() == 2U, "GHZ MPS bond dimension changed");
    }

    {
        constexpr std::size_t qubits = 8U;
        const MatrixProductState mps = MatrixProductState::cluster(qubits);
        QRegister reference(qubits);
        for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
            reference.apply_h(static_cast<QubitId>(qubit));
        }
        for (std::size_t qubit = 0U; qubit + 1U < qubits; ++qubit) {
            reference.apply_cz(
                static_cast<QubitId>(qubit),
                static_cast<QubitId>(qubit + 1U));
        }
        compare_dense(mps, reference, 2e-11, "cluster MPS");

        const PauliObservable stabilizer = single_term(
            qubits,
            {
                {2U, PauliAxis::Z},
                {3U, PauliAxis::X},
                {4U, PauliAxis::Z},
            });
        const QComplex mps_value = mps.expectation(stabilizer);
        const QComplex reference_value = stabilizer.expectation(reference);
        const MPSPauliPlan plan(MatrixProductState::cluster(qubits));
        const QComplex plan_value = plan.expectation(stabilizer);
        require(qubit::almost_equal(mps_value, reference_value, 2e-11),
                "cluster MPS Pauli expectation disagrees with QRegister");
        require(qubit::almost_equal(plan_value, mps_value, 2e-11),
                "compiled cluster MPS Pauli expectation disagrees with direct MPS");
        require(std::abs(mps_value.re - 1.0) <= 2e-11 && std::abs(mps_value.im) <= 2e-11,
                "cluster stabilizer expectation is not one");
    }

    {
        constexpr std::size_t qubits = 10U;
        constexpr double phase = std::numbers::pi / 5.0;
        const MatrixProductState mps = MatrixProductState::ghz(qubits, phase);
        std::vector<PauliFactor> factors;
        factors.reserve(qubits);
        for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
            factors.push_back({static_cast<QubitId>(qubit), PauliAxis::X});
        }
        const PauliObservable observable = single_term(qubits, std::move(factors));
        const QComplex value = mps.expectation(observable);
        const MPSPauliPlan plan(MatrixProductState::ghz(qubits, phase));
        const QComplex plan_value = plan.expectation(observable);
        require(std::abs(value.re - std::cos(phase)) <= 2e-11 && std::abs(value.im) <= 2e-11,
                "phased GHZ global-X expectation is incorrect");
        require(qubit::almost_equal(plan_value, value, 2e-11),
                "compiled global-span GHZ expectation disagrees with direct MPS");
    }

    {
        constexpr std::size_t qubits = 11U;
        const MatrixProductState mps = MatrixProductState::w(qubits);
        const std::vector<QComplex> dense = mps.materialize();
        const double scale = 1.0 / std::sqrt(static_cast<double>(qubits));
        for (std::size_t basis = 0U; basis < dense.size(); ++basis) {
            const unsigned count = std::popcount(static_cast<unsigned long long>(basis));
            const double expected = count == 1U ? scale : 0.0;
            require(std::abs(dense[basis].re - expected) <= 2e-12 &&
                        std::abs(dense[basis].im) <= 2e-12,
                    "W MPS amplitude is incorrect");
        }
        const PauliObservable z0 = single_term(qubits, {{0U, PauliAxis::Z}});
        const QComplex z = mps.expectation(z0);
        const MPSPauliPlan plan(MatrixProductState::w(qubits));
        require(qubit::almost_equal(plan.expectation(z0), z, 2e-12),
                "compiled W local-Z expectation disagrees with direct MPS");
        require(std::abs(z.re - static_cast<double>(qubits - 2U) / static_cast<double>(qubits)) <=
                    2e-12,
                "W MPS local-Z expectation is incorrect");
    }

    {
        constexpr std::size_t qubits = 30'000U;
        const MatrixProductState mps = MatrixProductState::cluster(qubits);
        require(mps.qubit_count() == qubits, "large cluster MPS width changed");
        require(mps.max_bond_dimension() == 2U, "large cluster MPS bond dimension changed");
        require(mps.scalar_count() == 239'992U, "large cluster MPS scalar count changed");
        require(std::abs(mps.norm2() - 1.0) <= 2e-9, "large cluster MPS norm drifted");
        const std::size_t center = qubits / 2U;
        const PauliObservable stabilizer = single_term(
            qubits,
            {
                {static_cast<QubitId>(center - 1U), PauliAxis::Z},
                {static_cast<QubitId>(center), PauliAxis::X},
                {static_cast<QubitId>(center + 1U), PauliAxis::Z},
            });
        const QComplex direct = mps.expectation(stabilizer);
        const MPSPauliPlan plan(MatrixProductState::cluster(qubits));
        const QComplex compiled = plan.expectation(stabilizer);
        require(plan.environment_scalar_count() == 239'996U,
                "large cluster MPS environment cache size changed");
        require(qubit::almost_equal(compiled, direct, 2e-9),
                "large compiled local MPS expectation disagrees with direct MPS");
        require(std::abs(direct.re - 1.0) <= 2e-9 && std::abs(direct.im) <= 2e-9,
                "large cluster MPS stabilizer expectation is incorrect");
        require(mps.validate(), "large cluster MPS failed validation");
    }

    {
        MPSConfig config;
        config.max_bond_dimension = 1U;
        bool rejected = false;
        try {
            static_cast<void>(MatrixProductState::cluster(8U, config));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "MPS bond limit did not fail closed");
    }

    {
        MPSConfig config;
        config.max_scalars = 7U;
        bool rejected = false;
        try {
            static_cast<void>(MatrixProductState::cluster(2U, config));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "MPS scalar limit did not fail closed");
    }

    {
        std::vector<MPSSiteTensor> sites{{
            1U,
            1U,
            {{2.0, 0.0}},
            {{0.0, 0.0}},
        }};
        bool rejected = false;
        try {
            static_cast<void>(MatrixProductState(std::move(sites)));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "nonnormalized MPS was accepted");
    }

    {
        const MatrixProductState mps = MatrixProductState::cluster(21U);
        bool rejected = false;
        try {
            static_cast<void>(mps.materialize());
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "MPS materialization width limit did not fail closed");
    }

    {
        bool rejected = false;
        try {
            static_cast<void>(MPSPauliPlan(MatrixProductState::cluster(8U), 59U));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "MPS Pauli environment limit did not fail closed");
    }

    {
        constexpr std::size_t bond_cap = 256U;
        for (std::size_t crossing_pairs = 1U; crossing_pairs <= 12U; ++crossing_pairs) {
            const std::size_t required_rank =
                qubit::required_schmidt_rank_cross_cut_bell_pairs(crossing_pairs);
            require((required_rank <= bond_cap) ==
                        qubit::bond_dimension_accepts_cross_cut_bell_pairs(
                            crossing_pairs, bond_cap),
                    "cross-cut Bell-pair bond certificate is inconsistent");
            require((required_rank <= bond_cap) == (crossing_pairs <= 8U),
                    "cross-cut Bell-pair Schmidt-rank collapse boundary changed");
        }
    }

    std::cout << "matrix-product state tests passed\n";
    return 0;
}

#include "qubit/qmps.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <optional>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::MPSPauliPlan;
using qubit::MatrixProductState;
using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
using qubit::QComplex;
using qubit::QRegister;
using qubit::QubitId;

template <class Function>
[[nodiscard]] double median_ms(
    Function&& function,
    int repeats = 7,
    int iterations = 1) {
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto start = Clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            function();
        }
        const auto finish = Clock::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(finish - start).count() /
            static_cast<double>(iterations));
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

PauliObservable cluster_stabilizer(std::size_t qubits, std::size_t center) {
    PauliObservable observable(qubits);
    const std::vector<PauliFactor> factors{
        {static_cast<QubitId>(center - 1U), PauliAxis::Z},
        {static_cast<QubitId>(center), PauliAxis::X},
        {static_cast<QubitId>(center + 1U), PauliAxis::Z},
    };
    observable.add_term({1.0, 0.0}, factors);
    return observable;
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);

    {
        constexpr std::size_t qubits = 18U;
        QRegister reference(qubits);
        const double qregister_setup_ms = median_ms([&] {
            QRegister next(qubits);
            for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
                next.apply_h(static_cast<QubitId>(qubit));
            }
            for (std::size_t qubit = 0U; qubit + 1U < qubits; ++qubit) {
                next.apply_cz(
                    static_cast<QubitId>(qubit),
                    static_cast<QubitId>(qubit + 1U));
            }
            reference = std::move(next);
        });

        std::optional<MatrixProductState> mps;
        const double mps_setup_ms = median_ms([&] {
            mps.emplace(MatrixProductState::cluster(qubits));
        });
        std::optional<MPSPauliPlan> plan;
        const double plan_setup_ms = median_ms([&] {
            plan.emplace(MatrixProductState::cluster(qubits));
        });

        const PauliObservable observable = cluster_stabilizer(qubits, qubits / 2U);
        QComplex reference_value{};
        QComplex mps_value{};
        QComplex plan_value{};
        const double qregister_query_ms = median_ms([&] {
            reference_value = observable.expectation(reference);
        });
        const double mps_query_ms = median_ms([&] {
            mps_value = mps->expectation(observable);
        }, 11, 100);
        const double plan_query_ms = median_ms([&] {
            plan_value = plan->expectation(observable);
        }, 11, 1000);

        std::cout << "mps18_qubits=" << qubits << '\n';
        std::cout << "mps18_qregister_setup_ms=" << qregister_setup_ms << '\n';
        std::cout << "mps18_setup_ms=" << mps_setup_ms << '\n';
        std::cout << "mps18_plan_setup_ms=" << plan_setup_ms << '\n';
        std::cout << "mps18_qregister_query_ms=" << qregister_query_ms << '\n';
        std::cout << "mps18_query_ms=" << mps_query_ms << '\n';
        std::cout << "mps18_plan_query_ms=" << plan_query_ms << '\n';
        std::cout << "mps18_query_ratio=" << qregister_query_ms / mps_query_ms << '\n';
        std::cout << "mps18_plan_over_direct_query_speed=" << mps_query_ms / plan_query_ms << '\n';
        std::cout << "mps18_value_error=" << (reference_value - mps_value).magnitude() << '\n';
        std::cout << "mps18_plan_value_error=" << (mps_value - plan_value).magnitude() << '\n';
        std::cout << "mps18_qregister_bytes=" << reference.estimated_bytes() << '\n';
        std::cout << "mps18_bytes=" << mps->estimated_bytes() << '\n';
        std::cout << "mps18_plan_bytes=" << plan->estimated_bytes() << '\n';
        std::cout << "mps18_plan_environment_scalars=" << plan->environment_scalar_count() << '\n';
        std::cout << "mps18_scalars=" << mps->scalar_count() << '\n';
        std::cout << "mps18_max_bond=" << mps->max_bond_dimension() << '\n';
        std::cout << "mps18_end_to_end_ratio="
                  << (qregister_setup_ms + qregister_query_ms) / (mps_setup_ms + mps_query_ms)
                  << '\n';
    }

    {
        constexpr std::size_t qubits = 30'000U;
        std::optional<MatrixProductState> mps;
        const double setup_ms = median_ms([&] {
            mps.emplace(MatrixProductState::cluster(qubits));
        }, 3);
        std::optional<MPSPauliPlan> plan;
        const double plan_setup_ms = median_ms([&] {
            plan.emplace(MatrixProductState::cluster(qubits));
        }, 3);
        const PauliObservable observable = cluster_stabilizer(qubits, qubits / 2U);
        QComplex value{};
        QComplex plan_value{};
        const double query_ms = median_ms([&] {
            value = mps->expectation(observable);
        }, 7);
        const double plan_query_ms = median_ms([&] {
            plan_value = plan->expectation(observable);
        }, 11, 1000);

        std::cout << "mps_large_qubits=" << qubits << '\n';
        std::cout << "mps_large_setup_ms=" << setup_ms << '\n';
        std::cout << "mps_large_plan_setup_ms=" << plan_setup_ms << '\n';
        std::cout << "mps_large_query_ms=" << query_ms << '\n';
        std::cout << "mps_large_plan_query_ms=" << plan_query_ms << '\n';
        std::cout << "mps_large_plan_over_direct_query_speed=" << query_ms / plan_query_ms << '\n';
        std::cout << "mps_large_plan_value_error=" << (value - plan_value).magnitude() << '\n';
        std::cout << "mps_large_bytes=" << mps->estimated_bytes() << '\n';
        std::cout << "mps_large_plan_bytes=" << plan->estimated_bytes() << '\n';
        std::cout << "mps_large_plan_environment_scalars=" << plan->environment_scalar_count() << '\n';
        std::cout << "mps_large_scalars=" << mps->scalar_count() << '\n';
        std::cout << "mps_large_max_bond=" << mps->max_bond_dimension() << '\n';
        std::cout << "mps_large_value_real=" << value.re << '\n';
        std::cout << "mps_large_value_imag=" << value.im << '\n';
        std::cout << "mps_large_norm_error=" << std::abs(mps->norm2() - 1.0) << '\n';
    }

    return 0;
}

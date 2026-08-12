#include "qubit/qbroker.hpp"
#include "qubit/qmps.hpp"

#include <algorithm>
#include <array>
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
using qubit::ExactExecutionBroker;
using qubit::ExactExecutionRoute;
using qubit::ExactPreparedExpectationPlan;
using qubit::ExactPreparedProbabilityPlan;
using qubit::MPSPauliPlan;
using qubit::MatrixProductState;
using qubit::Operation;
using qubit::OperationCode;
using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
using qubit::PauliPropagationConfig;
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

MatrixProductState evolved_cluster(std::size_t qubits) {
    MatrixProductState state = MatrixProductState::zero(qubits);
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        state.apply_unitary(qubit, qubit::gates::h());
    }
    for (std::size_t qubit = 0U; qubit + 1U < qubits; ++qubit) {
        state.apply_cz(qubit, qubit + 1U);
    }
    return state;
}

std::vector<Operation> cluster_operations(std::size_t qubits, double center_rz) {
    std::vector<Operation> operations;
    operations.reserve(2U * qubits);
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({
            OperationCode::H,
            static_cast<QubitId>(qubit),
            0U,
            0.0,
            0.0,
        });
    }
    for (std::size_t qubit = 0U; qubit + 1U < qubits; ++qubit) {
        operations.push_back({
            OperationCode::Cz,
            static_cast<QubitId>(qubit),
            static_cast<QubitId>(qubit + 1U),
            0.0,
            0.0,
        });
    }
    operations.push_back({
        OperationCode::Rz,
        static_cast<QubitId>(qubits / 2U),
        0U,
        center_rz,
        0.0,
    });
    return operations;
}

PauliObservable causal_collapse_observable(std::size_t qubits) {
    PauliPropagationConfig config;
    config.max_terms = 1U;
    PauliObservable observable(qubits, config);
    const std::vector<PauliFactor> factors{{
        static_cast<QubitId>(qubits / 2U),
        PauliAxis::X,
    }};
    observable.add_term({1.0, 0.0}, factors);
    return observable;
}

[[nodiscard]] double break_even_queries(
    double setup_ms,
    double one_shot_ms,
    double prepared_query_ms) noexcept {
    if (one_shot_ms <= prepared_query_ms) {
        return -1.0;
    }
    return setup_ms / (one_shot_ms - prepared_query_ms);
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
        std::optional<MatrixProductState> evolved;
        const double evolved_setup_ms = median_ms([&] {
            evolved.emplace(evolved_cluster(qubits));
        });
        std::optional<MPSPauliPlan> plan;
        const double plan_setup_ms = median_ms([&] {
            plan.emplace(MatrixProductState::cluster(qubits));
        });

        const PauliObservable observable = cluster_stabilizer(qubits, qubits / 2U);
        QComplex reference_value{};
        QComplex mps_value{};
        QComplex evolved_value{};
        QComplex plan_value{};
        const double qregister_query_ms = median_ms([&] {
            reference_value = observable.expectation(reference);
        });
        const double mps_query_ms = median_ms([&] {
            mps_value = mps->expectation(observable);
        }, 11, 100);
        const double evolved_query_ms = median_ms([&] {
            evolved_value = evolved->expectation(observable);
        }, 11, 100);
        const double plan_query_ms = median_ms([&] {
            plan_value = plan->expectation(observable);
        }, 11, 1000);

        const std::vector<Operation> broker_operations = cluster_operations(qubits, 0.37);
        const PauliObservable broker_observable = causal_collapse_observable(qubits);

        ExactExecutionBroker broker;
        qubit::ExactExpectationResult broker_result;
        const double broker_ms = median_ms([&] {
            broker_result = broker.expectation_from_zero(
                qubits,
                broker_operations,
                broker_observable);
        });

        QRegister broker_reference(qubits);
        QComplex broker_reference_value{};
        const double broker_qregister_ms = median_ms([&] {
            QRegister next(qubits);
            qubit::OperationPlan operation_plan(broker_operations);
            operation_plan.execute(next);
            broker_reference_value = broker_observable.expectation(next);
            broker_reference = std::move(next);
        });
        MatrixProductState broker_mps = evolved_cluster(qubits);
        broker_mps.apply_unitary(qubits / 2U, qubit::gates::rz(0.37));

        std::optional<ExactPreparedExpectationPlan> prepared;
        const double prepared_setup_ms = median_ms([&] {
            prepared.emplace(qubits, broker_operations);
        });
        qubit::ExactExpectationResult prepared_result;
        const double prepared_query_ms = median_ms([&] {
            prepared_result = prepared->expectation(broker_observable);
        }, 11, 1000);

        qubit::ExactExecutionBrokerConfig probability_config;
        probability_config.tensor.max_contraction_entries = 8U;
        ExactExecutionBroker probability_broker(probability_config);
        std::vector<Operation> probability_operations = broker_operations;
        probability_operations.push_back({
            OperationCode::Ry,
            static_cast<QubitId>(qubits / 2U),
            0U,
            0.19,
            0.0,
        });
        const std::vector<std::uint8_t> probability_bits(qubits, 0U);
        qubit::ExactProbabilityResult probability_result;
        const double probability_ms = median_ms([&] {
            probability_result = probability_broker.basis_probability_from_zero(
                qubits,
                probability_operations,
                probability_bits);
        });
        double probability_reference_value = 0.0;
        const double probability_qregister_ms = median_ms([&] {
            QRegister next(qubits);
            qubit::OperationPlan operation_plan(probability_operations);
            operation_plan.execute(next);
            probability_reference_value = next.amplitude_bits(probability_bits).norm2();
        });

        std::optional<ExactPreparedProbabilityPlan> probability_prepared;
        const double probability_prepared_setup_ms = median_ms([&] {
            probability_prepared.emplace(
                qubits,
                probability_operations,
                probability_config);
        });
        qubit::ExactProbabilityResult probability_prepared_result;
        const double probability_prepared_query_ms = median_ms([&] {
            probability_prepared_result = probability_prepared->probability(probability_bits);
        }, 11, 100);

        if (broker_result.route != ExactExecutionRoute::PersistentMPS) {
            std::cerr << "broker did not select persistent MPS after causal collapse\n";
            return 2;
        }
        if (prepared_result.route != ExactExecutionRoute::PersistentMPS ||
            prepared->prepared_fallback_route() != ExactExecutionRoute::PersistentMPS) {
            std::cerr << "prepared broker did not retain persistent MPS fallback\n";
            return 3;
        }
        if (probability_result.route != ExactExecutionRoute::PersistentMPS) {
            std::cerr << "probability broker did not select persistent MPS after tensor collapse\n";
            return 4;
        }
        if (probability_prepared->prepared_route() != ExactExecutionRoute::PersistentMPS ||
            probability_prepared_result.route != probability_result.route ||
            std::abs(probability_prepared_result.value - probability_result.value) > 2e-11) {
            std::cerr << "prepared probability broker did not retain persistent MPS semantics\n";
            return 5;
        }

        std::cout << "mps18_qubits=" << qubits << '\n';
        std::cout << "mps18_qregister_setup_ms=" << qregister_setup_ms << '\n';
        std::cout << "mps18_setup_ms=" << mps_setup_ms << '\n';
        std::cout << "mps18_evolved_setup_ms=" << evolved_setup_ms << '\n';
        std::cout << "mps18_plan_setup_ms=" << plan_setup_ms << '\n';
        std::cout << "mps18_qregister_query_ms=" << qregister_query_ms << '\n';
        std::cout << "mps18_query_ms=" << mps_query_ms << '\n';
        std::cout << "mps18_evolved_query_ms=" << evolved_query_ms << '\n';
        std::cout << "mps18_plan_query_ms=" << plan_query_ms << '\n';
        std::cout << "mps18_query_ratio=" << qregister_query_ms / mps_query_ms << '\n';
        std::cout << "mps18_evolved_setup_ratio=" << qregister_setup_ms / evolved_setup_ms << '\n';
        std::cout << "mps18_plan_over_direct_query_speed=" << mps_query_ms / plan_query_ms << '\n';
        std::cout << "mps18_value_error=" << (reference_value - mps_value).magnitude() << '\n';
        std::cout << "mps18_evolved_value_error=" << (mps_value - evolved_value).magnitude() << '\n';
        std::cout << "mps18_plan_value_error=" << (mps_value - plan_value).magnitude() << '\n';
        std::cout << "mps18_qregister_bytes=" << reference.estimated_bytes() << '\n';
        std::cout << "mps18_bytes=" << mps->estimated_bytes() << '\n';
        std::cout << "mps18_evolved_bytes=" << evolved->estimated_bytes() << '\n';
        std::cout << "mps18_plan_bytes=" << plan->estimated_bytes() << '\n';
        std::cout << "mps18_plan_environment_scalars=" << plan->environment_scalar_count() << '\n';
        std::cout << "mps18_scalars=" << mps->scalar_count() << '\n';
        std::cout << "mps18_evolved_scalars=" << evolved->scalar_count() << '\n';
        std::cout << "mps18_max_bond=" << mps->max_bond_dimension() << '\n';
        std::cout << "mps18_evolved_max_bond=" << evolved->max_bond_dimension() << '\n';
        std::cout << "mps18_end_to_end_ratio="
                  << (qregister_setup_ms + qregister_query_ms) / (mps_setup_ms + mps_query_ms)
                  << '\n';
        std::cout << "mps18_broker_route="
                  << qubit::exact_execution_route_name(broker_result.route) << '\n';
        std::cout << "mps18_broker_ms=" << broker_ms << '\n';
        std::cout << "mps18_broker_qregister_ms=" << broker_qregister_ms << '\n';
        std::cout << "mps18_broker_speed_ratio=" << broker_qregister_ms / broker_ms << '\n';
        std::cout << "mps18_broker_value_error="
                  << (broker_result.value - broker_reference_value).magnitude() << '\n';
        std::cout << "mps18_broker_qregister_bytes=" << broker_reference.estimated_bytes() << '\n';
        std::cout << "mps18_broker_mps_bytes=" << broker_mps.estimated_bytes() << '\n';
        std::cout << "mps18_broker_mps_scalars=" << broker_mps.scalar_count() << '\n';
        std::cout << "mps18_broker_mps_max_bond=" << broker_mps.max_bond_dimension() << '\n';
        std::cout << "mps18_prepared_setup_ms=" << prepared_setup_ms << '\n';
        std::cout << "mps18_prepared_query_ms=" << prepared_query_ms << '\n';
        std::cout << "mps18_prepared_over_one_shot_speed=" << broker_ms / prepared_query_ms << '\n';
        std::cout << "mps18_prepared_break_even_queries="
                  << break_even_queries(prepared_setup_ms, broker_ms, prepared_query_ms) << '\n';
        std::cout << "mps18_prepared_bytes=" << prepared->estimated_bytes() << '\n';
        std::cout << "mps18_prepared_value_error="
                  << (prepared_result.value - broker_reference_value).magnitude() << '\n';
        std::cout << "mps18_probability_broker_route="
                  << qubit::exact_execution_route_name(probability_result.route) << '\n';
        std::cout << "mps18_probability_broker_ms=" << probability_ms << '\n';
        std::cout << "mps18_probability_qregister_ms=" << probability_qregister_ms << '\n';
        std::cout << "mps18_probability_broker_speed_ratio="
                  << probability_qregister_ms / probability_ms << '\n';
        std::cout << "mps18_probability_value_error="
                  << std::abs(probability_result.value - probability_reference_value) << '\n';
        std::cout << "mps18_probability_prepared_setup_ms=" << probability_prepared_setup_ms << '\n';
        std::cout << "mps18_probability_prepared_query_ms=" << probability_prepared_query_ms << '\n';
        std::cout << "mps18_probability_prepared_over_one_shot_speed="
                  << probability_ms / probability_prepared_query_ms << '\n';
        std::cout << "mps18_probability_prepared_break_even_queries="
                  << break_even_queries(
                         probability_prepared_setup_ms,
                         probability_ms,
                         probability_prepared_query_ms) << '\n';
        std::cout << "mps18_probability_prepared_bytes="
                  << probability_prepared->estimated_bytes() << '\n';
        std::cout << "mps18_probability_prepared_value_error="
                  << std::abs(probability_prepared_result.value - probability_reference_value) << '\n';
    }

    {
        constexpr std::size_t qubits = 30'000U;
        std::optional<MatrixProductState> mps;
        const double setup_ms = median_ms([&] {
            mps.emplace(MatrixProductState::cluster(qubits));
        }, 3);
        std::optional<MatrixProductState> evolved;
        const double evolved_setup_ms = median_ms([&] {
            evolved.emplace(evolved_cluster(qubits));
        }, 3);
        std::optional<MPSPauliPlan> plan;
        const double plan_setup_ms = median_ms([&] {
            plan.emplace(MatrixProductState::cluster(qubits));
        }, 3);
        const PauliObservable observable = cluster_stabilizer(qubits, qubits / 2U);
        QComplex value{};
        QComplex evolved_value{};
        QComplex plan_value{};
        const double query_ms = median_ms([&] {
            value = mps->expectation(observable);
        }, 7);
        const double evolved_query_ms = median_ms([&] {
            evolved_value = evolved->expectation(observable);
        }, 7);
        const double plan_query_ms = median_ms([&] {
            plan_value = plan->expectation(observable);
        }, 11, 1000);

        const std::vector<Operation> broker_operations = cluster_operations(qubits, 0.37);
        const PauliObservable broker_observable = causal_collapse_observable(qubits);
        ExactExecutionBroker broker;
        qubit::ExactExpectationResult broker_result;
        const double broker_ms = median_ms([&] {
            broker_result = broker.expectation_from_zero(
                qubits,
                broker_operations,
                broker_observable);
        }, 3);

        std::optional<ExactPreparedExpectationPlan> prepared;
        const double prepared_setup_ms = median_ms([&] {
            prepared.emplace(qubits, broker_operations);
        }, 3);
        qubit::ExactExpectationResult prepared_result;
        const double prepared_query_ms = median_ms([&] {
            prepared_result = prepared->expectation(broker_observable);
        }, 11, 500);

        MatrixProductState broker_reference = evolved_cluster(qubits);
        broker_reference.apply_unitary(qubits / 2U, qubit::gates::rz(0.37));
        const QComplex broker_reference_value = broker_reference.expectation(broker_observable);

        const std::size_t center = qubits / 2U;
        std::vector<Operation> marginal_operations = broker_operations;
        marginal_operations.push_back({
            OperationCode::Ry,
            static_cast<QubitId>(center),
            0U,
            0.19,
            0.0,
        });
        MatrixProductState marginal_reference = broker_reference;
        marginal_reference.apply_unitary(center, qubit::gates::ry(0.19));
        const std::array<QubitId, 3> marginal_qubits{{
            static_cast<QubitId>(center - 1U),
            static_cast<QubitId>(center),
            static_cast<QubitId>(center + 1U),
        }};
        const std::array<std::uint8_t, 3> marginal_bits{{0U, 1U, 0U}};

        double marginal_direct_value = 0.0;
        const double marginal_direct_ms = median_ms([&] {
            marginal_direct_value = marginal_reference.marginal_probability(
                marginal_qubits, marginal_bits);
        }, 7);
        std::optional<MPSPauliPlan> marginal_plan;
        const double marginal_plan_setup_ms = median_ms([&] {
            marginal_plan.emplace(marginal_reference);
        }, 3);
        double marginal_plan_value = 0.0;
        const double marginal_plan_ms = median_ms([&] {
            marginal_plan_value = marginal_plan->marginal_probability(
                marginal_qubits, marginal_bits);
        }, 11, 1000);

        qubit::ExactExecutionBrokerConfig marginal_config;
        marginal_config.tensor.max_contraction_entries = 8U;
        ExactExecutionBroker marginal_broker(marginal_config);
        qubit::ExactProbabilityResult marginal_one_shot_result;
        const double marginal_one_shot_ms = median_ms([&] {
            marginal_one_shot_result = marginal_broker.marginal_probability_from_zero(
                qubits,
                marginal_operations,
                marginal_qubits,
                marginal_bits);
        }, 3);
        std::optional<ExactPreparedProbabilityPlan> marginal_prepared;
        const double marginal_prepared_setup_ms = median_ms([&] {
            marginal_prepared.emplace(qubits, marginal_operations, marginal_config);
        }, 3);
        qubit::ExactProbabilityResult marginal_prepared_result;
        const double marginal_prepared_ms = median_ms([&] {
            marginal_prepared_result = marginal_prepared->marginal_probability(
                marginal_qubits, marginal_bits);
        }, 11, 1000);

        if (broker_result.route != ExactExecutionRoute::PersistentMPS ||
            prepared_result.route != ExactExecutionRoute::PersistentMPS ||
            prepared->prepared_fallback_route() != ExactExecutionRoute::PersistentMPS) {
            std::cerr << "large prepared broker did not retain persistent MPS route\n";
            return 6;
        }
        if (marginal_one_shot_result.route != ExactExecutionRoute::PersistentMPS ||
            marginal_prepared->prepared_route() != ExactExecutionRoute::PersistentMPS ||
            marginal_prepared_result.route != ExactExecutionRoute::PersistentMPS ||
            std::abs(marginal_plan_value - marginal_direct_value) > 2e-9 ||
            std::abs(marginal_one_shot_result.value - marginal_direct_value) > 2e-9 ||
            std::abs(marginal_prepared_result.value - marginal_direct_value) > 2e-9) {
            std::cerr << "large MPS marginal probability route changed exact semantics\n";
            return 7;
        }

        std::cout << "mps_large_qubits=" << qubits << '\n';
        std::cout << "mps_large_setup_ms=" << setup_ms << '\n';
        std::cout << "mps_large_evolved_setup_ms=" << evolved_setup_ms << '\n';
        std::cout << "mps_large_plan_setup_ms=" << plan_setup_ms << '\n';
        std::cout << "mps_large_query_ms=" << query_ms << '\n';
        std::cout << "mps_large_evolved_query_ms=" << evolved_query_ms << '\n';
        std::cout << "mps_large_plan_query_ms=" << plan_query_ms << '\n';
        std::cout << "mps_large_plan_over_direct_query_speed=" << query_ms / plan_query_ms << '\n';
        std::cout << "mps_large_evolved_value_error=" << (value - evolved_value).magnitude() << '\n';
        std::cout << "mps_large_plan_value_error=" << (value - plan_value).magnitude() << '\n';
        std::cout << "mps_large_bytes=" << mps->estimated_bytes() << '\n';
        std::cout << "mps_large_evolved_bytes=" << evolved->estimated_bytes() << '\n';
        std::cout << "mps_large_plan_bytes=" << plan->estimated_bytes() << '\n';
        std::cout << "mps_large_plan_environment_scalars=" << plan->environment_scalar_count() << '\n';
        std::cout << "mps_large_scalars=" << mps->scalar_count() << '\n';
        std::cout << "mps_large_evolved_scalars=" << evolved->scalar_count() << '\n';
        std::cout << "mps_large_max_bond=" << mps->max_bond_dimension() << '\n';
        std::cout << "mps_large_evolved_max_bond=" << evolved->max_bond_dimension() << '\n';
        std::cout << "mps_large_value_real=" << value.re << '\n';
        std::cout << "mps_large_value_imag=" << value.im << '\n';
        std::cout << "mps_large_norm_error=" << std::abs(mps->norm2() - 1.0) << '\n';
        std::cout << "mps_large_evolved_norm_error=" << std::abs(evolved->norm2() - 1.0) << '\n';
        std::cout << "mps_large_broker_one_shot_ms=" << broker_ms << '\n';
        std::cout << "mps_large_prepared_setup_ms=" << prepared_setup_ms << '\n';
        std::cout << "mps_large_prepared_query_ms=" << prepared_query_ms << '\n';
        std::cout << "mps_large_prepared_over_one_shot_speed=" << broker_ms / prepared_query_ms << '\n';
        std::cout << "mps_large_prepared_break_even_queries="
                  << break_even_queries(prepared_setup_ms, broker_ms, prepared_query_ms) << '\n';
        std::cout << "mps_large_prepared_bytes=" << prepared->estimated_bytes() << '\n';
        std::cout << "mps_large_broker_value_error="
                  << (broker_result.value - broker_reference_value).magnitude() << '\n';
        std::cout << "mps_large_prepared_value_error="
                  << (prepared_result.value - broker_reference_value).magnitude() << '\n';
        std::cout << "mps_large_marginal_direct_ms=" << marginal_direct_ms << '\n';
        std::cout << "mps_large_marginal_plan_setup_ms=" << marginal_plan_setup_ms << '\n';
        std::cout << "mps_large_marginal_plan_ms=" << marginal_plan_ms << '\n';
        std::cout << "mps_large_marginal_plan_over_direct_speed="
                  << marginal_direct_ms / marginal_plan_ms << '\n';
        std::cout << "mps_large_marginal_plan_value_error="
                  << std::abs(marginal_plan_value - marginal_direct_value) << '\n';
        std::cout << "mps_large_marginal_one_shot_ms=" << marginal_one_shot_ms << '\n';
        std::cout << "mps_large_marginal_prepared_setup_ms=" << marginal_prepared_setup_ms << '\n';
        std::cout << "mps_large_marginal_prepared_ms=" << marginal_prepared_ms << '\n';
        std::cout << "mps_large_marginal_prepared_over_one_shot_speed="
                  << marginal_one_shot_ms / marginal_prepared_ms << '\n';
        std::cout << "mps_large_marginal_prepared_over_direct_speed="
                  << marginal_direct_ms / marginal_prepared_ms << '\n';
        std::cout << "mps_large_marginal_break_even_queries="
                  << break_even_queries(
                         marginal_prepared_setup_ms,
                         marginal_one_shot_ms,
                         marginal_prepared_ms) << '\n';
        std::cout << "mps_large_marginal_prepared_bytes="
                  << marginal_prepared->estimated_bytes() << '\n';
        std::cout << "mps_large_marginal_value_error="
                  << std::abs(marginal_prepared_result.value - marginal_direct_value) << '\n';
    }

    return 0;
}

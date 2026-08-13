#include "qubit/qbroker.hpp"
#include "qubit/qplan.hpp"
#include "qubit/qtensor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::ExactExecutionBrokerConfig;
using qubit::ExactExecutionRoute;
using qubit::ExactPreparedProbabilityPlan;
using qubit::Operation;
using qubit::OperationCode;
using qubit::QubitId;
using qubit::TensorContractionPlan;
using qubit::TensorNetworkCircuit;
using qubit::TensorNetworkConfig;

[[nodiscard]] std::vector<Operation> chain_operations(std::size_t qubits) {
    std::vector<Operation> operations;
    operations.reserve(2U * qubits + 2U);
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
        0.37,
        0.0,
    });
    operations.push_back({
        OperationCode::Ry,
        static_cast<QubitId>(qubits / 2U),
        0U,
        0.19,
        0.0,
    });
    return operations;
}

[[nodiscard]] std::vector<Operation> deferral_operations(std::size_t qubits) {
    std::vector<Operation> operations;
    operations.reserve(2U * qubits + 2U);
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        operations.push_back({
            OperationCode::Ry,
            static_cast<QubitId>(qubit),
            0U,
            0.001,
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
        0.37,
        0.0,
    });
    operations.push_back({
        OperationCode::Rz,
        static_cast<QubitId>(qubits / 2U),
        0U,
        -0.19,
        0.0,
    });
    return operations;
}

struct Result {
    double circuit_ms{0.0};
    double compile_ms{0.0};
    std::size_t operations{0U};
    std::size_t factors{0U};
    std::size_t steps{0U};
    std::size_t peak_entries{0U};
    std::size_t plan_bytes{0U};
};

[[nodiscard]] Result measure(std::size_t qubits) {
    const std::vector<Operation> operations = chain_operations(qubits);
    TensorNetworkConfig config;
    config.max_contraction_entries = 16U;
    config.max_factors = 1'000'000U;

    std::optional<TensorNetworkCircuit> circuit;
    const auto circuit_start = Clock::now();
    circuit.emplace(qubits, operations, config);
    const auto circuit_stop = Clock::now();

    std::optional<TensorContractionPlan> plan;
    const auto compile_start = Clock::now();
    plan.emplace(circuit->compile());
    const auto compile_stop = Clock::now();

    Result result;
    result.circuit_ms = std::chrono::duration<double, std::milli>(
        circuit_stop - circuit_start).count();
    result.compile_ms = std::chrono::duration<double, std::milli>(
        compile_stop - compile_start).count();
    result.operations = operations.size();
    result.factors = circuit->factor_count();
    result.steps = plan->step_count();
    result.peak_entries = plan->stats().peak_contraction_entries;
    result.plan_bytes = plan->estimated_bytes();
    return result;
}

[[nodiscard]] bool certified(std::size_t qubits, const Result& result) noexcept {
    return result.operations == 2U * qubits + 1U &&
           result.factors == 3U * qubits + 1U &&
           result.steps == 4U * qubits &&
           result.peak_entries == 16U;
}

template <typename Function>
[[nodiscard]] double median_ms(Function&& function, std::size_t repetitions) {
    std::vector<double> samples;
    samples.reserve(repetitions);
    for (std::size_t repetition = 0U; repetition < repetitions; ++repetition) {
        const auto start = Clock::now();
        function();
        const auto stop = Clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

void print(std::size_t qubits, const Result& result) {
    std::cout << "tensor_planner_qubits=" << qubits << '\n';
    std::cout << "tensor_planner_operations=" << result.operations << '\n';
    std::cout << "tensor_planner_factors=" << result.factors << '\n';
    std::cout << "tensor_planner_steps=" << result.steps << '\n';
    std::cout << "tensor_planner_peak_entries=" << result.peak_entries << '\n';
    std::cout << "tensor_planner_circuit_ms=" << result.circuit_ms << '\n';
    std::cout << "tensor_planner_compile_ms=" << result.compile_ms << '\n';
    std::cout << "tensor_planner_plan_bytes=" << result.plan_bytes << '\n';
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);
    const Result small = measure(1'000U);
    const Result medium = measure(5'000U);
    const Result large = measure(30'000U);
    if (!certified(1'000U, small) ||
        !certified(5'000U, medium) ||
        !certified(30'000U, large)) {
        std::cerr << "tensor planner scaling certificate changed\n";
        return 1;
    }
    print(1'000U, small);
    print(5'000U, medium);
    print(30'000U, large);
    std::cout << "tensor_planner_compile_growth_5k_over_1k="
              << medium.compile_ms / small.compile_ms << '\n';
    std::cout << "tensor_planner_compile_growth_30k_over_5k="
              << large.compile_ms / medium.compile_ms << '\n';

    constexpr std::size_t deferral_qubits = 30'000U;
    const std::vector<Operation> operations = deferral_operations(deferral_qubits);
    std::vector<std::uint8_t> zero(deferral_qubits, 0U);

    ExactExecutionBrokerConfig tensor_config;
    tensor_config.tensor.max_contraction_entries = 16U;
    tensor_config.tensor.max_factors = 1'000'000U;
    tensor_config.tensor_planning_defer_variables = 0U;
    ExactExecutionBrokerConfig deferred_config = tensor_config;
    deferred_config.tensor_planning_defer_variables = 65'536U;

    const double tensor_setup_ms = median_ms([&] {
        ExactPreparedProbabilityPlan candidate(deferral_qubits, operations, tensor_config);
        if (candidate.prepared_route() != ExactExecutionRoute::TensorNetwork) {
            throw std::runtime_error("disabled deferral did not retain TensorNetwork");
        }
    }, 5U);
    const double deferred_setup_ms = median_ms([&] {
        ExactPreparedProbabilityPlan candidate(deferral_qubits, operations, deferred_config);
        if (candidate.prepared_route() != ExactExecutionRoute::PersistentMPS) {
            throw std::runtime_error("large certified workload did not defer to PersistentMPS");
        }
    }, 5U);

    ExactPreparedProbabilityPlan tensor_plan(deferral_qubits, operations, tensor_config);
    ExactPreparedProbabilityPlan deferred_plan(deferral_qubits, operations, deferred_config);
    qubit::ExactProbabilityResult tensor_result;
    qubit::ExactProbabilityResult deferred_result;
    const double tensor_query_ms = median_ms([&] {
        tensor_result = tensor_plan.probability(zero);
    }, 7U);
    const double deferred_query_ms = median_ms([&] {
        deferred_result = deferred_plan.probability(zero);
    }, 7U);

    long double analytic = 1.0L;
    const long double one_qubit = std::cos(0.0005L) * std::cos(0.0005L);
    for (std::size_t qubit = 0U; qubit < deferral_qubits; ++qubit) {
        analytic *= one_qubit;
    }
    const double expected = static_cast<double>(analytic);
    const double tensor_error = std::abs(tensor_result.value - expected);
    const double deferred_error = std::abs(deferred_result.value - expected);
    const double route_error = std::abs(tensor_result.value - deferred_result.value);
    if (tensor_result.route != ExactExecutionRoute::TensorNetwork ||
        deferred_result.route != ExactExecutionRoute::PersistentMPS ||
        deferred_result.fallback_reason.find("tensor: planning deferred to certified MPS") ==
            std::string::npos ||
        tensor_error > 2e-9 || deferred_error > 2e-9 || route_error > 2e-9) {
        std::cerr << "tensor planning deferral evidence failed exact controls\n";
        return 1;
    }

    std::cout << "tensor_deferral_qubits=" << deferral_qubits << '\n';
    std::cout << "tensor_deferral_operations=" << operations.size() << '\n';
    std::cout << "tensor_deferral_planner_variables=120000\n";
    std::cout << "tensor_deferral_threshold="
              << deferred_config.tensor_planning_defer_variables << '\n';
    std::cout << "tensor_deferral_baseline_route="
              << qubit::exact_execution_route_name(tensor_plan.prepared_route()) << '\n';
    std::cout << "tensor_deferral_selected_route="
              << qubit::exact_execution_route_name(deferred_plan.prepared_route()) << '\n';
    std::cout << "tensor_deferral_reason_visible=1\n";
    std::cout << "tensor_deferral_tensor_setup_ms=" << tensor_setup_ms << '\n';
    std::cout << "tensor_deferral_mps_setup_ms=" << deferred_setup_ms << '\n';
    std::cout << "tensor_deferral_setup_speedup=" << tensor_setup_ms / deferred_setup_ms << '\n';
    std::cout << "tensor_deferral_tensor_query_ms=" << tensor_query_ms << '\n';
    std::cout << "tensor_deferral_mps_query_ms=" << deferred_query_ms << '\n';
    std::cout << "tensor_deferral_query_speedup=" << tensor_query_ms / deferred_query_ms << '\n';
    std::cout << "tensor_deferral_tensor_plan_bytes=" << tensor_plan.estimated_bytes() << '\n';
    std::cout << "tensor_deferral_mps_plan_bytes=" << deferred_plan.estimated_bytes() << '\n';
    std::cout << "tensor_deferral_expected_probability=" << expected << '\n';
    std::cout << "tensor_deferral_tensor_error=" << tensor_error << '\n';
    std::cout << "tensor_deferral_mps_error=" << deferred_error << '\n';
    std::cout << "tensor_deferral_route_error=" << route_error << '\n';
    return 0;
}

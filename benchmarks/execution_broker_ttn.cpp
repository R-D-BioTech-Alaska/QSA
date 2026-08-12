#include "qubit/qbroker.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::ExactExecutionBrokerConfig;
using qubit::ExactExecutionRoute;
using qubit::ExactPreparedProbabilityPlan;
using qubit::Operation;
using qubit::OperationCode;
using qubit::QRegister;
using qubit::QubitId;
using qubit::TreeTensorMarginalPlan;
using qubit::TreeTensorWorkspace;

#if defined(_MSC_VER)
#define QSA_BROKER_TTN_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define QSA_BROKER_TTN_NOINLINE __attribute__((noinline))
#else
#define QSA_BROKER_TTN_NOINLINE
#endif

volatile double broker_ttn_sink = 0.0;

[[nodiscard]] Operation single(OperationCode code, QubitId qubit, double parameter = 0.0) {
    Operation operation;
    operation.code = code;
    operation.first = qubit;
    operation.parameter = parameter;
    return operation;
}

[[nodiscard]] Operation two(OperationCode code, QubitId first, QubitId second) {
    Operation operation;
    operation.code = code;
    operation.first = first;
    operation.second = second;
    return operation;
}

void append_block(std::vector<Operation>& operations, std::size_t base, double theta) {
    operations.push_back(single(OperationCode::H, static_cast<QubitId>(base)));
    operations.push_back(single(OperationCode::Ry, static_cast<QubitId>(base + 1U), theta));
    operations.push_back(two(
        OperationCode::Cnot,
        static_cast<QubitId>(base),
        static_cast<QubitId>(base + 2U)));
    operations.push_back(two(
        OperationCode::Cnot,
        static_cast<QubitId>(base + 1U),
        static_cast<QubitId>(base + 3U)));
    operations.push_back(two(
        OperationCode::Cz,
        static_cast<QubitId>(base),
        static_cast<QubitId>(base + 3U)));
}

[[nodiscard]] std::vector<Operation> branching_operations(std::size_t qubits) {
    if (qubits == 0U || qubits % 4U != 0U) {
        throw std::runtime_error("branching carrier requires a nonzero multiple of four qubits");
    }
    std::vector<Operation> operations;
    operations.reserve((qubits / 4U) * 5U);
    for (std::size_t base = 0U; base < qubits; base += 4U) {
        const double theta = 0.31 + 0.00001 * static_cast<double>(base / 4U);
        append_block(operations, base, theta);
    }
    return operations;
}

template <class Function>
[[nodiscard]] double median_ms(
    Function&& function,
    std::size_t repeats,
    std::size_t iterations = 1U) {
    std::vector<double> samples;
    samples.reserve(repeats);
    for (std::size_t repeat = 0U; repeat < repeats; ++repeat) {
        const auto start = Clock::now();
        for (std::size_t iteration = 0U; iteration < iterations; ++iteration) {
            function();
        }
        const auto stop = Clock::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count() /
            static_cast<double>(iterations));
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

[[nodiscard]] QRegister evolve_register(
    std::size_t qubits,
    std::span<const Operation> operations) {
    QRegister state(qubits);
    qubit::OperationPlan plan(operations);
    plan.execute(state);
    return state;
}

QSA_BROKER_TTN_NOINLINE double run_broker_marginal(
    const ExactPreparedProbabilityPlan& plan,
    std::span<const QubitId> selected,
    std::span<const std::uint8_t> bits) {
    const auto result = plan.marginal_probability(selected, bits);
    broker_ttn_sink = result.value;
    return result.value;
}

QSA_BROKER_TTN_NOINLINE double run_direct_marginal(
    const TreeTensorMarginalPlan& plan,
    std::span<const QubitId> selected,
    std::span<const std::uint8_t> bits,
    TreeTensorWorkspace& workspace) {
    const double result = plan.marginal_probability(selected, bits, workspace);
    broker_ttn_sink = result;
    return result;
}

void matched_evidence() {
    constexpr std::size_t qubits = 16U;
    const std::vector<Operation> operations = branching_operations(qubits);
    ExactExecutionBrokerConfig config;
    config.tree_tensor.max_bond_dimension = 8U;
    config.tree_tensor.max_scalars = 32'768U;

    const double broker_setup_ms = median_ms([&] {
        const auto plan = ExactPreparedProbabilityPlan::for_marginals(
            qubits, operations, config);
        if (plan.prepared_route() != ExactExecutionRoute::TreeTensor) {
            throw std::runtime_error("matched broker carrier did not select TreeTensor");
        }
        broker_ttn_sink = static_cast<double>(plan.estimated_bytes());
    }, 7U);
    const double direct_setup_ms = median_ms([&] {
        const TreeTensorMarginalPlan plan(qubits, operations, config.tree_tensor);
        broker_ttn_sink = static_cast<double>(plan.estimated_bytes());
    }, 7U);
    const double register_setup_ms = median_ms([&] {
        const QRegister state = evolve_register(qubits, operations);
        broker_ttn_sink = static_cast<double>(state.estimated_bytes());
    }, 7U);

    const ExactPreparedProbabilityPlan broker =
        ExactPreparedProbabilityPlan::for_marginals(qubits, operations, config);
    const TreeTensorMarginalPlan direct(qubits, operations, config.tree_tensor);
    TreeTensorWorkspace direct_workspace = direct.workspace();
    const QRegister dense = evolve_register(qubits, operations);

    const std::array<QubitId, 4> selected{{0U, 1U, 2U, 3U}};
    const std::array<std::uint8_t, 4> bits{{1U, 1U, 1U, 1U}};
    double broker_value = 0.0;
    double direct_value = 0.0;
    const double broker_query_ms = median_ms([&] {
        broker_value = run_broker_marginal(broker, selected, bits);
    }, 11U, 100U);
    const double direct_query_ms = median_ms([&] {
        direct_value = run_direct_marginal(direct, selected, bits, direct_workspace);
    }, 11U, 100U);
    const double register_value = dense.marginal_probability(selected, bits);
    const double broker_error = std::abs(broker_value - register_value);
    const double direct_error = std::abs(direct_value - register_value);
    if (broker.prepared_route() != ExactExecutionRoute::TreeTensor ||
        broker_error > 5e-10 || direct_error > 5e-10) {
        throw std::runtime_error("matched marginal TTN broker evidence failed");
    }
    const auto routed = broker.marginal_probability(selected, bits);
    if (routed.fallback_reason.find("mps:") == std::string::npos ||
        routed.fallback_reason.find("ttn:") != std::string::npos) {
        throw std::runtime_error("matched broker fallback receipt is invalid");
    }

    std::cout << "broker_ttn_matched_qubits=" << qubits << '\n';
    std::cout << "broker_ttn_matched_operations=" << operations.size() << '\n';
    std::cout << "broker_ttn_matched_route=" << qubit::exact_execution_route_name(broker.prepared_route()) << '\n';
    std::cout << "broker_ttn_matched_setup_ms=" << broker_setup_ms << '\n';
    std::cout << "broker_ttn_matched_direct_setup_ms=" << direct_setup_ms << '\n';
    std::cout << "broker_ttn_matched_register_setup_ms=" << register_setup_ms << '\n';
    std::cout << "broker_ttn_matched_query_ms=" << broker_query_ms << '\n';
    std::cout << "broker_ttn_matched_direct_query_ms=" << direct_query_ms << '\n';
    std::cout << "broker_ttn_matched_plan_bytes=" << broker.estimated_bytes() << '\n';
    std::cout << "broker_ttn_matched_direct_plan_bytes=" << direct.estimated_bytes() << '\n';
    std::cout << "broker_ttn_matched_broker_error=" << broker_error << '\n';
    std::cout << "broker_ttn_matched_direct_error=" << direct_error << '\n';
}

void capability_evidence() {
    constexpr std::size_t qubits = 4096U;
    const std::vector<Operation> operations = branching_operations(qubits);
    ExactExecutionBrokerConfig config;
    config.tree_tensor.max_bond_dimension = 8U;
    config.tree_tensor.max_scalars = 1U << 20U;

    const double broker_setup_ms = median_ms([&] {
        const auto plan = ExactPreparedProbabilityPlan::for_marginals(
            qubits, operations, config);
        if (plan.prepared_route() != ExactExecutionRoute::TreeTensor) {
            throw std::runtime_error("large broker carrier did not select TreeTensor");
        }
        broker_ttn_sink = static_cast<double>(plan.estimated_bytes());
    }, 3U);
    const double direct_setup_ms = median_ms([&] {
        const TreeTensorMarginalPlan plan(qubits, operations, config.tree_tensor);
        broker_ttn_sink = static_cast<double>(plan.estimated_bytes());
    }, 3U);

    const ExactPreparedProbabilityPlan broker =
        ExactPreparedProbabilityPlan::for_marginals(qubits, operations, config);
    const TreeTensorMarginalPlan direct(qubits, operations, config.tree_tensor);
    TreeTensorWorkspace direct_workspace = direct.workspace();
    const std::array<QubitId, 4> selected{{0U, 1U, 2U, 3U}};
    const std::array<std::uint8_t, 4> bits{{1U, 1U, 1U, 1U}};

    std::vector<Operation> block_operations;
    block_operations.reserve(5U);
    append_block(block_operations, 0U, 0.31);
    const QRegister block = evolve_register(4U, block_operations);
    const double expected = block.marginal_probability(selected, bits);

    double broker_value = 0.0;
    double direct_value = 0.0;
    const double broker_query_ms = median_ms([&] {
        broker_value = run_broker_marginal(broker, selected, bits);
    }, 7U, 10U);
    const double direct_query_ms = median_ms([&] {
        direct_value = run_direct_marginal(direct, selected, bits, direct_workspace);
    }, 7U, 10U);
    const double broker_error = std::abs(broker_value - expected);
    const double direct_error = std::abs(direct_value - expected);
    if (broker.prepared_route() != ExactExecutionRoute::TreeTensor ||
        broker_error > 5e-10 || direct_error > 5e-10) {
        throw std::runtime_error("large marginal TTN broker evidence failed");
    }
    const auto routed = broker.marginal_probability(selected, bits);
    if (routed.fallback_reason.find("mps:") == std::string::npos ||
        routed.fallback_reason.find("ttn:") != std::string::npos) {
        throw std::runtime_error("large broker fallback receipt is invalid");
    }

    std::cout << "broker_ttn_capability_qubits=" << qubits << '\n';
    std::cout << "broker_ttn_capability_operations=" << operations.size() << '\n';
    std::cout << "broker_ttn_capability_route=" << qubit::exact_execution_route_name(broker.prepared_route()) << '\n';
    std::cout << "broker_ttn_capability_setup_ms=" << broker_setup_ms << '\n';
    std::cout << "broker_ttn_capability_direct_setup_ms=" << direct_setup_ms << '\n';
    std::cout << "broker_ttn_capability_query_ms=" << broker_query_ms << '\n';
    std::cout << "broker_ttn_capability_direct_query_ms=" << direct_query_ms << '\n';
    std::cout << "broker_ttn_capability_plan_bytes=" << broker.estimated_bytes() << '\n';
    std::cout << "broker_ttn_capability_direct_plan_bytes=" << direct.estimated_bytes() << '\n';
    std::cout << "broker_ttn_capability_selected_probability=" << broker_value << '\n';
    std::cout << "broker_ttn_capability_control_probability=" << expected << '\n';
    std::cout << "broker_ttn_capability_broker_error=" << broker_error << '\n';
    std::cout << "broker_ttn_capability_direct_error=" << direct_error << '\n';
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);
    matched_evidence();
    capability_evidence();
    return 0;
}

#include "qubit/qttn_marginal.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using qubit::Operation;
using qubit::OperationCode;
using qubit::QStateError;
using qubit::QubitId;
using qubit::TreeTensorConfig;
using qubit::TreeTensorMarginalCertificate;
using qubit::TreeTensorMarginalPlan;
using qubit::TreeTensorState;
using qubit::TreeTensorStats;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(double actual, double expected, const std::string& message) {
    const double scale = 1.0 + std::max(std::abs(actual), std::abs(expected));
    require(std::abs(actual - expected) <= 5e-10 * scale, message);
}

Operation single(OperationCode code, QubitId qubit, double parameter = 0.0) {
    Operation operation;
    operation.code = code;
    operation.first = qubit;
    operation.parameter = parameter;
    return operation;
}

Operation two(OperationCode code, QubitId first, QubitId second) {
    Operation operation;
    operation.code = code;
    operation.first = first;
    operation.second = second;
    return operation;
}

void require_stats(
    const TreeTensorStats& predicted,
    const TreeTensorStats& actual,
    const std::string& label) {
    require(predicted.qubit_count == actual.qubit_count, label + ": qubit count changed");
    require(predicted.node_count == actual.node_count, label + ": node count changed");
    require(predicted.scalar_count == actual.scalar_count, label + ": scalar count changed");
    require(predicted.max_bond_dimension == actual.max_bond_dimension,
            label + ": max bond changed");
    require(predicted.controlled_gate_count == actual.controlled_gate_count,
            label + ": controlled count changed");
    require(predicted.max_controlled_path_edges == actual.max_controlled_path_edges,
            label + ": max path changed");
    require(predicted.generation == actual.generation, label + ": generation changed");
}

bool direct_accepts(
    std::size_t qubits,
    std::span<const Operation> operations,
    TreeTensorConfig config) {
    try {
        (void)TreeTensorState::from_operations(qubits, operations, config);
        return true;
    } catch (const QStateError&) {
        return false;
    }
}

void require_rejection_agreement(
    std::size_t qubits,
    std::span<const Operation> operations,
    TreeTensorConfig config,
    const std::string& label) {
    const TreeTensorMarginalCertificate certificate =
        TreeTensorMarginalPlan::certify(qubits, operations, config);
    require(!certificate.eligible, label + ": certificate unexpectedly accepted");
    require(!direct_accepts(qubits, operations, config),
            label + ": direct TTN unexpectedly accepted");
}

void fixed_prediction() {
    TreeTensorConfig config;
    config.max_bond_dimension = 64U;
    config.max_scalars = 2'000'000U;

    const std::vector<Operation> operations{
        single(OperationCode::H, 0U),
        single(OperationCode::Ry, 1U, 0.42),
        single(OperationCode::Rz, 5U, -0.27),
        two(OperationCode::Cnot, 0U, 4U),
        two(OperationCode::Cz, 1U, 5U),
        two(OperationCode::Cnot, 4U, 2U),
        two(OperationCode::Cz, 0U, 3U),
        two(OperationCode::Cnot, 5U, 1U),
    };

    const TreeTensorMarginalCertificate certificate =
        TreeTensorMarginalPlan::certify(6U, operations, config);
    require(certificate.eligible, "fixed marginal certificate rejected");

    const TreeTensorState direct =
        TreeTensorState::from_operations(6U, operations, config);
    require_stats(certificate.predicted, direct.stats(), "fixed certificate");

    TreeTensorMarginalPlan plan(6U, operations, config);
    const std::array<QubitId, 3> selected{{0U, 2U, 5U}};
    for (std::uint8_t mask = 0U; mask < 8U; ++mask) {
        const std::array<std::uint8_t, 3> bits{{
            static_cast<std::uint8_t>(mask & 1U),
            static_cast<std::uint8_t>((mask >> 1U) & 1U),
            static_cast<std::uint8_t>((mask >> 2U) & 1U),
        }};
        require_close(
            plan.marginal_probability(selected, bits),
            direct.marginal_probability(selected, bits),
            "fixed marginal plan differs from direct TTN");
    }
}

void randomized_prediction() {
    std::mt19937_64 generator(0x7a11c37ULL);
    for (std::size_t trial = 0U; trial < 32U; ++trial) {
        const std::size_t qubits = 3U + static_cast<std::size_t>(generator() % 6U);
        TreeTensorConfig config;
        config.max_bond_dimension = 64U;
        config.max_scalars = 4'000'000U;

        std::vector<Operation> operations;
        std::size_t controlled = 0U;
        for (std::size_t step = 0U; step < 12U; ++step) {
            const bool use_two = controlled < 4U && generator() % 3U == 0U;
            if (!use_two) {
                const QubitId qubit = static_cast<QubitId>(generator() % qubits);
                switch (generator() % 5U) {
                    case 0U:
                        operations.push_back(single(OperationCode::X, qubit));
                        break;
                    case 1U:
                        operations.push_back(single(OperationCode::H, qubit));
                        break;
                    case 2U:
                        operations.push_back(single(
                            OperationCode::Ry,
                            qubit,
                            0.08 + 0.03 * static_cast<double>((trial + step) % 13U)));
                        break;
                    case 3U:
                        operations.push_back(single(
                            OperationCode::Rz,
                            qubit,
                            -0.24 + 0.04 * static_cast<double>((trial + 2U * step) % 11U)));
                        break;
                    default:
                        operations.push_back(single(OperationCode::T, qubit));
                        break;
                }
                continue;
            }

            QubitId first = static_cast<QubitId>(generator() % qubits);
            QubitId second = static_cast<QubitId>(generator() % qubits);
            while (second == first) {
                second = static_cast<QubitId>(generator() % qubits);
            }
            operations.push_back(two(
                (generator() & 1U) == 0U ? OperationCode::Cnot : OperationCode::Cz,
                first,
                second));
            ++controlled;
        }

        const TreeTensorMarginalCertificate certificate =
            TreeTensorMarginalPlan::certify(qubits, operations, config);
        require(certificate.eligible, "random marginal certificate rejected");
        const TreeTensorState direct =
            TreeTensorState::from_operations(qubits, operations, config);
        require_stats(certificate.predicted, direct.stats(), "random certificate");

        TreeTensorMarginalPlan plan(qubits, operations, config);
        QubitId first = static_cast<QubitId>(generator() % qubits);
        QubitId second = static_cast<QubitId>(generator() % qubits);
        while (second == first) {
            second = static_cast<QubitId>(generator() % qubits);
        }
        const std::array<QubitId, 2> selected{{first, second}};
        const std::array<std::uint8_t, 2> bits{{
            static_cast<std::uint8_t>(generator() & 1U),
            static_cast<std::uint8_t>(generator() & 1U),
        }};
        require_close(
            plan.marginal_probability(selected, bits),
            direct.marginal_probability(selected, bits),
            "random marginal plan differs from direct TTN");
    }
}

void rejection_agreement() {
    TreeTensorConfig config;
    config.max_bond_dimension = 64U;
    config.max_scalars = 1'000'000U;

    {
        const std::array<Operation, 1> operations{{two(OperationCode::Swap, 0U, 1U)}};
        require_rejection_agreement(4U, operations, config, "swap rejection");
    }
    {
        const std::array<Operation, 1> operations{{
            single(OperationCode::BitFlipTrajectory, 0U, 0.2),
        }};
        require_rejection_agreement(4U, operations, config, "trajectory rejection");
    }
    {
        const std::array<Operation, 1> operations{{single(OperationCode::H, 4U)}};
        require_rejection_agreement(4U, operations, config, "qubit-range rejection");
    }
    {
        const std::array<Operation, 1> operations{{single(
            OperationCode::Ry,
            0U,
            std::numeric_limits<double>::quiet_NaN())}};
        require_rejection_agreement(4U, operations, config, "nonfinite rotation rejection");
    }
    {
        TreeTensorConfig invalid = config;
        invalid.max_bond_dimension = 0U;
        require_rejection_agreement(4U, {}, invalid, "zero bond-cap rejection");
    }
    {
        TreeTensorConfig invalid = config;
        invalid.max_scalars = 0U;
        require_rejection_agreement(4U, {}, invalid, "zero scalar-cap rejection");
    }
    {
        TreeTensorConfig invalid = config;
        invalid.validation_tolerance = 0.0;
        require_rejection_agreement(4U, {}, invalid, "invalid tolerance rejection");
    }
}

void resource_boundaries() {
    const std::array<Operation, 1> cross_root{{two(OperationCode::Cnot, 0U, 3U)}};

    {
        TreeTensorConfig config;
        config.max_bond_dimension = 1U;
        config.max_scalars = 1'000U;
        require_rejection_agreement(4U, cross_root, config, "bond-cap rejection");
    }
    {
        TreeTensorConfig config;
        config.max_bond_dimension = 8U;
        config.max_scalars = 23U;
        require_rejection_agreement(4U, cross_root, config, "state-scalar rejection");
    }

    const std::array<Operation, 3> repeated{{
        two(OperationCode::Cnot, 0U, 3U),
        two(OperationCode::Cnot, 0U, 3U),
        two(OperationCode::Cnot, 0U, 3U),
    }};

    TreeTensorConfig roomy;
    roomy.max_bond_dimension = 8U;
    roomy.max_scalars = 1'000U;
    const TreeTensorMarginalCertificate roomy_certificate =
        TreeTensorMarginalPlan::certify(4U, repeated, roomy);
    require(roomy_certificate.eligible, "roomy repeated-path certificate rejected");
    require(roomy_certificate.predicted.scalar_count == 228U,
            "repeated-path state scalar prediction changed");
    require(roomy_certificate.marginal_workspace_scalars == 259U,
            "repeated-path marginal workspace prediction changed");

    TreeTensorConfig exact = roomy;
    exact.max_scalars = roomy_certificate.marginal_workspace_scalars;
    const TreeTensorMarginalCertificate exact_certificate =
        TreeTensorMarginalPlan::certify(4U, repeated, exact);
    require(exact_certificate.eligible, "exact workspace boundary rejected");
    TreeTensorMarginalPlan exact_plan(4U, repeated, exact);
    const std::array<QubitId, 1> selected{{0U}};
    const std::array<std::uint8_t, 1> bit{{0U}};
    require(std::isfinite(exact_plan.marginal_probability(selected, bit)),
            "exact workspace boundary query failed");

    TreeTensorConfig tight = roomy;
    tight.max_scalars = roomy_certificate.marginal_workspace_scalars - 1U;
    const TreeTensorMarginalCertificate tight_certificate =
        TreeTensorMarginalPlan::certify(4U, repeated, tight);
    require(!tight_certificate.eligible,
            "tight workspace certificate unexpectedly accepted");

    TreeTensorState direct = TreeTensorState::from_operations(4U, repeated, tight);
    require(direct.stats().scalar_count == 228U,
            "tight workspace direct state no longer fits");
    bool query_rejected = false;
    try {
        (void)direct.marginal_probability(selected, bit);
    } catch (const QStateError&) {
        query_rejected = true;
    }
    require(query_rejected,
            "tight workspace direct marginal unexpectedly allocated");
}

}  // namespace

int main() {
    fixed_prediction();
    randomized_prediction();
    rejection_agreement();
    resource_boundaries();
    return 0;
}

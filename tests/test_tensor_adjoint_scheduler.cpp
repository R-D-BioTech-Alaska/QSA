#include "qubit/qtensor_adjoint_scheduler.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using qubit::ExactAdjointGradientConfig;
using qubit::ExactAdjointGradientPlan;
using qubit::ExactAdjointGradientSchedulerConfig;
using qubit::ExactAdjointGradientSchedulerPlan;
using qubit::ExactAdjointScheduleRoute;
using qubit::OperationCode;
using qubit::ParameterizedOperation;
using qubit::PauliAxis;
using qubit::PauliFactor;
using qubit::PauliObservable;
using qubit::PauliPropagationConfig;
using qubit::QComplex;
using qubit::QStateError;
using qubit::TensorNetworkConfig;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<ParameterizedOperation> operations() {
    return {
        {{OperationCode::Ry, 0U, 0U, 0.0, 0.0}, 0, -1},
        {{OperationCode::Rz, 1U, 0U, 0.0, 0.0}, 1, -1},
        {{OperationCode::Cnot, 0U, 1U, 0.0, 0.0}, -1, -1},
        {{OperationCode::Rx, 2U, 0U, 0.0, 0.0}, 0, -1},
        {{OperationCode::Ry, 3U, 0U, 0.0, 0.0}, 2, -1},
        {{OperationCode::Cnot, 2U, 3U, 0.0, 0.0}, -1, -1},
        {{OperationCode::Rz, 4U, 0U, 0.0, 0.0}, 1, -1},
        {{OperationCode::Cnot, 4U, 5U, 0.0, 0.0}, -1, -1},
        {{OperationCode::Ry, 5U, 0U, 0.0, 0.0}, 2, -1},
    };
}

std::vector<PauliObservable> observables() {
    constexpr std::size_t qubits = 6U;
    std::vector<PauliObservable> result;
    PauliObservable first(qubits, PauliPropagationConfig{65'536U});
    first.add_term({0.7, -0.1}, std::vector<PauliFactor>{{0U, PauliAxis::X}, {1U, PauliAxis::Y}});
    result.push_back(std::move(first));
    PauliObservable second(qubits, PauliPropagationConfig{65'536U});
    second.add_term({-0.4, 0.0}, std::vector<PauliFactor>{{2U, PauliAxis::Z}});
    second.add_term({0.125, 0.0});
    result.push_back(std::move(second));
    PauliObservable third(qubits, PauliPropagationConfig{65'536U});
    third.add_term({0.55, 0.05}, std::vector<PauliFactor>{{3U, PauliAxis::X}, {4U, PauliAxis::Z}, {5U, PauliAxis::Y}});
    result.push_back(std::move(third));
    return result;
}

std::vector<double> points(std::size_t count) {
    std::vector<double> result(count * 3U);
    for (std::size_t point = 0U; point < count; ++point) {
        result[point * 3U + 0U] = -0.61 + 0.071 * static_cast<double>(point + 1U);
        result[point * 3U + 1U] = 0.37 - 0.043 * static_cast<double>(point + 2U);
        result[point * 3U + 2U] = -0.21 + 0.039 * static_cast<double>(point + 3U);
    }
    return result;
}

}  // namespace

int main() {
    constexpr std::size_t qubits = 6U;
    constexpr std::size_t point_count = 8U;
    const TensorNetworkConfig tensor{1U << 18U, 100'000U};
    const auto templated = operations();
    const auto queries = observables();
    const auto parameters = points(point_count);

    ExactAdjointGradientPlan reference(
        qubits, templated, queries, ExactAdjointGradientConfig{tensor, 1U});
    auto reference_workspace = reference.workspace();
    const std::size_t value_count = point_count * reference.observable_count();
    const std::size_t gradient_per_point =
        reference.observable_count() * reference.parameter_count();
    std::vector<QComplex> reference_values(value_count);
    std::vector<QComplex> reference_gradients(point_count * gradient_per_point);
    for (std::size_t point = 0U; point < point_count; ++point) {
        reference.value_and_gradient(
            std::span<const double>(parameters.data() + point * reference.parameter_count(), reference.parameter_count()),
            std::span<QComplex>(reference_values.data() + point * reference.observable_count(), reference.observable_count()),
            std::span<QComplex>(reference_gradients.data() + point * gradient_per_point, gradient_per_point),
            reference_workspace);
    }

    ExactAdjointGradientSchedulerPlan scheduler(
        qubits,
        templated,
        queries,
        ExactAdjointGradientSchedulerConfig{tensor, 4U, 0U});
    const auto single = scheduler.choose(1U);
    require(single.route == ExactAdjointScheduleRoute::TermParallel,
            "scheduler did not select term parallelism for one point");
    require(single.term_eligible, "single-point term route is not eligible");
    require(single.estimated_critical_work == single.term_estimated_critical_work,
            "single-point scheduler critical work changed");

    const auto many = scheduler.choose(point_count);
    require(many.point_eligible, "many-point point route is not eligible");
    require(many.route == ExactAdjointScheduleRoute::PointParallel,
            "scheduler did not select point parallelism for many points");
    require(many.point_estimated_critical_work < many.term_estimated_critical_work,
            "many-point structural model does not favor point parallelism");

    auto workspace = scheduler.workspace();
    std::vector<QComplex> values(value_count);
    std::vector<QComplex> gradients(reference_gradients.size());
    qubit::ExactAdjointGradientSchedule completed;
    scheduler.value_and_gradient_batch(
        parameters, point_count, values, gradients, workspace, &completed);
    require(values == reference_values,
            "scheduler values differ from repeated serial adjoint");
    require(gradients == reference_gradients,
            "scheduler gradients differ from repeated serial adjoint");
    require(completed.route == many.route,
            "scheduler executed a different route than choose reported");
    require(completed.estimated_workspace_bytes == workspace.estimated_bytes(),
            "scheduler predicted workspace does not match retained workspace");
    require(workspace.successful_point_count() == point_count,
            "scheduler successful point count changed");
    require(workspace.has_last_schedule(),
            "scheduler workspace did not retain completed route metadata");

    const auto first_values = values;
    const auto first_gradients = gradients;
    scheduler.value_and_gradient_batch(
        parameters, point_count, values, gradients, workspace, nullptr);
    require(values == first_values, "scheduler values are not deterministic");
    require(gradients == first_gradients, "scheduler gradients are not deterministic");
    require(workspace.estimated_bytes() == many.estimated_workspace_bytes,
            "reused scheduler workspace outgrew its retained-memory certificate");
    require(workspace.successful_point_count() == 2U * point_count,
            "scheduler cumulative successful point count changed");

    {
        ExactAdjointGradientSchedulerPlan one_worker(
            qubits,
            templated,
            queries,
            ExactAdjointGradientSchedulerConfig{tensor, 1U, 0U});
        const auto route = one_worker.choose(point_count);
        require(route.route == ExactAdjointScheduleRoute::Serial,
                "one-worker scheduler selected a parallel route");
    }

    {
        const std::size_t budget = many.term_estimated_workspace_bytes;
        require(many.point_estimated_workspace_bytes > budget,
                "scheduler memory test lacks a point-workspace separation");
        ExactAdjointGradientSchedulerPlan constrained(
            qubits,
            templated,
            queries,
            ExactAdjointGradientSchedulerConfig{tensor, 4U, budget});
        const auto route = constrained.choose(point_count);
        require(route.route != ExactAdjointScheduleRoute::PointParallel,
                "workspace budget did not exclude point parallelism");
        require(route.estimated_workspace_bytes <= budget,
                "scheduler selected a route over the workspace budget");
        auto constrained_workspace = constrained.workspace();
        std::vector<QComplex> constrained_values(value_count);
        std::vector<QComplex> constrained_gradients(reference_gradients.size());
        qubit::ExactAdjointGradientSchedule constrained_completed;
        constrained.value_and_gradient_batch(
            parameters,
            point_count,
            constrained_values,
            constrained_gradients,
            constrained_workspace,
            &constrained_completed);
        require(constrained_values == reference_values,
                "memory-constrained scheduler values changed");
        require(constrained_gradients == reference_gradients,
                "memory-constrained scheduler gradients changed");
        require(constrained_workspace.estimated_bytes() ==
                    constrained_completed.estimated_workspace_bytes,
                "memory-constrained scheduler retained size changed");
        require(constrained_workspace.estimated_bytes() <= budget,
                "memory-constrained scheduler exceeded its hard budget");
    }

    {
        const std::size_t impossible_budget = many.serial_estimated_workspace_bytes - 1U;
        ExactAdjointGradientSchedulerPlan constrained(
            qubits,
            templated,
            queries,
            ExactAdjointGradientSchedulerConfig{tensor, 4U, impossible_budget});
        bool rejected = false;
        try {
            static_cast<void>(constrained.choose(point_count));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "scheduler accepted a workspace budget below every route");
    }

    {
        const auto preserved_values = values;
        const auto preserved_gradients = gradients;
        std::vector<double> invalid = parameters;
        invalid[4U] = std::numeric_limits<double>::quiet_NaN();
        bool rejected = false;
        try {
            scheduler.value_and_gradient_batch(
                invalid, point_count, values, gradients, workspace, nullptr);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "scheduler accepted a nonfinite parameter");
        require(values == preserved_values,
                "rejected scheduler call mutated value outputs");
        require(gradients == preserved_gradients,
                "rejected scheduler call mutated gradient outputs");
    }

    {
        ExactAdjointGradientSchedulerPlan foreign(
            qubits,
            templated,
            queries,
            ExactAdjointGradientSchedulerConfig{tensor, 4U, 0U});
        auto foreign_workspace = foreign.workspace();
        bool rejected = false;
        try {
            scheduler.value_and_gradient_batch(
                parameters, point_count, values, gradients, foreign_workspace, nullptr);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "scheduler accepted a foreign workspace");
    }

    {
        bool rejected = false;
        try {
            static_cast<void>(ExactAdjointGradientSchedulerPlan(
                qubits,
                templated,
                queries,
                ExactAdjointGradientSchedulerConfig{tensor, 17U, 0U}));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "scheduler accepted an unbounded worker count");
    }

    return 0;
}

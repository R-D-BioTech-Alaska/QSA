#include "qubit/qexecution.hpp"

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::DependencyOperationPlan;
using qubit::IndependentComponentPlan;
using qubit::Operation;
using qubit::OperationCode;
using qubit::OperationPlan;
using qubit::QRegister;
using qubit::QubitId;

[[nodiscard]] QRegister make_components(
    std::size_t component_count,
    std::size_t width) {
    QRegister state(component_count * width);
    for (std::size_t component = 0; component < component_count; ++component) {
        const std::size_t offset = component * width;
        for (std::size_t local = 0; local < width; ++local) {
            const QubitId qubit = static_cast<QubitId>(offset + local);
            state.apply_ry(qubit, 0.17 + 0.013 * static_cast<double>(local));
            state.apply_rz(qubit, -0.11 + 0.009 * static_cast<double>(component));
        }
        for (std::size_t local = 1; local < width; ++local) {
            state.apply_cz(
                static_cast<QubitId>(offset + local - 1U),
                static_cast<QubitId>(offset + local));
        }
    }
    return state;
}

template <class Function>
[[nodiscard]] double timed_ms(Function&& function) {
    const auto start = Clock::now();
    function();
    const auto stop = Clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);
    std::vector<Operation> cancellable;
    cancellable.reserve(200'000U);
    for (std::size_t index = 0; index < 50'000U; ++index) {
        cancellable.push_back(Operation{OperationCode::H, 0});
        cancellable.push_back(Operation{OperationCode::X, 1});
        cancellable.push_back(Operation{OperationCode::H, 0});
        cancellable.push_back(Operation{OperationCode::X, 1});
    }
    OperationPlan ordinary(cancellable, true);
    DependencyOperationPlan dependency(cancellable);
    QRegister ordinary_state(2);
    const double ordinary_ms = timed_ms([&] { ordinary.execute(ordinary_state); });
    QRegister dependency_state(2);
    const double dependency_ms = timed_ms([&] { dependency.execute(dependency_state); });
    std::cout << "dependency source=" << cancellable.size()
              << " ordinary_steps=" << ordinary.compiled_step_count()
              << " dependency_operations=" << dependency.optimized_operation_count()
              << " ordinary_ms=" << ordinary_ms
              << " dependency_ms=" << dependency_ms
              << " speedup=" << ordinary_ms / dependency_ms << '\n';

    constexpr std::size_t components = 16U;
    constexpr std::size_t width = 14U;
    const QRegister seed = make_components(components, width);
    std::vector<Operation> operations;
    for (std::size_t component = 0; component < components; ++component) {
        const std::size_t offset = component * width;
        for (std::size_t round = 0; round < 12U; ++round) {
            operations.push_back(Operation{
                OperationCode::Ry,
                static_cast<QubitId>(offset + round % width),
                0,
                0.009 * static_cast<double>(round + 1U),
            });
        }
    }

    QRegister sequential = seed;
    const double sequential_ms = timed_ms([&] {
        for (const Operation& operation : operations) {
            sequential.apply_ry(operation.first, operation.parameter);
        }
    });
    QRegister parallel = seed;
    IndependentComponentPlan component_plan(operations);
    const double parallel_ms = timed_ms([&] { component_plan.execute(parallel, 4U); });
    std::cout << "components=" << components
              << " width=" << width
              << " operations=" << operations.size()
              << " sequential_ms=" << sequential_ms
              << " parallel_ms=" << parallel_ms
              << " speedup=" << sequential_ms / parallel_ms << '\n';
    return 0;
}

#include "qubit/qplan.hpp"
#include "qubit/qstate.hpp"

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

template <typename Function>
double elapsed_ms(Function&& function) {
    const auto start = Clock::now();
    function();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::vector<qubit::Operation> make_fusable_plan(std::size_t repetitions) {
    std::vector<qubit::Operation> operations;
    operations.reserve(repetitions * 4U + 2U);
    for (std::size_t index = 0; index < repetitions; ++index) {
        operations.push_back({qubit::OperationCode::Rx, 0, 0, 0.00031});
        operations.push_back({qubit::OperationCode::Ry, 0, 0, -0.00017});
        operations.push_back({qubit::OperationCode::X, 0});
        operations.push_back({qubit::OperationCode::H, 0});
    }
    operations.push_back({qubit::OperationCode::Cnot, 0, 1});
    operations.push_back({qubit::OperationCode::Ry, 1, 0, 0.37});
    return operations;
}

std::vector<qubit::Operation> make_ensemble_plan(std::size_t width, std::size_t layers) {
    std::vector<qubit::Operation> operations;
    for (std::size_t layer = 0; layer < layers; ++layer) {
        for (std::size_t qubit = 0; qubit < width; ++qubit) {
            operations.push_back({qubit::OperationCode::Ry,
                                  static_cast<qubit::QubitId>(qubit), 0,
                                  0.002 + static_cast<double>(qubit) * 0.00001});
            operations.push_back({qubit::OperationCode::Rz,
                                  static_cast<qubit::QubitId>(qubit), 0,
                                  -0.001 + static_cast<double>(layer) * 0.00001});
        }
        for (std::size_t qubit = 1; qubit < width; ++qubit) {
            operations.push_back({qubit::OperationCode::Cnot,
                                  static_cast<qubit::QubitId>(qubit - 1U),
                                  static_cast<qubit::QubitId>(qubit)});
        }
    }
    return operations;
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);

    const auto fusable = make_fusable_plan(5000);
    qubit::OperationPlan literal(fusable, false);
    qubit::OperationPlan optimized(fusable, true);
    qubit::QRegister literal_state(2);
    qubit::QRegister optimized_state(2);
    const double literal_ms = elapsed_ms([&] { literal.execute(literal_state); });
    const double optimized_ms = elapsed_ms([&] { optimized.execute(optimized_state); });
    std::cout << "plan_fusion,source=" << fusable.size()
              << ",steps=" << optimized.compiled_step_count()
              << ",literal_ms=" << literal_ms
              << ",optimized_ms=" << optimized_ms
              << ",speedup=" << literal_ms / optimized_ms << '\n';

    std::vector<qubit::Operation> diagonal_operations;
    for (std::size_t layer = 0; layer < 20; ++layer) {
        for (std::size_t qubit = 0; qubit < 16; ++qubit) {
            diagonal_operations.push_back({qubit::OperationCode::Rz,
                static_cast<qubit::QubitId>(qubit), 0, 0.001 + layer * 0.00001});
            diagonal_operations.push_back({qubit::OperationCode::T,
                static_cast<qubit::QubitId>(qubit)});
        }
    }
    qubit::OperationPlan diagonal_literal(diagonal_operations, false);
    qubit::OperationPlan diagonal_optimized(diagonal_operations, true);
    qubit::QRegister diagonal_literal_state(16);
    qubit::QRegister diagonal_optimized_state(16);
    for (std::size_t qubit = 0; qubit < 16; ++qubit) {
        diagonal_literal_state.apply_h(static_cast<qubit::QubitId>(qubit));
        diagonal_optimized_state.apply_h(static_cast<qubit::QubitId>(qubit));
    }
    for (std::size_t qubit = 1; qubit < 16; ++qubit) {
        diagonal_literal_state.apply_cz(0, static_cast<qubit::QubitId>(qubit));
        diagonal_optimized_state.apply_cz(0, static_cast<qubit::QubitId>(qubit));
    }
    const double diagonal_literal_ms = elapsed_ms([&] {
        diagonal_literal.execute(diagonal_literal_state);
    });
    const double diagonal_optimized_ms = elapsed_ms([&] {
        diagonal_optimized.execute(diagonal_optimized_state);
    });
    std::cout << "diagonal_layer,source=" << diagonal_operations.size()
              << ",steps=" << diagonal_optimized.compiled_step_count()
              << ",literal_ms=" << diagonal_literal_ms
              << ",optimized_ms=" << diagonal_optimized_ms
              << ",speedup=" << diagonal_literal_ms / diagonal_optimized_ms << '\n';

    constexpr std::size_t register_count = 256;
    constexpr std::size_t width = 10;
    const auto ensemble_operations = make_ensemble_plan(width, 4);
    qubit::OperationPlan ensemble_plan(ensemble_operations, true);
    std::vector<std::unique_ptr<qubit::QRegister>> serial_owned;
    std::vector<std::unique_ptr<qubit::QRegister>> parallel_owned;
    std::vector<qubit::QRegister*> serial_states;
    std::vector<qubit::QRegister*> parallel_states;
    for (std::size_t index = 0; index < register_count; ++index) {
        serial_owned.push_back(std::make_unique<qubit::QRegister>(width));
        parallel_owned.push_back(std::make_unique<qubit::QRegister>(width));
        serial_states.push_back(serial_owned.back().get());
        parallel_states.push_back(parallel_owned.back().get());
    }
    const double serial_ms = elapsed_ms([&] { ensemble_plan.execute_many(serial_states, 1); });
    const std::size_t workers = std::max(1U, std::thread::hardware_concurrency());
    const double parallel_ms = elapsed_ms([&] { ensemble_plan.execute_many(parallel_states, workers); });
    std::cout << "ensemble,registers=" << register_count
              << ",operations_per_register=" << ensemble_operations.size()
              << ",workers=" << workers
              << ",serial_ms=" << serial_ms
              << ",parallel_ms=" << parallel_ms
              << ",speedup=" << serial_ms / parallel_ms << '\n';

    qubit::QRegister readout_state(16);
    for (std::size_t qubit = 0; qubit < 16; ++qubit) {
        readout_state.apply_h(static_cast<qubit::QubitId>(qubit));
    }
    for (std::size_t qubit = 1; qubit < 16; ++qubit) {
        readout_state.apply_cz(0, static_cast<qubit::QubitId>(qubit));
    }
    std::vector<double> scalar(16);
    const double scalar_ms = elapsed_ms([&] {
        for (std::size_t repeat = 0; repeat < 100; ++repeat) {
            for (std::size_t qubit = 0; qubit < 16; ++qubit) {
                scalar[qubit] = readout_state.probability_one(static_cast<qubit::QubitId>(qubit));
            }
        }
    });
    std::vector<double> bulk;
    const double bulk_ms = elapsed_ms([&] {
        for (std::size_t repeat = 0; repeat < 100; ++repeat) {
            bulk = readout_state.probabilities_one();
        }
    });
    std::cout << "bulk_probabilities,qubits=16,repeats=100,scalar_ms=" << scalar_ms
              << ",bulk_ms=" << bulk_ms << ",speedup=" << scalar_ms / bulk_ms << '\n';
    return 0;
}

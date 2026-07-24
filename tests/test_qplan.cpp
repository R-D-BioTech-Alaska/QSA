#include "qubit/qplan.hpp"
#include "qubit/qstate.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "PLAN TEST FAILURE: " << message << '\n';
        std::exit(1);
    }
}

void require_equivalent(const qubit::QRegister& left, const qubit::QRegister& right) {
    require(left.qubit_count() == right.qubit_count(), "qubit counts differ");
    const auto left_state = left.materialize();
    const auto right_state = right.materialize();
    require(left_state.size() == right_state.size(), "materialized sizes differ");
    for (std::size_t index = 0; index < left_state.size(); ++index) {
        require(qubit::almost_equal(left_state[index], right_state[index], 2e-11),
                "compiled plan changed an amplitude");
    }
}

void test_fusion_equivalence() {
    const std::vector<qubit::Operation> operations{
        {qubit::OperationCode::H, 0},
        {qubit::OperationCode::Rz, 0, 0, 0.17},
        {qubit::OperationCode::Ry, 0, 0, -0.39},
        {qubit::OperationCode::T, 0},
        {qubit::OperationCode::Cnot, 0, 1},
        {qubit::OperationCode::Rx, 1, 0, 0.21},
        {qubit::OperationCode::Rx, 1, 0, -0.08},
        {qubit::OperationCode::Cz, 1, 2},
    };
    qubit::OperationPlan optimized(operations, true);
    qubit::OperationPlan literal(operations, false);
    require(optimized.source_operation_count() == operations.size(), "source operation count");
    require(optimized.compiled_step_count() < operations.size(), "single-qubit runs must fuse");

    qubit::QRegister optimized_state(3);
    qubit::QRegister literal_state(3);
    optimized.execute(optimized_state);
    literal.execute(literal_state);
    require_equivalent(optimized_state, literal_state);
}

void test_identity_elision() {
    std::vector<qubit::Operation> operations;
    for (std::size_t index = 0; index < 1000; ++index) {
        operations.push_back({qubit::OperationCode::H, 0});
    }
    qubit::OperationPlan plan(operations, true);
    require(plan.compiled_step_count() == 0U, "even Hadamard run should compile to identity");
    qubit::QRegister state(1);
    std::size_t completed = 0;
    plan.execute(state, &completed);
    require(completed == operations.size(), "identity-elided plan completion count");
    require(state.amplitude(0).norm2() > 1.0 - 1e-12, "identity-elided plan changed state");
}

void test_parallel_ensemble() {
    const std::vector<qubit::Operation> operations{
        {qubit::OperationCode::H, 0},
        {qubit::OperationCode::Cnot, 0, 1},
        {qubit::OperationCode::Ry, 2, 0, 0.31},
        {qubit::OperationCode::Rz, 2, 0, -0.22},
        {qubit::OperationCode::Cz, 1, 2},
    };
    qubit::OperationPlan plan(operations, true);
    std::vector<std::unique_ptr<qubit::QRegister>> owned;
    std::vector<qubit::QRegister*> states;
    for (std::size_t index = 0; index < 64; ++index) {
        owned.push_back(std::make_unique<qubit::QRegister>(3));
        states.push_back(owned.back().get());
    }
    std::size_t completed = 0;
    plan.execute_many(states, 4, &completed);
    require(completed == states.size(), "parallel ensemble completion count");
    for (std::size_t index = 1; index < states.size(); ++index) {
        require_equivalent(*states[0], *states[index]);
    }
}

void test_bulk_probabilities() {
    qubit::QRegister state(8);
    state.apply_h(0);
    state.apply_cnot(0, 1);
    state.apply_ry(3, 0.71);
    state.apply_h(6);
    const std::vector<double> probabilities = state.probabilities_one();
    require(probabilities.size() == state.qubit_count(), "bulk probability width");
    for (std::size_t qubit = 0; qubit < probabilities.size(); ++qubit) {
        require(std::abs(probabilities[qubit] - state.probability_one(
                    static_cast<qubit::QubitId>(qubit))) < 1e-12,
                "bulk probability mismatch");
    }
}

void test_diagonal_layer_equivalence() {
    const std::vector<qubit::Operation> operations{
        {qubit::OperationCode::Rz, 0, 0, 0.17},
        {qubit::OperationCode::S, 1},
        {qubit::OperationCode::T, 2},
        {qubit::OperationCode::Rz, 3, 0, -0.29},
        {qubit::OperationCode::Z, 1},
        {qubit::OperationCode::Tdg, 2},
        {qubit::OperationCode::H, 4},
    };
    qubit::OperationPlan optimized(operations, true);
    qubit::OperationPlan literal(operations, false);
    require(optimized.compiled_step_count() == 2U, "diagonal layer should become one pass");

    qubit::QRegister optimized_state(5);
    qubit::QRegister literal_state(5);
    for (qubit::QubitId qubit = 0; qubit < 5; ++qubit) {
        optimized_state.apply_h(qubit);
        literal_state.apply_h(qubit);
    }
    for (qubit::QubitId qubit = 1; qubit < 5; ++qubit) {
        optimized_state.apply_cz(0, qubit);
        literal_state.apply_cz(0, qubit);
    }
    optimized.execute(optimized_state);
    literal.execute(literal_state);
    require_equivalent(optimized_state, literal_state);
}

void test_randomized_plan_equivalence() {
    std::mt19937_64 generator(0x515341013ULL);
    std::uniform_int_distribution<int> gate(0, 8);
    std::uniform_int_distribution<int> qubit(0, 3);
    std::uniform_real_distribution<double> angle(-1.0, 1.0);

    for (std::size_t trial = 0; trial < 250; ++trial) {
        std::vector<qubit::Operation> operations;
        for (std::size_t index = 0; index < 80; ++index) {
            const auto first = static_cast<qubit::QubitId>(qubit(generator));
            switch (gate(generator)) {
                case 0: operations.push_back({qubit::OperationCode::H, first}); break;
                case 1: operations.push_back({qubit::OperationCode::X, first}); break;
                case 2: operations.push_back({qubit::OperationCode::S, first}); break;
                case 3: operations.push_back({qubit::OperationCode::Tdg, first}); break;
                case 4: operations.push_back({qubit::OperationCode::Rx, first, 0, angle(generator)}); break;
                case 5: operations.push_back({qubit::OperationCode::Ry, first, 0, angle(generator)}); break;
                case 6: operations.push_back({qubit::OperationCode::Rz, first, 0, angle(generator)}); break;
                case 7: {
                    const auto second = static_cast<qubit::QubitId>((first + 1U) % 4U);
                    operations.push_back({qubit::OperationCode::Cnot, first, second});
                    break;
                }
                default: {
                    const auto second = static_cast<qubit::QubitId>((first + 2U) % 4U);
                    operations.push_back({qubit::OperationCode::Cz, first, second});
                    break;
                }
            }
        }
        qubit::OperationPlan optimized(operations, true);
        qubit::OperationPlan literal(operations, false);
        qubit::QRegister optimized_state(4);
        qubit::QRegister literal_state(4);
        optimized.execute(optimized_state);
        literal.execute(literal_state);
        require_equivalent(optimized_state, literal_state);
    }
}

void test_parameterized_plan() {
    const std::vector<qubit::ParameterizedOperation> operations{
        {{qubit::OperationCode::Ry, 0}, 0, -1},
        {{qubit::OperationCode::Rz, 0}, 1, -1},
        {{qubit::OperationCode::Cnot, 0, 1}, -1, -1},
        {{qubit::OperationCode::Ry, 1}, 0, -1},
    };
    qubit::ParameterizedOperationPlan plan(operations, true);
    require(plan.parameter_count() == 2U, "parameterized plan slot count");
    const std::array<double, 2> values{0.31, -0.22};
    qubit::QRegister parameterized_state(2);
    plan.execute(parameterized_state, values);

    qubit::QRegister direct_state(2);
    direct_state.apply_ry(0, values[0]);
    direct_state.apply_rz(0, values[1]);
    direct_state.apply_cnot(0, 1);
    direct_state.apply_ry(1, values[0]);
    require_equivalent(parameterized_state, direct_state);

    std::vector<std::unique_ptr<qubit::QRegister>> owned;
    std::vector<qubit::QRegister*> states;
    for (std::size_t index = 0; index < 32; ++index) {
        owned.push_back(std::make_unique<qubit::QRegister>(2));
        states.push_back(owned.back().get());
    }
    std::size_t completed = 0;
    plan.execute_many(states, values, 4, &completed);
    require(completed == states.size(), "parameterized ensemble completion count");
    for (qubit::QRegister* state : states) {
        require_equivalent(*state, direct_state);
    }
}

}  // namespace

int main() {
    test_fusion_equivalence();
    test_identity_elision();
    test_parallel_ensemble();
    test_bulk_probabilities();
    test_diagonal_layer_equivalence();
    test_randomized_plan_equivalence();
    test_parameterized_plan();
    std::cout << "QSA compiled-plan and ensemble tests passed.\n";
    return 0;
}

#include "qubit/c_api.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "C API TEST FAILURE: " << message;
        const char* error = qstate_last_error();
        if (error != nullptr && std::strlen(error) != 0U) {
            std::cerr << " (native error: " << error << ')';
        }
        std::cerr << '\n';
        std::exit(1);
    }
}

void require_near(double actual, double expected, double tolerance, const std::string& message) {
    require(std::abs(actual - expected) <= tolerance, message);
}

}  // namespace

int main() {
    require(qstate_abi_version_major() == 1U, "ABI major must remain 1");
    require(qstate_abi_version_minor() >= 5U, "compiled-plan ABI must be available");
    require(std::string(qstate_version_string()) == "0.2.0", "native version string");

    qstate_handle state = qstate_create(2);
    require(state != nullptr, "qstate_create");
    require(qstate_qubit_count(state) == 2U, "qubit count");
    require(qstate_component_count(state) == 2U, "initial component count");

    const qstate_operation bell_operations[] = {
        {QSTATE_OP_H, 0U, 0U, 0U, 0.0, 0.0},
        {QSTATE_OP_CNOT, 0U, 1U, 0U, 0.0, 0.0},
    };
    size_t completed = 0U;
    require(qstate_apply_operations(state, bell_operations, 2U, &completed) == 0,
            "batched Bell preparation");
    require(completed == 2U, "batch completion count");

    double real = 0.0;
    double imag = 0.0;
    require(qstate_amplitude(state, 0, &real, &imag) == 0, "integer amplitude query");
    require_near(real, std::sqrt(0.5), 1e-12, "Bell |00> amplitude");
    require_near(imag, 0.0, 1e-12, "Bell |00> imaginary amplitude");

    const std::uint8_t bits[] = {1U, 1U};
    require(qstate_amplitude_bits(state, bits, 2U, &real, &imag) == 0, "bit-vector amplitude query");
    require_near(real, std::sqrt(0.5), 1e-12, "Bell |11> amplitude");

    int kind = -1;
    require(qstate_component_kind(state, 0, &kind) == 0, "component kind query");
    require(kind == 1, "Bell component must be sparse");
    require(qstate_validate(state) == 0, "native validation");

    const size_t qsc_size = qstate_qsc_size(state);
    require(qsc_size > 0U, "QSC size");
    std::vector<std::uint8_t> qsc(qsc_size);
    require(qstate_qsc_write(state, qsc.data(), qsc.size()) == 0, "QSC write");
    qstate_handle restored = qstate_qsc_read(qsc.data(), qsc.size());
    require(restored != nullptr, "QSC read");
    require(qstate_component_count(restored) == 1U, "QSC component partition");

    std::uint8_t measurements[2]{};
    require(qstate_measure_all(restored, 7U, measurements, 2U) == 0, "measure all");
    require(measurements[0] == measurements[1], "Bell measurements must correlate");

    qstate_destroy(restored);
    qstate_destroy(state);

    qstate_handle phase = qstate_create(1);
    require(phase != nullptr, "phase register creation");
    require(qstate_apply_x(phase, 0) == 0, "prepare |1>");
    require(qstate_apply_s(phase, 0) == 0, "S gate");
    require(qstate_apply_sdg(phase, 0) == 0, "S dagger gate");
    require(qstate_apply_t(phase, 0) == 0, "T gate");
    require(qstate_apply_tdg(phase, 0) == 0, "T dagger gate");
    require(qstate_apply_amplitude_damping_trajectory(phase, 0, 1.0, 0.0) == 0,
            "amplitude damping trajectory");
    double probability = 1.0;
    require(qstate_probability_one(phase, 0, &probability) == 0, "probability query");
    require_near(probability, 0.0, 1e-12, "full damping must produce |0>");
    qstate_destroy(phase);

    const qstate_operation fused_operations[] = {
        {QSTATE_OP_H, 0U, 0U, 0U, 0.0, 0.0},
        {QSTATE_OP_H, 0U, 0U, 0U, 0.0, 0.0},
        {QSTATE_OP_RY, 0U, 0U, 0U, 0.31, 0.0},
        {QSTATE_OP_RZ, 0U, 0U, 0U, -0.22, 0.0},
    };
    qstate_plan_handle plan = qstate_plan_create(
        fused_operations, 4U, static_cast<uint32_t>(QSTATE_PLAN_OPTIMIZE));
    require(plan != nullptr, "compiled plan creation");
    require(qstate_plan_source_operation_count(plan) == 4U, "compiled plan source count");
    require(qstate_plan_compiled_step_count(plan) == 2U, "single-qubit plan fusion");

    std::vector<qstate_handle> ensemble;
    for (std::size_t index = 0; index < 32U; ++index) {
        ensemble.push_back(qstate_create(1));
        require(ensemble.back() != nullptr, "ensemble register creation");
    }
    size_t completed_registers = 0U;
    require(qstate_plan_execute_many(
                plan,
                ensemble.data(),
                ensemble.size(),
                4U,
                &completed_registers) == 0,
            "parallel plan execution");
    require(completed_registers == ensemble.size(), "parallel plan completion count");
    for (qstate_handle member : ensemble) {
        double all_probabilities[1]{};
        require(qstate_probabilities_one(member, all_probabilities, 1U) == 0,
                "bulk probability query");
        require_near(all_probabilities[0], std::sin(0.31 / 2.0) * std::sin(0.31 / 2.0),
                     1e-12, "compiled plan probability");
        qstate_destroy(member);
    }
    qstate_plan_destroy(plan);

    const qstate_parameterized_operation parameterized_operations[] = {
        {QSTATE_OP_RY, 0U, 0U, 0U, 0.0, 0.0, 0, -1},
        {QSTATE_OP_RZ, 0U, 0U, 0U, 0.0, 0.0, 1, -1},
        {QSTATE_OP_CNOT, 0U, 1U, 0U, 0.0, 0.0, -1, -1},
    };
    qstate_parameterized_plan_handle parameterized = qstate_parameterized_plan_create(
        parameterized_operations, 3U, static_cast<uint32_t>(QSTATE_PLAN_OPTIMIZE));
    require(parameterized != nullptr, "parameterized plan creation");
    require(qstate_parameterized_plan_parameter_count(parameterized) == 2U,
            "parameterized plan slot count");
    qstate_handle parameterized_state = qstate_create(2);
    require(parameterized_state != nullptr, "parameterized state creation");
    const double values[] = {0.31, -0.22};
    completed = 0U;
    require(qstate_parameterized_plan_execute(
                parameterized_state, parameterized, values, 2U, &completed) == 0,
            "parameterized plan execution");
    require(completed == 3U, "parameterized plan completion count");
    require(qstate_probability_one(parameterized_state, 0U, &probability) == 0,
            "parameterized probability query");
    require_near(probability, std::sin(0.31 / 2.0) * std::sin(0.31 / 2.0),
                 1e-12, "parameterized plan probability");
    qstate_destroy(parameterized_state);
    qstate_parameterized_plan_destroy(parameterized);

    const uint64_t grover_marked[] = {5U};
    qstate_grover_handle grover = qstate_grover_create(3U, grover_marked, 1U);
    require(grover != nullptr, "compressed Grover creation");
    require(qstate_grover_space_size(grover) == 8U, "compressed Grover space size");
    require(qstate_grover_optimal_iterations(grover) == 2U, "compressed Grover optimum");
    require(qstate_grover_run_optimal(grover) == 0, "compressed Grover execution");
    require(qstate_grover_success_probability(grover, &probability) == 0,
            "compressed Grover probability");
    require_near(probability, 0.9453125, 1e-12, "compressed Grover amplification");
    uint64_t sampled_basis = 0U;
    require(qstate_grover_sample_basis(grover, 0.0, 0.5, &sampled_basis) == 0,
            "compressed Grover sampling");
    require(sampled_basis == 5U, "compressed Grover marked sample");
    require(qstate_grover_validate(grover) == 0, "compressed Grover validation");
    qstate_grover_destroy(grover);

    qstate_handle exact_grover = qstate_create(3U);
    require(exact_grover != nullptr, "exact Grover register creation");
    for (uint32_t qubit = 0; qubit < 3U; ++qubit) {
        require(qstate_apply_h(exact_grover, qubit) == 0, "exact Grover uniform state");
    }
    require(qstate_apply_grover_iterations(exact_grover, grover_marked, 1U, 2U) == 0,
            "exact Grover iterations");
    require(qstate_amplitude(exact_grover, 5U, &real, &imag) == 0,
            "exact Grover marked amplitude");
    require_near(real * real + imag * imag, 0.9453125, 1e-12,
                 "exact Grover amplification");
    qstate_destroy(exact_grover);

    const uint64_t symmetry_counts[] = {2U, 3U, 3U};
    qstate_symmetry_handle symmetry = qstate_symmetry_create_ordered(
        3U, symmetry_counts, 3U);
    require(symmetry != nullptr, "symmetry state creation");
    require(qstate_symmetry_class_count(symmetry) == 3U, "symmetry class count");
    require(qstate_symmetry_membership_mode(symmetry) == 1, "symmetry membership mode");
    require(qstate_symmetry_class_probability(symmetry, 0U, &probability) == 0,
            "symmetry probability query");
    require_near(probability, 0.25, 1e-12, "symmetry uniform probability");
    require(qstate_symmetry_apply_class_phase(symmetry, 0U, std::acos(-1.0)) == 0,
            "symmetry class phase");
    require(qstate_symmetry_apply_reflection(symmetry) == 0,
            "symmetry weighted reflection");
    require(qstate_symmetry_validate(symmetry) == 0, "symmetry validation");
    qstate_handle fallback = qstate_symmetry_to_register(symmetry, 3U);
    require(fallback != nullptr, "symmetry QRegister fallback");
    for (uint64_t basis = 0; basis < 8U; ++basis) {
        double symmetry_real = 0.0;
        double symmetry_imag = 0.0;
        require(qstate_symmetry_amplitude(
                    symmetry, basis, &symmetry_real, &symmetry_imag) == 0,
                "symmetry amplitude query");
        require(qstate_amplitude(fallback, basis, &real, &imag) == 0,
                "symmetry fallback amplitude query");
        require_near(real, symmetry_real, 1e-12, "symmetry fallback real amplitude");
        require_near(imag, symmetry_imag, 1e-12, "symmetry fallback imaginary amplitude");
    }
    qstate_destroy(fallback);
    qstate_symmetry_destroy(symmetry);

    qstate_handle uniform_source = qstate_create(6U);
    require(uniform_source != nullptr, "symmetry discovery source creation");
    for (uint32_t qubit = 0; qubit < 6U; ++qubit) {
        require(qstate_apply_h(uniform_source, qubit) == 0,
                "symmetry discovery uniform preparation");
    }
    qstate_symmetry_handle discovered = qstate_symmetry_discover(
        uniform_source, 6U, 1e-12, 16U);
    require(discovered != nullptr, "symmetry discovery");
    require(qstate_symmetry_class_count(discovered) == 1U,
            "symmetry discovery class count");
    double discovery_error = 1.0;
    require(qstate_symmetry_discovery_error(discovered, &discovery_error) == 0,
            "symmetry discovery error query");
    require(discovery_error < 1e-12, "symmetry discovery bounded error");
    qstate_symmetry_destroy(discovered);
    qstate_destroy(uniform_source);

    qstate_symmetry_handle hamming = qstate_symmetry_create_hamming_weight(60U);
    require(hamming != nullptr, "Hamming-weight symmetry creation");
    require(qstate_symmetry_class_count(hamming) == 61U,
            "Hamming-weight symmetry class count");
    require(qstate_symmetry_membership_mode(hamming) == 3,
            "Hamming-weight symmetry membership mode");
    require(qstate_symmetry_class_size(hamming, 30U) == 118264581564861424ULL,
            "Hamming-weight central binomial class size");
    qstate_symmetry_destroy(hamming);

    std::cout << "QSA C ABI compatibility tests passed.\n";
    return 0;
}
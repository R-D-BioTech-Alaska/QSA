#include "qubit/c_api.h"
#include "qubit/qplan.hpp"
#include "qubit/qgrover.hpp"
#include "qubit/qsymmetry.hpp"
#include "qubit/qstate.hpp"
#include "qubit/version.h"
#include <algorithm>
#include <cstring>
#include <exception>
#include <new>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

thread_local std::string last_error;

struct QStateHandle {
    explicit QStateHandle(std::size_t qubit_count) : state(qubit_count) {}
    explicit QStateHandle(qubit::QRegister&& value) : state(std::move(value)) {}

    qubit::QRegister state;
    std::vector<std::uint8_t> qsc_cache{};
    bool qsc_cache_valid{false};
};

struct QStatePlanHandle {
    QStatePlanHandle(std::span<const qubit::Operation> operations, bool optimize)
        : plan(operations, optimize) {}

    qubit::OperationPlan plan;
};

struct QStateParameterizedPlanHandle {
    QStateParameterizedPlanHandle(
        std::span<const qubit::ParameterizedOperation> operations,
        bool optimize)
        : plan(operations, optimize) {}

    qubit::ParameterizedOperationPlan plan;
};

struct QStateGroverHandle {
    explicit QStateGroverHandle(qubit::GroverSearch&& value) : search(std::move(value)) {}

    qubit::GroverSearch search;
};

struct QStateSymmetryHandle {
    explicit QStateSymmetryHandle(qubit::SymmetryState&& value) : state(std::move(value)) {}

    qubit::SymmetryState state;
};

QStateHandle* as_handle(qstate_handle handle) {
    if (handle == nullptr) {
        throw qubit::QStateError("Qubit state handle is null");
    }
    return static_cast<QStateHandle*>(handle);
}

qubit::QRegister* as_state(qstate_handle handle) {
    return &as_handle(handle)->state;
}

QStatePlanHandle* as_plan(qstate_plan_handle handle) {
    if (handle == nullptr) {
        throw qubit::QStateError("Qubit operation plan handle is null");
    }
    return static_cast<QStatePlanHandle*>(handle);
}

QStateParameterizedPlanHandle* as_parameterized_plan(qstate_parameterized_plan_handle handle) {
    if (handle == nullptr) {
        throw qubit::QStateError("Qubit parameterized plan handle is null");
    }
    return static_cast<QStateParameterizedPlanHandle*>(handle);
}

QStateGroverHandle* as_grover(qstate_grover_handle handle) {
    if (handle == nullptr) {
        throw qubit::QStateError("QSA Grover handle is null");
    }
    return static_cast<QStateGroverHandle*>(handle);
}

QStateSymmetryHandle* as_symmetry(qstate_symmetry_handle handle) {
    if (handle == nullptr) {
        throw qubit::QStateError("QSA symmetry handle is null");
    }
    return static_cast<QStateSymmetryHandle*>(handle);
}

std::vector<qubit::QComplex> complex_values(
    const double* real,
    const double* imag,
    std::size_t count,
    const char* label) {
    if (count != 0U && (real == nullptr || imag == nullptr)) {
        throw qubit::QStateError(std::string(label) + " buffer is null");
    }
    std::vector<qubit::QComplex> values;
    values.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        values.emplace_back(real[index], imag[index]);
    }
    return values;
}

void invalidate_qsc(qstate_handle handle) {
    QStateHandle* wrapper = as_handle(handle);
    wrapper->qsc_cache_valid = false;
    wrapper->qsc_cache.clear();
}

const std::vector<std::uint8_t>& encoded_qsc(qstate_handle handle) {
    QStateHandle* wrapper = as_handle(handle);
    if (!wrapper->qsc_cache_valid) {
        wrapper->qsc_cache = wrapper->state.encode_qsc();
        wrapper->qsc_cache_valid = true;
    }
    return wrapper->qsc_cache;
}

template <typename Function>
int guarded(Function&& function) {
    try {
        function();
        last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    } catch (...) {
        last_error = "Unknown Qubit native engine error";
        return -1;
    }
}

template <typename Function>
int guarded_mutation(qstate_handle handle, Function&& function) {
    return guarded([&] {
        invalidate_qsc(handle);
        function(*as_state(handle));
    });
}

void apply_operation(qubit::QRegister& state, const qstate_operation& operation) {
    switch (operation.opcode) {
        case QSTATE_OP_X:
            state.apply_x(operation.first);
            break;
        case QSTATE_OP_Y:
            state.apply_y(operation.first);
            break;
        case QSTATE_OP_Z:
            state.apply_z(operation.first);
            break;
        case QSTATE_OP_H:
            state.apply_h(operation.first);
            break;
        case QSTATE_OP_S:
            state.apply_s(operation.first);
            break;
        case QSTATE_OP_SDG:
            state.apply_sdg(operation.first);
            break;
        case QSTATE_OP_T:
            state.apply_t(operation.first);
            break;
        case QSTATE_OP_TDG:
            state.apply_tdg(operation.first);
            break;
        case QSTATE_OP_RX:
            state.apply_rx(operation.first, operation.parameter);
            break;
        case QSTATE_OP_RY:
            state.apply_ry(operation.first, operation.parameter);
            break;
        case QSTATE_OP_RZ:
            state.apply_rz(operation.first, operation.parameter);
            break;
        case QSTATE_OP_CNOT:
            state.apply_cnot(operation.first, operation.second);
            break;
        case QSTATE_OP_CZ:
            state.apply_cz(operation.first, operation.second);
            break;
        case QSTATE_OP_SWAP:
            state.apply_swap(operation.first, operation.second);
            break;
        case QSTATE_OP_BIT_FLIP_TRAJECTORY:
            state.apply_bit_flip_trajectory(
                operation.first, operation.parameter, operation.sample);
            break;
        case QSTATE_OP_PHASE_FLIP_TRAJECTORY:
            state.apply_phase_flip_trajectory(
                operation.first, operation.parameter, operation.sample);
            break;
        case QSTATE_OP_DEPOLARIZING_TRAJECTORY:
            state.apply_depolarizing_trajectory(
                operation.first, operation.parameter, operation.sample);
            break;
        case QSTATE_OP_AMPLITUDE_DAMPING_TRAJECTORY:
            state.apply_amplitude_damping_trajectory(
                operation.first, operation.parameter, operation.sample);
            break;
        default:
            throw qubit::QStateError("Batch operation contains an unknown opcode");
    }
}

qubit::Operation convert_operation(const qstate_operation& operation) {
    if (operation.opcode < QSTATE_OP_X || operation.opcode > QSTATE_OP_AMPLITUDE_DAMPING_TRAJECTORY) {
        throw qubit::QStateError("Operation contains an unknown opcode");
    }
    return qubit::Operation{
        static_cast<qubit::OperationCode>(operation.opcode),
        operation.first,
        operation.second,
        operation.parameter,
        operation.sample,
    };
}

qubit::ParameterizedOperation convert_parameterized_operation(
    const qstate_parameterized_operation& operation) {
    const qstate_operation concrete{
        operation.opcode,
        operation.first,
        operation.second,
        operation.reserved,
        operation.parameter,
        operation.sample,
    };
    return qubit::ParameterizedOperation{
        convert_operation(concrete),
        operation.parameter_slot,
        operation.sample_slot,
    };
}

} 

extern "C" {

qstate_handle qstate_create(size_t qubit_count) {
    try {
        auto* state = new QStateHandle(qubit_count);
        last_error.clear();
        return state;
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "Unknown Qubit native engine error";
        return nullptr;
    }
}

void qstate_destroy(qstate_handle handle) {
    delete static_cast<QStateHandle*>(handle);
}

const char* qstate_last_error(void) {
    return last_error.c_str();
}

int qstate_apply_x(qstate_handle handle, uint32_t qubit) {
    return guarded_mutation(handle, [&](qubit::QRegister& state) { state.apply_x(qubit); });
}

int qstate_apply_y(qstate_handle handle, uint32_t qubit) {
    return guarded_mutation(handle, [&](qubit::QRegister& state) { state.apply_y(qubit); });
}

int qstate_apply_z(qstate_handle handle, uint32_t qubit) {
    return guarded_mutation(handle, [&](qubit::QRegister& state) { state.apply_z(qubit); });
}

int qstate_apply_h(qstate_handle handle, uint32_t qubit) {
    return guarded_mutation(handle, [&](qubit::QRegister& state) { state.apply_h(qubit); });
}

int qstate_apply_s(qstate_handle handle, uint32_t qubit) {
    return guarded_mutation(handle, [&](qubit::QRegister& state) { state.apply_s(qubit); });
}

int qstate_apply_t(qstate_handle handle, uint32_t qubit) {
    return guarded_mutation(handle, [&](qubit::QRegister& state) { state.apply_t(qubit); });
}

int qstate_apply_rx(qstate_handle handle, uint32_t qubit, double theta) {
    return guarded_mutation(handle, [&](qubit::QRegister& state) { state.apply_rx(qubit, theta); });
}

int qstate_apply_ry(qstate_handle handle, uint32_t qubit, double theta) {
    return guarded_mutation(handle, [&](qubit::QRegister& state) { state.apply_ry(qubit, theta); });
}

int qstate_apply_rz(qstate_handle handle, uint32_t qubit, double theta) {
    return guarded_mutation(handle, [&](qubit::QRegister& state) { state.apply_rz(qubit, theta); });
}

int qstate_apply_cnot(qstate_handle handle, uint32_t control, uint32_t target) {
    return guarded_mutation(handle, [&](qubit::QRegister& state) {
        state.apply_cnot(control, target);
    });
}

int qstate_apply_cz(qstate_handle handle, uint32_t first, uint32_t second) {
    return guarded_mutation(handle, [&](qubit::QRegister& state) { state.apply_cz(first, second); });
}

int qstate_apply_swap(qstate_handle handle, uint32_t first, uint32_t second) {
    return guarded_mutation(handle, [&](qubit::QRegister& state) { state.apply_swap(first, second); });
}

int qstate_probability_one(qstate_handle handle, uint32_t qubit, double* result) {
    return guarded([&] {
        if (result == nullptr) {
            throw qubit::QStateError("Probability output pointer is null");
        }
        *result = as_state(handle)->probability_one(qubit);
    });
}

int qstate_measure(qstate_handle handle, uint32_t qubit, double sample, int* result) {
    return guarded_mutation(handle, [&](qubit::QRegister& state) {
        if (result == nullptr) {
            throw qubit::QStateError("Measurement output pointer is null");
        }
        *result = state.measure(qubit, sample);
    });
}

int qstate_amplitude(qstate_handle handle, uint64_t basis_index, double* real, double* imag) {
    return guarded([&] {
        if (real == nullptr || imag == nullptr) {
            throw qubit::QStateError("Amplitude output pointer is null");
        }
        const qubit::QComplex value = as_state(handle)->amplitude(basis_index);
        *real = value.re;
        *imag = value.im;
    });
}

size_t qstate_qubit_count(qstate_handle handle) {
    try {
        return as_state(handle)->qubit_count();
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0;
    }
}

size_t qstate_component_count(qstate_handle handle) {
    try {
        return as_state(handle)->component_count();
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0;
    }
}

size_t qstate_component_size(qstate_handle handle, uint32_t qubit) {
    try {
        return as_state(handle)->component_size(qubit);
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0;
    }
}

size_t qstate_component_nonzero_count(qstate_handle handle, uint32_t qubit) {
    try {
        return as_state(handle)->component_nonzero_count(qubit);
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0;
    }
}

size_t qstate_estimated_bytes(qstate_handle handle) {
    try {
        return as_state(handle)->estimated_bytes();
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0;
    }
}

size_t qstate_description_size(qstate_handle handle) {
    try {
        return as_state(handle)->describe().size() + 1U;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0;
    }
}

int qstate_description_write(qstate_handle handle, char* output, size_t output_size) {
    return guarded([&] {
        if (output == nullptr) {
            throw qubit::QStateError("Description output pointer is null");
        }
        const std::string description = as_state(handle)->describe();
        if (output_size < description.size() + 1U) {
            throw qubit::QStateError("Description output buffer is too small");
        }
        std::memcpy(output, description.c_str(), description.size() + 1U);
    });
}

size_t qstate_qsc_size(qstate_handle handle) {
    try {
        return encoded_qsc(handle).size();
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0;
    }
}

int qstate_qsc_write(qstate_handle handle, uint8_t* output, size_t output_size) {
    return guarded([&] {
        if (output == nullptr) {
            throw qubit::QStateError("QSC output pointer is null");
        }
        const std::vector<std::uint8_t>& encoded = encoded_qsc(handle);
        if (output_size < encoded.size()) {
            throw qubit::QStateError("QSC output buffer is too small");
        }
        std::copy(encoded.begin(), encoded.end(), output);
    });
}

qstate_handle qstate_qsc_read(const uint8_t* data, size_t data_size) {
    try {
        if (data == nullptr || data_size == 0U) {
            throw qubit::QStateError("QSC input buffer is empty");
        }
        qubit::QRegister decoded = qubit::QRegister::decode_qsc(
            std::span<const std::uint8_t>(data, data_size));
        auto* wrapper = new QStateHandle(std::move(decoded));
        wrapper->qsc_cache.assign(data, data + data_size);
        wrapper->qsc_cache_valid = true;
        last_error.clear();
        return wrapper;
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "Unknown Qubit native engine error";
        return nullptr;
    }
}

uint32_t qstate_abi_version_major(void) {
    return QSTATE_ABI_VERSION_MAJOR;
}

uint32_t qstate_abi_version_minor(void) {
    return QSTATE_ABI_VERSION_MINOR;
}

uint32_t qstate_abi_version_patch(void) {
    return QSTATE_ABI_VERSION_PATCH;
}

const char* qstate_version_string(void) {
    return QSTATE_VERSION_STRING;
}

int qstate_apply_sdg(qstate_handle handle, uint32_t qubit) {
    return guarded_mutation(handle, [&](qubit::QRegister& state) { state.apply_sdg(qubit); });
}

int qstate_apply_tdg(qstate_handle handle, uint32_t qubit) {
    return guarded_mutation(handle, [&](qubit::QRegister& state) { state.apply_tdg(qubit); });
}

int qstate_apply_bit_flip_trajectory(
    qstate_handle handle, uint32_t qubit, double probability, double sample) {
    return guarded_mutation(handle, [&](qubit::QRegister& state) {
        state.apply_bit_flip_trajectory(qubit, probability, sample);
    });
}

int qstate_apply_phase_flip_trajectory(
    qstate_handle handle, uint32_t qubit, double probability, double sample) {
    return guarded_mutation(handle, [&](qubit::QRegister& state) {
        state.apply_phase_flip_trajectory(qubit, probability, sample);
    });
}

int qstate_apply_depolarizing_trajectory(
    qstate_handle handle, uint32_t qubit, double probability, double sample) {
    return guarded_mutation(handle, [&](qubit::QRegister& state) {
        state.apply_depolarizing_trajectory(qubit, probability, sample);
    });
}

int qstate_apply_amplitude_damping_trajectory(
    qstate_handle handle, uint32_t qubit, double gamma, double sample) {
    return guarded_mutation(handle, [&](qubit::QRegister& state) {
        state.apply_amplitude_damping_trajectory(qubit, gamma, sample);
    });
}

int qstate_measure_all(
    qstate_handle handle, uint64_t seed, uint8_t* output, size_t output_size) {
    return guarded_mutation(handle, [&](qubit::QRegister& state) {
        if (output == nullptr) {
            throw qubit::QStateError("Measurement output buffer is null");
        }
        if (output_size < state.qubit_count()) {
            throw qubit::QStateError("Measurement output buffer is too small");
        }
        const std::vector<int> measurements = state.measure_all(seed);
        for (std::size_t index = 0; index < measurements.size(); ++index) {
            output[index] = static_cast<uint8_t>(measurements[index]);
        }
    });
}

int qstate_amplitude_bits(
    qstate_handle handle,
    const uint8_t* bits,
    size_t bit_count,
    double* real,
    double* imag) {
    return guarded([&] {
        if (bits == nullptr || real == nullptr || imag == nullptr) {
            throw qubit::QStateError("Amplitude bit-vector or output pointer is null");
        }
        const qubit::QComplex value = as_state(handle)->amplitude_bits(
            std::span<const std::uint8_t>(bits, bit_count));
        *real = value.re;
        *imag = value.im;
    });
}

int qstate_component_kind(qstate_handle handle, uint32_t qubit, int* result) {
    return guarded([&] {
        if (result == nullptr) {
            throw qubit::QStateError("Component-kind output pointer is null");
        }
        *result = static_cast<int>(as_state(handle)->component_kind(qubit));
    });
}

int qstate_validate(qstate_handle handle) {
    return guarded([&] {
        std::string reason;
        if (!as_state(handle)->validate(&reason)) {
            throw qubit::QStateError("QRegister validation failed: " + reason);
        }
    });
}

int qstate_apply_operations(
    qstate_handle handle,
    const qstate_operation* operations,
    size_t operation_count,
    size_t* completed_count) {
    if (completed_count != nullptr) {
        *completed_count = 0U;
    }
    return guarded_mutation(handle, [&](qubit::QRegister& state) {
        if (operation_count != 0U && operations == nullptr) {
            throw qubit::QStateError("Batch operation buffer is null");
        }
        for (std::size_t index = 0; index < operation_count; ++index) {
            try {
                apply_operation(state, operations[index]);
            } catch (const std::exception& error) {
                throw qubit::QStateError(
                    "Batch operation " + std::to_string(index) + " failed: " + error.what());
            }
            if (completed_count != nullptr) {
                *completed_count = index + 1U;
            }
        }
    });
}

qstate_plan_handle qstate_plan_create(
    const qstate_operation* operations,
    size_t operation_count,
    uint32_t flags) {
    try {
        if (operation_count != 0U && operations == nullptr) {
            throw qubit::QStateError("Operation plan buffer is null");
        }
        if ((flags & ~static_cast<uint32_t>(QSTATE_PLAN_OPTIMIZE)) != 0U) {
            throw qubit::QStateError("Operation plan contains unsupported flags");
        }
        std::vector<qubit::Operation> converted;
        converted.reserve(operation_count);
        for (std::size_t index = 0; index < operation_count; ++index) {
            try {
                converted.push_back(convert_operation(operations[index]));
            } catch (const std::exception& error) {
                throw qubit::QStateError(
                    "Operation plan entry " + std::to_string(index) + " failed: " + error.what());
            }
        }
        auto* plan = new QStatePlanHandle(
            converted, (flags & static_cast<uint32_t>(QSTATE_PLAN_OPTIMIZE)) != 0U);
        last_error.clear();
        return plan;
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "Unknown Qubit operation plan error";
        return nullptr;
    }
}

void qstate_plan_destroy(qstate_plan_handle plan) {
    delete static_cast<QStatePlanHandle*>(plan);
}

size_t qstate_plan_source_operation_count(qstate_plan_handle plan) {
    try {
        const std::size_t count = as_plan(plan)->plan.source_operation_count();
        last_error.clear();
        return count;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0U;
    }
}

size_t qstate_plan_compiled_step_count(qstate_plan_handle plan) {
    try {
        const std::size_t count = as_plan(plan)->plan.compiled_step_count();
        last_error.clear();
        return count;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0U;
    }
}

int qstate_plan_execute(
    qstate_handle handle,
    qstate_plan_handle plan,
    size_t* completed_operation_count) {
    if (completed_operation_count != nullptr) {
        *completed_operation_count = 0U;
    }
    return guarded_mutation(handle, [&](qubit::QRegister& state) {
        as_plan(plan)->plan.execute(state, completed_operation_count);
    });
}

int qstate_plan_execute_many(
    qstate_plan_handle plan,
    qstate_handle* handles,
    size_t handle_count,
    size_t worker_count,
    size_t* completed_handle_count) {
    if (completed_handle_count != nullptr) {
        *completed_handle_count = 0U;
    }
    return guarded([&] {
        if (handle_count != 0U && handles == nullptr) {
            throw qubit::QStateError("Operation-plan register buffer is null");
        }
        std::vector<qubit::QRegister*> states;
        states.reserve(handle_count);
        for (std::size_t index = 0; index < handle_count; ++index) {
            QStateHandle* wrapper = as_handle(handles[index]);
            wrapper->qsc_cache_valid = false;
            wrapper->qsc_cache.clear();
            states.push_back(&wrapper->state);
        }
        as_plan(plan)->plan.execute_many(states, worker_count, completed_handle_count);
    });
}

int qstate_probabilities_one(qstate_handle handle, double* output, size_t output_size) {
    return guarded([&] {
        if (output == nullptr) {
            throw qubit::QStateError("Probability output buffer is null");
        }
        const std::vector<double> probabilities = as_state(handle)->probabilities_one();
        if (output_size < probabilities.size()) {
            throw qubit::QStateError("Probability output buffer is too small");
        }
        std::copy(probabilities.begin(), probabilities.end(), output);
    });
}

qstate_parameterized_plan_handle qstate_parameterized_plan_create(
    const qstate_parameterized_operation* operations,
    size_t operation_count,
    uint32_t flags) {
    try {
        if (operation_count != 0U && operations == nullptr) {
            throw qubit::QStateError("Parameterized plan buffer is null");
        }
        if ((flags & ~static_cast<uint32_t>(QSTATE_PLAN_OPTIMIZE)) != 0U) {
            throw qubit::QStateError("Parameterized plan contains unsupported flags");
        }
        std::vector<qubit::ParameterizedOperation> converted;
        converted.reserve(operation_count);
        for (std::size_t index = 0; index < operation_count; ++index) {
            try {
                converted.push_back(convert_parameterized_operation(operations[index]));
            } catch (const std::exception& error) {
                throw qubit::QStateError(
                    "Parameterized plan entry " + std::to_string(index) + " failed: " +
                    error.what());
            }
        }
        auto* plan = new QStateParameterizedPlanHandle(
            converted, (flags & static_cast<uint32_t>(QSTATE_PLAN_OPTIMIZE)) != 0U);
        last_error.clear();
        return plan;
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "Unknown Qubit parameterized plan error";
        return nullptr;
    }
}

void qstate_parameterized_plan_destroy(qstate_parameterized_plan_handle plan) {
    delete static_cast<QStateParameterizedPlanHandle*>(plan);
}

size_t qstate_parameterized_plan_source_operation_count(qstate_parameterized_plan_handle plan) {
    try {
        const std::size_t count = as_parameterized_plan(plan)->plan.source_operation_count();
        last_error.clear();
        return count;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0U;
    }
}

size_t qstate_parameterized_plan_parameter_count(qstate_parameterized_plan_handle plan) {
    try {
        const std::size_t count = as_parameterized_plan(plan)->plan.parameter_count();
        last_error.clear();
        return count;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0U;
    }
}

int qstate_parameterized_plan_execute(
    qstate_handle handle,
    qstate_parameterized_plan_handle plan,
    const double* parameters,
    size_t parameter_count,
    size_t* completed_operation_count) {
    if (completed_operation_count != nullptr) {
        *completed_operation_count = 0U;
    }
    return guarded_mutation(handle, [&](qubit::QRegister& state) {
        if (parameter_count != 0U && parameters == nullptr) {
            throw qubit::QStateError("Parameterized plan values buffer is null");
        }
        as_parameterized_plan(plan)->plan.execute(
            state,
            std::span<const double>(parameters, parameter_count),
            completed_operation_count);
    });
}

int qstate_parameterized_plan_execute_many(
    qstate_parameterized_plan_handle plan,
    qstate_handle* handles,
    size_t handle_count,
    const double* parameters,
    size_t parameter_count,
    size_t worker_count,
    size_t* completed_handle_count) {
    if (completed_handle_count != nullptr) {
        *completed_handle_count = 0U;
    }
    return guarded([&] {
        if (handle_count != 0U && handles == nullptr) {
            throw qubit::QStateError("Parameterized-plan register buffer is null");
        }
        if (parameter_count != 0U && parameters == nullptr) {
            throw qubit::QStateError("Parameterized plan values buffer is null");
        }
        std::vector<qubit::QRegister*> states;
        states.reserve(handle_count);
        for (std::size_t index = 0; index < handle_count; ++index) {
            QStateHandle* wrapper = as_handle(handles[index]);
            wrapper->qsc_cache_valid = false;
            wrapper->qsc_cache.clear();
            states.push_back(&wrapper->state);
        }
        as_parameterized_plan(plan)->plan.execute_many(
            states,
            std::span<const double>(parameters, parameter_count),
            worker_count,
            completed_handle_count);
    });
}

int qstate_apply_grover_oracle(
    qstate_handle handle,
    const uint64_t* marked_indices,
    size_t marked_count) {
    return guarded_mutation(handle, [&](qubit::QRegister& state) {
        if (marked_count != 0U && marked_indices == nullptr) {
            throw qubit::QStateError("Grover marked-index buffer is null");
        }
        state.apply_grover_oracle(
            std::span<const qubit::BasisIndex>(marked_indices, marked_count));
    });
}

int qstate_apply_grover_diffusion(qstate_handle handle) {
    return guarded_mutation(handle, [&](qubit::QRegister& state) {
        state.apply_grover_diffusion();
    });
}

int qstate_apply_grover_iterations(
    qstate_handle handle,
    const uint64_t* marked_indices,
    size_t marked_count,
    uint64_t iteration_count) {
    return guarded_mutation(handle, [&](qubit::QRegister& state) {
        if (marked_count != 0U && marked_indices == nullptr) {
            throw qubit::QStateError("Grover marked-index buffer is null");
        }
        state.apply_grover_iterations(
            std::span<const qubit::BasisIndex>(marked_indices, marked_count),
            iteration_count);
    });
}

qstate_grover_handle qstate_grover_create(
    size_t qubit_count,
    const uint64_t* marked_indices,
    size_t marked_count) {
    try {
        if (marked_count != 0U && marked_indices == nullptr) {
            throw qubit::QStateError("Grover marked-index buffer is null");
        }
        auto* handle = new QStateGroverHandle(qubit::GroverSearch(
            qubit_count,
            std::span<const qubit::BasisIndex>(marked_indices, marked_count)));
        last_error.clear();
        return handle;
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "Unknown QSA Grover creation error";
        return nullptr;
    }
}

qstate_grover_handle qstate_grover_create_count(
    size_t qubit_count,
    uint64_t marked_count) {
    try {
        auto* handle = new QStateGroverHandle(
            qubit::GroverSearch::from_marked_count(qubit_count, marked_count));
        last_error.clear();
        return handle;
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "Unknown QSA Grover creation error";
        return nullptr;
    }
}

void qstate_grover_destroy(qstate_grover_handle handle) {
    delete static_cast<QStateGroverHandle*>(handle);
}

int qstate_grover_reset(qstate_grover_handle handle) {
    return guarded([&] { as_grover(handle)->search.reset(); });
}

int qstate_grover_apply_oracle(qstate_grover_handle handle) {
    return guarded([&] { as_grover(handle)->search.apply_oracle(); });
}

int qstate_grover_apply_diffusion(qstate_grover_handle handle) {
    return guarded([&] { as_grover(handle)->search.apply_diffusion(); });
}

int qstate_grover_iterate(qstate_grover_handle handle, uint64_t iteration_count) {
    return guarded([&] { as_grover(handle)->search.iterate(iteration_count); });
}

int qstate_grover_run_optimal(qstate_grover_handle handle) {
    return guarded([&] { as_grover(handle)->search.run_optimal(); });
}

size_t qstate_grover_qubit_count(qstate_grover_handle handle) {
    try {
        const std::size_t value = as_grover(handle)->search.qubit_count();
        last_error.clear();
        return value;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0U;
    }
}

uint64_t qstate_grover_space_size(qstate_grover_handle handle) {
    try {
        const uint64_t value = as_grover(handle)->search.space_size();
        last_error.clear();
        return value;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0U;
    }
}

uint64_t qstate_grover_marked_count(qstate_grover_handle handle) {
    try {
        const uint64_t value = as_grover(handle)->search.marked_count();
        last_error.clear();
        return value;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0U;
    }
}

uint64_t qstate_grover_iteration_count(qstate_grover_handle handle) {
    try {
        const uint64_t value = as_grover(handle)->search.iteration_count();
        last_error.clear();
        return value;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0U;
    }
}

uint64_t qstate_grover_optimal_iterations(qstate_grover_handle handle) {
    try {
        const uint64_t value = as_grover(handle)->search.optimal_iterations();
        last_error.clear();
        return value;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0U;
    }
}

size_t qstate_grover_estimated_bytes(qstate_grover_handle handle) {
    try {
        const std::size_t value = as_grover(handle)->search.estimated_bytes();
        last_error.clear();
        return value;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0U;
    }
}

int qstate_grover_has_explicit_marked_indices(qstate_grover_handle handle) {
    try {
        const bool value = as_grover(handle)->search.has_explicit_marked_indices();
        last_error.clear();
        return value ? 1 : 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

int qstate_grover_success_probability(qstate_grover_handle handle, double* result) {
    return guarded([&] {
        if (result == nullptr) {
            throw qubit::QStateError("Grover probability output pointer is null");
        }
        *result = as_grover(handle)->search.success_probability();
    });
}

int qstate_grover_marked_amplitude(
    qstate_grover_handle handle,
    double* real,
    double* imag) {
    return guarded([&] {
        if (real == nullptr || imag == nullptr) {
            throw qubit::QStateError("Grover amplitude output pointer is null");
        }
        const qubit::QComplex value = as_grover(handle)->search.marked_amplitude();
        *real = value.re;
        *imag = value.im;
    });
}

int qstate_grover_unmarked_amplitude(
    qstate_grover_handle handle,
    double* real,
    double* imag) {
    return guarded([&] {
        if (real == nullptr || imag == nullptr) {
            throw qubit::QStateError("Grover amplitude output pointer is null");
        }
        const qubit::QComplex value = as_grover(handle)->search.unmarked_amplitude();
        *real = value.re;
        *imag = value.im;
    });
}

int qstate_grover_amplitude(
    qstate_grover_handle handle,
    uint64_t basis_index,
    double* real,
    double* imag) {
    return guarded([&] {
        if (real == nullptr || imag == nullptr) {
            throw qubit::QStateError("Grover amplitude output pointer is null");
        }
        const qubit::QComplex value = as_grover(handle)->search.amplitude(basis_index);
        *real = value.re;
        *imag = value.im;
    });
}

int qstate_grover_sample_basis(
    qstate_grover_handle handle,
    double branch_sample,
    double index_sample,
    uint64_t* result) {
    return guarded([&] {
        if (result == nullptr) {
            throw qubit::QStateError("Grover sample output pointer is null");
        }
        *result = as_grover(handle)->search.sample_basis(branch_sample, index_sample);
    });
}

int qstate_grover_validate(qstate_grover_handle handle) {
    return guarded([&] {
        std::string reason;
        if (!as_grover(handle)->search.validate(&reason)) {
            throw qubit::QStateError("Grover validation failed: " + reason);
        }
    });
}

size_t qstate_grover_description_size(qstate_grover_handle handle) {
    try {
        const std::size_t size = as_grover(handle)->search.describe().size() + 1U;
        last_error.clear();
        return size;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0U;
    }
}

int qstate_grover_description_write(
    qstate_grover_handle handle,
    char* output,
    size_t output_size) {
    return guarded([&] {
        if (output == nullptr) {
            throw qubit::QStateError("Grover description output pointer is null");
        }
        const std::string description = as_grover(handle)->search.describe();
        if (output_size < description.size() + 1U) {
            throw qubit::QStateError("Grover description output buffer is too small");
        }
        std::memcpy(output, description.c_str(), description.size() + 1U);
    });
}

qstate_symmetry_handle qstate_symmetry_create_ordered(
    size_t qubit_count,
    const uint64_t* class_counts,
    size_t class_count) {
    try {
        if (class_count != 0U && class_counts == nullptr) {
            throw qubit::QStateError("Symmetry class-count buffer is null");
        }
        auto* handle = new QStateSymmetryHandle(qubit::SymmetryState(
            qubit_count,
            std::span<const qubit::BasisIndex>(class_counts, class_count)));
        last_error.clear();
        return handle;
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "Unknown QSA symmetry creation error";
        return nullptr;
    }
}

qstate_symmetry_handle qstate_symmetry_create_count_only(
    size_t qubit_count,
    const uint64_t* class_counts,
    size_t class_count) {
    try {
        if (class_count != 0U && class_counts == nullptr) {
            throw qubit::QStateError("Symmetry class-count buffer is null");
        }
        auto* handle = new QStateSymmetryHandle(qubit::SymmetryState::from_counts(
            qubit_count,
            std::span<const qubit::BasisIndex>(class_counts, class_count)));
        last_error.clear();
        return handle;
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "Unknown QSA symmetry creation error";
        return nullptr;
    }
}

qstate_symmetry_handle qstate_symmetry_create_labels(
    size_t qubit_count,
    const uint32_t* labels,
    size_t label_count) {
    try {
        if (label_count != 0U && labels == nullptr) {
            throw qubit::QStateError("Symmetry label buffer is null");
        }
        auto* handle = new QStateSymmetryHandle(qubit::SymmetryState::from_labels(
            qubit_count,
            std::span<const std::uint32_t>(labels, label_count)));
        last_error.clear();
        return handle;
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "Unknown QSA symmetry creation error";
        return nullptr;
    }
}

qstate_symmetry_handle qstate_symmetry_create_hamming_weight(size_t qubit_count) {
    try {
        auto* handle = new QStateSymmetryHandle(
            qubit::SymmetryState::hamming_weight(qubit_count));
        last_error.clear();
        return handle;
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "Unknown QSA Hamming-weight symmetry creation error";
        return nullptr;
    }
}

qstate_symmetry_handle qstate_symmetry_discover(
    qstate_handle source,
    size_t max_qubits,
    double tolerance,
    size_t max_classes) {
    try {
        auto* handle = new QStateSymmetryHandle(qubit::SymmetryState::discover(
            *as_state(source), max_qubits, tolerance, max_classes));
        last_error.clear();
        return handle;
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "Unknown QSA symmetry discovery error";
        return nullptr;
    }
}

void qstate_symmetry_destroy(qstate_symmetry_handle handle) {
    delete static_cast<QStateSymmetryHandle*>(handle);
}

int qstate_symmetry_reset_uniform(qstate_symmetry_handle handle) {
    return guarded([&] { as_symmetry(handle)->state.reset_uniform(); });
}

int qstate_symmetry_set_amplitudes(
    qstate_symmetry_handle handle,
    const double* real,
    const double* imag,
    size_t amplitude_count,
    int normalize) {
    return guarded([&] {
        const auto values = complex_values(real, imag, amplitude_count, "Symmetry amplitude");
        as_symmetry(handle)->state.set_class_amplitudes(values, normalize != 0);
    });
}

int qstate_symmetry_apply_class_phase(
    qstate_symmetry_handle handle,
    size_t class_index,
    double angle) {
    return guarded([&] { as_symmetry(handle)->state.apply_class_phase(class_index, angle); });
}

int qstate_symmetry_apply_class_phases(
    qstate_symmetry_handle handle,
    const double* angles,
    size_t angle_count) {
    return guarded([&] {
        if (angle_count != 0U && angles == nullptr) {
            throw qubit::QStateError("Symmetry phase buffer is null");
        }
        as_symmetry(handle)->state.apply_class_phases(
            std::span<const double>(angles, angle_count));
    });
}

int qstate_symmetry_apply_reflection(qstate_symmetry_handle handle) {
    return guarded([&] { as_symmetry(handle)->state.apply_weighted_reflection(); });
}

int qstate_symmetry_split_class(
    qstate_symmetry_handle handle,
    size_t class_index,
    uint64_t first_count,
    size_t* new_class_index) {
    return guarded([&] {
        if (new_class_index == nullptr) {
            throw qubit::QStateError("Symmetry split result pointer is null");
        }
        *new_class_index = as_symmetry(handle)->state.split_class(class_index, first_count);
    });
}

int qstate_symmetry_merge_equivalent(
    qstate_symmetry_handle handle,
    double tolerance,
    size_t* removed_count) {
    return guarded([&] {
        if (removed_count == nullptr) {
            throw qubit::QStateError("Symmetry merge result pointer is null");
        }
        *removed_count = as_symmetry(handle)->state.merge_equivalent(tolerance);
    });
}

int qstate_symmetry_apply_unitary(
    qstate_symmetry_handle handle,
    const double* real,
    const double* imag,
    size_t element_count,
    double tolerance) {
    return guarded([&] {
        const auto matrix = complex_values(real, imag, element_count, "Symmetry unitary");
        as_symmetry(handle)->state.apply_class_unitary(matrix, tolerance);
    });
}

int qstate_symmetry_iterate_unitary(
    qstate_symmetry_handle handle,
    const double* real,
    const double* imag,
    size_t element_count,
    uint64_t iteration_count,
    double tolerance) {
    return guarded([&] {
        const auto matrix = complex_values(real, imag, element_count, "Symmetry unitary");
        as_symmetry(handle)->state.iterate_class_unitary(
            matrix, iteration_count, tolerance);
    });
}

size_t qstate_symmetry_qubit_count(qstate_symmetry_handle handle) {
    try {
        const auto value = as_symmetry(handle)->state.qubit_count();
        last_error.clear();
        return value;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0U;
    }
}

uint64_t qstate_symmetry_space_size(qstate_symmetry_handle handle) {
    try {
        const auto value = as_symmetry(handle)->state.space_size();
        last_error.clear();
        return value;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0U;
    }
}

size_t qstate_symmetry_class_count(qstate_symmetry_handle handle) {
    try {
        const auto value = as_symmetry(handle)->state.class_count();
        last_error.clear();
        return value;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0U;
    }
}

int qstate_symmetry_membership_mode(qstate_symmetry_handle handle) {
    try {
        const auto value = static_cast<int>(as_symmetry(handle)->state.membership());
        last_error.clear();
        return value;
    } catch (const std::exception& error) {
        last_error = error.what();
        return -1;
    }
}

uint64_t qstate_symmetry_class_size(
    qstate_symmetry_handle handle,
    size_t class_index) {
    try {
        const auto value = as_symmetry(handle)->state.class_size(class_index);
        last_error.clear();
        return value;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0U;
    }
}

int qstate_symmetry_class_amplitude(
    qstate_symmetry_handle handle,
    size_t class_index,
    double* real,
    double* imag) {
    return guarded([&] {
        if (real == nullptr || imag == nullptr) {
            throw qubit::QStateError("Symmetry amplitude output pointer is null");
        }
        const auto value = as_symmetry(handle)->state.class_amplitude(class_index);
        *real = value.re;
        *imag = value.im;
    });
}

int qstate_symmetry_class_probability(
    qstate_symmetry_handle handle,
    size_t class_index,
    double* result) {
    return guarded([&] {
        if (result == nullptr) {
            throw qubit::QStateError("Symmetry probability output pointer is null");
        }
        *result = as_symmetry(handle)->state.class_probability(class_index);
    });
}

int qstate_symmetry_amplitude(
    qstate_symmetry_handle handle,
    uint64_t basis_index,
    double* real,
    double* imag) {
    return guarded([&] {
        if (real == nullptr || imag == nullptr) {
            throw qubit::QStateError("Symmetry amplitude output pointer is null");
        }
        const auto value = as_symmetry(handle)->state.amplitude(basis_index);
        *real = value.re;
        *imag = value.im;
    });
}

int qstate_symmetry_sample_class(
    qstate_symmetry_handle handle,
    double sample,
    size_t* result) {
    return guarded([&] {
        if (result == nullptr) {
            throw qubit::QStateError("Symmetry class sample output pointer is null");
        }
        *result = as_symmetry(handle)->state.sample_class(sample);
    });
}

int qstate_symmetry_sample_basis(
    qstate_symmetry_handle handle,
    double class_sample,
    double index_sample,
    uint64_t* result) {
    return guarded([&] {
        if (result == nullptr) {
            throw qubit::QStateError("Symmetry basis sample output pointer is null");
        }
        *result = as_symmetry(handle)->state.sample_basis(class_sample, index_sample);
    });
}

qstate_handle qstate_symmetry_to_register(
    qstate_symmetry_handle handle,
    size_t max_qubits) {
    try {
        auto* result = new QStateHandle(as_symmetry(handle)->state.to_register(max_qubits));
        last_error.clear();
        return result;
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "Unknown QSA symmetry materialization error";
        return nullptr;
    }
}

size_t qstate_symmetry_estimated_bytes(qstate_symmetry_handle handle) {
    try {
        const auto value = as_symmetry(handle)->state.estimated_bytes();
        last_error.clear();
        return value;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0U;
    }
}

int qstate_symmetry_discovery_error(
    qstate_symmetry_handle handle,
    double* result) {
    return guarded([&] {
        if (result == nullptr) {
            throw qubit::QStateError("Symmetry discovery-error output pointer is null");
        }
        *result = as_symmetry(handle)->state.discovery_error();
    });
}

int qstate_symmetry_validate(qstate_symmetry_handle handle) {
    return guarded([&] {
        std::string reason;
        if (!as_symmetry(handle)->state.validate(&reason)) {
            throw qubit::QStateError("Symmetry validation failed: " + reason);
        }
    });
}

size_t qstate_symmetry_description_size(qstate_symmetry_handle handle) {
    try {
        const auto value = as_symmetry(handle)->state.describe().size() + 1U;
        last_error.clear();
        return value;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 0U;
    }
}

int qstate_symmetry_description_write(
    qstate_symmetry_handle handle,
    char* output,
    size_t output_size) {
    return guarded([&] {
        if (output == nullptr) {
            throw qubit::QStateError("Symmetry description output pointer is null");
        }
        const std::string description = as_symmetry(handle)->state.describe();
        if (output_size < description.size() + 1U) {
            throw qubit::QStateError("Symmetry description output buffer is too small");
        }
        std::memcpy(output, description.c_str(), description.size() + 1U);
    });
}

} 

#include "qubit/qcausal_c_api.h"

#include "qubit/qcausal.hpp"
#include "qubit/qplan.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <new>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

thread_local std::string causal_last_error;

struct CausalHandle {
    explicit CausalHandle(std::size_t qubit_count) : state(qubit_count) {}
    explicit CausalHandle(qubit::QRegister value) : state(std::move(value)) {}
    explicit CausalHandle(qubit::CausalState value) : state(std::move(value)) {}

    qubit::CausalState state;
    std::vector<std::uint8_t> qsc_cache{};
    bool qsc_cache_valid{false};
};

struct CausalParameterizedPlanHandle {
    CausalParameterizedPlanHandle(
        std::span<const qubit::ParameterizedOperation> operations,
        bool optimize)
        : plan(operations, optimize) {}

    qubit::ParameterizedOperationPlan plan;
};

struct CausalPauliPlanHandle {
    CausalPauliPlanHandle(
        std::size_t qubit_count,
        std::span<const std::string> words,
        double imaginary_tolerance)
        : plan(
              qubit_count,
              words,
              qubit::PauliObservableConfig{imaginary_tolerance}) {}

    qubit::PauliObservablePlan plan;
};

CausalHandle* as_causal(qcausal_handle handle) {
    if (handle == nullptr) {
        throw qubit::QStateError("QSA causal state handle is null");
    }
    return static_cast<CausalHandle*>(handle);
}

CausalParameterizedPlanHandle* as_parameterized_plan(
    qcausal_parameterized_plan_handle handle) {
    if (handle == nullptr) {
        throw qubit::QStateError("QSA causal parameterized plan handle is null");
    }
    return static_cast<CausalParameterizedPlanHandle*>(handle);
}

CausalPauliPlanHandle* as_pauli_plan(qcausal_pauli_plan_handle handle) {
    if (handle == nullptr) {
        throw qubit::QStateError("QSA causal Pauli plan handle is null");
    }
    return static_cast<CausalPauliPlanHandle*>(handle);
}

void invalidate_qsc(CausalHandle& handle) {
    handle.qsc_cache_valid = false;
    handle.qsc_cache.clear();
}

const std::vector<std::uint8_t>& encoded_qsc(CausalHandle& handle) {
    if (!handle.qsc_cache_valid) {
        handle.qsc_cache = handle.state.read().encode_qsc();
        handle.qsc_cache_valid = true;
    }
    return handle.qsc_cache;
}

template <class Function>
int guarded(Function&& function) {
    try {
        function();
        causal_last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        causal_last_error = error.what();
        return -1;
    } catch (...) {
        causal_last_error = "Unknown QSA causal runtime error";
        return -1;
    }
}

qubit::Operation convert_operation(const qstate_parameterized_operation& operation) {
    if (operation.opcode < QSTATE_OP_X ||
        operation.opcode > QSTATE_OP_AMPLITUDE_DAMPING_TRAJECTORY) {
        throw qubit::QStateError("Causal plan contains an unknown opcode");
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
    return qubit::ParameterizedOperation{
        convert_operation(operation),
        operation.parameter_slot,
        operation.sample_slot,
    };
}

std::vector<std::string> decode_pauli_words(
    std::size_t qubit_count,
    const char* words,
    std::size_t word_count) {
    if (word_count != 0U && words == nullptr) {
        throw qubit::QStateError("Pauli word buffer is null");
    }
    if (qubit_count == 0U) {
        throw qubit::QStateError("Pauli plan qubit count must be positive");
    }
    std::vector<std::string> decoded;
    decoded.reserve(word_count);
    for (std::size_t index = 0; index < word_count; ++index) {
        const char* begin = words + index * qubit_count;
        decoded.emplace_back(begin, begin + qubit_count);
    }
    return decoded;
}

}  // namespace

extern "C" {

uint32_t qcausal_api_version_major(void) {
    return 0U;
}

uint32_t qcausal_api_version_minor(void) {
    return 1U;
}

uint32_t qcausal_api_version_patch(void) {
    return 0U;
}

const char* qcausal_last_error(void) {
    return causal_last_error.c_str();
}

qcausal_handle qcausal_create(size_t qubit_count) {
    try {
        auto* handle = new CausalHandle(qubit_count);
        causal_last_error.clear();
        return handle;
    } catch (const std::exception& error) {
        causal_last_error = error.what();
        return nullptr;
    } catch (...) {
        causal_last_error = "Unknown QSA causal runtime error";
        return nullptr;
    }
}

qcausal_handle qcausal_from_qsc(const uint8_t* data, size_t data_size) {
    try {
        if (data == nullptr || data_size == 0U) {
            throw qubit::QStateError("QSC input is null or empty");
        }
        qubit::QRegister decoded = qubit::QRegister::decode_qsc(
            std::span<const std::uint8_t>(data, data_size));
        auto* handle = new CausalHandle(std::move(decoded));
        causal_last_error.clear();
        return handle;
    } catch (const std::exception& error) {
        causal_last_error = error.what();
        return nullptr;
    } catch (...) {
        causal_last_error = "Unknown QSA causal runtime error";
        return nullptr;
    }
}

qcausal_handle qcausal_fork(qcausal_handle source) {
    try {
        auto* branch = new CausalHandle(as_causal(source)->state.fork());
        causal_last_error.clear();
        return branch;
    } catch (const std::exception& error) {
        causal_last_error = error.what();
        return nullptr;
    } catch (...) {
        causal_last_error = "Unknown QSA causal runtime error";
        return nullptr;
    }
}

int qcausal_adopt(qcausal_handle target, qcausal_handle selected) {
    return guarded([&] {
        CausalHandle* target_handle = as_causal(target);
        CausalHandle* selected_handle = as_causal(selected);
        if (target_handle == selected_handle) {
            throw qubit::QStateError("Causal state cannot adopt itself");
        }
        target_handle->state.adopt(std::move(selected_handle->state));
        invalidate_qsc(*target_handle);
        invalidate_qsc(*selected_handle);
    });
}

void qcausal_destroy(qcausal_handle handle) {
    delete static_cast<CausalHandle*>(handle);
}

size_t qcausal_qubit_count(qcausal_handle handle) {
    try {
        const std::size_t value = as_causal(handle)->state.read().qubit_count();
        causal_last_error.clear();
        return value;
    } catch (const std::exception& error) {
        causal_last_error = error.what();
        return 0U;
    }
}

size_t qcausal_component_count(qcausal_handle handle) {
    try {
        const std::size_t value = as_causal(handle)->state.read().component_count();
        causal_last_error.clear();
        return value;
    } catch (const std::exception& error) {
        causal_last_error = error.what();
        return 0U;
    }
}

size_t qcausal_estimated_bytes(qcausal_handle handle) {
    try {
        const std::size_t value = as_causal(handle)->state.estimated_bytes();
        causal_last_error.clear();
        return value;
    } catch (const std::exception& error) {
        causal_last_error = error.what();
        return 0U;
    }
}

size_t qcausal_shared_owner_count(qcausal_handle handle) {
    try {
        const long count = as_causal(handle)->state.shared_owner_count();
        if (count < 0L) {
            throw qubit::QStateError("Causal state owner count is invalid");
        }
        causal_last_error.clear();
        return static_cast<std::size_t>(count);
    } catch (const std::exception& error) {
        causal_last_error = error.what();
        return 0U;
    }
}

int qcausal_validate(qcausal_handle handle) {
    return guarded([&] {
        std::string reason;
        if (!as_causal(handle)->state.read().validate(&reason)) {
            throw qubit::QStateError(
                reason.empty() ? "Causal state validation failed" : reason);
        }
    });
}

int qcausal_amplitude(
    qcausal_handle handle,
    uint64_t basis_index,
    double* real,
    double* imag) {
    return guarded([&] {
        if (real == nullptr || imag == nullptr) {
            throw qubit::QStateError("Causal amplitude output is null");
        }
        const qubit::QComplex value =
            as_causal(handle)->state.read().amplitude(basis_index);
        *real = value.re;
        *imag = value.im;
    });
}

int qcausal_probabilities_one(
    qcausal_handle handle,
    double* output,
    size_t output_size) {
    return guarded([&] {
        if (output == nullptr && output_size != 0U) {
            throw qubit::QStateError("Causal probability output is null");
        }
        as_causal(handle)->state.read().probabilities_one_into(
            std::span<double>(output, output_size));
    });
}

size_t qcausal_qsc_size(qcausal_handle handle) {
    try {
        const std::size_t value = encoded_qsc(*as_causal(handle)).size();
        causal_last_error.clear();
        return value;
    } catch (const std::exception& error) {
        causal_last_error = error.what();
        return 0U;
    }
}

int qcausal_qsc_write(
    qcausal_handle handle,
    uint8_t* output,
    size_t output_size) {
    return guarded([&] {
        const auto& encoded = encoded_qsc(*as_causal(handle));
        if (output == nullptr || output_size < encoded.size()) {
            throw qubit::QStateError("Causal QSC output buffer is too small");
        }
        std::memcpy(output, encoded.data(), encoded.size());
    });
}

qcausal_parameterized_plan_handle qcausal_parameterized_plan_create(
    const qstate_parameterized_operation* operations,
    size_t operation_count,
    uint32_t flags) {
    try {
        if (operation_count != 0U && operations == nullptr) {
            throw qubit::QStateError("Causal plan operation buffer is null");
        }
        if ((flags & ~static_cast<std::uint32_t>(QSTATE_PLAN_OPTIMIZE)) != 0U) {
            throw qubit::QStateError("Causal plan flags contain an unknown bit");
        }
        std::vector<qubit::ParameterizedOperation> converted;
        converted.reserve(operation_count);
        for (std::size_t index = 0; index < operation_count; ++index) {
            converted.push_back(convert_parameterized_operation(operations[index]));
        }
        auto* plan = new CausalParameterizedPlanHandle(
            converted,
            (flags & static_cast<std::uint32_t>(QSTATE_PLAN_OPTIMIZE)) != 0U);
        causal_last_error.clear();
        return plan;
    } catch (const std::exception& error) {
        causal_last_error = error.what();
        return nullptr;
    } catch (...) {
        causal_last_error = "Unknown QSA causal runtime error";
        return nullptr;
    }
}

void qcausal_parameterized_plan_destroy(
    qcausal_parameterized_plan_handle plan) {
    delete static_cast<CausalParameterizedPlanHandle*>(plan);
}

size_t qcausal_parameterized_plan_parameter_count(
    qcausal_parameterized_plan_handle plan) {
    try {
        const std::size_t value = as_parameterized_plan(plan)->plan.parameter_count();
        causal_last_error.clear();
        return value;
    } catch (const std::exception& error) {
        causal_last_error = error.what();
        return 0U;
    }
}

int qcausal_parameterized_plan_execute(
    qcausal_handle handle,
    qcausal_parameterized_plan_handle plan,
    const double* parameters,
    size_t parameter_count,
    size_t* completed_operation_count) {
    return guarded([&] {
        if (parameter_count != 0U && parameters == nullptr) {
            throw qubit::QStateError("Causal parameter buffer is null");
        }
        CausalHandle* causal = as_causal(handle);
        invalidate_qsc(*causal);
        as_parameterized_plan(plan)->plan.execute(
            causal->state.write(),
            std::span<const double>(parameters, parameter_count),
            completed_operation_count);
    });
}

qcausal_pauli_plan_handle qcausal_pauli_plan_create(
    size_t qubit_count,
    const char* words,
    size_t word_count,
    double imaginary_tolerance) {
    try {
        std::vector<std::string> decoded =
            decode_pauli_words(qubit_count, words, word_count);
        auto* plan = new CausalPauliPlanHandle(
            qubit_count, decoded, imaginary_tolerance);
        causal_last_error.clear();
        return plan;
    } catch (const std::exception& error) {
        causal_last_error = error.what();
        return nullptr;
    } catch (...) {
        causal_last_error = "Unknown QSA causal runtime error";
        return nullptr;
    }
}

void qcausal_pauli_plan_destroy(qcausal_pauli_plan_handle plan) {
    delete static_cast<CausalPauliPlanHandle*>(plan);
}

size_t qcausal_pauli_plan_observable_count(qcausal_pauli_plan_handle plan) {
    try {
        const std::size_t value = as_pauli_plan(plan)->plan.observable_count();
        causal_last_error.clear();
        return value;
    } catch (const std::exception& error) {
        causal_last_error = error.what();
        return 0U;
    }
}

int qcausal_pauli_plan_execute(
    qcausal_pauli_plan_handle plan,
    qcausal_handle handle,
    double* output,
    size_t output_size) {
    return guarded([&] {
        CausalPauliPlanHandle* pauli = as_pauli_plan(plan);
        if (output == nullptr && output_size != 0U) {
            throw qubit::QStateError("Causal Pauli output is null");
        }
        const std::vector<double> values =
            pauli->plan.execute(as_causal(handle)->state);
        if (output_size < values.size()) {
            throw qubit::QStateError("Causal Pauli output buffer is too small");
        }
        std::copy(values.begin(), values.end(), output);
    });
}

}  // extern "C"

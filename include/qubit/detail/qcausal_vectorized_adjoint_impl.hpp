#pragma once

#include "qubit/qadjoint_vectorized_batch.hpp"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <span>
#include <string>
#include <thread>
#include <utility>

namespace {

struct VectorizedBatchSizes {
    std::size_t parameters{0U};
    std::size_t cotangents{0U};
    std::size_t values{0U};
    std::size_t gradients{0U};
};

VectorizedBatchSizes validate_vectorized_batch_sizes(
    std::size_t row_count,
    std::size_t parameter_count,
    std::size_t observable_count,
    bool include_cotangents) {
    const std::size_t limit = std::numeric_limits<std::size_t>::max();
    if ((parameter_count != 0U && row_count > limit / parameter_count) ||
        (observable_count != 0U && row_count > limit / observable_count)) {
        throw qubit::QStateError("Vectorized batch size overflow");
    }
    return VectorizedBatchSizes{
        row_count * parameter_count,
        include_cotangents ? row_count * observable_count : 0U,
        row_count * observable_count,
        include_cotangents ? row_count * parameter_count : 0U,
    };
}

constexpr std::size_t kVectorizedParallelThreshold = 4096U;
constexpr std::size_t kVectorizedRowsPerWorker = 2048U;
constexpr std::size_t kVectorizedMaximumWorkers = 32U;

std::size_t requested_vectorized_workers() {
    const char* raw = std::getenv("QSA_CAUSAL_VECTORIZED_WORKERS");
    if (raw == nullptr || *raw == '\0') {
        return 0U;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0' || parsed == 0ULL ||
        parsed > 256ULL) {
        throw qubit::QStateError(
            "QSA_CAUSAL_VECTORIZED_WORKERS must be an integer from 1 to 256");
    }
    return static_cast<std::size_t>(parsed);
}

std::size_t vectorized_worker_count(std::size_t row_count) {
    if (row_count < kVectorizedParallelThreshold) {
        return 1U;
    }
    const std::size_t maximum_by_rows = std::max<std::size_t>(
        1U,
        row_count / kVectorizedRowsPerWorker);
    const std::size_t requested = requested_vectorized_workers();
    if (requested == 1U) {
        return 1U;
    }
    std::size_t workers = requested;
    if (workers == 0U) {
        workers = static_cast<std::size_t>(std::thread::hardware_concurrency());
        if (workers == 0U) {
            workers = 1U;
        }
    }
    workers = std::min(workers, kVectorizedMaximumWorkers);
    workers = std::min(workers, maximum_by_rows);
    workers = std::min(workers, row_count);
    return std::max<std::size_t>(1U, workers);
}

std::pair<std::size_t, std::size_t> vectorized_chunk_bounds(
    std::size_t chunk,
    std::size_t chunk_count,
    std::size_t row_count) {
    const std::size_t quotient = row_count / chunk_count;
    const std::size_t remainder = row_count % chunk_count;
    const std::size_t begin =
        chunk * quotient + std::min(chunk, remainder);
    const std::size_t count = quotient + (chunk < remainder ? 1U : 0U);
    return {begin, count};
}

template <class Function>
void run_vectorized_chunks(
    std::size_t row_count,
    std::size_t worker_count,
    Function&& function) {
    if (row_count == 0U || worker_count <= 1U) {
        function(0U, row_count);
        return;
    }
    std::size_t completed_chunks = 0U;
    const int status = run_causal_parallel(
        worker_count,
        worker_count,
        &completed_chunks,
        [&](std::size_t chunk) {
            const auto [begin, count] = vectorized_chunk_bounds(
                chunk,
                worker_count,
                row_count);
            function(begin, count);
        });
    if (status != 0 || completed_chunks != worker_count) {
        const std::string message = causal_last_error.empty()
            ? "Parallel vectorized QSA batch failed"
            : causal_last_error;
        throw qubit::QStateError(message);
    }
}

template <class T>
T* offset_pointer(T* pointer, std::size_t offset) {
    return pointer == nullptr ? nullptr : pointer + offset;
}

template <class T>
const T* offset_pointer(const T* pointer, std::size_t offset) {
    return pointer == nullptr ? nullptr : pointer + offset;
}

}  // namespace

extern "C" {

int qcausal_observables_many_vectorized(
    qcausal_handle initial,
    qcausal_parameterized_plan_handle plan,
    qcausal_pauli_support_plan_handle observables,
    const double* parameter_rows,
    size_t row_count,
    size_t parameter_count,
    size_t max_qubits,
    double* values_output,
    size_t values_output_size,
    size_t* completed_row_count) {
    return guarded_causal([&] {
        if (completed_row_count == nullptr) {
            throw qubit::QStateError(
                "Vectorized primal completed-row output is null");
        }
        *completed_row_count = 0U;
        CausalHandle* causal = as_causal(initial);
        CausalParameterizedPlanHandle* parameterized =
            as_causal_parameterized_plan(plan);
        CausalPauliSupportPlanHandle* support =
            as_causal_pauli_support_plan(observables);
        if (parameter_count != parameterized->plan.parameter_count()) {
            throw qubit::QStateError(
                "Vectorized primal parameter width differs from plan");
        }
        const std::size_t observable_count = support->plan.observable_count();
        const VectorizedBatchSizes sizes = validate_vectorized_batch_sizes(
            row_count,
            parameter_count,
            observable_count,
            false);
        if (sizes.parameters != 0U && parameter_rows == nullptr) {
            throw qubit::QStateError(
                "Vectorized primal parameter buffer is null");
        }
        if (values_output_size < sizes.values ||
            (sizes.values != 0U && values_output == nullptr)) {
            throw qubit::QStateError(
                "Vectorized primal value output buffer is too small");
        }

        const qubit::QRegister& root = causal->state.read();
        const std::size_t workers = vectorized_worker_count(row_count);
        run_vectorized_chunks(
            row_count,
            workers,
            [&](std::size_t begin, std::size_t count) {
                const std::size_t parameter_offset = begin * parameter_count;
                const std::size_t value_offset = begin * observable_count;
                qubit::observables_many_vectorized(
                    root,
                    parameterized->plan,
                    std::span<const double>(
                        offset_pointer(parameter_rows, parameter_offset),
                        count * parameter_count),
                    count,
                    support->plan,
                    max_qubits,
                    support->imaginary_tolerance,
                    std::span<double>(
                        offset_pointer(values_output, value_offset),
                        count * observable_count));
            });
        *completed_row_count = row_count;
    });
}

int qcausal_weighted_adjoint_many_vectorized(
    qcausal_handle initial,
    qcausal_parameterized_plan_handle plan,
    qcausal_pauli_support_plan_handle observables,
    const double* parameter_rows,
    size_t row_count,
    size_t parameter_count,
    const double* cotangent_rows,
    size_t cotangent_count,
    size_t max_qubits,
    double* values_output,
    size_t values_output_size,
    double* gradient_output,
    size_t gradient_output_size,
    size_t* completed_row_count) {
    return guarded_causal([&] {
        if (completed_row_count == nullptr) {
            throw qubit::QStateError(
                "Vectorized adjoint completed-row output is null");
        }
        *completed_row_count = 0U;
        CausalHandle* causal = as_causal(initial);
        CausalParameterizedPlanHandle* parameterized =
            as_causal_parameterized_plan(plan);
        CausalPauliSupportPlanHandle* support =
            as_causal_pauli_support_plan(observables);
        if (parameter_count != parameterized->plan.parameter_count()) {
            throw qubit::QStateError(
                "Vectorized adjoint parameter width differs from plan");
        }
        if (cotangent_count != support->plan.observable_count()) {
            throw qubit::QStateError(
                "Vectorized adjoint cotangent width differs from observables");
        }
        const VectorizedBatchSizes sizes = validate_vectorized_batch_sizes(
            row_count,
            parameter_count,
            cotangent_count,
            true);
        if (sizes.parameters != 0U && parameter_rows == nullptr) {
            throw qubit::QStateError(
                "Vectorized adjoint parameter buffer is null");
        }
        if (sizes.cotangents != 0U && cotangent_rows == nullptr) {
            throw qubit::QStateError(
                "Vectorized adjoint cotangent buffer is null");
        }
        if (values_output_size < sizes.values ||
            (sizes.values != 0U && values_output == nullptr)) {
            throw qubit::QStateError(
                "Vectorized adjoint value output buffer is too small");
        }
        if (gradient_output_size < sizes.gradients ||
            (sizes.gradients != 0U && gradient_output == nullptr)) {
            throw qubit::QStateError(
                "Vectorized adjoint gradient output buffer is too small");
        }

        const qubit::QRegister& root = causal->state.read();
        const std::size_t workers = vectorized_worker_count(row_count);
        run_vectorized_chunks(
            row_count,
            workers,
            [&](std::size_t begin, std::size_t count) {
                const std::size_t parameter_offset = begin * parameter_count;
                const std::size_t cotangent_offset = begin * cotangent_count;
                const std::size_t value_offset = begin * cotangent_count;
                qubit::weighted_adjoint_many_vectorized(
                    root,
                    parameterized->plan,
                    std::span<const double>(
                        offset_pointer(parameter_rows, parameter_offset),
                        count * parameter_count),
                    count,
                    support->plan,
                    std::span<const double>(
                        offset_pointer(cotangent_rows, cotangent_offset),
                        count * cotangent_count),
                    max_qubits,
                    support->imaginary_tolerance,
                    std::span<double>(
                        offset_pointer(values_output, value_offset),
                        count * cotangent_count),
                    std::span<double>(
                        offset_pointer(gradient_output, parameter_offset),
                        count * parameter_count));
            });
        *completed_row_count = row_count;
    });
}

}  // extern "C"

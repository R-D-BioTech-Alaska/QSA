#pragma once

#include "qubit/qpauli_support.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace {

struct CausalPauliSupportPlanHandle {
    CausalPauliSupportPlanHandle(
        std::size_t qubit_count,
        std::span<const std::vector<qubit::PauliSupportTerm>> observables,
        double imaginary_tolerance)
        : plan(
              qubit_count,
              observables,
              qubit::PauliSupportConfig{imaginary_tolerance}) {}

    qubit::PauliSupportPlan plan;
};

CausalPauliSupportPlanHandle* as_causal_pauli_support_plan(
    qcausal_pauli_support_plan_handle handle) {
    if (handle == nullptr) {
        throw qubit::QStateError("QSA causal Pauli support plan handle is null");
    }
    return static_cast<CausalPauliSupportPlanHandle*>(handle);
}

std::vector<std::vector<qubit::PauliSupportTerm>> decode_causal_pauli_supports(
    const qcausal_pauli_support_term* terms,
    std::size_t term_count,
    const std::size_t* offsets,
    std::size_t observable_count) {
    if (term_count != 0U && terms == nullptr) {
        throw qubit::QStateError("Pauli support term buffer is null");
    }
    if (observable_count != 0U && offsets == nullptr) {
        throw qubit::QStateError("Pauli support offset buffer is null");
    }
    if (observable_count == 0U) {
        if (term_count != 0U) {
            throw qubit::QStateError("Pauli support terms have no observable");
        }
        return {};
    }
    if (offsets[0] != 0U || offsets[observable_count] != term_count) {
        throw qubit::QStateError("Pauli support offsets do not cover the term buffer");
    }

    std::vector<std::vector<qubit::PauliSupportTerm>> observables;
    observables.reserve(observable_count);
    for (std::size_t observable = 0; observable < observable_count; ++observable) {
        const std::size_t begin = offsets[observable];
        const std::size_t end = offsets[observable + 1U];
        if (begin > end || end > term_count) {
            throw qubit::QStateError("Pauli support offsets are not ordered");
        }
        std::vector<qubit::PauliSupportTerm> decoded;
        decoded.reserve(end - begin);
        for (std::size_t index = begin; index < end; ++index) {
            decoded.push_back(qubit::PauliSupportTerm{
                terms[index].qubit,
                static_cast<char>(terms[index].axis),
            });
        }
        observables.push_back(std::move(decoded));
    }
    return observables;
}

}  // namespace

extern "C" {

qcausal_pauli_support_plan_handle qcausal_pauli_support_plan_create(
    size_t qubit_count,
    const qcausal_pauli_support_term* terms,
    size_t term_count,
    const size_t* observable_offsets,
    size_t observable_count,
    double imaginary_tolerance) {
    try {
        auto observables = decode_causal_pauli_supports(
            terms,
            term_count,
            observable_offsets,
            observable_count);
        auto* plan = new CausalPauliSupportPlanHandle(
            qubit_count,
            observables,
            imaginary_tolerance);
        causal_last_error.clear();
        return plan;
    } catch (const std::exception& error) {
        causal_last_error = error.what();
        return nullptr;
    } catch (...) {
        causal_last_error = "Unknown QSA causal Pauli support error";
        return nullptr;
    }
}

void qcausal_pauli_support_plan_destroy(
    qcausal_pauli_support_plan_handle plan) {
    delete static_cast<CausalPauliSupportPlanHandle*>(plan);
}

size_t qcausal_pauli_support_plan_observable_count(
    qcausal_pauli_support_plan_handle plan) {
    try {
        const std::size_t value =
            as_causal_pauli_support_plan(plan)->plan.observable_count();
        causal_last_error.clear();
        return value;
    } catch (const std::exception& error) {
        causal_last_error = error.what();
        return 0U;
    }
}

size_t qcausal_pauli_support_plan_term_count(
    qcausal_pauli_support_plan_handle plan) {
    try {
        const std::size_t value =
            as_causal_pauli_support_plan(plan)->plan.term_count();
        causal_last_error.clear();
        return value;
    } catch (const std::exception& error) {
        causal_last_error = error.what();
        return 0U;
    }
}

int qcausal_pauli_support_plan_execute(
    qcausal_pauli_support_plan_handle plan,
    qcausal_handle handle,
    double* output,
    size_t output_size) {
    return guarded_causal([&] {
        CausalPauliSupportPlanHandle* support =
            as_causal_pauli_support_plan(plan);
        if (output == nullptr && output_size != 0U) {
            throw qubit::QStateError("Causal Pauli support output is null");
        }
        const std::vector<double> values =
            support->plan.execute(as_causal(handle)->state.read());
        if (output_size < values.size()) {
            throw qubit::QStateError("Causal Pauli support output is too small");
        }
        std::copy(values.begin(), values.end(), output);
    });
}

int qcausal_pauli_support_plan_execute_many(
    qcausal_pauli_support_plan_handle plan,
    qcausal_handle* handles,
    size_t handle_count,
    double* output,
    size_t output_size,
    size_t worker_count,
    size_t* completed_handle_count) {
    try {
        CausalPauliSupportPlanHandle* support =
            as_causal_pauli_support_plan(plan);
        if (handle_count != 0U && handles == nullptr) {
            throw qubit::QStateError("Causal Pauli support handle buffer is null");
        }
        const std::size_t observable_count = support->plan.observable_count();
        if (observable_count != 0U &&
            handle_count > std::numeric_limits<std::size_t>::max() / observable_count) {
            throw qubit::QStateError("Causal Pauli support output is too large");
        }
        const std::size_t required = handle_count * observable_count;
        if (output_size < required || (required != 0U && output == nullptr)) {
            throw qubit::QStateError("Causal Pauli support output is too small");
        }

        std::size_t selected_workers = worker_count;
        const std::size_t term_count = support->plan.term_count();
        if (selected_workers == 0U &&
            (term_count == 0U ||
             handle_count <= std::numeric_limits<std::size_t>::max() / term_count) &&
            handle_count * term_count < 4'096U) {
            selected_workers = 1U;
        }

        return run_causal_parallel(
            handle_count,
            selected_workers,
            completed_handle_count,
            [&](std::size_t index) {
                const std::vector<double> values = support->plan.execute(
                    as_causal(handles[index])->state.read());
                std::copy(
                    values.begin(),
                    values.end(),
                    output + index * observable_count);
            });
    } catch (const std::exception& error) {
        if (completed_handle_count != nullptr) {
            *completed_handle_count = 0U;
        }
        causal_last_error = error.what();
        return -1;
    } catch (...) {
        if (completed_handle_count != nullptr) {
            *completed_handle_count = 0U;
        }
        causal_last_error = "Unknown QSA causal Pauli support error";
        return -1;
    }
}

}  // extern "C"

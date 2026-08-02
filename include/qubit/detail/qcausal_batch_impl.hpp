#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

template <class Function>
int run_causal_inline(
    std::size_t item_count,
    std::size_t* completed_count,
    Function&& function) {
    std::size_t completed = 0U;
    try {
        for (; completed < item_count; ++completed) {
            function(completed);
        }
        if (completed_count != nullptr) {
            *completed_count = completed;
        }
        causal_last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        if (completed_count != nullptr) {
            *completed_count = completed;
        }
        causal_last_error = error.what();
        return -1;
    } catch (...) {
        if (completed_count != nullptr) {
            *completed_count = completed;
        }
        causal_last_error = "Unknown QSA causal batch error";
        return -1;
    }
}

template <class Function>
int run_causal_parallel(
    std::size_t item_count,
    std::size_t worker_count,
    std::size_t* completed_count,
    Function&& function) {
    if (completed_count != nullptr) {
        *completed_count = 0U;
    }
    if (item_count == 0U) {
        causal_last_error.clear();
        return 0;
    }

    constexpr std::size_t minimum_parallel_items = 64U;
    std::size_t workers = worker_count;
    if (workers == 0U) {
        workers = item_count < minimum_parallel_items
            ? 1U
            : static_cast<std::size_t>(std::thread::hardware_concurrency());
    }
    workers = std::max<std::size_t>(1U, std::min(workers, item_count));
    if (workers == 1U) {
        return run_causal_inline(
            item_count,
            completed_count,
            std::forward<Function>(function));
    }

    std::atomic<std::size_t> next{0U};
    std::atomic<std::size_t> completed{0U};
    std::atomic<bool> failed{false};
    std::mutex error_mutex;
    std::string first_error;

    auto worker = [&] {
        while (!failed.load(std::memory_order_acquire)) {
            const std::size_t index = next.fetch_add(1U, std::memory_order_relaxed);
            if (index >= item_count) {
                return;
            }
            try {
                function(index);
                completed.fetch_add(1U, std::memory_order_relaxed);
            } catch (const std::exception& error) {
                if (!failed.exchange(true, std::memory_order_acq_rel)) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    first_error = error.what();
                }
                return;
            } catch (...) {
                if (!failed.exchange(true, std::memory_order_acq_rel)) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    first_error = "Unknown QSA causal batch error";
                }
                return;
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (std::size_t index = 0; index < workers; ++index) {
        threads.emplace_back(worker);
    }
    for (std::thread& thread : threads) {
        thread.join();
    }

    const std::size_t completed_value = completed.load(std::memory_order_relaxed);
    if (completed_count != nullptr) {
        *completed_count = completed_value;
    }
    if (failed.load(std::memory_order_acquire)) {
        causal_last_error = first_error.empty()
            ? "QSA causal batch failed"
            : first_error;
        return -1;
    }
    causal_last_error.clear();
    return 0;
}

}  // namespace

extern "C" {

int qcausal_fork_many(
    qcausal_handle source,
    qcausal_handle* output,
    size_t branch_count) {
    try {
        CausalHandle* root = as_causal(source);
        if (branch_count != 0U && output == nullptr) {
            throw qubit::QStateError("Causal fork output is null");
        }
        for (std::size_t index = 0; index < branch_count; ++index) {
            output[index] = nullptr;
        }
        std::size_t created = 0U;
        try {
            for (; created < branch_count; ++created) {
                output[created] = new CausalHandle(root->state.fork());
            }
        } catch (...) {
            for (std::size_t index = 0; index < created; ++index) {
                delete static_cast<CausalHandle*>(output[index]);
                output[index] = nullptr;
            }
            throw;
        }
        causal_last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        causal_last_error = error.what();
        return -1;
    } catch (...) {
        causal_last_error = "Unknown QSA causal fork error";
        return -1;
    }
}

int qcausal_parameterized_plan_execute_many(
    qcausal_parameterized_plan_handle plan,
    qcausal_handle* handles,
    size_t handle_count,
    const double* parameters,
    size_t parameter_count,
    size_t worker_count,
    size_t* completed_handle_count) {
    try {
        CausalParameterizedPlanHandle* native_plan =
            as_causal_parameterized_plan(plan);
        if (handle_count != 0U && handles == nullptr) {
            throw qubit::QStateError("Causal batch handle buffer is null");
        }
        if (parameter_count != native_plan->plan.parameter_count()) {
            throw qubit::QStateError("Causal batch parameter width is wrong");
        }
        if (parameter_count != 0U &&
            handle_count > std::numeric_limits<std::size_t>::max() / parameter_count) {
            throw qubit::QStateError("Causal batch parameter matrix is too large");
        }
        const std::size_t parameter_total = handle_count * parameter_count;
        if (parameter_total != 0U && parameters == nullptr) {
            throw qubit::QStateError("Causal batch parameter matrix is null");
        }
        return run_causal_parallel(
            handle_count,
            worker_count,
            completed_handle_count,
            [&](std::size_t index) {
                CausalHandle* causal = as_causal(handles[index]);
                invalidate_causal_qsc(*causal);
                const double* row = parameter_count == 0U
                    ? parameters
                    : parameters + index * parameter_count;
                native_plan->plan.execute(
                    causal->state.write(),
                    std::span<const double>(row, parameter_count),
                    nullptr);
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
        causal_last_error = "Unknown QSA causal batch error";
        return -1;
    }
}

int qcausal_pauli_plan_execute_many(
    qcausal_pauli_plan_handle plan,
    qcausal_handle* handles,
    size_t handle_count,
    double* output,
    size_t output_size,
    size_t worker_count,
    size_t* completed_handle_count) {
    try {
        CausalPauliPlanHandle* pauli = as_causal_pauli_plan(plan);
        if (handle_count != 0U && handles == nullptr) {
            throw qubit::QStateError("Causal Pauli batch handle buffer is null");
        }
        const std::size_t observable_count = pauli->plan.observable_count();
        if (observable_count != 0U &&
            handle_count > std::numeric_limits<std::size_t>::max() / observable_count) {
            throw qubit::QStateError("Causal Pauli output matrix is too large");
        }
        const std::size_t required = handle_count * observable_count;
        if (output_size < required || (required != 0U && output == nullptr)) {
            throw qubit::QStateError("Causal Pauli batch output is too small");
        }
        return run_causal_parallel(
            handle_count,
            worker_count,
            completed_handle_count,
            [&](std::size_t index) {
                const std::vector<double> values =
                    pauli->plan.execute(as_causal(handles[index])->state);
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
        causal_last_error = "Unknown QSA causal Pauli batch error";
        return -1;
    }
}

}  // extern "C"

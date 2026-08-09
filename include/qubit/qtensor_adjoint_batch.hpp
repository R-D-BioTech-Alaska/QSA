#pragma once

#include "qubit/qtensor_adjoint.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <span>
#include <thread>
#include <vector>

namespace qubit {

struct ExactAdjointGradientBatchConfig {
    TensorNetworkConfig tensor{};
    std::size_t point_worker_count{1U};
};

struct ExactAdjointGradientBatchStats {
    std::size_t parameter_count{0U};
    std::size_t observable_count{0U};
    std::size_t point_worker_count{1U};
    ExactAdjointGradientStats point{};
};

class ExactAdjointGradientBatchPlan;

class ExactAdjointGradientBatchWorkspace {
public:
    ExactAdjointGradientBatchWorkspace() = default;

    [[nodiscard]] std::size_t rebind_count() const noexcept {
        std::size_t count = 0U;
        for (const ExactAdjointGradientWorkspace& workspace : point_workspaces_) {
            count += workspace.rebind_count();
        }
        return count;
    }

    [[nodiscard]] std::size_t point_capacity() const noexcept {
        return staged_values_.empty() || observable_count_ == 0U
            ? 0U
            : staged_values_.size() / observable_count_;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) +
                            point_workspaces_.capacity() *
                                sizeof(ExactAdjointGradientWorkspace) +
                            staged_values_.capacity() * sizeof(QComplex) +
                            staged_gradients_.capacity() * sizeof(QComplex);
        for (const ExactAdjointGradientWorkspace& workspace : point_workspaces_) {
            bytes += workspace.estimated_bytes();
        }
        return bytes;
    }

private:
    std::uint64_t token_{0U};
    std::size_t observable_count_{0U};
    std::size_t parameter_count_{0U};
    std::vector<ExactAdjointGradientWorkspace> point_workspaces_{};
    std::vector<QComplex> staged_values_{};
    std::vector<QComplex> staged_gradients_{};

    friend class ExactAdjointGradientBatchPlan;
};

class ExactAdjointGradientBatchPlan {
public:
    ExactAdjointGradientBatchPlan(
        std::size_t qubit_count,
        std::span<const ParameterizedOperation> operations,
        std::span<const PauliObservable> observables,
        ExactAdjointGradientBatchConfig config = {})
        : point_plan_(
              qubit_count,
              operations,
              observables,
              ExactAdjointGradientConfig{config.tensor, 1U}),
          config_(config),
          workspace_token_(next_workspace_token()) {
        if (config_.point_worker_count > kMaxPointWorkers) {
            throw QStateError(
                "Exact adjoint gradient batch point_worker_count exceeds the supported limit");
        }

        std::size_t requested = config_.point_worker_count;
        if (requested == 0U) {
            const unsigned int hardware = std::thread::hardware_concurrency();
            requested = hardware == 0U ? 1U : static_cast<std::size_t>(hardware);
            requested = std::min(requested, kAutoPointWorkers);
        }
        point_worker_count_ = std::max<std::size_t>(1U, requested);
    }

    [[nodiscard]] ExactAdjointGradientBatchWorkspace workspace() const {
        ExactAdjointGradientBatchWorkspace result;
        result.token_ = workspace_token_;
        result.observable_count_ = observable_count();
        result.parameter_count_ = parameter_count();
        result.point_workspaces_.reserve(point_worker_count_);
        for (std::size_t lane = 0U; lane < point_worker_count_; ++lane) {
            result.point_workspaces_.push_back(point_plan_.workspace());
        }
        return result;
    }

    void value_and_gradient_batch(
        std::span<const double> parameters,
        std::size_t point_count,
        std::span<QComplex> values,
        std::span<QComplex> gradients) const {
        ExactAdjointGradientBatchWorkspace local = workspace();
        value_and_gradient_batch(
            parameters, point_count, values, gradients, local);
    }

    void value_and_gradient_batch(
        std::span<const double> parameters,
        std::size_t point_count,
        std::span<QComplex> values,
        std::span<QComplex> gradients,
        ExactAdjointGradientBatchWorkspace& workspace_value) const {
        validate_workspace(workspace_value);
        const std::size_t parameter_entries = checked_multiply(
            point_count, parameter_count(),
            "Exact adjoint gradient batch parameter shape overflowed");
        const std::size_t value_entries = checked_multiply(
            point_count, observable_count(),
            "Exact adjoint gradient batch value shape overflowed");
        const std::size_t point_gradient_entries = checked_multiply(
            observable_count(), parameter_count(),
            "Exact adjoint gradient batch point gradient shape overflowed");
        const std::size_t gradient_entries = checked_multiply(
            point_count, point_gradient_entries,
            "Exact adjoint gradient batch gradient shape overflowed");

        if (parameters.size() != parameter_entries) {
            throw QStateError("Exact adjoint gradient batch parameter count is invalid");
        }
        if (values.size() != value_entries) {
            throw QStateError("Exact adjoint gradient batch value result count is invalid");
        }
        if (gradients.size() != gradient_entries) {
            throw QStateError("Exact adjoint gradient batch gradient result count is invalid");
        }
        for (const double parameter : parameters) {
            if (!std::isfinite(parameter)) {
                throw QStateError("Exact adjoint gradient batch parameters must be finite");
            }
        }
        if (point_count == 0U) {
            return;
        }

        workspace_value.staged_values_.resize(value_entries);
        workspace_value.staged_gradients_.resize(gradient_entries);

        const std::size_t active_workers =
            std::min(point_worker_count_, point_count);
        std::vector<std::exception_ptr> errors(active_workers);
        const auto worker = [&](std::size_t lane) {
            try {
                for (std::size_t point = lane;
                     point < point_count;
                     point += active_workers) {
                    const std::size_t parameter_offset =
                        point * parameter_count();
                    const std::size_t value_offset =
                        point * observable_count();
                    const std::size_t gradient_offset =
                        point * point_gradient_entries;

                    point_plan_.value_and_gradient(
                        std::span<const double>(
                            parameters.data() + parameter_offset,
                            parameter_count()),
                        std::span<QComplex>(
                            workspace_value.staged_values_.data() + value_offset,
                            observable_count()),
                        std::span<QComplex>(
                            workspace_value.staged_gradients_.data() + gradient_offset,
                            point_gradient_entries),
                        workspace_value.point_workspaces_[lane]);
                }
            } catch (...) {
                errors[lane] = std::current_exception();
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(active_workers > 0U ? active_workers - 1U : 0U);
        for (std::size_t lane = 1U; lane < active_workers; ++lane) {
            threads.emplace_back(worker, lane);
        }
        worker(0U);
        for (std::thread& thread : threads) {
            thread.join();
        }
        for (const std::exception_ptr& error : errors) {
            if (error != nullptr) {
                std::rethrow_exception(error);
            }
        }

        std::copy(
            workspace_value.staged_values_.begin(),
            workspace_value.staged_values_.end(),
            values.begin());
        std::copy(
            workspace_value.staged_gradients_.begin(),
            workspace_value.staged_gradients_.end(),
            gradients.begin());
    }

    [[nodiscard]] std::size_t parameter_count() const noexcept {
        return point_plan_.parameter_count();
    }

    [[nodiscard]] std::size_t observable_count() const noexcept {
        return point_plan_.observable_count();
    }

    [[nodiscard]] std::size_t point_worker_count() const noexcept {
        return point_worker_count_;
    }

    [[nodiscard]] std::size_t effective_point_worker_count(
        std::size_t point_count) const noexcept {
        return point_count == 0U
            ? 0U
            : std::min(point_worker_count_, point_count);
    }

    [[nodiscard]] ExactAdjointGradientBatchStats stats() const noexcept {
        ExactAdjointGradientBatchStats result;
        result.parameter_count = parameter_count();
        result.observable_count = observable_count();
        result.point_worker_count = point_worker_count_;
        result.point = point_plan_.stats();
        return result;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        return sizeof(*this) + point_plan_.estimated_bytes();
    }

private:
    static constexpr std::size_t kMaxPointWorkers = 16U;
    static constexpr std::size_t kAutoPointWorkers = 8U;

    ExactAdjointGradientPlan point_plan_;
    ExactAdjointGradientBatchConfig config_{};
    std::size_t point_worker_count_{1U};
    std::uint64_t workspace_token_{0U};

    [[nodiscard]] static std::uint64_t next_workspace_token() noexcept {
        static std::atomic<std::uint64_t> next{1U};
        return next.fetch_add(1U, std::memory_order_relaxed);
    }

    [[nodiscard]] static std::size_t checked_multiply(
        std::size_t first,
        std::size_t second,
        const char* message) {
        if (first != 0U && second > std::numeric_limits<std::size_t>::max() / first) {
            throw QStateError(message);
        }
        return first * second;
    }

    void validate_workspace(
        const ExactAdjointGradientBatchWorkspace& workspace_value) const {
        if (workspace_value.token_ != workspace_token_ ||
            workspace_value.observable_count_ != observable_count() ||
            workspace_value.parameter_count_ != parameter_count() ||
            workspace_value.point_workspaces_.size() != point_worker_count_) {
            throw QStateError(
                "Exact adjoint gradient batch workspace does not match its plan");
        }
    }
};

}  // namespace qubit

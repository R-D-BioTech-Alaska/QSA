#pragma once

#include "qubit/qtensor_adjoint.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <span>
#include <thread>
#include <vector>

namespace qubit {

enum class ExactAdjointScheduleRoute : std::uint8_t {
    Serial = 0,
    TermParallel = 1,
    PointParallel = 2,
};

struct ExactAdjointGradientSchedulerConfig {
    TensorNetworkConfig tensor{};
    std::size_t total_worker_count{0U};
    std::size_t max_workspace_bytes{0U};
};

struct ExactAdjointGradientSchedule {
    ExactAdjointScheduleRoute route{ExactAdjointScheduleRoute::Serial};
    std::size_t point_count{0U};
    std::size_t worker_count{0U};
    std::uint64_t estimated_critical_work{0U};
    std::size_t estimated_workspace_bytes{0U};

    std::uint64_t serial_estimated_critical_work{0U};
    std::uint64_t term_estimated_critical_work{
        std::numeric_limits<std::uint64_t>::max()};
    std::uint64_t point_estimated_critical_work{
        std::numeric_limits<std::uint64_t>::max()};

    std::size_t serial_estimated_workspace_bytes{0U};
    std::size_t term_estimated_workspace_bytes{
        std::numeric_limits<std::size_t>::max()};
    std::size_t point_estimated_workspace_bytes{
        std::numeric_limits<std::size_t>::max()};

    bool serial_eligible{false};
    bool term_eligible{false};
    bool point_eligible{false};
};

struct ExactAdjointGradientSchedulerStats {
    std::size_t resolved_worker_count{1U};
    std::size_t max_workspace_bytes{0U};
    std::size_t parameter_count{0U};
    std::size_t observable_count{0U};
    std::size_t differentiated_term_count{0U};
    std::uint64_t serial_estimated_work{0U};
    std::uint64_t term_balanced_peak_estimated_work{0U};
    std::size_t serial_workspace_bytes{0U};
    std::size_t term_workspace_bytes{0U};
};

class ExactAdjointGradientSchedulerPlan;

class ExactAdjointGradientSchedulerWorkspace {
public:
    ExactAdjointGradientSchedulerWorkspace() = default;

    [[nodiscard]] std::size_t successful_point_count() const noexcept {
        return successful_point_count_;
    }

    [[nodiscard]] bool has_last_schedule() const noexcept {
        return last_schedule_.has_value();
    }

    [[nodiscard]] const ExactAdjointGradientSchedule& last_schedule() const {
        if (!last_schedule_.has_value()) {
            throw QStateError("Exact adjoint scheduler workspace has no completed schedule");
        }
        return *last_schedule_;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) +
                            point_workspaces_.capacity() *
                                sizeof(ExactAdjointGradientWorkspace) +
                            staged_values_.capacity() * sizeof(QComplex) +
                            staged_gradients_.capacity() * sizeof(QComplex);
        if (single_workspace_.has_value()) {
            bytes += single_workspace_->estimated_bytes();
        }
        for (const ExactAdjointGradientWorkspace& workspace : point_workspaces_) {
            bytes += workspace.estimated_bytes();
        }
        return bytes;
    }

private:
    std::uint64_t token_{0U};
    std::optional<ExactAdjointGradientWorkspace> single_workspace_{};
    std::optional<ExactAdjointScheduleRoute> single_route_{};
    std::vector<ExactAdjointGradientWorkspace> point_workspaces_{};
    std::vector<QComplex> staged_values_{};
    std::vector<QComplex> staged_gradients_{};
    std::optional<ExactAdjointGradientSchedule> last_schedule_{};
    std::size_t successful_point_count_{0U};

    friend class ExactAdjointGradientSchedulerPlan;
};

class ExactAdjointGradientSchedulerPlan {
public:
    ExactAdjointGradientSchedulerPlan(
        std::size_t qubit_count,
        std::span<const ParameterizedOperation> operations,
        std::span<const PauliObservable> observables,
        ExactAdjointGradientSchedulerConfig config = {})
        : serial_(
              qubit_count,
              operations,
              observables,
              ExactAdjointGradientConfig{config.tensor, 1U}),
          config_(config),
          workspace_token_(next_workspace_token()) {
        if (serial_.parameter_count() == 0U) {
            throw QStateError(
                "Exact adjoint scheduler requires at least one parameterized operation");
        }
        if (config_.total_worker_count > kMaxWorkers) {
            throw QStateError(
                "Exact adjoint scheduler total_worker_count exceeds the supported limit");
        }

        std::size_t requested = config_.total_worker_count;
        if (requested == 0U) {
            const unsigned int hardware = std::thread::hardware_concurrency();
            requested = hardware == 0U ? 1U : static_cast<std::size_t>(hardware);
            requested = std::min(requested, kMaxWorkers);
        }
        resolved_worker_count_ = std::max<std::size_t>(1U, requested);

        const ExactAdjointGradientStats serial_stats = serial_.stats();
        serial_estimated_work_ = serial_stats.estimated_work;
        serial_workspace_bytes_ = serial_.workspace().estimated_bytes();
        if (serial_estimated_work_ == 0U) {
            throw QStateError("Exact adjoint scheduler received an empty differentiated workload");
        }

        if (resolved_worker_count_ > 1U &&
            serial_stats.differentiated_term_count > 1U) {
            term_.emplace(
                qubit_count,
                operations,
                observables,
                ExactAdjointGradientConfig{
                    config_.tensor,
                    resolved_worker_count_,
                });
            if (term_->worker_count() > 1U) {
                const ExactAdjointGradientStats term_stats = term_->stats();
                term_peak_estimated_work_ =
                    term_stats.balanced_peak_estimated_work;
                term_workspace_bytes_ = term_->workspace().estimated_bytes();
            } else {
                term_.reset();
            }
        }
    }

    [[nodiscard]] ExactAdjointGradientSchedulerWorkspace workspace() const {
        ExactAdjointGradientSchedulerWorkspace result;
        result.token_ = workspace_token_;
        return result;
    }

    [[nodiscard]] ExactAdjointGradientSchedule choose(
        std::size_t point_count) const {
        ExactAdjointGradientSchedule schedule;
        schedule.point_count = point_count;
        if (point_count == 0U) {
            schedule.route = ExactAdjointScheduleRoute::Serial;
            schedule.serial_eligible = true;
            return schedule;
        }

        const std::size_t staging_bytes = staged_output_bytes(point_count);
        schedule.serial_estimated_critical_work = saturating_multiply(
            serial_estimated_work_, static_cast<std::uint64_t>(point_count));
        schedule.serial_estimated_workspace_bytes = saturating_add_size(
            serial_workspace_bytes_, staging_bytes);
        schedule.serial_eligible = fits_workspace(
            schedule.serial_estimated_workspace_bytes);

        if (term_.has_value()) {
            schedule.term_estimated_critical_work = saturating_multiply(
                term_peak_estimated_work_, static_cast<std::uint64_t>(point_count));
            schedule.term_estimated_workspace_bytes = saturating_add_size(
                term_workspace_bytes_, staging_bytes);
            schedule.term_eligible = fits_workspace(
                schedule.term_estimated_workspace_bytes);
        }

        const std::size_t point_workers =
            std::min(resolved_worker_count_, point_count);
        if (point_workers > 1U) {
            const std::size_t waves =
                point_count / point_workers +
                static_cast<std::size_t>(point_count % point_workers != 0U);
            schedule.point_estimated_critical_work = saturating_multiply(
                serial_estimated_work_, static_cast<std::uint64_t>(waves));
            const std::size_t point_workspace = saturating_multiply_size(
                serial_workspace_bytes_, point_workers);
            schedule.point_estimated_workspace_bytes = saturating_add_size(
                point_workspace, staging_bytes);
            schedule.point_eligible = fits_workspace(
                schedule.point_estimated_workspace_bytes);
        }

        struct Candidate {
            ExactAdjointScheduleRoute route{ExactAdjointScheduleRoute::Serial};
            std::size_t workers{1U};
            std::uint64_t work{0U};
            std::size_t bytes{0U};
        };

        std::optional<Candidate> selected;
        const auto consider = [&](Candidate candidate, bool eligible) {
            if (!eligible) {
                return;
            }
            if (!selected.has_value() ||
                candidate.work < selected->work ||
                (candidate.work == selected->work &&
                 candidate.bytes < selected->bytes) ||
                (candidate.work == selected->work &&
                 candidate.bytes == selected->bytes &&
                 static_cast<std::uint8_t>(candidate.route) <
                     static_cast<std::uint8_t>(selected->route))) {
                selected = candidate;
            }
        };

        consider(
            {ExactAdjointScheduleRoute::Serial,
             1U,
             schedule.serial_estimated_critical_work,
             schedule.serial_estimated_workspace_bytes},
            schedule.serial_eligible);
        if (term_.has_value()) {
            consider(
                {ExactAdjointScheduleRoute::TermParallel,
                 term_->worker_count(),
                 schedule.term_estimated_critical_work,
                 schedule.term_estimated_workspace_bytes},
                schedule.term_eligible);
        }
        if (point_workers > 1U) {
            consider(
                {ExactAdjointScheduleRoute::PointParallel,
                 point_workers,
                 schedule.point_estimated_critical_work,
                 schedule.point_estimated_workspace_bytes},
                schedule.point_eligible);
        }

        if (!selected.has_value()) {
            throw QStateError(
                "Exact adjoint scheduler has no route within max_workspace_bytes");
        }
        schedule.route = selected->route;
        schedule.worker_count = selected->workers;
        schedule.estimated_critical_work = selected->work;
        schedule.estimated_workspace_bytes = selected->bytes;
        return schedule;
    }

    void value_and_gradient_batch(
        std::span<const double> parameters,
        std::size_t point_count,
        std::span<QComplex> values,
        std::span<QComplex> gradients,
        ExactAdjointGradientSchedulerWorkspace& workspace_value,
        ExactAdjointGradientSchedule* completed_schedule = nullptr) const {
        validate_workspace(workspace_value);
        const std::size_t parameter_entries = checked_multiply(
            point_count,
            parameter_count(),
            "Exact adjoint scheduler parameter shape overflowed");
        const std::size_t value_entries = checked_multiply(
            point_count,
            observable_count(),
            "Exact adjoint scheduler value shape overflowed");
        const std::size_t point_gradient_entries = checked_multiply(
            observable_count(),
            parameter_count(),
            "Exact adjoint scheduler point gradient shape overflowed");
        const std::size_t gradient_entries = checked_multiply(
            point_count,
            point_gradient_entries,
            "Exact adjoint scheduler gradient shape overflowed");

        if (parameters.size() != parameter_entries) {
            throw QStateError("Exact adjoint scheduler parameter count is invalid");
        }
        if (values.size() != value_entries) {
            throw QStateError("Exact adjoint scheduler value result count is invalid");
        }
        if (gradients.size() != gradient_entries) {
            throw QStateError("Exact adjoint scheduler gradient result count is invalid");
        }
        for (const double parameter : parameters) {
            if (!std::isfinite(parameter)) {
                throw QStateError("Exact adjoint scheduler parameters must be finite");
            }
        }

        const ExactAdjointGradientSchedule schedule = choose(point_count);
        if (point_count == 0U) {
            workspace_value.last_schedule_ = schedule;
            if (completed_schedule != nullptr) {
                *completed_schedule = schedule;
            }
            return;
        }

        workspace_value.staged_values_.resize(value_entries);
        workspace_value.staged_gradients_.resize(gradient_entries);
        prepare_workspace(workspace_value, schedule);

        switch (schedule.route) {
            case ExactAdjointScheduleRoute::Serial:
                evaluate_sequential(
                    serial_,
                    *workspace_value.single_workspace_,
                    parameters,
                    point_count,
                    point_gradient_entries,
                    workspace_value);
                break;
            case ExactAdjointScheduleRoute::TermParallel:
                evaluate_sequential(
                    *term_,
                    *workspace_value.single_workspace_,
                    parameters,
                    point_count,
                    point_gradient_entries,
                    workspace_value);
                break;
            case ExactAdjointScheduleRoute::PointParallel:
                evaluate_points_parallel(
                    parameters,
                    point_count,
                    point_gradient_entries,
                    schedule.worker_count,
                    workspace_value);
                break;
        }

        std::copy(
            workspace_value.staged_values_.begin(),
            workspace_value.staged_values_.end(),
            values.begin());
        std::copy(
            workspace_value.staged_gradients_.begin(),
            workspace_value.staged_gradients_.end(),
            gradients.begin());
        workspace_value.successful_point_count_ += point_count;
        workspace_value.last_schedule_ = schedule;
        if (completed_schedule != nullptr) {
            *completed_schedule = schedule;
        }
    }

    void value_and_gradient_batch(
        std::span<const double> parameters,
        std::size_t point_count,
        std::span<QComplex> values,
        std::span<QComplex> gradients) const {
        ExactAdjointGradientSchedulerWorkspace local = workspace();
        value_and_gradient_batch(
            parameters, point_count, values, gradients, local, nullptr);
    }

    [[nodiscard]] std::size_t parameter_count() const noexcept {
        return serial_.parameter_count();
    }

    [[nodiscard]] std::size_t observable_count() const noexcept {
        return serial_.observable_count();
    }

    [[nodiscard]] std::size_t resolved_worker_count() const noexcept {
        return resolved_worker_count_;
    }

    [[nodiscard]] ExactAdjointGradientSchedulerStats stats() const noexcept {
        ExactAdjointGradientSchedulerStats result;
        const ExactAdjointGradientStats serial_stats = serial_.stats();
        result.resolved_worker_count = resolved_worker_count_;
        result.max_workspace_bytes = config_.max_workspace_bytes;
        result.parameter_count = parameter_count();
        result.observable_count = observable_count();
        result.differentiated_term_count =
            serial_stats.differentiated_term_count;
        result.serial_estimated_work = serial_estimated_work_;
        result.term_balanced_peak_estimated_work =
            term_peak_estimated_work_;
        result.serial_workspace_bytes = serial_workspace_bytes_;
        result.term_workspace_bytes = term_workspace_bytes_;
        return result;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        return sizeof(*this) +
               serial_.estimated_bytes() +
               (term_.has_value() ? term_->estimated_bytes() : 0U);
    }

private:
    static constexpr std::size_t kMaxWorkers = 16U;

    ExactAdjointGradientPlan serial_;
    std::optional<ExactAdjointGradientPlan> term_{};
    ExactAdjointGradientSchedulerConfig config_{};
    std::size_t resolved_worker_count_{1U};
    std::uint64_t serial_estimated_work_{0U};
    std::uint64_t term_peak_estimated_work_{0U};
    std::size_t serial_workspace_bytes_{0U};
    std::size_t term_workspace_bytes_{0U};
    std::uint64_t workspace_token_{0U};

    [[nodiscard]] static std::uint64_t next_workspace_token() noexcept {
        static std::atomic<std::uint64_t> next{1U};
        return next.fetch_add(1U, std::memory_order_relaxed);
    }

    [[nodiscard]] static std::uint64_t saturating_multiply(
        std::uint64_t first,
        std::uint64_t second) noexcept {
        if (first == 0U || second == 0U) {
            return 0U;
        }
        if (first > std::numeric_limits<std::uint64_t>::max() / second) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        return first * second;
    }

    [[nodiscard]] static std::size_t saturating_add_size(
        std::size_t first,
        std::size_t second) noexcept {
        if (second > std::numeric_limits<std::size_t>::max() - first) {
            return std::numeric_limits<std::size_t>::max();
        }
        return first + second;
    }

    [[nodiscard]] static std::size_t saturating_multiply_size(
        std::size_t first,
        std::size_t second) noexcept {
        if (first == 0U || second == 0U) {
            return 0U;
        }
        if (first > std::numeric_limits<std::size_t>::max() / second) {
            return std::numeric_limits<std::size_t>::max();
        }
        return first * second;
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

    [[nodiscard]] std::size_t staged_output_bytes(
        std::size_t point_count) const {
        const std::size_t values = checked_multiply(
            point_count,
            observable_count(),
            "Exact adjoint scheduler staged value shape overflowed");
        const std::size_t gradients_per_point = checked_multiply(
            observable_count(),
            parameter_count(),
            "Exact adjoint scheduler staged gradient shape overflowed");
        const std::size_t gradients = checked_multiply(
            point_count,
            gradients_per_point,
            "Exact adjoint scheduler staged gradient count overflowed");
        const std::size_t entries = saturating_add_size(values, gradients);
        return saturating_multiply_size(entries, sizeof(QComplex));
    }

    [[nodiscard]] bool fits_workspace(std::size_t bytes) const noexcept {
        return config_.max_workspace_bytes == 0U ||
               bytes <= config_.max_workspace_bytes;
    }

    void validate_workspace(
        const ExactAdjointGradientSchedulerWorkspace& workspace_value) const {
        if (workspace_value.token_ != workspace_token_) {
            throw QStateError(
                "Exact adjoint scheduler workspace does not match its plan");
        }
    }

    void prepare_workspace(
        ExactAdjointGradientSchedulerWorkspace& workspace_value,
        const ExactAdjointGradientSchedule& schedule) const {
        if (schedule.route == ExactAdjointScheduleRoute::PointParallel) {
            workspace_value.single_workspace_.reset();
            workspace_value.single_route_.reset();
            if (workspace_value.point_workspaces_.size() != schedule.worker_count) {
                std::vector<ExactAdjointGradientWorkspace> fresh;
                fresh.reserve(schedule.worker_count);
                for (std::size_t lane = 0U; lane < schedule.worker_count; ++lane) {
                    fresh.push_back(serial_.workspace());
                }
                workspace_value.point_workspaces_ = std::move(fresh);
            }
            return;
        }

        std::vector<ExactAdjointGradientWorkspace>().swap(
            workspace_value.point_workspaces_);
        if (!workspace_value.single_workspace_.has_value() ||
            !workspace_value.single_route_.has_value() ||
            *workspace_value.single_route_ != schedule.route) {
            workspace_value.single_workspace_.reset();
            if (schedule.route == ExactAdjointScheduleRoute::TermParallel) {
                workspace_value.single_workspace_.emplace(term_->workspace());
            } else {
                workspace_value.single_workspace_.emplace(serial_.workspace());
            }
            workspace_value.single_route_ = schedule.route;
        }
    }

    void evaluate_sequential(
        const ExactAdjointGradientPlan& plan,
        ExactAdjointGradientWorkspace& plan_workspace,
        std::span<const double> parameters,
        std::size_t point_count,
        std::size_t point_gradient_entries,
        ExactAdjointGradientSchedulerWorkspace& workspace_value) const {
        for (std::size_t point = 0U; point < point_count; ++point) {
            const std::size_t parameter_offset = point * parameter_count();
            const std::size_t value_offset = point * observable_count();
            const std::size_t gradient_offset = point * point_gradient_entries;
            plan.value_and_gradient(
                std::span<const double>(
                    parameters.data() + parameter_offset,
                    parameter_count()),
                std::span<QComplex>(
                    workspace_value.staged_values_.data() + value_offset,
                    observable_count()),
                std::span<QComplex>(
                    workspace_value.staged_gradients_.data() + gradient_offset,
                    point_gradient_entries),
                plan_workspace);
        }
    }

    void evaluate_points_parallel(
        std::span<const double> parameters,
        std::size_t point_count,
        std::size_t point_gradient_entries,
        std::size_t worker_count,
        ExactAdjointGradientSchedulerWorkspace& workspace_value) const {
        std::vector<std::exception_ptr> errors(worker_count);
        const auto worker = [&](std::size_t lane) {
            try {
                for (std::size_t point = lane;
                     point < point_count;
                     point += worker_count) {
                    const std::size_t parameter_offset = point * parameter_count();
                    const std::size_t value_offset = point * observable_count();
                    const std::size_t gradient_offset = point * point_gradient_entries;
                    serial_.value_and_gradient(
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
        threads.reserve(worker_count > 0U ? worker_count - 1U : 0U);
        for (std::size_t lane = 1U; lane < worker_count; ++lane) {
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
    }
};

}  // namespace qubit

#pragma once

#include "qubit/qplan.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace qubit {

class DependencyOperationPlan {
public:
    explicit DependencyOperationPlan(std::span<const Operation> operations);

    [[nodiscard]] std::size_t source_operation_count() const noexcept {
        return source_operation_count_;
    }
    [[nodiscard]] std::size_t optimized_operation_count() const noexcept {
        return operations_.size();
    }
    [[nodiscard]] std::size_t compiled_step_count() const noexcept {
        return compiled_.compiled_step_count();
    }
    [[nodiscard]] std::span<const Operation> optimized_operations() const noexcept {
        return operations_;
    }

    void execute(QRegister& state) const;
    void execute_many(
        std::span<QRegister* const> states,
        std::size_t worker_count = 0) const;

private:
    std::size_t source_operation_count_{0};
    std::vector<Operation> operations_{};
    OperationPlan compiled_;

    [[nodiscard]] static std::vector<Operation> optimize(
        std::span<const Operation> operations);
};

class IndependentComponentPlan {
public:
    explicit IndependentComponentPlan(std::span<const Operation> operations);

    [[nodiscard]] std::size_t operation_count() const noexcept {
        return operations_.size();
    }

    void execute(QRegister& state, std::size_t worker_count = 0) const;

private:
    std::vector<Operation> operations_{};

    [[nodiscard]] static bool supported(OperationCode code) noexcept;
};

}  // namespace qubit

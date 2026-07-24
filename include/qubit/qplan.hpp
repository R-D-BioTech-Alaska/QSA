#pragma once

#include "qubit/qstate.hpp"
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace qubit {

enum class OperationCode : std::uint32_t {
    X = 1,
    Y = 2,
    Z = 3,
    H = 4,
    S = 5,
    Sdg = 6,
    T = 7,
    Tdg = 8,
    Rx = 9,
    Ry = 10,
    Rz = 11,
    Cnot = 12,
    Cz = 13,
    Swap = 14,
    BitFlipTrajectory = 15,
    PhaseFlipTrajectory = 16,
    DepolarizingTrajectory = 17,
    AmplitudeDampingTrajectory = 18,
};

struct Operation {
    OperationCode code{OperationCode::X};
    QubitId first{0};
    QubitId second{0};
    double parameter{0.0};
    double sample{0.0};
};

struct ParameterizedOperation {
    Operation operation{};
    std::int32_t parameter_slot{-1};
    std::int32_t sample_slot{-1};
};

class OperationPlan {
public:
    explicit OperationPlan(std::span<const Operation> operations, bool optimize = true);

    [[nodiscard]] std::size_t source_operation_count() const noexcept {
        return source_operation_count_;
    }
    [[nodiscard]] std::size_t compiled_step_count() const noexcept { return steps_.size(); }
    [[nodiscard]] bool optimized() const noexcept { return optimized_; }

    void execute(QRegister& state, std::size_t* completed_operations = nullptr) const;
    void execute_many(
        std::span<QRegister* const> states,
        std::size_t worker_count = 0,
        std::size_t* completed_states = nullptr) const;

private:
    enum class StepKind : std::uint8_t {
        Direct,
        Matrix2,
        Diagonal,
    };

    struct Step {
        StepKind kind{StepKind::Direct};
        Operation operation{};
        QMatrix2 matrix{};
        std::vector<QDiagonalPhase> diagonal{};
        std::size_t source_begin{0};
        std::size_t source_end{0};
    };

    std::size_t source_operation_count_{0};
    bool optimized_{true};
    std::vector<Step> steps_{};

    static void apply_direct(QRegister& state, const Operation& operation);
};

class ParameterizedOperationPlan {
public:
    explicit ParameterizedOperationPlan(
        std::span<const ParameterizedOperation> operations,
        bool optimize = true);

    [[nodiscard]] std::size_t source_operation_count() const noexcept {
        return operations_.size();
    }
    [[nodiscard]] std::size_t parameter_count() const noexcept { return parameter_count_; }
    [[nodiscard]] bool optimized() const noexcept { return optimize_; }

    [[nodiscard]] OperationPlan bind(std::span<const double> parameters) const;
    void execute(
        QRegister& state,
        std::span<const double> parameters,
        std::size_t* completed_operations = nullptr) const;
    void execute_many(
        std::span<QRegister* const> states,
        std::span<const double> parameters,
        std::size_t worker_count = 0,
        std::size_t* completed_states = nullptr) const;

private:
    std::vector<ParameterizedOperation> operations_{};
    std::size_t parameter_count_{0};
    bool optimize_{true};
};

} 

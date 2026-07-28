#pragma once

#include "qubit/qstate.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace qubit {

struct CompiledDiagonalConfig {
    std::size_t max_coefficient_bytes{512ULL << 20};
    double unitary_tolerance{2e-12};
};

class CompiledDiagonalPlan {
public:
    CompiledDiagonalPlan(
        const QRegister& prototype,
        std::span<const QDiagonalPhase> phases,
        CompiledDiagonalConfig config = {});

    [[nodiscard]] std::size_t phase_count() const noexcept { return phase_count_; }
    [[nodiscard]] std::size_t component_count() const noexcept { return components_.size(); }
    [[nodiscard]] std::size_t coefficient_bytes() const noexcept;

    void execute(QRegister& state) const;
    void execute_many(
        std::span<QRegister* const> states,
        std::size_t worker_count = 0) const;

private:
    struct LocalPhase {
        std::size_t position{0};
        QComplex zero{1.0, 0.0};
        QComplex one{1.0, 0.0};
    };

    struct ComponentPlan {
        std::vector<QubitId> qubits{};
        ComponentKind kind{ComponentKind::Cell};
        std::vector<LocalPhase> phases{};
        std::vector<QComplex> coefficients{};
    };

    std::size_t phase_count_{0};
    CompiledDiagonalConfig config_{};
    std::vector<ComponentPlan> components_{};

    static void validate_config(const CompiledDiagonalConfig& config);
    static void validate_phase(
        const QDiagonalPhase& phase,
        const CompiledDiagonalConfig& config);
};

}  // namespace qubit

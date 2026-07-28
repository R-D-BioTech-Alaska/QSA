#pragma once

#include "qubit/qplan.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace qubit {

struct AdaptiveCompactionConfig {
    bool enabled{true};
    std::uint32_t warmup_checks{8};
    std::uint32_t audit_interval{64};
    double prior_success{1.0};
    double prior_failure{3.0};
    double minimum_posterior_success{0.20};
};

struct AdaptiveCompactionMetrics {
    std::uint64_t checks{0};
    std::uint64_t successes{0};
    std::uint64_t skips{0};
    std::uint64_t audit_checks{0};
};

class AdaptiveOperationPlan {
public:
    explicit AdaptiveOperationPlan(
        std::span<const Operation> operations,
        AdaptiveCompactionConfig config = {});

    [[nodiscard]] std::size_t operation_count() const noexcept {
        return operations_.size();
    }
    [[nodiscard]] const AdaptiveCompactionConfig& config() const noexcept {
        return config_;
    }
    [[nodiscard]] AdaptiveCompactionMetrics metrics() const noexcept;

    void reset_learning() noexcept;
    void execute(QRegister& state, std::size_t* completed_operations = nullptr);

private:
    enum class GateFamily : std::uint8_t {
        Cnot = 0,
        Cz = 1,
        Swap = 2,
    };

    struct Posterior {
        std::uint64_t checks{0};
        std::uint64_t successes{0};
        std::uint64_t skips{0};
        std::uint64_t audit_checks{0};
        std::uint32_t since_audit{0};
    };

    std::vector<Operation> operations_{};
    AdaptiveCompactionConfig config_{};
    std::array<Posterior, 18> posteriors_{};

    [[nodiscard]] std::size_t context_index(
        GateFamily family,
        std::size_t component_width) const noexcept;
    [[nodiscard]] bool should_check(Posterior& posterior, bool& audit) noexcept;
    void maybe_compact(
        QRegister& state,
        std::size_t component,
        std::span<const QubitId> candidates,
        GateFamily family);

    void apply_operation(QRegister& state, const Operation& operation);
    void apply_cnot(QRegister& state, QubitId control, QubitId target);
    void apply_cz(QRegister& state, QubitId first, QubitId second);
    void apply_swap(QRegister& state, QubitId first, QubitId second);
};

}  // namespace qubit

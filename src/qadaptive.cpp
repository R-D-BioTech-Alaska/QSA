#include "qubit/qadaptive.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <string>

namespace qubit {
namespace {

void validate_config(const AdaptiveCompactionConfig& config) {
    if (!std::isfinite(config.prior_success) || config.prior_success <= 0.0 ||
        !std::isfinite(config.prior_failure) || config.prior_failure <= 0.0) {
        throw QStateError("Adaptive compaction priors must be finite and positive");
    }
    if (!std::isfinite(config.minimum_posterior_success) ||
        config.minimum_posterior_success < 0.0 ||
        config.minimum_posterior_success > 1.0) {
        throw QStateError("Adaptive compaction threshold must be finite and in [0, 1]");
    }
}

}  // namespace

AdaptiveOperationPlan::AdaptiveOperationPlan(
    std::span<const Operation> operations,
    AdaptiveCompactionConfig config)
    : operations_(operations.begin(), operations.end()), config_(config) {
    validate_config(config_);
}

AdaptiveCompactionMetrics AdaptiveOperationPlan::metrics() const noexcept {
    AdaptiveCompactionMetrics result;
    for (const Posterior& posterior : posteriors_) {
        result.checks += posterior.checks;
        result.successes += posterior.successes;
        result.skips += posterior.skips;
        result.audit_checks += posterior.audit_checks;
    }
    return result;
}

void AdaptiveOperationPlan::reset_learning() noexcept {
    posteriors_.fill(Posterior{});
}

std::size_t AdaptiveOperationPlan::context_index(
    GateFamily family,
    std::size_t component_width) const noexcept {
    std::size_t width_bucket = 0U;
    if (component_width > 2U) {
        width_bucket = 1U;
    }
    if (component_width > 4U) {
        width_bucket = 2U;
    }
    if (component_width > 8U) {
        width_bucket = 3U;
    }
    if (component_width > 16U) {
        width_bucket = 4U;
    }
    if (component_width > 32U) {
        width_bucket = 5U;
    }
    return static_cast<std::size_t>(family) * 6U + width_bucket;
}

bool AdaptiveOperationPlan::should_check(Posterior& posterior, bool& audit) noexcept {
    audit = false;
    if (!config_.enabled) {
        return true;
    }
    if (posterior.checks < config_.warmup_checks) {
        return true;
    }
    if (config_.audit_interval != 0U && posterior.since_audit >= config_.audit_interval) {
        audit = true;
        return true;
    }

    const double success = config_.prior_success + static_cast<double>(posterior.successes);
    const double failure = config_.prior_failure +
                           static_cast<double>(posterior.checks - posterior.successes);
    if (success / (success + failure) >= config_.minimum_posterior_success) {
        return true;
    }

    ++posterior.skips;
    ++posterior.since_audit;
    return false;
}

void AdaptiveOperationPlan::maybe_compact(
    QRegister& state,
    std::size_t component,
    std::span<const QubitId> candidates,
    GateFamily family) {
    if (component >= state.components_.size()) {
        throw QStateError("Adaptive compaction received an invalid component");
    }

    Posterior& posterior =
        posteriors_[context_index(family, state.components_[component].qubits.size())];
    bool audit = false;
    if (!should_check(posterior, audit)) {
        return;
    }

    const std::size_t before = state.components_.size();
    state.compact_component_targets(component, candidates);
    ++posterior.checks;
    posterior.since_audit = 0U;
    if (audit) {
        ++posterior.audit_checks;
    }
    if (state.components_.size() > before) {
        ++posterior.successes;
    }
}

void AdaptiveOperationPlan::apply_cnot(
    QRegister& state,
    QubitId control,
    QubitId target) {
    state.validate_qubit(control);
    state.validate_qubit(target);
    if (control == target) {
        throw QStateError("A two-qubit gate requires two distinct qubits");
    }

    std::size_t component = state.component_index(control);
    const std::size_t other = state.component_index(target);
    if (component != other) {
        if (const auto control_value = QRegister::exact_z_basis(state.components_[component]);
            control_value.has_value()) {
            if (*control_value == 1) {
                state.apply_x(target);
            }
            return;
        }
        if (const auto target_eigenvalue =
                QRegister::exact_x_eigenvalue(state.components_[other]);
            target_eigenvalue.has_value()) {
            if (*target_eigenvalue == -1) {
                state.apply_z(control);
            }
            return;
        }
        component = state.merge_components_cached(component, other);
    }

    StateComponent& selected = state.components_[component];
    const std::size_t control_position = state.local_position(selected, control);
    const std::size_t target_position = state.local_position(selected, target);
    std::get<AmplitudeStore>(selected.state).apply_cnot(control_position, target_position);
    const std::array<QubitId, 2> candidates{control, target};
    maybe_compact(state, component, candidates, GateFamily::Cnot);
}

void AdaptiveOperationPlan::apply_cz(
    QRegister& state,
    QubitId first,
    QubitId second) {
    state.validate_qubit(first);
    state.validate_qubit(second);
    if (first == second) {
        throw QStateError("A two-qubit gate requires two distinct qubits");
    }

    std::size_t component = state.component_index(first);
    const std::size_t other = state.component_index(second);
    if (component != other) {
        if (const auto first_value = QRegister::exact_z_basis(state.components_[component]);
            first_value.has_value()) {
            if (*first_value == 1) {
                state.apply_z(second);
            }
            return;
        }
        if (const auto second_value = QRegister::exact_z_basis(state.components_[other]);
            second_value.has_value()) {
            if (*second_value == 1) {
                state.apply_z(first);
            }
            return;
        }
        component = state.merge_components_cached(component, other);
    }

    StateComponent& selected = state.components_[component];
    const std::size_t first_position = state.local_position(selected, first);
    const std::size_t second_position = state.local_position(selected, second);
    std::get<AmplitudeStore>(selected.state).apply_cz(first_position, second_position);
    const std::array<QubitId, 2> candidates{first, second};
    maybe_compact(state, component, candidates, GateFamily::Cz);
}

void AdaptiveOperationPlan::apply_swap(
    QRegister& state,
    QubitId first,
    QubitId second) {
    state.validate_qubit(first);
    state.validate_qubit(second);
    if (first == second) {
        throw QStateError("A two-qubit gate requires two distinct qubits");
    }

    const std::size_t component = state.component_index(first);
    const std::size_t other = state.component_index(second);
    if (component != other) {
        state.swap_disconnected_qubits(component, first, other, second);
        return;
    }

    StateComponent& selected = state.components_[component];
    const std::size_t first_position = state.local_position(selected, first);
    const std::size_t second_position = state.local_position(selected, second);
    std::get<AmplitudeStore>(selected.state).apply_swap(first_position, second_position);
    const std::array<QubitId, 2> candidates{first, second};
    maybe_compact(state, component, candidates, GateFamily::Swap);
}

void AdaptiveOperationPlan::apply_operation(QRegister& state, const Operation& operation) {
    switch (operation.code) {
        case OperationCode::X:
            state.apply_x(operation.first);
            break;
        case OperationCode::Y:
            state.apply_y(operation.first);
            break;
        case OperationCode::Z:
            state.apply_z(operation.first);
            break;
        case OperationCode::H:
            state.apply_h(operation.first);
            break;
        case OperationCode::S:
            state.apply_s(operation.first);
            break;
        case OperationCode::Sdg:
            state.apply_sdg(operation.first);
            break;
        case OperationCode::T:
            state.apply_t(operation.first);
            break;
        case OperationCode::Tdg:
            state.apply_tdg(operation.first);
            break;
        case OperationCode::Rx:
            state.apply_rx(operation.first, operation.parameter);
            break;
        case OperationCode::Ry:
            state.apply_ry(operation.first, operation.parameter);
            break;
        case OperationCode::Rz:
            state.apply_rz(operation.first, operation.parameter);
            break;
        case OperationCode::Cnot:
            apply_cnot(state, operation.first, operation.second);
            break;
        case OperationCode::Cz:
            apply_cz(state, operation.first, operation.second);
            break;
        case OperationCode::Swap:
            apply_swap(state, operation.first, operation.second);
            break;
        case OperationCode::BitFlipTrajectory:
            state.apply_bit_flip_trajectory(
                operation.first, operation.parameter, operation.sample);
            break;
        case OperationCode::PhaseFlipTrajectory:
            state.apply_phase_flip_trajectory(
                operation.first, operation.parameter, operation.sample);
            break;
        case OperationCode::DepolarizingTrajectory:
            state.apply_depolarizing_trajectory(
                operation.first, operation.parameter, operation.sample);
            break;
        case OperationCode::AmplitudeDampingTrajectory:
            state.apply_amplitude_damping_trajectory(
                operation.first, operation.parameter, operation.sample);
            break;
        default:
            throw QStateError("Adaptive operation plan contains an unknown opcode");
    }
}

void AdaptiveOperationPlan::execute(
    QRegister& state,
    std::size_t* completed_operations) {
    if (completed_operations != nullptr) {
        *completed_operations = 0U;
    }
    for (std::size_t index = 0; index < operations_.size(); ++index) {
        try {
            apply_operation(state, operations_[index]);
        } catch (const std::exception& error) {
            throw QStateError(
                "Adaptive plan operation " + std::to_string(index) +
                " failed: " + error.what());
        }
        if (completed_operations != nullptr) {
            *completed_operations = index + 1U;
        }
    }
}

}  // namespace qubit

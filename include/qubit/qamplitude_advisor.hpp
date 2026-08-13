#pragma once

#include "qubit/qfactor.hpp"
#include "qubit/qplan.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace qubit {

enum class ExactAmplitudeRoute : std::uint8_t {
    None = 0,
    PhaseGraphBranchSum = 1,
    HadamardPathFactor = 2,
    Indeterminate = 3,
};

enum class ExactAmplitudeDecisionCode : std::uint8_t {
    BothIneligible = 0,
    PhaseOnly = 1,
    HPathOnly = 2,
    HPathLowerPeakEnvelope = 3,
    CommonEnvelopeUnresolved = 4,
};

struct ExactAmplitudeAdvisorConfig {
    ExactFactorConfig factor{};
    std::size_t max_qubits{1U << 20U};
    std::size_t max_operations{1U << 24U};
    std::size_t max_phase_h_defects{20U};
    std::size_t max_phase_branches{1U << 20U};
    std::size_t max_hpath_events{1U << 20U};
    std::size_t minimum_hpath_log2_margin{8U};
};

struct ExactHadamardPathWidthCertificate {
    std::size_t qubits{0U};
    std::size_t operations{0U};
    std::size_t h_events{0U};
    std::size_t h_active_qubits{0U};
    std::size_t factor_variables{0U};
    std::size_t factor_count{0U};
    std::size_t peak_union_variables{0U};
    std::size_t peak_factor_entries{0U};
    std::size_t peak_factor_log2_ceiling{0U};
    std::size_t compiled_index_entries{0U};
    std::size_t workspace_slots{0U};
    std::size_t graph_estimated_bytes{0U};
    std::size_t plan_estimated_bytes{0U};
};

struct ExactAmplitudeDecision {
    ExactAmplitudeRoute route{ExactAmplitudeRoute::None};
    ExactAmplitudeDecisionCode code{ExactAmplitudeDecisionCode::BothIneligible};
    bool phase_graph_eligible{false};
    bool hpath_eligible{false};
    std::size_t phase_h_defects{0U};
    bool phase_branch_envelope_fits_size_t{true};
    std::size_t phase_branch_envelope{1U};
    std::size_t structural_log2_margin{0U};
    ExactHadamardPathWidthCertificate hpath{};
};

class ExactAmplitudeRepresentationAdvisor {
public:
    [[nodiscard]] static ExactAmplitudeDecision analyze_plus_state(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        ExactAmplitudeAdvisorConfig config = {}) {
        validate_configuration(qubit_count, operations.size(), config);

        ExactAmplitudeDecision decision;
        bool phase_compatible = true;
        bool hpath_compatible = true;
        for (const Operation& operation : operations) {
            validate_operation(qubit_count, operation);
            if (operation.code == OperationCode::H) {
                ++decision.phase_h_defects;
            }
            phase_compatible = phase_compatible && phase_operation(operation.code);
            hpath_compatible = hpath_compatible && hpath_operation(operation.code);
        }

        set_phase_envelope(decision);
        decision.phase_graph_eligible =
            phase_compatible &&
            decision.phase_h_defects <= config.max_phase_h_defects &&
            decision.phase_branch_envelope_fits_size_t &&
            decision.phase_branch_envelope <= config.max_phase_branches;

        const std::size_t hpath_event_cap =
            std::min(config.max_hpath_events, config.factor.max_variables);
        if (hpath_compatible && decision.phase_h_defects <= hpath_event_cap) {
            decision.hpath = build_hpath_certificate(qubit_count, operations, config);
            decision.hpath_eligible = true;
        }

        if (!decision.phase_graph_eligible && !decision.hpath_eligible) {
            return decision;
        }
        if (decision.phase_graph_eligible && !decision.hpath_eligible) {
            decision.route = ExactAmplitudeRoute::PhaseGraphBranchSum;
            decision.code = ExactAmplitudeDecisionCode::PhaseOnly;
            return decision;
        }
        if (!decision.phase_graph_eligible && decision.hpath_eligible) {
            decision.route = ExactAmplitudeRoute::HadamardPathFactor;
            decision.code = ExactAmplitudeDecisionCode::HPathOnly;
            return decision;
        }

        const std::size_t phase_log2 = decision.phase_h_defects;
        const std::size_t hpath_log2 = decision.hpath.peak_factor_log2_ceiling;
        if (phase_log2 >= hpath_log2 &&
            phase_log2 - hpath_log2 >= config.minimum_hpath_log2_margin) {
            decision.route = ExactAmplitudeRoute::HadamardPathFactor;
            decision.code = ExactAmplitudeDecisionCode::HPathLowerPeakEnvelope;
            decision.structural_log2_margin = phase_log2 - hpath_log2;
            return decision;
        }

        decision.route = ExactAmplitudeRoute::Indeterminate;
        decision.code = ExactAmplitudeDecisionCode::CommonEnvelopeUnresolved;
        return decision;
    }

private:
    [[nodiscard]] static ExactHadamardPathWidthCertificate build_hpath_certificate(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        const ExactAmplitudeAdvisorConfig& config) {
        const std::size_t npos = std::numeric_limits<std::size_t>::max();
        std::vector<std::size_t> first(qubit_count, npos);
        std::vector<std::size_t> last(qubit_count, npos);
        std::vector<std::size_t> next;
        std::vector<std::size_t> operation_event(operations.size(), npos);
        next.reserve(std::min(operations.size(), config.max_hpath_events));

        std::size_t h_events = 0U;
        std::size_t h_active_qubits = 0U;
        for (std::size_t operation_index = 0U;
             operation_index < operations.size();
             ++operation_index) {
            const Operation& operation = operations[operation_index];
            if (operation.code != OperationCode::H) {
                continue;
            }
            const std::size_t qubit = static_cast<std::size_t>(operation.first);
            const std::size_t event = h_events++;
            operation_event[operation_index] = event;
            next.push_back(npos);
            if (last[qubit] == npos) {
                first[qubit] = event;
                ++h_active_qubits;
            } else {
                next[last[qubit]] = event;
            }
            last[qubit] = event;
        }

        ExactFactorGraph graph(config.factor);
        std::vector<FactorVariableId> variables(h_events);
        for (std::size_t event = 0U; event < h_events; ++event) {
            variables[event] = graph.add_variable(2U);
        }
        std::vector<std::size_t> current = first;
        const std::array<QComplex, 2> unary{
            QComplex{1.0, 0.0}, QComplex{1.0, 0.0}};
        const std::array<QComplex, 4> binary{
            QComplex{1.0, 0.0}, QComplex{1.0, 0.0},
            QComplex{1.0, 0.0}, QComplex{1.0, 0.0}};

        const auto add_unary = [&](QubitId qubit) {
            const std::size_t event = current[static_cast<std::size_t>(qubit)];
            if (event != npos) {
                const std::array<FactorVariableId, 1> scope{variables[event]};
                (void)graph.add_dense_factor(scope, unary);
            }
        };
        const auto add_cz = [&](QubitId first_qubit, QubitId second_qubit) {
            const std::size_t left = current[static_cast<std::size_t>(first_qubit)];
            const std::size_t right = current[static_cast<std::size_t>(second_qubit)];
            if (left != npos && right != npos) {
                const std::array<FactorVariableId, 2> scope{
                    variables[left], variables[right]};
                (void)graph.add_dense_factor(scope, binary);
            } else if (left != npos || right != npos) {
                const std::array<FactorVariableId, 1> scope{
                    variables[left != npos ? left : right]};
                (void)graph.add_dense_factor(scope, unary);
            }
        };

        for (std::size_t operation_index = 0U;
             operation_index < operations.size();
             ++operation_index) {
            const Operation& operation = operations[operation_index];
            switch (operation.code) {
                case OperationCode::H: {
                    const std::size_t qubit = static_cast<std::size_t>(operation.first);
                    const std::size_t event = operation_event[operation_index];
                    if (event == npos || current[qubit] != event) {
                        throw QStateError("Amplitude advisor Hadamard event sequence is inconsistent");
                    }
                    const std::size_t following = next[event];
                    if (following != npos) {
                        const std::array<FactorVariableId, 2> scope{
                            variables[event], variables[following]};
                        (void)graph.add_dense_factor(scope, binary);
                    } else {
                        const std::array<FactorVariableId, 1> scope{variables[event]};
                        (void)graph.add_dense_factor(scope, unary);
                    }
                    current[qubit] = following;
                    break;
                }
                case OperationCode::Z:
                case OperationCode::S:
                case OperationCode::Sdg:
                case OperationCode::T:
                case OperationCode::Tdg:
                case OperationCode::Rz:
                    add_unary(operation.first);
                    break;
                case OperationCode::Cz:
                    add_cz(operation.first, operation.second);
                    break;
                default:
                    throw QStateError(
                        "Amplitude advisor operation left Hadamard-path contract");
            }
        }

        ExactFactorStats stats{};
        std::size_t plan_bytes = 0U;
        if (h_events != 0U) {
            ExactFactorPlan plan = graph.compile();
            stats = plan.stats();
            plan_bytes = plan.estimated_bytes();
        }
        return ExactHadamardPathWidthCertificate{
            qubit_count,
            operations.size(),
            h_events,
            h_active_qubits,
            graph.variable_count(),
            graph.factor_count(),
            stats.peak_union_variables,
            stats.peak_factor_entries,
            ceil_log2(std::max<std::size_t>(1U, stats.peak_factor_entries)),
            stats.compiled_index_entries,
            stats.workspace_slots,
            graph.estimated_bytes(),
            plan_bytes,
        };
    }

    static void validate_configuration(
        std::size_t qubit_count,
        std::size_t operation_count,
        const ExactAmplitudeAdvisorConfig& config) {
        if (qubit_count == 0U || qubit_count > config.max_qubits ||
            qubit_count > static_cast<std::size_t>(std::numeric_limits<QubitId>::max()) ||
            operation_count > config.max_operations ||
            config.max_phase_h_defects == 0U ||
            config.max_phase_branches == 0U ||
            config.max_hpath_events == 0U ||
            config.factor.max_variables == 0U) {
            throw QStateError("Amplitude advisor dimensions or configuration are invalid");
        }
    }

    static void validate_operation(
        std::size_t qubit_count,
        const Operation& operation) {
        const std::size_t first = static_cast<std::size_t>(operation.first);
        if (first >= qubit_count) {
            throw QStateError("Amplitude advisor operation target is out of range");
        }
        if (two_qubit(operation.code)) {
            const std::size_t second = static_cast<std::size_t>(operation.second);
            if (second >= qubit_count || first == second) {
                throw QStateError("Amplitude advisor two-qubit support is invalid");
            }
        }
        if ((operation.code == OperationCode::Rx ||
             operation.code == OperationCode::Ry ||
             operation.code == OperationCode::Rz) &&
            !std::isfinite(operation.parameter)) {
            throw QStateError("Amplitude advisor rotation parameter must be finite");
        }
    }

    static void set_phase_envelope(ExactAmplitudeDecision& decision) noexcept {
        constexpr std::size_t bits = std::numeric_limits<std::size_t>::digits;
        if (decision.phase_h_defects >= bits) {
            decision.phase_branch_envelope_fits_size_t = false;
            decision.phase_branch_envelope = std::numeric_limits<std::size_t>::max();
            return;
        }
        decision.phase_branch_envelope = std::size_t{1U} << decision.phase_h_defects;
    }

    [[nodiscard]] static std::size_t ceil_log2(std::size_t value) noexcept {
        if (value <= 1U) {
            return 0U;
        }
        std::size_t result = 0U;
        std::size_t current = value - 1U;
        while (current != 0U) {
            ++result;
            current >>= 1U;
        }
        return result;
    }

    [[nodiscard]] static bool phase_operation(OperationCode code) noexcept {
        switch (code) {
            case OperationCode::X:
            case OperationCode::Y:
            case OperationCode::Z:
            case OperationCode::H:
            case OperationCode::S:
            case OperationCode::Sdg:
            case OperationCode::T:
            case OperationCode::Tdg:
            case OperationCode::Rz:
            case OperationCode::Cz:
            case OperationCode::Swap:
                return true;
            default:
                return false;
        }
    }

    [[nodiscard]] static bool hpath_operation(OperationCode code) noexcept {
        switch (code) {
            case OperationCode::H:
            case OperationCode::Z:
            case OperationCode::S:
            case OperationCode::Sdg:
            case OperationCode::T:
            case OperationCode::Tdg:
            case OperationCode::Rz:
            case OperationCode::Cz:
                return true;
            default:
                return false;
        }
    }

    [[nodiscard]] static bool two_qubit(OperationCode code) noexcept {
        return code == OperationCode::Cnot ||
            code == OperationCode::Cz ||
            code == OperationCode::Swap;
    }
};

}  // namespace qubit

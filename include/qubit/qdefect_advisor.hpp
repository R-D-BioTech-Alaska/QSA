#pragma once

#include "qubit/qplan.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace qubit {

enum class ExactDefectRoute : std::uint8_t {
    None = 0,
    PhaseGraphLowH = 1,
    StabilizerLowT = 2,
    Indeterminate = 3,
};

enum class ExactDefectQuery : std::uint8_t {
    StateCarrier = 0,
    Amplitude = 1,
    ProbabilityOne = 2,
    PauliExpectation = 3,
};

enum class ExactDefectDecisionCode : std::uint8_t {
    BothIneligible = 0,
    PhaseOnly = 1,
    StabilizerOnly = 2,
    LowerHEnvelope = 3,
    LowerTEnvelope = 4,
    EqualEnvelope = 5,
};

struct ExactDefectAdvisorConfig {
    std::size_t max_qubits{1U << 20U};
    std::size_t max_operations{1U << 24U};
    std::size_t max_phase_h_defects{20U};
    std::size_t max_magic_t_defects{20U};
};

struct ExactDefectRouteEstimate {
    bool eligible{false};
    std::size_t defects{0U};
    std::size_t incompatible_operations{0U};
    bool branch_envelope_fits_size_t{true};
    std::size_t branch_envelope{1U};
};

struct ExactDefectDecision {
    ExactDefectRoute route{ExactDefectRoute::None};
    ExactDefectDecisionCode code{ExactDefectDecisionCode::BothIneligible};
    ExactDefectQuery query{ExactDefectQuery::StateCarrier};
    ExactDefectRouteEstimate phase_graph{};
    ExactDefectRouteEstimate low_magic{};
    std::size_t branch_envelope_gap_log2{0U};
};

class ExactDefectRepresentationAdvisor {
public:
    [[nodiscard]] static ExactDefectDecision analyze_plus_state(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        ExactDefectQuery query = ExactDefectQuery::StateCarrier,
        ExactDefectAdvisorConfig config = {}) {
        validate_configuration(qubit_count, operations.size(), config);

        ExactDefectRouteEstimate phase;
        ExactDefectRouteEstimate magic;
        for (const Operation& operation : operations) {
            validate_operation(qubit_count, operation);
            if (operation.code == OperationCode::H) {
                ++phase.defects;
            }
            if (operation.code == OperationCode::T ||
                operation.code == OperationCode::Tdg) {
                ++magic.defects;
            }
            if (!phase_operation(operation.code)) {
                ++phase.incompatible_operations;
            }
            if (!magic_operation(operation.code)) {
                ++magic.incompatible_operations;
            }
        }

        apply_query_contract(query, phase, magic);
        phase.eligible = phase.incompatible_operations == 0U &&
            phase.defects <= config.max_phase_h_defects;
        magic.eligible = magic.incompatible_operations == 0U &&
            magic.defects <= config.max_magic_t_defects;
        set_branch_envelope(phase);
        set_branch_envelope(magic);

        ExactDefectDecision decision;
        decision.query = query;
        decision.phase_graph = phase;
        decision.low_magic = magic;
        if (!phase.eligible && !magic.eligible) {
            return decision;
        }
        if (phase.eligible && !magic.eligible) {
            decision.route = ExactDefectRoute::PhaseGraphLowH;
            decision.code = ExactDefectDecisionCode::PhaseOnly;
            return decision;
        }
        if (!phase.eligible && magic.eligible) {
            decision.route = ExactDefectRoute::StabilizerLowT;
            decision.code = ExactDefectDecisionCode::StabilizerOnly;
            return decision;
        }

        if (phase.defects < magic.defects) {
            decision.route = ExactDefectRoute::PhaseGraphLowH;
            decision.code = ExactDefectDecisionCode::LowerHEnvelope;
            decision.branch_envelope_gap_log2 = magic.defects - phase.defects;
        } else if (magic.defects < phase.defects) {
            decision.route = ExactDefectRoute::StabilizerLowT;
            decision.code = ExactDefectDecisionCode::LowerTEnvelope;
            decision.branch_envelope_gap_log2 = phase.defects - magic.defects;
        } else {
            decision.route = ExactDefectRoute::Indeterminate;
            decision.code = ExactDefectDecisionCode::EqualEnvelope;
        }
        return decision;
    }

private:
    static void validate_configuration(
        std::size_t qubit_count,
        std::size_t operation_count,
        const ExactDefectAdvisorConfig& config) {
        if (qubit_count == 0U || qubit_count > config.max_qubits ||
            qubit_count > static_cast<std::size_t>(std::numeric_limits<QubitId>::max()) ||
            operation_count > config.max_operations ||
            config.max_phase_h_defects == 0U ||
            config.max_magic_t_defects == 0U) {
            throw QStateError("Defect advisor dimensions or configuration are invalid");
        }
    }

    static void validate_operation(
        std::size_t qubit_count,
        const Operation& operation) {
        if (static_cast<std::size_t>(operation.first) >= qubit_count) {
            throw QStateError("Defect advisor operation target is out of range");
        }
        if (two_qubit(operation.code)) {
            if (static_cast<std::size_t>(operation.second) >= qubit_count ||
                operation.first == operation.second) {
                throw QStateError("Defect advisor two-qubit operation is invalid");
            }
        }
        if ((operation.code == OperationCode::Rx ||
             operation.code == OperationCode::Ry ||
             operation.code == OperationCode::Rz) &&
            !std::isfinite(operation.parameter)) {
            throw QStateError("Defect advisor rotation parameter must be finite");
        }
    }

    static void apply_query_contract(
        ExactDefectQuery query,
        ExactDefectRouteEstimate& phase,
        ExactDefectRouteEstimate& magic) {
        switch (query) {
            case ExactDefectQuery::StateCarrier:
                return;
            case ExactDefectQuery::Amplitude:
                ++magic.incompatible_operations;
                return;
            case ExactDefectQuery::ProbabilityOne:
            case ExactDefectQuery::PauliExpectation:
                ++phase.incompatible_operations;
                return;
            default:
                throw QStateError("Defect advisor received an unknown query contract");
        }
    }

    static void set_branch_envelope(ExactDefectRouteEstimate& estimate) noexcept {
        constexpr std::size_t bits = std::numeric_limits<std::size_t>::digits;
        if (estimate.defects >= bits) {
            estimate.branch_envelope_fits_size_t = false;
            estimate.branch_envelope = std::numeric_limits<std::size_t>::max();
            return;
        }
        estimate.branch_envelope = std::size_t{1U} << estimate.defects;
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
            case OperationCode::Cz:
            case OperationCode::Swap:
                return true;
            case OperationCode::Rx:
            case OperationCode::Ry:
            case OperationCode::Rz:
            case OperationCode::Cnot:
            case OperationCode::BitFlipTrajectory:
            case OperationCode::PhaseFlipTrajectory:
            case OperationCode::DepolarizingTrajectory:
            case OperationCode::AmplitudeDampingTrajectory:
                return false;
            default:
                return false;
        }
    }

    [[nodiscard]] static bool magic_operation(OperationCode code) noexcept {
        switch (code) {
            case OperationCode::X:
            case OperationCode::Y:
            case OperationCode::Z:
            case OperationCode::H:
            case OperationCode::S:
            case OperationCode::Sdg:
            case OperationCode::T:
            case OperationCode::Tdg:
            case OperationCode::Cnot:
            case OperationCode::Cz:
            case OperationCode::Swap:
                return true;
            case OperationCode::Rx:
            case OperationCode::Ry:
            case OperationCode::Rz:
            case OperationCode::BitFlipTrajectory:
            case OperationCode::PhaseFlipTrajectory:
            case OperationCode::DepolarizingTrajectory:
            case OperationCode::AmplitudeDampingTrajectory:
                return false;
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

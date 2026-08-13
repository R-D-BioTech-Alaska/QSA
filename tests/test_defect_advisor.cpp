#include "qubit/qdefect_advisor.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

qubit::ExactDefectAdvisorConfig wide_config() {
    qubit::ExactDefectAdvisorConfig config;
    config.max_phase_h_defects = 128U;
    config.max_magic_t_defects = 128U;
    return config;
}

void route_by_defect_exponent() {
    using qubit::ExactDefectQuery;
    using qubit::ExactDefectRepresentationAdvisor;
    using qubit::ExactDefectRoute;
    using qubit::Operation;
    using qubit::OperationCode;

    std::vector<Operation> low_h;
    for (std::size_t index = 0U; index < 4U; ++index) {
        low_h.push_back({OperationCode::H, static_cast<qubit::QubitId>(index)});
    }
    for (std::size_t index = 0U; index < 64U; ++index) {
        low_h.push_back({OperationCode::T, static_cast<qubit::QubitId>(index % 64U)});
    }
    const auto phase = ExactDefectRepresentationAdvisor::analyze_plus_state(
        64U, low_h, ExactDefectQuery::StateCarrier, wide_config());
    require(phase.route == ExactDefectRoute::PhaseGraphLowH,
        "H4/T64 circuit did not choose low-H PhaseGraph");
    require(phase.phase_graph.defects == 4U && phase.low_magic.defects == 64U,
        "H4/T64 defect counts are wrong");
    require(phase.branch_envelope_gap_log2 == 60U,
        "H4/T64 branch-envelope gap is wrong");
    require(phase.phase_graph.branch_envelope == 16U,
        "H4 PhaseGraph branch envelope is wrong");
    require(!phase.low_magic.branch_envelope_fits_size_t,
        "T64 envelope should not be shifted into size_t");

    std::vector<Operation> low_t;
    for (std::size_t index = 0U; index < 64U; ++index) {
        low_t.push_back({OperationCode::H, static_cast<qubit::QubitId>(index % 64U)});
    }
    for (std::size_t index = 0U; index < 4U; ++index) {
        low_t.push_back({OperationCode::Tdg, static_cast<qubit::QubitId>(index)});
    }
    const auto magic = ExactDefectRepresentationAdvisor::analyze_plus_state(
        64U, low_t, ExactDefectQuery::StateCarrier, wide_config());
    require(magic.route == ExactDefectRoute::StabilizerLowT,
        "H64/T4 circuit did not choose low-T stabilizer");
    require(magic.branch_envelope_gap_log2 == 60U,
        "H64/T4 branch-envelope gap is wrong");
    require(magic.low_magic.branch_envelope == 16U,
        "T4 stabilizer branch envelope is wrong");
}

void query_and_gate_contracts() {
    using qubit::ExactDefectQuery;
    using qubit::ExactDefectRepresentationAdvisor;
    using qubit::ExactDefectRoute;
    using qubit::Operation;
    using qubit::OperationCode;

    const std::vector<Operation> common{
        {OperationCode::H, 0U},
        {OperationCode::T, 1U},
        {OperationCode::Cz, 0U, 1U},
    };
    const auto equal = ExactDefectRepresentationAdvisor::analyze_plus_state(
        4U, common, ExactDefectQuery::StateCarrier, wide_config());
    require(equal.route == ExactDefectRoute::Indeterminate,
        "equal H/T envelope should remain indeterminate");

    const auto amplitude = ExactDefectRepresentationAdvisor::analyze_plus_state(
        4U, common, ExactDefectQuery::Amplitude, wide_config());
    require(amplitude.route == ExactDefectRoute::PhaseGraphLowH,
        "amplitude query should restrict the advisor to PhaseGraph");

    const auto pauli = ExactDefectRepresentationAdvisor::analyze_plus_state(
        4U, common, ExactDefectQuery::PauliExpectation, wide_config());
    require(pauli.route == ExactDefectRoute::StabilizerLowT,
        "Pauli query should restrict the advisor to low-magic stabilizer");

    const std::vector<Operation> rz{{OperationCode::Rz, 0U, 0U, 0.41}};
    const auto rz_route = ExactDefectRepresentationAdvisor::analyze_plus_state(
        4U, rz, ExactDefectQuery::StateCarrier, wide_config());
    require(rz_route.route == ExactDefectRoute::None,
        "anonymous Rz should reject both current defect routes");

    const std::vector<Operation> cnot{{OperationCode::Cnot, 0U, 1U}};
    const auto cnot_route = ExactDefectRepresentationAdvisor::analyze_plus_state(
        4U, cnot, ExactDefectQuery::StateCarrier, wide_config());
    require(cnot_route.route == ExactDefectRoute::StabilizerLowT,
        "CNOT should disqualify the current PhaseGraph route");

    const std::vector<Operation> rx{{OperationCode::Rx, 0U, 0U, 0.23}};
    const auto unsupported = ExactDefectRepresentationAdvisor::analyze_plus_state(
        4U, rx, ExactDefectQuery::StateCarrier, wide_config());
    require(unsupported.route == ExactDefectRoute::None,
        "Rx should reject both exact defect routes");
}

void caps_and_validation() {
    using qubit::ExactDefectAdvisorConfig;
    using qubit::ExactDefectRepresentationAdvisor;
    using qubit::ExactDefectRoute;
    using qubit::Operation;
    using qubit::OperationCode;
    using qubit::QStateError;

    ExactDefectAdvisorConfig config;
    config.max_phase_h_defects = 3U;
    config.max_magic_t_defects = 3U;
    std::vector<Operation> operations;
    for (std::size_t index = 0U; index < 4U; ++index) {
        operations.push_back({OperationCode::H, 0U});
        operations.push_back({OperationCode::T, 0U});
    }
    const auto rejected = ExactDefectRepresentationAdvisor::analyze_plus_state(
        2U, operations, qubit::ExactDefectQuery::StateCarrier, config);
    require(rejected.route == ExactDefectRoute::None,
        "advisor defect caps did not reject both routes");

    bool threw = false;
    try {
        const std::vector<Operation> bad{{OperationCode::Cz, 0U, 0U}};
        (void)ExactDefectRepresentationAdvisor::analyze_plus_state(
            2U, bad, qubit::ExactDefectQuery::StateCarrier, wide_config());
    } catch (const QStateError&) {
        threw = true;
    }
    require(threw, "advisor accepted an invalid two-qubit support");

    threw = false;
    try {
        const std::vector<Operation> bad{{
            OperationCode::Rz, 0U, 0U, std::numeric_limits<double>::infinity()}};
        (void)ExactDefectRepresentationAdvisor::analyze_plus_state(
            2U, bad, qubit::ExactDefectQuery::StateCarrier, wide_config());
    } catch (const QStateError&) {
        threw = true;
    }
    require(threw, "advisor accepted a non-finite rotation parameter");
}

}  // namespace

int main() {
    route_by_defect_exponent();
    query_and_gate_contracts();
    caps_and_validation();
    return 0;
}

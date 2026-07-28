#include "qubit/qdot.hpp"
#include "qdot_detail.hpp"

namespace qubit::qdot {

using namespace detail;

void apply_reference_step(
    QRegister& state,
    const PocketConfig& config,
    std::span<const DotInput> inputs) {
    validate_pocket_config(config);
    if (inputs.size() != config.dot_count || state.qubit_count() != 2U * config.dot_count) {
        throw QStateError("Reference quantum-dot state/input dimensions do not match config");
    }
    for (const DotInput& input : inputs) {
        validate_input(input);
    }
    validate_derived_step(config, inputs);
    const auto edges = topology_edges(config.dot_count, config.topology);
    const double sub_dt = config.dt / static_cast<double>(config.trotter_steps);
    const double half = 0.5 * sub_dt;
    for (std::size_t trotter = 0; trotter < config.trotter_steps; ++trotter) {
        for (std::size_t dot = 0; dot < config.dot_count; ++dot) {
            reference_local(state, dot, config.dot, inputs[dot], half);
        }
        for (const auto& [first, second] : edges) {
            const QubitId occ_a = static_cast<QubitId>(2U * first);
            const QubitId spin_a = static_cast<QubitId>(2U * first + 1U);
            const QubitId occ_b = static_cast<QubitId>(2U * second);
            const QubitId spin_b = static_cast<QubitId>(2U * second + 1U);
            const auto& c = config.coupling;
            if (c.capacitive != 0.0) {
                reference_zz(state, occ_a, occ_b, 2.0 * c.capacitive * c.scale * sub_dt);
            }
            if (c.spin_exchange != 0.0) {
                const double angle = c.spin_exchange * c.scale * sub_dt;
                reference_xx(state, spin_a, spin_b, angle);
                reference_yy(state, spin_a, spin_b, angle);
            }
            if (c.charge_tunneling != 0.0) {
                const double angle = c.charge_tunneling * c.scale * sub_dt;
                reference_xx(state, occ_a, occ_b, angle);
                reference_yy(state, occ_a, occ_b, angle);
            }
        }
        for (std::size_t dot = config.dot_count; dot-- > 0U;) {
            const Coordinates c = coordinates(inputs[dot]);
            const auto diag = diagonal_coefficients(config.dot, c);
            const QubitId occ = static_cast<QubitId>(2U * dot);
            const QubitId spin = static_cast<QubitId>(2U * dot + 1U);
            state.apply_rx(occ, config.dot.charge_drive * c.x * half);
            state.apply_ry(occ, config.dot.charge_drive * c.y * half);
            state.apply_rx(spin, config.dot.spin_drive * c.y * half);
            state.apply_ry(spin, config.dot.spin_drive * (0.65 * c.x + 0.35 * c.z) * half);
            state.apply_rz(occ, 2.0 * diag.occ * half);
            state.apply_rz(spin, 2.0 * diag.spin * half);
            reference_zz(state, occ, spin, 2.0 * diag.zz * half);
        }
    }
}

}  // namespace qubit::qdot

#include "qubit/qdot.hpp"
#include "qdot_detail.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace qubit::qdot {

using namespace detail;

std::vector<std::pair<std::size_t, std::size_t>> topology_edges(
    std::size_t dot_count, Topology topology) {
    std::vector<std::pair<std::size_t, std::size_t>> edges;
    if (topology == Topology::Chain) {
        for (std::size_t dot = 0; dot + 1U < dot_count; ++dot) {
            edges.emplace_back(dot, dot + 1U);
        }
    } else if (topology == Topology::PairBlocks ||
               topology == Topology::PairPlusContext) {
        for (std::size_t dot = 0; dot + 1U < dot_count; dot += 2U) {
            edges.emplace_back(dot, dot + 1U);
        }
    } else {
        throw QStateError("Unknown quantum-dot topology");
    }
    return edges;
}

QuantumDotPocket::QuantumDotPocket(PocketConfig config) : config_(config) {
    validate_pocket_config(config_);
    if (config_.topology == Topology::Chain) {
        throw QStateError("Specialized QuantumDotPocket supports PairBlocks or PairPlusContext");
    }
    edges_ = topology_edges(config_.dot_count, config_.topology);
    build_blocks();
}

void QuantumDotPocket::build_blocks() {
    blocks_.clear();
    dot_block_.assign(config_.dot_count, 0U);
    dot_local_.assign(config_.dot_count, 0U);
    for (std::size_t dot = 0; dot < config_.dot_count;) {
        Block block;
        block.dots.push_back(dot);
        if (dot + 1U < config_.dot_count) {
            block.dots.push_back(dot + 1U);
        }
        const std::size_t block_index = blocks_.size();
        for (std::size_t local = 0; local < block.dots.size(); ++local) {
            dot_block_[block.dots[local]] = block_index;
            dot_local_[block.dots[local]] = local;
        }
        const std::size_t dimension = std::size_t{1} << (2U * block.dots.size());
        block.amplitudes.assign(dimension, QComplex{});
        block.amplitudes[0] = QComplex{1.0, 0.0};
        blocks_.push_back(std::move(block));
        dot += 2U;
    }
}

void QuantumDotPocket::reset() {
    for (auto& block : blocks_) {
        std::fill(block.amplitudes.begin(), block.amplitudes.end(), QComplex{});
        block.amplitudes[0] = QComplex{1.0, 0.0};
    }
}

std::pair<std::size_t, std::size_t> QuantumDotPocket::locate(std::size_t dot) const {
    if (dot >= config_.dot_count) {
        throw QStateError("Quantum dot index is out of range");
    }
    return {dot_block_[dot], dot_local_[dot]};
}

void QuantumDotPocket::apply_single(
    std::size_t dot, std::size_t within_dot, const QMatrix2& matrix) {
    const auto [block_index, local_dot] = locate(dot);
    apply_single_to_vector(blocks_[block_index].amplitudes, 2U * local_dot + within_dot, matrix);
}

void QuantumDotPocket::apply_cnot(
    std::size_t first_dot, std::size_t first_within,
    std::size_t second_dot, std::size_t second_within) {
    const auto [first_block, first_local] = locate(first_dot);
    const auto [second_block, second_local] = locate(second_dot);
    if (first_block != second_block) {
        throw QStateError("Quantum-dot block topology forbids cross-block coupling");
    }
    apply_cnot_to_vector(
        blocks_[first_block].amplitudes,
        2U * first_local + first_within,
        2U * second_local + second_within);
}

void QuantumDotPocket::apply_h(std::size_t dot, std::size_t within) {
    apply_single(dot, within, gates::h());
}
void QuantumDotPocket::apply_s(std::size_t dot, std::size_t within) {
    apply_single(dot, within, gates::s());
}
void QuantumDotPocket::apply_sdg(std::size_t dot, std::size_t within) {
    apply_single(dot, within, gates::sdg());
}
void QuantumDotPocket::apply_rx(std::size_t dot, std::size_t within, double angle) {
    apply_single(dot, within, gates::rx(angle));
}
void QuantumDotPocket::apply_ry(std::size_t dot, std::size_t within, double angle) {
    apply_single(dot, within, gates::ry(angle));
}
void QuantumDotPocket::apply_rz(std::size_t dot, std::size_t within, double angle) {
    const auto [block_index, local_dot] = locate(dot);
    auto& amplitudes = blocks_[block_index].amplitudes;
    const std::size_t bit = 2U * local_dot + within;
    const std::size_t mask = std::size_t{1} << bit;
    const QComplex zero_phase = QComplex::from_polar(1.0, -0.5 * angle);
    const QComplex one_phase = QComplex::from_polar(1.0, 0.5 * angle);
    for (std::size_t basis = 0; basis < amplitudes.size(); ++basis) {
        amplitudes[basis] *= (basis & mask) == 0U ? zero_phase : one_phase;
    }
}

void QuantumDotPocket::apply_zz(
    std::size_t first_dot, std::size_t first_within,
    std::size_t second_dot, std::size_t second_within, double angle) {
    const auto [first_block, first_local] = locate(first_dot);
    const auto [second_block, second_local] = locate(second_dot);
    if (first_block != second_block) {
        throw QStateError("Quantum-dot block topology forbids cross-block coupling");
    }
    auto& amplitudes = blocks_[first_block].amplitudes;
    const std::size_t first_mask = std::size_t{1} << (2U * first_local + first_within);
    const std::size_t second_mask = std::size_t{1} << (2U * second_local + second_within);
    const QComplex same = QComplex::from_polar(1.0, -0.5 * angle);
    const QComplex different = QComplex::from_polar(1.0, 0.5 * angle);
    for (std::size_t basis = 0; basis < amplitudes.size(); ++basis) {
        const bool first = (basis & first_mask) != 0U;
        const bool second = (basis & second_mask) != 0U;
        amplitudes[basis] *= first == second ? same : different;
    }
}

void QuantumDotPocket::apply_exchange(
    std::size_t first_dot, std::size_t first_within,
    std::size_t second_dot, std::size_t second_within, double angle) {
    const auto [first_block, first_local] = locate(first_dot);
    const auto [second_block, second_local] = locate(second_dot);
    if (first_block != second_block) {
        throw QStateError("Quantum-dot block topology forbids cross-block coupling");
    }
    auto& amplitudes = blocks_[first_block].amplitudes;
    const std::size_t first_mask = std::size_t{1} << (2U * first_local + first_within);
    const std::size_t second_mask = std::size_t{1} << (2U * second_local + second_within);
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const QComplex minus_i_sine{0.0, -sine};
    for (std::size_t basis = 0; basis < amplitudes.size(); ++basis) {
        if ((basis & first_mask) == 0U && (basis & second_mask) != 0U) {
            const std::size_t partner = basis ^ first_mask ^ second_mask;
            const QComplex left = amplitudes[basis];
            const QComplex right = amplitudes[partner];
            amplitudes[basis] = left * cosine + right * minus_i_sine;
            amplitudes[partner] = right * cosine + left * minus_i_sine;
        }
    }
}

void QuantumDotPocket::apply_xx(
    std::size_t first_dot, std::size_t first_within,
    std::size_t second_dot, std::size_t second_within, double angle) {
    apply_h(first_dot, first_within);
    apply_h(second_dot, second_within);
    apply_zz(first_dot, first_within, second_dot, second_within, angle);
    apply_h(first_dot, first_within);
    apply_h(second_dot, second_within);
}

void QuantumDotPocket::apply_yy(
    std::size_t first_dot, std::size_t first_within,
    std::size_t second_dot, std::size_t second_within, double angle) {
    apply_sdg(first_dot, first_within);
    apply_h(first_dot, first_within);
    apply_sdg(second_dot, second_within);
    apply_h(second_dot, second_within);
    apply_zz(first_dot, first_within, second_dot, second_within, angle);
    apply_h(first_dot, first_within);
    apply_s(first_dot, first_within);
    apply_h(second_dot, second_within);
    apply_s(second_dot, second_within);
}

void QuantumDotPocket::apply_local(
    std::size_t dot, const DotInput& input, double duration) {
    const Coordinates c = coordinates(input);
    const auto diag = diagonal_coefficients(config_.dot, c);
    apply_rz(dot, 0U, 2.0 * diag.occ * duration);
    apply_rz(dot, 1U, 2.0 * diag.spin * duration);
    apply_zz(dot, 0U, dot, 1U, 2.0 * diag.zz * duration);
    apply_rx(dot, 0U, config_.dot.charge_drive * c.x * duration);
    apply_ry(dot, 0U, config_.dot.charge_drive * c.y * duration);
    apply_rx(dot, 1U, config_.dot.spin_drive * c.y * duration);
    apply_ry(dot, 1U, config_.dot.spin_drive * (0.65 * c.x + 0.35 * c.z) * duration);
}

void QuantumDotPocket::apply_couplings(double duration) {
    const CouplingConfig coupling = config_.coupling;
    for (const auto& [first, second] : edges_) {
        if (coupling.capacitive != 0.0) {
            apply_zz(first, 0U, second, 0U,
                     2.0 * coupling.capacitive * coupling.scale * duration);
        }
        if (coupling.spin_exchange != 0.0) {
            const double angle = coupling.spin_exchange * coupling.scale * duration;
            apply_exchange(first, 1U, second, 1U, angle);
        }
        if (coupling.charge_tunneling != 0.0) {
            const double angle = coupling.charge_tunneling * coupling.scale * duration;
            apply_exchange(first, 0U, second, 0U, angle);
        }
    }
}

void QuantumDotPocket::step(std::span<const DotInput> inputs) {
    if (inputs.size() != config_.dot_count) {
        throw QStateError("QuantumDotPocket input count does not match dot count");
    }
    for (const DotInput& input : inputs) {
        validate_input(input);
    }
    validate_derived_step(config_, inputs);
    const double sub_dt = config_.dt / static_cast<double>(config_.trotter_steps);
    const double half = 0.5 * sub_dt;
    for (std::size_t trotter = 0; trotter < config_.trotter_steps; ++trotter) {
        for (std::size_t dot = 0; dot < config_.dot_count; ++dot) {
            apply_local(dot, inputs[dot], half);
        }
        apply_couplings(sub_dt);
        for (std::size_t dot = config_.dot_count; dot-- > 0U;) {
            const Coordinates c = coordinates(inputs[dot]);
            const auto diag = diagonal_coefficients(config_.dot, c);
            apply_rx(dot, 0U, config_.dot.charge_drive * c.x * half);
            apply_ry(dot, 0U, config_.dot.charge_drive * c.y * half);
            apply_rx(dot, 1U, config_.dot.spin_drive * c.y * half);
            apply_ry(dot, 1U, config_.dot.spin_drive * (0.65 * c.x + 0.35 * c.z) * half);
            apply_rz(dot, 0U, 2.0 * diag.occ * half);
            apply_rz(dot, 1U, 2.0 * diag.spin * half);
            apply_zz(dot, 0U, dot, 1U, 2.0 * diag.zz * half);
        }
    }
}

std::size_t QuantumDotPocket::max_component_qubits() const noexcept {
    std::size_t result = 0U;
    for (const auto& block : blocks_) {
        result = std::max(result, block.dots.size() * 2U);
    }
    return result;
}

std::size_t QuantumDotPocket::estimated_bytes() const noexcept {
    std::size_t total = sizeof(*this) + blocks_.capacity() * sizeof(Block) +
                        edges_.capacity() * sizeof(std::pair<std::size_t, std::size_t>) +
                        dot_block_.capacity() * sizeof(std::size_t) +
                        dot_local_.capacity() * sizeof(std::size_t);
    for (const auto& block : blocks_) {
        total += block.dots.capacity() * sizeof(std::size_t);
        total += block.amplitudes.capacity() * sizeof(QComplex);
    }
    return total;
}

std::vector<double> QuantumDotPocket::probabilities_one() const {
    std::vector<double> result(logical_qubit_count(), 0.0);
    for (const auto& block : blocks_) {
        for (std::size_t local_dot = 0; local_dot < block.dots.size(); ++local_dot) {
            for (std::size_t within = 0; within < 2U; ++within) {
                const std::size_t bit = 2U * local_dot + within;
                const std::size_t mask = std::size_t{1} << bit;
                long double probability = 0.0L;
                for (std::size_t basis = 0; basis < block.amplitudes.size(); ++basis) {
                    if ((basis & mask) != 0U) {
                        probability += static_cast<long double>(block.amplitudes[basis].norm2());
                    }
                }
                result[2U * block.dots[local_dot] + within] =
                    std::clamp(static_cast<double>(probability), 0.0, 1.0);
            }
        }
    }
    return result;
}

std::vector<QComplex> QuantumDotPocket::materialize(std::size_t max_qubits) const {
    const std::size_t qubits = logical_qubit_count();
    if (qubits > max_qubits || qubits >= std::numeric_limits<std::size_t>::digits) {
        throw QStateError("QuantumDotPocket materialization exceeds limit");
    }
    std::vector<QComplex> result(std::size_t{1} << qubits, QComplex{});
    const auto fill = [&](const auto& self, std::size_t block_index,
                          std::size_t global_basis, QComplex amplitude) -> void {
        if (block_index == blocks_.size()) {
            result[global_basis] = amplitude;
            return;
        }
        const Block& block = blocks_[block_index];
        for (std::size_t local_basis = 0; local_basis < block.amplitudes.size(); ++local_basis) {
            const QComplex value = block.amplitudes[local_basis];
            if (value.norm2() <= 0.0) {
                continue;
            }
            std::size_t mapped = 0U;
            for (std::size_t local_dot = 0; local_dot < block.dots.size(); ++local_dot) {
                for (std::size_t within = 0; within < 2U; ++within) {
                    const std::size_t local_bit = 2U * local_dot + within;
                    if (((local_basis >> local_bit) & 1U) != 0U) {
                        mapped |= std::size_t{1} << (2U * block.dots[local_dot] + within);
                    }
                }
            }
            self(self, block_index + 1U, global_basis | mapped, amplitude * value);
        }
    };
    fill(fill, 0U, 0U, QComplex{1.0, 0.0});
    return result;
}

double QuantumDotPocket::current_max_norm_error() const noexcept {
    double result = 0.0;
    for (const auto& block : blocks_) {
        result = std::max(result, std::abs(norm2(block.amplitudes) - 1.0));
    }
    return result;
}

bool QuantumDotPocket::validate(std::string* reason) const {
    const auto fail = [reason](const char* message) {
        if (reason != nullptr) {
            *reason = message;
        }
        return false;
    };
    try {
        validate_pocket_config(config_);
    } catch (const std::exception& error) {
        if (reason != nullptr) {
            *reason = error.what();
        }
        return false;
    }
    if (dot_block_.size() != config_.dot_count || dot_local_.size() != config_.dot_count) {
        return fail("quantum-dot lookup tables have the wrong size");
    }
    const std::size_t expected_blocks = (config_.dot_count + 1U) / 2U;
    if (blocks_.size() != expected_blocks) {
        return fail("quantum-dot block count does not match topology");
    }
    const std::size_t expected_edges = config_.dot_count / 2U;
    if (edges_.size() != expected_edges) {
        return fail("quantum-dot cached edge count does not match topology");
    }
    for (std::size_t edge = 0; edge < edges_.size(); ++edge) {
        const auto expected = std::pair<std::size_t, std::size_t>{2U * edge, 2U * edge + 1U};
        if (edges_[edge] != expected) {
            return fail("quantum-dot cached edge topology is inconsistent");
        }
    }
    std::vector<bool> seen(config_.dot_count, false);
    for (std::size_t block_index = 0; block_index < blocks_.size(); ++block_index) {
        const Block& block = blocks_[block_index];
        if (block.dots.empty() || block.dots.size() > 2U) {
            return fail("quantum-dot block width is invalid");
        }
        if (config_.topology == Topology::PairBlocks && block.dots.size() != 2U) {
            return fail("PairBlocks contains a non-pair block");
        }
        if (config_.topology == Topology::PairPlusContext &&
            block_index + 1U < blocks_.size() && block.dots.size() != 2U) {
            return fail("PairPlusContext contains an early singleton");
        }
        if (config_.topology == Topology::PairPlusContext &&
            block_index + 1U == blocks_.size() && block.dots.size() != 1U) {
            return fail("PairPlusContext is missing its final context singleton");
        }
        const std::size_t expected_dimension = std::size_t{1} << (2U * block.dots.size());
        if (block.amplitudes.size() != expected_dimension) {
            return fail("quantum-dot block dimension is invalid");
        }
        for (std::size_t local = 0; local < block.dots.size(); ++local) {
            const std::size_t dot = block.dots[local];
            if (dot >= config_.dot_count || seen[dot]) {
                return fail("quantum-dot block membership is invalid");
            }
            if (dot_block_[dot] != block_index || dot_local_[dot] != local) {
                return fail("quantum-dot lookup table is inconsistent");
            }
            seen[dot] = true;
        }
        long double total = 0.0L;
        for (const QComplex& amplitude : block.amplitudes) {
            if (!finite(amplitude.re) || !finite(amplitude.im)) {
                return fail("quantum-dot block contains a non-finite amplitude");
            }
            total += static_cast<long double>(amplitude.norm2());
        }
        const double value = static_cast<double>(total);
        if (!finite(value) || std::abs(value - 1.0) > 1e-8) {
            return fail("quantum-dot block is not normalized");
        }
    }
    if (std::find(seen.begin(), seen.end(), false) != seen.end()) {
        return fail("quantum-dot topology does not cover every dot");
    }
    return true;
}

}  // namespace qubit::qdot

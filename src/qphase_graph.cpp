#include "qubit/qphase_graph.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace qubit {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

struct PhaseGraphSplitMix64 {
    std::uint64_t state;

    [[nodiscard]] std::uint64_t next() noexcept {
        std::uint64_t value = (state += 0x9E3779B97F4A7C15ULL);
        value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
        return value ^ (value >> 31U);
    }
};

struct PhaseGraphInternalEdge {
    std::size_t first{0U};
    std::size_t second{0U};
    double angle{0.0};
};

struct PhaseGraphBoundaryEdge {
    std::size_t flip{0U};
    std::size_t external{0U};
    double angle{0.0};
};

void require_finite(double value, const char* label) {
    if (!std::isfinite(value)) {
        throw QStateError(std::string(label) + " must be finite");
    }
}

[[nodiscard]] double checked_sum(double first, double second, const char* label) {
    const double value = first + second;
    if (!std::isfinite(value)) {
        throw QStateError(std::string(label) + " overflowed");
    }
    return value;
}

[[nodiscard]] QComplex half_phase_factor(double angle, bool negative) {
    const QComplex phase = QComplex::from_polar(1.0, angle);
    return negative
        ? QComplex{0.5 * (1.0 - phase.re), -0.5 * phase.im}
        : QComplex{0.5 * (1.0 + phase.re), 0.5 * phase.im};
}

}  // namespace

PhaseGraphState::PhaseGraphState(
    std::size_t qubit_count,
    PhaseGraphConfig config)
    : qubit_count_(qubit_count), config_(config), local_phases_(qubit_count, 0.0) {
    if (qubit_count_ == 0U) {
        throw QStateError("PhaseGraphState requires at least one qubit");
    }
    if (qubit_count_ > static_cast<std::size_t>(std::numeric_limits<QubitId>::max())) {
        throw QStateError("PhaseGraphState qubit count exceeds QubitId capacity");
    }
    if (config_.max_edges == 0U) {
        throw QStateError("PhaseGraphState max_edges must be positive");
    }
}

void PhaseGraphState::validate_qubit(QubitId qubit) const {
    if (static_cast<std::size_t>(qubit) >= qubit_count_) {
        throw QStateError("Phase-graph qubit index is out of range");
    }
}

std::uint64_t PhaseGraphState::edge_key(QubitId first, QubitId second) noexcept {
    if (second < first) {
        std::swap(first, second);
    }
    return (static_cast<std::uint64_t>(first) << 32U) |
           static_cast<std::uint64_t>(second);
}

std::pair<QubitId, QubitId> PhaseGraphState::decode_edge(
    std::uint64_t key) noexcept {
    return {
        static_cast<QubitId>(key >> 32U),
        static_cast<QubitId>(key & 0xFFFFFFFFULL),
    };
}

void PhaseGraphState::add_global_phase(double angle) {
    require_finite(angle, "Phase-graph global angle");
    global_phase_ = checked_sum(global_phase_, angle, "Phase-graph global phase");
}

void PhaseGraphState::add_local_phase(QubitId qubit, double angle) {
    validate_qubit(qubit);
    require_finite(angle, "Phase-graph local angle");
    local_phases_[qubit] = checked_sum(
        local_phases_[qubit], angle, "Phase-graph local phase");
}

void PhaseGraphState::apply_x(QubitId qubit) {
    validate_qubit(qubit);
    const double next_global = checked_sum(
        global_phase_, local_phases_[qubit], "Phase-graph X global phase");
    const double next_local = -local_phases_[qubit];
    require_finite(next_local, "Phase-graph X local phase");

    for (const auto& [key, angle] : edge_phases_) {
        const auto [first, second] = decode_edge(key);
        if (first != qubit && second != qubit) {
            continue;
        }
        const QubitId other = first == qubit ? second : first;
        (void)checked_sum(
            local_phases_[other], angle, "Phase-graph X neighbor phase");
        require_finite(-angle, "Phase-graph X edge phase");
    }

    global_phase_ = next_global;
    local_phases_[qubit] = next_local;
    for (auto& [key, angle] : edge_phases_) {
        const auto [first, second] = decode_edge(key);
        if (first != qubit && second != qubit) {
            continue;
        }
        const QubitId other = first == qubit ? second : first;
        local_phases_[other] += angle;
        angle = -angle;
    }
}

void PhaseGraphState::apply_y(QubitId qubit) {
    validate_qubit(qubit);
    PhaseGraphState next = *this;
    next.apply_z(qubit);
    next.apply_x(qubit);
    next.add_global_phase(0.5 * kPi);
    *this = std::move(next);
}

void PhaseGraphState::apply_z(QubitId qubit) {
    add_local_phase(qubit, kPi);
}

void PhaseGraphState::apply_s(QubitId qubit) {
    add_local_phase(qubit, 0.5 * kPi);
}

void PhaseGraphState::apply_sdg(QubitId qubit) {
    add_local_phase(qubit, -0.5 * kPi);
}

void PhaseGraphState::apply_t(QubitId qubit) {
    add_local_phase(qubit, 0.25 * kPi);
}

void PhaseGraphState::apply_tdg(QubitId qubit) {
    add_local_phase(qubit, -0.25 * kPi);
}

void PhaseGraphState::apply_rz(QubitId qubit, double angle) {
    validate_qubit(qubit);
    require_finite(angle, "Phase-graph Rz angle");
    const double next_global = checked_sum(
        global_phase_, -0.5 * angle, "Phase-graph Rz global phase");
    const double next_local = checked_sum(
        local_phases_[qubit], angle, "Phase-graph Rz local phase");
    global_phase_ = next_global;
    local_phases_[qubit] = next_local;
}

void PhaseGraphState::apply_cz(QubitId first, QubitId second) {
    apply_controlled_phase(first, second, kPi);
}

void PhaseGraphState::apply_controlled_phase(
    QubitId first,
    QubitId second,
    double angle) {
    validate_qubit(first);
    validate_qubit(second);
    if (first == second) {
        throw QStateError("Phase-graph controlled phase requires distinct qubits");
    }
    require_finite(angle, "Phase-graph controlled phase angle");
    const std::uint64_t key = edge_key(first, second);
    const auto iterator = edge_phases_.find(key);
    if (iterator == edge_phases_.end()) {
        if (edge_phases_.size() >= config_.max_edges) {
            throw QStateError("PhaseGraphState exceeded configured edge limit");
        }
        edge_phases_.emplace(key, angle);
        return;
    }
    const double next = checked_sum(
        iterator->second, angle, "Phase-graph controlled phase");
    if (next == 0.0) {
        edge_phases_.erase(iterator);
    } else {
        iterator->second = next;
    }
}

void PhaseGraphState::apply_swap(QubitId first, QubitId second) {
    validate_qubit(first);
    validate_qubit(second);
    if (first == second) {
        throw QStateError("Phase-graph SWAP requires distinct qubits");
    }

    std::unordered_map<std::uint64_t, double> next;
    next.reserve(edge_phases_.size());
    for (const auto& [key, angle] : edge_phases_) {
        auto [left, right] = decode_edge(key);
        if (left == first) {
            left = second;
        } else if (left == second) {
            left = first;
        }
        if (right == first) {
            right = second;
        } else if (right == second) {
            right = first;
        }
        const std::uint64_t mapped = edge_key(left, right);
        const auto iterator = next.find(mapped);
        if (iterator == next.end()) {
            next.emplace(mapped, angle);
        } else {
            iterator->second = checked_sum(
                iterator->second, angle, "Phase-graph SWAP edge phase");
        }
    }
    std::swap(local_phases_[first], local_phases_[second]);
    edge_phases_ = std::move(next);
}

double PhaseGraphState::probability_one(QubitId qubit) const {
    validate_qubit(qubit);
    return 0.5;
}

PhaseGraphCoherenceResult PhaseGraphState::equatorial_coherence() const {
    PhaseGraphCoherenceResult result;
    result.values.reserve(qubit_count_);
    for (double angle : local_phases_) {
        result.values.push_back(QComplex::from_polar(1.0, angle));
    }
    for (const auto& [key, angle] : edge_phases_) {
        const auto [first, second] = decode_edge(key);
        const QComplex factor = half_phase_factor(angle, false);
        result.values[first] *= factor;
        result.values[second] *= factor;
    }
    result.receipt.qubits = qubit_count_;
    result.receipt.phase_edges = edge_phases_.size();
    result.receipt.structural_factors = qubit_count_ + 2U * edge_phases_.size();
    return result;
}

PhaseGraphPauliResult PhaseGraphState::pauli_expectation(
    std::span<const PauliFactor> factors,
    PhaseGraphPauliConfig config) const {
    if (config.max_flip_qubits >= std::numeric_limits<std::size_t>::digits ||
        config.max_enumerated_assignments == 0U) {
        throw QStateError("Phase-graph Pauli configuration is invalid");
    }

    std::vector<PauliAxis> axes(qubit_count_, PauliAxis::I);
    std::vector<bool> seen(qubit_count_, false);
    std::vector<QubitId> flips;
    std::size_t active_factors = 0U;
    for (const PauliFactor& factor : factors) {
        validate_qubit(factor.qubit);
        const std::size_t qubit = static_cast<std::size_t>(factor.qubit);
        if (seen[qubit]) {
            throw QStateError("Phase-graph Pauli observable contains a duplicate qubit");
        }
        seen[qubit] = true;
        axes[qubit] = factor.axis;
        if (factor.axis == PauliAxis::X || factor.axis == PauliAxis::Y) {
            flips.push_back(factor.qubit);
        }
        if (factor.axis != PauliAxis::I) {
            ++active_factors;
        }
    }
    if (flips.size() > config.max_flip_qubits) {
        throw QStateError("Phase-graph Pauli flip support exceeds the configured limit");
    }
    const std::size_t assignments = std::size_t{1} << flips.size();
    if (assignments > config.max_enumerated_assignments) {
        throw QStateError("Phase-graph Pauli enumeration exceeds the configured limit");
    }

    std::vector<int> flip_position(qubit_count_, -1);
    for (std::size_t index = 0U; index < flips.size(); ++index) {
        flip_position[flips[index]] = static_cast<int>(index);
    }

    std::vector<QubitId> external_qubits;
    std::vector<int> external_position(qubit_count_, -1);
    const auto ensure_external = [&](QubitId qubit) -> std::size_t {
        const std::size_t value = static_cast<std::size_t>(qubit);
        if (external_position[value] >= 0) {
            return static_cast<std::size_t>(external_position[value]);
        }
        const std::size_t index = external_qubits.size();
        external_qubits.push_back(qubit);
        external_position[value] = static_cast<int>(index);
        return index;
    };

    std::vector<PhaseGraphInternalEdge> internal_edges;
    std::vector<PhaseGraphBoundaryEdge> boundary_edges;
    internal_edges.reserve(edge_phases_.size());
    boundary_edges.reserve(edge_phases_.size());
    for (const auto& [key, angle] : edge_phases_) {
        const auto [first, second] = decode_edge(key);
        const int first_position = flip_position[first];
        const int second_position = flip_position[second];
        if (first_position >= 0 && second_position >= 0) {
            internal_edges.push_back({
                static_cast<std::size_t>(first_position),
                static_cast<std::size_t>(second_position),
                angle,
            });
        } else if (first_position >= 0 || second_position >= 0) {
            const std::size_t flip = first_position >= 0
                ? static_cast<std::size_t>(first_position)
                : static_cast<std::size_t>(second_position);
            const QubitId external = first_position >= 0 ? second : first;
            boundary_edges.push_back({flip, ensure_external(external), angle});
        }
    }
    for (const PauliFactor& factor : factors) {
        if (factor.axis == PauliAxis::Z) {
            (void)ensure_external(factor.qubit);
        }
    }

    std::vector<double> external_angles(external_qubits.size(), 0.0);
    QComplex total{};
    for (std::size_t assignment = 0U; assignment < assignments; ++assignment) {
        double phase = 0.0;
        QComplex pauli_phase{1.0, 0.0};
        for (std::size_t index = 0U; index < flips.size(); ++index) {
            const bool bit = ((assignment >> index) & 1U) != 0U;
            const double sign = bit ? 1.0 : -1.0;
            phase = checked_sum(
                phase,
                sign * local_phases_[flips[index]],
                "Phase-graph Pauli local phase");
            if (axes[flips[index]] == PauliAxis::Y) {
                pauli_phase *= QComplex{0.0, bit ? -1.0 : 1.0};
            }
        }
        for (const PhaseGraphInternalEdge& edge : internal_edges) {
            const double first = ((assignment >> edge.first) & 1U) != 0U ? 1.0 : 0.0;
            const double second = ((assignment >> edge.second) & 1U) != 0U ? 1.0 : 0.0;
            phase = checked_sum(
                phase,
                edge.angle * (first + second - 1.0),
                "Phase-graph Pauli internal phase");
        }

        std::fill(external_angles.begin(), external_angles.end(), 0.0);
        for (const PhaseGraphBoundaryEdge& edge : boundary_edges) {
            const bool bit = ((assignment >> edge.flip) & 1U) != 0U;
            const double sign = bit ? 1.0 : -1.0;
            external_angles[edge.external] = checked_sum(
                external_angles[edge.external],
                sign * edge.angle,
                "Phase-graph Pauli boundary phase");
        }

        QComplex term = QComplex::from_polar(1.0, phase) * pauli_phase;
        for (std::size_t index = 0U; index < external_qubits.size(); ++index) {
            const bool negative = axes[external_qubits[index]] == PauliAxis::Z;
            term *= half_phase_factor(external_angles[index], negative);
        }
        total += term;
    }
    total /= static_cast<double>(assignments);

    PhaseGraphPauliResult result;
    result.value = total;
    result.receipt.qubits = qubit_count_;
    result.receipt.phase_edges = edge_phases_.size();
    result.receipt.pauli_factors = active_factors;
    result.receipt.flipped_qubits = flips.size();
    result.receipt.internal_phase_edges = internal_edges.size();
    result.receipt.boundary_phase_edges = boundary_edges.size();
    result.receipt.external_factors = external_qubits.size();
    result.receipt.enumerated_assignments = assignments;
    return result;
}

QComplex PhaseGraphState::unit_phase(BasisIndex basis) const {
    if (qubit_count_ >= 63U) {
        throw QStateError(
            "Integer phase-graph phase lookup supports at most 62 qubits; use unit_phase_bits");
    }
    const BasisIndex dimension = BasisIndex{1} << qubit_count_;
    if (basis >= dimension) {
        throw QStateError("Phase-graph basis index is out of range");
    }
    std::vector<std::uint8_t> bits(qubit_count_);
    for (std::size_t qubit = 0; qubit < qubit_count_; ++qubit) {
        bits[qubit] = static_cast<std::uint8_t>((basis >> qubit) & 1U);
    }
    return unit_phase_bits(bits);
}

QComplex PhaseGraphState::unit_phase_bits(
    std::span<const std::uint8_t> bits) const {
    if (bits.size() != qubit_count_) {
        throw QStateError("Phase-graph bit-vector length does not match qubit count");
    }
    double phase = global_phase_;
    for (std::size_t qubit = 0; qubit < qubit_count_; ++qubit) {
        if (bits[qubit] > 1U) {
            throw QStateError("Phase-graph basis bits must be 0 or 1");
        }
        if (bits[qubit] != 0U) {
            phase = checked_sum(
                phase, local_phases_[qubit], "Phase-graph amplitude phase");
        }
    }
    for (const auto& [key, angle] : edge_phases_) {
        const auto [first, second] = decode_edge(key);
        if (bits[first] != 0U && bits[second] != 0U) {
            phase = checked_sum(phase, angle, "Phase-graph amplitude edge phase");
        }
    }
    return QComplex::from_polar(1.0, phase);
}

QComplex PhaseGraphState::amplitude(BasisIndex basis) const {
    const QComplex phase = unit_phase(basis);
    const double scale = std::exp2(log2_uniform_amplitude_scale());
    if (scale == 0.0) {
        throw QStateError("Phase-graph amplitude magnitude underflows double precision");
    }
    return phase * scale;
}

QComplex PhaseGraphState::amplitude_bits(
    std::span<const std::uint8_t> bits) const {
    const QComplex phase = unit_phase_bits(bits);
    const double scale = std::exp2(log2_uniform_amplitude_scale());
    if (scale == 0.0) {
        throw QStateError("Phase-graph amplitude magnitude underflows double precision");
    }
    return phase * scale;
}

std::vector<QComplex> PhaseGraphState::materialize(std::size_t max_qubits) const {
    if (qubit_count_ > max_qubits || qubit_count_ >= 63U) {
        throw QStateError("Phase-graph materialization exceeds the requested qubit limit");
    }
    const BasisIndex dimension = BasisIndex{1} << qubit_count_;
    std::vector<QComplex> result(static_cast<std::size_t>(dimension));
    for (BasisIndex basis = 0; basis < dimension; ++basis) {
        result[static_cast<std::size_t>(basis)] = amplitude(basis);
    }
    return result;
}

std::vector<int> PhaseGraphState::sample_bits(std::uint64_t seed) const {
    PhaseGraphSplitMix64 generator{seed};
    std::vector<int> result(qubit_count_);
    std::uint64_t word = 0U;
    for (std::size_t qubit = 0; qubit < qubit_count_; ++qubit) {
        if ((qubit & 63U) == 0U) {
            word = generator.next();
        }
        result[qubit] = static_cast<int>((word >> (qubit & 63U)) & 1U);
    }
    return result;
}

BasisIndex PhaseGraphState::sample_basis(std::uint64_t seed) const {
    if (qubit_count_ > 63U) {
        throw QStateError("Integer phase-graph sampling supports at most 63 qubits");
    }
    const auto bits = sample_bits(seed);
    BasisIndex result = 0U;
    for (std::size_t qubit = 0; qubit < qubit_count_; ++qubit) {
        if (bits[qubit] != 0) {
            result |= BasisIndex{1} << qubit;
        }
    }
    return result;
}

std::size_t PhaseGraphState::estimated_bytes() const noexcept {
    return sizeof(*this) + local_phases_.capacity() * sizeof(double) +
           edge_phases_.size() *
               (sizeof(std::uint64_t) + sizeof(double) + 2U * sizeof(void*));
}

bool PhaseGraphState::validate(std::string* reason) const {
    const auto fail = [reason](const char* message) {
        if (reason != nullptr) {
            *reason = message;
        }
        return false;
    };
    if (qubit_count_ == 0U || local_phases_.size() != qubit_count_) {
        return fail("phase-graph dimensions are invalid");
    }
    if (!std::isfinite(global_phase_)) {
        return fail("phase-graph global phase is non-finite");
    }
    for (double angle : local_phases_) {
        if (!std::isfinite(angle)) {
            return fail("phase-graph local phase is non-finite");
        }
    }
    if (edge_phases_.size() > config_.max_edges) {
        return fail("phase-graph edge count exceeds the configured limit");
    }
    for (const auto& [key, angle] : edge_phases_) {
        const auto [first, second] = decode_edge(key);
        if (first >= second || static_cast<std::size_t>(second) >= qubit_count_) {
            return fail("phase-graph edge is invalid");
        }
        if (!std::isfinite(angle)) {
            return fail("phase-graph edge phase is non-finite");
        }
    }
    return true;
}

std::string PhaseGraphState::describe() const {
    std::ostringstream stream;
    stream << "QSA uniform phase graph\n"
           << "qubits: " << qubit_count_ << "\n"
           << "quadratic phase edges: " << edge_phases_.size() << "\n"
           << "estimated engine bytes: " << estimated_bytes() << "\n";
    return stream.str();
}

}  // namespace qubit

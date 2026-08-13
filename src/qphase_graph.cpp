#include "qubit/qphase_graph.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace qubit {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kInvSqrt2 = 0.707106781186547524400844362104849039;

struct PhaseGraphSplitMix64 {
    std::uint64_t state;

    [[nodiscard]] std::uint64_t next() noexcept {
        std::uint64_t value = (state += 0x9E3779B97F4A7C15ULL);
        value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
        return value ^ (value >> 31U);
    }
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

[[nodiscard]] std::size_t saturating_add(std::size_t first, std::size_t second) noexcept {
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (second > maximum - first) {
        return maximum;
    }
    return first + second;
}

[[nodiscard]] std::size_t saturating_multiply(
    std::size_t first,
    std::size_t second) noexcept {
    if (first == 0U || second == 0U) {
        return 0U;
    }
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (first > maximum / second) {
        return maximum;
    }
    return first * second;
}

[[nodiscard]] QComplex materialize_scaled(
    const PhaseGraphScaledAmplitude& amplitude,
    const char* label) {
    if (amplitude.mantissa.norm2() == 0.0) {
        return {};
    }
    const double scale = std::exp2(amplitude.log2_scale);
    if (scale == 0.0) {
        throw QStateError(std::string(label) + " underflows double precision");
    }
    if (!std::isfinite(scale)) {
        throw QStateError(std::string(label) + " overflows double precision");
    }
    return amplitude.mantissa * scale;
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

PhaseGraphScaledAmplitude PhaseGraphState::scaled_amplitude(BasisIndex basis) const {
    if (qubit_count_ >= 63U) {
        throw QStateError(
            "Integer phase-graph amplitude lookup supports at most 62 qubits; use scaled_amplitude_bits");
    }
    const BasisIndex dimension = BasisIndex{1} << qubit_count_;
    if (basis >= dimension) {
        throw QStateError("Phase-graph basis index is out of range");
    }
    std::vector<std::uint8_t> bits(qubit_count_);
    for (std::size_t qubit = 0; qubit < qubit_count_; ++qubit) {
        bits[qubit] = static_cast<std::uint8_t>((basis >> qubit) & 1U);
    }
    return scaled_amplitude_bits(bits);
}

PhaseGraphScaledAmplitude PhaseGraphState::scaled_amplitude_bits(
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
    return {
        QComplex::from_polar(1.0, phase),
        -0.5 * static_cast<double>(qubit_count_),
    };
}

QComplex PhaseGraphState::amplitude(BasisIndex basis) const {
    return materialize_scaled(scaled_amplitude(basis), "Phase-graph amplitude magnitude");
}

QComplex PhaseGraphState::amplitude_bits(
    std::span<const std::uint8_t> bits) const {
    return materialize_scaled(
        scaled_amplitude_bits(bits), "Phase-graph amplitude magnitude");
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

PhaseGraphBranchState::PhaseGraphBranchState(
    std::size_t qubit_count,
    PhaseGraphBranchConfig config)
    : qubit_count_(qubit_count), config_(config) {
    if (config_.max_branches == 0U) {
        throw QStateError("PhaseGraphBranchState max_branches must be positive");
    }
    if (config_.max_estimated_bytes == 0U) {
        throw QStateError("PhaseGraphBranchState max_estimated_bytes must be positive");
    }
    branches_.emplace_back(
        QComplex{1.0, 0.0}, PhaseGraphState(qubit_count_, config_.phase_graph));
    enforce_resources(branches_);
}

std::size_t PhaseGraphBranchState::estimated_bytes(
    const std::vector<Branch>& branches) const noexcept {
    std::size_t bytes = sizeof(*this);
    bytes = saturating_add(
        bytes, saturating_multiply(branches.capacity(), sizeof(Branch)));
    for (const auto& branch : branches) {
        const std::size_t state_bytes = branch.state.estimated_bytes();
        if (state_bytes > sizeof(PhaseGraphState)) {
            bytes = saturating_add(bytes, state_bytes - sizeof(PhaseGraphState));
        }
    }
    return bytes;
}

std::size_t PhaseGraphBranchState::estimated_bytes() const noexcept {
    return estimated_bytes(branches_);
}

void PhaseGraphBranchState::enforce_resources(
    const std::vector<Branch>& branches) const {
    if (branches.empty()) {
        throw QStateError("PhaseGraphBranchState requires at least one branch");
    }
    if (branches.size() > config_.max_branches) {
        throw QStateError("PhaseGraphBranchState exceeded configured branch limit");
    }
    if (estimated_bytes(branches) > config_.max_estimated_bytes) {
        throw QStateError("PhaseGraphBranchState exceeded configured memory limit");
    }
}

void PhaseGraphBranchState::apply_h(QubitId qubit) {
    if (branches_.size() > config_.max_branches / 2U) {
        throw QStateError("PhaseGraphBranchState exceeded configured branch limit");
    }
    const std::size_t next_count = branches_.size() * 2U;
    std::size_t projected = sizeof(*this);
    projected = saturating_add(
        projected, saturating_multiply(next_count, sizeof(Branch)));
    for (const auto& branch : branches_) {
        const std::size_t state_bytes = branch.state.estimated_bytes();
        const std::size_t dynamic_bytes = state_bytes > sizeof(PhaseGraphState)
            ? state_bytes - sizeof(PhaseGraphState)
            : 0U;
        projected = saturating_add(
            projected, saturating_multiply(2U, dynamic_bytes));
    }
    if (projected > config_.max_estimated_bytes) {
        throw QStateError("PhaseGraphBranchState exceeded configured memory limit");
    }

    std::vector<Branch> next;
    next.reserve(next_count);
    for (const auto& branch : branches_) {
        PhaseGraphState left = branch.state;
        left.apply_x(qubit);
        QComplex coefficient = branch.coefficient;
        coefficient *= kInvSqrt2;
        next.emplace_back(coefficient, std::move(left));

        PhaseGraphState right = branch.state;
        right.apply_z(qubit);
        next.emplace_back(coefficient, std::move(right));
    }
    enforce_resources(next);
    branches_ = std::move(next);
    ++hadamard_defects_;
}

void PhaseGraphBranchState::apply_single(
    void (PhaseGraphState::*operation)(QubitId),
    QubitId qubit) {
    std::vector<Branch> next = branches_;
    for (auto& branch : next) {
        (branch.state.*operation)(qubit);
    }
    enforce_resources(next);
    branches_ = std::move(next);
}

void PhaseGraphBranchState::apply_single_angle(
    void (PhaseGraphState::*operation)(QubitId, double),
    QubitId qubit,
    double angle) {
    std::vector<Branch> next = branches_;
    for (auto& branch : next) {
        (branch.state.*operation)(qubit, angle);
    }
    enforce_resources(next);
    branches_ = std::move(next);
}

void PhaseGraphBranchState::apply_two(
    void (PhaseGraphState::*operation)(QubitId, QubitId),
    QubitId first,
    QubitId second) {
    std::vector<Branch> next = branches_;
    for (auto& branch : next) {
        (branch.state.*operation)(first, second);
    }
    enforce_resources(next);
    branches_ = std::move(next);
}

void PhaseGraphBranchState::apply_two_angle(
    void (PhaseGraphState::*operation)(QubitId, QubitId, double),
    QubitId first,
    QubitId second,
    double angle) {
    std::vector<Branch> next = branches_;
    for (auto& branch : next) {
        (branch.state.*operation)(first, second, angle);
    }
    enforce_resources(next);
    branches_ = std::move(next);
}

void PhaseGraphBranchState::apply_x(QubitId qubit) {
    apply_single(&PhaseGraphState::apply_x, qubit);
}

void PhaseGraphBranchState::apply_y(QubitId qubit) {
    apply_single(&PhaseGraphState::apply_y, qubit);
}

void PhaseGraphBranchState::apply_z(QubitId qubit) {
    apply_single(&PhaseGraphState::apply_z, qubit);
}

void PhaseGraphBranchState::apply_s(QubitId qubit) {
    apply_single(&PhaseGraphState::apply_s, qubit);
}

void PhaseGraphBranchState::apply_sdg(QubitId qubit) {
    apply_single(&PhaseGraphState::apply_sdg, qubit);
}

void PhaseGraphBranchState::apply_t(QubitId qubit) {
    apply_single(&PhaseGraphState::apply_t, qubit);
}

void PhaseGraphBranchState::apply_tdg(QubitId qubit) {
    apply_single(&PhaseGraphState::apply_tdg, qubit);
}

void PhaseGraphBranchState::apply_rz(QubitId qubit, double angle) {
    apply_single_angle(&PhaseGraphState::apply_rz, qubit, angle);
}

void PhaseGraphBranchState::apply_cz(QubitId first, QubitId second) {
    apply_two(&PhaseGraphState::apply_cz, first, second);
}

void PhaseGraphBranchState::apply_controlled_phase(
    QubitId first,
    QubitId second,
    double angle) {
    apply_two_angle(&PhaseGraphState::apply_controlled_phase, first, second, angle);
}

void PhaseGraphBranchState::apply_swap(QubitId first, QubitId second) {
    apply_two(&PhaseGraphState::apply_swap, first, second);
}

PhaseGraphScaledAmplitude PhaseGraphBranchState::scaled_amplitude(
    BasisIndex basis) const {
    if (qubit_count_ >= 63U) {
        throw QStateError(
            "Integer phase-graph branch amplitude lookup supports at most 62 qubits; use scaled_amplitude_bits");
    }
    const BasisIndex dimension = BasisIndex{1} << qubit_count_;
    if (basis >= dimension) {
        throw QStateError("Phase-graph branch basis index is out of range");
    }
    std::vector<std::uint8_t> bits(qubit_count_);
    for (std::size_t qubit = 0; qubit < qubit_count_; ++qubit) {
        bits[qubit] = static_cast<std::uint8_t>((basis >> qubit) & 1U);
    }
    return scaled_amplitude_bits(bits);
}

PhaseGraphScaledAmplitude PhaseGraphBranchState::scaled_amplitude_bits(
    std::span<const std::uint8_t> bits) const {
    QComplex mantissa{};
    double log2_scale = 0.0;
    bool first = true;
    for (const auto& branch : branches_) {
        const PhaseGraphScaledAmplitude value = branch.state.scaled_amplitude_bits(bits);
        if (first) {
            log2_scale = value.log2_scale;
            first = false;
        } else if (value.log2_scale != log2_scale) {
            throw QStateError("PhaseGraphBranchState branch amplitude scales diverged");
        }
        mantissa += branch.coefficient * value.mantissa;
    }
    return {mantissa, log2_scale};
}

QComplex PhaseGraphBranchState::amplitude(BasisIndex basis) const {
    return materialize_scaled(
        scaled_amplitude(basis), "Phase-graph branch amplitude magnitude");
}

QComplex PhaseGraphBranchState::amplitude_bits(
    std::span<const std::uint8_t> bits) const {
    return materialize_scaled(
        scaled_amplitude_bits(bits), "Phase-graph branch amplitude magnitude");
}

std::vector<QComplex> PhaseGraphBranchState::materialize(
    std::size_t max_qubits) const {
    if (qubit_count_ > max_qubits || qubit_count_ >= 63U) {
        throw QStateError(
            "Phase-graph branch materialization exceeds the requested qubit limit");
    }
    const BasisIndex dimension = BasisIndex{1} << qubit_count_;
    std::vector<QComplex> result(static_cast<std::size_t>(dimension));
    for (BasisIndex basis = 0; basis < dimension; ++basis) {
        result[static_cast<std::size_t>(basis)] = amplitude(basis);
    }
    return result;
}

bool PhaseGraphBranchState::validate(std::string* reason) const {
    const auto fail = [reason](const char* message) {
        if (reason != nullptr) {
            *reason = message;
        }
        return false;
    };
    if (qubit_count_ == 0U) {
        return fail("phase-graph branch qubit count is invalid");
    }
    if (config_.max_branches == 0U || branches_.empty() ||
        branches_.size() > config_.max_branches) {
        return fail("phase-graph branch count is invalid");
    }
    if (config_.max_estimated_bytes == 0U ||
        estimated_bytes() > config_.max_estimated_bytes) {
        return fail("phase-graph branch memory estimate exceeds the configured limit");
    }
    for (const auto& branch : branches_) {
        if (!std::isfinite(branch.coefficient.re) ||
            !std::isfinite(branch.coefficient.im)) {
            return fail("phase-graph branch coefficient is non-finite");
        }
        if (branch.state.qubit_count() != qubit_count_) {
            return fail("phase-graph branch qubit count differs from carrier");
        }
        std::string branch_reason;
        if (!branch.state.validate(&branch_reason)) {
            if (reason != nullptr) {
                *reason = "phase-graph branch invalid: " + branch_reason;
            }
            return false;
        }
    }
    return true;
}

std::string PhaseGraphBranchState::describe() const {
    std::ostringstream stream;
    stream << "QSA bounded phase-graph branch sum\n"
           << "qubits: " << qubit_count_ << "\n"
           << "Hadamard defects: " << hadamard_defects_ << "\n"
           << "branches: " << branches_.size() << " / " << config_.max_branches << "\n"
           << "estimated engine bytes: " << estimated_bytes() << " / "
           << config_.max_estimated_bytes << "\n";
    return stream.str();
}

}  // namespace qubit

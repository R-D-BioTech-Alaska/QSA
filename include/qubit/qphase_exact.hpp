#pragma once

#include "qubit/qplan.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <utility>
#include <vector>

namespace qubit {

struct ExactCyclotomic8 {
    std::array<std::int64_t, 4> values{0, 0, 0, 0};

    [[nodiscard]] static ExactCyclotomic8 one() noexcept {
        ExactCyclotomic8 value;
        value.values[0] = 1;
        return value;
    }

    [[nodiscard]] bool zero() const noexcept {
        return values[0] == 0 && values[1] == 0 && values[2] == 0 && values[3] == 0;
    }

    void add(const ExactCyclotomic8& other) {
        for (std::size_t index = 0U; index < values.size(); ++index) {
            values[index] = checked_add(values[index], other.values[index]);
        }
    }

    void rotate(std::uint8_t ticks) {
        ticks &= 7U;
        for (std::uint8_t tick = 0U; tick < ticks; ++tick) {
            const std::int64_t a = values[0];
            const std::int64_t b = values[1];
            const std::int64_t c = values[2];
            const std::int64_t d = values[3];
            values = {checked_negate(d), a, b, c};
        }
    }

    void divide_by_two(std::size_t shifts) noexcept {
        for (std::size_t shift = 0U; shift < shifts; ++shift) {
            for (std::int64_t& value : values) {
                value /= 2;
            }
        }
    }

    [[nodiscard]] std::uint64_t max_abs() const noexcept {
        std::uint64_t maximum = 0U;
        for (const std::int64_t value : values) {
            maximum = std::max(maximum, magnitude(value));
        }
        return maximum;
    }

    [[nodiscard]] QComplex evaluate() const noexcept {
        constexpr double inverse_sqrt_two = 0.707106781186547524400844362104849039;
        const double a = static_cast<double>(values[0]);
        const double b = static_cast<double>(values[1]);
        const double c = static_cast<double>(values[2]);
        const double d = static_cast<double>(values[3]);
        return {
            a + inverse_sqrt_two * (b - d),
            c + inverse_sqrt_two * (b + d),
        };
    }

private:
    [[nodiscard]] static std::int64_t checked_add(std::int64_t left, std::int64_t right) {
        if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
            (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right)) {
            throw QStateError("Exact cyclotomic coefficient addition overflowed int64");
        }
        return left + right;
    }

    [[nodiscard]] static std::int64_t checked_negate(std::int64_t value) {
        if (value == std::numeric_limits<std::int64_t>::min()) {
            throw QStateError("Exact cyclotomic coefficient negation overflowed int64");
        }
        return -value;
    }

    [[nodiscard]] static std::uint64_t magnitude(std::int64_t value) noexcept {
        if (value >= 0) {
            return static_cast<std::uint64_t>(value);
        }
        return static_cast<std::uint64_t>(-(value + 1)) + 1U;
    }
};

struct DiscretePhaseEdge {
    std::uint64_t key{0U};
    std::uint8_t ticks{0U};

    [[nodiscard]] bool operator<(const DiscretePhaseEdge& other) const noexcept {
        if (key != other.key) {
            return key < other.key;
        }
        return ticks < other.ticks;
    }

    [[nodiscard]] bool operator==(const DiscretePhaseEdge& other) const noexcept {
        return key == other.key && ticks == other.ticks;
    }
};

struct DiscretePhaseDescriptor {
    std::vector<std::uint8_t> local_ticks{};
    std::vector<DiscretePhaseEdge> edges{};
};

struct DiscretePhaseDescriptorLess {
    [[nodiscard]] bool operator()(
        const DiscretePhaseDescriptor& left,
        const DiscretePhaseDescriptor& right) const noexcept {
        if (left.local_ticks != right.local_ticks) {
            return left.local_ticks < right.local_ticks;
        }
        return left.edges < right.edges;
    }
};

struct ExactCompressedPhaseBranch {
    DiscretePhaseDescriptor descriptor{};
    ExactCyclotomic8 coefficient{};
};

struct ExactCompressedPhaseConfig {
    std::size_t max_qubits{1U << 20U};
    std::size_t max_live_branches{1U << 20U};
    std::size_t max_intermediate_branches{1U << 21U};
    std::size_t max_edges_per_branch{4'000'000U};
    std::size_t max_retained_estimated_bytes{1U << 30U};
    std::uint64_t max_abs_coefficient{std::uint64_t{1U} << 60U};
    std::size_t max_materialize_qubits{24U};
};

struct ExactCompressedPhaseStats {
    std::size_t qubits{0U};
    std::size_t live_branches{0U};
    std::size_t hadamard_defects{0U};
    std::size_t max_live_branches{0U};
    std::size_t merged_branches{0U};
    std::size_t exact_cancellations{0U};
    std::size_t extracted_power_of_two_bits{0U};
    std::size_t total_phase_edges{0U};
    std::size_t retained_estimated_bytes{0U};
    std::int64_t scale_half_power{0};
};

struct ExactCyclotomicScaledAmplitude {
    ExactCyclotomic8 coefficient{};
    std::int64_t scale_half_power{0};

    [[nodiscard]] double log2_scale() const noexcept {
        return 0.5 * static_cast<double>(scale_half_power);
    }
};

class ExactCompressedPhaseGraphSum {
public:
    explicit ExactCompressedPhaseGraphSum(
        std::size_t qubit_count,
        ExactCompressedPhaseConfig config = {})
        : qubit_count_(qubit_count), config_(config) {
        validate_configuration();
        ExactCompressedPhaseBranch branch;
        branch.descriptor.local_ticks.assign(qubit_count_, 0U);
        branch.coefficient = ExactCyclotomic8::one();
        branches_.push_back(std::move(branch));
        scale_half_power_ = -static_cast<std::int64_t>(qubit_count_);
        refresh_stats();
        enforce_resources();
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::size_t branch_count() const noexcept { return branches_.size(); }
    [[nodiscard]] const std::vector<ExactCompressedPhaseBranch>& branches() const noexcept {
        return branches_;
    }
    [[nodiscard]] const ExactCompressedPhaseStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const ExactCompressedPhaseConfig& config() const noexcept { return config_; }

    void apply_x(QubitId qubit) {
        validate_qubit(qubit);
        transform([qubit](WorkingBranch& branch) { x(branch, qubit); });
    }

    void apply_y(QubitId qubit) {
        validate_qubit(qubit);
        transform([qubit](WorkingBranch& branch) {
            add_local(branch.descriptor, qubit, 4U);
            x(branch, qubit);
            branch.global_ticks = add_ticks(branch.global_ticks, 2U);
        });
    }

    void apply_z(QubitId qubit) {
        validate_qubit(qubit);
        transform([qubit](WorkingBranch& branch) {
            add_local(branch.descriptor, qubit, 4U);
        });
    }

    void apply_s(QubitId qubit) {
        validate_qubit(qubit);
        transform([qubit](WorkingBranch& branch) {
            add_local(branch.descriptor, qubit, 2U);
        });
    }

    void apply_sdg(QubitId qubit) {
        validate_qubit(qubit);
        transform([qubit](WorkingBranch& branch) {
            add_local(branch.descriptor, qubit, 6U);
        });
    }

    void apply_t(QubitId qubit) {
        validate_qubit(qubit);
        transform([qubit](WorkingBranch& branch) {
            add_local(branch.descriptor, qubit, 1U);
        });
    }

    void apply_tdg(QubitId qubit) {
        validate_qubit(qubit);
        transform([qubit](WorkingBranch& branch) {
            add_local(branch.descriptor, qubit, 7U);
        });
    }

    void apply_cz(QubitId first, QubitId second) {
        apply_controlled_phase_ticks(first, second, 4U);
    }

    void apply_controlled_phase_ticks(
        QubitId first,
        QubitId second,
        std::uint8_t ticks) {
        validate_qubit(first);
        validate_qubit(second);
        if (first == second) {
            throw QStateError("Exact compressed phase controlled phase requires distinct qubits");
        }
        ticks &= 7U;
        transform([first, second, ticks](WorkingBranch& branch) {
            add_edge(branch.descriptor, first, second, ticks);
        });
    }

    void apply_swap(QubitId first, QubitId second) {
        validate_qubit(first);
        validate_qubit(second);
        if (first == second) {
            throw QStateError("Exact compressed phase SWAP requires distinct qubits");
        }
        transform([first, second](WorkingBranch& branch) {
            std::swap(branch.descriptor.local_ticks[first], branch.descriptor.local_ticks[second]);
            for (DiscretePhaseEdge& edge : branch.descriptor.edges) {
                auto [left, right] = decode_edge(edge.key);
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
                edge.key = edge_key(left, right);
            }
            std::sort(branch.descriptor.edges.begin(), branch.descriptor.edges.end());
        });
    }

    void apply_h(QubitId qubit) {
        validate_qubit(qubit);
        if (scale_half_power_ == std::numeric_limits<std::int64_t>::min()) {
            throw QStateError("Exact compressed phase common scale exponent underflowed");
        }

        MergeMap merged;
        std::size_t inputs = 0U;
        for (const ExactCompressedPhaseBranch& stored : branches_) {
            WorkingBranch x_child{0U, stored.descriptor, stored.coefficient};
            x(x_child, qubit);
            merge_one(merged, std::move(x_child));
            ++inputs;
            enforce_intermediate(merged.size());

            WorkingBranch z_child{0U, stored.descriptor, stored.coefficient};
            add_local(z_child.descriptor, qubit, 4U);
            merge_one(merged, std::move(z_child));
            ++inputs;
            enforce_intermediate(merged.size());
        }

        const CompressionResult compressed = finalize(
            merged, inputs, scale_half_power_ - 1);
        commit(compressed, true);
    }

    void apply(const Operation& operation) {
        switch (operation.code) {
            case OperationCode::X: apply_x(operation.first); return;
            case OperationCode::Y: apply_y(operation.first); return;
            case OperationCode::Z: apply_z(operation.first); return;
            case OperationCode::H: apply_h(operation.first); return;
            case OperationCode::S: apply_s(operation.first); return;
            case OperationCode::Sdg: apply_sdg(operation.first); return;
            case OperationCode::T: apply_t(operation.first); return;
            case OperationCode::Tdg: apply_tdg(operation.first); return;
            case OperationCode::Cz: apply_cz(operation.first, operation.second); return;
            case OperationCode::Swap: apply_swap(operation.first, operation.second); return;
            case OperationCode::Rz:
            case OperationCode::Rx:
            case OperationCode::Ry:
            case OperationCode::Cnot:
            case OperationCode::BitFlipTrajectory:
            case OperationCode::PhaseFlipTrajectory:
            case OperationCode::DepolarizingTrajectory:
            case OperationCode::AmplitudeDampingTrajectory:
                throw QStateError(
                    "Operation is outside the exact discrete pi/4 compressed phase contract");
            default:
                throw QStateError("Exact compressed phase sum received an unknown opcode");
        }
    }

    [[nodiscard]] ExactCyclotomicScaledAmplitude exact_amplitude_bits(
        std::span<const std::uint8_t> bits) const {
        validate_bits(bits);
        ExactCyclotomic8 total;
        for (const ExactCompressedPhaseBranch& branch : branches_) {
            ExactCyclotomic8 contribution = branch.coefficient;
            contribution.rotate(phase_ticks(branch.descriptor, bits));
            total.add(contribution);
        }
        return ExactCyclotomicScaledAmplitude{total, scale_half_power_};
    }

    [[nodiscard]] double log2_probability_bits(
        std::span<const std::uint8_t> bits) const {
        const ExactCyclotomicScaledAmplitude exact = exact_amplitude_bits(bits);
        if (exact.coefficient.zero()) {
            return -std::numeric_limits<double>::infinity();
        }
        const QComplex value = exact.coefficient.evaluate();
        const double norm2 = value.norm2();
        if (!std::isfinite(norm2) || norm2 <= 0.0) {
            throw QStateError("Exact compressed phase probability evaluation became invalid");
        }
        return std::log2(norm2) + static_cast<double>(exact.scale_half_power);
    }

    [[nodiscard]] QComplex amplitude_bits(
        std::span<const std::uint8_t> bits) const {
        const ExactCyclotomicScaledAmplitude exact = exact_amplitude_bits(bits);
        const double scale = std::exp2(exact.log2_scale());
        if (scale == 0.0 && !exact.coefficient.zero()) {
            throw QStateError(
                "Exact compressed phase amplitude underflows; use exact_amplitude_bits or log2_probability_bits");
        }
        return exact.coefficient.evaluate() * scale;
    }

    [[nodiscard]] std::vector<QComplex> materialize(
        std::size_t max_qubits = 24U) const {
        const std::size_t limit = std::min(max_qubits, config_.max_materialize_qubits);
        if (qubit_count_ > limit || qubit_count_ >= 63U) {
            throw QStateError("Exact compressed phase materialization exceeds configured qubit cap");
        }
        const std::size_t dimension = std::size_t{1U} << qubit_count_;
        std::vector<QComplex> output(dimension);
        std::vector<std::uint8_t> bits(qubit_count_);
        for (std::size_t basis = 0U; basis < dimension; ++basis) {
            for (std::size_t qubit = 0U; qubit < qubit_count_; ++qubit) {
                bits[qubit] = static_cast<std::uint8_t>((basis >> qubit) & 1U);
            }
            output[basis] = amplitude_bits(bits);
        }
        return output;
    }

private:
    struct WorkingBranch {
        std::uint8_t global_ticks{0U};
        DiscretePhaseDescriptor descriptor{};
        ExactCyclotomic8 coefficient{};
    };

    using MergeMap = std::map<
        DiscretePhaseDescriptor,
        ExactCyclotomic8,
        DiscretePhaseDescriptorLess>;

    struct CompressionResult {
        std::vector<ExactCompressedPhaseBranch> branches{};
        std::int64_t scale_half_power{0};
        std::size_t merged{0U};
        std::size_t cancelled{0U};
        std::size_t extracted_bits{0U};
    };

    std::size_t qubit_count_{0U};
    ExactCompressedPhaseConfig config_{};
    std::vector<ExactCompressedPhaseBranch> branches_{};
    std::int64_t scale_half_power_{0};
    std::size_t hadamard_defects_{0U};
    std::size_t max_live_branches_{1U};
    std::size_t merged_branches_{0U};
    std::size_t exact_cancellations_{0U};
    std::size_t extracted_power_of_two_bits_{0U};
    ExactCompressedPhaseStats stats_{};

    void validate_configuration() const {
        if (qubit_count_ == 0U || qubit_count_ > config_.max_qubits ||
            qubit_count_ > static_cast<std::size_t>(std::numeric_limits<QubitId>::max()) ||
            qubit_count_ > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
            config_.max_live_branches == 0U ||
            config_.max_intermediate_branches < config_.max_live_branches ||
            config_.max_edges_per_branch == 0U ||
            config_.max_retained_estimated_bytes == 0U ||
            config_.max_abs_coefficient == 0U ||
            config_.max_materialize_qubits == 0U) {
            throw QStateError("Exact compressed phase dimensions or configuration are invalid");
        }
    }

    void validate_qubit(QubitId qubit) const {
        if (static_cast<std::size_t>(qubit) >= qubit_count_) {
            throw QStateError("Exact compressed phase qubit is out of range");
        }
    }

    void validate_bits(std::span<const std::uint8_t> bits) const {
        if (bits.size() != qubit_count_) {
            throw QStateError("Exact compressed phase bit-vector length does not match qubit count");
        }
        for (const std::uint8_t bit : bits) {
            if (bit > 1U) {
                throw QStateError("Exact compressed phase basis bits must be zero or one");
            }
        }
    }

    template <typename Mutation>
    void transform(Mutation&& mutation) {
        MergeMap merged;
        for (const ExactCompressedPhaseBranch& stored : branches_) {
            WorkingBranch branch{0U, stored.descriptor, stored.coefficient};
            mutation(branch);
            merge_one(merged, std::move(branch));
            enforce_intermediate(merged.size());
        }
        const CompressionResult compressed = finalize(
            merged, branches_.size(), scale_half_power_);
        commit(compressed, false);
    }

    void merge_one(MergeMap& merged, WorkingBranch branch) const {
        normalize_global(branch);
        if (branch.descriptor.edges.size() > config_.max_edges_per_branch) {
            throw QStateError("Exact compressed phase branch exceeds configured edge cap");
        }
        const auto [iterator, inserted] = merged.emplace(
            std::move(branch.descriptor), branch.coefficient);
        if (!inserted) {
            iterator->second.add(branch.coefficient);
        }
    }

    [[nodiscard]] CompressionResult finalize(
        const MergeMap& merged,
        std::size_t input_count,
        std::int64_t scale_half_power) const {
        CompressionResult result;
        result.scale_half_power = scale_half_power;
        result.merged = input_count > merged.size() ? input_count - merged.size() : 0U;
        result.branches.reserve(merged.size());
        for (const auto& [descriptor, coefficient] : merged) {
            if (coefficient.zero()) {
                ++result.cancelled;
                continue;
            }
            result.branches.push_back(ExactCompressedPhaseBranch{descriptor, coefficient});
        }
        if (result.branches.empty()) {
            throw QStateError("Exact compressed phase cancellation produced a zero state");
        }
        if (result.branches.size() > config_.max_live_branches) {
            throw QStateError("Exact compressed phase live branch count exceeds configured cap");
        }

        const std::size_t common_two = common_power_of_two(result.branches);
        if (common_two != 0U) {
            if (common_two > static_cast<std::size_t>(
                    std::numeric_limits<std::int64_t>::max() / 2)) {
                throw QStateError("Exact compressed phase common scale extraction overflowed");
            }
            const std::int64_t delta = static_cast<std::int64_t>(2U * common_two);
            if (result.scale_half_power > std::numeric_limits<std::int64_t>::max() - delta) {
                throw QStateError("Exact compressed phase common scale exponent overflowed");
            }
            for (ExactCompressedPhaseBranch& branch : result.branches) {
                branch.coefficient.divide_by_two(common_two);
            }
            result.scale_half_power += delta;
            result.extracted_bits = common_two;
        }

        for (const ExactCompressedPhaseBranch& branch : result.branches) {
            if (branch.coefficient.max_abs() > config_.max_abs_coefficient) {
                throw QStateError("Exact compressed phase coefficient exceeds configured exact cap");
            }
        }
        const std::size_t retained = estimated_bytes(result.branches);
        if (retained > config_.max_retained_estimated_bytes) {
            throw QStateError("Exact compressed phase retained-byte estimate exceeds configured cap");
        }
        return result;
    }

    void commit(const CompressionResult& result, bool hadamard) {
        branches_ = result.branches;
        scale_half_power_ = result.scale_half_power;
        merged_branches_ = checked_sum(
            merged_branches_, result.merged,
            "Exact compressed phase merge counter overflowed");
        exact_cancellations_ = checked_sum(
            exact_cancellations_, result.cancelled,
            "Exact compressed phase cancellation counter overflowed");
        extracted_power_of_two_bits_ = checked_sum(
            extracted_power_of_two_bits_, result.extracted_bits,
            "Exact compressed phase extraction counter overflowed");
        if (hadamard) {
            hadamard_defects_ = checked_sum(
                hadamard_defects_, 1U,
                "Exact compressed phase Hadamard counter overflowed");
        }
        max_live_branches_ = std::max(max_live_branches_, branches_.size());
        refresh_stats();
    }

    void refresh_stats() {
        std::size_t total_edges = 0U;
        for (const ExactCompressedPhaseBranch& branch : branches_) {
            total_edges = checked_sum(
                total_edges, branch.descriptor.edges.size(),
                "Exact compressed phase total edge count overflowed");
        }
        stats_ = ExactCompressedPhaseStats{
            qubit_count_,
            branches_.size(),
            hadamard_defects_,
            max_live_branches_,
            merged_branches_,
            exact_cancellations_,
            extracted_power_of_two_bits_,
            total_edges,
            estimated_bytes(branches_),
            scale_half_power_,
        };
    }

    void enforce_resources() const {
        if (branches_.size() > config_.max_live_branches ||
            stats_.retained_estimated_bytes > config_.max_retained_estimated_bytes) {
            throw QStateError("Exact compressed phase initial state exceeds configured resources");
        }
    }

    void enforce_intermediate(std::size_t branches) const {
        if (branches > config_.max_intermediate_branches) {
            throw QStateError("Exact compressed phase intermediate branch count exceeds configured cap");
        }
    }

    static void normalize_global(WorkingBranch& branch) {
        branch.coefficient.rotate(branch.global_ticks);
        branch.global_ticks = 0U;
    }

    static void x(WorkingBranch& branch, QubitId qubit) {
        const std::uint8_t local = branch.descriptor.local_ticks[qubit];
        branch.global_ticks = add_ticks(branch.global_ticks, local);
        branch.descriptor.local_ticks[qubit] = negate_ticks(local);
        for (DiscretePhaseEdge& edge : branch.descriptor.edges) {
            const auto [first, second] = decode_edge(edge.key);
            if (first != qubit && second != qubit) {
                continue;
            }
            const QubitId other = first == qubit ? second : first;
            branch.descriptor.local_ticks[other] = add_ticks(
                branch.descriptor.local_ticks[other], edge.ticks);
            edge.ticks = negate_ticks(edge.ticks);
        }
    }

    static void add_local(
        DiscretePhaseDescriptor& descriptor,
        QubitId qubit,
        std::uint8_t ticks) noexcept {
        descriptor.local_ticks[qubit] = add_ticks(descriptor.local_ticks[qubit], ticks);
    }

    static void add_edge(
        DiscretePhaseDescriptor& descriptor,
        QubitId first,
        QubitId second,
        std::uint8_t ticks) {
        ticks &= 7U;
        if (ticks == 0U) {
            return;
        }
        const std::uint64_t key = edge_key(first, second);
        const auto found = std::lower_bound(
            descriptor.edges.begin(), descriptor.edges.end(), key,
            [](const DiscretePhaseEdge& edge, std::uint64_t value) {
                return edge.key < value;
            });
        if (found == descriptor.edges.end() || found->key != key) {
            descriptor.edges.insert(found, DiscretePhaseEdge{key, ticks});
            return;
        }
        const std::uint8_t next = add_ticks(found->ticks, ticks);
        if (next == 0U) {
            descriptor.edges.erase(found);
        } else {
            found->ticks = next;
        }
    }

    [[nodiscard]] static std::uint8_t phase_ticks(
        const DiscretePhaseDescriptor& descriptor,
        std::span<const std::uint8_t> bits) noexcept {
        std::uint8_t phase = 0U;
        for (std::size_t qubit = 0U; qubit < bits.size(); ++qubit) {
            if (bits[qubit] != 0U) {
                phase = add_ticks(phase, descriptor.local_ticks[qubit]);
            }
        }
        for (const DiscretePhaseEdge& edge : descriptor.edges) {
            const auto [first, second] = decode_edge(edge.key);
            if (bits[first] != 0U && bits[second] != 0U) {
                phase = add_ticks(phase, edge.ticks);
            }
        }
        return phase;
    }

    [[nodiscard]] static std::size_t common_power_of_two(
        const std::vector<ExactCompressedPhaseBranch>& branches) noexcept {
        std::size_t common = 64U;
        bool seen = false;
        for (const ExactCompressedPhaseBranch& branch : branches) {
            for (const std::int64_t value : branch.coefficient.values) {
                if (value == 0) {
                    continue;
                }
                seen = true;
                std::uint64_t magnitude = value >= 0
                    ? static_cast<std::uint64_t>(value)
                    : static_cast<std::uint64_t>(-(value + 1)) + 1U;
                std::size_t power = 0U;
                while ((magnitude & 1U) == 0U) {
                    ++power;
                    magnitude >>= 1U;
                }
                common = std::min(common, power);
            }
        }
        return seen ? common : 0U;
    }

    [[nodiscard]] std::size_t estimated_bytes(
        const std::vector<ExactCompressedPhaseBranch>& branches) const {
        std::size_t total = sizeof(*this);
        total = checked_sum(
            total,
            checked_product(branches.capacity(), sizeof(ExactCompressedPhaseBranch),
                "Exact compressed phase branch vector storage overflowed"),
            "Exact compressed phase retained storage overflowed");
        for (const ExactCompressedPhaseBranch& branch : branches) {
            total = checked_sum(
                total,
                checked_product(branch.descriptor.local_ticks.capacity(), sizeof(std::uint8_t),
                    "Exact compressed phase local storage overflowed"),
                "Exact compressed phase retained storage overflowed");
            total = checked_sum(
                total,
                checked_product(branch.descriptor.edges.capacity(), sizeof(DiscretePhaseEdge),
                    "Exact compressed phase edge storage overflowed"),
                "Exact compressed phase retained storage overflowed");
        }
        return total;
    }

    [[nodiscard]] static std::uint8_t add_ticks(
        std::uint8_t left,
        std::uint8_t right) noexcept {
        return static_cast<std::uint8_t>((left + right) & 7U);
    }

    [[nodiscard]] static std::uint8_t negate_ticks(std::uint8_t ticks) noexcept {
        return static_cast<std::uint8_t>((8U - (ticks & 7U)) & 7U);
    }

    [[nodiscard]] static std::uint64_t edge_key(QubitId first, QubitId second) noexcept {
        if (second < first) {
            std::swap(first, second);
        }
        return (static_cast<std::uint64_t>(first) << 32U) |
               static_cast<std::uint64_t>(second);
    }

    [[nodiscard]] static std::pair<QubitId, QubitId> decode_edge(
        std::uint64_t key) noexcept {
        return {
            static_cast<QubitId>(key >> 32U),
            static_cast<QubitId>(key & 0xFFFFFFFFULL),
        };
    }

    [[nodiscard]] static std::size_t checked_product(
        std::size_t left,
        std::size_t right,
        const char* message) {
        if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
            throw QStateError(message);
        }
        return left * right;
    }

    [[nodiscard]] static std::size_t checked_sum(
        std::size_t left,
        std::size_t right,
        const char* message) {
        if (right > std::numeric_limits<std::size_t>::max() - left) {
            throw QStateError(message);
        }
        return left + right;
    }
};

}  // namespace qubit

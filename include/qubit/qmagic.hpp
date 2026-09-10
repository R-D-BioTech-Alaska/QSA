#pragma once

#include "qubit/qpauli_support.hpp"
#include "qubit/qplan.hpp"
#include "qubit/qstabilizer.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <utility>
#include <vector>

namespace qubit {

struct LowMagicCyclotomic8 {
    std::array<std::int64_t, 4> values{0, 0, 0, 0};

    [[nodiscard]] static LowMagicCyclotomic8 one() noexcept {
        LowMagicCyclotomic8 result;
        result.values[0] = 1;
        return result;
    }

    [[nodiscard]] bool zero() const noexcept {
        return values[0] == 0 && values[1] == 0 && values[2] == 0 && values[3] == 0;
    }

    void add(const LowMagicCyclotomic8& other) {
        for (std::size_t index = 0U; index < values.size(); ++index) {
            values[index] = checked_add(values[index], other.values[index]);
        }
    }

    void subtract(const LowMagicCyclotomic8& other) {
        for (std::size_t index = 0U; index < values.size(); ++index) {
            values[index] = checked_subtract(values[index], other.values[index]);
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
    [[nodiscard]] static std::int64_t checked_add(
        std::int64_t left,
        std::int64_t right) {
        if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
            (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right)) {
            throw QStateError("Low-magic cyclotomic coefficient addition overflowed int64");
        }
        return left + right;
    }

    [[nodiscard]] static std::int64_t checked_subtract(
        std::int64_t left,
        std::int64_t right) {
        if ((right > 0 && left < std::numeric_limits<std::int64_t>::min() + right) ||
            (right < 0 && left > std::numeric_limits<std::int64_t>::max() + right)) {
            throw QStateError("Low-magic cyclotomic coefficient subtraction overflowed int64");
        }
        return left - right;
    }

    [[nodiscard]] static std::int64_t checked_negate(std::int64_t value) {
        if (value == std::numeric_limits<std::int64_t>::min()) {
            throw QStateError("Low-magic cyclotomic coefficient negation overflowed int64");
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

struct LowMagicPauliKey {
    std::vector<std::uint64_t> x{};
    std::vector<std::uint64_t> z{};
};

struct LowMagicPauliKeyLess {
    [[nodiscard]] bool operator()(
        const LowMagicPauliKey& left,
        const LowMagicPauliKey& right) const noexcept {
        if (left.x != right.x) {
            return left.x < right.x;
        }
        return left.z < right.z;
    }
};

struct ExactLowMagicConfig {
    std::size_t max_qubits{1U << 20U};
    std::size_t max_live_branches{1U << 20U};
    std::size_t max_intermediate_branches{1U << 21U};
    std::size_t max_retained_estimated_bytes{1U << 30U};
    std::uint64_t max_abs_coefficient{std::uint64_t{1U} << 60U};
    std::size_t max_observable_terms{1U << 16U};
    std::size_t max_expectation_word_ops{200'000'000U};
    double imaginary_tolerance{1e-10};
};

struct ExactLowMagicStats {
    std::size_t qubits{0U};
    std::size_t live_branches{0U};
    std::size_t t_defects{0U};
    std::size_t max_live_branches{0U};
    std::size_t merged_branches{0U};
    std::size_t exact_cancellations{0U};
    std::size_t extracted_power_of_two_bits{0U};
    std::size_t base_tableau_bytes{0U};
    std::size_t retained_estimated_bytes{0U};
    std::int64_t scale_power_two{0};
};

class ExactLowMagicStabilizerSum {
public:
    explicit ExactLowMagicStabilizerSum(
        std::size_t qubit_count,
        ExactLowMagicConfig config = {})
        : qubit_count_(qubit_count),
          word_count_((qubit_count + 63U) / 64U),
          config_(config),
          base_(qubit_count) {
        initialize();
    }

    explicit ExactLowMagicStabilizerSum(
        StabilizerState base,
        ExactLowMagicConfig config = {})
        : qubit_count_(base.qubit_count()),
          word_count_((base.qubit_count() + 63U) / 64U),
          config_(config),
          base_(std::move(base)) {
        initialize();
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::size_t branch_count() const noexcept { return branches_.size(); }
    [[nodiscard]] const ExactLowMagicStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const StabilizerState& base_state() const noexcept { return base_; }

    void apply(const Operation& operation) {
        switch (operation.code) {
            case OperationCode::T:
                apply_t_like(operation.first, 1U);
                return;
            case OperationCode::Tdg:
                apply_t_like(operation.first, 7U);
                return;
            case OperationCode::X:
            case OperationCode::Y:
            case OperationCode::Z:
            case OperationCode::H:
            case OperationCode::S:
            case OperationCode::Sdg:
            case OperationCode::Cnot:
            case OperationCode::Cz:
            case OperationCode::Swap:
                apply_clifford(operation);
                return;
            case OperationCode::Rx:
            case OperationCode::Ry:
            case OperationCode::Rz:
            case OperationCode::BitFlipTrajectory:
            case OperationCode::PhaseFlipTrajectory:
            case OperationCode::DepolarizingTrajectory:
            case OperationCode::AmplitudeDampingTrajectory:
                throw QStateError(
                    "Operation is outside the exact bounded-magic Clifford+T contract");
            default:
                throw QStateError("Low-magic stabilizer sum received an unknown opcode");
        }
    }

    void apply_many(std::span<const Operation> operations) {
        for (const Operation& operation : operations) {
            apply(operation);
        }
    }

    [[nodiscard]] double expectation_pauli(
        std::span<const PauliSupportTerm> observable) const {
        const PauliFrame target = make_observable(observable);
        preflight_expectation();

        double numerator = 0.0;
        for (std::size_t left = 0U; left < branches_.size(); ++left) {
            const QComplex left_coefficient = branches_[left].coefficient.evaluate();
            const PauliFrame left_frame{branches_[left].key, 0U};
            const PauliFrame left_dagger = dagger(left_frame);

            for (std::size_t right = left; right < branches_.size(); ++right) {
                const QComplex right_coefficient = branches_[right].coefficient.evaluate();
                const PauliFrame right_frame{branches_[right].key, 0U};
                const PauliFrame product = multiply(
                    multiply(left_dagger, target), right_frame);
                const QComplex base_value = base_pauli_expectation(product);
                const QComplex value =
                    left_coefficient.conjugate() * right_coefficient * base_value;
                if (left == right) {
                    const double scale = 1.0 + std::abs(value.re);
                    if (std::abs(value.im) > config_.imaginary_tolerance * scale) {
                        throw QStateError(
                            "Low-magic diagonal Pauli expectation developed an imaginary part");
                    }
                    numerator += value.re;
                } else {
                    numerator += 2.0 * value.re;
                }
            }
        }

        const double scale = std::exp2(2.0 * static_cast<double>(scale_power_two_));
        if (scale == 0.0 && numerator != 0.0) {
            throw QStateError(
                "Low-magic observable scale underflows double precision");
        }
        const double result = numerator * scale;
        if (!std::isfinite(result)) {
            throw QStateError("Low-magic Pauli expectation became non-finite");
        }
        return result;
    }

    [[nodiscard]] double norm_squared() const {
        return expectation_pauli({});
    }

    [[nodiscard]] double probability_one(QubitId qubit) const {
        validate_qubit(qubit);
        const std::array<PauliSupportTerm, 1> observable{{{qubit, 'Z'}}};
        const double z = expectation_pauli(observable);
        const double probability = 0.5 * (1.0 - z);
        const double tolerance = 16.0 * config_.imaginary_tolerance;
        if (probability < -tolerance || probability > 1.0 + tolerance) {
            throw QStateError("Low-magic probability escaped the physical interval");
        }
        return std::clamp(probability, 0.0, 1.0);
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        return stats_.retained_estimated_bytes;
    }

private:
    struct Branch {
        LowMagicPauliKey key{};
        LowMagicCyclotomic8 coefficient{};
    };

    struct PauliFrame {
        LowMagicPauliKey key{};
        std::uint8_t phase{0U};
    };

    using MergeMap = std::map<
        LowMagicPauliKey,
        LowMagicCyclotomic8,
        LowMagicPauliKeyLess>;

    struct CompressionResult {
        std::vector<Branch> branches{};
        std::int64_t scale_power_two{0};
        std::size_t merged{0U};
        std::size_t cancelled{0U};
        std::size_t extracted_bits{0U};
    };

    std::size_t qubit_count_{0U};
    std::size_t word_count_{0U};
    ExactLowMagicConfig config_{};
    StabilizerState base_;
    std::vector<Branch> branches_{};
    std::int64_t scale_power_two_{0};
    std::size_t t_defects_{0U};
    std::size_t max_live_branches_{1U};
    std::size_t merged_branches_{0U};
    std::size_t exact_cancellations_{0U};
    std::size_t extracted_power_of_two_bits_{0U};
    ExactLowMagicStats stats_{};

    void initialize() {
        if (qubit_count_ == 0U || qubit_count_ > config_.max_qubits ||
            config_.max_live_branches == 0U ||
            config_.max_intermediate_branches < config_.max_live_branches ||
            config_.max_retained_estimated_bytes == 0U ||
            config_.max_abs_coefficient == 0U ||
            config_.max_observable_terms == 0U ||
            config_.max_expectation_word_ops == 0U ||
            !std::isfinite(config_.imaginary_tolerance) ||
            config_.imaginary_tolerance <= 0.0) {
            throw QStateError("Low-magic stabilizer dimensions or configuration are invalid");
        }
        std::string reason;
        if (!base_.validate(&reason)) {
            throw QStateError("Low-magic base stabilizer is invalid: " + reason);
        }
        Branch branch;
        branch.key.x.assign(word_count_, 0U);
        branch.key.z.assign(word_count_, 0U);
        branch.coefficient = LowMagicCyclotomic8::one();
        branches_.push_back(std::move(branch));
        refresh_stats();
        if (stats_.retained_estimated_bytes > config_.max_retained_estimated_bytes) {
            throw QStateError("Low-magic initial state exceeds configured retained-byte cap");
        }
    }

    void validate_qubit(QubitId qubit) const {
        if (static_cast<std::size_t>(qubit) >= qubit_count_) {
            throw QStateError("Low-magic stabilizer qubit is out of range");
        }
    }

    void validate_clifford(const Operation& operation) const {
        validate_qubit(operation.first);
        if (operation.code == OperationCode::Cnot ||
            operation.code == OperationCode::Cz ||
            operation.code == OperationCode::Swap) {
            validate_qubit(operation.second);
            if (operation.first == operation.second) {
                throw QStateError("Low-magic two-qubit Clifford requires distinct qubits");
            }
        }
    }

    void apply_clifford(const Operation& operation) {
        validate_clifford(operation);
        for (Branch& branch : branches_) {
            std::uint8_t phase = 0U;
            conjugate(branch.key, phase, operation);
            branch.coefficient.rotate(static_cast<std::uint8_t>(2U * phase));
            if (branch.coefficient.max_abs() > config_.max_abs_coefficient) {
                throw QStateError("Low-magic coefficient exceeds configured exact cap");
            }
        }
        apply_base(operation);
        refresh_stats();
    }

    void apply_base(const Operation& operation) {
        switch (operation.code) {
            case OperationCode::X: base_.apply_x(operation.first); return;
            case OperationCode::Y: base_.apply_y(operation.first); return;
            case OperationCode::Z: base_.apply_z(operation.first); return;
            case OperationCode::H: base_.apply_h(operation.first); return;
            case OperationCode::S: base_.apply_s(operation.first); return;
            case OperationCode::Sdg: base_.apply_sdg(operation.first); return;
            case OperationCode::Cnot:
                base_.apply_cnot(operation.first, operation.second);
                return;
            case OperationCode::Cz:
                base_.apply_cz(operation.first, operation.second);
                return;
            case OperationCode::Swap:
                base_.apply_swap(operation.first, operation.second);
                return;
            default:
                throw QStateError("Low-magic base received a non-Clifford operation");
        }
    }

    void apply_t_like(QubitId qubit, std::uint8_t omega_ticks) {
        validate_qubit(qubit);
        if (scale_power_two_ == std::numeric_limits<std::int64_t>::min()) {
            throw QStateError("Low-magic common denominator exponent underflowed");
        }
        if (branches_.size() > config_.max_intermediate_branches / 2U) {
            throw QStateError("Low-magic T expansion exceeds intermediate branch cap");
        }

        MergeMap merged;
        const std::size_t input_count = checked_product(
            branches_.size(), 2U,
            "Low-magic T expansion branch count overflowed");
        for (const Branch& branch : branches_) {
            LowMagicCyclotomic8 rotated = branch.coefficient;
            rotated.rotate(omega_ticks);

            LowMagicCyclotomic8 same = branch.coefficient;
            same.add(rotated);
            merge_one(merged, branch.key, same, 0U);
            enforce_intermediate(merged.size());

            LowMagicCyclotomic8 z_coefficient = branch.coefficient;
            z_coefficient.subtract(rotated);
            LowMagicPauliKey z_key = branch.key;
            std::uint8_t phase = 0U;
            left_multiply_z(z_key, phase, qubit);
            merge_one(merged, std::move(z_key), z_coefficient, phase);
            enforce_intermediate(merged.size());
        }

        CompressionResult compressed = finalize(
            merged, input_count, scale_power_two_ - 1);
        branches_ = std::move(compressed.branches);
        scale_power_two_ = compressed.scale_power_two;
        merged_branches_ = checked_sum(
            merged_branches_, compressed.merged,
            "Low-magic merge counter overflowed");
        exact_cancellations_ = checked_sum(
            exact_cancellations_, compressed.cancelled,
            "Low-magic cancellation counter overflowed");
        extracted_power_of_two_bits_ = checked_sum(
            extracted_power_of_two_bits_, compressed.extracted_bits,
            "Low-magic extraction counter overflowed");
        t_defects_ = checked_sum(
            t_defects_, 1U,
            "Low-magic T-defect counter overflowed");
        max_live_branches_ = std::max(max_live_branches_, branches_.size());
        refresh_stats();
    }

    void merge_one(
        MergeMap& merged,
        LowMagicPauliKey key,
        LowMagicCyclotomic8 coefficient,
        std::uint8_t phase) const {
        coefficient.rotate(static_cast<std::uint8_t>(2U * (phase & 3U)));
        const auto [iterator, inserted] = merged.emplace(
            std::move(key), coefficient);
        if (!inserted) {
            iterator->second.add(coefficient);
        }
    }

    [[nodiscard]] CompressionResult finalize(
        const MergeMap& merged,
        std::size_t input_count,
        std::int64_t scale_power_two) const {
        CompressionResult result;
        result.scale_power_two = scale_power_two;
        result.merged = input_count > merged.size() ? input_count - merged.size() : 0U;
        result.branches.reserve(merged.size());
        for (const auto& [key, coefficient] : merged) {
            if (coefficient.zero()) {
                ++result.cancelled;
                continue;
            }
            result.branches.push_back(Branch{key, coefficient});
        }
        if (result.branches.empty()) {
            throw QStateError("Low-magic exact cancellation produced a zero state");
        }
        if (result.branches.size() > config_.max_live_branches) {
            throw QStateError("Low-magic live branch count exceeds configured cap");
        }

        const std::size_t common_two = common_power_of_two(result.branches);
        if (common_two != 0U) {
            if (common_two > static_cast<std::size_t>(
                    std::numeric_limits<std::int64_t>::max())) {
                throw QStateError("Low-magic common scale extraction overflowed");
            }
            const std::int64_t delta = static_cast<std::int64_t>(common_two);
            if (result.scale_power_two > std::numeric_limits<std::int64_t>::max() - delta) {
                throw QStateError("Low-magic common denominator exponent overflowed");
            }
            for (Branch& branch : result.branches) {
                branch.coefficient.divide_by_two(common_two);
            }
            result.scale_power_two += delta;
            result.extracted_bits = common_two;
        }

        for (const Branch& branch : result.branches) {
            if (branch.coefficient.max_abs() > config_.max_abs_coefficient) {
                throw QStateError("Low-magic coefficient exceeds configured exact cap");
            }
        }
        const std::size_t retained = estimated_bytes(result.branches);
        if (retained > config_.max_retained_estimated_bytes) {
            throw QStateError("Low-magic retained-byte estimate exceeds configured cap");
        }
        return result;
    }

    [[nodiscard]] PauliFrame make_observable(
        std::span<const PauliSupportTerm> observable) const {
        if (observable.size() > config_.max_observable_terms) {
            throw QStateError("Low-magic Pauli observable exceeds configured term cap");
        }
        std::vector<PauliSupportTerm> normalized(observable.begin(), observable.end());
        for (PauliSupportTerm& term : normalized) {
            if (static_cast<std::size_t>(term.qubit) >= qubit_count_) {
                throw QStateError("Low-magic Pauli observable qubit is out of range");
            }
            switch (term.axis) {
                case 'x': term.axis = 'X'; break;
                case 'y': term.axis = 'Y'; break;
                case 'z': term.axis = 'Z'; break;
                case 'i': term.axis = 'I'; break;
                case 'X': case 'Y': case 'Z': case 'I': break;
                default:
                    throw QStateError("Low-magic Pauli observable axis is invalid");
            }
        }
        normalized.erase(
            std::remove_if(
                normalized.begin(), normalized.end(),
                [](const PauliSupportTerm& term) { return term.axis == 'I'; }),
            normalized.end());
        std::sort(
            normalized.begin(), normalized.end(),
            [](const PauliSupportTerm& left, const PauliSupportTerm& right) {
                return left.qubit < right.qubit;
            });
        for (std::size_t index = 1U; index < normalized.size(); ++index) {
            if (normalized[index - 1U].qubit == normalized[index].qubit) {
                throw QStateError("Low-magic Pauli observable contains a duplicate qubit");
            }
        }

        PauliFrame frame;
        frame.key.x.assign(word_count_, 0U);
        frame.key.z.assign(word_count_, 0U);
        for (const PauliSupportTerm& term : normalized) {
            switch (term.axis) {
                case 'X': set(frame.key.x, term.qubit); break;
                case 'Z': set(frame.key.z, term.qubit); break;
                case 'Y':
                    set(frame.key.x, term.qubit);
                    set(frame.key.z, term.qubit);
                    frame.phase = phase_mod4(static_cast<int>(frame.phase) + 1);
                    break;
                default:
                    throw QStateError("Low-magic normalized Pauli observable is invalid");
            }
        }
        return frame;
    }

    void preflight_expectation() const {
        const std::size_t pairs = checked_product(
            branches_.size(), branches_.size() + 1U,
            "Low-magic expectation pair count overflowed") / 2U;
        const std::size_t row_words = checked_product(
            qubit_count_, word_count_,
            "Low-magic expectation tableau work overflowed");
        const std::size_t work = checked_product(
            pairs, row_words,
            "Low-magic expectation work estimate overflowed");
        if (work > config_.max_expectation_word_ops) {
            throw QStateError("Low-magic Pauli expectation exceeds configured work cap");
        }
    }

    [[nodiscard]] QComplex base_pauli_expectation(const PauliFrame& frame) const {
        for (std::size_t generator = 0U; generator < qubit_count_; ++generator) {
            if (anticommutes_with_row(frame.key, qubit_count_ + generator)) {
                return {};
            }
        }

        std::vector<std::uint64_t> scratch_x(word_count_, 0U);
        std::vector<std::uint64_t> scratch_z(word_count_, 0U);
        std::uint8_t scratch_phase = 0U;
        for (std::size_t generator = 0U; generator < qubit_count_; ++generator) {
            if (anticommutes_with_row(frame.key, generator)) {
                base_.multiply_scratch(
                    scratch_x,
                    scratch_z,
                    scratch_phase,
                    qubit_count_ + generator);
            }
        }
        if (scratch_x != frame.key.x || scratch_z != frame.key.z) {
            throw QStateError("Low-magic stabilizer membership reconstruction failed");
        }
        const std::uint8_t phase = phase_mod4(
            static_cast<int>(frame.phase) - static_cast<int>(scratch_phase));
        switch (phase) {
            case 0U: return {1.0, 0.0};
            case 1U: return {0.0, 1.0};
            case 2U: return {-1.0, 0.0};
            case 3U: return {0.0, -1.0};
            default: return {};
        }
    }

    [[nodiscard]] bool anticommutes_with_row(
        const LowMagicPauliKey& frame,
        std::size_t row) const noexcept {
        unsigned parity = 0U;
        for (std::size_t word = 0U; word < word_count_; ++word) {
            parity ^= static_cast<unsigned>(
                std::popcount(frame.x[word] & base_.z_[base_.offset(row, word)]) & 1U);
            parity ^= static_cast<unsigned>(
                std::popcount(frame.z[word] & base_.x_[base_.offset(row, word)]) & 1U);
        }
        return parity != 0U;
    }

    [[nodiscard]] static PauliFrame dagger(const PauliFrame& input) {
        PauliFrame result = input;
        unsigned parity = 0U;
        for (std::size_t word = 0U; word < input.key.x.size(); ++word) {
            parity ^= static_cast<unsigned>(
                std::popcount(input.key.x[word] & input.key.z[word]) & 1U);
        }
        result.phase = phase_mod4(
            -static_cast<int>(input.phase) + static_cast<int>(2U * parity));
        return result;
    }

    [[nodiscard]] static PauliFrame multiply(
        const PauliFrame& left,
        const PauliFrame& right) {
        if (left.key.x.size() != right.key.x.size() ||
            left.key.z.size() != right.key.z.size()) {
            throw QStateError("Low-magic Pauli frame widths differ");
        }
        PauliFrame result;
        result.key.x.resize(left.key.x.size());
        result.key.z.resize(left.key.z.size());
        unsigned parity = 0U;
        for (std::size_t word = 0U; word < left.key.x.size(); ++word) {
            parity ^= static_cast<unsigned>(
                std::popcount(left.key.z[word] & right.key.x[word]) & 1U);
            result.key.x[word] = left.key.x[word] ^ right.key.x[word];
            result.key.z[word] = left.key.z[word] ^ right.key.z[word];
        }
        result.phase = phase_mod4(
            static_cast<int>(left.phase) + static_cast<int>(right.phase) +
            static_cast<int>(2U * parity));
        return result;
    }

    static void conjugate(
        LowMagicPauliKey& key,
        std::uint8_t& phase,
        const Operation& operation) {
        const QubitId first = operation.first;
        switch (operation.code) {
            case OperationCode::X:
                if (get(key.z, first)) {
                    phase = phase_mod4(static_cast<int>(phase) + 2);
                }
                return;
            case OperationCode::Y:
                if (get(key.x, first) != get(key.z, first)) {
                    phase = phase_mod4(static_cast<int>(phase) + 2);
                }
                return;
            case OperationCode::Z:
                if (get(key.x, first)) {
                    phase = phase_mod4(static_cast<int>(phase) + 2);
                }
                return;
            case OperationCode::H: conjugate_h(key, phase, first); return;
            case OperationCode::S: conjugate_s(key, phase, first, false); return;
            case OperationCode::Sdg: conjugate_s(key, phase, first, true); return;
            case OperationCode::Cnot:
                conjugate_cnot(key, phase, operation.first, operation.second);
                return;
            case OperationCode::Cz:
                conjugate_h(key, phase, operation.second);
                conjugate_cnot(key, phase, operation.first, operation.second);
                conjugate_h(key, phase, operation.second);
                return;
            case OperationCode::Swap:
                swap_bits(key.x, operation.first, operation.second);
                swap_bits(key.z, operation.first, operation.second);
                return;
            default:
                throw QStateError("Low-magic Pauli frame received a non-Clifford operation");
        }
    }

    static void conjugate_h(
        LowMagicPauliKey& key,
        std::uint8_t& phase,
        QubitId qubit) {
        const bool x = get(key.x, qubit);
        const bool z = get(key.z, qubit);
        if (x && z) {
            phase = phase_mod4(static_cast<int>(phase) + 2);
        }
        assign(key.x, qubit, z);
        assign(key.z, qubit, x);
    }

    static void conjugate_s(
        LowMagicPauliKey& key,
        std::uint8_t& phase,
        QubitId qubit,
        bool inverse) {
        if (!get(key.x, qubit)) {
            return;
        }
        phase = phase_mod4(
            static_cast<int>(phase) + (inverse ? 3 : 1));
        toggle(key.z, qubit);
    }

    static void conjugate_cnot(
        LowMagicPauliKey& key,
        std::uint8_t& phase,
        QubitId control,
        QubitId target) {
        const bool xc = get(key.x, control);
        const bool zc = get(key.z, control);
        const bool xt = get(key.x, target);
        const bool zt = get(key.z, target);
        const int old_xz = static_cast<int>(xc && zc) + static_cast<int>(xt && zt);
        const bool next_xt = xt != xc;
        const bool next_zc = zc != zt;
        const int next_xz = static_cast<int>(xc && next_zc) +
                            static_cast<int>(next_xt && zt);
        const bool sign_flip = xc && zt && (xt == zc);
        phase = phase_mod4(
            static_cast<int>(phase) + next_xz - old_xz +
            (sign_flip ? 2 : 0));
        assign(key.x, target, next_xt);
        assign(key.z, control, next_zc);
    }

    static void left_multiply_z(
        LowMagicPauliKey& key,
        std::uint8_t& phase,
        QubitId qubit) {
        if (get(key.x, qubit)) {
            phase = phase_mod4(static_cast<int>(phase) + 2);
        }
        toggle(key.z, qubit);
    }

    static void swap_bits(
        std::vector<std::uint64_t>& words,
        QubitId first,
        QubitId second) {
        const bool first_value = get(words, first);
        const bool second_value = get(words, second);
        assign(words, first, second_value);
        assign(words, second, first_value);
    }

    [[nodiscard]] static bool get(
        const std::vector<std::uint64_t>& words,
        QubitId qubit) noexcept {
        const std::size_t word = static_cast<std::size_t>(qubit) >> 6U;
        const std::uint64_t mask = std::uint64_t{1U} << (qubit & 63U);
        return (words[word] & mask) != 0U;
    }

    static void set(std::vector<std::uint64_t>& words, QubitId qubit) noexcept {
        const std::size_t word = static_cast<std::size_t>(qubit) >> 6U;
        words[word] |= std::uint64_t{1U} << (qubit & 63U);
    }

    static void toggle(std::vector<std::uint64_t>& words, QubitId qubit) noexcept {
        const std::size_t word = static_cast<std::size_t>(qubit) >> 6U;
        words[word] ^= std::uint64_t{1U} << (qubit & 63U);
    }

    static void assign(
        std::vector<std::uint64_t>& words,
        QubitId qubit,
        bool value) noexcept {
        const std::size_t word = static_cast<std::size_t>(qubit) >> 6U;
        const std::uint64_t mask = std::uint64_t{1U} << (qubit & 63U);
        if (value) {
            words[word] |= mask;
        } else {
            words[word] &= ~mask;
        }
    }

    [[nodiscard]] static std::uint8_t phase_mod4(int value) noexcept {
        value %= 4;
        if (value < 0) {
            value += 4;
        }
        return static_cast<std::uint8_t>(value);
    }

    void enforce_intermediate(std::size_t branches) const {
        if (branches > config_.max_intermediate_branches) {
            throw QStateError("Low-magic intermediate branch count exceeds configured cap");
        }
    }

    [[nodiscard]] static std::size_t common_power_of_two(
        const std::vector<Branch>& branches) noexcept {
        std::size_t common = 64U;
        bool seen = false;
        for (const Branch& branch : branches) {
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
        const std::vector<Branch>& branches) const {
        std::size_t total = sizeof(*this);
        total = checked_sum(
            total, base_.estimated_bytes(),
            "Low-magic retained storage overflowed");
        total = checked_sum(
            total,
            checked_product(branches.size(), sizeof(Branch),
                "Low-magic branch storage overflowed"),
            "Low-magic retained storage overflowed");
        for (const Branch& branch : branches) {
            total = checked_sum(
                total,
                checked_product(branch.key.x.size(), sizeof(std::uint64_t),
                    "Low-magic X-frame storage overflowed"),
                "Low-magic retained storage overflowed");
            total = checked_sum(
                total,
                checked_product(branch.key.z.size(), sizeof(std::uint64_t),
                    "Low-magic Z-frame storage overflowed"),
                "Low-magic retained storage overflowed");
        }
        return total;
    }

    void refresh_stats() {
        stats_ = ExactLowMagicStats{
            qubit_count_,
            branches_.size(),
            t_defects_,
            max_live_branches_,
            merged_branches_,
            exact_cancellations_,
            extracted_power_of_two_bits_,
            base_.estimated_bytes(),
            estimated_bytes(branches_),
            scale_power_two_,
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

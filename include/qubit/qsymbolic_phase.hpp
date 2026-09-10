#pragma once

#include "qubit/qplan.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <span>
#include <utility>
#include <vector>

namespace qubit {

using SymbolicPhaseId = std::uint32_t;

struct SymbolicPhaseKey {
    std::uint8_t eighths{0U};
    std::vector<std::pair<SymbolicPhaseId, std::int64_t>> terms{};

    [[nodiscard]] bool operator<(const SymbolicPhaseKey& other) const noexcept {
        if (eighths != other.eighths) {
            return eighths < other.eighths;
        }
        return terms < other.terms;
    }

    [[nodiscard]] bool operator==(const SymbolicPhaseKey& other) const noexcept = default;
};

struct SymbolicLinearPhase {
    std::uint8_t eighths{0U};
    std::map<SymbolicPhaseId, std::int64_t> terms{};

    [[nodiscard]] bool zero() const noexcept {
        return eighths == 0U && terms.empty();
    }
};

struct SymbolicPhaseLocal {
    QubitId qubit{0U};
    SymbolicPhaseKey phase{};

    [[nodiscard]] bool operator<(const SymbolicPhaseLocal& other) const noexcept {
        if (qubit != other.qubit) {
            return qubit < other.qubit;
        }
        return phase < other.phase;
    }

    [[nodiscard]] bool operator==(const SymbolicPhaseLocal& other) const noexcept = default;
};

struct SymbolicPhaseEdge {
    QubitId first{0U};
    QubitId second{0U};
    SymbolicPhaseKey phase{};

    [[nodiscard]] bool operator<(const SymbolicPhaseEdge& other) const noexcept {
        if (first != other.first) {
            return first < other.first;
        }
        if (second != other.second) {
            return second < other.second;
        }
        return phase < other.phase;
    }

    [[nodiscard]] bool operator==(const SymbolicPhaseEdge& other) const noexcept = default;
};

struct SymbolicPhaseGraphKey {
    std::vector<SymbolicPhaseLocal> local{};
    std::vector<SymbolicPhaseEdge> edges{};

    [[nodiscard]] bool operator<(const SymbolicPhaseGraphKey& other) const noexcept {
        if (local != other.local) {
            return local < other.local;
        }
        return edges < other.edges;
    }

    [[nodiscard]] bool operator==(const SymbolicPhaseGraphKey& other) const noexcept = default;
};

struct ExactSymbolicPhaseConfig {
    std::size_t max_qubits{1U << 20U};
    std::size_t max_symbols{1U << 20U};
    std::size_t max_live_branches{1U << 16U};
    std::size_t max_intermediate_branches{1U << 17U};
    std::size_t max_coefficient_terms{1U << 20U};
    std::size_t max_symbol_terms{1U << 22U};
    std::size_t max_integer_bits{60U};
    std::size_t max_retained_estimated_bytes{1U << 30U};
    std::size_t max_materialize_qubits{22U};
};

struct ExactSymbolicPhaseStats {
    std::size_t qubits{0U};
    std::size_t symbols{0U};
    std::size_t live_branches{0U};
    std::size_t hadamard_defects{0U};
    std::size_t max_live_branches{0U};
    std::size_t coefficient_terms{0U};
    std::size_t max_coefficient_terms{0U};
    std::size_t symbol_terms{0U};
    std::size_t max_integer_bits{0U};
    std::size_t graph_merges{0U};
    std::size_t exact_cancellations{0U};
    std::size_t extracted_power_of_two_bits{0U};
    std::size_t retained_estimated_bytes{0U};
    std::int64_t sqrt2_denominator_power{0};
};

struct ExactSymbolicScaledAmplitude {
    std::map<SymbolicPhaseKey, std::int64_t> terms{};
    std::int64_t sqrt2_denominator_power{0};
    std::size_t qubits{0U};
};

class ExactSymbolicPhaseGraphSum {
public:
    explicit ExactSymbolicPhaseGraphSum(
        std::size_t qubit_count,
        ExactSymbolicPhaseConfig config = {})
        : qubit_count_(validated_qubit_count(qubit_count, config)), config_(config) {
        Branch branch;
        branch.weights.emplace(SymbolicPhaseKey{}, 1);
        branches_.push_back(std::move(branch));
        max_live_branches_ = 1U;
        max_coefficient_terms_ = 1U;
        max_integer_bits_ = 1U;
        refresh_and_enforce();
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::size_t branch_count() const noexcept { return branches_.size(); }
    [[nodiscard]] const ExactSymbolicPhaseStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const ExactSymbolicPhaseConfig& config() const noexcept { return config_; }

    void bind_symbol(SymbolicPhaseId symbol, double value) {
        if (!std::isfinite(value)) {
            throw QStateError("Symbolic phase value must be finite");
        }
        if (value == 0.0) {
            value = 0.0;
        }
        const auto found = symbols_.find(symbol);
        if (found != symbols_.end()) {
            if (found->second != value) {
                throw QStateError("Symbolic phase ID was rebound to a different value");
            }
            return;
        }
        if (symbols_.size() >= config_.max_symbols) {
            throw QStateError("Symbolic phase table exceeds configured symbol cap");
        }
        const std::size_t next_symbol_bytes = checked_product(
            symbols_.size() + 1U,
            sizeof(std::pair<SymbolicPhaseId, double>) + 3U * sizeof(void*),
            "Symbolic phase symbol storage overflowed");
        const std::size_t current_without_symbols = estimated_bytes(branches_) -
            checked_product(
                symbols_.size(),
                sizeof(std::pair<SymbolicPhaseId, double>) + 3U * sizeof(void*),
                "Symbolic phase symbol storage overflowed");
        if (checked_sum(
                current_without_symbols,
                next_symbol_bytes,
                "Symbolic phase storage overflowed") > config_.max_retained_estimated_bytes) {
            throw QStateError("Symbolic phase symbol binding exceeds retained-byte cap");
        }
        symbols_.emplace(symbol, value);
        refresh_and_enforce();
    }

    [[nodiscard]] bool has_symbol(SymbolicPhaseId symbol) const noexcept {
        return symbols_.find(symbol) != symbols_.end();
    }

    [[nodiscard]] double symbol_value(SymbolicPhaseId symbol) const {
        const auto found = symbols_.find(symbol);
        if (found == symbols_.end()) {
            throw QStateError("Symbolic phase ID is not bound");
        }
        return found->second;
    }

    void apply_h(QubitId qubit) {
        validate_qubit(qubit);
        if (branches_.size() > config_.max_intermediate_branches / 2U) {
            throw QStateError("Symbolic phase Hadamard exceeds intermediate branch cap");
        }
        const std::size_t next_count = checked_product(
            branches_.size(), 2U,
            "Symbolic phase Hadamard branch count overflowed");
        if (next_count > config_.max_intermediate_branches) {
            throw QStateError("Symbolic phase Hadamard exceeds intermediate branch cap");
        }
        if (sqrt2_denominator_power_ == std::numeric_limits<std::int64_t>::max()) {
            throw QStateError("Symbolic phase denominator exponent overflowed");
        }

        std::vector<Branch> next;
        next.reserve(next_count);
        for (const Branch& branch : branches_) {
            Branch left = branch;
            Branch right = branch;
            graph_x(left.graph, qubit);
            graph_z(right.graph, qubit);
            next.push_back(std::move(left));
            next.push_back(std::move(right));
        }

        CompressionResult compressed = compress(
            std::move(next), sqrt2_denominator_power_ + 1);
        commit_compression(std::move(compressed));
        hadamard_defects_ = checked_sum(
            hadamard_defects_, 1U,
            "Symbolic phase Hadamard counter overflowed");
        max_live_branches_ = std::max(max_live_branches_, next_count);
        refresh_and_enforce();
    }

    void apply_x(QubitId qubit) {
        validate_qubit(qubit);
        mutate_graphs([qubit](Graph& graph) { graph_x(graph, qubit); });
    }

    void apply_y(QubitId qubit) {
        validate_qubit(qubit);
        mutate_graphs([qubit](Graph& graph) {
            graph_z(graph, qubit);
            graph_x(graph, qubit);
            add_constant(graph.global, 4);
        });
    }

    void apply_z(QubitId qubit) {
        validate_qubit(qubit);
        mutate_graphs([qubit](Graph& graph) { graph_z(graph, qubit); });
    }

    void apply_s(QubitId qubit) {
        validate_qubit(qubit);
        mutate_graphs([qubit](Graph& graph) { add_constant(graph.local[qubit], 4); });
    }

    void apply_sdg(QubitId qubit) {
        validate_qubit(qubit);
        mutate_graphs([qubit](Graph& graph) { add_constant(graph.local[qubit], 12); });
    }

    void apply_t(QubitId qubit) {
        validate_qubit(qubit);
        mutate_graphs([qubit](Graph& graph) { add_constant(graph.local[qubit], 2); });
    }

    void apply_tdg(QubitId qubit) {
        validate_qubit(qubit);
        mutate_graphs([qubit](Graph& graph) { add_constant(graph.local[qubit], 14); });
    }

    void apply_cz(QubitId first, QubitId second) {
        validate_pair(first, second);
        mutate_graphs([first, second](Graph& graph) {
            add_constant(graph.edges[edge_key(first, second)], 8);
        });
    }

    void apply_swap(QubitId first, QubitId second) {
        validate_pair(first, second);
        mutate_graphs([first, second](Graph& graph) {
            graph_swap(graph, first, second);
        });
    }

    void apply_rz_symbol(
        QubitId qubit,
        SymbolicPhaseId symbol,
        int sign = 1) {
        validate_qubit(qubit);
        require_symbol(symbol);
        validate_sign(sign);
        mutate_graphs([qubit, symbol, sign](Graph& graph) {
            add_symbol(graph.global, symbol, sign > 0 ? -1 : 1);
            add_symbol(graph.local[qubit], symbol, sign > 0 ? 2 : -2);
        });
    }

    void apply_controlled_phase_symbol(
        QubitId first,
        QubitId second,
        SymbolicPhaseId symbol,
        int sign = 1) {
        validate_pair(first, second);
        require_symbol(symbol);
        validate_sign(sign);
        mutate_graphs([first, second, symbol, sign](Graph& graph) {
            add_symbol(
                graph.edges[edge_key(first, second)],
                symbol,
                sign > 0 ? 2 : -2);
        });
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
                throw QStateError(
                    "Runtime Rz requires apply_rz_symbol with an explicit stable symbol ID");
            case OperationCode::Rx:
            case OperationCode::Ry:
            case OperationCode::Cnot:
            case OperationCode::BitFlipTrajectory:
            case OperationCode::PhaseFlipTrajectory:
            case OperationCode::DepolarizingTrajectory:
            case OperationCode::AmplitudeDampingTrajectory:
                throw QStateError("Operation is outside the exact symbolic PhaseGraph contract");
            default:
                throw QStateError("Symbolic PhaseGraph received an unknown opcode");
        }
    }

    [[nodiscard]] ExactSymbolicScaledAmplitude exact_amplitude_bits(
        std::span<const std::uint8_t> bits) const {
        validate_bits(bits);
        std::map<SymbolicPhaseKey, std::int64_t> terms;
        for (const Branch& branch : branches_) {
            const SymbolicPhaseKey graph_phase = graph_phase_bits(branch.graph, bits);
            for (const auto& [weight_phase, multiplicity] : branch.weights) {
                SymbolicPhaseKey key = add_keys(weight_phase, graph_phase);
                auto normalized = normalize_weight(std::move(key), multiplicity);
                merge_term(terms, std::move(normalized.first), normalized.second, nullptr);
            }
        }
        std::int64_t denominator = sqrt2_denominator_power_;
        normalize_common_power(terms, denominator, nullptr);
        enforce_weight_map(terms);
        return {std::move(terms), denominator, qubit_count_};
    }

    [[nodiscard]] QComplex scaled_mantissa_bits(
        std::span<const std::uint8_t> bits) const {
        const ExactSymbolicScaledAmplitude exact = exact_amplitude_bits(bits);
        QComplex value{};
        for (const auto& [key, multiplicity] : exact.terms) {
            value += static_cast<double>(multiplicity) *
                QComplex::from_polar(1.0, phase_value(key));
        }
        if (!finite(value)) {
            throw QStateError("Symbolic phase amplitude mantissa became non-finite");
        }
        return value;
    }

    [[nodiscard]] double log2_probability_bits(
        std::span<const std::uint8_t> bits) const {
        const ExactSymbolicScaledAmplitude exact = exact_amplitude_bits(bits);
        QComplex mantissa{};
        for (const auto& [key, multiplicity] : exact.terms) {
            mantissa += static_cast<double>(multiplicity) *
                QComplex::from_polar(1.0, phase_value(key));
        }
        const double norm2 = mantissa.norm2();
        if (!std::isfinite(norm2)) {
            throw QStateError("Symbolic phase probability became non-finite");
        }
        if (norm2 == 0.0) {
            return -std::numeric_limits<double>::infinity();
        }
        return std::log2(norm2) - static_cast<double>(qubit_count_) -
            static_cast<double>(exact.sqrt2_denominator_power);
    }

    [[nodiscard]] QComplex amplitude_bits(
        std::span<const std::uint8_t> bits) const {
        const ExactSymbolicScaledAmplitude exact = exact_amplitude_bits(bits);
        QComplex mantissa{};
        for (const auto& [key, multiplicity] : exact.terms) {
            mantissa += static_cast<double>(multiplicity) *
                QComplex::from_polar(1.0, phase_value(key));
        }
        const double exponent = -0.5 * (
            static_cast<double>(qubit_count_) +
            static_cast<double>(exact.sqrt2_denominator_power));
        const double scale = std::exp2(exponent);
        if (scale == 0.0 && mantissa.norm2() != 0.0) {
            throw QStateError(
                "Symbolic phase amplitude underflows; use exact_amplitude_bits or log2_probability_bits");
        }
        return mantissa * scale;
    }

    [[nodiscard]] std::vector<QComplex> materialize(
        std::size_t max_qubits = 22U) const {
        const std::size_t limit = std::min(max_qubits, config_.max_materialize_qubits);
        if (qubit_count_ > limit || qubit_count_ >= 63U) {
            throw QStateError("Symbolic phase materialization exceeds configured qubit cap");
        }
        const BasisIndex dimension = BasisIndex{1U} << qubit_count_;
        std::vector<QComplex> result(static_cast<std::size_t>(dimension));
        std::vector<std::uint8_t> bits(qubit_count_, 0U);
        for (BasisIndex basis = 0U; basis < dimension; ++basis) {
            for (std::size_t qubit = 0U; qubit < qubit_count_; ++qubit) {
                bits[qubit] = static_cast<std::uint8_t>((basis >> qubit) & 1U);
            }
            result[static_cast<std::size_t>(basis)] = amplitude_bits(bits);
        }
        return result;
    }

private:
    struct Graph {
        SymbolicLinearPhase global{};
        std::map<QubitId, SymbolicLinearPhase> local{};
        std::map<std::pair<QubitId, QubitId>, SymbolicLinearPhase> edges{};
    };

    struct Branch {
        std::map<SymbolicPhaseKey, std::int64_t> weights{};
        Graph graph{};
    };

    struct CompressionResult {
        std::vector<Branch> branches{};
        std::int64_t denominator{0};
        std::size_t merges{0U};
        std::size_t cancellations{0U};
        std::size_t extracted_bits{0U};
    };

    std::size_t qubit_count_{0U};
    ExactSymbolicPhaseConfig config_{};
    std::map<SymbolicPhaseId, double> symbols_{};
    std::vector<Branch> branches_{};
    std::int64_t sqrt2_denominator_power_{0};
    std::size_t hadamard_defects_{0U};
    std::size_t max_live_branches_{0U};
    std::size_t max_coefficient_terms_{0U};
    std::size_t max_integer_bits_{0U};
    std::size_t graph_merges_{0U};
    std::size_t exact_cancellations_{0U};
    std::size_t extracted_power_of_two_bits_{0U};
    ExactSymbolicPhaseStats stats_{};

    [[nodiscard]] static std::size_t validated_qubit_count(
        std::size_t qubit_count,
        const ExactSymbolicPhaseConfig& config) {
        if (qubit_count == 0U ||
            qubit_count > static_cast<std::size_t>(std::numeric_limits<QubitId>::max()) ||
            qubit_count > config.max_qubits ||
            config.max_symbols == 0U ||
            config.max_live_branches == 0U ||
            config.max_intermediate_branches < config.max_live_branches ||
            config.max_coefficient_terms == 0U ||
            config.max_symbol_terms == 0U ||
            config.max_integer_bits == 0U || config.max_integer_bits > 63U ||
            config.max_retained_estimated_bytes == 0U ||
            config.max_materialize_qubits == 0U) {
            throw QStateError("Symbolic phase dimensions or configuration are invalid");
        }
        return qubit_count;
    }

    void validate_qubit(QubitId qubit) const {
        if (static_cast<std::size_t>(qubit) >= qubit_count_) {
            throw QStateError("Symbolic phase qubit is out of range");
        }
    }

    void validate_pair(QubitId first, QubitId second) const {
        validate_qubit(first);
        validate_qubit(second);
        if (first == second) {
            throw QStateError("Symbolic phase two-qubit operation requires distinct qubits");
        }
    }

    void validate_sign(int sign) const {
        if (sign != 1 && sign != -1) {
            throw QStateError("Symbolic phase parameter sign must be +1 or -1");
        }
    }

    void require_symbol(SymbolicPhaseId symbol) const {
        if (!has_symbol(symbol)) {
            throw QStateError("Symbolic phase operation references an unbound symbol ID");
        }
    }

    void validate_bits(std::span<const std::uint8_t> bits) const {
        if (bits.size() != qubit_count_) {
            throw QStateError("Symbolic phase bit-vector length does not match qubit count");
        }
        for (const std::uint8_t bit : bits) {
            if (bit > 1U) {
                throw QStateError("Symbolic phase basis bits must be zero or one");
            }
        }
    }

    template <typename Mutation>
    void mutate_graphs(Mutation&& mutation) {
        std::vector<Branch> next = branches_;
        for (Branch& branch : next) {
            mutation(branch.graph);
            canonicalize_graph(branch.graph);
        }
        CompressionResult compressed = compress(
            std::move(next), sqrt2_denominator_power_);
        commit_compression(std::move(compressed));
        refresh_and_enforce();
    }

    [[nodiscard]] CompressionResult compress(
        std::vector<Branch> input,
        std::int64_t denominator) const {
        std::map<SymbolicPhaseGraphKey, Branch> merged;
        CompressionResult result;
        result.denominator = denominator;

        for (Branch& branch : input) {
            const SymbolicPhaseKey global = phase_key(branch.graph.global);
            branch.graph.global = {};
            canonicalize_graph(branch.graph);
            std::map<SymbolicPhaseKey, std::int64_t> shifted;
            for (const auto& [key, multiplicity] : branch.weights) {
                SymbolicPhaseKey target = add_keys(key, global);
                auto normalized = normalize_weight(std::move(target), multiplicity);
                merge_term(
                    shifted, std::move(normalized.first), normalized.second,
                    &result.cancellations);
            }
            branch.weights = std::move(shifted);
            if (branch.weights.empty()) {
                continue;
            }
            const SymbolicPhaseGraphKey signature = graph_key(branch.graph);
            const auto found = merged.find(signature);
            if (found == merged.end()) {
                merged.emplace(signature, std::move(branch));
            } else {
                for (const auto& [key, multiplicity] : branch.weights) {
                    merge_term(
                        found->second.weights, key, multiplicity,
                        &result.cancellations);
                }
                ++result.merges;
            }
        }

        result.branches.reserve(merged.size());
        for (auto& entry : merged) {
            if (!entry.second.weights.empty()) {
                result.branches.push_back(std::move(entry.second));
            }
        }
        if (result.branches.empty()) {
            throw QStateError("Symbolic phase exact cancellation produced a zero state");
        }
        if (result.branches.size() > config_.max_live_branches) {
            throw QStateError("Symbolic phase live branch count exceeds configured cap");
        }
        normalize_common_power(
            result.branches, result.denominator, &result.extracted_bits);
        enforce_branches(result.branches);
        return result;
    }

    void commit_compression(CompressionResult result) {
        branches_ = std::move(result.branches);
        sqrt2_denominator_power_ = result.denominator;
        graph_merges_ = checked_sum(
            graph_merges_, result.merges,
            "Symbolic phase merge counter overflowed");
        exact_cancellations_ = checked_sum(
            exact_cancellations_, result.cancellations,
            "Symbolic phase cancellation counter overflowed");
        extracted_power_of_two_bits_ = checked_sum(
            extracted_power_of_two_bits_, result.extracted_bits,
            "Symbolic phase extraction counter overflowed");
    }

    void enforce_branches(const std::vector<Branch>& branches) const {
        std::size_t coefficient_terms = 0U;
        std::size_t symbol_terms = symbols_.size();
        std::size_t integer_bits = 1U;
        for (const Branch& branch : branches) {
            coefficient_terms = checked_sum(
                coefficient_terms, branch.weights.size(),
                "Symbolic phase coefficient term count overflowed");
            for (const auto& [key, multiplicity] : branch.weights) {
                symbol_terms = checked_sum(
                    symbol_terms, key.terms.size(),
                    "Symbolic phase symbol term count overflowed");
                integer_bits = std::max(integer_bits, integer_bit_width(multiplicity));
                for (const auto& term : key.terms) {
                    integer_bits = std::max(integer_bits, integer_bit_width(term.second));
                }
            }
            symbol_terms = checked_sum(
                symbol_terms, graph_symbol_terms(branch.graph),
                "Symbolic phase graph symbol term count overflowed");
            integer_bits = std::max(integer_bits, graph_integer_bits(branch.graph));
        }
        if (coefficient_terms > config_.max_coefficient_terms) {
            throw QStateError("Symbolic phase coefficient term count exceeds configured cap");
        }
        if (symbol_terms > config_.max_symbol_terms) {
            throw QStateError("Symbolic phase symbol term count exceeds configured cap");
        }
        if (integer_bits > config_.max_integer_bits) {
            throw QStateError("Symbolic phase integer coefficient width exceeds configured cap");
        }
        if (estimated_bytes(branches) > config_.max_retained_estimated_bytes) {
            throw QStateError("Symbolic phase retained-byte estimate exceeds configured cap");
        }
    }

    void enforce_weight_map(
        const std::map<SymbolicPhaseKey, std::int64_t>& weights) const {
        if (weights.size() > config_.max_coefficient_terms) {
            throw QStateError("Symbolic phase point amplitude exceeds coefficient-term cap");
        }
        std::size_t terms = 0U;
        std::size_t bits = 1U;
        for (const auto& [key, multiplicity] : weights) {
            terms = checked_sum(
                terms, key.terms.size(),
                "Symbolic phase point symbol term count overflowed");
            bits = std::max(bits, integer_bit_width(multiplicity));
            for (const auto& term : key.terms) {
                bits = std::max(bits, integer_bit_width(term.second));
            }
        }
        if (terms > config_.max_symbol_terms || bits > config_.max_integer_bits) {
            throw QStateError("Symbolic phase point amplitude exceeds symbolic resource cap");
        }
    }

    void refresh_and_enforce() {
        enforce_branches(branches_);
        std::size_t coefficient_terms = 0U;
        std::size_t symbol_terms = symbols_.size();
        std::size_t integer_bits = 1U;
        for (const Branch& branch : branches_) {
            coefficient_terms = checked_sum(
                coefficient_terms, branch.weights.size(),
                "Symbolic phase coefficient term count overflowed");
            for (const auto& [key, multiplicity] : branch.weights) {
                symbol_terms = checked_sum(
                    symbol_terms, key.terms.size(),
                    "Symbolic phase symbol term count overflowed");
                integer_bits = std::max(integer_bits, integer_bit_width(multiplicity));
                for (const auto& term : key.terms) {
                    integer_bits = std::max(integer_bits, integer_bit_width(term.second));
                }
            }
            symbol_terms = checked_sum(
                symbol_terms, graph_symbol_terms(branch.graph),
                "Symbolic phase graph term count overflowed");
            integer_bits = std::max(integer_bits, graph_integer_bits(branch.graph));
        }
        max_coefficient_terms_ = std::max(max_coefficient_terms_, coefficient_terms);
        max_integer_bits_ = std::max(max_integer_bits_, integer_bits);
        max_live_branches_ = std::max(max_live_branches_, branches_.size());
        stats_ = {
            qubit_count_, symbols_.size(), branches_.size(), hadamard_defects_,
            max_live_branches_, coefficient_terms, max_coefficient_terms_,
            symbol_terms, max_integer_bits_, graph_merges_, exact_cancellations_,
            extracted_power_of_two_bits_, estimated_bytes(branches_),
            sqrt2_denominator_power_,
        };
    }

    [[nodiscard]] std::size_t estimated_bytes(
        const std::vector<Branch>& branches) const {
        std::size_t total = sizeof(*this);
        total = checked_sum(
            total,
            checked_product(
                symbols_.size(),
                sizeof(std::pair<SymbolicPhaseId, double>) + 3U * sizeof(void*),
                "Symbolic phase symbol storage overflowed"),
            "Symbolic phase storage overflowed");
        total = checked_sum(
            total,
            checked_product(
                branches.size(), sizeof(Branch),
                "Symbolic phase branch storage overflowed"),
            "Symbolic phase storage overflowed");
        for (const Branch& branch : branches) {
            total = checked_sum(
                total,
                checked_product(
                    branch.weights.size(),
                    sizeof(std::int64_t) + sizeof(SymbolicPhaseKey) + 3U * sizeof(void*),
                    "Symbolic phase coefficient storage overflowed"),
                "Symbolic phase storage overflowed");
            for (const auto& weight : branch.weights) {
                total = checked_sum(
                    total,
                    checked_product(
                        weight.first.terms.size(),
                        sizeof(std::pair<SymbolicPhaseId, std::int64_t>),
                        "Symbolic phase coefficient-symbol storage overflowed"),
                    "Symbolic phase storage overflowed");
            }
            total = checked_sum(
                total, graph_estimated_bytes(branch.graph),
                "Symbolic phase graph storage overflowed");
        }
        return total;
    }

    [[nodiscard]] static std::size_t graph_estimated_bytes(const Graph& graph) {
        std::size_t total = 0U;
        const std::size_t symbol_entry =
            sizeof(std::pair<SymbolicPhaseId, std::int64_t>) + 3U * sizeof(void*);
        total = checked_sum(
            total,
            checked_product(graph.global.terms.size(), symbol_entry,
                "Symbolic graph global storage overflowed"),
            "Symbolic graph storage overflowed");
        total = checked_sum(
            total,
            checked_product(
                graph.local.size(),
                sizeof(QubitId) + sizeof(SymbolicLinearPhase) + 3U * sizeof(void*),
                "Symbolic graph local storage overflowed"),
            "Symbolic graph storage overflowed");
        total = checked_sum(
            total,
            checked_product(
                graph.edges.size(),
                2U * sizeof(QubitId) + sizeof(SymbolicLinearPhase) + 3U * sizeof(void*),
                "Symbolic graph edge storage overflowed"),
            "Symbolic graph storage overflowed");
        for (const auto& local : graph.local) {
            total = checked_sum(
                total,
                checked_product(local.second.terms.size(), symbol_entry,
                    "Symbolic graph local-symbol storage overflowed"),
                "Symbolic graph storage overflowed");
        }
        for (const auto& edge : graph.edges) {
            total = checked_sum(
                total,
                checked_product(edge.second.terms.size(), symbol_entry,
                    "Symbolic graph edge-symbol storage overflowed"),
                "Symbolic graph storage overflowed");
        }
        return total;
    }

    [[nodiscard]] static SymbolicPhaseGraphKey graph_key(const Graph& graph) {
        SymbolicPhaseGraphKey key;
        key.local.reserve(graph.local.size());
        key.edges.reserve(graph.edges.size());
        for (const auto& [qubit, phase] : graph.local) {
            key.local.push_back({qubit, phase_key(phase)});
        }
        for (const auto& [edge, phase] : graph.edges) {
            key.edges.push_back({edge.first, edge.second, phase_key(phase)});
        }
        return key;
    }

    [[nodiscard]] static SymbolicPhaseKey graph_phase_bits(
        const Graph& graph,
        std::span<const std::uint8_t> bits) {
        SymbolicPhaseKey result = phase_key(graph.global);
        for (const auto& [qubit, phase] : graph.local) {
            if (bits[qubit] != 0U) {
                result = add_keys(result, phase_key(phase));
            }
        }
        for (const auto& [edge, phase] : graph.edges) {
            if (bits[edge.first] != 0U && bits[edge.second] != 0U) {
                result = add_keys(result, phase_key(phase));
            }
        }
        return result;
    }

    [[nodiscard]] double phase_value(const SymbolicPhaseKey& key) const {
        constexpr double pi_over_eight = 0.392699081698724154807830422909937861;
        double value = static_cast<double>(key.eighths) * pi_over_eight;
        for (const auto& [symbol, coefficient] : key.terms) {
            const auto found = symbols_.find(symbol);
            if (found == symbols_.end()) {
                throw QStateError("Symbolic phase amplitude references an unbound symbol");
            }
            value += 0.5 * static_cast<double>(coefficient) * found->second;
        }
        if (!std::isfinite(value)) {
            throw QStateError("Symbolic phase evaluation became non-finite");
        }
        return value;
    }

    [[nodiscard]] static SymbolicPhaseKey phase_key(const SymbolicLinearPhase& phase) {
        SymbolicPhaseKey key;
        key.eighths = static_cast<std::uint8_t>(phase.eighths & 15U);
        key.terms.reserve(phase.terms.size());
        for (const auto& term : phase.terms) {
            if (term.second != 0) {
                key.terms.push_back(term);
            }
        }
        return key;
    }

    [[nodiscard]] static SymbolicPhaseKey add_keys(
        const SymbolicPhaseKey& first,
        const SymbolicPhaseKey& second) {
        SymbolicPhaseKey result;
        result.eighths = static_cast<std::uint8_t>(
            (static_cast<unsigned>(first.eighths) +
             static_cast<unsigned>(second.eighths)) & 15U);
        std::map<SymbolicPhaseId, std::int64_t> terms;
        for (const auto& term : first.terms) {
            terms.emplace(term);
        }
        for (const auto& [symbol, coefficient] : second.terms) {
            const auto found = terms.find(symbol);
            if (found == terms.end()) {
                terms.emplace(symbol, coefficient);
            } else {
                found->second = checked_add(
                    found->second, coefficient,
                    "Symbolic phase-key addition overflowed");
                if (found->second == 0) {
                    terms.erase(found);
                }
            }
        }
        result.terms.reserve(terms.size());
        for (const auto& term : terms) {
            result.terms.push_back(term);
        }
        return result;
    }

    [[nodiscard]] static std::pair<SymbolicPhaseKey, std::int64_t> normalize_weight(
        SymbolicPhaseKey key,
        std::int64_t multiplicity) {
        key.eighths &= 15U;
        if (key.eighths >= 8U) {
            key.eighths = static_cast<std::uint8_t>(key.eighths - 8U);
            multiplicity = checked_negate(
                multiplicity,
                "Symbolic coefficient sign normalization overflowed");
        }
        return {std::move(key), multiplicity};
    }

    static void merge_term(
        std::map<SymbolicPhaseKey, std::int64_t>& target,
        SymbolicPhaseKey key,
        std::int64_t multiplicity,
        std::size_t* cancellations) {
        if (multiplicity == 0) {
            return;
        }
        const auto found = target.find(key);
        if (found == target.end()) {
            target.emplace(std::move(key), multiplicity);
            return;
        }
        const std::int64_t value = checked_add(
            found->second, multiplicity,
            "Symbolic coefficient addition overflowed");
        if (value == 0) {
            target.erase(found);
            if (cancellations != nullptr) {
                ++(*cancellations);
            }
        } else {
            found->second = value;
        }
    }

    static void normalize_common_power(
        std::vector<Branch>& branches,
        std::int64_t& denominator,
        std::size_t* extracted) {
        while (denominator >= 2) {
            bool seen = false;
            bool all_even = true;
            for (const Branch& branch : branches) {
                for (const auto& weight : branch.weights) {
                    seen = true;
                    if ((weight.second & 1) != 0) {
                        all_even = false;
                        break;
                    }
                }
                if (!all_even) {
                    break;
                }
            }
            if (!seen || !all_even) {
                break;
            }
            for (Branch& branch : branches) {
                for (auto& weight : branch.weights) {
                    weight.second /= 2;
                }
            }
            denominator -= 2;
            if (extracted != nullptr) {
                ++(*extracted);
            }
        }
    }

    static void normalize_common_power(
        std::map<SymbolicPhaseKey, std::int64_t>& weights,
        std::int64_t& denominator,
        std::size_t* extracted) {
        while (denominator >= 2 && !weights.empty()) {
            bool all_even = true;
            for (const auto& weight : weights) {
                if ((weight.second & 1) != 0) {
                    all_even = false;
                    break;
                }
            }
            if (!all_even) {
                break;
            }
            for (auto& weight : weights) {
                weight.second /= 2;
            }
            denominator -= 2;
            if (extracted != nullptr) {
                ++(*extracted);
            }
        }
    }

    static void graph_x(Graph& graph, QubitId qubit) {
        const auto target = graph.local.find(qubit);
        if (target != graph.local.end()) {
            add_phase(graph.global, target->second);
            negate_phase(target->second);
        }
        for (auto& [edge, phase] : graph.edges) {
            if (edge.first != qubit && edge.second != qubit) {
                continue;
            }
            const QubitId other = edge.first == qubit ? edge.second : edge.first;
            add_phase(graph.local[other], phase);
            negate_phase(phase);
        }
        canonicalize_graph(graph);
    }

    static void graph_z(Graph& graph, QubitId qubit) {
        add_constant(graph.local[qubit], 8);
        canonicalize_graph(graph);
    }

    static void graph_swap(Graph& graph, QubitId first, QubitId second) {
        SymbolicLinearPhase first_phase;
        SymbolicLinearPhase second_phase;
        const auto first_it = graph.local.find(first);
        const auto second_it = graph.local.find(second);
        const bool have_first = first_it != graph.local.end();
        const bool have_second = second_it != graph.local.end();
        if (have_first) {
            first_phase = first_it->second;
        }
        if (have_second) {
            second_phase = second_it->second;
        }
        graph.local.erase(first);
        graph.local.erase(second);
        if (have_first) {
            graph.local[second] = std::move(first_phase);
        }
        if (have_second) {
            graph.local[first] = std::move(second_phase);
        }

        std::map<std::pair<QubitId, QubitId>, SymbolicLinearPhase> remapped;
        for (const auto& [edge, phase] : graph.edges) {
            QubitId left = edge.first;
            QubitId right = edge.second;
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
            add_phase(remapped[edge_key(left, right)], phase);
        }
        graph.edges = std::move(remapped);
        canonicalize_graph(graph);
    }

    static void canonicalize_graph(Graph& graph) {
        canonicalize_phase(graph.global);
        for (auto iterator = graph.local.begin(); iterator != graph.local.end();) {
            canonicalize_phase(iterator->second);
            if (iterator->second.zero()) {
                iterator = graph.local.erase(iterator);
            } else {
                ++iterator;
            }
        }
        for (auto iterator = graph.edges.begin(); iterator != graph.edges.end();) {
            canonicalize_phase(iterator->second);
            if (iterator->second.zero()) {
                iterator = graph.edges.erase(iterator);
            } else {
                ++iterator;
            }
        }
    }

    static void canonicalize_phase(SymbolicLinearPhase& phase) {
        phase.eighths &= 15U;
        for (auto iterator = phase.terms.begin(); iterator != phase.terms.end();) {
            if (iterator->second == 0) {
                iterator = phase.terms.erase(iterator);
            } else {
                ++iterator;
            }
        }
    }

    static void add_constant(SymbolicLinearPhase& phase, int eighths) noexcept {
        const int value = (static_cast<int>(phase.eighths) + eighths) % 16;
        phase.eighths = static_cast<std::uint8_t>(value < 0 ? value + 16 : value);
    }

    static void add_symbol(
        SymbolicLinearPhase& phase,
        SymbolicPhaseId symbol,
        std::int64_t coefficient) {
        if (coefficient == 0) {
            return;
        }
        const auto found = phase.terms.find(symbol);
        if (found == phase.terms.end()) {
            phase.terms.emplace(symbol, coefficient);
        } else {
            found->second = checked_add(
                found->second, coefficient,
                "Symbolic linear coefficient overflowed");
            if (found->second == 0) {
                phase.terms.erase(found);
            }
        }
    }

    static void add_phase(
        SymbolicLinearPhase& target,
        const SymbolicLinearPhase& source) {
        add_constant(target, static_cast<int>(source.eighths));
        for (const auto& [symbol, coefficient] : source.terms) {
            add_symbol(target, symbol, coefficient);
        }
    }

    static void negate_phase(SymbolicLinearPhase& phase) {
        add_constant(phase, -2 * static_cast<int>(phase.eighths));
        for (auto& [symbol, coefficient] : phase.terms) {
            coefficient = checked_negate(
                coefficient,
                "Symbolic linear negation overflowed");
        }
        canonicalize_phase(phase);
    }

    [[nodiscard]] static std::pair<QubitId, QubitId> edge_key(
        QubitId first,
        QubitId second) noexcept {
        return first < second
            ? std::pair<QubitId, QubitId>{first, second}
            : std::pair<QubitId, QubitId>{second, first};
    }

    [[nodiscard]] static std::size_t graph_symbol_terms(const Graph& graph) {
        std::size_t total = graph.global.terms.size();
        for (const auto& local : graph.local) {
            total = checked_sum(total, local.second.terms.size(),
                "Symbolic graph term count overflowed");
        }
        for (const auto& edge : graph.edges) {
            total = checked_sum(total, edge.second.terms.size(),
                "Symbolic graph term count overflowed");
        }
        return total;
    }

    [[nodiscard]] static std::size_t graph_integer_bits(const Graph& graph) noexcept {
        std::size_t bits = 1U;
        for (const auto& term : graph.global.terms) {
            bits = std::max(bits, integer_bit_width(term.second));
        }
        for (const auto& local : graph.local) {
            for (const auto& term : local.second.terms) {
                bits = std::max(bits, integer_bit_width(term.second));
            }
        }
        for (const auto& edge : graph.edges) {
            for (const auto& term : edge.second.terms) {
                bits = std::max(bits, integer_bit_width(term.second));
            }
        }
        return bits;
    }

    [[nodiscard]] static std::size_t integer_bit_width(std::int64_t value) noexcept {
        std::uint64_t magnitude = value >= 0
            ? static_cast<std::uint64_t>(value)
            : static_cast<std::uint64_t>(-(value + 1)) + 1U;
        std::size_t bits = 1U;
        while (magnitude > 1U) {
            ++bits;
            magnitude >>= 1U;
        }
        return bits;
    }

    [[nodiscard]] static std::int64_t checked_add(
        std::int64_t left,
        std::int64_t right,
        const char* message) {
        if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
            (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right)) {
            throw QStateError(message);
        }
        return left + right;
    }

    [[nodiscard]] static std::int64_t checked_negate(
        std::int64_t value,
        const char* message) {
        if (value == std::numeric_limits<std::int64_t>::min()) {
            throw QStateError(message);
        }
        return -value;
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

    [[nodiscard]] static bool finite(QComplex value) noexcept {
        return std::isfinite(value.re) && std::isfinite(value.im);
    }
};

}  // namespace qubit

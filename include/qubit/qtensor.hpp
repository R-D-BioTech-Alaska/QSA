#pragma once

#include "qubit/qplan.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace qubit {

class TensorContractionPlan;
class TensorContractionWorkspace;

struct TensorNetworkConfig {
    std::size_t max_contraction_entries{1U << 20U};
    std::size_t max_factors{1'000'000U};
};

struct TensorContractionStats {
    std::size_t source_operations{0};
    std::size_t source_factors{0};
    std::size_t eliminated_variables{0};
    std::size_t peak_union_variables{0};
    std::size_t peak_contraction_entries{0};
};

class TensorNetworkCircuit {
public:
    explicit TensorNetworkCircuit(
        std::size_t qubit_count,
        TensorNetworkConfig config = {});
    TensorNetworkCircuit(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        TensorNetworkConfig config = {});

    void apply(const Operation& operation);
    void apply(std::span<const Operation> operations);

    [[nodiscard]] QComplex amplitude(
        std::span<const std::uint8_t> basis_bits,
        TensorContractionStats* stats = nullptr) const;
    [[nodiscard]] QComplex amplitude(
        BasisIndex basis,
        TensorContractionStats* stats = nullptr) const;
    [[nodiscard]] std::vector<QComplex> materialize(
        std::size_t max_qubits = 20U) const;
    [[nodiscard]] TensorContractionPlan compile() const;

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::size_t operation_count() const noexcept { return operation_count_; }
    [[nodiscard]] std::size_t factor_count() const noexcept { return factors_.size(); }
    [[nodiscard]] std::size_t estimated_bytes() const noexcept;
    [[nodiscard]] const TensorNetworkConfig& config() const noexcept { return config_; }
    [[nodiscard]] bool validate(std::string* reason = nullptr) const noexcept;

private:
    using VariableId = std::uint32_t;

    struct Factor {
        std::vector<VariableId> variables{};
        std::vector<QComplex> values{};
    };

    std::size_t qubit_count_{0};
    TensorNetworkConfig config_{};
    std::vector<Factor> factors_{};
    std::vector<VariableId> current_wires_{};
    VariableId next_variable_{0};
    std::size_t operation_count_{0};

    void validate_operation(const Operation& operation) const;
    void apply_single(QubitId qubit, const QMatrix2& matrix);
    void apply_two(QubitId first, QubitId second, const QMatrix4& matrix);
    void push_factor(Factor factor);

    [[nodiscard]] QComplex contract(
        std::vector<Factor> factors,
        TensorContractionStats* stats) const;

    friend class TensorContractionPlan;
};

class TensorContractionWorkspace {
public:
    TensorContractionWorkspace() = default;

    [[nodiscard]] std::size_t estimated_bytes() const noexcept;

private:
    std::vector<std::array<QComplex, 2>> pins_{};
    std::vector<std::vector<QComplex>> outputs_{};

    friend class TensorContractionPlan;
};

class TensorContractionPlan {
public:
    explicit TensorContractionPlan(const TensorNetworkCircuit& circuit);

    [[nodiscard]] TensorContractionWorkspace workspace() const;
    [[nodiscard]] QComplex amplitude(
        std::span<const std::uint8_t> basis_bits,
        TensorContractionStats* stats = nullptr) const;
    [[nodiscard]] QComplex amplitude(
        std::span<const std::uint8_t> basis_bits,
        TensorContractionWorkspace& workspace,
        TensorContractionStats* stats = nullptr) const;
    [[nodiscard]] QComplex amplitude(
        BasisIndex basis,
        TensorContractionStats* stats = nullptr) const;
    [[nodiscard]] QComplex amplitude(
        BasisIndex basis,
        TensorContractionWorkspace& workspace,
        TensorContractionStats* stats = nullptr) const;

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::size_t step_count() const noexcept { return steps_.size(); }
    [[nodiscard]] std::size_t estimated_bytes() const noexcept;
    [[nodiscard]] const TensorContractionStats& stats() const noexcept { return stats_; }

private:
    using VariableId = std::uint32_t;

    struct SourceFactor {
        std::vector<VariableId> variables{};
        std::vector<QComplex> values{};
    };

    struct InputMap {
        std::size_t node{0};
        std::vector<std::size_t> positions{};
    };

    struct Step {
        VariableId eliminated{0};
        std::vector<VariableId> union_variables{};
        std::vector<VariableId> output_variables{};
        std::vector<InputMap> inputs{};
        std::size_t selected_position{0};
        std::size_t output_entries{0};
    };

    std::size_t qubit_count_{0};
    std::size_t source_factor_count_{0};
    std::size_t pin_node_begin_{0};
    std::size_t step_node_begin_{0};
    std::vector<SourceFactor> sources_{};
    std::vector<Step> steps_{};
    std::vector<std::size_t> terminal_nodes_{};
    TensorContractionStats stats_{};

    [[nodiscard]] std::span<const QComplex> node_values(
        std::size_t node,
        const TensorContractionWorkspace& workspace) const;
    void validate_workspace(const TensorContractionWorkspace& workspace) const;
};

}  // namespace qubit

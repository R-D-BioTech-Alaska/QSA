#pragma once

#include "qubit/qstate.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qubit {

struct PhaseGraphConfig {
    std::size_t max_edges{4'000'000};
};

struct PhaseGraphScaledAmplitude {
    QComplex mantissa{};
    double log2_scale{0.0};
};

class PhaseGraphState {
public:
    explicit PhaseGraphState(
        std::size_t qubit_count,
        PhaseGraphConfig config = {});

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::size_t edge_count() const noexcept { return edge_phases_.size(); }
    [[nodiscard]] std::size_t estimated_bytes() const noexcept;

    void apply_x(QubitId qubit);
    void apply_y(QubitId qubit);
    void apply_z(QubitId qubit);
    void apply_s(QubitId qubit);
    void apply_sdg(QubitId qubit);
    void apply_t(QubitId qubit);
    void apply_tdg(QubitId qubit);
    void apply_rz(QubitId qubit, double angle);
    void apply_cz(QubitId first, QubitId second);
    void apply_controlled_phase(QubitId first, QubitId second, double angle);
    void apply_swap(QubitId first, QubitId second);

    [[nodiscard]] double probability_one(QubitId qubit) const;
    [[nodiscard]] QComplex amplitude(BasisIndex basis) const;
    [[nodiscard]] QComplex amplitude_bits(std::span<const std::uint8_t> bits) const;
    [[nodiscard]] PhaseGraphScaledAmplitude scaled_amplitude(BasisIndex basis) const;
    [[nodiscard]] PhaseGraphScaledAmplitude scaled_amplitude_bits(
        std::span<const std::uint8_t> bits) const;
    [[nodiscard]] std::vector<QComplex> materialize(std::size_t max_qubits = 24) const;
    [[nodiscard]] std::vector<int> sample_bits(std::uint64_t seed) const;
    [[nodiscard]] BasisIndex sample_basis(std::uint64_t seed) const;

    [[nodiscard]] bool validate(std::string* reason = nullptr) const;
    [[nodiscard]] std::string describe() const;

private:
    std::size_t qubit_count_{0};
    PhaseGraphConfig config_{};
    double global_phase_{0.0};
    std::vector<double> local_phases_{};
    std::unordered_map<std::uint64_t, double> edge_phases_{};

    void validate_qubit(QubitId qubit) const;
    void add_global_phase(double angle);
    void add_local_phase(QubitId qubit, double angle);
    [[nodiscard]] static std::uint64_t edge_key(QubitId first, QubitId second) noexcept;
    [[nodiscard]] static std::pair<QubitId, QubitId> decode_edge(
        std::uint64_t key) noexcept;
};

struct PhaseGraphBranchConfig {
    PhaseGraphConfig phase_graph{};
    std::size_t max_branches{256};
    std::size_t max_estimated_bytes{1024ULL * 1024ULL * 1024ULL};
};

class PhaseGraphBranchState {
public:
    explicit PhaseGraphBranchState(
        std::size_t qubit_count,
        PhaseGraphBranchConfig config = {});

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::size_t branch_count() const noexcept { return branches_.size(); }
    [[nodiscard]] std::size_t hadamard_defects() const noexcept { return hadamard_defects_; }
    [[nodiscard]] std::size_t estimated_bytes() const noexcept;

    void apply_h(QubitId qubit);
    void apply_x(QubitId qubit);
    void apply_y(QubitId qubit);
    void apply_z(QubitId qubit);
    void apply_s(QubitId qubit);
    void apply_sdg(QubitId qubit);
    void apply_t(QubitId qubit);
    void apply_tdg(QubitId qubit);
    void apply_rz(QubitId qubit, double angle);
    void apply_cz(QubitId first, QubitId second);
    void apply_controlled_phase(QubitId first, QubitId second, double angle);
    void apply_swap(QubitId first, QubitId second);

    [[nodiscard]] QComplex amplitude(BasisIndex basis) const;
    [[nodiscard]] QComplex amplitude_bits(std::span<const std::uint8_t> bits) const;
    [[nodiscard]] PhaseGraphScaledAmplitude scaled_amplitude(BasisIndex basis) const;
    [[nodiscard]] PhaseGraphScaledAmplitude scaled_amplitude_bits(
        std::span<const std::uint8_t> bits) const;
    [[nodiscard]] std::vector<QComplex> materialize(std::size_t max_qubits = 24) const;

    [[nodiscard]] bool validate(std::string* reason = nullptr) const;
    [[nodiscard]] std::string describe() const;

private:
    struct Branch {
        QComplex coefficient{1.0, 0.0};
        PhaseGraphState state;

        Branch(QComplex value, PhaseGraphState graph)
            : coefficient(value), state(std::move(graph)) {}
    };

    std::size_t qubit_count_{0};
    PhaseGraphBranchConfig config_{};
    std::size_t hadamard_defects_{0};
    std::vector<Branch> branches_{};

    [[nodiscard]] std::size_t estimated_bytes(
        const std::vector<Branch>& branches) const noexcept;
    void enforce_resources(const std::vector<Branch>& branches) const;
    void apply_single(void (PhaseGraphState::*operation)(QubitId), QubitId qubit);
    void apply_single_angle(
        void (PhaseGraphState::*operation)(QubitId, double),
        QubitId qubit,
        double angle);
    void apply_two(
        void (PhaseGraphState::*operation)(QubitId, QubitId),
        QubitId first,
        QubitId second);
    void apply_two_angle(
        void (PhaseGraphState::*operation)(QubitId, QubitId, double),
        QubitId first,
        QubitId second,
        double angle);
};

}  // namespace qubit

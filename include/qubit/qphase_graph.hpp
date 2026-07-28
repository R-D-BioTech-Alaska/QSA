#pragma once

#include "qubit/qstate.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace qubit {

struct PhaseGraphConfig {
    std::size_t max_edges{4'000'000};
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

}  // namespace qubit

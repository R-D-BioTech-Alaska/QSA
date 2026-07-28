#pragma once

#include "qubit/qstate.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace qubit {

struct StabilizerConfig {
    std::size_t max_tableau_bytes{1ULL << 30};
};

class StabilizerState {
public:
    explicit StabilizerState(
        std::size_t qubit_count,
        StabilizerConfig config = {});

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::size_t estimated_bytes() const noexcept;

    void apply_x(QubitId qubit);
    void apply_y(QubitId qubit);
    void apply_z(QubitId qubit);
    void apply_h(QubitId qubit);
    void apply_s(QubitId qubit);
    void apply_sdg(QubitId qubit);
    void apply_cnot(QubitId control, QubitId target);
    void apply_cz(QubitId first, QubitId second);
    void apply_swap(QubitId first, QubitId second);

    [[nodiscard]] double probability_one(QubitId qubit) const;
    [[nodiscard]] int measure_z(QubitId qubit, double sample);
    [[nodiscard]] std::vector<int> measure_all(std::uint64_t seed);

    [[nodiscard]] bool validate(std::string* reason = nullptr) const;
    [[nodiscard]] bool validate_full(std::string* reason = nullptr) const;
    [[nodiscard]] std::string describe() const;

private:
    std::size_t qubit_count_{0};
    std::size_t word_count_{0};
    StabilizerConfig config_{};
    std::vector<std::uint64_t> x_{};
    std::vector<std::uint64_t> z_{};
    std::vector<std::uint8_t> phase_{};

    [[nodiscard]] std::size_t row_count() const noexcept { return qubit_count_ * 2U; }
    [[nodiscard]] std::size_t offset(std::size_t row, std::size_t word) const noexcept {
        return row * word_count_ + word;
    }
    [[nodiscard]] bool bit(
        const std::vector<std::uint64_t>& table,
        std::size_t row,
        std::size_t qubit) const noexcept;
    void set_bit(
        std::vector<std::uint64_t>& table,
        std::size_t row,
        std::size_t qubit,
        bool value) noexcept;
    void clear_row(std::size_t row) noexcept;
    void copy_row(std::size_t target, std::size_t source) noexcept;
    void multiply_row(std::size_t target, std::size_t source) noexcept;
    void multiply_scratch(
        std::vector<std::uint64_t>& scratch_x,
        std::vector<std::uint64_t>& scratch_z,
        std::uint8_t& scratch_phase,
        std::size_t source) const noexcept;
    [[nodiscard]] bool symplectic_anticommutes(
        std::size_t first,
        std::size_t second) const noexcept;
    [[nodiscard]] int deterministic_z(QubitId qubit) const;
    [[nodiscard]] std::size_t random_measurement_row(QubitId qubit) const noexcept;
    void validate_qubit(QubitId qubit) const;
};

}  // namespace qubit

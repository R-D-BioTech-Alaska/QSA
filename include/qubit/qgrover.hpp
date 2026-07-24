#pragma once

#include "qubit/qstate.hpp"
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace qubit {

class GroverSearch {
public:
    GroverSearch(std::size_t qubit_count, std::span<const BasisIndex> marked_indices);

    [[nodiscard]] static GroverSearch from_marked_count(
        std::size_t qubit_count,
        BasisIndex marked_count);

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] BasisIndex space_size() const noexcept { return space_size_; }
    [[nodiscard]] BasisIndex marked_count() const noexcept { return marked_count_; }
    [[nodiscard]] BasisIndex unmarked_count() const noexcept {
        return space_size_ - marked_count_;
    }
    [[nodiscard]] bool has_explicit_marked_indices() const noexcept {
        return !marked_indices_.empty();
    }
    [[nodiscard]] std::uint64_t iteration_count() const noexcept { return iteration_count_; }

    void reset();
    void apply_oracle();
    void apply_diffusion();
    void iterate(std::uint64_t count = 1);
    void run_optimal();

    [[nodiscard]] std::uint64_t optimal_iterations() const noexcept;
    [[nodiscard]] double success_probability() const noexcept;
    [[nodiscard]] QComplex marked_amplitude() const noexcept { return marked_amplitude_; }
    [[nodiscard]] QComplex unmarked_amplitude() const noexcept { return unmarked_amplitude_; }
    [[nodiscard]] QComplex amplitude(BasisIndex basis_index) const;

    [[nodiscard]] bool sample_is_marked(double sample) const;
    [[nodiscard]] BasisIndex sample_basis(double branch_sample, double index_sample) const;
    [[nodiscard]] BasisIndex sample_basis(std::uint64_t seed) const;

    [[nodiscard]] std::vector<QComplex> materialize(std::size_t max_qubits = 24) const;
    [[nodiscard]] std::size_t estimated_bytes() const noexcept;
    [[nodiscard]] bool validate(std::string* reason = nullptr) const;
    [[nodiscard]] std::string describe() const;

private:
    struct CountOnlyTag {};

    GroverSearch(std::size_t qubit_count, BasisIndex marked_count, CountOnlyTag);

    std::size_t qubit_count_{0};
    BasisIndex space_size_{0};
    BasisIndex marked_count_{0};
    std::vector<BasisIndex> marked_indices_{};
    QComplex marked_amplitude_{};
    QComplex unmarked_amplitude_{};
    std::uint64_t iteration_count_{0};

    void initialize_space(std::size_t qubit_count, BasisIndex marked_count);
    [[nodiscard]] bool is_marked(BasisIndex basis_index) const;
    [[nodiscard]] BasisIndex select_unmarked(BasisIndex rank) const;
};

} 

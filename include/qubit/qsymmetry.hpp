#pragma once

#include "qubit/qstate.hpp"
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace qubit {

enum class SymmetryMembership : std::uint8_t {
    CountOnly = 0,
    OrderedRanges = 1,
    ExplicitLabels = 2,
    HammingWeight = 3,
};

class SymmetryState {
public:
    SymmetryState(std::size_t qubit_count, std::span<const BasisIndex> class_counts);

    [[nodiscard]] static SymmetryState from_counts(
        std::size_t qubit_count,
        std::span<const BasisIndex> class_counts);

    [[nodiscard]] static SymmetryState from_labels(
        std::size_t qubit_count,
        std::span<const std::uint32_t> labels);

    [[nodiscard]] static SymmetryState hamming_weight(std::size_t qubit_count);

    [[nodiscard]] static SymmetryState discover(
        const QRegister& state,
        std::size_t max_qubits = 24,
        double tolerance = 0.0,
        std::size_t max_classes = 1'000'000);

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] BasisIndex space_size() const noexcept { return space_size_; }
    [[nodiscard]] std::size_t class_count() const noexcept { return counts_.size(); }
    [[nodiscard]] SymmetryMembership membership() const noexcept { return membership_; }
    [[nodiscard]] bool has_basis_membership() const noexcept {
        return membership_ != SymmetryMembership::CountOnly;
    }
    [[nodiscard]] double discovery_error() const noexcept { return discovery_error_; }

    [[nodiscard]] BasisIndex class_size(std::size_t class_index) const;
    [[nodiscard]] QComplex class_amplitude(std::size_t class_index) const;
    [[nodiscard]] double class_probability(std::size_t class_index) const;
    [[nodiscard]] std::size_t class_for_basis(BasisIndex basis_index) const;
    [[nodiscard]] QComplex amplitude(BasisIndex basis_index) const;

    void reset_uniform();
    void set_class_amplitudes(std::span<const QComplex> amplitudes, bool normalize = true);
    void apply_class_phase(std::size_t class_index, double angle);
    void apply_class_phases(std::span<const double> angles);
    void apply_weighted_reflection();

    [[nodiscard]] std::size_t split_class(
        std::size_t class_index,
        BasisIndex first_count);

    [[nodiscard]] std::size_t merge_equivalent(double tolerance = 1e-12);

    void apply_class_unitary(std::span<const QComplex> matrix, double tolerance = 1e-10);
    void iterate_class_unitary(
        std::span<const QComplex> matrix,
        std::uint64_t count,
        double tolerance = 1e-10);

    [[nodiscard]] std::size_t sample_class(double sample) const;
    [[nodiscard]] BasisIndex sample_basis(double class_sample, double index_sample) const;
    [[nodiscard]] BasisIndex sample_basis(std::uint64_t seed) const;

    [[nodiscard]] std::vector<QComplex> materialize(std::size_t max_qubits = 24) const;
    [[nodiscard]] QRegister to_register(std::size_t max_qubits = 24) const;
    [[nodiscard]] std::size_t estimated_bytes() const noexcept;
    [[nodiscard]] bool validate(std::string* reason = nullptr) const;
    [[nodiscard]] std::string describe() const;

private:
    struct CountOnlyTag {};
    struct ExplicitLabelsTag {};

    SymmetryState(
        std::size_t qubit_count,
        std::span<const BasisIndex> class_counts,
        CountOnlyTag);
    SymmetryState(
        std::size_t qubit_count,
        std::span<const std::uint32_t> labels,
        ExplicitLabelsTag);

    std::size_t qubit_count_{0};
    BasisIndex space_size_{0};
    SymmetryMembership membership_{SymmetryMembership::CountOnly};
    std::vector<BasisIndex> counts_{};
    std::vector<BasisIndex> offsets_{};
    std::vector<std::uint32_t> labels_{};
    std::vector<QComplex> amplitudes_{};
    double discovery_error_{0.0};

    void initialize_space(std::size_t qubit_count);
    void initialize_counts(std::span<const BasisIndex> class_counts);
    void normalize();
    void validate_class(std::size_t class_index) const;

    [[nodiscard]] std::vector<QComplex> normalized_coefficients() const;
    void assign_normalized_coefficients(std::span<const QComplex> coefficients);
};

} 

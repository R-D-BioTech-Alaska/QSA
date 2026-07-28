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

// Exact amplitude-class representation for permutation-symmetric state spaces.
//
// Every basis state in one class shares one complex amplitude. Class sizes are
// retained explicitly, so normalization and class-space unitary evolution are
// exact without allocating the logical 2^n statevector. Operations that remain
// inside the class partition cost O(class_count) or O(class_count^3 log k), not
// O(2^n). A state can be materialized into QRegister when an operation breaks
// the symmetry and the requested fallback size is safe.
class SymmetryState {
public:
    // Classes occupy consecutive basis-index ranges in the supplied order.
    SymmetryState(std::size_t qubit_count, std::span<const BasisIndex> class_counts);

    // Class counts are known, but basis membership is intentionally symbolic.
    [[nodiscard]] static SymmetryState from_counts(
        std::size_t qubit_count,
        std::span<const BasisIndex> class_counts);

    // One class label per basis state. Labels must be dense in [0, class_count).
    [[nodiscard]] static SymmetryState from_labels(
        std::size_t qubit_count,
        std::span<const std::uint32_t> labels);

    // Permutation-invariant basis partition. Class k contains every basis
    // state with Hamming weight k, requiring only qubit_count + 1 amplitudes.
    [[nodiscard]] static SymmetryState hamming_weight(std::size_t qubit_count);

    // Discover amplitude equivalence classes in an existing register. The
    // default tolerance of zero is bit-exact. Nonzero tolerance is explicit,
    // bounded approximation and is reported through discovery_error().
    [[nodiscard]] static SymmetryState discover(
        const QRegister& state,
        std::size_t max_qubits = 24,
        double tolerance = 0.0,
        std::size_t max_classes = 1'000'000);

    // Discover exact count-only amplitude classes from QSA components without
    // materializing the full register. Basis membership remains symbolic.
    [[nodiscard]] static SymmetryState discover_components(
        const QRegister& state,
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

    // Refine one equivalence class without materializing the logical state.
    // The original class keeps first_count members and the returned class
    // receives the remainder with the same amplitude.
    [[nodiscard]] std::size_t split_class(
        std::size_t class_index,
        BasisIndex first_count);

    // Merge classes whose amplitudes are equal within tolerance. Ordered-range
    // states merge adjacent classes; symbolic and explicit states can merge any
    // equivalent classes. Returns the number of removed classes.
    [[nodiscard]] std::size_t merge_equivalent(double tolerance = 1e-12);

    // Matrix acts on normalized class coefficients c_i = sqrt(count_i) * a_i.
    // The matrix is row-major and must be unitary within tolerance.
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

}  // namespace qubit

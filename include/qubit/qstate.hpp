#pragma once

#include "qubit/qcomplex.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace qubit {

using QubitId = std::uint32_t;
using BasisIndex = std::uint64_t;

class QStateError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct QStateConfig {
    double epsilon{1e-12};
    double factor_tolerance{1e-10};
    std::size_t max_component_qubits{60};
    std::uint64_t max_dense_amplitudes{1ULL << 24};
    std::size_t max_sparse_entries{4'000'000};
};

struct QMatrix2 {
    std::array<QComplex, 4> values{};

    [[nodiscard]] constexpr const QComplex& operator()(std::size_t row, std::size_t column) const {
        return values[row * 2 + column];
    }
};

struct QMatrix4 {
    std::array<QComplex, 16> values{};

    [[nodiscard]] constexpr const QComplex& operator()(std::size_t row, std::size_t column) const {
        return values[row * 4 + column];
    }
};

struct QDiagonalPhase {
    QubitId qubit{0};
    QComplex zero{1.0, 0.0};
    QComplex one{1.0, 0.0};
};

namespace gates {
[[nodiscard]] QMatrix2 identity();
[[nodiscard]] QMatrix2 x();
[[nodiscard]] QMatrix2 y();
[[nodiscard]] QMatrix2 z();
[[nodiscard]] QMatrix2 h();
[[nodiscard]] QMatrix2 s();
[[nodiscard]] QMatrix2 sdg();
[[nodiscard]] QMatrix2 t();
[[nodiscard]] QMatrix2 tdg();
[[nodiscard]] QMatrix2 rx(double theta);
[[nodiscard]] QMatrix2 ry(double theta);
[[nodiscard]] QMatrix2 rz(double theta);
[[nodiscard]] QMatrix4 cnot();
[[nodiscard]] QMatrix4 cz();
[[nodiscard]] QMatrix4 swap();
} 

struct BlochCell {
    double x{0.0};
    double y{0.0};
    double z{1.0};

    void normalize(double epsilon = 1e-12);
    [[nodiscard]] double probability_one() const noexcept;
    [[nodiscard]] std::array<QComplex, 2> amplitudes(double epsilon = 1e-12) const;
    [[nodiscard]] static BlochCell from_amplitudes(
        QComplex zero,
        QComplex one,
        double epsilon = 1e-12);

    void apply_x() noexcept;
    void apply_y() noexcept;
    void apply_z() noexcept;
    void apply_h() noexcept;
    void rotate_x(double theta) noexcept;
    void rotate_y(double theta) noexcept;
    void rotate_z(double theta) noexcept;
};

enum class StorageMode : std::uint8_t {
    Sparse = 1,
    Dense = 2,
};

enum class ComponentKind : std::uint8_t {
    Cell = 0,
    Sparse = 1,
    Dense = 2,
};

class AmplitudeStore {
public:
    using SparseEntry = std::pair<BasisIndex, QComplex>;

    AmplitudeStore() = default;

    [[nodiscard]] static AmplitudeStore from_entries(
        BasisIndex dimension,
        std::vector<SparseEntry> entries,
        const QStateConfig& config,
        bool normalize = true);

    [[nodiscard]] static AmplitudeStore from_dense(
        std::vector<QComplex> amplitudes,
        const QStateConfig& config,
        bool normalize = true);

    [[nodiscard]] BasisIndex dimension() const noexcept { return dimension_; }
    [[nodiscard]] StorageMode mode() const noexcept { return mode_; }
    [[nodiscard]] std::size_t nonzero_count() const noexcept;
    [[nodiscard]] QComplex at(BasisIndex index) const;
    [[nodiscard]] std::vector<SparseEntry> entries(double epsilon = 0.0) const;
    [[nodiscard]] std::vector<QComplex> dense_copy() const;
    [[nodiscard]] std::size_t estimated_bytes() const noexcept;

    void normalize(double epsilon = 1e-12);
    void apply_single(
        std::size_t bit_position,
        const QMatrix2& matrix,
        const QStateConfig& config,
        bool renormalize = true);
    void apply_two(
        std::size_t first_bit,
        std::size_t second_bit,
        const QMatrix4& matrix,
        const QStateConfig& config,
        bool renormalize = true);

    void apply_x(std::size_t bit_position);
    void apply_y(std::size_t bit_position);
    void apply_z(std::size_t bit_position);
    void apply_phase(std::size_t bit_position, QComplex phase_zero, QComplex phase_one);
    void apply_cnot(std::size_t control_bit, std::size_t target_bit);
    void apply_cz(std::size_t first_bit, std::size_t second_bit);
    void apply_swap(std::size_t first_bit, std::size_t second_bit);

private:
    BasisIndex dimension_{0};
    StorageMode mode_{StorageMode::Sparse};
    std::vector<QComplex> dense_{};
    std::vector<SparseEntry> sparse_{};

    void assign_entries(
        BasisIndex dimension,
        std::vector<SparseEntry> entries,
        const QStateConfig& config,
        bool normalize_values);
    void assign_sorted_entries(
        BasisIndex dimension,
        std::vector<SparseEntry> entries,
        const QStateConfig& config,
        bool normalize_values);
    void rebalance(const QStateConfig& config);
    void sort_sparse();

    friend class QRegister;
};

struct StateComponent {
    std::vector<QubitId> qubits{};
    std::variant<BlochCell, AmplitudeStore> state{BlochCell{}};

    [[nodiscard]] bool is_cell() const noexcept {
        return std::holds_alternative<BlochCell>(state);
    }
};

class QStateCodec;

class QRegister {
public:
    explicit QRegister(std::size_t qubit_count, QStateConfig config = {});
    [[nodiscard]] static QRegister from_amplitudes(
        std::vector<QComplex> amplitudes,
        QStateConfig config = {});

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] std::size_t component_count() const noexcept { return components_.size(); }
    [[nodiscard]] const QStateConfig& config() const noexcept { return config_; }

    void apply_x(QubitId qubit);
    void apply_y(QubitId qubit);
    void apply_z(QubitId qubit);
    void apply_h(QubitId qubit);
    void apply_s(QubitId qubit);
    void apply_sdg(QubitId qubit);
    void apply_t(QubitId qubit);
    void apply_tdg(QubitId qubit);
    void apply_rx(QubitId qubit, double theta);
    void apply_ry(QubitId qubit, double theta);
    void apply_rz(QubitId qubit, double theta);
    void apply_single(QubitId qubit, const QMatrix2& matrix);
    void apply_diagonal(std::span<const QDiagonalPhase> phases);

    void apply_cnot(QubitId control, QubitId target);
    void apply_cz(QubitId first, QubitId second);
    void apply_swap(QubitId first, QubitId second);
    void apply_two(QubitId first, QubitId second, const QMatrix4& matrix);

    void apply_grover_oracle(std::span<const BasisIndex> marked_indices);
    void apply_grover_diffusion();
    void apply_grover_iterations(
        std::span<const BasisIndex> marked_indices,
        std::uint64_t iteration_count = 1);

    void apply_bit_flip_trajectory(QubitId qubit, double probability, double sample);
    void apply_phase_flip_trajectory(QubitId qubit, double probability, double sample);
    void apply_depolarizing_trajectory(QubitId qubit, double probability, double sample);
    void apply_amplitude_damping_trajectory(QubitId qubit, double gamma, double sample);

    [[nodiscard]] double probability_one(QubitId qubit) const;
    [[nodiscard]] std::vector<double> probabilities_one() const;
    [[nodiscard]] int measure(QubitId qubit, double sample);
    [[nodiscard]] std::vector<int> measure_all(std::uint64_t seed);

    [[nodiscard]] QComplex amplitude(BasisIndex global_basis_index) const;
    [[nodiscard]] QComplex amplitude_bits(std::span<const std::uint8_t> bits) const;
    [[nodiscard]] std::vector<QComplex> materialize(std::size_t max_qubits = 24) const;

    [[nodiscard]] std::size_t component_size(QubitId qubit) const;
    [[nodiscard]] StorageMode component_storage_mode(QubitId qubit) const;
    [[nodiscard]] ComponentKind component_kind(QubitId qubit) const;
    [[nodiscard]] std::size_t component_nonzero_count(QubitId qubit) const;
    [[nodiscard]] std::size_t estimated_bytes() const noexcept;
    [[nodiscard]] std::string describe() const;
    [[nodiscard]] bool validate(std::string* reason = nullptr) const;

    [[nodiscard]] std::vector<std::uint8_t> encode_qsc() const;
    [[nodiscard]] static QRegister decode_qsc(std::span<const std::uint8_t> bytes);

private:
    std::size_t qubit_count_{0};
    QStateConfig config_{};
    std::vector<StateComponent> components_{};
    std::vector<std::uint32_t> component_order_{};
    std::uint64_t next_component_order_{0};
    std::vector<std::size_t> qubit_component_{};

    void validate_qubit(QubitId qubit) const;
    void reindex_components();
    [[nodiscard]] std::vector<std::size_t> ordered_component_indices() const;
    void renumber_component_order();
    [[nodiscard]] std::size_t append_component(StateComponent component);
    void remove_component(std::size_t component_index);
    [[nodiscard]] std::size_t component_index(QubitId qubit) const;
    [[nodiscard]] std::size_t local_position(const StateComponent& component, QubitId qubit) const;
    void apply_cell_matrix(BlochCell& cell, const QMatrix2& matrix);
    [[nodiscard]] std::size_t merge_components(std::size_t first, std::size_t second);
    void compact_component(std::size_t component_index);
    void compact_component_targets(
        std::size_t component_index,
        std::span<const QubitId> candidate_qubits);
    [[nodiscard]] std::optional<std::pair<StateComponent, StateComponent>>
    factor_singleton(const StateComponent& component, std::size_t local_position) const;
    void collapse_patch(std::size_t component_index, std::size_t local_position, int outcome);
    void apply_nonunitary_single(QubitId qubit, const QMatrix2& matrix);
    [[nodiscard]] std::size_t promote_global_dense();

    friend class QStateCodec;
};

class QStateCodec {
public:
    [[nodiscard]] static std::vector<std::uint8_t> encode(const QRegister& state);
    [[nodiscard]] static QRegister decode(std::span<const std::uint8_t> bytes);
};

} 

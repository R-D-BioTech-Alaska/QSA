#pragma once

#include "qubit/qpauli.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace qubit {

struct MPSSiteTensor {
    std::size_t left_dimension{0};
    std::size_t right_dimension{0};
    std::vector<QComplex> zero{};
    std::vector<QComplex> one{};
};

struct MPSConfig {
    std::size_t max_bond_dimension{65'536};
    std::size_t max_scalars{4'000'000};
    std::size_t max_materialize_qubits{20};
    double normalization_tolerance{1e-9};
};

class MatrixProductState {
public:
    explicit MatrixProductState(
        std::vector<MPSSiteTensor> sites,
        MPSConfig config = {});

    [[nodiscard]] static MatrixProductState ghz(
        std::size_t qubit_count,
        double phase = 0.0,
        MPSConfig config = {});
    [[nodiscard]] static MatrixProductState w(
        std::size_t qubit_count,
        MPSConfig config = {});
    [[nodiscard]] static MatrixProductState cluster(
        std::size_t qubit_count,
        MPSConfig config = {});

    [[nodiscard]] std::size_t qubit_count() const noexcept { return sites_.size(); }
    [[nodiscard]] std::size_t max_bond_dimension() const noexcept;
    [[nodiscard]] std::size_t scalar_count() const noexcept;
    [[nodiscard]] std::size_t estimated_bytes() const noexcept;
    [[nodiscard]] const MPSConfig& config() const noexcept { return config_; }
    [[nodiscard]] const std::vector<MPSSiteTensor>& sites() const noexcept { return sites_; }

    [[nodiscard]] QComplex amplitude(std::span<const std::uint8_t> bits) const;
    [[nodiscard]] QComplex pauli_expectation(std::span<const PauliAxis> axes) const;
    [[nodiscard]] QComplex expectation(const PauliObservable& observable) const;
    [[nodiscard]] double norm2() const;
    [[nodiscard]] std::vector<QComplex> materialize() const;
    [[nodiscard]] bool validate(std::string* reason = nullptr) const;

private:
    std::vector<MPSSiteTensor> sites_{};
    MPSConfig config_{};

    [[nodiscard]] bool validate_structure(std::string* reason) const;
    [[nodiscard]] QComplex product_expectation(std::span<const PauliAxis> axes) const;
};

[[nodiscard]] std::size_t required_schmidt_rank_cross_cut_bell_pairs(
    std::size_t pair_count);
[[nodiscard]] bool bond_dimension_accepts_cross_cut_bell_pairs(
    std::size_t pair_count,
    std::size_t bond_dimension);

}  // namespace qubit

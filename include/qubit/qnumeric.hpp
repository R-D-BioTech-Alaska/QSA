#pragma once

#include "qubit/qcomplex.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace qubit {

enum class NumericBackend : std::uint8_t {
    Scalar = 0,
    Avx2Fma = 1,
};

struct NumericConfig {
    std::size_t worker_count{0};
    std::size_t grain_size{16'384};
    bool enable_simd{true};
};

class NumericExecutor {
public:
    explicit NumericExecutor(NumericConfig config = {});
    ~NumericExecutor();

    NumericExecutor(const NumericExecutor&) = delete;
    NumericExecutor& operator=(const NumericExecutor&) = delete;
    NumericExecutor(NumericExecutor&&) = delete;
    NumericExecutor& operator=(NumericExecutor&&) = delete;

    [[nodiscard]] std::size_t worker_count() const noexcept;
    [[nodiscard]] std::size_t grain_size() const noexcept;
    [[nodiscard]] NumericBackend backend() const noexcept;
    [[nodiscard]] const char* backend_name() const noexcept;

    void fused_affine(
        std::span<const double> first,
        std::span<const double> second,
        double first_scale,
        double second_scale,
        double bias,
        std::span<double> output) const;

    [[nodiscard]] double fused_affine_norm2(
        std::span<const double> first,
        std::span<const double> second,
        double first_scale,
        double second_scale,
        double bias,
        std::span<double> output) const;

    void fused_complex_affine(
        std::span<const QComplex> first,
        std::span<const QComplex> second,
        QComplex first_scale,
        QComplex second_scale,
        QComplex bias,
        std::span<QComplex> output) const;

    [[nodiscard]] double fused_complex_affine_norm2(
        std::span<const QComplex> first,
        std::span<const QComplex> second,
        QComplex first_scale,
        QComplex second_scale,
        QComplex bias,
        std::span<QComplex> output) const;

    [[nodiscard]] double dot(
        std::span<const double> first,
        std::span<const double> second) const;

    [[nodiscard]] QComplex inner_product(
        std::span<const QComplex> first,
        std::span<const QComplex> second) const;

    void matrix2_batch(
        std::span<const QComplex, 4> matrix,
        std::span<const QComplex> input,
        std::span<QComplex> output) const;

    void matrix4_batch(
        std::span<const QComplex, 16> matrix,
        std::span<const QComplex> input,
        std::span<QComplex> output) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace qubit

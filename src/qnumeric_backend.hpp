#pragma once

#include "qubit/qcomplex.hpp"

#include <cstddef>

namespace qubit::numeric_detail {

#if defined(QSTATE_NUMERIC_HAS_AVX2)
[[nodiscard]] bool avx2_fma_available() noexcept;

void fused_affine_avx2(
    const double* first,
    const double* second,
    double first_scale,
    double second_scale,
    double bias,
    double* output,
    std::size_t size) noexcept;

[[nodiscard]] double fused_affine_norm2_avx2(
    const double* first,
    const double* second,
    double first_scale,
    double second_scale,
    double bias,
    double* output,
    std::size_t size) noexcept;

[[nodiscard]] double dot_avx2(
    const double* first,
    const double* second,
    std::size_t size) noexcept;

void fused_complex_affine_avx2(
    const QComplex* first,
    const QComplex* second,
    QComplex first_scale,
    QComplex second_scale,
    QComplex bias,
    QComplex* output,
    std::size_t size) noexcept;

[[nodiscard]] double fused_complex_affine_norm2_avx2(
    const QComplex* first,
    const QComplex* second,
    QComplex first_scale,
    QComplex second_scale,
    QComplex bias,
    QComplex* output,
    std::size_t size) noexcept;

void matrix2_batch_avx2(
    const QComplex* matrix,
    const QComplex* input,
    QComplex* output,
    std::size_t vector_count) noexcept;

void matrix4_batch_avx2(
    const QComplex* matrix,
    const QComplex* input,
    QComplex* output,
    std::size_t vector_count) noexcept;
#endif

}  // namespace qubit::numeric_detail

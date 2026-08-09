#include "qnumeric_backend.hpp"

#if defined(QSTATE_NUMERIC_HAS_AVX2)

#include <immintrin.h>

#include <cmath>

namespace qubit::numeric_detail {
namespace {

[[nodiscard]] __m256d load_complex2(const QComplex* values) noexcept {
    return _mm256_setr_pd(
        values[0].re, values[0].im, values[1].re, values[1].im);
}

void store_complex2(QComplex* values, __m256d packed) noexcept {
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, packed);
    values[0] = {lanes[0], lanes[1]};
    values[1] = {lanes[2], lanes[3]};
}

[[nodiscard]] QComplex scalar_multiply(QComplex first, QComplex second) noexcept {
    return {
        std::fma(first.re, second.re, -first.im * second.im),
        std::fma(first.re, second.im, first.im * second.re),
    };
}

[[nodiscard]] __m256d complex_multiply_pair(
    __m256d left,
    __m256d right) noexcept {
    const __m256d left_real = _mm256_permute4x64_pd(left, 0xA0);
    const __m256d left_imag = _mm256_permute4x64_pd(left, 0xF5);
    const __m256d right_swapped = _mm256_permute4x64_pd(right, 0xB1);
    return _mm256_fmaddsub_pd(
        left_real, right, _mm256_mul_pd(left_imag, right_swapped));
}

[[nodiscard]] __m128d complex_dot2(
    __m256d values,
    __m256d coefficients) noexcept {
    const __m256d products = complex_multiply_pair(coefficients, values);
    return _mm_add_pd(
        _mm256_castpd256_pd128(products),
        _mm256_extractf128_pd(products, 1));
}

[[nodiscard]] __m128d complex_dot4(
    __m256d first_values,
    __m256d second_values,
    __m256d first_coefficients,
    __m256d second_coefficients) noexcept {
    const __m256d first_products = complex_multiply_pair(
        first_coefficients, first_values);
    const __m256d second_products = complex_multiply_pair(
        second_coefficients, second_values);
    __m128d sum = _mm256_castpd256_pd128(first_products);
    sum = _mm_add_pd(sum, _mm256_extractf128_pd(first_products, 1));
    sum = _mm_add_pd(sum, _mm256_castpd256_pd128(second_products));
    return _mm_add_pd(sum, _mm256_extractf128_pd(second_products, 1));
}

[[nodiscard]] __m256d pack_complex2(__m128d first, __m128d second) noexcept {
    __m256d packed = _mm256_castpd128_pd256(first);
    return _mm256_insertf128_pd(packed, second, 1);
}

[[nodiscard]] double horizontal_sum(__m256d values) noexcept {
    const __m128d pair = _mm_add_pd(
        _mm256_castpd256_pd128(values),
        _mm256_extractf128_pd(values, 1));
    alignas(16) double lanes[2];
    _mm_store_pd(lanes, pair);
    return lanes[0] + lanes[1];
}

[[nodiscard]] __m256d repeated_complex(QComplex value) noexcept {
    return _mm256_setr_pd(value.re, value.im, value.re, value.im);
}

}  // namespace

void fused_affine_avx2(
    const double* first,
    const double* second,
    double first_scale,
    double second_scale,
    double bias,
    double* output,
    std::size_t size) noexcept {
    const __m256d first_scale_vector = _mm256_set1_pd(first_scale);
    const __m256d second_scale_vector = _mm256_set1_pd(second_scale);
    const __m256d bias_vector = _mm256_set1_pd(bias);
    std::size_t index = 0U;
    for (; index + 4U <= size; index += 4U) {
        const __m256d left = _mm256_loadu_pd(first + index);
        const __m256d right = _mm256_loadu_pd(second + index);
        const __m256d value = _mm256_fmadd_pd(
            left,
            first_scale_vector,
            _mm256_fmadd_pd(right, second_scale_vector, bias_vector));
        _mm256_storeu_pd(output + index, value);
    }
    for (; index < size; ++index) {
        output[index] = std::fma(
            first[index], first_scale, std::fma(second[index], second_scale, bias));
    }
}

double fused_affine_norm2_avx2(
    const double* first,
    const double* second,
    double first_scale,
    double second_scale,
    double bias,
    double* output,
    std::size_t size) noexcept {
    const __m256d first_scale_vector = _mm256_set1_pd(first_scale);
    const __m256d second_scale_vector = _mm256_set1_pd(second_scale);
    const __m256d bias_vector = _mm256_set1_pd(bias);
    __m256d sum = _mm256_setzero_pd();
    std::size_t index = 0U;
    for (; index + 4U <= size; index += 4U) {
        const __m256d left = _mm256_loadu_pd(first + index);
        const __m256d right = _mm256_loadu_pd(second + index);
        const __m256d value = _mm256_fmadd_pd(
            left,
            first_scale_vector,
            _mm256_fmadd_pd(right, second_scale_vector, bias_vector));
        _mm256_storeu_pd(output + index, value);
        sum = _mm256_fmadd_pd(value, value, sum);
    }
    double result = horizontal_sum(sum);
    for (; index < size; ++index) {
        const double value = std::fma(
            first[index], first_scale, std::fma(second[index], second_scale, bias));
        output[index] = value;
        result = std::fma(value, value, result);
    }
    return result;
}

double dot_avx2(
    const double* first,
    const double* second,
    std::size_t size) noexcept {
    __m256d sum = _mm256_setzero_pd();
    std::size_t index = 0U;
    for (; index + 4U <= size; index += 4U) {
        sum = _mm256_fmadd_pd(
            _mm256_loadu_pd(first + index),
            _mm256_loadu_pd(second + index),
            sum);
    }
    double result = horizontal_sum(sum);
    for (; index < size; ++index) {
        result = std::fma(first[index], second[index], result);
    }
    return result;
}

void fused_complex_affine_avx2(
    const QComplex* first,
    const QComplex* second,
    QComplex first_scale,
    QComplex second_scale,
    QComplex bias,
    QComplex* output,
    std::size_t size) noexcept {
    const __m256d first_coefficients = repeated_complex(first_scale);
    const __m256d second_coefficients = repeated_complex(second_scale);
    const __m256d bias_vector = repeated_complex(bias);
    std::size_t index = 0U;
    for (; index + 2U <= size; index += 2U) {
        const __m256d left = complex_multiply_pair(
            load_complex2(first + index), first_coefficients);
        const __m256d right = complex_multiply_pair(
            load_complex2(second + index), second_coefficients);
        store_complex2(output + index, _mm256_add_pd(_mm256_add_pd(left, right), bias_vector));
    }
    for (; index < size; ++index) {
        const QComplex left = scalar_multiply(first[index], first_scale);
        const QComplex right = scalar_multiply(second[index], second_scale);
        output[index] = left + right + bias;
    }
}

double fused_complex_affine_norm2_avx2(
    const QComplex* first,
    const QComplex* second,
    QComplex first_scale,
    QComplex second_scale,
    QComplex bias,
    QComplex* output,
    std::size_t size) noexcept {
    const __m256d first_coefficients = repeated_complex(first_scale);
    const __m256d second_coefficients = repeated_complex(second_scale);
    const __m256d bias_vector = repeated_complex(bias);
    __m256d sum = _mm256_setzero_pd();
    std::size_t index = 0U;
    for (; index + 2U <= size; index += 2U) {
        const __m256d left = complex_multiply_pair(
            load_complex2(first + index), first_coefficients);
        const __m256d right = complex_multiply_pair(
            load_complex2(second + index), second_coefficients);
        const __m256d value = _mm256_add_pd(_mm256_add_pd(left, right), bias_vector);
        store_complex2(output + index, value);
        sum = _mm256_fmadd_pd(value, value, sum);
    }
    double result = horizontal_sum(sum);
    for (; index < size; ++index) {
        const QComplex value = scalar_multiply(first[index], first_scale) +
                               scalar_multiply(second[index], second_scale) + bias;
        output[index] = value;
        result += value.norm2();
    }
    return result;
}

void matrix2_batch_avx2(
    const QComplex* matrix,
    const QComplex* input,
    QComplex* output,
    std::size_t vector_count) noexcept {
    const __m256d row0 = load_complex2(matrix);
    const __m256d row1 = load_complex2(matrix + 2U);
    for (std::size_t vector = 0; vector < vector_count; ++vector) {
        const __m256d values = load_complex2(input + vector * 2U);
        store_complex2(
            output + vector * 2U,
            pack_complex2(complex_dot2(values, row0), complex_dot2(values, row1)));
    }
}

void matrix4_batch_avx2(
    const QComplex* matrix,
    const QComplex* input,
    QComplex* output,
    std::size_t vector_count) noexcept {
    for (std::size_t vector = 0; vector < vector_count; ++vector) {
        const QComplex* source = input + vector * 4U;
        const __m256d first_values = load_complex2(source);
        const __m256d second_values = load_complex2(source + 2U);
        __m128d rows[4];
        for (std::size_t row = 0; row < 4U; ++row) {
            const QComplex* coefficients = matrix + row * 4U;
            rows[row] = complex_dot4(
                first_values, second_values,
                load_complex2(coefficients), load_complex2(coefficients + 2U));
        }
        QComplex* destination = output + vector * 4U;
        store_complex2(destination, pack_complex2(rows[0], rows[1]));
        store_complex2(destination + 2U, pack_complex2(rows[2], rows[3]));
    }
}

}  // namespace qubit::numeric_detail

#endif

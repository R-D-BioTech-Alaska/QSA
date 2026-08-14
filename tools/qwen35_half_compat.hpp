#pragma once

#include <cstdint>
#include <immintrin.h>

inline float qwen35_half_to_float(std::uint16_t value) noexcept {
    const __m128i half = _mm_cvtsi32_si128(static_cast<int>(value));
    return _mm_cvtss_f32(_mm_cvtph_ps(half));
}

inline std::uint16_t qwen35_float_to_half(float value) noexcept {
    const __m128 input = _mm_set_ss(value);
    return static_cast<std::uint16_t>(_mm_cvtsi128_si32(_mm_cvtps_ph(input, 0)) & 0xffff);
}

#ifndef _cvtsh_ss
#define _cvtsh_ss(value) qwen35_half_to_float(static_cast<std::uint16_t>(value))
#endif

#ifndef _cvtss_sh
#define _cvtss_sh(value, rounding) qwen35_float_to_half(value)
#endif

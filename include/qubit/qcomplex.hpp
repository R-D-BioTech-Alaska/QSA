#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace qubit {

struct QComplex {
    double re{0.0};
    double im{0.0};

    constexpr QComplex() = default;
    constexpr QComplex(double real, double imag = 0.0) : re(real), im(imag) {}

    [[nodiscard]] constexpr QComplex conjugate() const noexcept { return {re, -im}; }
    [[nodiscard]] constexpr double norm2() const noexcept { return re * re + im * im; }
    [[nodiscard]] double magnitude() const noexcept { return std::sqrt(norm2()); }
    [[nodiscard]] double phase() const noexcept { return std::atan2(im, re); }

    [[nodiscard]] static QComplex from_polar(double magnitude, double phase) noexcept {
        return {magnitude * std::cos(phase), magnitude * std::sin(phase)};
    }

    constexpr QComplex& operator+=(const QComplex& other) noexcept {
        re += other.re;
        im += other.im;
        return *this;
    }

    constexpr QComplex& operator-=(const QComplex& other) noexcept {
        re -= other.re;
        im -= other.im;
        return *this;
    }

    constexpr QComplex& operator*=(const QComplex& other) noexcept {
        const double next_re = re * other.re - im * other.im;
        const double next_im = re * other.im + im * other.re;
        re = next_re;
        im = next_im;
        return *this;
    }

    QComplex& operator/=(const QComplex& other) {
        const double denominator = other.norm2();
        if (denominator <= std::numeric_limits<double>::min()) {
            throw std::domain_error("QComplex division by zero");
        }
        const double next_re = (re * other.re + im * other.im) / denominator;
        const double next_im = (im * other.re - re * other.im) / denominator;
        re = next_re;
        im = next_im;
        return *this;
    }

    constexpr QComplex& operator*=(double scalar) noexcept {
        re *= scalar;
        im *= scalar;
        return *this;
    }

    QComplex& operator/=(double scalar) {
        if (std::abs(scalar) <= std::numeric_limits<double>::min()) {
            throw std::domain_error("QComplex division by zero scalar");
        }
        re /= scalar;
        im /= scalar;
        return *this;
    }
};

[[nodiscard]] constexpr inline QComplex operator+(QComplex lhs, const QComplex& rhs) noexcept {
    lhs += rhs;
    return lhs;
}

[[nodiscard]] constexpr inline QComplex operator-(QComplex lhs, const QComplex& rhs) noexcept {
    lhs -= rhs;
    return lhs;
}

[[nodiscard]] constexpr inline QComplex operator-(const QComplex& value) noexcept {
    return {-value.re, -value.im};
}

[[nodiscard]] constexpr inline QComplex operator*(QComplex lhs, const QComplex& rhs) noexcept {
    lhs *= rhs;
    return lhs;
}

[[nodiscard]] constexpr inline QComplex operator*(QComplex lhs, double rhs) noexcept {
    lhs *= rhs;
    return lhs;
}

[[nodiscard]] constexpr inline QComplex operator*(double lhs, QComplex rhs) noexcept {
    rhs *= lhs;
    return rhs;
}

[[nodiscard]] inline QComplex operator/(QComplex lhs, const QComplex& rhs) {
    lhs /= rhs;
    return lhs;
}

[[nodiscard]] inline QComplex operator/(QComplex lhs, double rhs) {
    lhs /= rhs;
    return lhs;
}

[[nodiscard]] constexpr inline bool operator==(const QComplex& lhs, const QComplex& rhs) noexcept {
    return lhs.re == rhs.re && lhs.im == rhs.im;
}

[[nodiscard]] inline bool almost_equal(
    const QComplex& lhs,
    const QComplex& rhs,
    double tolerance = 1e-12) noexcept {
    const double scale = 1.0 + std::max(lhs.magnitude(), rhs.magnitude());
    return (lhs - rhs).magnitude() <= tolerance * scale;
}

inline constexpr QComplex QI{0.0, 1.0};

} 

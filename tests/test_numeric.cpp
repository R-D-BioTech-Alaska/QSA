#include "qubit/qkron.hpp"
#include "qubit/qnumeric.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using qubit::KronVector;
using qubit::NumericConfig;
using qubit::NumericExecutor;
using qubit::QComplex;
using qubit::QStateError;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(double actual, double expected, double tolerance, const char* message) {
    if (std::abs(actual - expected) > tolerance * (1.0 + std::abs(expected))) {
        throw std::runtime_error(message);
    }
}

void require_close(QComplex actual, QComplex expected, double tolerance, const char* message) {
    if (!qubit::almost_equal(actual, expected, tolerance)) {
        throw std::runtime_error(message);
    }
}

[[nodiscard]] QComplex multiply(QComplex first, QComplex second) noexcept {
    return {
        std::fma(first.re, second.re, -first.im * second.im),
        std::fma(first.re, second.im, first.im * second.re),
    };
}

[[nodiscard]] QComplex affine_value(
    QComplex first,
    QComplex second,
    QComplex first_scale,
    QComplex second_scale,
    QComplex bias) noexcept {
    const QComplex left = multiply(first, first_scale);
    const QComplex right = multiply(second, second_scale);
    return {left.re + right.re + bias.re, left.im + right.im + bias.im};
}

[[nodiscard]] std::vector<double> real_data(std::size_t size, double shift) {
    std::vector<double> values(size);
    for (std::size_t index = 0; index < size; ++index) {
        const double x = static_cast<double>(index + 1U);
        values[index] = std::sin(0.0013 * x + shift) +
                        0.25 * std::cos(0.0007 * x - 0.5 * shift);
    }
    return values;
}

[[nodiscard]] std::vector<QComplex> complex_data(std::size_t size, double shift) {
    std::vector<QComplex> values(size);
    for (std::size_t index = 0; index < size; ++index) {
        const double x = static_cast<double>(index + 1U);
        values[index] = {
            std::sin(0.0011 * x + shift),
            0.7 * std::cos(0.0009 * x - shift),
        };
    }
    return values;
}

[[nodiscard]] bool same_bits(double first, double second) noexcept {
    return std::bit_cast<std::uint64_t>(first) == std::bit_cast<std::uint64_t>(second);
}

}  // namespace

int main() {
    constexpr std::size_t size = 32'777U;
    const NumericConfig serial_config{1U, 257U};
    const NumericConfig parallel_config{4U, 257U};
    NumericExecutor serial(serial_config);
    NumericExecutor parallel(parallel_config);

    require(serial.worker_count() == 1U, "serial numeric worker count changed");
    require(parallel.worker_count() == 4U, "parallel numeric worker count changed");
    require(serial.grain_size() == 257U && parallel.grain_size() == 257U,
            "numeric grain size changed");

    {
        const std::vector<double> first = real_data(size, 0.11);
        const std::vector<double> second = real_data(size, -0.27);
        std::vector<double> output(size);
        std::vector<double> expected(size);
        constexpr double first_scale = 1.25;
        constexpr double second_scale = -0.375;
        constexpr double bias = 0.0625;
        for (std::size_t index = 0; index < size; ++index) {
            expected[index] = std::fma(
                first[index], first_scale,
                std::fma(second[index], second_scale, bias));
        }

        parallel.fused_affine(
            first, second, first_scale, second_scale, bias, output);
        for (std::size_t index = 0; index < size; ++index) {
            require(same_bits(output[index], expected[index]),
                    "fused real affine changed scalar arithmetic");
        }

        std::vector<double> in_place_first = first;
        parallel.fused_affine(
            in_place_first, second, first_scale, second_scale, bias, in_place_first);
        require(in_place_first == expected, "fused real affine failed first-input in-place execution");

        std::vector<double> in_place_second = second;
        parallel.fused_affine(
            first, in_place_second, first_scale, second_scale, bias, in_place_second);
        require(in_place_second == expected, "fused real affine failed second-input in-place execution");

        std::vector<double> serial_output(size);
        std::vector<double> parallel_output(size);
        const double serial_norm = serial.fused_affine_norm2(
            first, second, first_scale, second_scale, bias, serial_output);
        const double parallel_norm = parallel.fused_affine_norm2(
            first, second, first_scale, second_scale, bias, parallel_output);
        require(serial_output == parallel_output,
                "parallel fused real affine changed element results");
        require(same_bits(serial_norm, parallel_norm),
                "parallel fused real reduction changed deterministic result");
    }

    {
        const std::vector<QComplex> first = complex_data(size, 0.21);
        const std::vector<QComplex> second = complex_data(size, -0.13);
        const QComplex first_scale{0.75, -0.25};
        const QComplex second_scale{-0.125, 0.5};
        const QComplex bias{0.03125, -0.0625};
        std::vector<QComplex> expected(size);
        for (std::size_t index = 0; index < size; ++index) {
            expected[index] = affine_value(
                first[index], second[index], first_scale, second_scale, bias);
        }

        std::vector<QComplex> output(size);
        parallel.fused_complex_affine(
            first, second, first_scale, second_scale, bias, output);
        for (std::size_t index = 0; index < size; ++index) {
            require(output[index] == expected[index],
                    "fused complex affine changed scalar arithmetic");
        }

        std::vector<QComplex> in_place = first;
        parallel.fused_complex_affine(
            in_place, second, first_scale, second_scale, bias, in_place);
        require(in_place == expected,
                "fused complex affine failed in-place execution");

        std::vector<QComplex> serial_output(size);
        std::vector<QComplex> parallel_output(size);
        const double serial_norm = serial.fused_complex_affine_norm2(
            first, second, first_scale, second_scale, bias, serial_output);
        const double parallel_norm = parallel.fused_complex_affine_norm2(
            first, second, first_scale, second_scale, bias, parallel_output);
        require(serial_output == parallel_output,
                "parallel fused complex affine changed element results");
        require(same_bits(serial_norm, parallel_norm),
                "parallel fused complex reduction changed deterministic result");
    }

    {
        const std::vector<double> first = real_data(size, 0.37);
        const std::vector<double> second = real_data(size, -0.41);
        const double serial_value = serial.dot(first, second);
        const double parallel_value = parallel.dot(first, second);
        require(same_bits(serial_value, parallel_value),
                "parallel dot product changed deterministic result");

        const std::vector<QComplex> complex_first = complex_data(size, 0.43);
        const std::vector<QComplex> complex_second = complex_data(size, -0.47);
        const QComplex serial_inner = serial.inner_product(complex_first, complex_second);
        const QComplex parallel_inner = parallel.inner_product(complex_first, complex_second);
        require(serial_inner == parallel_inner,
                "parallel complex inner product changed deterministic result");
    }

    {
        const std::array<QComplex, 4> matrix{{
            {0.5, 0.25}, {-0.125, 0.75},
            {0.625, -0.375}, {0.25, 0.125},
        }};
        std::vector<QComplex> input = complex_data(8'194U, 0.19);
        std::vector<QComplex> expected(input.size());
        for (std::size_t offset = 0; offset < input.size(); offset += 2U) {
            expected[offset] = multiply(matrix[0], input[offset]) +
                               multiply(matrix[1], input[offset + 1U]);
            expected[offset + 1U] = multiply(matrix[2], input[offset]) +
                                    multiply(matrix[3], input[offset + 1U]);
        }
        std::vector<QComplex> output(input.size());
        parallel.matrix2_batch(std::span<const QComplex, 4>(matrix), input, output);
        require(output == expected, "matrix2 batch differs from scalar reference");

        parallel.matrix2_batch(std::span<const QComplex, 4>(matrix), input, input);
        require(input == expected, "matrix2 batch failed in-place execution");
    }

    {
        std::array<QComplex, 16> matrix{};
        for (std::size_t row = 0; row < 4U; ++row) {
            for (std::size_t column = 0; column < 4U; ++column) {
                const double x = static_cast<double>(1U + row * 4U + column);
                matrix[row * 4U + column] = {
                    0.03 * x,
                    0.01 * static_cast<double>(static_cast<int>(row) - static_cast<int>(column)),
                };
            }
        }
        std::vector<QComplex> input = complex_data(8'196U, -0.31);
        std::vector<QComplex> expected(input.size());
        for (std::size_t offset = 0; offset < input.size(); offset += 4U) {
            const std::array<QComplex, 4> values{{
                input[offset], input[offset + 1U], input[offset + 2U], input[offset + 3U]}};
            for (std::size_t row = 0; row < 4U; ++row) {
                QComplex sum{};
                for (std::size_t column = 0; column < 4U; ++column) {
                    sum += multiply(matrix[row * 4U + column], values[column]);
                }
                expected[offset + row] = sum;
            }
        }
        std::vector<QComplex> output(input.size());
        parallel.matrix4_batch(std::span<const QComplex, 16>(matrix), input, output);
        require(output == expected, "matrix4 batch differs from scalar reference");

        parallel.matrix4_batch(std::span<const QComplex, 16>(matrix), input, input);
        require(input == expected, "matrix4 batch failed in-place execution");
    }

    {
        KronVector structured({2U, 3U, 2U}, 8U);
        const std::vector<std::vector<QComplex>> first{
            {{0.8, 0.1}, {0.3, -0.2}},
            {{0.4, 0.0}, {-0.2, 0.3}, {0.7, -0.1}},
            {{0.6, 0.2}, {0.1, 0.5}},
        };
        const std::vector<std::vector<QComplex>> second{
            {{0.2, -0.4}, {0.5, 0.1}},
            {{-0.1, 0.2}, {0.3, 0.4}, {0.6, 0.0}},
            {{0.9, -0.1}, {-0.2, 0.2}},
        };
        structured.add_term({0.7, -0.15}, first);
        structured.add_term({-0.25, 0.3}, second);
        require(structured.logical_size() == 12U && structured.term_count() == 2U,
                "structured Kronecker dimensions changed");

        const std::vector<QComplex> dense = structured.materialize(12U);
        QComplex dense_norm{};
        for (const QComplex value : dense) {
            dense_norm += value.conjugate() * value;
        }
        require_close(structured.norm2(), dense_norm.re, 5e-13,
                      "structured Kronecker norm differs from dense materialization");

        for (std::size_t first_index = 0; first_index < 2U; ++first_index) {
            for (std::size_t second_index = 0; second_index < 3U; ++second_index) {
                for (std::size_t third_index = 0; third_index < 2U; ++third_index) {
                    const std::array<std::size_t, 3> index{
                        first_index, second_index, third_index};
                    const std::size_t flat =
                        (first_index * 3U + second_index) * 2U + third_index;
                    require_close(structured.element(index), dense[flat], 0.0,
                                  "structured Kronecker element order changed");
                }
            }
        }

        KronVector transformed = structured;
        const std::array<QComplex, 4> matrix{{
            {0.5, 0.25}, {-0.125, 0.75},
            {0.625, -0.375}, {0.25, 0.125},
        }};
        transformed.apply_local_operator(0U, matrix);
        const std::vector<QComplex> transformed_dense = transformed.materialize(12U);
        for (std::size_t out0 = 0; out0 < 2U; ++out0) {
            for (std::size_t i1 = 0; i1 < 3U; ++i1) {
                for (std::size_t i2 = 0; i2 < 2U; ++i2) {
                    QComplex expected{};
                    for (std::size_t in0 = 0; in0 < 2U; ++in0) {
                        const std::size_t source = (in0 * 3U + i1) * 2U + i2;
                        expected += matrix[out0 * 2U + in0] * dense[source];
                    }
                    const std::size_t target = (out0 * 3U + i1) * 2U + i2;
                    require_close(transformed_dense[target], expected, 2e-13,
                                  "structured local operator differs from dense execution");
                }
            }
        }

        QComplex dense_inner{};
        for (std::size_t index = 0; index < dense.size(); ++index) {
            dense_inner += dense[index].conjugate() * transformed_dense[index];
        }
        require_close(structured.inner_product(transformed), dense_inner, 5e-13,
                      "structured Kronecker inner product differs from dense execution");
    }

    {
        constexpr std::size_t sites = 100U;
        KronVector huge(std::vector<std::size_t>(sites, 2U), 4U);
        std::vector<std::vector<QComplex>> factors;
        factors.reserve(sites);
        for (std::size_t site = 0; site < sites; ++site) {
            const double angle = 0.01 * static_cast<double>(site + 1U);
            factors.push_back({{std::cos(angle), 0.0}, {std::sin(angle), 0.0}});
        }
        huge.add_term({1.0, 0.0}, factors);
        require(!huge.logical_size_fits(),
                "100-factor Kronecker vector unexpectedly fit in a dense size_t extent");
        require_close(static_cast<double>(huge.log2_logical_size()), 100.0, 1e-14,
                      "100-factor Kronecker logical size changed");
        require_close(huge.norm2(), 1.0, 2e-13,
                      "100-factor structured norm lost product normalization");
        const std::vector<std::size_t> zero_index(sites, 0U);
        QComplex expected{1.0, 0.0};
        for (std::size_t site = 0; site < sites; ++site) {
            expected *= factors[site][0];
        }
        require_close(huge.element(zero_index), expected, 2e-13,
                      "100-factor structured element differs from product reference");

        bool rejected = false;
        try {
            static_cast<void>(huge.materialize());
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "100-factor Kronecker vector allowed dense materialization");
    }

    {
        KronVector left({2U}, 2U);
        KronVector right({2U}, 2U);
        const std::vector<std::vector<QComplex>> a{{{1.0, 0.0}, {0.0, 0.0}}};
        const std::vector<std::vector<QComplex>> b{{{0.0, 0.0}, {1.0, 0.0}}};
        left.add_term({1.0, 0.0}, a);
        left.add_term({1.0, 0.0}, b);
        right.add_term({1.0, 0.0}, a);
        right.add_term({1.0, 0.0}, b);
        bool rejected = false;
        try {
            static_cast<void>(left.tensor_product(right));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "Kronecker tensor product exceeded its exact rank cap");
    }

    {
        std::vector<double> empty_real;
        std::vector<QComplex> empty_complex;
        const std::array<QComplex, 4> matrix2{};
        const std::array<QComplex, 16> matrix4{};
        parallel.fused_affine(empty_real, empty_real, 1.0, 1.0, 0.0, empty_real);
        require(parallel.fused_affine_norm2(
                    empty_real, empty_real, 1.0, 1.0, 0.0, empty_real) == 0.0,
                "empty fused real reduction is not zero");
        require(parallel.dot(empty_real, empty_real) == 0.0,
                "empty dot product is not zero");
        parallel.fused_complex_affine(
            empty_complex, empty_complex, {}, {}, {}, empty_complex);
        require(parallel.fused_complex_affine_norm2(
                    empty_complex, empty_complex, {}, {}, {}, empty_complex) == 0.0,
                "empty fused complex reduction is not zero");
        require(parallel.inner_product(empty_complex, empty_complex) == QComplex{},
                "empty inner product is not zero");
        parallel.matrix2_batch(std::span<const QComplex, 4>(matrix2), empty_complex, empty_complex);
        parallel.matrix4_batch(std::span<const QComplex, 16>(matrix4), empty_complex, empty_complex);
    }

    {
        bool rejected = false;
        try {
            std::vector<double> first(4U);
            std::vector<double> second(5U);
            std::vector<double> output(4U);
            parallel.fused_affine(first, second, 1.0, 1.0, 0.0, output);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "fused affine accepted mismatched sizes");

        rejected = false;
        try {
            std::vector<double> storage(17U);
            std::vector<double> second(16U);
            const std::span<const double> first(storage.data(), 16U);
            const std::span<double> output(storage.data() + 1U, 16U);
            parallel.fused_affine(first, second, 1.0, 1.0, 0.0, output);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "fused affine accepted partial output overlap");

        rejected = false;
        try {
            const std::array<QComplex, 4> matrix{};
            std::vector<QComplex> input(3U);
            std::vector<QComplex> output(3U);
            parallel.matrix2_batch(std::span<const QComplex, 4>(matrix), input, output);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "matrix2 batch accepted an incomplete vector");

        rejected = false;
        try {
            NumericExecutor invalid(NumericConfig{1U, 0U});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "numeric executor accepted zero grain size");

        rejected = false;
        try {
            NumericExecutor invalid(NumericConfig{129U, 256U});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "numeric executor accepted an unbounded worker count");

        rejected = false;
        try {
            KronVector invalid({2U, 0U});
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "KronVector accepted a zero factor dimension");

        rejected = false;
        try {
            KronVector vector({2U});
            const std::vector<std::vector<QComplex>> wrong{{{1.0, 0.0}}};
            vector.add_term({1.0, 0.0}, wrong);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "KronVector accepted a mismatched factor dimension");
    }

    {
        std::vector<double> first = real_data(16'384U, 0.09);
        std::vector<double> second = real_data(16'384U, -0.17);
        std::vector<double> output(first.size());
        double checksum = 0.0;
        for (std::size_t iteration = 0; iteration < 128U; ++iteration) {
            checksum += parallel.fused_affine_norm2(
                first,
                second,
                0.5 + 0.0001 * static_cast<double>(iteration),
                -0.25,
                0.125,
                output);
        }
        require(std::isfinite(checksum) && checksum > 0.0,
                "persistent numeric scheduler stress produced an invalid checksum");
    }

    return 0;
}
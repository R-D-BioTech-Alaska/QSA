#include "qubit/qqtt.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

using qubit::BasisIndex;
using qubit::ExactQTTFunction;
using qubit::QComplex;
using qubit::QStateError;
using qubit::QTTConfig;
using qubit::QTTCore;

[[nodiscard]] bool close(QComplex left, QComplex right, double tolerance = 1e-10) {
    const double scale = 1.0 + std::max(left.magnitude(), right.magnitude());
    return (left - right).magnitude() <= tolerance * scale;
}

[[nodiscard]] bool close(double left, double right, double tolerance = 1e-10) {
    const double scale = 1.0 + std::max(std::abs(left), std::abs(right));
    return std::abs(left - right) <= tolerance * scale;
}

template <class Function>
void require_reject(Function&& function, const std::string& label) {
    bool rejected = false;
    try {
        function();
    } catch (const QStateError&) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error(label + " did not reject");
    }
}

void test_product_and_affine() {
    const std::array<std::array<QComplex, 2>, 5> factors{{
        {QComplex{1.0}, QComplex{2.0}},
        {QComplex{3.0}, QComplex{-1.0}},
        {QComplex{0.5}, QComplex{4.0}},
        {QComplex{2.0}, QComplex{0.25}},
        {QComplex{-2.0}, QComplex{1.0}},
    }};
    const ExactQTTFunction product = ExactQTTFunction::product(factors);
    if (product.stats().maximum_rank != 1U) {
        throw std::runtime_error("rank-one product field changed rank");
    }
    for (BasisIndex index = 0U; index < 32U; ++index) {
        QComplex expected{1.0};
        for (std::size_t position = 0U; position < factors.size(); ++position) {
            const std::size_t shift = factors.size() - 1U - position;
            expected *= factors[position][static_cast<std::size_t>((index >> shift) & 1U)];
        }
        if (!close(product.value(index), expected)) {
            throw std::runtime_error("rank-one product field differential failed");
        }
    }

    const QComplex offset{1.25, -0.5};
    const QComplex slope{0.75, 0.125};
    const ExactQTTFunction affine = ExactQTTFunction::affine_index(12U, offset, slope);
    if (affine.stats().maximum_rank != 2U || affine.stats().descriptor_scalars != 88U) {
        throw std::runtime_error("affine QTT rank/scalar certificate changed");
    }
    for (BasisIndex index : std::array<BasisIndex, 6>{{0U, 1U, 7U, 255U, 2048U, 4095U}}) {
        const QComplex expected = offset + slope * static_cast<double>(index);
        if (!close(affine.value(index), expected, 2e-10)) {
            throw std::runtime_error("affine QTT selected value failed");
        }
    }
}

void test_complex_exponential() {
    constexpr double omega = 0.037;
    const QComplex amplitude{0.4, -0.2};
    const ExactQTTFunction state =
        ExactQTTFunction::complex_exponential(10U, omega, amplitude);
    if (state.stats().maximum_rank != 1U) {
        throw std::runtime_error("complex exponential is no longer rank one");
    }
    for (BasisIndex index : std::array<BasisIndex, 6>{{0U, 1U, 17U, 255U, 511U, 1023U}}) {
        const QComplex expected =
            amplitude * QComplex::from_polar(1.0, omega * static_cast<double>(index));
        if (!close(state.value(index), expected, 2e-10)) {
            throw std::runtime_error("complex exponential selected value failed");
        }
    }

    constexpr std::size_t wide_bits = 4096U;
    const ExactQTTFunction wide = ExactQTTFunction::complex_exponential(wide_bits, 1.0e-6);
    if (wide.logical_bits() != wide_bits || wide.stats().maximum_rank != 1U ||
        wide.stats().descriptor_scalars != 8192U) {
        throw std::runtime_error("4096-bit complex exponential descriptor changed");
    }
    for (const QTTCore& core : wide.cores()) {
        const QComplex phase = core.one.front();
        if (!std::isfinite(phase.re) || !std::isfinite(phase.im) ||
            !close(phase.norm2(), 1.0, 2e-12)) {
            throw std::runtime_error("4096-bit complex exponential phase lost unit magnitude");
        }
    }
    std::vector<std::uint8_t> selected(wide_bits);
    for (std::size_t position = 0U; position < wide_bits; ++position) {
        selected[position] = static_cast<std::uint8_t>(((position * 19U + 7U) % 23U) < 11U);
    }
    const QComplex selected_value = wide.value_bits(selected);
    if (!std::isfinite(selected_value.re) || !std::isfinite(selected_value.im) ||
        !close(selected_value.norm2(), 1.0, 2e-9)) {
        throw std::runtime_error("4096-bit complex exponential selected value is unstable");
    }
}

void test_algebra_and_reductions() {
    const ExactQTTFunction left = ExactQTTFunction::affine_index(8U, QComplex{0.5}, QComplex{0.25});
    const ExactQTTFunction right =
        ExactQTTFunction::complex_exponential(8U, 0.11, QComplex{0.75});
    const ExactQTTFunction added = left.add(right);
    const ExactQTTFunction product = left.hadamard(right);
    const std::vector<QComplex> dense_left = left.materialize();
    const std::vector<QComplex> dense_right = right.materialize();
    const std::vector<QComplex> dense_added = added.materialize();
    const std::vector<QComplex> dense_product = product.materialize();

    QComplex expected_sum{};
    QComplex expected_inner{};
    double expected_norm = 0.0;
    for (std::size_t index = 0U; index < dense_left.size(); ++index) {
        if (!close(dense_added[index], dense_left[index] + dense_right[index])) {
            throw std::runtime_error("QTT exact addition differential failed");
        }
        if (!close(dense_product[index], dense_left[index] * dense_right[index])) {
            throw std::runtime_error("QTT exact Hadamard differential failed");
        }
        expected_sum += dense_left[index];
        expected_inner += dense_left[index].conjugate() * dense_right[index];
        expected_norm += dense_left[index].norm2();
    }
    if (!close(left.sum_all(), expected_sum, 2e-10)) {
        throw std::runtime_error("QTT sum contraction failed");
    }
    if (!close(left.inner_product(right), expected_inner, 2e-10)) {
        throw std::runtime_error("QTT inner product contraction failed");
    }
    if (!close(left.norm_squared(), expected_norm, 2e-10)) {
        throw std::runtime_error("QTT norm contraction failed");
    }

    const std::array<std::size_t, 3> positions{{0U, 3U, 7U}};
    const std::array<std::uint8_t, 3> bits{{1U, 0U, 1U}};
    QComplex expected_conditioned{};
    for (std::size_t index = 0U; index < dense_left.size(); ++index) {
        bool selected = true;
        for (std::size_t query = 0U; query < positions.size(); ++query) {
            const std::size_t shift = 7U - positions[query];
            if (((index >> shift) & 1U) != bits[query]) {
                selected = false;
                break;
            }
        }
        if (selected) {
            expected_conditioned += dense_left[index];
        }
    }
    if (!close(left.conditioned_sum(positions, bits), expected_conditioned, 2e-10)) {
        throw std::runtime_error("QTT conditioned reduction failed");
    }
}

void test_random_weighted_bit_differential() {
    std::mt19937_64 random(0x51454c4dULL);
    std::uniform_real_distribution<double> distribution(-0.75, 0.75);
    for (std::size_t trial = 0U; trial < 48U; ++trial) {
        const std::size_t logical_bits = 3U + static_cast<std::size_t>(random() % 6U);
        std::vector<QComplex> weights(logical_bits);
        for (QComplex& weight : weights) {
            weight = QComplex{distribution(random), distribution(random)};
        }
        const QComplex offset{distribution(random), distribution(random)};
        const ExactQTTFunction state = ExactQTTFunction::weighted_bit_sum(weights, offset);
        const std::size_t count = std::size_t{1} << logical_bits;
        QComplex direct_sum{};
        for (std::size_t index = 0U; index < count; ++index) {
            QComplex expected = offset;
            for (std::size_t position = 0U; position < logical_bits; ++position) {
                const std::size_t shift = logical_bits - 1U - position;
                if (((index >> shift) & 1U) != 0U) {
                    expected += weights[position];
                }
            }
            const QComplex actual = state.value(static_cast<BasisIndex>(index));
            if (!close(actual, expected, 5e-11)) {
                throw std::runtime_error("random weighted-bit QTT differential failed");
            }
            direct_sum += expected;
        }
        if (!close(state.sum_all(), direct_sum, 5e-10)) {
            throw std::runtime_error("random weighted-bit QTT sum failed");
        }
    }
}

void test_4096_bit_capability() {
    constexpr std::size_t logical_bits = 4096U;
    const ExactQTTFunction state = ExactQTTFunction::hamming_weight(logical_bits);
    if (state.logical_bits() != logical_bits || state.stats().maximum_rank != 2U ||
        state.stats().descriptor_scalars != 32760U) {
        throw std::runtime_error("4096-bit QTT descriptor certificate failed");
    }

    std::vector<std::uint8_t> all_ones(logical_bits, 1U);
    if (!close(state.value_bits(all_ones), QComplex{4096.0}, 1e-12)) {
        throw std::runtime_error("4096-bit all-ones selected value failed");
    }
    std::vector<std::uint8_t> alternating(logical_bits);
    for (std::size_t position = 0U; position < logical_bits; ++position) {
        alternating[position] = static_cast<std::uint8_t>(position & 1U);
    }
    if (!close(state.value_bits(alternating), QComplex{2048.0}, 1e-12)) {
        throw std::runtime_error("4096-bit alternating selected value failed");
    }

    constexpr std::size_t fixed_count = logical_bits - 100U;
    std::vector<std::size_t> fixed_positions(fixed_count);
    std::vector<std::uint8_t> fixed_bits(fixed_count);
    std::size_t fixed_ones = 0U;
    for (std::size_t position = 0U; position < fixed_count; ++position) {
        fixed_positions[position] = position;
        fixed_bits[position] = static_cast<std::uint8_t>(position & 1U);
        fixed_ones += fixed_bits[position];
    }
    const double expected =
        std::ldexp(static_cast<double>(fixed_ones), 100) +
        std::ldexp(100.0, 99);
    const QComplex reduced = state.conditioned_sum(fixed_positions, fixed_bits);
    if (!close(reduced, QComplex{expected}, 1e-12)) {
        throw std::runtime_error("4096-bit conditioned 2^100 reduction failed");
    }

    require_reject(
        [&] { static_cast<void>(state.value(BasisIndex{0})); },
        "4096-bit BasisIndex query");
    require_reject([&] { static_cast<void>(state.materialize()); }, "4096-bit materialization");
}

void test_fail_closed_resources_and_shape() {
    require_reject(
        [] {
            static_cast<void>(ExactQTTFunction::from_certified_cores({
                QTTCore{2U, 1U, {QComplex{1.0}, QComplex{}}, {QComplex{}, QComplex{1.0}}},
            }));
        },
        "invalid first rank");

    require_reject(
        [] {
            static_cast<void>(ExactQTTFunction::from_certified_cores({
                QTTCore{1U, 2U, {QComplex{1.0}, QComplex{}}, {QComplex{}, QComplex{1.0}}},
                QTTCore{3U, 1U, {QComplex{1.0}, QComplex{}, QComplex{}},
                        {QComplex{}, QComplex{}, QComplex{1.0}}},
            }));
        },
        "adjacent rank mismatch");

    require_reject(
        [] {
            static_cast<void>(ExactQTTFunction::from_certified_cores({
                QTTCore{1U, 1U, {QComplex{std::numeric_limits<double>::infinity()}},
                        {QComplex{1.0}}},
            }));
        },
        "non-finite core");

    QTTConfig rank_two;
    rank_two.max_rank = 2U;
    const ExactQTTFunction left =
        ExactQTTFunction::affine_index(8U, QComplex{}, QComplex{1.0}, rank_two);
    const ExactQTTFunction right =
        ExactQTTFunction::affine_index(8U, QComplex{1.0}, QComplex{2.0}, rank_two);
    require_reject([&] { static_cast<void>(left.add(right)); }, "addition rank growth");
    require_reject([&] { static_cast<void>(left.hadamard(right)); }, "Hadamard rank growth");

    QTTConfig environment_three;
    environment_three.max_environment_scalars = 3U;
    const ExactQTTFunction env_left =
        ExactQTTFunction::affine_index(6U, QComplex{}, QComplex{1.0}, environment_three);
    const ExactQTTFunction env_right =
        ExactQTTFunction::affine_index(6U, QComplex{1.0}, QComplex{0.5}, environment_three);
    require_reject(
        [&] { static_cast<void>(env_left.inner_product(env_right)); },
        "inner-product environment cap");

    QTTConfig total_seven;
    total_seven.max_total_scalars = 7U;
    const std::array<std::array<QComplex, 2>, 4> factors{{
        {QComplex{1.0}, QComplex{1.0}},
        {QComplex{1.0}, QComplex{1.0}},
        {QComplex{1.0}, QComplex{1.0}},
        {QComplex{1.0}, QComplex{1.0}},
    }};
    require_reject(
        [&] { static_cast<void>(ExactQTTFunction::product(factors, total_seven)); },
        "total scalar cap");

    QTTConfig materialize_two;
    materialize_two.max_materialize_bits = 2U;
    const ExactQTTFunction three = ExactQTTFunction::hamming_weight(3U, materialize_two);
    require_reject([&] { static_cast<void>(three.materialize()); }, "materialization cap");

    const std::array<std::size_t, 2> duplicate_positions{{1U, 1U}};
    const std::array<std::uint8_t, 2> duplicate_bits{{0U, 1U}};
    require_reject(
        [&] { static_cast<void>(three.conditioned_sum(duplicate_positions, duplicate_bits)); },
        "duplicate conditioned positions");
}

}  // namespace

int main() {
    try {
        test_product_and_affine();
        test_complex_exponential();
        test_algebra_and_reductions();
        test_random_weighted_bit_differential();
        test_4096_bit_capability();
        test_fail_closed_resources_and_shape();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
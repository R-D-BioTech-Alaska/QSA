#include "qubit/qqtt.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::BasisIndex;
using qubit::ExactQTTFunction;
using qubit::QComplex;

#if defined(_MSC_VER)
#define QSA_NOINLINE __declspec(noinline)
#else
#define QSA_NOINLINE __attribute__((noinline))
#endif

volatile double observed = 0.0;

template <class Function>
[[nodiscard]] double median_ms(Function&& function, int repeats = 7, int iterations = 1) {
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto start = Clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            function();
        }
        const auto finish = Clock::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(finish - start).count() /
            static_cast<double>(iterations));
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

QSA_NOINLINE QComplex qtt_value(const ExactQTTFunction& state, BasisIndex index) {
    return state.value(index);
}

QSA_NOINLINE QComplex qtt_value_bits(
    const ExactQTTFunction& state,
    const std::vector<std::uint8_t>& bits) {
    return state.value_bits(bits);
}

QSA_NOINLINE QComplex qtt_sum(const ExactQTTFunction& state) {
    return state.sum_all();
}

QSA_NOINLINE QComplex qtt_conditioned(
    const ExactQTTFunction& state,
    const std::vector<std::size_t>& positions,
    const std::vector<std::uint8_t>& bits) {
    return state.conditioned_sum(positions, bits);
}

QSA_NOINLINE double qtt_norm(const ExactQTTFunction& state) {
    return state.norm_squared();
}

QSA_NOINLINE double direct_hamming_value(BasisIndex index) {
    return static_cast<double>(std::popcount(index));
}

QSA_NOINLINE double dense_hamming_sum(std::size_t logical_bits) {
    const std::size_t count = std::size_t{1} << logical_bits;
    std::uint64_t sum = 0U;
    for (std::size_t index = 0U; index < count; ++index) {
        sum += static_cast<std::uint64_t>(std::popcount(index));
    }
    return static_cast<double>(sum);
}

QSA_NOINLINE double closed_hamming_sum(std::size_t logical_bits) {
    return std::ldexp(static_cast<double>(logical_bits), static_cast<int>(logical_bits - 1U));
}

QSA_NOINLINE double closed_hamming_norm(std::size_t logical_bits) {
    const double coefficient =
        static_cast<double>(logical_bits) * static_cast<double>(logical_bits + 1U);
    return std::ldexp(coefficient, static_cast<int>(logical_bits - 2U));
}

QSA_NOINLINE std::size_t direct_bit_count(const std::vector<std::uint8_t>& bits) {
    std::size_t result = 0U;
    for (std::uint8_t bit : bits) {
        result += static_cast<std::size_t>(bit);
    }
    return result;
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);

    constexpr std::size_t matched_bits = 20U;
    ExactQTTFunction matched = ExactQTTFunction::hamming_weight(matched_bits);
    double setup_observed = 0.0;
    const double matched_setup_ms = median_ms([&] {
        ExactQTTFunction candidate = ExactQTTFunction::hamming_weight(matched_bits);
        setup_observed += static_cast<double>(candidate.stats().descriptor_scalars);
    });
    observed = setup_observed;

    constexpr BasisIndex selected = 0xABCDEULL;
    QComplex selected_qtt{};
    const double qtt_query_ms = median_ms([&] {
        selected_qtt = qtt_value(matched, selected);
        observed = selected_qtt.re;
    }, 9, 1000);
    double selected_direct = 0.0;
    const double direct_query_ms = median_ms([&] {
        selected_direct = direct_hamming_value(selected);
        observed = selected_direct;
    }, 9, 100000);

    QComplex qtt_partition{};
    const double qtt_sum_ms = median_ms([&] {
        qtt_partition = qtt_sum(matched);
        observed = qtt_partition.re;
    }, 9, 1000);
    double dense_partition = 0.0;
    const double dense_sum_ms = median_ms([&] {
        dense_partition = dense_hamming_sum(matched_bits);
        observed = dense_partition;
    }, 7, 1);
    double closed_partition = 0.0;
    const double closed_sum_ms = median_ms([&] {
        closed_partition = closed_hamming_sum(matched_bits);
        observed = closed_partition;
    }, 9, 100000);

    const double matched_query_error = std::abs(selected_qtt.re - selected_direct);
    const double matched_sum_error = std::abs(qtt_partition.re - dense_partition);
    if (matched_query_error > 1e-12 || matched_sum_error > 1e-9 ||
        std::abs(qtt_partition.im) > 1e-12 || dense_partition != closed_partition) {
        std::cerr << "matched QTT Hamming control failed\n";
        return 2;
    }

    std::cout << "qtt_matched_bits=" << matched_bits << '\n';
    std::cout << "qtt_matched_logical_entries=" << (std::size_t{1} << matched_bits) << '\n';
    std::cout << "qtt_matched_rank=" << matched.stats().maximum_rank << '\n';
    std::cout << "qtt_matched_descriptor_scalars=" << matched.stats().descriptor_scalars << '\n';
    std::cout << "qtt_matched_setup_ms=" << matched_setup_ms << '\n';
    std::cout << "qtt_matched_query_ms=" << qtt_query_ms << '\n';
    std::cout << "qtt_matched_direct_query_ms=" << direct_query_ms << '\n';
    std::cout << "qtt_matched_sum_ms=" << qtt_sum_ms << '\n';
    std::cout << "qtt_matched_dense_sum_ms=" << dense_sum_ms << '\n';
    std::cout << "qtt_matched_closed_sum_ms=" << closed_sum_ms << '\n';
    std::cout << "qtt_matched_dense_sum_speedup=" << dense_sum_ms / qtt_sum_ms << '\n';
    std::cout << "qtt_matched_query_error=" << matched_query_error << '\n';
    std::cout << "qtt_matched_sum_error=" << matched_sum_error << '\n';

    constexpr std::size_t norm_bits = 512U;
    const ExactQTTFunction norm_state = ExactQTTFunction::hamming_weight(norm_bits);
    double qtt_norm_value = 0.0;
    const double qtt_norm_ms = median_ms([&] {
        qtt_norm_value = qtt_norm(norm_state);
        observed = qtt_norm_value;
    }, 7, 20);
    const double norm_control = closed_hamming_norm(norm_bits);
    const double norm_relative_error =
        std::abs(qtt_norm_value - norm_control) / std::max(1.0, std::abs(norm_control));
    if (norm_relative_error > 2e-12) {
        std::cerr << "QTT 512-bit norm control failed\n";
        return 3;
    }
    std::cout << "qtt_norm_bits=" << norm_bits << '\n';
    std::cout << "qtt_norm_rank=" << norm_state.stats().maximum_rank << '\n';
    std::cout << "qtt_norm_ms=" << qtt_norm_ms << '\n';
    std::cout << "qtt_norm_relative_error=" << norm_relative_error << '\n';

    constexpr std::size_t capability_bits = 4096U;
    ExactQTTFunction capability = ExactQTTFunction::hamming_weight(capability_bits);
    double capability_setup_observed = 0.0;
    const double capability_setup_ms = median_ms([&] {
        ExactQTTFunction candidate = ExactQTTFunction::hamming_weight(capability_bits);
        capability_setup_observed += static_cast<double>(candidate.stats().descriptor_scalars);
    });
    observed = capability_setup_observed;

    std::vector<std::uint8_t> capability_query(capability_bits);
    for (std::size_t position = 0U; position < capability_bits; ++position) {
        capability_query[position] =
            static_cast<std::uint8_t>(((position * 37U + 11U) % 7U) < 3U ? 1U : 0U);
    }
    const double capability_direct = static_cast<double>(direct_bit_count(capability_query));
    QComplex capability_value{};
    const double capability_query_ms = median_ms([&] {
        capability_value = qtt_value_bits(capability, capability_query);
        observed = capability_value.re;
    }, 7, 100);
    double capability_direct_value = 0.0;
    const double capability_direct_query_ms = median_ms([&] {
        capability_direct_value = static_cast<double>(direct_bit_count(capability_query));
        observed = capability_direct_value;
    }, 7, 1000);

    constexpr std::size_t free_bits = 100U;
    constexpr std::size_t fixed_count = capability_bits - free_bits;
    std::vector<std::size_t> fixed_positions(fixed_count);
    std::vector<std::uint8_t> fixed_values(fixed_count);
    std::size_t fixed_ones = 0U;
    for (std::size_t position = 0U; position < fixed_count; ++position) {
        fixed_positions[position] = position;
        fixed_values[position] = static_cast<std::uint8_t>(position & 1U);
        fixed_ones += static_cast<std::size_t>(fixed_values[position]);
    }
    const double conditioned_control =
        std::ldexp(static_cast<double>(fixed_ones), static_cast<int>(free_bits)) +
        std::ldexp(static_cast<double>(free_bits), static_cast<int>(free_bits - 1U));
    QComplex conditioned_value{};
    const double conditioned_ms = median_ms([&] {
        conditioned_value = qtt_conditioned(capability, fixed_positions, fixed_values);
        observed = conditioned_value.re;
    }, 7, 50);

    const double capability_query_error = std::abs(capability_value.re - capability_direct);
    const double conditioned_relative_error =
        std::abs(conditioned_value.re - conditioned_control) /
        std::max(1.0, std::abs(conditioned_control));
    if (capability.stats().maximum_rank != 2U ||
        capability.stats().descriptor_scalars != 32760U ||
        capability_query_error > 1e-12 || conditioned_relative_error > 2e-12 ||
        std::abs(capability_value.im) > 1e-12 || std::abs(conditioned_value.im) > 1e-12) {
        std::cerr << "4096-bit QTT capability control failed\n";
        return 4;
    }

    const double descriptor_bytes =
        static_cast<double>(capability.stats().descriptor_scalars * sizeof(QComplex));
    const double dense_log2_bytes =
        static_cast<double>(capability_bits) + std::log2(static_cast<double>(sizeof(QComplex)));
    const double descriptor_log2_bytes = std::log2(descriptor_bytes);
    const double dense_decimal_digits = std::floor(
        std::log10(static_cast<double>(sizeof(QComplex))) +
        static_cast<double>(capability_bits) * std::log10(2.0)) + 1.0;

    std::cout << "qtt_capability_bits=" << capability_bits << '\n';
    std::cout << "qtt_capability_log2_entries=" << capability_bits << '\n';
    std::cout << "qtt_capability_rank=" << capability.stats().maximum_rank << '\n';
    std::cout << "qtt_capability_descriptor_scalars="
              << capability.stats().descriptor_scalars << '\n';
    std::cout << "qtt_capability_descriptor_bytes=" << descriptor_bytes << '\n';
    std::cout << "qtt_capability_estimated_bytes=" << capability.estimated_bytes() << '\n';
    std::cout << "qtt_capability_dense_log2_bytes=" << dense_log2_bytes << '\n';
    std::cout << "qtt_capability_descriptor_log2_bytes=" << descriptor_log2_bytes << '\n';
    std::cout << "qtt_capability_dense_descriptor_log2_ratio="
              << dense_log2_bytes - descriptor_log2_bytes << '\n';
    std::cout << "qtt_capability_dense_byte_decimal_digits=" << dense_decimal_digits << '\n';
    std::cout << "qtt_capability_setup_ms=" << capability_setup_ms << '\n';
    std::cout << "qtt_capability_query_ms=" << capability_query_ms << '\n';
    std::cout << "qtt_capability_direct_query_ms=" << capability_direct_query_ms << '\n';
    std::cout << "qtt_capability_selected_value=" << capability_value.re << '\n';
    std::cout << "qtt_capability_selected_control=" << capability_direct << '\n';
    std::cout << "qtt_capability_selected_error=" << capability_query_error << '\n';
    std::cout << "qtt_capability_conditioned_free_bits=" << free_bits << '\n';
    std::cout << "qtt_capability_conditioned_ms=" << conditioned_ms << '\n';
    std::cout << "qtt_capability_conditioned_relative_error=" << conditioned_relative_error << '\n';

    return observed == -1.0 ? 1 : 0;
}

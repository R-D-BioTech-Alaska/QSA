#include "qubit/qtropical_qtt.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::BasisIndex;
using qubit::ExactTropicalQTT;
using qubit::TropicalQTTCore;
using qubit::TropicalQTTEntry;
using qubit::TropicalQTTMinimum;

#if defined(_MSC_VER)
#define QSA_NOINLINE __declspec(noinline)
#else
#define QSA_NOINLINE __attribute__((noinline))
#endif

volatile std::int64_t observed_energy = 0;
volatile std::size_t observed_size = 0U;

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

QSA_NOINLINE TropicalQTTMinimum qtt_minimum(const ExactTropicalQTT& state) {
    return state.global_minimum();
}

QSA_NOINLINE std::int64_t qtt_energy(
    const ExactTropicalQTT& state,
    const std::vector<std::uint8_t>& bits) {
    return state.energy_bits(bits);
}

QSA_NOINLINE std::int64_t direct_viterbi_minimum(const ExactTropicalQTT& state) {
    struct Cost {
        bool supported{false};
        std::int64_t value{0};
    };

    std::vector<Cost> costs(1U, Cost{true, 0});
    for (const TropicalQTTCore& core : state.cores()) {
        std::vector<Cost> next(core.right_rank);
        for (std::size_t following = 0U; following < core.right_rank; ++following) {
            for (std::uint8_t bit = 0U; bit < 2U; ++bit) {
                const std::vector<TropicalQTTEntry>& matrix = bit == 0U ? core.zero : core.one;
                for (std::size_t previous = 0U; previous < core.left_rank; ++previous) {
                    const TropicalQTTEntry& edge =
                        matrix[previous * core.right_rank + following];
                    if (!costs[previous].supported || !edge.supported) {
                        continue;
                    }
                    const std::int64_t candidate = costs[previous].value + edge.cost;
                    if (!next[following].supported || candidate < next[following].value) {
                        next[following] = Cost{true, candidate};
                    }
                }
            }
        }
        costs = std::move(next);
    }
    return costs.front().value;
}

QSA_NOINLINE std::int64_t exhaustive_minimum(
    const ExactTropicalQTT& state,
    std::size_t logical_bits) {
    const std::size_t count = std::size_t{1} << logical_bits;
    bool have = false;
    std::int64_t best = 0;
    for (std::size_t index = 0U; index < count; ++index) {
        const std::int64_t energy = state.energy(static_cast<BasisIndex>(index));
        if (!have || energy < best) {
            have = true;
            best = energy;
        }
    }
    return best;
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);

    constexpr std::size_t matched_bits = 16U;
    constexpr std::size_t matched_rank = 4U;
    ExactTropicalQTT matched =
        ExactTropicalQTT::structured_markov_energy(matched_bits, matched_rank, 31);
    double setup_sink = 0.0;
    const double matched_setup_ms = median_ms([&] {
        ExactTropicalQTT candidate =
            ExactTropicalQTT::structured_markov_energy(matched_bits, matched_rank, 31);
        setup_sink += static_cast<double>(candidate.stats().descriptor_entries);
    });
    observed_size = static_cast<std::size_t>(setup_sink);

    TropicalQTTMinimum matched_qtt{};
    const double matched_qtt_ms = median_ms([&] {
        matched_qtt = qtt_minimum(matched);
        observed_energy = matched_qtt.energy;
    }, 9, 100);

    std::int64_t matched_viterbi = 0;
    const double matched_viterbi_ms = median_ms([&] {
        matched_viterbi = direct_viterbi_minimum(matched);
        observed_energy = matched_viterbi;
    }, 9, 1000);

    std::int64_t matched_exhaustive = 0;
    const double matched_exhaustive_ms = median_ms([&] {
        matched_exhaustive = exhaustive_minimum(matched, matched_bits);
        observed_energy = matched_exhaustive;
    }, 5, 1);

    if (matched_qtt.energy != matched_viterbi || matched_qtt.energy != matched_exhaustive ||
        matched.stats().descriptor_entries != 464U ||
        matched_qtt.candidate_evaluations != matched.stats().descriptor_entries ||
        qtt_energy(matched, matched_qtt.bits) != matched_qtt.energy) {
        std::cerr << "matched Tropical QTT controls failed\n";
        return 2;
    }

    std::cout << "tropical_matched_bits=" << matched_bits << '\n';
    std::cout << "tropical_matched_sector_count=" << (std::size_t{1} << matched_bits) << '\n';
    std::cout << "tropical_matched_rank=" << matched.stats().maximum_rank << '\n';
    std::cout << "tropical_matched_descriptor_entries="
              << matched.stats().descriptor_entries << '\n';
    std::cout << "tropical_matched_setup_ms=" << matched_setup_ms << '\n';
    std::cout << "tropical_matched_qtt_min_ms=" << matched_qtt_ms << '\n';
    std::cout << "tropical_matched_viterbi_min_ms=" << matched_viterbi_ms << '\n';
    std::cout << "tropical_matched_exhaustive_min_ms=" << matched_exhaustive_ms << '\n';
    std::cout << "tropical_matched_vs_enumeration_speedup="
              << matched_exhaustive_ms / matched_qtt_ms << '\n';
    std::cout << "tropical_matched_qtt_over_viterbi_ratio="
              << matched_qtt_ms / matched_viterbi_ms << '\n';
    std::cout << "tropical_matched_energy=" << matched_qtt.energy << '\n';
    std::cout << "tropical_matched_candidate_evaluations="
              << matched_qtt.candidate_evaluations << '\n';

    constexpr std::size_t capability_bits = 4096U;
    constexpr std::size_t capability_rank = 8U;
    constexpr std::size_t capability_entries = 524064U;
    ExactTropicalQTT capability =
        ExactTropicalQTT::structured_markov_energy(capability_bits, capability_rank, 101);
    double capability_setup_sink = 0.0;
    const double capability_setup_ms = median_ms([&] {
        ExactTropicalQTT candidate =
            ExactTropicalQTT::structured_markov_energy(capability_bits, capability_rank, 101);
        capability_setup_sink += static_cast<double>(candidate.stats().descriptor_entries);
    }, 5, 1);
    observed_size = static_cast<std::size_t>(capability_setup_sink);

    TropicalQTTMinimum capability_qtt{};
    const double capability_qtt_ms = median_ms([&] {
        capability_qtt = qtt_minimum(capability);
        observed_energy = capability_qtt.energy;
    }, 7, 1);

    std::int64_t capability_viterbi = 0;
    const double capability_viterbi_ms = median_ms([&] {
        capability_viterbi = direct_viterbi_minimum(capability);
        observed_energy = capability_viterbi;
    }, 7, 1);

    std::int64_t capability_verified = 0;
    const double capability_verify_ms = median_ms([&] {
        capability_verified = qtt_energy(capability, capability_qtt.bits);
        observed_energy = capability_verified;
    }, 7, 10);

    if (capability.stats().maximum_rank != capability_rank ||
        capability.stats().descriptor_entries != capability_entries ||
        capability.stats().supported_entries != capability_entries ||
        capability_qtt.candidate_evaluations != capability_entries ||
        capability_qtt.maximum_active_bond_states > capability_rank ||
        capability_qtt.bits.size() != capability_bits ||
        capability_qtt.energy != capability_viterbi ||
        capability_qtt.energy != capability_verified) {
        std::cerr << "4096-bit Tropical QTT capability controls failed\n";
        return 3;
    }

    const double descriptor_log2 = std::log2(static_cast<double>(capability_entries));
    std::cout << "tropical_capability_bits=" << capability_bits << '\n';
    std::cout << "tropical_capability_log2_sectors=" << capability_bits << '\n';
    std::cout << "tropical_capability_rank=" << capability.stats().maximum_rank << '\n';
    std::cout << "tropical_capability_descriptor_entries="
              << capability.stats().descriptor_entries << '\n';
    std::cout << "tropical_capability_supported_entries="
              << capability.stats().supported_entries << '\n';
    std::cout << "tropical_capability_estimated_bytes=" << capability.estimated_bytes() << '\n';
    std::cout << "tropical_capability_descriptor_log2=" << descriptor_log2 << '\n';
    std::cout << "tropical_capability_sector_descriptor_log2_ratio="
              << static_cast<double>(capability_bits) - descriptor_log2 << '\n';
    std::cout << "tropical_capability_setup_ms=" << capability_setup_ms << '\n';
    std::cout << "tropical_capability_qtt_min_ms=" << capability_qtt_ms << '\n';
    std::cout << "tropical_capability_viterbi_min_ms=" << capability_viterbi_ms << '\n';
    std::cout << "tropical_capability_qtt_over_viterbi_ratio="
              << capability_qtt_ms / capability_viterbi_ms << '\n';
    std::cout << "tropical_capability_verify_ms=" << capability_verify_ms << '\n';
    std::cout << "tropical_capability_energy=" << capability_qtt.energy << '\n';
    std::cout << "tropical_capability_candidate_evaluations="
              << capability_qtt.candidate_evaluations << '\n';
    std::cout << "tropical_capability_maximum_active_bond_states="
              << capability_qtt.maximum_active_bond_states << '\n';

    return observed_energy == std::numeric_limits<std::int64_t>::min() && observed_size == 0U ? 1 : 0;
}

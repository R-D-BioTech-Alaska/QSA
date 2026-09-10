#include "qubit/qpersistent_tropical_qtt.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::ExactTropicalQTT;
using qubit::PersistentTropicalQTT;
using qubit::PersistentTropicalQTTUpdate;
using qubit::TropicalQTTCore;
using qubit::TropicalQTTEntry;

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

[[nodiscard]] TropicalQTTCore perturb(
    const TropicalQTTCore& source,
    std::uint8_t bit,
    std::int64_t delta,
    std::size_t seed) {
    TropicalQTTCore core = source;
    std::vector<TropicalQTTEntry>& matrix = bit == 0U ? core.zero : core.one;
    for (std::size_t row = 0U; row < core.left_rank; ++row) {
        for (std::size_t column = 0U; column < core.right_rank; ++column) {
            TropicalQTTEntry& entry = matrix[row * core.right_rank + column];
            if (entry.supported) {
                entry.cost += delta + static_cast<std::int64_t>((row + 3U * column + seed) % 5U);
            }
        }
    }
    return core;
}

QSA_NOINLINE std::int64_t direct_viterbi_override(
    const ExactTropicalQTT& state,
    std::size_t replacement_index,
    const TropicalQTTCore& replacement) {
    struct Cost {
        bool supported{false};
        std::int64_t value{0};
    };
    std::vector<Cost> costs(1U, Cost{true, 0});
    for (std::size_t index = 0U; index < state.logical_bits(); ++index) {
        const TropicalQTTCore& core = index == replacement_index ? replacement : state.cores()[index];
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

QSA_NOINLINE std::int64_t persistent_minimum(const PersistentTropicalQTT& tree) {
    return tree.minimum_energy();
}

QSA_NOINLINE std::uint8_t persistent_selected_bit(
    const PersistentTropicalQTT& tree,
    std::size_t index) {
    return tree.selected_bit(index);
}

QSA_NOINLINE std::vector<std::uint8_t> persistent_bits(const PersistentTropicalQTT& tree) {
    return tree.minimum_bits();
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);

    constexpr std::size_t logical_bits = 4096U;
    constexpr std::size_t rank = 8U;
    constexpr std::size_t update_index = 2048U;
    ExactTropicalQTT source =
        ExactTropicalQTT::structured_markov_energy(logical_bits, rank, 101);

    std::optional<PersistentTropicalQTT> built;
    const double build_ms = median_ms([&] {
        built.emplace(PersistentTropicalQTT::build(source));
        observed_size = built->stats().transfer_entries;
    }, 5, 1);
    PersistentTropicalQTT tree = std::move(*built);

    const std::int64_t old_energy = tree.minimum_energy();
    const std::uint8_t old_selected = tree.selected_bit(update_index);
    TropicalQTTCore replacement = perturb(
        source.cores()[update_index], old_selected, 1000, 17U);

    std::int64_t direct_energy = 0;
    const double direct_resolve_ms = median_ms([&] {
        direct_energy = direct_viterbi_override(source, update_index, replacement);
        observed_energy = direct_energy;
    }, 7, 1);

    std::optional<PersistentTropicalQTT> updated;
    PersistentTropicalQTTUpdate receipt{};
    const double update_ms = median_ms([&] {
        auto result = tree.update_core(update_index, replacement);
        updated.emplace(std::move(result.first));
        receipt = result.second;
        observed_energy = updated->minimum_energy();
    }, 9, 10);

    std::int64_t root_energy = 0;
    const double root_query_ms = median_ms([&] {
        root_energy = persistent_minimum(*updated);
        observed_energy = root_energy;
    }, 9, 100000);

    std::uint8_t selected = 0U;
    const double selected_bit_ms = median_ms([&] {
        selected = persistent_selected_bit(*updated, update_index);
        observed_size = selected;
    }, 9, 10000);

    std::vector<std::uint8_t> all_bits;
    const double all_bits_ms = median_ms([&] {
        all_bits = persistent_bits(*updated);
        observed_size = all_bits.size();
    }, 9, 100);

    ExactTropicalQTT reconstructed = updated->to_state();
    const std::int64_t reconstructed_energy = reconstructed.global_minimum().energy;
    const std::int64_t bits_energy = reconstructed.energy_bits(all_bits);

    if (tree.stats().tree_nodes != 8191U || tree.stats().tree_depth != 12U ||
        tree.stats().transfer_entries != 522817U ||
        receipt.nodes_created != 13U || receipt.transfer_entries_created != 713U ||
        tree.minimum_energy() != old_energy ||
        receipt.old_minimum_energy != old_energy ||
        receipt.new_minimum_energy != direct_energy ||
        updated->minimum_energy() != direct_energy || root_energy != direct_energy ||
        reconstructed_energy != direct_energy || bits_energy != direct_energy ||
        selected != all_bits[update_index]) {
        std::cerr << "persistent Tropical QTT controls failed\n";
        return 2;
    }

    std::cout << "persistent_tropical_bits=" << logical_bits << '\n';
    std::cout << "persistent_tropical_log2_sectors=" << logical_bits << '\n';
    std::cout << "persistent_tropical_rank=" << rank << '\n';
    std::cout << "persistent_tropical_tree_nodes=" << tree.stats().tree_nodes << '\n';
    std::cout << "persistent_tropical_tree_depth=" << tree.stats().tree_depth << '\n';
    std::cout << "persistent_tropical_transfer_entries="
              << tree.stats().transfer_entries << '\n';
    std::cout << "persistent_tropical_build_ms=" << build_ms << '\n';
    std::cout << "persistent_tropical_update_index=" << update_index << '\n';
    std::cout << "persistent_tropical_update_nodes_created=" << receipt.nodes_created << '\n';
    std::cout << "persistent_tropical_update_transfer_entries_created="
              << receipt.transfer_entries_created << '\n';
    std::cout << "persistent_tropical_update_ms=" << update_ms << '\n';
    std::cout << "persistent_tropical_direct_viterbi_resolve_ms=" << direct_resolve_ms << '\n';
    std::cout << "persistent_tropical_update_speedup=" << direct_resolve_ms / update_ms << '\n';
    std::cout << "persistent_tropical_root_query_ms=" << root_query_ms << '\n';
    std::cout << "persistent_tropical_selected_bit_ms=" << selected_bit_ms << '\n';
    std::cout << "persistent_tropical_all_bits_ms=" << all_bits_ms << '\n';
    std::cout << "persistent_tropical_selected_over_all_bits_speedup="
              << all_bits_ms / selected_bit_ms << '\n';
    std::cout << "persistent_tropical_old_energy=" << old_energy << '\n';
    std::cout << "persistent_tropical_new_energy=" << direct_energy << '\n';
    std::cout << "persistent_tropical_update_energy_error="
              << (updated->minimum_energy() - direct_energy) << '\n';
    std::cout << "persistent_tropical_reconstructed_energy_error="
              << (reconstructed_energy - direct_energy) << '\n';

    return observed_energy == std::numeric_limits<std::int64_t>::min() && observed_size == 0U ? 1 : 0;
}

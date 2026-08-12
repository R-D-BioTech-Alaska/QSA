#include "qubit/qtropical_qtt.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using qubit::BasisIndex;
using qubit::ExactTropicalQTT;
using qubit::QStateError;
using qubit::TropicalQTTConfig;
using qubit::TropicalQTTCore;
using qubit::TropicalQTTEntry;
using qubit::TropicalQTTMinimum;

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

[[nodiscard]] TropicalQTTMinimum exhaustive_minimum(
    const ExactTropicalQTT& state,
    std::size_t logical_bits) {
    if (logical_bits >= std::numeric_limits<std::size_t>::digits) {
        throw std::runtime_error("test exhaustive domain is too wide");
    }
    const std::size_t count = std::size_t{1} << logical_bits;
    bool have = false;
    std::int64_t best_energy = 0;
    std::size_t best_index = 0U;
    for (std::size_t index = 0U; index < count; ++index) {
        try {
            const std::int64_t energy = state.energy(static_cast<BasisIndex>(index));
            if (!have || energy < best_energy || (energy == best_energy && index < best_index)) {
                have = true;
                best_energy = energy;
                best_index = index;
            }
        } catch (const QStateError&) {
        }
    }
    if (!have) {
        throw std::runtime_error("test exhaustive reference found no supported sector");
    }
    std::vector<std::uint8_t> bits(logical_bits);
    for (std::size_t position = 0U; position < logical_bits; ++position) {
        const std::size_t shift = logical_bits - 1U - position;
        bits[position] = static_cast<std::uint8_t>((best_index >> shift) & 1U);
    }
    return TropicalQTTMinimum{std::move(bits), best_energy, count, 0U};
}

void test_rank_one_weighted_minimum() {
    const std::array<std::int64_t, 8> weights{{17, -5, 11, -3, 29, -7, 13, -2}};
    constexpr std::int64_t offset = 31;
    const ExactTropicalQTT state = ExactTropicalQTT::weighted_bit_sum(weights, offset);
    const TropicalQTTMinimum minimum = state.global_minimum();

    std::int64_t expected = offset;
    std::vector<std::uint8_t> expected_bits(weights.size());
    for (std::size_t index = 0U; index < weights.size(); ++index) {
        if (weights[index] < 0) {
            expected += weights[index];
            expected_bits[index] = 1U;
        }
    }
    if (state.stats().maximum_rank != 1U || state.stats().descriptor_entries != 16U ||
        minimum.energy != expected || minimum.bits != expected_bits ||
        minimum.candidate_evaluations != state.stats().descriptor_entries ||
        minimum.maximum_active_bond_states != 1U ||
        state.energy_bits(minimum.bits) != minimum.energy) {
        throw std::runtime_error("rank-one Tropical QTT minimum certificate failed");
    }
}

void test_structured_against_exhaustive() {
    constexpr std::size_t logical_bits = 14U;
    const ExactTropicalQTT state =
        ExactTropicalQTT::structured_markov_energy(logical_bits, 3U, 31);
    const TropicalQTTMinimum dynamic = state.global_minimum();
    const TropicalQTTMinimum exhaustive = exhaustive_minimum(state, logical_bits);
    if (dynamic.energy != exhaustive.energy ||
        state.energy_bits(dynamic.bits) != dynamic.energy ||
        dynamic.candidate_evaluations != state.stats().descriptor_entries ||
        dynamic.candidate_evaluations >= (std::size_t{1} << logical_bits) ||
        dynamic.maximum_active_bond_states > 3U) {
        throw std::runtime_error("structured Tropical QTT exhaustive differential failed");
    }

    for (BasisIndex index : std::array<BasisIndex, 5>{{0U, 1U, 17U, 8191U, 16383U}}) {
        std::vector<std::uint8_t> bits(logical_bits);
        for (std::size_t position = 0U; position < logical_bits; ++position) {
            const std::size_t shift = logical_bits - 1U - position;
            bits[position] = static_cast<std::uint8_t>((index >> shift) & 1U);
        }
        if (state.energy(index) != state.energy_bits(bits)) {
            throw std::runtime_error("Tropical QTT fixed-sector index semantics changed");
        }
    }
}

void test_sparse_support_and_reachability() {
    const TropicalQTTEntry missing = TropicalQTTEntry::missing();
    const ExactTropicalQTT state = ExactTropicalQTT::from_certified_cores({
        TropicalQTTCore{
            1U,
            2U,
            {TropicalQTTEntry::edge(0), missing},
            {missing, TropicalQTTEntry::edge(1)},
        },
        TropicalQTTCore{
            2U,
            1U,
            {TropicalQTTEntry::edge(2), missing},
            {missing, TropicalQTTEntry::edge(3)},
        },
    });
    const std::array<std::uint8_t, 2> zero_path{{0U, 0U}};
    const std::array<std::uint8_t, 2> one_path{{1U, 1U}};
    const std::array<std::uint8_t, 2> unsupported{{0U, 1U}};
    if (state.energy_bits(zero_path) != 2 || state.energy_bits(one_path) != 4) {
        throw std::runtime_error("Tropical QTT sparse supported-path energy failed");
    }
    require_reject(
        [&] { static_cast<void>(state.energy_bits(unsupported)); },
        "unsupported Tropical QTT sector");
    const TropicalQTTMinimum minimum = state.global_minimum();
    if (minimum.energy != 2 || minimum.bits != std::vector<std::uint8_t>({0U, 0U})) {
        throw std::runtime_error("Tropical QTT sparse global minimum failed");
    }

    require_reject(
        [&] {
            static_cast<void>(ExactTropicalQTT::from_certified_cores({
                TropicalQTTCore{
                    1U,
                    2U,
                    {TropicalQTTEntry::edge(0), missing},
                    {missing, TropicalQTTEntry::edge(1)},
                },
                TropicalQTTCore{2U, 1U, {missing, missing}, {missing, missing}},
            }));
        },
        "dead Tropical QTT support");
}

void test_4096_bit_rank_eight_capability() {
    constexpr std::size_t logical_bits = 4096U;
    constexpr std::size_t rank = 8U;
    constexpr std::size_t expected_entries = 524064U;
    const ExactTropicalQTT state =
        ExactTropicalQTT::structured_markov_energy(logical_bits, rank, 101);
    if (state.logical_bits() != logical_bits || state.stats().maximum_rank != rank ||
        state.stats().descriptor_entries != expected_entries ||
        state.stats().supported_entries != expected_entries) {
        throw std::runtime_error("4096-bit Tropical QTT descriptor certificate failed");
    }
    const TropicalQTTMinimum minimum = state.global_minimum();
    if (minimum.bits.size() != logical_bits ||
        minimum.candidate_evaluations != expected_entries ||
        minimum.maximum_active_bond_states > rank ||
        state.energy_bits(minimum.bits) != minimum.energy) {
        throw std::runtime_error("4096-bit Tropical QTT global minimum certificate failed");
    }
    require_reject(
        [&] { static_cast<void>(state.energy(BasisIndex{0})); },
        "4096-bit Tropical QTT BasisIndex query");
}

void test_resource_and_integer_fail_closed() {
    TropicalQTTConfig rank_cap;
    rank_cap.max_rank = 7U;
    require_reject(
        [&] { static_cast<void>(ExactTropicalQTT::structured_markov_energy(4096U, 8U, 101, rank_cap)); },
        "Tropical QTT rank cap");

    TropicalQTTConfig core_cap;
    core_cap.max_core_entries = 127U;
    require_reject(
        [&] { static_cast<void>(ExactTropicalQTT::structured_markov_energy(4096U, 8U, 101, core_cap)); },
        "Tropical QTT core entry cap");

    TropicalQTTConfig total_cap;
    total_cap.max_total_entries = 524063U;
    require_reject(
        [&] { static_cast<void>(ExactTropicalQTT::structured_markov_energy(4096U, 8U, 101, total_cap)); },
        "Tropical QTT total entry cap");

    TropicalQTTConfig backpointer_cap;
    backpointer_cap.max_backpointer_entries = 32760U;
    const ExactTropicalQTT no_minimum =
        ExactTropicalQTT::structured_markov_energy(4096U, 8U, 101, backpointer_cap);
    require_reject(
        [&] { static_cast<void>(no_minimum.global_minimum()); },
        "Tropical QTT backpointer cap");

    const std::array<std::int64_t, 1> one{{1}};
    require_reject(
        [&] {
            static_cast<void>(ExactTropicalQTT::weighted_bit_sum(
                one, std::numeric_limits<std::int64_t>::max()));
        },
        "Tropical QTT generator integer overflow");

    const ExactTropicalQTT accumulation_overflow =
        ExactTropicalQTT::from_certified_cores({
            TropicalQTTCore{
                1U,
                1U,
                {TropicalQTTEntry::edge(std::numeric_limits<std::int64_t>::max())},
                {TropicalQTTEntry::missing()},
            },
            TropicalQTTCore{
                1U,
                1U,
                {TropicalQTTEntry::edge(1)},
                {TropicalQTTEntry::missing()},
            },
        });
    const std::array<std::uint8_t, 2> zero_bits{{0U, 0U}};
    require_reject(
        [&] { static_cast<void>(accumulation_overflow.energy_bits(zero_bits)); },
        "Tropical QTT path integer overflow");
    require_reject(
        [&] { static_cast<void>(accumulation_overflow.global_minimum()); },
        "Tropical QTT minimum integer overflow");

    require_reject(
        [] {
            static_cast<void>(ExactTropicalQTT::from_certified_cores({
                TropicalQTTCore{
                    1U,
                    2U,
                    {TropicalQTTEntry::edge(0), TropicalQTTEntry::edge(0)},
                    {TropicalQTTEntry::edge(0), TropicalQTTEntry::edge(0)},
                },
                TropicalQTTCore{
                    3U,
                    1U,
                    {TropicalQTTEntry::edge(0), TropicalQTTEntry::edge(0), TropicalQTTEntry::edge(0)},
                    {TropicalQTTEntry::edge(0), TropicalQTTEntry::edge(0), TropicalQTTEntry::edge(0)},
                },
            }));
        },
        "Tropical QTT adjacent rank mismatch");

    require_reject(
        [] {
            static_cast<void>(ExactTropicalQTT::from_certified_cores({
                TropicalQTTCore{
                    1U,
                    2U,
                    {TropicalQTTEntry::edge(0)},
                    {TropicalQTTEntry::edge(0)},
                },
            }));
        },
        "Tropical QTT physical slice size mismatch");
}

}  // namespace

int main() {
    try {
        test_rank_one_weighted_minimum();
        test_structured_against_exhaustive();
        test_sparse_support_and_reachability();
        test_4096_bit_rank_eight_capability();
        test_resource_and_integer_fail_closed();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}

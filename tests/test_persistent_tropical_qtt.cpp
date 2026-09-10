#include "qubit/qpersistent_tropical_qtt.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using qubit::ExactTropicalQTT;
using qubit::PersistentTropicalQTT;
using qubit::PersistentTropicalQTTConfig;
using qubit::QStateError;
using qubit::TropicalQTTCore;
using qubit::TropicalQTTEntry;

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

void test_build_and_selected_bits() {
    const ExactTropicalQTT source = ExactTropicalQTT::structured_markov_energy(64U, 4U, 31);
    const std::int64_t source_minimum = source.global_minimum().energy;
    const PersistentTropicalQTT tree = PersistentTropicalQTT::build(source);
    if (tree.logical_bits() != 64U || tree.maximum_rank() != 4U ||
        tree.stats().tree_nodes != 127U || tree.stats().tree_depth != 6U ||
        tree.minimum_energy() != source_minimum) {
        throw std::runtime_error("persistent Tropical QTT build certificate failed");
    }

    const std::vector<std::uint8_t> bits = tree.minimum_bits();
    if (source.energy_bits(bits) != source_minimum) {
        throw std::runtime_error("persistent Tropical QTT minimum-bit materialization failed");
    }
    for (std::size_t index : std::array<std::size_t, 6>{{0U, 1U, 7U, 31U, 47U, 63U}}) {
        if (tree.selected_bit(index) != bits[index]) {
            throw std::runtime_error("persistent Tropical QTT selected-bit descent failed");
        }
    }
    const ExactTropicalQTT restored = tree.to_state();
    if (restored.global_minimum().energy != source_minimum ||
        restored.energy_bits(bits) != source_minimum) {
        throw std::runtime_error("persistent Tropical QTT source reconstruction failed");
    }
}

void test_transactional_update() {
    const ExactTropicalQTT source = ExactTropicalQTT::structured_markov_energy(64U, 4U, 47);
    const PersistentTropicalQTT tree = PersistentTropicalQTT::build(source);
    const std::int64_t old_energy = tree.minimum_energy();
    TropicalQTTCore replacement = perturb(source.cores()[37U], 0U, 23, 11U);
    auto [updated, receipt] = tree.update_core(37U, std::move(replacement));
    const ExactTropicalQTT updated_state = updated.to_state();
    const std::vector<std::uint8_t> bits = updated.minimum_bits();

    if (tree.minimum_energy() != old_energy || receipt.core_index != 37U ||
        receipt.old_minimum_energy != old_energy ||
        receipt.new_minimum_energy != updated.minimum_energy() ||
        receipt.nodes_created != 7U || receipt.transfer_entries_created == 0U ||
        updated_state.global_minimum().energy != updated.minimum_energy() ||
        updated_state.energy_bits(bits) != updated.minimum_energy()) {
        throw std::runtime_error("persistent Tropical QTT transactional update failed");
    }
}

void test_4096_bit_logarithmic_update() {
    constexpr std::size_t logical_bits = 4096U;
    constexpr std::size_t rank = 8U;
    constexpr std::size_t expected_tree_entries = 522817U;
    const ExactTropicalQTT source =
        ExactTropicalQTT::structured_markov_energy(logical_bits, rank, 101);
    const std::int64_t old_reference = source.global_minimum().energy;
    const PersistentTropicalQTT tree = PersistentTropicalQTT::build(source);
    if (tree.stats().tree_nodes != 8191U || tree.stats().tree_depth != 12U ||
        tree.stats().transfer_entries != expected_tree_entries ||
        tree.minimum_energy() != old_reference) {
        throw std::runtime_error("4096-bit persistent Tropical QTT build certificate failed");
    }

    constexpr std::size_t update_index = 2048U;
    TropicalQTTCore replacement = perturb(source.cores()[update_index], 1U, 41, 17U);
    auto [updated, receipt] = tree.update_core(update_index, std::move(replacement));
    if (receipt.nodes_created != 13U || receipt.transfer_entries_created > 13U * rank * rank ||
        tree.minimum_energy() != old_reference) {
        throw std::runtime_error("4096-bit persistent Tropical QTT update-path certificate failed");
    }

    const ExactTropicalQTT updated_state = updated.to_state();
    const std::int64_t rebuilt_reference = updated_state.global_minimum().energy;
    if (updated.minimum_energy() != rebuilt_reference ||
        receipt.new_minimum_energy != rebuilt_reference) {
        throw std::runtime_error("4096-bit persistent Tropical QTT updated minimum failed");
    }
    const std::vector<std::uint8_t> bits = updated.minimum_bits();
    if (updated_state.energy_bits(bits) != rebuilt_reference ||
        updated.selected_bit(update_index) != bits[update_index]) {
        throw std::runtime_error("4096-bit persistent Tropical QTT updated optimum bits failed");
    }
}

void test_resource_and_failure_boundaries() {
    const ExactTropicalQTT source = ExactTropicalQTT::structured_markov_energy(4096U, 8U, 101);

    PersistentTropicalQTTConfig node_cap;
    node_cap.max_tree_nodes = 8190U;
    require_reject(
        [&] { static_cast<void>(PersistentTropicalQTT::build(source, node_cap)); },
        "persistent Tropical QTT tree-node cap");

    PersistentTropicalQTTConfig transfer_cap;
    transfer_cap.max_total_transfer_entries = 522816U;
    require_reject(
        [&] { static_cast<void>(PersistentTropicalQTT::build(source, transfer_cap)); },
        "persistent Tropical QTT transfer-entry cap");

    PersistentTropicalQTTConfig update_cap;
    update_cap.max_update_transfer_entries = 700U;
    const PersistentTropicalQTT bounded = PersistentTropicalQTT::build(source, update_cap);
    TropicalQTTCore replacement = perturb(source.cores()[2048U], 0U, 17, 5U);
    const std::int64_t old_energy = bounded.minimum_energy();
    require_reject(
        [&] { static_cast<void>(bounded.update_core(2048U, replacement)); },
        "persistent Tropical QTT update transfer cap");
    if (bounded.minimum_energy() != old_energy) {
        throw std::runtime_error("failed persistent Tropical QTT update mutated source tree");
    }

    TropicalQTTCore wrong_shape = source.cores()[2048U];
    wrong_shape.left_rank = 1U;
    require_reject(
        [&] { static_cast<void>(bounded.update_core(2048U, wrong_shape)); },
        "persistent Tropical QTT shape-changing update");

    TropicalQTTCore no_support = source.cores()[2048U];
    for (TropicalQTTEntry& entry : no_support.zero) {
        entry = TropicalQTTEntry::missing();
    }
    for (TropicalQTTEntry& entry : no_support.one) {
        entry = TropicalQTTEntry::missing();
    }
    require_reject(
        [&] { static_cast<void>(PersistentTropicalQTT::build(source).update_core(2048U, no_support)); },
        "persistent Tropical QTT support-removing update");

    const ExactTropicalQTT overflow_source = ExactTropicalQTT::from_certified_cores({
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
    require_reject(
        [&] { static_cast<void>(PersistentTropicalQTT::build(overflow_source)); },
        "persistent Tropical QTT min-plus overflow");
}

}  // namespace

int main() {
    try {
        test_build_and_selected_bits();
        test_transactional_update();
        test_4096_bit_logarithmic_update();
        test_resource_and_failure_boundaries();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}

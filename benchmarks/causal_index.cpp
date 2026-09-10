#include "qubit/qcausal_index.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace qubit;

namespace {

std::size_t naive_causal_count(
    std::size_t qubits,
    const std::vector<Operation>& operations,
    QubitId query) {
    std::vector<std::uint8_t> active(qubits, 0U);
    active[static_cast<std::size_t>(query)] = 1U;
    std::size_t retained = 0U;
    for (std::size_t cursor = operations.size(); cursor-- > 0U;) {
        const Operation& operation = operations[cursor];
        const std::size_t first = static_cast<std::size_t>(operation.first);
        if (operation.code == OperationCode::Cnot ||
            operation.code == OperationCode::Cz ||
            operation.code == OperationCode::Swap) {
            const std::size_t second = static_cast<std::size_t>(operation.second);
            if (active[first] != 0U || active[second] != 0U) {
                active[first] = 1U;
                active[second] = 1U;
                ++retained;
            }
        } else if (active[first] != 0U) {
            ++retained;
        }
    }
    return retained;
}

}  // namespace

int main() {
    constexpr std::size_t qubits = 10000U;
    constexpr std::size_t layers = 100U;
    constexpr std::size_t prefix_h = 8U;

    std::vector<Operation> operations;
    operations.reserve(prefix_h + layers * (qubits - 1U));
    for (std::size_t index = 0U; index < prefix_h; ++index) {
        operations.push_back({OperationCode::H, 0U});
    }
    for (std::size_t layer = 0U; layer < layers; ++layer) {
        for (std::size_t qubit = 0U; qubit + 1U < qubits; ++qubit) {
            operations.push_back({
                OperationCode::Cnot,
                static_cast<QubitId>(qubit),
                static_cast<QubitId>(qubit + 1U),
            });
        }
    }

    const auto build_start = std::chrono::steady_clock::now();
    ExactCausalOperationIndex index(qubits, operations);
    const auto build_end = std::chrono::steady_clock::now();

    const std::vector<QubitId> query{0U};
    ExactCausalSlice warm = index.slice(query);

    constexpr std::size_t repeats = 100U;
    std::uint64_t indexed_checksum = 0U;
    const auto indexed_start = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0U; repeat < repeats; ++repeat) {
        const ExactCausalSlice slice = index.slice(query);
        indexed_checksum += static_cast<std::uint64_t>(slice.operations.size());
        indexed_checksum += static_cast<std::uint64_t>(slice.global_qubits.size());
    }
    const auto indexed_end = std::chrono::steady_clock::now();

    std::uint64_t naive_checksum = 0U;
    const auto naive_start = std::chrono::steady_clock::now();
    for (std::size_t repeat = 0U; repeat < repeats; ++repeat) {
        naive_checksum += static_cast<std::uint64_t>(
            naive_causal_count(qubits, operations, query[0]));
    }
    const auto naive_end = std::chrono::steady_clock::now();

    const double build_ms =
        std::chrono::duration<double, std::milli>(build_end - build_start).count();
    const double indexed_us =
        std::chrono::duration<double, std::micro>(indexed_end - indexed_start).count() /
        static_cast<double>(repeats);
    const double naive_us =
        std::chrono::duration<double, std::micro>(naive_end - naive_start).count() /
        static_cast<double>(repeats);
    const double speedup = naive_us / indexed_us;

    std::cout << std::setprecision(17);
    std::cout << "index_qubits=" << qubits << '\n';
    std::cout << "index_operations=" << operations.size() << '\n';
    std::cout << "index_dependency_edges=" << index.stats().dependency_edges << '\n';
    std::cout << "index_estimated_bytes=" << index.stats().estimated_bytes << '\n';
    std::cout << "index_build_ms=" << build_ms << '\n';
    std::cout << "slice_qubits=" << warm.global_qubits.size() << '\n';
    std::cout << "slice_operations=" << warm.operations.size() << '\n';
    std::cout << "indexed_query_us=" << indexed_us << '\n';
    std::cout << "naive_query_us=" << naive_us << '\n';
    std::cout << "indexed_over_naive_speedup=" << speedup << '\n';
    std::cout << "indexed_checksum=" << indexed_checksum << '\n';
    std::cout << "naive_checksum=" << naive_checksum << '\n';
    return 0;
}

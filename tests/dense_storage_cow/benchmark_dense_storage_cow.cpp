#include "dense_fixture.hpp"

#include "qubit/qcausal.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] double milliseconds(
    Clock::time_point start,
    Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

[[nodiscard]] double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

}  // namespace

int main() {
    constexpr std::size_t group_count = 16U;
    constexpr std::size_t group_width = 16U;
    constexpr int repeats = 9;
    constexpr double angle = 0.217;

    qubit::QRegister prototype =
        qsa_dense_cow_test::make_dense_components(group_count, group_width);
    const std::size_t dense_payload_bytes = prototype.estimated_bytes();
    const std::vector<std::uint8_t> root_qsc = prototype.encode_qsc();
    qubit::CausalState root(std::move(prototype));

    std::vector<double> qsc_samples;
    std::vector<double> cow_samples;
    qsc_samples.reserve(repeats);
    cow_samples.reserve(repeats);
    volatile double sink = 0.0;

    for (int repeat = 0; repeat < repeats; ++repeat) {
        {
            const auto start = Clock::now();
            qubit::QRegister state = qubit::QRegister::decode_qsc(root_qsc);
            state.apply_rz(0U, angle);
            sink += state.probability_one(0U);
            const auto stop = Clock::now();
            qsc_samples.push_back(milliseconds(start, stop));
        }
        {
            const auto start = Clock::now();
            qubit::CausalState branch = root.fork();
            branch.mutate([](qubit::QRegister& state) {
                state.apply_rz(0U, angle);
            });
            sink += branch.read().probability_one(0U);
            const auto stop = Clock::now();
            cow_samples.push_back(milliseconds(start, stop));
        }
    }

    qubit::CausalState receipt_branch = root.fork();
    receipt_branch.mutate([](qubit::QRegister& state) {
        state.apply_rz(0U, angle);
    });
    const long touched_root_owners =
        root.read().component_storage_owner_count(0U);
    const long touched_branch_owners =
        receipt_branch.read().component_storage_owner_count(0U);
    const auto untouched = static_cast<qubit::QubitId>(group_width);
    const long untouched_root_owners =
        root.read().component_storage_owner_count(untouched);
    const long untouched_branch_owners =
        receipt_branch.read().component_storage_owner_count(untouched);

    qubit::QRegister control = qubit::QRegister::decode_qsc(root_qsc);
    control.apply_rz(0U, angle);
    const double maximum_error = qsa_dense_cow_test::max_state_error(
        receipt_branch.read(),
        control);

    const double qsc_ms = median(qsc_samples);
    const double cow_ms = median(cow_samples);
    if (sink < -1.0) {
        throw std::runtime_error("unreachable benchmark sink");
    }

    std::cout << std::setprecision(12)
              << "dense_storage_cow"
              << " groups=" << group_count
              << " group_width=" << group_width
              << " components=" << root.read().component_count()
              << " dense_payload_bytes=" << dense_payload_bytes
              << " qsc_bytes=" << root_qsc.size()
              << " qsc_clone_mutate_ms=" << qsc_ms
              << " dense_cow_mutate_ms=" << cow_ms
              << " speedup=" << qsc_ms / cow_ms
              << " touched_root_owners=" << touched_root_owners
              << " touched_branch_owners=" << touched_branch_owners
              << " untouched_root_owners=" << untouched_root_owners
              << " untouched_branch_owners=" << untouched_branch_owners
              << " max_error=" << maximum_error
              << '\n';
    return maximum_error <= 2.0e-12 ? 0 : 1;
}

#include "qubit/qcausal.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::CausalState;
using qubit::PauliObservablePlan;
using qubit::QRegister;

[[nodiscard]] double elapsed_ms(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

[[nodiscard]] double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

}  // namespace

int main() {
    constexpr std::size_t qubits = 10'000U;
    constexpr std::size_t branch_count = 128U;
    constexpr int repeats = 7;

    QRegister prototype(qubits);
    for (std::size_t qubit = 0; qubit < 32U; ++qubit) {
        prototype.apply_h(static_cast<qubit::QubitId>(qubit));
    }
    prototype.apply_x(static_cast<qubit::QubitId>(qubits - 1U));
    const std::vector<std::uint8_t> qsc = prototype.encode_qsc();
    CausalState root(prototype);

    std::vector<double> qsc_fork_samples;
    std::vector<double> causal_fork_samples;
    qsc_fork_samples.reserve(repeats);
    causal_fork_samples.reserve(repeats);

    for (int repeat = 0; repeat < repeats; ++repeat) {
        {
            const auto start = Clock::now();
            std::vector<QRegister> branches;
            branches.reserve(branch_count);
            for (std::size_t branch = 0; branch < branch_count; ++branch) {
                branches.push_back(QRegister::decode_qsc(qsc));
            }
            const auto stop = Clock::now();
            volatile std::size_t sink = branches.size();
            (void)sink;
            qsc_fork_samples.push_back(elapsed_ms(start, stop));
        }
        {
            const auto start = Clock::now();
            const auto branches = root.fork_many(branch_count);
            const auto stop = Clock::now();
            volatile long sink = branches.front().shared_owner_count();
            (void)sink;
            causal_fork_samples.push_back(elapsed_ms(start, stop));
        }
    }

    std::string observable(qubits, 'I');
    for (std::size_t qubit = 0; qubit < 32U; ++qubit) {
        observable[qubit] = 'X';
    }
    observable[qubits - 1U] = 'Z';
    const std::vector<std::string> words{observable};
    const PauliObservablePlan plan(qubits, words);

    const auto observable_start = Clock::now();
    const auto values = plan.execute(root);
    const auto observable_stop = Clock::now();

    const double qsc_ms = median(qsc_fork_samples);
    const double causal_ms = median(causal_fork_samples);
    std::cout << std::setprecision(12)
              << "causal_runtime qubits=" << qubits
              << " branches=" << branch_count
              << " qsc_bytes=" << qsc.size()
              << " qsc_clone_ms=" << qsc_ms
              << " causal_fork_ms=" << causal_ms
              << " fork_speedup=" << qsc_ms / causal_ms
              << " observable_ms="
              << elapsed_ms(observable_start, observable_stop)
              << " observable_value=" << values.front()
              << " components=" << root.read().component_count()
              << '\n';
    return 0;
}

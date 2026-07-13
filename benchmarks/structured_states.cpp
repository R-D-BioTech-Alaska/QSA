#include "qubit/qstate.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

std::string format_bytes(std::size_t bytes) {
    const char* units[] = {"B", "KiB", "MiB", "GiB"};
    double value = static_cast<double>(bytes);
    std::size_t unit = 0;
    while (value >= 1024.0 && unit < 3) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << value << ' ' << units[unit];
    return stream.str();
}

void report(const std::string& label, const qubit::QRegister& state, double milliseconds) {
    std::cout << std::left << std::setw(28) << label
              << " components=" << std::setw(6) << state.component_count()
              << " engine-memory=" << std::setw(12) << format_bytes(state.estimated_bytes())
              << " build=" << std::fixed << std::setprecision(3) << milliseconds << " ms\n";
}

}  // namespace

int main() {
    using clock = std::chrono::steady_clock;

    auto begin = clock::now();
    qubit::QRegister product(10'000);
    for (std::uint32_t q = 0; q < 10'000; q += 3) {
        product.apply_h(q);
    }
    auto end = clock::now();
    report("10,000 independent qubits", product,
           std::chrono::duration<double, std::milli>(end - begin).count());

    begin = clock::now();
    qubit::QRegister ghz(50);
    ghz.apply_h(0);
    for (std::uint32_t q = 1; q < 50; ++q) {
        ghz.apply_cnot(0, q);
    }
    end = clock::now();
    report("50-qubit exact GHZ", ghz,
           std::chrono::duration<double, std::milli>(end - begin).count());
    std::cout << "  GHZ support=" << ghz.component_nonzero_count(0)
              << ", patch-width=" << ghz.component_size(0) << " qubits\n";

    begin = clock::now();
    qubit::QRegister local_pairs(200);
    for (std::uint32_t q = 0; q < 200; q += 2) {
        local_pairs.apply_h(q);
        local_pairs.apply_cnot(q, q + 1);
    }
    end = clock::now();
    report("100 independent Bell pairs", local_pairs,
           std::chrono::duration<double, std::milli>(end - begin).count());

    std::cout << "\nA dense 50-qubit statevector would require roughly 16 PiB at complex128.\n"
                 "This benchmark is for structured states; arbitrary highly entangled states can still grow exponentially.\n";
    return 0;
}

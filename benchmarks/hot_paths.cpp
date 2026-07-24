#include "qubit/qstate.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace {

using Clock = std::chrono::steady_clock;

template <typename Function>
double elapsed_ms(Function&& function) {
    const auto start = Clock::now();
    function();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void bench_bell_pairs(std::size_t pairs) {
    qubit::QRegister state(pairs * 2U);
    const double milliseconds = elapsed_ms([&] {
        for (std::size_t pair = 0; pair < pairs; ++pair) {
            const auto first = static_cast<qubit::QubitId>(pair * 2U);
            state.apply_h(first);
            state.apply_cnot(first, first + 1U);
        }
    });
    std::cout << "bell_pairs," << pairs << ',' << std::setprecision(12) << milliseconds << ','
              << state.component_count() << ',' << state.estimated_bytes() << '\n';
}

void bench_sparse_rz(std::size_t operations) {
    constexpr std::size_t width = 50;
    qubit::QRegister state(width);
    state.apply_h(0);
    for (std::size_t target = 1; target < width; ++target) {
        state.apply_cnot(0, static_cast<qubit::QubitId>(target));
    }
    const double milliseconds = elapsed_ms([&] {
        for (std::size_t operation = 0; operation < operations; ++operation) {
            state.apply_rz(static_cast<qubit::QubitId>(operation % width), 0.00031);
        }
    });
    std::cout << "sparse_rz," << operations << ',' << std::setprecision(12) << milliseconds << ','
              << state.component_nonzero_count(0) << ',' << state.estimated_bytes() << '\n';
}

void bench_sparse_cnot(std::size_t operations) {
    constexpr std::size_t width = 50;
    qubit::QRegister state(width);
    state.apply_h(0);
    for (std::size_t target = 1; target < width; ++target) {
        state.apply_cnot(0, static_cast<qubit::QubitId>(target));
    }
    const double milliseconds = elapsed_ms([&] {
        for (std::size_t operation = 0; operation < operations; ++operation) {
            const auto target = static_cast<qubit::QubitId>(1U + (operation % (width - 1U)));
            state.apply_cnot(0, target);
        }
    });
    std::cout << "sparse_cnot," << operations << ',' << std::setprecision(12) << milliseconds << ','
              << state.component_nonzero_count(0) << ',' << state.estimated_bytes() << '\n';
}

void bench_sparse_ry(std::size_t operations) {
    constexpr std::size_t width = 50;
    qubit::QRegister state(width);
    state.apply_h(0);
    for (std::size_t target = 1; target < width; ++target) {
        state.apply_cnot(0, static_cast<qubit::QubitId>(target));
    }
    for (std::size_t qubit = 0; qubit < 10; ++qubit) {
        state.apply_ry(static_cast<qubit::QubitId>(qubit), 0.17);
    }
    const double milliseconds = elapsed_ms([&] {
        for (std::size_t operation = 0; operation < operations; ++operation) {
            state.apply_ry(static_cast<qubit::QubitId>(operation % 10U), 0.00031);
        }
    });
    std::cout << "sparse_ry," << operations << ',' << std::setprecision(12) << milliseconds << ','
              << state.component_nonzero_count(0) << ',' << state.estimated_bytes() << '\n';
}

void bench_dense_cnot(std::size_t operations) {
    constexpr std::size_t width = 16;
    qubit::QRegister state(width);
    for (std::size_t qubit = 0; qubit < width; ++qubit) {
        state.apply_h(static_cast<qubit::QubitId>(qubit));
    }
    for (std::size_t qubit = 1; qubit < width; ++qubit) {
        state.apply_cz(0, static_cast<qubit::QubitId>(qubit));
    }
    const double milliseconds = elapsed_ms([&] {
        for (std::size_t operation = 0; operation < operations; ++operation) {
            const auto target = static_cast<qubit::QubitId>(1U + (operation % (width - 1U)));
            state.apply_cnot(0, target);
        }
    });
    std::cout << "dense_cnot," << operations << ',' << std::setprecision(12) << milliseconds << ','
              << state.component_nonzero_count(0) << ',' << state.estimated_bytes() << '\n';
}

void bench_dense_ry(std::size_t operations) {
    constexpr std::size_t width = 16;
    qubit::QRegister state(width);
    for (std::size_t qubit = 0; qubit < width; ++qubit) {
        state.apply_h(static_cast<qubit::QubitId>(qubit));
    }
    for (std::size_t qubit = 1; qubit < width; ++qubit) {
        state.apply_cz(0, static_cast<qubit::QubitId>(qubit));
    }
    const double milliseconds = elapsed_ms([&] {
        for (std::size_t operation = 0; operation < operations; ++operation) {
            state.apply_ry(static_cast<qubit::QubitId>(operation % width), 0.00031);
        }
    });
    std::cout << "dense_ry," << operations << ',' << std::setprecision(12) << milliseconds << ','
              << state.component_nonzero_count(0) << ',' << state.estimated_bytes() << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t scale = argc > 1 ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 10)) : 1U;
    std::cout << "benchmark,count,milliseconds,support_or_components,estimated_bytes\n";
    bench_bell_pairs(1000U * scale);
    bench_sparse_rz(20000U * scale);
    bench_sparse_cnot(2000U * scale);
    bench_sparse_ry(2000U * scale);
    bench_dense_cnot(20U * scale);
    bench_dense_ry(20U * scale);
    return 0;
}

#include "qubit/qexact.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace qubit;

namespace {

constexpr std::size_t kIterations = 20000U;
constexpr std::size_t kRounds = 7U;
constexpr std::size_t kSamples = 8U;

QRational evaluate(std::size_t index) {
    const std::int64_t i = static_cast<std::int64_t>(index % 997U) + 2;
    const QRational a(i + 1, i + 2);
    const QRational b(2 * i + 3, i + 5);
    const QRational c(3 * i + 7, 2 * i + 9);
    const QRational d(i + 11, 4 * i + 13);
    return (a + b) * (c - d);
}

double run_round() {
    volatile double sink = 0.0;
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t index = 0U; index < kIterations; ++index) {
        sink += evaluate(index).to_double();
    }
    const auto stopped = std::chrono::steady_clock::now();
    if (!std::isfinite(sink)) throw std::runtime_error("QMath exact benchmark sink became nonfinite");
    return std::chrono::duration<double>(stopped - started).count();
}

}  // namespace

int main() {
    for (std::size_t warmup = 0U; warmup < 2U; ++warmup) (void)run_round();

    std::vector<double> rounds;
    rounds.reserve(kRounds);
    for (std::size_t round = 0U; round < kRounds; ++round) rounds.push_back(run_round());
    std::vector<double> ordered = rounds;
    std::sort(ordered.begin(), ordered.end());
    const double median = ordered[ordered.size() / 2U];

    std::vector<std::string> samples;
    samples.reserve(kSamples);
    for (std::size_t index = 0U; index < kSamples; ++index) {
        samples.push_back(evaluate(index).canonical());
    }

    std::cout << std::setprecision(17);
    std::cout << "{\"schema\":\"qsa.qmath-exact-platform.v1\","
              << "\"implementation\":\"QSA::QRational\","
              << "\"iterations\":" << kIterations << ','
              << "\"rounds\":" << kRounds << ','
              << "\"median_seconds\":" << median << ','
              << "\"samples\":[";
    for (std::size_t index = 0U; index < samples.size(); ++index) {
        if (index != 0U) std::cout << ',';
        std::cout << '\"' << samples[index] << '\"';
    }
    std::cout << "]}\n";
}

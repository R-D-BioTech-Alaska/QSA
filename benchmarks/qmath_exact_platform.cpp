#include "qubit/qexact.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qubit;

namespace {

constexpr std::size_t kIterations = 20000U;
constexpr std::size_t kRounds = 7U;
constexpr std::size_t kSamples = 8U;
constexpr std::uint64_t kOffset = 1469598103934665603ULL;
constexpr std::uint64_t kPrime = 1099511628211ULL;

struct RoundResult {
    double seconds{0.0};
    std::uint64_t checksum{0U};
};

QRational evaluate(std::size_t index) {
    const std::int64_t i = static_cast<std::int64_t>(index % 997U) + 2;
    const QRational a(i + 1, i + 2);
    const QRational b(2 * i + 3, i + 5);
    const QRational c(3 * i + 7, 2 * i + 9);
    const QRational d(i + 11, 4 * i + 13);
    return (a + b) * (c - d);
}

std::uint64_t observe(std::uint64_t checksum, const QRational& value) {
    if (!value.numerator().fits_int64() || !value.denominator().fits_int64()) {
        throw std::runtime_error("QMath exact benchmark observation exceeded int64 control range");
    }
    checksum ^= static_cast<std::uint64_t>(value.numerator().to_int64());
    checksum *= kPrime;
    checksum ^= static_cast<std::uint64_t>(value.denominator().to_int64());
    checksum *= kPrime;
    return checksum;
}

RoundResult run_round() {
    std::uint64_t checksum = kOffset;
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t index = 0U; index < kIterations; ++index) {
        checksum = observe(checksum, evaluate(index));
    }
    const auto stopped = std::chrono::steady_clock::now();
    return {
        std::chrono::duration<double>(stopped - started).count(),
        checksum,
    };
}

}  // namespace

int main() {
    for (std::size_t warmup = 0U; warmup < 2U; ++warmup) (void)run_round();

    std::vector<double> rounds;
    rounds.reserve(kRounds);
    std::uint64_t checksum = 0U;
    for (std::size_t round = 0U; round < kRounds; ++round) {
        const RoundResult result = run_round();
        rounds.push_back(result.seconds);
        if (round == 0U) checksum = result.checksum;
        else if (result.checksum != checksum) {
            throw std::runtime_error("QMath exact benchmark checksum changed between identical rounds");
        }
    }
    std::vector<double> ordered = rounds;
    std::sort(ordered.begin(), ordered.end());
    const double median = ordered[ordered.size() / 2U];
    if (!std::isfinite(median) || median <= 0.0) {
        throw std::runtime_error("QMath exact benchmark timing is invalid");
    }

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
              << "\"checksum\":" << checksum << ','
              << "\"samples\":[";
    for (std::size_t index = 0U; index < samples.size(); ++index) {
        if (index != 0U) std::cout << ',';
        std::cout << '\"' << samples[index] << '\"';
    }
    std::cout << "]}\n";
}

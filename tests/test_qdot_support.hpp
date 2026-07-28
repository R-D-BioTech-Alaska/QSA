#pragma once

#include "qubit/qdot.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

inline void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "TEST FAILURE: " << message << '\n';
        std::exit(1);
    }
}

inline void require_near(
    double actual,
    double expected,
    double tolerance,
    const std::string& message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::cerr << "TEST FAILURE: " << message << " actual=" << actual
                  << " expected=" << expected << " tolerance=" << tolerance << '\n';
        std::exit(1);
    }
}

inline double state_fidelity(
    const std::vector<qubit::QComplex>& actual,
    const std::vector<qubit::QComplex>& expected) {
    require(actual.size() == expected.size(), "state dimensions differ");
    qubit::QComplex overlap{};
    for (std::size_t i = 0; i < actual.size(); ++i) {
        overlap += expected[i].conjugate() * actual[i];
    }
    return overlap.norm2();
}

inline double max_probability_error(
    const std::vector<double>& actual,
    const std::vector<double>& expected) {
    require(actual.size() == expected.size(), "probability dimensions differ");
    double error = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        error = std::max(error, std::abs(actual[i] - expected[i]));
    }
    return error;
}

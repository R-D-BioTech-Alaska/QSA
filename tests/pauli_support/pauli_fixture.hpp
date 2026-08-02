#pragma once

#include "qubit/qcausal.hpp"
#include "qubit/qpauli_support.hpp"
#include "qubit/qstate.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace qsa_pauli_support_test {

inline std::vector<std::vector<qubit::PauliSupportTerm>> tripair_support(
    qubit::QubitId first,
    qubit::QubitId second,
    qubit::QubitId third) {
    using Term = qubit::PauliSupportTerm;
    return {
        {{first, 'X'}},
        {{first, 'Y'}},
        {{first, 'Z'}},
        {{second, 'X'}},
        {{second, 'Y'}},
        {{second, 'Z'}},
        {{third, 'X'}},
        {{third, 'Y'}},
        {{third, 'Z'}},
        {{first, 'Z'}, {second, 'Z'}},
        {{second, 'Z'}, {third, 'Z'}},
        {{first, 'Z'}, {third, 'Z'}},
        {{first, 'X'}, {second, 'X'}, {third, 'X'}},
        {{first, 'Y'}, {second, 'Y'}, {third, 'Y'}},
    };
}

inline std::vector<std::string> tripair_words(
    std::size_t qubit_count,
    qubit::QubitId first,
    qubit::QubitId second,
    qubit::QubitId third) {
    const auto support = tripair_support(first, second, third);
    std::vector<std::string> words;
    words.reserve(support.size());
    for (const auto& observable : support) {
        std::string word(qubit_count, 'I');
        for (const qubit::PauliSupportTerm& term : observable) {
            word[static_cast<std::size_t>(term.qubit)] = term.axis;
        }
        words.push_back(std::move(word));
    }
    return words;
}

inline qubit::QRegister make_tripair(
    std::size_t qubit_count,
    qubit::QubitId first,
    qubit::QubitId second,
    qubit::QubitId third) {
    qubit::QRegister state(qubit_count);
    state.apply_ry(first, 0.37);
    state.apply_rz(first, -0.21);
    state.apply_ry(second, -0.44);
    state.apply_rz(second, 0.19);
    state.apply_ry(third, 0.28);
    state.apply_rz(third, 0.33);
    state.apply_cnot(first, second);
    state.apply_cnot(second, third);
    state.apply_cnot(third, first);
    state.apply_ry(first, -0.17);
    state.apply_rz(second, 0.14);
    state.apply_ry(third, 0.23);
    return state;
}

inline double maximum_error(
    const std::vector<double>& first,
    const std::vector<double>& second) {
    if (first.size() != second.size()) {
        return std::numeric_limits<double>::infinity();
    }
    double maximum = 0.0;
    for (std::size_t index = 0; index < first.size(); ++index) {
        maximum = std::max(maximum, std::abs(first[index] - second[index]));
    }
    return maximum;
}

}  // namespace qsa_pauli_support_test

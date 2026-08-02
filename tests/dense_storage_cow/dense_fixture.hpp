#pragma once

#include "qubit/qstate.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace qsa_dense_cow_test {

inline qubit::QRegister make_dense_components(
    std::size_t group_count,
    std::size_t group_width) {
    if (group_count == 0U || group_width < 2U) {
        throw std::invalid_argument("dense component fixture dimensions are invalid");
    }

    qubit::QRegister state(group_count * group_width);
    for (std::size_t group = 0; group < group_count; ++group) {
        const std::size_t base = group * group_width;
        state.apply_h(static_cast<qubit::QubitId>(base));
        for (std::size_t offset = 1; offset < group_width; ++offset) {
            state.apply_cnot(
                static_cast<qubit::QubitId>(base + offset - 1U),
                static_cast<qubit::QubitId>(base + offset));
        }
        for (std::size_t offset = 0; offset < group_width; ++offset) {
            state.apply_h(static_cast<qubit::QubitId>(base + offset));
        }
        state.apply_ry(
            static_cast<qubit::QubitId>(base),
            0.37 + 0.01 * static_cast<double>(group));

        const auto qubit = static_cast<qubit::QubitId>(base);
        if (state.component_size(qubit) != group_width ||
            state.component_storage_mode(qubit) != qubit::StorageMode::Dense) {
            throw std::runtime_error("dense component fixture was not retained");
        }
    }
    return state;
}

inline double complex_error(qubit::QComplex first, qubit::QComplex second) {
    return std::hypot(first.re - second.re, first.im - second.im);
}

inline double max_state_error(
    const qubit::QRegister& first,
    const qubit::QRegister& second) {
    const auto first_views = first.component_read_views();
    const auto second_views = second.component_read_views();
    if (first_views.size() != second_views.size()) {
        return std::numeric_limits<double>::infinity();
    }

    double maximum = 0.0;
    for (std::size_t index = 0; index < first_views.size(); ++index) {
        const auto& left = first_views[index];
        const auto& right = second_views[index];
        if (left.kind != right.kind ||
            left.dimension != right.dimension ||
            left.qubits.size() != right.qubits.size() ||
            !std::equal(left.qubits.begin(), left.qubits.end(), right.qubits.begin())) {
            return std::numeric_limits<double>::infinity();
        }

        if (left.kind == qubit::ComponentKind::Cell) {
            if (left.cell == nullptr || right.cell == nullptr) {
                return std::numeric_limits<double>::infinity();
            }
            maximum = std::max(maximum, std::abs(left.cell->x - right.cell->x));
            maximum = std::max(maximum, std::abs(left.cell->y - right.cell->y));
            maximum = std::max(maximum, std::abs(left.cell->z - right.cell->z));
            continue;
        }

        if (left.kind == qubit::ComponentKind::Dense) {
            if (left.dense.size() != right.dense.size()) {
                return std::numeric_limits<double>::infinity();
            }
            for (std::size_t amplitude = 0; amplitude < left.dense.size(); ++amplitude) {
                maximum = std::max(
                    maximum,
                    complex_error(left.dense[amplitude], right.dense[amplitude]));
            }
            continue;
        }

        if (left.sparse.size() != right.sparse.size()) {
            return std::numeric_limits<double>::infinity();
        }
        for (std::size_t entry = 0; entry < left.sparse.size(); ++entry) {
            if (left.sparse[entry].first != right.sparse[entry].first) {
                return std::numeric_limits<double>::infinity();
            }
            maximum = std::max(
                maximum,
                complex_error(left.sparse[entry].second, right.sparse[entry].second));
        }
    }
    return maximum;
}

}  // namespace qsa_dense_cow_test

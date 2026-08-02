#pragma once

#include "qubit/qstate.hpp"

#include <cstddef>
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

}  // namespace qsa_dense_cow_test

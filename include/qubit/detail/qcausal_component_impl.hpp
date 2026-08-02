#pragma once

#include "qubit/qcausal_component_api.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace {

[[nodiscard]] bool same_component_view(
    const qubit::QComponentReadView& first,
    const qubit::QComponentReadView& second) noexcept {
    return first.qubits.data() == second.qubits.data() &&
        first.qubits.size() == second.qubits.size();
}

[[nodiscard]] qubit::QubitId first_component_qubit(
    const qubit::QComponentReadView& view) {
    if (view.qubits.empty()) {
        throw qubit::QStateError("QSA component has no logical qubits");
    }
    return *std::min_element(view.qubits.begin(), view.qubits.end());
}

[[nodiscard]] std::vector<qubit::QComplex> component_amplitudes(
    const qubit::QComponentReadView& view) {
    if (view.kind == qubit::ComponentKind::Cell) {
        if (view.cell == nullptr || view.qubits.size() != 1U) {
            throw qubit::QStateError("Invalid Bloch component closure");
        }
        const auto amplitudes = view.cell->amplitudes();
        return {amplitudes[0], amplitudes[1]};
    }

    if (view.dimension == 0U ||
        view.dimension > static_cast<qubit::BasisIndex>(
            std::numeric_limits<std::size_t>::max())) {
        throw qubit::QStateError("Component closure dimension is invalid");
    }
    const std::size_t dimension = static_cast<std::size_t>(view.dimension);

    if (view.kind == qubit::ComponentKind::Dense) {
        if (view.dense.size() != dimension) {
            throw qubit::QStateError("Invalid dense component closure");
        }
        return {view.dense.begin(), view.dense.end()};
    }

    if (view.kind == qubit::ComponentKind::Sparse) {
        std::vector<qubit::QComplex> amplitudes(dimension);
        for (const auto& [index, amplitude] : view.sparse) {
            if (index >= view.dimension) {
                throw qubit::QStateError("Sparse component closure index is invalid");
            }
            amplitudes[static_cast<std::size_t>(index)] = amplitude;
        }
        return amplitudes;
    }

    throw qubit::QStateError("Unknown component closure kind");
}

[[nodiscard]] qubit::QRegister extract_component_closure(
    const qubit::QRegister& source,
    std::span<const std::uint32_t> requested_qubits,
    std::size_t max_local_qubits,
    std::vector<qubit::QubitId>& global_qubits) {
    if (requested_qubits.empty()) {
        throw qubit::QStateError("Component closure request is empty");
    }
    if (max_local_qubits == 0U || max_local_qubits > 24U) {
        throw qubit::QStateError("Component closure limit must be between 1 and 24 qubits");
    }

    std::vector<qubit::QComponentReadView> components;
    components.reserve(requested_qubits.size());
    for (std::uint32_t qubit : requested_qubits) {
        const qubit::QComponentReadView view = source.component_read_view(qubit);
        const bool duplicate = std::any_of(
            components.begin(),
            components.end(),
            [&](const qubit::QComponentReadView& existing) {
                return same_component_view(existing, view);
            });
        if (!duplicate) {
            components.push_back(view);
        }
    }
    std::sort(
        components.begin(),
        components.end(),
        [](const qubit::QComponentReadView& first,
           const qubit::QComponentReadView& second) {
            return first_component_qubit(first) < first_component_qubit(second);
        });

    std::size_t local_qubit_count = 0U;
    for (const qubit::QComponentReadView& component : components) {
        if (component.qubits.size() > max_local_qubits - local_qubit_count) {
            throw qubit::QStateError("Component closure exceeds the local qubit limit");
        }
        local_qubit_count += component.qubits.size();
    }

    std::vector<qubit::QComplex> amplitudes{{1.0, 0.0}};
    global_qubits.clear();
    global_qubits.reserve(local_qubit_count);
    for (const qubit::QComponentReadView& component : components) {
        const std::vector<qubit::QComplex> local = component_amplitudes(component);
        if (local.empty() ||
            amplitudes.size() > std::numeric_limits<std::size_t>::max() / local.size()) {
            throw qubit::QStateError("Component closure amplitude count is too large");
        }
        std::vector<qubit::QComplex> combined(amplitudes.size() * local.size());
        for (std::size_t local_index = 0; local_index < local.size(); ++local_index) {
            const std::size_t offset = local_index * amplitudes.size();
            for (std::size_t root_index = 0; root_index < amplitudes.size(); ++root_index) {
                combined[offset + root_index] =
                    amplitudes[root_index] * local[local_index];
            }
        }
        amplitudes = std::move(combined);
        global_qubits.insert(
            global_qubits.end(),
            component.qubits.begin(),
            component.qubits.end());
    }

    if (global_qubits.size() != local_qubit_count) {
        throw qubit::QStateError("Component closure mapping is incomplete");
    }
    return qubit::QRegister::from_amplitudes(
        std::move(amplitudes),
        source.config());
}

}  // namespace

extern "C" {

qcausal_handle qcausal_extract_component_closure(
    qcausal_handle source,
    const uint32_t* requested_qubits,
    size_t requested_count,
    size_t max_local_qubits,
    uint32_t* global_qubits,
    size_t global_qubit_capacity,
    size_t* global_qubit_count) {
    try {
        if (global_qubit_count == nullptr) {
            throw qubit::QStateError("Component closure count output is null");
        }
        *global_qubit_count = 0U;
        if (requested_count != 0U && requested_qubits == nullptr) {
            throw qubit::QStateError("Component closure request buffer is null");
        }
        if (global_qubit_capacity != 0U && global_qubits == nullptr) {
            throw qubit::QStateError("Component closure mapping output is null");
        }

        CausalHandle* root = as_causal(source);
        std::vector<qubit::QubitId> mapping;
        qubit::QRegister local = extract_component_closure(
            root->state.read(),
            std::span<const std::uint32_t>(requested_qubits, requested_count),
            max_local_qubits,
            mapping);
        if (global_qubit_capacity < mapping.size()) {
            throw qubit::QStateError("Component closure mapping output is too small");
        }
        std::copy(mapping.begin(), mapping.end(), global_qubits);
        *global_qubit_count = mapping.size();

        auto* handle = new CausalHandle(std::move(local));
        causal_last_error.clear();
        return handle;
    } catch (const std::exception& error) {
        causal_last_error = error.what();
        return nullptr;
    } catch (...) {
        causal_last_error = "Unknown QSA component closure error";
        return nullptr;
    }
}

}  // extern "C"

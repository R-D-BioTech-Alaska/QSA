#pragma once

#include "qubit/qstate.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace qubit {

struct ExactStructuredOverlapConfig {
    std::size_t max_qubits{1U << 20U};
    std::size_t max_union_block_qubits{24U};
    std::size_t max_basis_evaluations{1U << 24U};
};

struct ExactStructuredOverlapStats {
    std::size_t qubits{0U};
    std::size_t left_components{0U};
    std::size_t right_components{0U};
    std::size_t union_blocks{0U};
    std::size_t max_union_block_qubits{0U};
    std::size_t basis_evaluations{0U};
};

struct ExactStructuredOverlapResult {
    QComplex inner_product{};
    double fidelity{0.0};
    double log2_fidelity{-std::numeric_limits<double>::infinity()};
    ExactStructuredOverlapStats stats{};
};

namespace detail {

class OverlapDisjointSet {
public:
    explicit OverlapDisjointSet(std::size_t size)
        : parent_(size), rank_(size, 0U) {
        for (std::size_t index = 0U; index < size; ++index) {
            parent_[index] = index;
        }
    }

    [[nodiscard]] std::size_t find(std::size_t value) {
        std::size_t root = value;
        while (parent_[root] != root) {
            root = parent_[root];
        }
        while (parent_[value] != value) {
            const std::size_t next = parent_[value];
            parent_[value] = root;
            value = next;
        }
        return root;
    }

    void unite(std::size_t left, std::size_t right) {
        left = find(left);
        right = find(right);
        if (left == right) {
            return;
        }
        if (rank_[left] < rank_[right]) {
            std::swap(left, right);
        }
        parent_[right] = left;
        if (rank_[left] == rank_[right]) {
            ++rank_[left];
        }
    }

private:
    std::vector<std::size_t> parent_{};
    std::vector<std::uint8_t> rank_{};
};

struct OverlapAccessor {
    QComponentReadView view{};
    std::vector<std::size_t> positions{};
    std::array<QComplex, 2> cell_amplitudes{};
};

struct OverlapBlock {
    std::vector<QubitId> qubits{};
    std::vector<OverlapAccessor> left{};
    std::vector<OverlapAccessor> right{};
};

[[nodiscard]] inline QComplex overlap_component_amplitude(
    const OverlapAccessor& accessor,
    std::size_t block_basis) {
    std::size_t local_basis = 0U;
    for (std::size_t local = 0U; local < accessor.positions.size(); ++local) {
        local_basis |= ((block_basis >> accessor.positions[local]) & 1U) << local;
    }

    switch (accessor.view.kind) {
        case ComponentKind::Cell:
            return accessor.cell_amplitudes[local_basis];
        case ComponentKind::Dense:
            if (local_basis >= accessor.view.dense.size()) {
                throw QStateError("Structured overlap dense component index is out of range");
            }
            return accessor.view.dense[local_basis];
        case ComponentKind::Sparse: {
            const auto begin = accessor.view.sparse.begin();
            const auto end = accessor.view.sparse.end();
            const auto found = std::lower_bound(
                begin,
                end,
                static_cast<BasisIndex>(local_basis),
                [](const AmplitudeStore::SparseEntry& entry, BasisIndex index) {
                    return entry.first < index;
                });
            if (found == end || found->first != static_cast<BasisIndex>(local_basis)) {
                return {};
            }
            return found->second;
        }
    }
    throw QStateError("Structured overlap encountered an unknown component kind");
}

[[nodiscard]] inline std::size_t overlap_checked_sum(
    std::size_t left,
    std::size_t right,
    const char* message) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw QStateError(message);
    }
    return left + right;
}

}  // namespace detail

[[nodiscard]] inline ExactStructuredOverlapResult exact_structured_overlap(
    const QRegister& left,
    const QRegister& right,
    ExactStructuredOverlapConfig config = {}) {
    const std::size_t qubits = left.qubit_count();
    if (qubits == 0U || qubits != right.qubit_count() ||
        qubits > config.max_qubits || config.max_union_block_qubits == 0U ||
        config.max_union_block_qubits >= std::numeric_limits<std::size_t>::digits ||
        config.max_basis_evaluations == 0U) {
        throw QStateError("Structured overlap dimensions or configuration are invalid");
    }

    std::string reason;
    if (!left.validate(&reason)) {
        throw QStateError("Structured overlap left state is invalid: " + reason);
    }
    if (!right.validate(&reason)) {
        throw QStateError("Structured overlap right state is invalid: " + reason);
    }

    const auto left_views = left.component_read_views();
    const auto right_views = right.component_read_views();
    detail::OverlapDisjointSet sets(qubits);
    const auto join_views = [&](std::span<const QComponentReadView> views) {
        for (const QComponentReadView& view : views) {
            if (view.qubits.empty()) {
                throw QStateError("Structured overlap encountered an empty component");
            }
            const std::size_t first = static_cast<std::size_t>(view.qubits.front());
            for (const QubitId qubit : view.qubits.subspan(1U)) {
                sets.unite(first, static_cast<std::size_t>(qubit));
            }
        }
    };
    join_views(left_views);
    join_views(right_views);

    const std::size_t npos = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> block_by_root(qubits, npos);
    std::vector<std::size_t> block_by_qubit(qubits, npos);
    std::vector<detail::OverlapBlock> blocks;
    for (std::size_t qubit = 0U; qubit < qubits; ++qubit) {
        const std::size_t root = sets.find(qubit);
        std::size_t block = block_by_root[root];
        if (block == npos) {
            block = blocks.size();
            block_by_root[root] = block;
            blocks.emplace_back();
        }
        blocks[block].qubits.push_back(static_cast<QubitId>(qubit));
        block_by_qubit[qubit] = block;
    }

    std::vector<std::size_t> position_by_qubit(qubits, npos);
    std::size_t total_evaluations = 0U;
    std::size_t maximum_block = 0U;
    for (detail::OverlapBlock& block : blocks) {
        const std::size_t width = block.qubits.size();
        if (width > config.max_union_block_qubits) {
            throw QStateError("Structured overlap union block exceeds configured qubit cap");
        }
        maximum_block = std::max(maximum_block, width);
        const std::size_t evaluations = std::size_t{1U} << width;
        total_evaluations = detail::overlap_checked_sum(
            total_evaluations,
            evaluations,
            "Structured overlap basis-evaluation count overflowed");
        if (total_evaluations > config.max_basis_evaluations) {
            throw QStateError("Structured overlap exceeds configured basis-evaluation cap");
        }
        for (std::size_t position = 0U; position < width; ++position) {
            position_by_qubit[static_cast<std::size_t>(block.qubits[position])] = position;
        }
    }

    const auto attach_views = [&](std::span<const QComponentReadView> views, bool is_left) {
        for (const QComponentReadView& view : views) {
            const std::size_t block_index = block_by_qubit[static_cast<std::size_t>(view.qubits.front())];
            detail::OverlapAccessor accessor;
            accessor.view = view;
            accessor.positions.reserve(view.qubits.size());
            for (const QubitId qubit : view.qubits) {
                if (block_by_qubit[static_cast<std::size_t>(qubit)] != block_index) {
                    throw QStateError("Structured overlap component crosses union blocks");
                }
                accessor.positions.push_back(position_by_qubit[static_cast<std::size_t>(qubit)]);
            }
            if (view.kind == ComponentKind::Cell) {
                if (view.cell == nullptr || view.qubits.size() != 1U) {
                    throw QStateError("Structured overlap cell view is invalid");
                }
                accessor.cell_amplitudes = view.cell->amplitudes();
            }
            if (is_left) {
                blocks[block_index].left.push_back(std::move(accessor));
            } else {
                blocks[block_index].right.push_back(std::move(accessor));
            }
        }
    };
    attach_views(left_views, true);
    attach_views(right_views, false);

    QComplex overlap{1.0, 0.0};
    double log2_fidelity = 0.0;
    bool zero_fidelity = false;
    for (const detail::OverlapBlock& block : blocks) {
        const std::size_t dimension = std::size_t{1U} << block.qubits.size();
        QComplex block_overlap{};
        for (std::size_t basis = 0U; basis < dimension; ++basis) {
            QComplex left_amplitude{1.0, 0.0};
            for (const detail::OverlapAccessor& accessor : block.left) {
                left_amplitude *= detail::overlap_component_amplitude(accessor, basis);
            }
            QComplex right_amplitude{1.0, 0.0};
            for (const detail::OverlapAccessor& accessor : block.right) {
                right_amplitude *= detail::overlap_component_amplitude(accessor, basis);
            }
            block_overlap += left_amplitude.conjugate() * right_amplitude;
        }
        const double block_fidelity = block_overlap.norm2();
        if (!std::isfinite(block_fidelity)) {
            throw QStateError("Structured overlap block fidelity became non-finite");
        }
        if (block_fidelity == 0.0) {
            zero_fidelity = true;
        } else if (!zero_fidelity) {
            log2_fidelity += std::log2(block_fidelity);
        }
        overlap *= block_overlap;
        if (!std::isfinite(overlap.re) || !std::isfinite(overlap.im)) {
            throw QStateError("Structured overlap became non-finite");
        }
    }
    if (zero_fidelity) {
        log2_fidelity = -std::numeric_limits<double>::infinity();
    }

    return ExactStructuredOverlapResult{
        overlap,
        overlap.norm2(),
        log2_fidelity,
        ExactStructuredOverlapStats{
            qubits,
            left_views.size(),
            right_views.size(),
            blocks.size(),
            maximum_block,
            total_evaluations,
        },
    };
}

}  // namespace qubit

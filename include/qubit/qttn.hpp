#pragma once

#include "qubit/qplan.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace qubit {

struct TreeTensorConfig {
    std::size_t max_bond_dimension{1024U};
    std::size_t max_scalars{1U << 24U};
    std::size_t max_materialize_qubits{24U};
    double validation_tolerance{1e-10};
};

struct TreeTensorStats {
    std::size_t qubit_count{0U};
    std::size_t node_count{0U};
    std::size_t scalar_count{0U};
    std::size_t max_bond_dimension{1U};
    std::size_t controlled_gate_count{0U};
    std::size_t max_controlled_path_edges{0U};
    std::size_t generation{0U};
};

class TreeTensorState;

class TreeTensorWorkspace {
public:
    TreeTensorWorkspace() = default;

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) +
                            vectors_.capacity() * sizeof(std::vector<QComplex>) +
                            environments_.capacity() * sizeof(std::vector<QComplex>) +
                            assignment_.capacity() * sizeof(std::int8_t);
        for (const auto& values : vectors_) {
            bytes += values.capacity() * sizeof(QComplex);
        }
        for (const auto& values : environments_) {
            bytes += values.capacity() * sizeof(QComplex);
        }
        return bytes;
    }

private:
    std::vector<std::vector<QComplex>> vectors_{};
    std::vector<std::vector<QComplex>> environments_{};
    std::vector<std::int8_t> assignment_{};
    std::size_t generation_{std::numeric_limits<std::size_t>::max()};

    friend class TreeTensorState;
};

class TreeTensorState {
public:
    explicit TreeTensorState(
        std::size_t qubit_count,
        TreeTensorConfig config = {})
        : config_(config), qubit_count_(qubit_count) {
        if (qubit_count_ == 0U) {
            throw QStateError("Tree tensor state requires at least one qubit");
        }
        if (qubit_count_ - 1U >
            static_cast<std::size_t>(std::numeric_limits<QubitId>::max())) {
            throw QStateError("Tree tensor state qubit count exceeds QubitId range");
        }
        if (config_.max_bond_dimension == 0U || config_.max_scalars == 0U) {
            throw QStateError("Tree tensor configuration contains a zero resource cap");
        }
        if (!std::isfinite(config_.validation_tolerance) ||
            config_.validation_tolerance <= 0.0) {
            throw QStateError("Tree tensor validation tolerance must be positive and finite");
        }
        if (qubit_count_ > (std::numeric_limits<std::size_t>::max() + 1U) / 2U) {
            throw QStateError("Tree tensor topology size overflowed");
        }

        qubit_node_.resize(qubit_count_);
        nodes_.reserve(2U * qubit_count_ - 1U);
        root_ = build_topology(0U, qubit_count_, 0U);
        stats_.qubit_count = qubit_count_;
        stats_.node_count = nodes_.size();
        stats_.scalar_count = checked_initial_scalars();
        if (stats_.scalar_count > config_.max_scalars) {
            throw QStateError("Tree tensor zero state exceeds configured scalar cap");
        }
    }

    [[nodiscard]] static TreeTensorState from_operations(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        TreeTensorConfig config = {}) {
        TreeTensorState state(qubit_count, config);
        for (const Operation& operation : operations) {
            state.apply(operation);
        }
        return state;
    }

    [[nodiscard]] std::size_t qubit_count() const noexcept { return qubit_count_; }
    [[nodiscard]] const TreeTensorConfig& config() const noexcept { return config_; }
    [[nodiscard]] const TreeTensorStats& stats() const noexcept { return stats_; }

    [[nodiscard]] TreeTensorWorkspace workspace() const {
        TreeTensorWorkspace result;
        refresh_workspace(result);
        return result;
    }

    void apply(const Operation& operation) {
        if (operation.kind == OperationKind::Trajectory) {
            throw QStateError("Tree tensor state does not support trajectory operations");
        }
        if (operation.kind == OperationKind::Single) {
            apply_unitary(operation.first, operation.single);
            return;
        }
        if (operation.kind == OperationKind::Two) {
            if (operation.code == OperationCode::Cnot) {
                apply_cnot(operation.first, operation.second);
                return;
            }
            if (operation.code == OperationCode::Cz) {
                apply_cz(operation.first, operation.second);
                return;
            }
            throw QStateError(
                "Tree tensor state currently supports only CNOT/CZ two-qubit operations");
        }
        throw QStateError("Tree tensor operation kind is invalid");
    }

    void apply_unitary(QubitId qubit, const QMatrix2& matrix) {
        validate_qubit(qubit);
        validate_unitary(matrix);
        Node& node = nodes_[qubit_node_[qubit]];
        const std::size_t dimension = node.parent_dimension;
        std::vector<QComplex> next_zero(dimension);
        std::vector<QComplex> next_one(dimension);
        for (std::size_t bond = 0U; bond < dimension; ++bond) {
            next_zero[bond] = matrix(0U, 0U) * node.zero[bond] +
                              matrix(0U, 1U) * node.one[bond];
            next_one[bond] = matrix(1U, 0U) * node.zero[bond] +
                             matrix(1U, 1U) * node.one[bond];
        }
        node.zero = std::move(next_zero);
        node.one = std::move(next_one);
        advance_generation();
    }

    void apply_cnot(QubitId control, QubitId target) {
        static const QMatrix2 identity{{
            QComplex{1.0}, QComplex{}, QComplex{}, QComplex{1.0},
        }};
        static const QMatrix2 x{{
            QComplex{}, QComplex{1.0}, QComplex{1.0}, QComplex{},
        }};
        apply_controlled(control, target, identity, x);
    }

    void apply_cz(QubitId first, QubitId second) {
        static const QMatrix2 identity{{
            QComplex{1.0}, QComplex{}, QComplex{}, QComplex{1.0},
        }};
        static const QMatrix2 z{{
            QComplex{1.0}, QComplex{}, QComplex{}, QComplex{-1.0},
        }};
        apply_controlled(first, second, identity, z);
    }

    [[nodiscard]] QComplex amplitude(
        std::span<const std::uint8_t> bits,
        TreeTensorWorkspace& workspace_value) const {
        if (bits.size() != qubit_count_) {
            throw QStateError("Tree tensor amplitude bit count does not match qubit count");
        }
        refresh_workspace(workspace_value);
        for (std::size_t node_index = 0U; node_index < nodes_.size(); ++node_index) {
            const Node& node = nodes_[node_index];
            std::vector<QComplex>& output = workspace_value.vectors_[node_index];
            if (node.leaf) {
                const std::uint8_t bit = bits[node.qubit];
                if (bit > 1U) {
                    throw QStateError("Tree tensor amplitude bits must be 0 or 1");
                }
                const std::vector<QComplex>& source = bit == 0U ? node.zero : node.one;
                std::copy(source.begin(), source.end(), output.begin());
                continue;
            }

            std::fill(output.begin(), output.end(), QComplex{});
            const Node& left = nodes_[node.left];
            const Node& right = nodes_[node.right];
            const std::vector<QComplex>& left_value = workspace_value.vectors_[node.left];
            const std::vector<QComplex>& right_value = workspace_value.vectors_[node.right];
            const std::size_t left_dimension = left.parent_dimension;
            const std::size_t right_dimension = right.parent_dimension;
            for (std::size_t left_index = 0U; left_index < left_dimension; ++left_index) {
                for (std::size_t right_index = 0U;
                     right_index < right_dimension;
                     ++right_index) {
                    const QComplex children =
                        left_value[left_index] * right_value[right_index];
                    if (children.re == 0.0 && children.im == 0.0) {
                        continue;
                    }
                    for (std::size_t parent_index = 0U;
                         parent_index < node.parent_dimension;
                         ++parent_index) {
                        output[parent_index] += children * node.values[
                            tensor_index(
                                left_index,
                                right_index,
                                parent_index,
                                right_dimension,
                                node.parent_dimension)];
                    }
                }
            }
        }
        return workspace_value.vectors_[root_][0U];
    }

    [[nodiscard]] QComplex amplitude(std::span<const std::uint8_t> bits) const {
        TreeTensorWorkspace local = workspace();
        return amplitude(bits, local);
    }

    [[nodiscard]] QComplex amplitude(
        BasisIndex basis,
        TreeTensorWorkspace& workspace_value) const {
        if (qubit_count_ > std::numeric_limits<BasisIndex>::digits) {
            throw QStateError("Tree tensor BasisIndex amplitude exceeds BasisIndex width");
        }
        std::vector<std::uint8_t> bits(qubit_count_);
        for (std::size_t qubit = 0U; qubit < qubit_count_; ++qubit) {
            bits[qubit] = static_cast<std::uint8_t>((basis >> qubit) & 1U);
        }
        return amplitude(bits, workspace_value);
    }

    [[nodiscard]] QComplex amplitude(BasisIndex basis) const {
        TreeTensorWorkspace local = workspace();
        return amplitude(basis, local);
    }

    [[nodiscard]] double probability(
        std::span<const std::uint8_t> bits,
        TreeTensorWorkspace& workspace_value) const {
        const double value = amplitude(bits, workspace_value).norm2();
        return checked_probability(value);
    }

    [[nodiscard]] double probability(std::span<const std::uint8_t> bits) const {
        TreeTensorWorkspace local = workspace();
        return probability(bits, local);
    }

    [[nodiscard]] double marginal_probability(
        std::span<const QubitId> selected_qubits,
        std::span<const std::uint8_t> selected_bits,
        TreeTensorWorkspace& workspace_value) const {
        if (selected_qubits.size() != selected_bits.size()) {
            throw QStateError("Tree tensor marginal qubit/bit sizes differ");
        }
        refresh_workspace(workspace_value);
        std::fill(
            workspace_value.assignment_.begin(),
            workspace_value.assignment_.end(),
            static_cast<std::int8_t>(-1));
        for (std::size_t index = 0U; index < selected_qubits.size(); ++index) {
            const QubitId qubit = selected_qubits[index];
            validate_qubit(qubit);
            if (selected_bits[index] > 1U) {
                throw QStateError("Tree tensor marginal bits must be 0 or 1");
            }
            if (workspace_value.assignment_[qubit] >= 0) {
                throw QStateError("Tree tensor marginal repeats a qubit");
            }
            workspace_value.assignment_[qubit] =
                static_cast<std::int8_t>(selected_bits[index]);
        }

        ensure_environments(workspace_value);
        for (std::size_t node_index = 0U; node_index < nodes_.size(); ++node_index) {
            const Node& node = nodes_[node_index];
            std::vector<QComplex>& environment =
                workspace_value.environments_[node_index];
            std::fill(environment.begin(), environment.end(), QComplex{});
            if (node.leaf) {
                const std::int8_t selected = workspace_value.assignment_[node.qubit];
                for (std::size_t bra = 0U; bra < node.parent_dimension; ++bra) {
                    for (std::size_t ket = 0U; ket < node.parent_dimension; ++ket) {
                        QComplex value{};
                        if (selected <= 0) {
                            value += node.zero[bra].conjugate() * node.zero[ket];
                        }
                        if (selected < 0 || selected == 1) {
                            value += node.one[bra].conjugate() * node.one[ket];
                        }
                        environment[bra * node.parent_dimension + ket] = value;
                    }
                }
                continue;
            }

            const Node& left = nodes_[node.left];
            const Node& right = nodes_[node.right];
            const std::size_t left_dimension = left.parent_dimension;
            const std::size_t right_dimension = right.parent_dimension;
            const std::vector<QComplex>& left_environment =
                workspace_value.environments_[node.left];
            const std::vector<QComplex>& right_environment =
                workspace_value.environments_[node.right];
            for (std::size_t parent_bra = 0U;
                 parent_bra < node.parent_dimension;
                 ++parent_bra) {
                for (std::size_t parent_ket = 0U;
                     parent_ket < node.parent_dimension;
                     ++parent_ket) {
                    QComplex value{};
                    for (std::size_t left_bra = 0U;
                         left_bra < left_dimension;
                         ++left_bra) {
                        for (std::size_t left_ket = 0U;
                             left_ket < left_dimension;
                             ++left_ket) {
                            const QComplex left_value = left_environment[
                                left_bra * left_dimension + left_ket];
                            if (left_value.re == 0.0 && left_value.im == 0.0) {
                                continue;
                            }
                            for (std::size_t right_bra = 0U;
                                 right_bra < right_dimension;
                                 ++right_bra) {
                                for (std::size_t right_ket = 0U;
                                     right_ket < right_dimension;
                                     ++right_ket) {
                                    const QComplex right_value = right_environment[
                                        right_bra * right_dimension + right_ket];
                                    if (right_value.re == 0.0 && right_value.im == 0.0) {
                                        continue;
                                    }
                                    const QComplex bra_value = node.values[
                                        tensor_index(
                                            left_bra,
                                            right_bra,
                                            parent_bra,
                                            right_dimension,
                                            node.parent_dimension)];
                                    const QComplex ket_value = node.values[
                                        tensor_index(
                                            left_ket,
                                            right_ket,
                                            parent_ket,
                                            right_dimension,
                                            node.parent_dimension)];
                                    value += bra_value.conjugate() * ket_value *
                                             left_value * right_value;
                                }
                            }
                        }
                    }
                    environment[parent_bra * node.parent_dimension + parent_ket] = value;
                }
            }
        }

        const QComplex result = workspace_value.environments_[root_][0U];
        const double scale = 1.0 + std::abs(result.re);
        if (std::abs(result.im) > config_.validation_tolerance * scale) {
            throw QStateError("Tree tensor marginal produced a complex probability");
        }
        return checked_probability(result.re);
    }

    [[nodiscard]] double marginal_probability(
        std::span<const QubitId> selected_qubits,
        std::span<const std::uint8_t> selected_bits) const {
        TreeTensorWorkspace local = workspace();
        return marginal_probability(selected_qubits, selected_bits, local);
    }

    [[nodiscard]] double norm2(TreeTensorWorkspace& workspace_value) const {
        return marginal_probability({}, {}, workspace_value);
    }

    [[nodiscard]] double norm2() const {
        TreeTensorWorkspace local = workspace();
        return norm2(local);
    }

    [[nodiscard]] std::vector<QComplex> materialize(
        std::size_t max_qubits = 0U) const {
        const std::size_t limit = max_qubits == 0U
            ? config_.max_materialize_qubits
            : std::min(max_qubits, config_.max_materialize_qubits);
        if (qubit_count_ > limit || qubit_count_ >= std::numeric_limits<std::size_t>::digits) {
            throw QStateError("Tree tensor materialization exceeds configured qubit cap");
        }
        const std::size_t entries = std::size_t{1U} << qubit_count_;
        std::vector<QComplex> output(entries);
        TreeTensorWorkspace local = workspace();
        for (std::size_t index = 0U; index < entries; ++index) {
            output[index] = amplitude(static_cast<BasisIndex>(index), local);
        }
        return output;
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        std::size_t bytes = sizeof(*this) +
                            nodes_.capacity() * sizeof(Node) +
                            qubit_node_.capacity() * sizeof(std::size_t);
        for (const Node& node : nodes_) {
            bytes += node.zero.capacity() * sizeof(QComplex) +
                     node.one.capacity() * sizeof(QComplex) +
                     node.values.capacity() * sizeof(QComplex);
        }
        return bytes;
    }

    [[nodiscard]] bool validate(std::string* reason = nullptr) const noexcept {
        try {
            if (nodes_.empty() || root_ >= nodes_.size() || nodes_[root_].parent != no_node()) {
                if (reason != nullptr) {
                    *reason = "Tree tensor root topology is invalid";
                }
                return false;
            }
            std::size_t scalar_count = 0U;
            std::size_t max_bond = 1U;
            for (std::size_t index = 0U; index < nodes_.size(); ++index) {
                const Node& node = nodes_[index];
                if (node.parent_dimension == 0U ||
                    node.parent_dimension > config_.max_bond_dimension) {
                    if (reason != nullptr) {
                        *reason = "Tree tensor bond dimension is invalid";
                    }
                    return false;
                }
                max_bond = std::max(max_bond, node.parent_dimension);
                if (node.leaf) {
                    if (node.zero.size() != node.parent_dimension ||
                        node.one.size() != node.parent_dimension ||
                        node.qubit >= qubit_count_ || qubit_node_[node.qubit] != index) {
                        if (reason != nullptr) {
                            *reason = "Tree tensor leaf shape is invalid";
                        }
                        return false;
                    }
                    scalar_count += node.zero.size() + node.one.size();
                    for (const QComplex& value : node.zero) {
                        if (!finite(value)) {
                            if (reason != nullptr) {
                                *reason = "Tree tensor contains non-finite values";
                            }
                            return false;
                        }
                    }
                    for (const QComplex& value : node.one) {
                        if (!finite(value)) {
                            if (reason != nullptr) {
                                *reason = "Tree tensor contains non-finite values";
                            }
                            return false;
                        }
                    }
                    continue;
                }
                if (node.left >= index || node.right >= index ||
                    nodes_[node.left].parent != index || nodes_[node.right].parent != index) {
                    if (reason != nullptr) {
                        *reason = "Tree tensor internal topology is invalid";
                    }
                    return false;
                }
                const std::size_t expected = checked_tensor_size(
                    nodes_[node.left].parent_dimension,
                    nodes_[node.right].parent_dimension,
                    node.parent_dimension);
                if (node.values.size() != expected) {
                    if (reason != nullptr) {
                        *reason = "Tree tensor internal shape is invalid";
                    }
                    return false;
                }
                scalar_count += node.values.size();
                for (const QComplex& value : node.values) {
                    if (!finite(value)) {
                        if (reason != nullptr) {
                            *reason = "Tree tensor contains non-finite values";
                        }
                        return false;
                    }
                }
            }
            if (scalar_count != stats_.scalar_count || max_bond != stats_.max_bond_dimension) {
                if (reason != nullptr) {
                    *reason = "Tree tensor statistics do not match storage";
                }
                return false;
            }
            return true;
        } catch (...) {
            if (reason != nullptr) {
                *reason = "Tree tensor validation encountered an invalid shape";
            }
            return false;
        }
    }

private:
    struct Node {
        bool leaf{false};
        std::size_t parent{no_node()};
        std::size_t left{no_node()};
        std::size_t right{no_node()};
        std::size_t depth{0U};
        std::size_t parent_dimension{1U};
        QubitId qubit{0U};
        std::vector<QComplex> zero{};
        std::vector<QComplex> one{};
        std::vector<QComplex> values{};
    };

    struct PendingNode {
        std::size_t node{0U};
        std::size_t parent_dimension{1U};
        std::vector<QComplex> zero{};
        std::vector<QComplex> one{};
        std::vector<QComplex> values{};
    };

    TreeTensorConfig config_{};
    std::size_t qubit_count_{0U};
    std::vector<Node> nodes_{};
    std::vector<std::size_t> qubit_node_{};
    std::size_t root_{no_node()};
    TreeTensorStats stats_{};

    [[nodiscard]] static constexpr std::size_t no_node() noexcept {
        return std::numeric_limits<std::size_t>::max();
    }

    [[nodiscard]] static bool finite(const QComplex& value) noexcept {
        return std::isfinite(value.re) && std::isfinite(value.im);
    }

    [[nodiscard]] static std::size_t tensor_index(
        std::size_t left,
        std::size_t right,
        std::size_t parent,
        std::size_t right_dimension,
        std::size_t parent_dimension) noexcept {
        return (left * right_dimension + right) * parent_dimension + parent;
    }

    [[nodiscard]] static std::size_t checked_product(
        std::size_t first,
        std::size_t second) {
        if (first != 0U && second > std::numeric_limits<std::size_t>::max() / first) {
            throw QStateError("Tree tensor size multiplication overflowed");
        }
        return first * second;
    }

    [[nodiscard]] static std::size_t checked_tensor_size(
        std::size_t left,
        std::size_t right,
        std::size_t parent) {
        return checked_product(checked_product(left, right), parent);
    }

    [[nodiscard]] std::size_t build_topology(
        std::size_t begin,
        std::size_t end,
        std::size_t depth) {
        if (end - begin == 1U) {
            Node node;
            node.leaf = true;
            node.depth = depth;
            node.qubit = static_cast<QubitId>(begin);
            node.zero = {QComplex{1.0, 0.0}};
            node.one = {QComplex{}};
            const std::size_t index = nodes_.size();
            nodes_.push_back(std::move(node));
            qubit_node_[begin] = index;
            return index;
        }

        const std::size_t middle = begin + (end - begin) / 2U;
        const std::size_t left = build_topology(begin, middle, depth + 1U);
        const std::size_t right = build_topology(middle, end, depth + 1U);
        Node node;
        node.left = left;
        node.right = right;
        node.depth = depth;
        node.values = {QComplex{1.0, 0.0}};
        const std::size_t index = nodes_.size();
        nodes_.push_back(std::move(node));
        nodes_[left].parent = index;
        nodes_[right].parent = index;
        return index;
    }

    [[nodiscard]] std::size_t checked_initial_scalars() const {
        std::size_t total = 0U;
        for (const Node& node : nodes_) {
            const std::size_t local = node.leaf
                ? node.zero.size() + node.one.size()
                : node.values.size();
            if (total > std::numeric_limits<std::size_t>::max() - local) {
                throw QStateError("Tree tensor scalar count overflowed");
            }
            total += local;
        }
        return total;
    }

    void validate_qubit(QubitId qubit) const {
        if (static_cast<std::size_t>(qubit) >= qubit_count_) {
            throw QStateError("Tree tensor qubit is out of range");
        }
    }

    void validate_unitary(const QMatrix2& matrix) const {
        for (const QComplex& value : matrix.values) {
            if (!finite(value)) {
                throw QStateError("Tree tensor gate contains non-finite values");
            }
        }
        for (std::size_t first = 0U; first < 2U; ++first) {
            for (std::size_t second = 0U; second < 2U; ++second) {
                QComplex inner{};
                for (std::size_t row = 0U; row < 2U; ++row) {
                    inner += matrix(row, first).conjugate() * matrix(row, second);
                }
                const QComplex expected = first == second
                    ? QComplex{1.0, 0.0}
                    : QComplex{};
                if (!almost_equal(inner, expected, config_.validation_tolerance)) {
                    throw QStateError("Tree tensor single-qubit matrix is not unitary");
                }
            }
        }
    }

    [[nodiscard]] std::size_t lca(std::size_t first, std::size_t second) const {
        std::size_t left = first;
        std::size_t right = second;
        while (nodes_[left].depth > nodes_[right].depth) {
            left = nodes_[left].parent;
        }
        while (nodes_[right].depth > nodes_[left].depth) {
            right = nodes_[right].parent;
        }
        while (left != right) {
            left = nodes_[left].parent;
            right = nodes_[right].parent;
        }
        return left;
    }

    [[nodiscard]] static bool contains(
        std::span<const std::size_t> values,
        std::size_t target) {
        return std::find(values.begin(), values.end(), target) != values.end();
    }

    static void append_unique(std::vector<std::size_t>& values, std::size_t value) {
        if (!contains(values, value)) {
            values.push_back(value);
        }
    }

    void apply_controlled(
        QubitId control,
        QubitId target,
        const QMatrix2& inactive,
        const QMatrix2& active) {
        validate_qubit(control);
        validate_qubit(target);
        if (control == target) {
            throw QStateError("Tree tensor controlled gate requires distinct qubits");
        }
        validate_unitary(inactive);
        validate_unitary(active);

        const std::size_t control_node = qubit_node_[control];
        const std::size_t target_node = qubit_node_[target];
        const std::size_t ancestor = lca(control_node, target_node);
        std::vector<std::size_t> path_edges;
        std::vector<std::size_t> affected;
        std::size_t cursor = control_node;
        while (cursor != ancestor) {
            path_edges.push_back(cursor);
            append_unique(affected, cursor);
            append_unique(affected, nodes_[cursor].parent);
            cursor = nodes_[cursor].parent;
        }
        cursor = target_node;
        while (cursor != ancestor) {
            path_edges.push_back(cursor);
            append_unique(affected, cursor);
            append_unique(affected, nodes_[cursor].parent);
            cursor = nodes_[cursor].parent;
        }
        if (path_edges.empty()) {
            throw QStateError("Tree tensor controlled path is empty");
        }

        std::size_t next_scalar_count = stats_.scalar_count;
        std::size_t next_max_bond = stats_.max_bond_dimension;
        for (const std::size_t child : path_edges) {
            const std::size_t old_dimension = nodes_[child].parent_dimension;
            if (old_dimension > config_.max_bond_dimension / 2U) {
                throw QStateError("Tree tensor controlled gate exceeds bond dimension cap");
            }
            next_max_bond = std::max(next_max_bond, old_dimension * 2U);
        }

        std::vector<PendingNode> pending;
        pending.reserve(affected.size());
        for (const std::size_t node_index : affected) {
            const Node& node = nodes_[node_index];
            PendingNode update;
            update.node = node_index;
            update.parent_dimension = contains(path_edges, node_index)
                ? node.parent_dimension * 2U
                : node.parent_dimension;
            std::size_t old_local = 0U;
            std::size_t new_local = 0U;
            if (node.leaf) {
                if (node_index != control_node && node_index != target_node) {
                    throw QStateError("Tree tensor controlled path contains an unexpected leaf");
                }
                const QMatrix2 project_zero{{
                    QComplex{1.0}, QComplex{}, QComplex{}, QComplex{},
                }};
                const QMatrix2 project_one{{
                    QComplex{}, QComplex{}, QComplex{}, QComplex{1.0},
                }};
                const QMatrix2 control_ops[2]{project_zero, project_one};
                const QMatrix2 target_ops[2]{inactive, active};
                const QMatrix2* operators =
                    node_index == control_node ? control_ops : target_ops;
                update.zero.assign(update.parent_dimension, QComplex{});
                update.one.assign(update.parent_dimension, QComplex{});
                for (std::size_t bond = 0U; bond < node.parent_dimension; ++bond) {
                    for (std::size_t branch = 0U; branch < 2U; ++branch) {
                        const std::size_t next_bond = bond * 2U + branch;
                        update.zero[next_bond] =
                            operators[branch](0U, 0U) * node.zero[bond] +
                            operators[branch](0U, 1U) * node.one[bond];
                        update.one[next_bond] =
                            operators[branch](1U, 0U) * node.zero[bond] +
                            operators[branch](1U, 1U) * node.one[bond];
                    }
                }
                old_local = node.zero.size() + node.one.size();
                new_local = update.zero.size() + update.one.size();
            } else {
                const bool parent_path = contains(path_edges, node_index);
                const bool left_path = contains(path_edges, node.left);
                const bool right_path = contains(path_edges, node.right);
                const std::size_t incident =
                    static_cast<std::size_t>(parent_path) +
                    static_cast<std::size_t>(left_path) +
                    static_cast<std::size_t>(right_path);
                if (incident != 2U) {
                    throw QStateError("Tree tensor controlled path topology is invalid");
                }
                const std::size_t old_left = nodes_[node.left].parent_dimension;
                const std::size_t old_right = nodes_[node.right].parent_dimension;
                const std::size_t new_left = left_path ? old_left * 2U : old_left;
                const std::size_t new_right = right_path ? old_right * 2U : old_right;
                const std::size_t new_size = checked_tensor_size(
                    new_left, new_right, update.parent_dimension);
                update.values.assign(new_size, QComplex{});
                for (std::size_t left = 0U; left < old_left; ++left) {
                    for (std::size_t right = 0U; right < old_right; ++right) {
                        for (std::size_t parent = 0U;
                             parent < node.parent_dimension;
                             ++parent) {
                            const QComplex old_value = node.values[
                                tensor_index(
                                    left,
                                    right,
                                    parent,
                                    old_right,
                                    node.parent_dimension)];
                            for (std::size_t branch = 0U; branch < 2U; ++branch) {
                                const std::size_t next_left =
                                    left_path ? left * 2U + branch : left;
                                const std::size_t next_right =
                                    right_path ? right * 2U + branch : right;
                                const std::size_t next_parent =
                                    parent_path ? parent * 2U + branch : parent;
                                update.values[tensor_index(
                                    next_left,
                                    next_right,
                                    next_parent,
                                    new_right,
                                    update.parent_dimension)] += old_value;
                            }
                        }
                    }
                }
                old_local = node.values.size();
                new_local = update.values.size();
            }

            if (new_local < old_local ||
                next_scalar_count >
                    std::numeric_limits<std::size_t>::max() - (new_local - old_local)) {
                throw QStateError("Tree tensor controlled scalar count overflowed");
            }
            next_scalar_count += new_local - old_local;
            if (next_scalar_count > config_.max_scalars) {
                throw QStateError("Tree tensor controlled gate exceeds scalar cap");
            }
            pending.push_back(std::move(update));
        }

        for (PendingNode& update : pending) {
            Node& node = nodes_[update.node];
            node.parent_dimension = update.parent_dimension;
            if (node.leaf) {
                node.zero = std::move(update.zero);
                node.one = std::move(update.one);
            } else {
                node.values = std::move(update.values);
            }
        }
        stats_.scalar_count = next_scalar_count;
        stats_.max_bond_dimension = next_max_bond;
        ++stats_.controlled_gate_count;
        stats_.max_controlled_path_edges = std::max(
            stats_.max_controlled_path_edges, path_edges.size());
        advance_generation();
    }

    void advance_generation() noexcept {
        ++stats_.generation;
    }

    void refresh_workspace(TreeTensorWorkspace& workspace_value) const {
        if (workspace_value.generation_ == stats_.generation &&
            workspace_value.vectors_.size() == nodes_.size()) {
            return;
        }
        workspace_value.vectors_.resize(nodes_.size());
        workspace_value.environments_.clear();
        workspace_value.environments_.resize(nodes_.size());
        for (std::size_t index = 0U; index < nodes_.size(); ++index) {
            workspace_value.vectors_[index].assign(
                nodes_[index].parent_dimension, QComplex{});
        }
        workspace_value.assignment_.assign(qubit_count_, static_cast<std::int8_t>(-1));
        workspace_value.generation_ = stats_.generation;
    }

    void ensure_environments(TreeTensorWorkspace& workspace_value) const {
        for (std::size_t index = 0U; index < nodes_.size(); ++index) {
            const std::size_t dimension = nodes_[index].parent_dimension;
            const std::size_t entries = checked_product(dimension, dimension);
            if (entries > config_.max_scalars) {
                throw QStateError("Tree tensor marginal workspace exceeds scalar cap");
            }
            if (workspace_value.environments_[index].size() != entries) {
                workspace_value.environments_[index].assign(entries, QComplex{});
            }
        }
    }

    [[nodiscard]] double checked_probability(double value) const {
        if (!std::isfinite(value)) {
            throw QStateError("Tree tensor probability is non-finite");
        }
        if (value < -config_.validation_tolerance ||
            value > 1.0 + config_.validation_tolerance) {
            throw QStateError("Tree tensor probability is outside [0, 1]");
        }
        return std::clamp(value, 0.0, 1.0);
    }
};

}  // namespace qubit

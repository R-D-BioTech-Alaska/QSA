#pragma once

#include "qubit/qttn.hpp"

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

struct TreeTensorMarginalCertificate {
    bool eligible{false};
    std::string reason{};
    TreeTensorStats predicted{};
    std::size_t marginal_workspace_scalars{0U};
};

class TreeTensorMarginalPlan {
public:
    TreeTensorMarginalPlan(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        TreeTensorConfig config = {})
        : config_(config),
          certificate_(certify(qubit_count, operations, config)),
          state_(build_state(qubit_count, operations, config, certificate_)) {
        const TreeTensorStats actual = state_.stats();
        const TreeTensorStats& predicted = certificate_.predicted;
        if (actual.qubit_count != predicted.qubit_count ||
            actual.node_count != predicted.node_count ||
            actual.scalar_count != predicted.scalar_count ||
            actual.max_bond_dimension != predicted.max_bond_dimension ||
            actual.controlled_gate_count != predicted.controlled_gate_count ||
            actual.max_controlled_path_edges != predicted.max_controlled_path_edges ||
            actual.generation != predicted.generation) {
            throw QStateError("Tree tensor marginal resource certificate disagrees with executor");
        }
    }

    [[nodiscard]] static TreeTensorMarginalCertificate certify(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        TreeTensorConfig config = {}) noexcept {
        try {
            return certify_checked(qubit_count, operations, config);
        } catch (const QStateError& error) {
            TreeTensorMarginalCertificate result;
            result.reason = error.what();
            return result;
        } catch (...) {
            TreeTensorMarginalCertificate result;
            result.reason = "Tree tensor marginal certification failed unexpectedly";
            return result;
        }
    }

    [[nodiscard]] const TreeTensorMarginalCertificate& certificate() const noexcept {
        return certificate_;
    }

    [[nodiscard]] const TreeTensorState& state() const noexcept { return state_; }

    [[nodiscard]] TreeTensorWorkspace workspace() const {
        return state_.workspace();
    }

    [[nodiscard]] double marginal_probability(
        std::span<const QubitId> selected_qubits,
        std::span<const std::uint8_t> selected_bits,
        TreeTensorWorkspace& workspace_value) const {
        return state_.marginal_probability(
            selected_qubits, selected_bits, workspace_value);
    }

    [[nodiscard]] double marginal_probability(
        std::span<const QubitId> selected_qubits,
        std::span<const std::uint8_t> selected_bits) const {
        return state_.marginal_probability(selected_qubits, selected_bits);
    }

    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        return state_.estimated_bytes() +
               sizeof(*this) - sizeof(TreeTensorState) +
               certificate_.reason.capacity();
    }

private:
    struct Node {
        std::size_t parent{no_node()};
        std::size_t left{no_node()};
        std::size_t right{no_node()};
        std::size_t depth{0U};
        std::size_t parent_dimension{1U};
        bool leaf{false};
    };

    struct Model {
        std::vector<Node> nodes{};
        std::vector<std::size_t> qubit_node{};
        std::size_t root{no_node()};
        TreeTensorStats stats{};
    };

    TreeTensorConfig config_{};
    TreeTensorMarginalCertificate certificate_{};
    TreeTensorState state_;

    [[nodiscard]] static constexpr std::size_t no_node() noexcept {
        return std::numeric_limits<std::size_t>::max();
    }

    [[nodiscard]] static bool finite(const QComplex& value) noexcept {
        return std::isfinite(value.re) && std::isfinite(value.im);
    }

    [[nodiscard]] static std::size_t checked_product(
        std::size_t first,
        std::size_t second,
        const char* message) {
        if (first != 0U && second > std::numeric_limits<std::size_t>::max() / first) {
            throw QStateError(message);
        }
        return first * second;
    }

    [[nodiscard]] static std::size_t checked_tensor_size(
        std::size_t left,
        std::size_t right,
        std::size_t parent) {
        return checked_product(
            checked_product(left, right, "Tree tensor marginal tensor size overflowed"),
            parent,
            "Tree tensor marginal tensor size overflowed");
    }

    [[nodiscard]] static std::size_t build_topology(
        Model& model,
        std::size_t begin,
        std::size_t end,
        std::size_t depth) {
        if (end - begin == 1U) {
            Node node;
            node.depth = depth;
            node.leaf = true;
            const std::size_t index = model.nodes.size();
            model.nodes.push_back(node);
            model.qubit_node[begin] = index;
            return index;
        }
        const std::size_t middle = begin + (end - begin) / 2U;
        const std::size_t left = build_topology(model, begin, middle, depth + 1U);
        const std::size_t right = build_topology(model, middle, end, depth + 1U);
        Node node;
        node.depth = depth;
        node.left = left;
        node.right = right;
        const std::size_t index = model.nodes.size();
        model.nodes.push_back(node);
        model.nodes[left].parent = index;
        model.nodes[right].parent = index;
        return index;
    }

    [[nodiscard]] static Model initial_model(
        std::size_t qubit_count,
        const TreeTensorConfig& config) {
        if (qubit_count == 0U) {
            throw QStateError("Tree tensor marginal requires at least one qubit");
        }
        if (qubit_count - 1U >
            static_cast<std::size_t>(std::numeric_limits<QubitId>::max())) {
            throw QStateError("Tree tensor marginal qubit count exceeds QubitId range");
        }
        if (config.max_bond_dimension == 0U || config.max_scalars == 0U) {
            throw QStateError("Tree tensor marginal configuration contains a zero resource cap");
        }
        if (!std::isfinite(config.validation_tolerance) ||
            config.validation_tolerance <= 0.0) {
            throw QStateError("Tree tensor marginal validation tolerance must be positive and finite");
        }
        if (qubit_count > std::numeric_limits<std::size_t>::max() / 2U + 1U) {
            throw QStateError("Tree tensor marginal topology size overflowed");
        }

        Model model;
        model.qubit_node.resize(qubit_count);
        model.nodes.reserve(2U * qubit_count - 1U);
        model.root = build_topology(model, 0U, qubit_count, 0U);
        model.stats.qubit_count = qubit_count;
        model.stats.node_count = model.nodes.size();
        if (qubit_count > std::numeric_limits<std::size_t>::max() / 3U) {
            throw QStateError("Tree tensor marginal initial scalar count overflowed");
        }
        model.stats.scalar_count = 3U * qubit_count - 1U;
        model.stats.max_bond_dimension = 1U;
        if (model.stats.scalar_count > config.max_scalars) {
            throw QStateError("Tree tensor marginal zero state exceeds scalar cap");
        }
        return model;
    }

    [[nodiscard]] static bool single_matrix(
        const Operation& operation,
        QMatrix2& matrix) {
        switch (operation.code) {
            case OperationCode::X:
                matrix = gates::x();
                return true;
            case OperationCode::Y:
                matrix = gates::y();
                return true;
            case OperationCode::Z:
                matrix = gates::z();
                return true;
            case OperationCode::H:
                matrix = gates::h();
                return true;
            case OperationCode::S:
                matrix = gates::s();
                return true;
            case OperationCode::Sdg:
                matrix = gates::sdg();
                return true;
            case OperationCode::T:
                matrix = gates::t();
                return true;
            case OperationCode::Tdg:
                matrix = gates::tdg();
                return true;
            case OperationCode::Rx:
                matrix = gates::rx(operation.parameter);
                return true;
            case OperationCode::Ry:
                matrix = gates::ry(operation.parameter);
                return true;
            case OperationCode::Rz:
                matrix = gates::rz(operation.parameter);
                return true;
            default:
                return false;
        }
    }

    static void validate_single(
        QubitId qubit,
        const QMatrix2& matrix,
        std::size_t qubit_count,
        double tolerance) {
        if (static_cast<std::size_t>(qubit) >= qubit_count) {
            throw QStateError("Tree tensor marginal single-qubit operation is out of range");
        }
        for (const QComplex& value : matrix.values) {
            if (!finite(value)) {
                throw QStateError("Tree tensor marginal single-qubit matrix is non-finite");
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
                if (!almost_equal(inner, expected, tolerance)) {
                    throw QStateError("Tree tensor marginal single-qubit matrix is not unitary");
                }
            }
        }
    }

    [[nodiscard]] static std::size_t lca(
        const Model& model,
        std::size_t first,
        std::size_t second) {
        std::size_t left = first;
        std::size_t right = second;
        while (model.nodes[left].depth > model.nodes[right].depth) {
            left = model.nodes[left].parent;
        }
        while (model.nodes[right].depth > model.nodes[left].depth) {
            right = model.nodes[right].parent;
        }
        while (left != right) {
            left = model.nodes[left].parent;
            right = model.nodes[right].parent;
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

    [[nodiscard]] static std::size_t old_local(
        const Model& model,
        std::size_t node_index) {
        const Node& node = model.nodes[node_index];
        if (node.leaf) {
            return checked_product(
                node.parent_dimension,
                2U,
                "Tree tensor marginal leaf scalar count overflowed");
        }
        return checked_tensor_size(
            model.nodes[node.left].parent_dimension,
            model.nodes[node.right].parent_dimension,
            node.parent_dimension);
    }

    [[nodiscard]] static std::size_t new_local(
        const Model& model,
        std::size_t node_index,
        std::span<const std::size_t> path_edges) {
        const Node& node = model.nodes[node_index];
        const bool parent_path = contains(path_edges, node_index);
        if (node.leaf) {
            if (!parent_path) {
                throw QStateError("Tree tensor marginal endpoint is not on its path");
            }
            return checked_product(
                checked_product(
                    node.parent_dimension,
                    2U,
                    "Tree tensor marginal leaf bond overflowed"),
                2U,
                "Tree tensor marginal leaf scalar count overflowed");
        }
        const bool left_path = contains(path_edges, node.left);
        const bool right_path = contains(path_edges, node.right);
        const std::size_t incident =
            static_cast<std::size_t>(parent_path) +
            static_cast<std::size_t>(left_path) +
            static_cast<std::size_t>(right_path);
        if (incident != 2U) {
            throw QStateError("Tree tensor marginal controlled path topology is invalid");
        }
        const std::size_t left_dimension = model.nodes[node.left].parent_dimension;
        const std::size_t right_dimension = model.nodes[node.right].parent_dimension;
        return checked_tensor_size(
            left_path
                ? checked_product(left_dimension, 2U, "Tree tensor marginal left bond overflowed")
                : left_dimension,
            right_path
                ? checked_product(right_dimension, 2U, "Tree tensor marginal right bond overflowed")
                : right_dimension,
            parent_path
                ? checked_product(node.parent_dimension, 2U, "Tree tensor marginal parent bond overflowed")
                : node.parent_dimension);
    }

    static void apply_controlled_model(
        Model& model,
        QubitId first,
        QubitId second,
        const TreeTensorConfig& config) {
        if (static_cast<std::size_t>(first) >= model.stats.qubit_count ||
            static_cast<std::size_t>(second) >= model.stats.qubit_count) {
            throw QStateError("Tree tensor marginal controlled qubit is out of range");
        }
        if (first == second) {
            throw QStateError("Tree tensor marginal controlled gate requires distinct qubits");
        }

        const std::size_t first_node = model.qubit_node[first];
        const std::size_t second_node = model.qubit_node[second];
        const std::size_t ancestor = lca(model, first_node, second_node);
        std::vector<std::size_t> path_edges;
        std::vector<std::size_t> affected;
        std::size_t cursor = first_node;
        while (cursor != ancestor) {
            path_edges.push_back(cursor);
            append_unique(affected, cursor);
            append_unique(affected, model.nodes[cursor].parent);
            cursor = model.nodes[cursor].parent;
        }
        cursor = second_node;
        while (cursor != ancestor) {
            path_edges.push_back(cursor);
            append_unique(affected, cursor);
            append_unique(affected, model.nodes[cursor].parent);
            cursor = model.nodes[cursor].parent;
        }
        if (path_edges.empty()) {
            throw QStateError("Tree tensor marginal controlled path is empty");
        }

        std::size_t next_scalars = model.stats.scalar_count;
        std::size_t next_max_bond = model.stats.max_bond_dimension;
        for (const std::size_t child : path_edges) {
            const std::size_t dimension = model.nodes[child].parent_dimension;
            if (dimension > config.max_bond_dimension / 2U) {
                throw QStateError("Tree tensor marginal controlled gate exceeds bond cap");
            }
            next_max_bond = std::max(next_max_bond, dimension * 2U);
        }
        for (const std::size_t node_index : affected) {
            const std::size_t old_count = old_local(model, node_index);
            const std::size_t new_count = new_local(model, node_index, path_edges);
            if (new_count < old_count) {
                throw QStateError("Tree tensor marginal scalar growth is invalid");
            }
            const std::size_t delta = new_count - old_count;
            if (delta > config.max_scalars || next_scalars > config.max_scalars - delta) {
                throw QStateError("Tree tensor marginal controlled gate exceeds scalar cap");
            }
            next_scalars += delta;
        }

        for (const std::size_t child : path_edges) {
            model.nodes[child].parent_dimension *= 2U;
        }
        model.stats.scalar_count = next_scalars;
        model.stats.max_bond_dimension = next_max_bond;
        ++model.stats.controlled_gate_count;
        model.stats.max_controlled_path_edges = std::max(
            model.stats.max_controlled_path_edges,
            path_edges.size());
        ++model.stats.generation;
    }

    [[nodiscard]] static std::size_t marginal_workspace_scalars(
        const Model& model,
        const TreeTensorConfig& config) {
        std::size_t total = 0U;
        for (const Node& node : model.nodes) {
            const std::size_t entries = checked_product(
                node.parent_dimension,
                node.parent_dimension,
                "Tree tensor marginal workspace size overflowed");
            if (entries > config.max_scalars || total > config.max_scalars - entries) {
                throw QStateError("Tree tensor marginal workspace exceeds scalar cap");
            }
            total += entries;
        }
        return total;
    }

    [[nodiscard]] static TreeTensorMarginalCertificate certify_checked(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        const TreeTensorConfig& config) {
        Model model = initial_model(qubit_count, config);
        for (const Operation& operation : operations) {
            QMatrix2 matrix{};
            if (single_matrix(operation, matrix)) {
                validate_single(
                    operation.first,
                    matrix,
                    qubit_count,
                    config.validation_tolerance);
                ++model.stats.generation;
                continue;
            }
            switch (operation.code) {
                case OperationCode::Cnot:
                case OperationCode::Cz:
                    apply_controlled_model(model, operation.first, operation.second, config);
                    break;
                case OperationCode::Swap:
                    throw QStateError(
                        "Tree tensor marginal currently supports only CNOT/CZ two-qubit operations");
                case OperationCode::BitFlipTrajectory:
                case OperationCode::PhaseFlipTrajectory:
                case OperationCode::DepolarizingTrajectory:
                case OperationCode::AmplitudeDampingTrajectory:
                    throw QStateError("Tree tensor marginal does not support trajectory operations");
                default:
                    throw QStateError("Tree tensor marginal operation code is invalid");
            }
        }

        TreeTensorMarginalCertificate result;
        result.eligible = true;
        result.predicted = model.stats;
        result.marginal_workspace_scalars =
            marginal_workspace_scalars(model, config);
        return result;
    }

    [[nodiscard]] static TreeTensorState build_state(
        std::size_t qubit_count,
        std::span<const Operation> operations,
        const TreeTensorConfig& config,
        const TreeTensorMarginalCertificate& certificate) {
        if (!certificate.eligible) {
            throw QStateError(
                certificate.reason.empty()
                    ? "Tree tensor marginal certificate rejected the circuit"
                    : certificate.reason);
        }
        return TreeTensorState::from_operations(qubit_count, operations, config);
    }
};

}  // namespace qubit

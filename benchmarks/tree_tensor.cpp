#include "qubit/qttn.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::QComplex;
using qubit::QMatrix2;
using qubit::QubitId;
using qubit::TreeTensorConfig;
using qubit::TreeTensorState;
using qubit::TreeTensorWorkspace;

#if defined(_MSC_VER)
#define QSA_TTN_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define QSA_TTN_NOINLINE __attribute__((noinline))
#else
#define QSA_TTN_NOINLINE
#endif

volatile double ttn_sink = 0.0;

[[nodiscard]] QMatrix2 h_gate() {
    const double scale = 1.0 / std::sqrt(2.0);
    return {{
        QComplex{scale}, QComplex{scale},
        QComplex{scale}, QComplex{-scale},
    }};
}

[[nodiscard]] QMatrix2 ry_gate(double theta) {
    const double half = theta * 0.5;
    return {{
        QComplex{std::cos(half)}, QComplex{-std::sin(half)},
        QComplex{std::sin(half)}, QComplex{std::cos(half)},
    }};
}

class DenseControl {
public:
    explicit DenseControl(std::size_t qubits)
        : amplitudes_(std::size_t{1U} << qubits) {
        amplitudes_[0U] = {1.0, 0.0};
    }

    void apply_single(std::size_t qubit, const QMatrix2& matrix) {
        const std::size_t mask = std::size_t{1U} << qubit;
        for (std::size_t base = 0U; base < amplitudes_.size(); ++base) {
            if ((base & mask) != 0U) {
                continue;
            }
            const std::size_t one = base | mask;
            const QComplex zero_value = amplitudes_[base];
            const QComplex one_value = amplitudes_[one];
            amplitudes_[base] =
                matrix(0U, 0U) * zero_value + matrix(0U, 1U) * one_value;
            amplitudes_[one] =
                matrix(1U, 0U) * zero_value + matrix(1U, 1U) * one_value;
        }
    }

    void apply_cnot(std::size_t control, std::size_t target) {
        const std::size_t control_mask = std::size_t{1U} << control;
        const std::size_t target_mask = std::size_t{1U} << target;
        std::vector<QComplex> next(amplitudes_.size());
        for (std::size_t index = 0U; index < amplitudes_.size(); ++index) {
            const std::size_t destination = (index & control_mask) != 0U
                ? index ^ target_mask
                : index;
            next[destination] += amplitudes_[index];
        }
        amplitudes_ = std::move(next);
    }

    void apply_cz(std::size_t first, std::size_t second) {
        const std::size_t first_mask = std::size_t{1U} << first;
        const std::size_t second_mask = std::size_t{1U} << second;
        for (std::size_t index = 0U; index < amplitudes_.size(); ++index) {
            if ((index & first_mask) != 0U && (index & second_mask) != 0U) {
                amplitudes_[index] *= -1.0;
            }
        }
    }

    [[nodiscard]] const std::vector<QComplex>& amplitudes() const noexcept {
        return amplitudes_;
    }

    [[nodiscard]] double marginal(
        std::span<const QubitId> selected,
        std::span<const std::uint8_t> bits) const {
        double result = 0.0;
        for (std::size_t index = 0U; index < amplitudes_.size(); ++index) {
            bool match = true;
            for (std::size_t position = 0U; position < selected.size(); ++position) {
                if (((index >> selected[position]) & 1U) != bits[position]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                result += amplitudes_[index].norm2();
            }
        }
        return result;
    }

private:
    std::vector<QComplex> amplitudes_{};
};

void apply_block(TreeTensorState& state, std::size_t base, double theta) {
    state.apply_unitary(static_cast<QubitId>(base), h_gate());
    state.apply_unitary(static_cast<QubitId>(base + 1U), ry_gate(theta));
    state.apply_cnot(
        static_cast<QubitId>(base),
        static_cast<QubitId>(base + 2U));
    state.apply_cnot(
        static_cast<QubitId>(base + 1U),
        static_cast<QubitId>(base + 3U));
    state.apply_cz(
        static_cast<QubitId>(base),
        static_cast<QubitId>(base + 3U));
}

void apply_block(DenseControl& state, std::size_t base, double theta) {
    state.apply_single(base, h_gate());
    state.apply_single(base + 1U, ry_gate(theta));
    state.apply_cnot(base, base + 2U);
    state.apply_cnot(base + 1U, base + 3U);
    state.apply_cz(base, base + 3U);
}

void apply_blocks(TreeTensorState& state, std::size_t qubits) {
    for (std::size_t base = 0U; base < qubits; base += 4U) {
        const double theta = 0.31 + 0.00001 * static_cast<double>(base / 4U);
        apply_block(state, base, theta);
    }
}

void apply_blocks(DenseControl& state, std::size_t qubits) {
    for (std::size_t base = 0U; base < qubits; base += 4U) {
        const double theta = 0.31 + 0.00001 * static_cast<double>(base / 4U);
        apply_block(state, base, theta);
    }
}

template <class Function>
[[nodiscard]] double median_ms(
    Function&& function,
    std::size_t repeats,
    std::size_t iterations = 1U) {
    std::vector<double> samples;
    samples.reserve(repeats);
    for (std::size_t repeat = 0U; repeat < repeats; ++repeat) {
        const auto start = Clock::now();
        for (std::size_t iteration = 0U; iteration < iterations; ++iteration) {
            function();
        }
        const auto stop = Clock::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count() /
            static_cast<double>(iterations));
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

[[nodiscard]] double max_error(
    std::span<const QComplex> first,
    std::span<const QComplex> second) {
    double error = 0.0;
    for (std::size_t index = 0U; index < first.size(); ++index) {
        error = std::max(error, (first[index] - second[index]).magnitude());
    }
    return error;
}

QSA_TTN_NOINLINE double run_tree_marginal(
    const TreeTensorState& state,
    std::span<const QubitId> selected,
    std::span<const std::uint8_t> bits,
    TreeTensorWorkspace& workspace) {
    const double result = state.marginal_probability(selected, bits, workspace);
    ttn_sink = result;
    return result;
}

void matched_evidence() {
    constexpr std::size_t qubits = 16U;
    TreeTensorConfig config;
    config.max_bond_dimension = 8U;
    config.max_scalars = 32'768U;
    config.max_materialize_qubits = qubits;

    std::optional<TreeTensorState> tree;
    const double tree_setup_ms = median_ms([&] {
        tree.emplace(qubits, config);
        apply_blocks(*tree, qubits);
    }, 7U);
    std::optional<DenseControl> dense;
    const double dense_setup_ms = median_ms([&] {
        dense.emplace(qubits);
        apply_blocks(*dense, qubits);
    }, 7U);

    const std::vector<QComplex> materialized = tree->materialize();
    const double amplitude_error = max_error(materialized, dense->amplitudes());
    const std::array<QubitId, 4> selected{{0U, 1U, 2U, 3U}};
    const std::array<std::uint8_t, 4> bits{{1U, 1U, 1U, 1U}};
    auto workspace = tree->workspace();
    double tree_marginal = 0.0;
    const double tree_query_ms = median_ms([&] {
        tree_marginal = run_tree_marginal(*tree, selected, bits, workspace);
    }, 11U, 100U);
    const double dense_marginal = dense->marginal(selected, bits);
    const double marginal_error = std::abs(tree_marginal - dense_marginal);
    if (amplitude_error > 5e-10 || marginal_error > 5e-10 ||
        std::abs(tree->norm2() - 1.0) > 5e-10) {
        throw std::runtime_error("matched tree tensor evidence failed");
    }

    std::cout << "ttn_matched_qubits=" << qubits << '\n';
    std::cout << "ttn_matched_controlled_gates=" << tree->stats().controlled_gate_count << '\n';
    std::cout << "ttn_matched_max_path_edges=" << tree->stats().max_controlled_path_edges << '\n';
    std::cout << "ttn_matched_max_bond=" << tree->stats().max_bond_dimension << '\n';
    std::cout << "ttn_matched_scalars=" << tree->stats().scalar_count << '\n';
    std::cout << "ttn_matched_setup_ms=" << tree_setup_ms << '\n';
    std::cout << "ttn_matched_dense_setup_ms=" << dense_setup_ms << '\n';
    std::cout << "ttn_matched_query_ms=" << tree_query_ms << '\n';
    std::cout << "ttn_matched_plan_bytes=" << tree->estimated_bytes() << '\n';
    std::cout << "ttn_matched_workspace_bytes=" << workspace.estimated_bytes() << '\n';
    std::cout << "ttn_matched_amplitude_error=" << amplitude_error << '\n';
    std::cout << "ttn_matched_marginal_error=" << marginal_error << '\n';
}

void capability_evidence() {
    constexpr std::size_t qubits = 4096U;
    constexpr std::size_t blocks = qubits / 4U;
    TreeTensorConfig config;
    config.max_bond_dimension = 8U;
    config.max_scalars = 1U << 20U;
    config.max_materialize_qubits = 20U;

    std::optional<TreeTensorState> tree;
    const double setup_ms = median_ms([&] {
        tree.emplace(qubits, config);
        apply_blocks(*tree, qubits);
    }, 3U);
    std::string reason;
    if (!tree->validate(&reason) || tree->stats().max_bond_dimension != 8U ||
        tree->stats().controlled_gate_count != 3U * blocks) {
        throw std::runtime_error("large tree tensor validation failed: " + reason);
    }

    DenseControl block(4U);
    apply_block(block, 0U, 0.31);
    const std::array<QubitId, 4> selected{{0U, 1U, 2U, 3U}};
    const std::array<std::uint8_t, 4> bits{{1U, 1U, 1U, 1U}};
    const double expected = block.marginal(selected, bits);
    auto workspace = tree->workspace();
    double actual = 0.0;
    const double query_ms = median_ms([&] {
        actual = run_tree_marginal(*tree, selected, bits, workspace);
    }, 11U, 10U);
    const double error = std::abs(actual - expected);
    if (error > 5e-10 || std::abs(tree->norm2(workspace) - 1.0) > 5e-9) {
        throw std::runtime_error("large tree tensor branching control mismatch");
    }

    std::cout << "ttn_capability_qubits=" << qubits << '\n';
    std::cout << "ttn_capability_blocks=" << blocks << '\n';
    std::cout << "ttn_capability_controlled_gates=" << tree->stats().controlled_gate_count << '\n';
    std::cout << "ttn_capability_max_path_edges=" << tree->stats().max_controlled_path_edges << '\n';
    std::cout << "ttn_capability_max_bond=" << tree->stats().max_bond_dimension << '\n';
    std::cout << "ttn_capability_scalars=" << tree->stats().scalar_count << '\n';
    std::cout << "ttn_capability_setup_ms=" << setup_ms << '\n';
    std::cout << "ttn_capability_query_ms=" << query_ms << '\n';
    std::cout << "ttn_capability_plan_bytes=" << tree->estimated_bytes() << '\n';
    std::cout << "ttn_capability_workspace_bytes=" << workspace.estimated_bytes() << '\n';
    std::cout << "ttn_capability_selected_probability=" << actual << '\n';
    std::cout << "ttn_capability_control_probability=" << expected << '\n';
    std::cout << "ttn_capability_error=" << error << '\n';
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);
    matched_evidence();
    capability_evidence();
    return 0;
}

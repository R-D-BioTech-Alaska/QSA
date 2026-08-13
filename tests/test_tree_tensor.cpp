#include "qubit/qttn.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using qubit::QComplex;
using qubit::QMatrix2;
using qubit::QStateError;
using qubit::QubitId;
using qubit::TreeTensorConfig;
using qubit::TreeTensorState;
using qubit::TreeTensorWorkspace;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(QComplex actual, QComplex expected, const std::string& message) {
    require(qubit::almost_equal(actual, expected, 5e-10), message);
}

void require_close(double actual, double expected, const std::string& message) {
    const double scale = 1.0 + std::max(std::abs(actual), std::abs(expected));
    require(std::abs(actual - expected) <= 5e-10 * scale, message);
}

template <class Function>
void require_reject(Function&& function, const std::string& message) {
    bool rejected = false;
    try {
        function();
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, message);
}

[[nodiscard]] QMatrix2 identity() {
    return {{QComplex{1.0}, QComplex{}, QComplex{}, QComplex{1.0}}};
}

[[nodiscard]] QMatrix2 x_gate() {
    return {{QComplex{}, QComplex{1.0}, QComplex{1.0}, QComplex{}}};
}

[[nodiscard]] QMatrix2 z_gate() {
    return {{QComplex{1.0}, QComplex{}, QComplex{}, QComplex{-1.0}}};
}

[[nodiscard]] QMatrix2 h_gate() {
    const double scale = 1.0 / std::sqrt(2.0);
    return {{
        QComplex{scale}, QComplex{scale},
        QComplex{scale}, QComplex{-scale},
    }};
}

[[nodiscard]] QMatrix2 ry_gate(double theta) {
    const double half = theta * 0.5;
    const double c = std::cos(half);
    const double s = std::sin(half);
    return {{QComplex{c}, QComplex{-s}, QComplex{s}, QComplex{c}}};
}

[[nodiscard]] QMatrix2 rz_gate(double theta) {
    const double half = theta * 0.5;
    return {{
        QComplex::from_polar(1.0, -half), QComplex{},
        QComplex{}, QComplex::from_polar(1.0, half),
    }};
}

class DenseControl {
public:
    explicit DenseControl(std::size_t qubits)
        : qubits_(qubits), amplitudes_(std::size_t{1U} << qubits) {
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
    std::size_t qubits_{0U};
    std::vector<QComplex> amplitudes_{};
};

void compare_state(const TreeTensorState& tree, const DenseControl& dense) {
    const std::vector<QComplex> materialized = tree.materialize();
    const std::vector<QComplex>& expected = dense.amplitudes();
    require(materialized.size() == expected.size(), "tree materialized size changed");
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        require_close(materialized[index], expected[index],
                      "tree amplitude differs from dense control");
    }
    require_close(tree.norm2(), 1.0, "tree norm changed");
    std::string reason;
    require(tree.validate(&reason), "tree validation failed: " + reason);
}

void zero_and_single_gates() {
    TreeTensorState tree(4U);
    DenseControl dense(4U);
    compare_state(tree, dense);
    require_close(tree.amplitude(0U), {1.0, 0.0}, "tree zero amplitude changed");
    require_close(tree.amplitude(3U), {}, "tree nonzero basis amplitude changed");

    const std::array<QMatrix2, 5> gates{{
        h_gate(), ry_gate(0.37), rz_gate(-0.61), x_gate(), z_gate(),
    }};
    for (std::size_t index = 0U; index < gates.size(); ++index) {
        const QubitId qubit = static_cast<QubitId>(index % 4U);
        tree.apply_unitary(qubit, gates[index]);
        dense.apply_single(qubit, gates[index]);
    }
    compare_state(tree, dense);
}

void branching_controlled_paths() {
    TreeTensorConfig config;
    config.max_bond_dimension = 64U;
    config.max_scalars = 2'000'000U;
    TreeTensorState tree(6U, config);
    DenseControl dense(6U);

    const QMatrix2 first = h_gate();
    const QMatrix2 second = ry_gate(0.42);
    const QMatrix2 third = rz_gate(-0.27);
    tree.apply_unitary(0U, first);
    dense.apply_single(0U, first);
    tree.apply_unitary(1U, second);
    dense.apply_single(1U, second);
    tree.apply_unitary(5U, third);
    dense.apply_single(5U, third);

    tree.apply_cnot(0U, 4U);
    dense.apply_cnot(0U, 4U);
    tree.apply_cz(1U, 5U);
    dense.apply_cz(1U, 5U);
    tree.apply_cnot(4U, 2U);
    dense.apply_cnot(4U, 2U);
    tree.apply_cz(0U, 3U);
    dense.apply_cz(0U, 3U);
    tree.apply_cnot(5U, 1U);
    dense.apply_cnot(5U, 1U);

    compare_state(tree, dense);
    require(tree.stats().controlled_gate_count == 5U,
            "tree controlled gate count changed");
    require(tree.stats().max_controlled_path_edges >= 4U,
            "tree did not record nonlocal controlled paths");

    TreeTensorWorkspace workspace = tree.workspace();
    const std::array<QubitId, 2> selected{{0U, 5U}};
    for (std::uint8_t first_bit = 0U; first_bit < 2U; ++first_bit) {
        for (std::uint8_t second_bit = 0U; second_bit < 2U; ++second_bit) {
            const std::array<std::uint8_t, 2> bits{{first_bit, second_bit}};
            require_close(
                tree.marginal_probability(selected, bits, workspace),
                dense.marginal(selected, bits),
                "tree selected marginal differs from dense control");
        }
    }
}

void randomized_circuits() {
    std::mt19937_64 generator(0x771e5a11ULL);
    for (std::size_t trial = 0U; trial < 24U; ++trial) {
        const std::size_t qubits = 3U + static_cast<std::size_t>(generator() % 5U);
        TreeTensorConfig config;
        config.max_bond_dimension = 64U;
        config.max_scalars = 4'000'000U;
        TreeTensorState tree(qubits, config);
        DenseControl dense(qubits);

        std::size_t controlled = 0U;
        for (std::size_t step = 0U; step < 10U; ++step) {
            const bool use_controlled = controlled < 4U && (generator() % 3U == 0U);
            if (!use_controlled) {
                const QubitId qubit = static_cast<QubitId>(generator() % qubits);
                QMatrix2 matrix;
                switch (generator() % 4U) {
                    case 0U:
                        matrix = h_gate();
                        break;
                    case 1U:
                        matrix = x_gate();
                        break;
                    case 2U:
                        matrix = ry_gate(
                            0.1 + 0.07 * static_cast<double>((trial + step) % 9U));
                        break;
                    default:
                        matrix = rz_gate(
                            -0.2 + 0.05 * static_cast<double>((trial + 2U * step) % 11U));
                        break;
                }
                tree.apply_unitary(qubit, matrix);
                dense.apply_single(qubit, matrix);
                continue;
            }

            QubitId first = static_cast<QubitId>(generator() % qubits);
            QubitId second = static_cast<QubitId>(generator() % qubits);
            while (second == first) {
                second = static_cast<QubitId>(generator() % qubits);
            }
            if ((generator() & 1U) == 0U) {
                tree.apply_cnot(first, second);
                dense.apply_cnot(first, second);
            } else {
                tree.apply_cz(first, second);
                dense.apply_cz(first, second);
            }
            ++controlled;
        }

        compare_state(tree, dense);
        const std::array<QubitId, 2> selected{{
            static_cast<QubitId>(generator() % qubits),
            static_cast<QubitId>(generator() % qubits),
        }};
        if (selected[0] == selected[1]) {
            continue;
        }
        const std::array<std::uint8_t, 2> bits{{
            static_cast<std::uint8_t>(generator() & 1U),
            static_cast<std::uint8_t>(generator() & 1U),
        }};
        require_close(
            tree.marginal_probability(selected, bits),
            dense.marginal(selected, bits),
            "random tree marginal differs from dense control");
    }
}

void transactional_caps() {
    TreeTensorConfig bond_config;
    bond_config.max_bond_dimension = 2U;
    bond_config.max_scalars = 128U;
    TreeTensorState bond_limited(2U, bond_config);
    bond_limited.apply_unitary(0U, h_gate());
    bond_limited.apply_cnot(0U, 1U);
    const std::vector<QComplex> before_bond = bond_limited.materialize();
    const auto bond_stats = bond_limited.stats();
    require_reject([&] {
        bond_limited.apply_cnot(0U, 1U);
    }, "tree bond cap did not reject repeated controlled gate");
    require(bond_limited.stats().scalar_count == bond_stats.scalar_count &&
            bond_limited.stats().generation == bond_stats.generation,
            "failed tree bond update changed statistics");
    const std::vector<QComplex> after_bond = bond_limited.materialize();
    require(before_bond == after_bond,
            "failed tree bond update changed state");

    TreeTensorConfig scalar_config;
    scalar_config.max_bond_dimension = 8U;
    scalar_config.max_scalars = 20U;
    TreeTensorState scalar_limited(4U, scalar_config);
    const std::vector<QComplex> before_scalar = scalar_limited.materialize();
    const auto scalar_stats = scalar_limited.stats();
    require_reject([&] {
        scalar_limited.apply_cnot(0U, 3U);
    }, "tree scalar cap did not reject nonlocal controlled gate");
    require(scalar_limited.stats().scalar_count == scalar_stats.scalar_count &&
            scalar_limited.stats().generation == scalar_stats.generation,
            "failed tree scalar update changed statistics");
    require(scalar_limited.materialize() == before_scalar,
            "failed tree scalar update changed state");

    TreeTensorConfig environment_config;
    environment_config.max_bond_dimension = 4U;
    environment_config.max_scalars = 32U;
    TreeTensorState environment_limited(2U, environment_config);
    environment_limited.apply_unitary(0U, h_gate());
    environment_limited.apply_cnot(0U, 1U);
    environment_limited.apply_cnot(0U, 1U);
    require(environment_limited.stats().scalar_count == 32U,
            "tree environment control did not reach expected scalar boundary");
    require_reject([&] {
        static_cast<void>(environment_limited.norm2());
    }, "tree marginal workspace cap was not enforced");
}

void workspace_and_validation_adversaries() {
    TreeTensorState single_only(4U);
    single_only.apply_unitary(0U, h_gate());
    TreeTensorWorkspace foreign = single_only.workspace();

    TreeTensorConfig config;
    config.max_bond_dimension = 8U;
    config.max_scalars = 1024U;
    TreeTensorState entangled(4U, config);
    entangled.apply_cnot(0U, 3U);
    const std::array<std::uint8_t, 4> zero_bits{{0U, 0U, 0U, 0U}};
    require_close(
        entangled.amplitude(zero_bits, foreign),
        entangled.amplitude(zero_bits),
        "foreign tree workspace did not refresh to current shapes");

    const QMatrix2 invalid{{
        QComplex{1.0}, QComplex{1.0}, QComplex{}, QComplex{1.0},
    }};
    const std::vector<QComplex> before = entangled.materialize();
    const auto stats = entangled.stats();
    require_reject([&] {
        entangled.apply_unitary(1U, invalid);
    }, "tree accepted non-unitary single matrix");
    require(entangled.materialize() == before &&
            entangled.stats().generation == stats.generation,
            "failed tree single gate changed state");

    require_reject([&] {
        entangled.apply_cnot(2U, 2U);
    }, "tree accepted same-qubit controlled gate");

    const std::array<std::uint8_t, 3> short_bits{{0U, 0U, 0U}};
    require_reject([&] {
        static_cast<void>(entangled.amplitude(short_bits));
    }, "tree accepted wrong amplitude bit count");
    const std::array<std::uint8_t, 4> invalid_bits{{0U, 0U, 2U, 0U}};
    require_reject([&] {
        static_cast<void>(entangled.amplitude(invalid_bits));
    }, "tree accepted invalid amplitude bit");

    const std::array<QubitId, 2> repeated{{1U, 1U}};
    const std::array<std::uint8_t, 2> repeated_bits{{0U, 1U}};
    require_reject([&] {
        static_cast<void>(entangled.marginal_probability(repeated, repeated_bits));
    }, "tree accepted repeated marginal qubit");

    TreeTensorConfig materialize_config;
    materialize_config.max_materialize_qubits = 3U;
    TreeTensorState materialize_limited(4U, materialize_config);
    require_reject([&] {
        static_cast<void>(materialize_limited.materialize());
    }, "tree materialization cap was not enforced");
}

}  // namespace

int main() {
    zero_and_single_gates();
    branching_controlled_paths();
    randomized_circuits();
    transactional_caps();
    workspace_and_validation_adversaries();
    return 0;
}

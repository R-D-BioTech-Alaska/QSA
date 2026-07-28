#include "qubit/qstate.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using qubit::ComponentKind;
using qubit::QComplex;
using qubit::QRegister;
using qubit::QStateError;
using qubit::QubitId;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

[[nodiscard]] QRegister dense_state(std::size_t qubits) {
    const std::size_t dimension = std::size_t{1} << qubits;
    const double scale = 1.0 / std::sqrt(static_cast<double>(dimension));
    std::vector<QComplex> amplitudes(dimension);
    for (std::size_t basis = 0; basis < dimension; ++basis) {
        amplitudes[basis] = QComplex::from_polar(
            scale,
            0.013 * static_cast<double>((basis * 17U) % 101U));
    }
    return QRegister::from_amplitudes(std::move(amplitudes));
}

}  // namespace

int main() {
    {
        QRegister cells(4);
        const auto view = cells.component_read_view(2);
        require(view.kind == ComponentKind::Cell, "cell view has the wrong kind");
        require(view.cell != nullptr, "cell view is missing its cell pointer");
        require(view.qubits.size() == 1U && view.qubits.front() == 2U,
                "cell view has the wrong qubit membership");
        require(view.dense.empty() && view.sparse.empty(),
                "cell view exposed amplitude storage");
        require(view.dimension == 2U, "cell view has the wrong dimension");
    }

    {
        QRegister sparse(8);
        sparse.apply_h(0);
        for (QubitId qubit = 1; qubit < 8; ++qubit) {
            sparse.apply_cnot(0, qubit);
        }
        const auto view = sparse.component_read_view(0);
        require(view.kind == ComponentKind::Sparse, "GHZ view is not sparse");
        require(view.sparse.size() == 2U, "GHZ view has the wrong support size");
        require(view.dense.empty() && view.cell == nullptr,
                "sparse view exposed the wrong storage");
        require(view.dimension == (std::uint64_t{1} << 8U),
                "sparse view has the wrong dimension");
        require(view.sparse.data() == sparse.component_read_view(7).sparse.data(),
                "views of the same component do not share storage");
    }

    {
        QRegister dense = dense_state(8);
        const auto view = dense.component_read_view(3);
        require(view.kind == ComponentKind::Dense, "dense view has the wrong kind");
        require(view.dense.size() == 256U, "dense view has the wrong size");
        require(view.sparse.empty() && view.cell == nullptr,
                "dense view exposed the wrong storage");
        const auto materialized = dense.materialize(12);
        require(view.dense.data() != nullptr, "dense view has no data pointer");
        for (std::size_t index = 0; index < materialized.size(); ++index) {
            require(qubit::almost_equal(view.dense[index], materialized[index], 2e-12),
                    "dense view differs from materialized amplitudes");
        }
    }

    {
        QRegister mixed(5);
        mixed.apply_h(0);
        mixed.apply_cnot(0, 1);
        const auto views = mixed.component_read_views();
        require(views.size() == mixed.component_count(),
                "component read-view count is wrong");
        std::vector<bool> seen(5, false);
        for (const auto& view : views) {
            for (QubitId qubit : view.qubits) {
                require(!seen[qubit], "component read views duplicate a qubit");
                seen[qubit] = true;
            }
        }
        for (bool value : seen) {
            require(value, "component read views omit a qubit");
        }
    }

    {
        QRegister state = dense_state(12);
        const auto expected = state.probabilities_one();
        std::vector<double> output(state.qubit_count());
        state.probabilities_one_into(output);
        require(output == expected, "preallocated probabilities differ");

        bool rejected = false;
        try {
            std::vector<double> short_output(state.qubit_count() - 1U);
            state.probabilities_one_into(short_output);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "probability output size mismatch was accepted");
    }

    {
        QRegister first(6);
        first.apply_h(0);
        first.apply_cnot(0, 1);
        first.apply_h(2);
        first.apply_cnot(2, 3);
        QRegister second = first;
        std::vector<int> output(first.qubit_count());
        first.measure_all_into(0x123456789ABCDEF0ULL, output);
        const auto expected = second.measure_all(0x123456789ABCDEF0ULL);
        require(output == expected, "preallocated sequential measurement differs");
        require(first.validate() && second.validate(),
                "sequential buffer measurement produced an invalid state");
    }

    std::cout << "read-view and output-buffer tests passed\n";
    return 0;
}

#include "pauli_fixture.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class Function>
void require_rejected(Function&& function, const char* message) {
    try {
        function();
    } catch (const qubit::QStateError&) {
        return;
    }
    throw std::runtime_error(message);
}

}  // namespace

int main() {
    {
        constexpr std::size_t qubit_count = 3U;
        qubit::QRegister state =
            qsa_pauli_support_test::make_tripair(qubit_count, 0U, 1U, 2U);
        const auto supports =
            qsa_pauli_support_test::tripair_support(0U, 1U, 2U);
        const auto words =
            qsa_pauli_support_test::tripair_words(qubit_count, 0U, 1U, 2U);
        const qubit::PauliSupportPlan compact(qubit_count, supports);
        const qubit::PauliObservablePlan full(qubit_count, words);
        const auto compact_values = compact.execute(state);
        const auto full_values = full.execute(state);
        require(compact.observable_count() == 14U,
                "Tripair support observable count is wrong");
        require(compact.term_count() == 21U,
                "Tripair support term count is wrong");
        require(
            qsa_pauli_support_test::maximum_error(compact_values, full_values)
                <= 2.0e-12,
            "compact Tripair observables differ from full Pauli words");
    }

    {
        constexpr std::size_t qubit_count = 10'000U;
        qubit::QRegister state(qubit_count);
        state.apply_h(0U);
        state.apply_x(static_cast<qubit::QubitId>(qubit_count - 1U));
        const std::vector<std::vector<qubit::PauliSupportTerm>> supports{
            {},
            {{0U, 'X'}, {static_cast<qubit::QubitId>(qubit_count - 1U), 'Z'}},
        };
        const qubit::PauliSupportPlan plan(qubit_count, supports);
        const auto values = plan.execute(state);
        require(values.size() == 2U, "factorized support result count is wrong");
        require(std::abs(values[0] - 1.0) <= 2.0e-12,
                "identity support is wrong");
        require(std::abs(values[1] + 1.0) <= 2.0e-12,
                "factorized 10,000-qubit support is wrong");
        require(state.component_count() == qubit_count,
                "compact support merged independent components");
    }

    require_rejected(
        [] {
            const std::vector<std::vector<qubit::PauliSupportTerm>> supports{
                {{0U, 'X'}, {0U, 'Z'}},
            };
            const qubit::PauliSupportPlan plan(2U, supports);
            (void)plan;
        },
        "duplicate Pauli support qubit was accepted");

    require_rejected(
        [] {
            const std::vector<std::vector<qubit::PauliSupportTerm>> supports{
                {{0U, 'A'}},
            };
            const qubit::PauliSupportPlan plan(2U, supports);
            (void)plan;
        },
        "invalid Pauli support axis was accepted");

    require_rejected(
        [] {
            const std::vector<std::vector<qubit::PauliSupportTerm>> supports{
                {{2U, 'Z'}},
            };
            const qubit::PauliSupportPlan plan(2U, supports);
            (void)plan;
        },
        "out-of-range Pauli support qubit was accepted");

    std::cout << "QSA compact Pauli support tests passed.\n";
    return 0;
}

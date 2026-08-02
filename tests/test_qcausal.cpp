#include "qubit/qcausal.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using qubit::BasisIndex;
using qubit::CausalState;
using qubit::PauliObservablePlan;
using qubit::QComplex;
using qubit::QRegister;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(double actual, double expected, const char* message) {
    if (std::abs(actual - expected) > 2e-11) {
        throw std::runtime_error(message);
    }
}

[[nodiscard]] QComplex coefficient_for(
    BasisIndex source,
    const std::string& word,
    BasisIndex& target) {
    QComplex coefficient{1.0, 0.0};
    target = source;
    for (std::size_t qubit = 0; qubit < word.size(); ++qubit) {
        const bool one = ((source >> qubit) & BasisIndex{1}) != 0U;
        switch (word[qubit]) {
            case 'I':
                break;
            case 'X':
                target ^= BasisIndex{1} << qubit;
                break;
            case 'Y':
                target ^= BasisIndex{1} << qubit;
                coefficient *= one ? QComplex{0.0, -1.0} : QComplex{0.0, 1.0};
                break;
            case 'Z':
                if (one) {
                    coefficient *= -1.0;
                }
                break;
            default:
                throw std::runtime_error("invalid test Pauli word");
        }
    }
    return coefficient;
}

[[nodiscard]] double dense_reference(
    const std::vector<QComplex>& amplitudes,
    const std::string& word) {
    QComplex expectation{};
    for (BasisIndex source = 0; source < amplitudes.size(); ++source) {
        BasisIndex target = 0;
        const QComplex coefficient = coefficient_for(source, word, target);
        expectation += amplitudes[static_cast<std::size_t>(target)].conjugate()
            * coefficient
            * amplitudes[static_cast<std::size_t>(source)];
    }
    require(std::abs(expectation.im) <= 2e-11,
            "dense reference expectation is unexpectedly complex");
    return expectation.re;
}

}  // namespace

int main() {
    {
        QRegister initial(3);
        initial.apply_h(0);
        initial.apply_cnot(0, 1);
        CausalState root(std::move(initial));
        CausalState branch = root.fork();

        require(root.shares_state_with(branch),
                "fresh causal branch does not share its immutable state");
        require(root.shared_owner_count() == 2L,
                "fresh causal branch has the wrong owner count");

        branch.mutate([](QRegister& state) { state.apply_x(2); });
        require(!root.shares_state_with(branch),
                "mutated causal branch did not detach");
        require_close(root.read().probability_one(2), 0.0,
                      "branch mutation changed the parent state");
        require_close(branch.read().probability_one(2), 1.0,
                      "branch mutation was not applied");

        root.adopt(std::move(branch));
        require(!branch.valid(), "adopted branch retained state ownership");
        require_close(root.read().probability_one(2), 1.0,
                      "selected branch was not adopted");
        require(root.unique(), "adopted state is not uniquely owned");
    }

    {
        CausalState root(64);
        const auto branches = root.fork_many(128);
        require(root.shared_owner_count() == 129L,
                "fork_many did not retain one shared immutable state");
        for (const CausalState& branch : branches) {
            require(root.shares_state_with(branch),
                    "fork_many produced an eager state copy");
        }
    }

    {
        QRegister bell(2);
        bell.apply_h(0);
        bell.apply_cnot(0, 1);
        const std::vector<std::string> words{"ZZ", "XX", "YY", "ZI", "IZ"};
        const PauliObservablePlan plan(2, words);
        const auto values = plan.execute(bell);
        require_close(values[0], 1.0, "Bell <ZZ> is wrong");
        require_close(values[1], 1.0, "Bell <XX> is wrong");
        require_close(values[2], -1.0, "Bell <YY> is wrong");
        require_close(values[3], 0.0, "Bell <ZI> is wrong");
        require_close(values[4], 0.0, "Bell <IZ> is wrong");
    }

    {
        QRegister structured(10'000);
        structured.apply_h(0);
        structured.apply_x(9'999);
        std::string word(10'000, 'I');
        word[0] = 'X';
        word[9'999] = 'Z';
        const std::vector<std::string> words{word};
        const PauliObservablePlan plan(10'000, words);
        const auto values = plan.execute(structured);
        require_close(values.front(), -1.0,
                      "factorized 10,000-qubit observable is wrong");
        require(structured.component_count() == 10'000U,
                "observable execution merged independent components");
    }

    {
        QRegister ghz(3);
        ghz.apply_h(0);
        ghz.apply_cnot(0, 1);
        ghz.apply_cnot(1, 2);
        const std::vector<std::string> words{
            "ZZI", "IZZ", "ZIZ", "XXX", "YYY"
        };
        const PauliObservablePlan plan(3, words);
        const auto values = plan.execute(ghz);
        require_close(values[0], 1.0, "GHZ <ZZI> is wrong");
        require_close(values[1], 1.0, "GHZ <IZZ> is wrong");
        require_close(values[2], 1.0, "GHZ <ZIZ> is wrong");
        require_close(values[3], 1.0, "GHZ <XXX> is wrong");
        require_close(values[4], 0.0, "GHZ <YYY> is wrong");
    }

    {
        std::vector<QComplex> amplitudes{
            {0.22, 0.11}, {-0.17, 0.08}, {0.31, -0.09}, {0.04, 0.27},
            {-0.13, -0.21}, {0.19, 0.06}, {0.07, -0.24}, {0.29, 0.14},
        };
        QRegister dense = QRegister::from_amplitudes(amplitudes);
        const std::vector<QComplex> normalized = dense.materialize(8);
        const std::vector<std::string> tripair_words{
            "XII", "YII", "ZII",
            "IXI", "IYI", "IZI",
            "IIX", "IIY", "IIZ",
            "ZZI", "IZZ", "ZIZ",
            "XXX", "YYY",
        };
        const PauliObservablePlan plan(3, tripair_words);
        const auto values = plan.execute(dense);
        require(values.size() == tripair_words.size(),
                "Tripair observable closure has the wrong size");
        for (std::size_t index = 0; index < tripair_words.size(); ++index) {
            require_close(values[index],
                          dense_reference(normalized, tripair_words[index]),
                          "component observable differs from dense reference");
        }
    }

    std::cout << "causal state-graph and Pauli observable tests passed\n";
    return 0;
}

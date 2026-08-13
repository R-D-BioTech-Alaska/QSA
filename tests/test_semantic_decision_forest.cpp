#include "qubit/qsemantic_forest.hpp"

#include <array>
#include <stdexcept>

static void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

int main() {
    qubit::ExactSemanticDecisionForest forest;
    const auto a = forest.atom("alpha");
    const auto a_again = forest.atom("alpha");
    const auto b = forest.atom("beta");

    const std::array same{a, a_again};
    require(forest.merge(same).unique(), "duplicate semantics did not collapse");
    require(forest.stats().nodes == 2U, "unexpected atom count");

    const std::array distinct{a, b};
    const auto ambiguity = forest.merge(distinct);
    require(ambiguity.ambiguous(), "distinct semantics were not ambiguous");

    const std::array children{ambiguity, a};
    const auto composite = forest.compose("IMPLIES", children, "brain.semantic.v1");
    require(composite.ambiguous(), "composite lost ambiguity");
    require(composite.witness_count() == 2U, "composite stored excess witnesses");

    const std::array left{a, b};
    const std::array right{b, a};
    const auto and_left = forest.compose("AND", left, {}, true);
    const auto and_right = forest.compose("AND", right, {}, true);
    require(and_left.witness(0U) == and_right.witness(0U), "commutative identity mismatch");
    return 0;
}

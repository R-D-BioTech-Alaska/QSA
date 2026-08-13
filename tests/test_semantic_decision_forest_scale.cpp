#include "qubit/qsemantic_forest.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

int main() {
    constexpr std::size_t width = 4096U;
    qubit::ExactSemanticForestConfig config;
    config.max_nodes = 10000U;
    config.max_children_per_node = width;
    qubit::ExactSemanticDecisionForest forest(config);
    std::vector<qubit::ExactSemanticDecision> choices;
    choices.reserve(width);
    for (std::size_t i = 0U; i < width; ++i) {
        const auto zero = forest.atom("s" + std::to_string(i) + "a");
        const auto one = forest.atom("s" + std::to_string(i) + "b");
        const std::array pair{zero, one};
        choices.push_back(forest.merge(pair));
    }
    const auto root = forest.compose("AND", choices, {}, true);
    return root.ambiguous() && root.witness_count() == 2U &&
        forest.stats().nodes == width * 2U + 2U ? 0 : 1;
}

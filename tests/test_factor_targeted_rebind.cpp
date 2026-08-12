#include "qubit/qfactor.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using qubit::ExactFactorGraph;
using qubit::FactorId;
using qubit::FactorSparseEntry;
using qubit::FactorVariableId;
using qubit::QComplex;
using qubit::QStateError;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(
    const std::vector<QComplex>& actual,
    const std::vector<QComplex>& expected,
    const std::string& message) {
    require(actual.size() == expected.size(), message + " size");
    for (std::size_t index = 0U; index < actual.size(); ++index) {
        require(qubit::almost_equal(actual[index], expected[index], 2e-12),
                message + " value");
    }
}

}  // namespace

int main() {
    ExactFactorGraph graph;
    const FactorVariableId a = graph.add_variable(2U);
    const FactorVariableId b = graph.add_variable(3U);
    const std::array<FactorVariableId, 2> scope{{a, b}};
    const std::array<QComplex, 6> initial{{
        QComplex{1.0}, QComplex{2.0}, QComplex{3.0},
        QComplex{4.0}, QComplex{5.0}, QComplex{6.0},
    }};
    const FactorId factor = graph.add_dense_factor(scope, initial);
    const std::array<FactorVariableId, 1> retained{{b}};
    auto plan = graph.compile(retained);
    auto workspace = plan.workspace();

    const std::array<FactorSparseEntry, 3> sparse{{
        FactorSparseEntry{0U, QComplex{2.0}},
        FactorSparseEntry{3U, QComplex{7.0}},
        FactorSparseEntry{5U, QComplex{11.0}},
    }};
    plan.rebind_sparse_factor(factor, sparse);
    graph.set_sparse_factor(factor, sparse);
    require_close(plan.evaluate(workspace), graph.marginal(retained),
                  "targeted sparse rebind");
    require(plan.rebind_count() == 1U &&
                plan.stats().source_dense_factors == 0U &&
                plan.stats().source_sparse_factors == 1U,
            "targeted sparse rebind stats are wrong");

    const std::array<QComplex, 6> dense{{
        QComplex{6.0}, QComplex{5.0}, QComplex{4.0},
        QComplex{3.0}, QComplex{2.0}, QComplex{1.0},
    }};
    plan.rebind_dense_factor(factor, dense);
    graph.set_dense_factor(factor, dense);
    require_close(plan.evaluate(workspace), graph.marginal(retained),
                  "targeted dense rebind");
    require(plan.rebind_count() == 2U &&
                plan.stats().source_dense_factors == 1U &&
                plan.stats().source_sparse_factors == 0U,
            "targeted dense rebind stats are wrong");

    const auto stable = plan.evaluate(workspace);
    const std::size_t stable_count = plan.rebind_count();

    bool rejected = false;
    const std::array<QComplex, 2> wrong_size{{QComplex{1.0}, QComplex{2.0}}};
    try {
        plan.rebind_dense_factor(factor, wrong_size);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "targeted dense rebind accepted a changed size");

    rejected = false;
    const std::array<FactorSparseEntry, 2> duplicate{{
        FactorSparseEntry{1U, QComplex{1.0}},
        FactorSparseEntry{1U, QComplex{2.0}},
    }};
    try {
        plan.rebind_sparse_factor(factor, duplicate);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "targeted sparse rebind accepted a duplicate index");

    rejected = false;
    const std::array<FactorSparseEntry, 1> out_of_range{{
        FactorSparseEntry{6U, QComplex{1.0}},
    }};
    try {
        plan.rebind_sparse_factor(factor, out_of_range);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "targeted sparse rebind accepted an out-of-range index");

    rejected = false;
    const std::array<QComplex, 6> nonfinite{{
        QComplex{1.0}, QComplex{2.0}, QComplex{3.0},
        QComplex{4.0}, QComplex{5.0},
        QComplex{std::numeric_limits<double>::infinity()},
    }};
    try {
        plan.rebind_dense_factor(factor, nonfinite);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "targeted dense rebind accepted a non-finite value");

    rejected = false;
    try {
        plan.rebind_dense_factor(static_cast<FactorId>(factor + 1U), dense);
    } catch (const QStateError&) {
        rejected = true;
    }
    require(rejected, "targeted rebind accepted an out-of-range factor id");

    require(plan.rebind_count() == stable_count,
            "failed targeted rebind changed the rebind count");
    require_close(plan.evaluate(workspace), stable,
                  "failed targeted rebind was not transactional");
    return 0;
}

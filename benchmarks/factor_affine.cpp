#include "qubit/qfactor_affine.hpp"
#include "qubit/qfactor_decision.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using qubit::ExactFactorAffineConfig;
using qubit::ExactFactorAffinePlan;
using qubit::ExactFactorConfig;
using qubit::ExactFactorDecisionConfig;
using qubit::ExactFactorDecisionPlan;
using qubit::ExactFactorDecisionWorkspace;
using qubit::ExactFactorGraph;
using qubit::ExactFactorPlan;
using qubit::ExactFactorWorkspace;
using qubit::FactorSparseEntry;
using qubit::FactorVariableId;
using qubit::QComplex;
using qubit::QStateError;

#if defined(_MSC_VER)
#define QSA_BENCH_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define QSA_BENCH_NOINLINE __attribute__((noinline))
#else
#define QSA_BENCH_NOINLINE
#endif

volatile double query_sink = 0.0;

struct Equation {
    std::array<FactorVariableId, 3> variables{};
    bool rhs{false};
};

struct System {
    std::size_t variable_count{0U};
    std::vector<std::uint8_t> witness{};
    std::vector<Equation> equations{};
};

[[nodiscard]] System make_system(
    std::size_t variable_count,
    std::size_t equation_count,
    std::uint64_t seed) {
    std::mt19937_64 generator(seed);
    System system;
    system.variable_count = variable_count;
    system.witness.resize(variable_count);
    for (std::uint8_t& bit : system.witness) {
        bit = static_cast<std::uint8_t>(generator() & 1U);
    }
    system.equations.reserve(equation_count);
    while (system.equations.size() < equation_count) {
        std::array<FactorVariableId, 3> scope{};
        for (std::size_t position = 0U; position < scope.size();) {
            const FactorVariableId candidate =
                static_cast<FactorVariableId>(generator() % variable_count);
            bool duplicate = false;
            for (std::size_t previous = 0U; previous < position; ++previous) {
                duplicate = duplicate || scope[previous] == candidate;
            }
            if (!duplicate) {
                scope[position++] = candidate;
            }
        }
        bool rhs = false;
        for (const FactorVariableId variable : scope) {
            rhs = rhs != (system.witness[variable] != 0U);
        }
        system.equations.push_back({scope, rhs});
    }
    return system;
}

[[nodiscard]] std::array<QComplex, 8> parity_table(bool rhs) {
    std::array<QComplex, 8> table{};
    for (std::size_t index = 0U; index < table.size(); ++index) {
        if (((std::popcount(index) & 1U) != 0U) == rhs) {
            table[index] = {1.0, 0.0};
        }
    }
    return table;
}

[[nodiscard]] ExactFactorGraph make_graph(const System& system) {
    ExactFactorConfig config;
    config.max_variables = system.variable_count + 1U;
    config.max_factors = system.equations.size() + 1U;
    ExactFactorGraph graph(config);
    for (std::size_t variable = 0U; variable < system.variable_count; ++variable) {
        static_cast<void>(graph.add_variable(2U));
    }
    for (std::size_t index = 0U; index < system.equations.size(); ++index) {
        const Equation& equation = system.equations[index];
        const std::array<QComplex, 8> table = parity_table(equation.rhs);
        if ((index & 1U) == 0U) {
            static_cast<void>(graph.add_dense_factor(equation.variables, table));
        } else {
            std::vector<FactorSparseEntry> sparse;
            sparse.reserve(4U);
            for (std::size_t entry = 0U; entry < table.size(); ++entry) {
                if (table[entry].re != 0.0 || table[entry].im != 0.0) {
                    sparse.push_back({entry, table[entry]});
                }
            }
            static_cast<void>(graph.add_sparse_factor(equation.variables, sparse));
        }
    }
    return graph;
}

class PackedGF2Control {
public:
    PackedGF2Control(const System& system, FactorVariableId retained)
        : variable_count_(system.variable_count),
          word_count_((variable_count_ + 63U) / 64U),
          basis_(variable_count_ * word_count_, 0U),
          basis_rhs_(variable_count_, 0U),
          active_(variable_count_, 0U),
          retained_(retained) {
        std::vector<std::uint64_t> row(word_count_);
        for (const Equation& equation : system.equations) {
            std::fill(row.begin(), row.end(), 0U);
            for (const FactorVariableId variable : equation.variables) {
                row[variable / 64U] |= std::uint64_t{1U} << (variable % 64U);
            }
            insert(row, equation.rhs);
        }
        free_variables_ = inconsistent_ ? 0U : variable_count_ - rank_;
        if (!inconsistent_) {
            std::fill(row.begin(), row.end(), 0U);
            row[retained_ / 64U] |= std::uint64_t{1U} << (retained_ % 64U);
            bool rhs = false;
            reduce(row, rhs);
            retained_fixed_ = first_set(row) == variable_count_;
            retained_value_ = rhs;
        }
    }

    [[nodiscard]] std::array<QComplex, 2> marginal() const {
        if (inconsistent_) {
            return {};
        }
        if (retained_fixed_) {
            std::array<QComplex, 2> output{};
            output[retained_value_ ? 1U : 0U] = power_of_two(free_variables_);
            return output;
        }
        if (free_variables_ == 0U) {
            throw std::runtime_error("packed GF(2) control retained balance invariant failed");
        }
        const QComplex half = power_of_two(free_variables_ - 1U);
        return {{half, half}};
    }

    [[nodiscard]] std::size_t rank() const noexcept { return rank_; }
    [[nodiscard]] std::size_t free_variables() const noexcept { return free_variables_; }
    [[nodiscard]] bool consistent() const noexcept { return !inconsistent_; }
    [[nodiscard]] std::size_t estimated_bytes() const noexcept {
        return sizeof(*this) +
               basis_.capacity() * sizeof(std::uint64_t) +
               basis_rhs_.capacity() * sizeof(std::uint8_t) +
               active_.capacity() * sizeof(std::uint8_t);
    }

private:
    std::size_t variable_count_{0U};
    std::size_t word_count_{0U};
    std::vector<std::uint64_t> basis_{};
    std::vector<std::uint8_t> basis_rhs_{};
    std::vector<std::uint8_t> active_{};
    FactorVariableId retained_{0U};
    std::size_t rank_{0U};
    std::size_t free_variables_{0U};
    bool inconsistent_{false};
    bool retained_fixed_{false};
    bool retained_value_{false};

    [[nodiscard]] std::uint64_t* basis_row(std::size_t pivot) noexcept {
        return basis_.data() + pivot * word_count_;
    }

    [[nodiscard]] const std::uint64_t* basis_row(std::size_t pivot) const noexcept {
        return basis_.data() + pivot * word_count_;
    }

    [[nodiscard]] std::size_t first_set(
        std::span<const std::uint64_t> row) const noexcept {
        for (std::size_t word = 0U; word < row.size(); ++word) {
            if (row[word] != 0U) {
                const std::size_t bit = static_cast<std::size_t>(std::countr_zero(row[word]));
                const std::size_t variable = word * 64U + bit;
                return variable < variable_count_ ? variable : variable_count_;
            }
        }
        return variable_count_;
    }

    void xor_basis(std::vector<std::uint64_t>& row, std::size_t pivot) const noexcept {
        const std::uint64_t* source = basis_row(pivot);
        for (std::size_t word = pivot / 64U; word < word_count_; ++word) {
            row[word] ^= source[word];
        }
    }

    void reduce(std::vector<std::uint64_t>& row, bool& rhs) const noexcept {
        while (true) {
            const std::size_t pivot = first_set(row);
            if (pivot == variable_count_ || active_[pivot] == 0U) {
                return;
            }
            xor_basis(row, pivot);
            rhs = rhs != (basis_rhs_[pivot] != 0U);
        }
    }

    void insert(std::vector<std::uint64_t>& row, bool rhs) {
        reduce(row, rhs);
        const std::size_t pivot = first_set(row);
        if (pivot == variable_count_) {
            inconsistent_ = inconsistent_ || rhs;
            return;
        }
        std::copy(row.begin(), row.end(), basis_row(pivot));
        basis_rhs_[pivot] = rhs ? 1U : 0U;
        active_[pivot] = 1U;
        ++rank_;
    }

    [[nodiscard]] static QComplex power_of_two(std::size_t exponent) noexcept {
        return {std::ldexp(1.0, static_cast<int>(exponent)), 0.0};
    }
};

QSA_BENCH_NOINLINE void run_affine_query(
    const ExactFactorAffinePlan& plan,
    std::array<QComplex, 2>& output) {
    plan.evaluate(output);
    query_sink = output[0].re + output[1].re;
}

QSA_BENCH_NOINLINE void run_control_query(
    const PackedGF2Control& plan,
    std::array<QComplex, 2>& output) {
    output = plan.marginal();
    query_sink = output[0].re + output[1].re;
}

QSA_BENCH_NOINLINE void run_decision_query(
    const ExactFactorDecisionPlan& plan,
    ExactFactorDecisionWorkspace& workspace,
    std::array<QComplex, 2>& output) {
    plan.evaluate(output, workspace);
    query_sink = output[0].re + output[1].re;
}

QSA_BENCH_NOINLINE void run_generic_query(
    const ExactFactorPlan& plan,
    ExactFactorWorkspace& workspace,
    std::array<QComplex, 2>& output) {
    plan.evaluate(output, workspace);
    query_sink = output[0].re + output[1].re;
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

[[nodiscard]] std::array<QComplex, 2> brute_marginal(
    const System& system,
    FactorVariableId retained) {
    if (system.variable_count >= std::numeric_limits<std::size_t>::digits) {
        throw std::runtime_error("brute affine control exceeds size_t bit width");
    }
    const std::size_t assignments = std::size_t{1U} << system.variable_count;
    std::array<QComplex, 2> output{};
    for (std::size_t assignment = 0U; assignment < assignments; ++assignment) {
        bool valid = true;
        for (const Equation& equation : system.equations) {
            bool rhs = false;
            for (const FactorVariableId variable : equation.variables) {
                rhs = rhs != (((assignment >> variable) & 1U) != 0U);
            }
            if (rhs != equation.rhs) {
                valid = false;
                break;
            }
        }
        if (valid) {
            output[(assignment >> retained) & 1U] += {1.0, 0.0};
        }
    }
    return output;
}

void matched_evidence() {
    constexpr std::size_t variables = 16U;
    constexpr std::size_t equations = 40U;
    const System system = make_system(variables, equations, 0xa6611eULL);
    ExactFactorGraph graph = make_graph(system);
    const std::array<FactorVariableId, 1> retained{{0U}};

    std::optional<ExactFactorAffinePlan> affine;
    const double affine_compile_ms = median_ms([&] {
        affine.emplace(graph, retained);
    }, 11U);
    std::optional<PackedGF2Control> control;
    const double control_compile_ms = median_ms([&] {
        control.emplace(system, retained.front());
    }, 11U);
    std::optional<ExactFactorDecisionPlan> decision;
    const double decision_compile_ms = median_ms([&] {
        decision.emplace(graph, retained);
    }, 7U);
    std::optional<ExactFactorPlan> generic;
    const double generic_compile_ms = median_ms([&] {
        generic.emplace(graph, retained);
    }, 5U);

    std::array<QComplex, 2> affine_output{};
    std::array<QComplex, 2> control_output{};
    std::array<QComplex, 2> decision_output{};
    std::array<QComplex, 2> generic_output{};
    auto decision_workspace = decision->workspace();
    auto generic_workspace = generic->workspace();
    const double affine_query_ms = median_ms([&] {
        run_affine_query(*affine, affine_output);
    }, 11U, 100000U);
    const double control_query_ms = median_ms([&] {
        run_control_query(*control, control_output);
    }, 11U, 100000U);
    const double decision_query_ms = median_ms([&] {
        run_decision_query(*decision, decision_workspace, decision_output);
    }, 11U, 1000U);
    const double generic_query_ms = median_ms([&] {
        run_generic_query(*generic, generic_workspace, generic_output);
    }, 7U, 3U);
    const std::array<QComplex, 2> brute_output = brute_marginal(system, retained.front());

    const double error = std::max({
        max_error(affine_output, control_output),
        max_error(affine_output, decision_output),
        max_error(affine_output, generic_output),
        max_error(affine_output, brute_output),
    });
    if (error > 2e-11 || affine->stats().rank != control->rank() ||
        affine->stats().free_variables != control->free_variables()) {
        throw std::runtime_error("matched affine evidence failed");
    }

    std::cout << "affine_matched_variables=" << variables << '\n';
    std::cout << "affine_matched_equations=" << equations << '\n';
    std::cout << "affine_matched_rank=" << affine->stats().rank << '\n';
    std::cout << "affine_matched_free_variables=" << affine->stats().free_variables << '\n';
    std::cout << "affine_matched_basis_terms=" << affine->stats().basis_terms << '\n';
    std::cout << "affine_matched_peak_row_terms=" << affine->stats().peak_row_terms << '\n';
    std::cout << "affine_compile_ms=" << affine_compile_ms << '\n';
    std::cout << "affine_control_compile_ms=" << control_compile_ms << '\n';
    std::cout << "affine_decision_compile_ms=" << decision_compile_ms << '\n';
    std::cout << "affine_generic_compile_ms=" << generic_compile_ms << '\n';
    std::cout << "affine_compile_speedup_vs_decision=" << decision_compile_ms / affine_compile_ms << '\n';
    std::cout << "affine_compile_speedup_vs_generic=" << generic_compile_ms / affine_compile_ms << '\n';
    std::cout << "affine_control_compile_over_affine=" << control_compile_ms / affine_compile_ms << '\n';
    std::cout << "affine_query_ms=" << affine_query_ms << '\n';
    std::cout << "affine_control_query_ms=" << control_query_ms << '\n';
    std::cout << "affine_decision_query_ms=" << decision_query_ms << '\n';
    std::cout << "affine_generic_query_ms=" << generic_query_ms << '\n';
    std::cout << "affine_query_speedup_vs_decision=" << decision_query_ms / affine_query_ms << '\n';
    std::cout << "affine_query_speedup_vs_generic=" << generic_query_ms / affine_query_ms << '\n';
    std::cout << "affine_control_query_over_affine=" << control_query_ms / affine_query_ms << '\n';
    std::cout << "affine_plan_bytes=" << affine->estimated_bytes() << '\n';
    std::cout << "affine_control_bytes=" << control->estimated_bytes() << '\n';
    std::cout << "affine_decision_plan_bytes=" << decision->estimated_bytes() << '\n';
    std::cout << "affine_decision_workspace_bytes=" << decision_workspace.estimated_bytes() << '\n';
    std::cout << "affine_generic_plan_bytes=" << generic->estimated_bytes() << '\n';
    std::cout << "affine_generic_workspace_bytes=" << generic_workspace.estimated_bytes() << '\n';
    std::cout << "affine_matched_error=" << error << '\n';
}

void capability_evidence() {
    constexpr std::size_t variables = 512U;
    constexpr std::size_t equations = 1536U;
    const System system = make_system(variables, equations, 0x3a6611eULL);
    ExactFactorGraph graph = make_graph(system);
    const std::array<FactorVariableId, 1> retained{{0U}};

    ExactFactorAffineConfig affine_config;
    affine_config.max_variables = variables;
    affine_config.max_equations = equations;
    affine_config.max_row_terms = variables;
    affine_config.max_basis_terms = variables * variables;
    std::optional<ExactFactorAffinePlan> affine;
    const double affine_compile_ms = median_ms([&] {
        affine.emplace(graph, retained, affine_config);
    }, 3U);
    std::optional<PackedGF2Control> control;
    const double control_compile_ms = median_ms([&] {
        control.emplace(system, retained.front());
    }, 5U);

    std::array<QComplex, 2> affine_output{};
    std::array<QComplex, 2> control_output{};
    const double affine_query_ms = median_ms([&] {
        run_affine_query(*affine, affine_output);
    }, 11U, 100000U);
    const double control_query_ms = median_ms([&] {
        run_control_query(*control, control_output);
    }, 11U, 100000U);
    const double control_error = max_error(affine_output, control_output);
    if (control_error > 2e-11 || affine->stats().rank != control->rank() ||
        affine->stats().free_variables != control->free_variables() ||
        !control->consistent()) {
        throw std::runtime_error("large affine GF(2) control mismatch");
    }

    ExactFactorDecisionConfig decision_config;
    decision_config.max_variables = variables;
    decision_config.max_nodes = 65'536U;
    decision_config.max_apply_pairs = 262'144U;
    bool decision_accepted = false;
    double decision_compile_ms = 0.0;
    double decision_query_ms = 0.0;
    std::size_t decision_plan_bytes = 0U;
    const auto decision_start = Clock::now();
    try {
        ExactFactorDecisionPlan decision(graph, retained, decision_config);
        decision_compile_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - decision_start).count();
        auto workspace = decision.workspace();
        std::array<QComplex, 2> output{};
        decision_query_ms = median_ms([&] {
            run_decision_query(decision, workspace, output);
        }, 7U, 100U);
        if (max_error(output, affine_output) > 2e-11) {
            throw std::runtime_error("large DecisionDiagram result differs from AffineXOR");
        }
        decision_plan_bytes = decision.estimated_bytes();
        decision_accepted = true;
    } catch (const QStateError&) {
        decision_compile_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - decision_start).count();
    }

    bool generic_accepted = false;
    double generic_compile_ms = 0.0;
    double generic_query_ms = 0.0;
    std::size_t generic_plan_bytes = 0U;
    const auto generic_start = Clock::now();
    try {
        ExactFactorPlan generic(graph, retained);
        generic_compile_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - generic_start).count();
        auto workspace = generic.workspace();
        std::array<QComplex, 2> output{};
        generic_query_ms = median_ms([&] {
            run_generic_query(generic, workspace, output);
        }, 3U);
        if (max_error(output, affine_output) > 2e-11) {
            throw std::runtime_error("large generic VE result differs from AffineXOR");
        }
        generic_plan_bytes = generic.estimated_bytes();
        generic_accepted = true;
    } catch (const QStateError&) {
        generic_compile_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - generic_start).count();
    }

    std::cout << "affine_capability_variables=" << variables << '\n';
    std::cout << "affine_capability_equations=" << equations << '\n';
    std::cout << "affine_capability_rank=" << affine->stats().rank << '\n';
    std::cout << "affine_capability_free_variables=" << affine->stats().free_variables << '\n';
    std::cout << "affine_capability_basis_terms=" << affine->stats().basis_terms << '\n';
    std::cout << "affine_capability_peak_row_terms=" << affine->stats().peak_row_terms << '\n';
    std::cout << "affine_capability_row_reductions=" << affine->stats().row_reductions << '\n';
    std::cout << "affine_capability_compile_ms=" << affine_compile_ms << '\n';
    std::cout << "affine_capability_query_ms=" << affine_query_ms << '\n';
    std::cout << "affine_capability_plan_bytes=" << affine->estimated_bytes() << '\n';
    std::cout << "affine_capability_control_compile_ms=" << control_compile_ms << '\n';
    std::cout << "affine_capability_control_query_ms=" << control_query_ms << '\n';
    std::cout << "affine_capability_control_bytes=" << control->estimated_bytes() << '\n';
    std::cout << "affine_capability_control_compile_over_affine=" << control_compile_ms / affine_compile_ms << '\n';
    std::cout << "affine_capability_control_query_over_affine=" << control_query_ms / affine_query_ms << '\n';
    std::cout << "affine_capability_decision_accepted=" << decision_accepted << '\n';
    std::cout << "affine_capability_decision_compile_ms=" << decision_compile_ms << '\n';
    std::cout << "affine_capability_decision_query_ms=" << decision_query_ms << '\n';
    std::cout << "affine_capability_decision_plan_bytes=" << decision_plan_bytes << '\n';
    std::cout << "affine_capability_generic_accepted=" << generic_accepted << '\n';
    std::cout << "affine_capability_generic_compile_ms=" << generic_compile_ms << '\n';
    std::cout << "affine_capability_generic_query_ms=" << generic_query_ms << '\n';
    std::cout << "affine_capability_generic_plan_bytes=" << generic_plan_bytes << '\n';
    std::cout << "affine_capability_error=" << control_error << '\n';
}

void collapse_adversary() {
    ExactFactorGraph graph;
    static_cast<void>(graph.add_variable(2U));
    const std::array<FactorVariableId, 1> scope{{0U}};
    const std::array<QComplex, 2> weighted{{QComplex{0.4}, QComplex{0.6}}};
    static_cast<void>(graph.add_dense_factor(scope, weighted));
    bool rejected = false;
    try {
        ExactFactorAffinePlan affine(graph);
        static_cast<void>(affine);
    } catch (const QStateError&) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error("non-affine collapse adversary was accepted");
    }
    std::cout << "affine_non_affine_rejected=1\n";
}

}  // namespace

int main() {
    std::cout << std::setprecision(12);
    matched_evidence();
    capability_evidence();
    collapse_adversary();
    return 0;
}

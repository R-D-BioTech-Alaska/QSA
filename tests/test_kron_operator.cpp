#include "qubit/qkron.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

using qubit::KronOperator;
using qubit::KronVector;
using qubit::QComplex;
using qubit::QStateError;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(QComplex actual, QComplex expected, double tolerance, const char* message) {
    if (!qubit::almost_equal(actual, expected, tolerance)) {
        throw std::runtime_error(message);
    }
}

KronVector small_vector(std::size_t max_terms = 8U) {
    KronVector result({2U, 3U}, max_terms);
    const std::vector<std::vector<QComplex>> first{
        {{0.8, 0.1}, {0.3, -0.2}},
        {{0.4, 0.0}, {-0.2, 0.3}, {0.7, -0.1}},
    };
    const std::vector<std::vector<QComplex>> second{
        {{0.2, -0.4}, {0.5, 0.1}},
        {{-0.1, 0.2}, {0.3, 0.4}, {0.6, 0.0}},
    };
    result.add_term({0.7, -0.15}, first);
    result.add_term({-0.25, 0.3}, second);
    return result;
}

KronOperator small_operator(std::size_t max_terms = 8U) {
    KronOperator result({2U, 3U}, max_terms);
    const std::vector<std::vector<QComplex>> first{
        {
            {0.5, 0.1}, {-0.2, 0.3},
            {0.4, -0.1}, {0.7, 0.0},
        },
        {
            {0.8, 0.0}, {0.1, -0.2}, {0.0, 0.1},
            {-0.3, 0.2}, {0.6, 0.0}, {0.2, 0.1},
            {0.05, 0.0}, {-0.1, -0.15}, {0.9, 0.0},
        },
    };
    const std::vector<std::vector<QComplex>> second{
        {
            {0.2, 0.0}, {0.0, -0.4},
            {0.0, 0.4}, {-0.2, 0.0},
        },
        {
            {0.4, 0.0}, {0.0, 0.1}, {-0.2, 0.0},
            {0.0, -0.1}, {-0.5, 0.0}, {0.0, 0.2},
            {-0.2, 0.0}, {0.0, -0.2}, {0.1, 0.0},
        },
    };
    result.add_term({0.6, -0.05}, first);
    result.add_term({-0.2, 0.15}, second);
    return result;
}

KronVector huge_vector(std::size_t sites, std::size_t max_terms = 8U) {
    KronVector result(std::vector<std::size_t>(sites, 2U), max_terms);
    std::vector<std::vector<QComplex>> first;
    std::vector<std::vector<QComplex>> second;
    first.reserve(sites);
    second.reserve(sites);
    for (std::size_t site = 0; site < sites; ++site) {
        const double a = 0.007 * static_cast<double>(site + 1U);
        const double b = 0.009 * static_cast<double>(site + 1U);
        first.push_back({{std::cos(a), 0.0}, {std::sin(a), 0.0}});
        second.push_back({{std::cos(b), 0.0}, {0.0, std::sin(b)}});
    }
    result.add_term({0.8, 0.1}, first);
    result.add_term({-0.15, 0.05}, second);
    return result;
}

KronOperator huge_operator(std::size_t sites, std::size_t max_terms = 8U) {
    KronOperator result(std::vector<std::size_t>(sites, 2U), max_terms);
    std::vector<std::vector<QComplex>> first;
    std::vector<std::vector<QComplex>> second;
    first.reserve(sites);
    second.reserve(sites);
    for (std::size_t site = 0; site < sites; ++site) {
        const double a = 0.011 * static_cast<double>(site + 1U);
        const double b = 0.013 * static_cast<double>(site + 1U);
        const double ca = std::cos(a);
        const double sa = std::sin(a);
        const double cb = std::cos(b);
        const double sb = std::sin(b);
        first.push_back({
            {ca, 0.0}, {sa, 0.0},
            {sa, 0.0}, {-ca, 0.0},
        });
        second.push_back({
            {cb, 0.0}, {0.0, -sb},
            {0.0, sb}, {-cb, 0.0},
        });
    }
    result.add_term({0.55, 0.0}, first);
    result.add_term({-0.3, 0.0}, second);
    return result;
}

}  // namespace

int main() {
    {
        const KronVector state = small_vector();
        const KronOperator operation = small_operator();
        const std::vector<QComplex> dense_state = state.materialize(6U);
        const std::vector<QComplex> dense_operator = operation.materialize(36U);
        const KronVector structured_output = operation.apply(state);
        const std::vector<QComplex> dense_output = structured_output.materialize(6U);

        require(structured_output.term_count() == 4U,
                "Kronecker operator application did not expose exact rank growth");
        for (std::size_t row = 0; row < dense_state.size(); ++row) {
            QComplex expected{};
            for (std::size_t column = 0; column < dense_state.size(); ++column) {
                expected += dense_operator[row * dense_state.size() + column] *
                            dense_state[column];
            }
            require_close(dense_output[row], expected, 2e-13,
                          "structured Kronecker matvec differs from dense execution");
        }

        QComplex dense_expectation{};
        for (std::size_t index = 0; index < dense_state.size(); ++index) {
            dense_expectation += dense_state[index].conjugate() * dense_output[index];
        }
        require_close(operation.expectation(state), dense_expectation, 3e-13,
                      "structured Kronecker expectation differs from dense execution");
        require_close(operation.matrix_element(state, state), dense_expectation, 3e-13,
                      "structured Kronecker matrix element differs from dense execution");
    }

    {
        constexpr std::size_t sites = 100U;
        const KronVector state = huge_vector(sites);
        const KronOperator operation = huge_operator(sites);
        const QComplex expectation = operation.expectation(state);
        require(std::isfinite(expectation.re) && std::isfinite(expectation.im),
                "100-site Kronecker expectation became nonfinite");
        const KronVector output = operation.apply(state);
        require(output.factor_count() == sites && output.term_count() == 4U,
                "100-site Kronecker matvec changed exact structural rank");
        require(output.estimated_bytes() < 100'000U,
                "100-site Kronecker matvec unexpectedly expanded storage");

        bool rejected = false;
        try {
            static_cast<void>(operation.materialize());
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "100-site Kronecker operator allowed dense matrix materialization");
    }

    {
        const KronVector state = small_vector(2U);
        const KronOperator operation = small_operator(2U);
        bool rejected = false;
        try {
            static_cast<void>(operation.apply(state));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "Kronecker matvec did not fail closed on exact rank growth");
    }

    {
        bool rejected = false;
        try {
            KronOperator invalid({2U, 3U});
            const std::vector<std::vector<QComplex>> wrong{
                std::vector<QComplex>(4U),
                std::vector<QComplex>(8U),
            };
            invalid.add_term({1.0, 0.0}, wrong);
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "KronOperator accepted a malformed local matrix");

        rejected = false;
        try {
            const KronOperator operation = small_operator();
            KronVector wrong({2U, 2U});
            static_cast<void>(operation.apply(wrong));
        } catch (const QStateError&) {
            rejected = true;
        }
        require(rejected, "KronOperator accepted an incompatible vector shape");
    }

    return 0;
}

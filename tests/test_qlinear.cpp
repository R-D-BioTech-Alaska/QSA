#include "qubit/qlinear.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace qubit;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    const auto length = QPhysicalDimension::base(QPhysicalDimension::Length);
    const QMathType length_scalar = QMathType::scalar_type(QMathScalar::Rational, length);

    QExactLinearSystem unique({"x", "y"}, length_scalar);
    unique.add_equation(std::vector<QRational>{QRational(1), QRational(1)}, QRational(5));
    unique.add_equation(std::vector<QRational>{QRational(2), QRational(-1)}, QRational(1));
    const QExactLinearResult solved = unique.solve();
    require(solved.unique() && solved.consistent() && solved.rank == 2U &&
            solved.augmented_rank == 2U && solved.pivot_columns == std::vector<std::size_t>({0U, 1U}) &&
            solved.solution == std::vector<QRational>({QRational(2), QRational(3)}) &&
            solved.variable_type == length_scalar && solved.exact,
            "exact linear unique solve failed");

    QExactLinearSystem reordered({"x", "y"}, length_scalar);
    reordered.add_equation(std::vector<QRational>{QRational(2), QRational(-1)}, QRational(1));
    reordered.add_equation(std::vector<QRational>{QRational(1), QRational(1)}, QRational(5));
    const QExactLinearResult reordered_result = reordered.solve();
    require(reordered_result.solution == solved.solution && reordered_result.canonical == solved.canonical,
            "exact linear canonical result depends on equation insertion order");

    QExactLinearSystem underdetermined({"x", "y"});
    underdetermined.add_equation(std::vector<QRational>{QRational(1), QRational(1)}, QRational(5));
    const QExactLinearResult under = underdetermined.solve();
    require(under.status == QLinearStatus::Underdetermined && under.rank == 1U &&
            under.augmented_rank == 1U && under.solution.empty() && under.consistent(),
            "exact linear underdetermined classification failed");

    QExactLinearSystem inconsistent({"x", "y"});
    inconsistent.add_equation(std::vector<QRational>{QRational(1), QRational(1)}, QRational(5));
    inconsistent.add_equation(std::vector<QRational>{QRational(1), QRational(1)}, QRational(6));
    const QExactLinearResult contradiction = inconsistent.solve();
    require(contradiction.status == QLinearStatus::Inconsistent && contradiction.rank == 1U &&
            contradiction.augmented_rank == 2U && contradiction.solution.empty() && !contradiction.consistent(),
            "exact linear inconsistency classification failed");

    QExactLinearSystem fractional({"x", "y"});
    fractional.add_equation(
        std::vector<QRational>{QRational(1, 2), QRational(1, 3)},
        QRational(1));
    fractional.add_equation(
        std::vector<QRational>{QRational(1), QRational(-1)},
        QRational(0));
    const QExactLinearResult fractional_result = fractional.solve();
    require(fractional_result.unique() &&
            fractional_result.solution == std::vector<QRational>({QRational(6, 5), QRational(6, 5)}),
            "exact linear fractional rational arithmetic failed");

    const QRational large = QRational::parse("1000000000000000000000000000000");
    QExactLinearSystem arbitrary_precision({"x"});
    arbitrary_precision.add_equation(std::vector<QRational>{QRational(3)}, large);
    const QExactLinearResult arbitrary = arbitrary_precision.solve();
    require(arbitrary.unique() && arbitrary.solution.front() == large / QRational(3),
            "exact linear solver lost arbitrary-precision rational state");

    bool rejected = false;
    try { (void)QExactLinearSystem({"y", "x"}); } catch (const QMathError&) { rejected = true; }
    require(rejected, "exact linear system accepted noncanonical variable ordering");

    rejected = false;
    try {
        QExactLinearSystem bad_width({"x", "y"});
        bad_width.add_equation(std::vector<QRational>{QRational(1)}, QRational(1));
    } catch (const QMathError&) {
        rejected = true;
    }
    require(rejected, "exact linear system accepted a coefficient-width mismatch");

    QExactLinearSystem capped({"x"}, QMathType::scalar_type(QMathScalar::Rational), QExactLinearConfig{1U, 1U, 2U});
    capped.add_equation(std::vector<QRational>{QRational(1)}, QRational(1));
    rejected = false;
    try { capped.add_equation(std::vector<QRational>{QRational(1)}, QRational(2)); } catch (const QMathError&) { rejected = true; }
    require(rejected && capped.equation_count() == 1U,
            "exact linear equation cap did not fail closed");

    std::cout << "QExact linear-system tests passed\n";
}

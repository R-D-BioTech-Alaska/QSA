#include "qubit/qpolynomial.hpp"

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
    const QPolynomial quadratic("x", {QRational(-2), QRational(0), QRational(1)});
    require(quadratic.canonical() == "poly:x:[-2,0,1]" && quadratic.degree() == 2U &&
            quadratic.evaluate(QRational(3, 2)) == QRational(1, 4),
            "exact polynomial construction or Horner evaluation failed");
    require(quadratic.derivative().canonical() == "poly:x:[0,2]",
            "exact polynomial derivative failed");

    const QPolynomial cubic_minus_one(
        "x", {QRational(-1), QRational(0), QRational(0), QRational(1)});
    const QPolynomial x_minus_one("x", {QRational(-1), QRational(1)});
    const auto division = cubic_minus_one.divmod(x_minus_one);
    require(division.first == QPolynomial("x", {QRational(1), QRational(1), QRational(1)}) &&
            division.second.is_zero(),
            "exact polynomial Euclidean division failed");

    const QPolynomial repeated("x", {QRational(2), QRational(-3), QRational(0), QRational(1)});
    const QPolynomial repeated_gcd = QPolynomial::gcd(repeated, repeated.derivative());
    require(repeated_gcd == QPolynomial("x", {QRational(-1), QRational(1)}),
            "exact polynomial gcd failed to identify a repeated factor");
    require(repeated.square_free() == QPolynomial("x", {QRational(-2), QRational(1), QRational(1)}),
            "exact polynomial square-free reduction failed");

    const QPolynomial three_roots("x", {QRational(0), QRational(-1), QRational(0), QRational(1)});
    const QPolynomialRootCountReceipt all = three_roots.count_distinct_real_roots(QRational(-2), QRational(2));
    const QPolynomialRootCountReceipt center = three_roots.count_distinct_real_roots(QRational(-1, 2), QRational(1, 2));
    require(all.distinct_roots == 3U && all.degree == 3U && all.square_free_degree == 3U &&
            all.sturm_terms >= 2U && all.exact &&
            center.distinct_roots == 1U,
            "exact Sturm root counting failed");

    const QPolynomialRootCountReceipt repeated_roots =
        repeated.count_distinct_real_roots(QRational(-3), QRational(3));
    require(repeated_roots.distinct_roots == 2U && repeated_roots.square_free_degree == 2U,
            "Sturm counting did not reduce repeated roots exactly");

    const QPolynomial no_real_roots("x", {QRational(1), QRational(0), QRational(1)});
    require(no_real_roots.count_distinct_real_roots(QRational(-10), QRational(10)).distinct_roots == 0U,
            "exact Sturm counting invented real roots");

    bool rejected = false;
    try {
        (void)three_roots.count_distinct_real_roots(QRational(-1), QRational(2));
    } catch (const QMathError&) {
        rejected = true;
    }
    require(rejected, "Sturm interval accepted an exact root endpoint");

    rejected = false;
    try {
        (void)QPolynomial("x").count_distinct_real_roots(QRational(-1), QRational(1));
    } catch (const QMathError&) {
        rejected = true;
    }
    require(rejected, "zero polynomial did not fail closed as an infinite-root case");

    rejected = false;
    try {
        (void)(QPolynomial("x", {QRational(1), QRational(1)}) +
               QPolynomial("y", {QRational(1), QRational(1)}));
    } catch (const QMathError&) {
        rejected = true;
    }
    require(rejected, "polynomial algebra mixed distinct admitted variable identities");

    rejected = false;
    try {
        (void)QPolynomial(
            "x",
            {QRational(1), QRational(0), QRational(0), QRational(1)},
            QPolynomialConfig{2U, 64U});
    } catch (const QMathError&) {
        rejected = true;
    }
    require(rejected, "polynomial construction exceeded the configured degree cap");

    rejected = false;
    try {
        const QPolynomial capped("x", {QRational(1), QRational(1)}, QPolynomialConfig{1U, 64U});
        (void)(capped * capped);
    } catch (const QMathError&) {
        rejected = true;
    }
    require(rejected, "polynomial multiplication exceeded the configured degree cap");

    const QRational huge = QRational::parse("1000000000000000000000000000000");
    const QPolynomial arbitrary("u", {huge, QRational(1)});
    require(arbitrary.evaluate(QRational(1)).canonical() == "1000000000000000000000000000001" &&
            arbitrary.receipt().canonical == arbitrary.canonical(),
            "polynomial algebra lost arbitrary-precision rational state or receipt identity");

    std::cout << "Exact rational polynomial and Sturm tests passed\n";
}

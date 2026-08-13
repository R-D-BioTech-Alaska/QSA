#include "qubit/qmath.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace qubit;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    QMathArena math;
    const auto length = QPhysicalDimension::base(QPhysicalDimension::Length);
    const auto time = QPhysicalDimension::base(QPhysicalDimension::Time);
    const auto x = math.symbol("x", QMathType::scalar_type(QMathScalar::Real, length));
    const auto t = math.symbol("t", QMathType::scalar_type(QMathScalar::Real, time));
    const auto y = math.symbol("y", QMathType::scalar_type(QMathScalar::Real, length));
    const auto z = math.symbol("z", QMathType::scalar_type(QMathScalar::Real));

    require(QRational(2, 4) == QRational(1, 2), "rational normalization failed");
    require(math.add({x, y}).get() == math.add({y, x}).get(), "commutative sum was not interned canonically");

    bool rejected = false;
    try { (void)math.add({x, t}); } catch (const QMathError&) { rejected = true; }
    require(rejected, "dimensionally invalid addition was accepted");

    const auto velocity = math.multiply({x, math.power(t, QRational(-1))});
    require(velocity->type.dimension == length.divided(time), "velocity dimension is wrong");
    require(math.power(x, QRational(1, 2))->type.dimension == length.powered(QRational(1, 2)),
            "fractional physical dimension was lost");
    require(math.derivative(x, x)->type.dimension.is_dimensionless(),
            "self derivative did not cancel physical dimensions");
    require(math.integral(x, x)->type.dimension == length.powered(QRational(2)),
            "integral physical dimension is wrong");

    const auto f = math.add({math.power(z, QRational(3)), math.multiply({math.rational(2), z})});
    const auto df = math.derivative(f, z);
    const auto expected_df = math.add({math.multiply({math.rational(3), math.power(z, QRational(2))}), math.rational(2)});
    require(math.equivalent(df, expected_df), "exact polynomial derivative failed");

    const auto integral = math.integral(math.power(z, QRational(2)), z);
    const auto expected_integral = math.multiply({math.rational(1, 3), math.power(z, QRational(3))});
    require(math.equivalent(integral, expected_integral), "exact polynomial integral failed");

    QMathType operator_type{QMathScalar::Complex, QMathSpace::Operator, {2, 2}, {}};
    const auto A = math.symbol("A", operator_type, false);
    const auto B = math.symbol("B", operator_type, false);
    require(!math.equivalent(math.multiply({A, B}), math.multiply({B, A})), "operator order was incorrectly commuted");
    require(math.commutator(A, A)->kind == QMathKind::Zero, "self commutator did not simplify to zero");
    require(math.commutator(A, B)->kind == QMathKind::Commutator, "operator commutator was not retained");
    require(math.multiply({math.rational(2), A, B})->args.back().get() == B.get(),
            "noncommutative operator order was not preserved");

    QMathType state_type{QMathScalar::Complex, QMathSpace::State, {2}, {}};
    const auto psi = math.symbol("psi", state_type, false);
    const auto phi = math.symbol("phi", state_type, false);
    const auto pair = math.tensor_product(psi, phi);
    require(pair->type.space == QMathSpace::State && pair->type.shape == std::vector<std::size_t>({2, 2}),
            "state tensor product type is wrong");
    require(math.apply(A, psi)->type.space == QMathSpace::State, "operator application lost state type");
    rejected = false;
    try { (void)math.multiply({psi, phi}); } catch (const QMathError&) { rejected = true; }
    require(rejected, "ambiguous state multiplication was accepted instead of requiring tensor_product/apply");

    const auto deps = math.dependencies(math.multiply({f, math.apply(A, psi)}));
    require(deps == std::vector<std::string>({"A", "psi", "z"}), "dependency extraction failed");

    require(math.equation(x, y)->type.scalar == QMathScalar::Boolean, "equation did not produce boolean type");

    QMathArena bounded(QMathConfig{2, 8, 8});
    (void)bounded.symbol("a", QMathType::scalar_type());
    (void)bounded.symbol("b", QMathType::scalar_type());
    rejected = false;
    try { (void)bounded.symbol("c", QMathType::scalar_type()); } catch (const QMathError&) { rejected = true; }
    require(rejected, "node limit did not fail closed");

    rejected = false;
    try { (void)(QRational(std::numeric_limits<std::int64_t>::max()) + QRational(1)); }
    catch (const QMathError&) { rejected = true; }
    require(rejected, "rational overflow did not fail closed");

    std::cout << "QMath core tests passed\n";
}

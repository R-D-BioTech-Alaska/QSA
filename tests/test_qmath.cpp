#include "qubit/qmath.hpp"
#include "qubit/qsigned.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

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

    const auto promoted = QRational(std::numeric_limits<std::int64_t>::max()) + QRational(1);
    require(promoted.canonical() == "9223372036854775808", "exact rational did not promote beyond int64");
    require(QInteger(7).is_small() && !QInteger::parse("9223372036854775808").is_small(),
            "small-value arbitrary-precision representation contract changed");
    require(QRational::parse("100000000000000000000/25000000000000000000").canonical() == "4",
            "arbitrary-precision rational normalization failed");

    const QTernaryVector sx{QPolarity::Positive, QPolarity::Negative, QPolarity::Neutral, QPolarity::Positive, QPolarity::Negative};
    const QTernaryVector sy{QPolarity::Positive, QPolarity::Positive, QPolarity::Negative, QPolarity::Neutral, QPolarity::Negative};
    const auto interaction = sx.interaction(sy);
    require(interaction.aligned == 2 && interaction.opposed == 1 && interaction.unresolved == 2,
            "signed ternary interaction classification failed");
    require(interaction.net_alignment().canonical() == "1" && sx.dot(sy).canonical() == "1",
            "signed ternary dot product failed");
    const QTernaryVector product = sx.multiplied(sy);
    require(product.get(0) == QPolarity::Positive && product.get(1) == QPolarity::Negative &&
            product.get(2) == QPolarity::Neutral && product.get(3) == QPolarity::Neutral &&
            product.get(4) == QPolarity::Positive,
            "signed ternary coordinate product failed");
    const std::vector<std::uint8_t> packed = sx.pack_base243();
    require(QTernaryVector::from_base243(packed, sx.size()) == sx, "base-243 ternary round trip failed");
    require(sx.negated().negated() == sx, "ternary negation involution failed");

    QSignedChannels brain(QSignedDomain::BrainState);
    brain.add(QPolarity::Positive, QRational(100));
    brain.add(QPolarity::Negative, QRational(100));
    brain.add(QPolarity::Neutral, QRational(7));
    require(brain.net().is_zero(), "opposed signed structure did not project to zero");
    require(brain.active() == QRational(200) && brain.canceled() == QRational(100) && brain.neutral() == QRational(7),
            "deferred cancellation lost signed or neutral structure");
    brain.resolve_neutral(QPolarity::Positive, QRational(2));
    require(brain.net() == QRational(2) && brain.neutral() == QRational(5),
            "neutral-to-polarized transition failed");

    const QMathType evidence_type = QMathType::scalar_type(QMathScalar::Real);
    const auto hypothesis = math.symbol("hypothesis", evidence_type);
    QSignedExpression evidence(evidence_type, QSignedDomain::BrainState);
    evidence.add(QPolarity::Positive, hypothesis, QRational(9, 10));
    evidence.add(QPolarity::Negative, hypothesis, QRational(9, 10));
    evidence.add(QPolarity::Neutral, hypothesis, QRational(1, 5));
    const auto evidence_projection = evidence.project(math);
    require(evidence_projection.expression->kind == QMathKind::Zero,
            "balanced Brain signed structure did not project to numerical zero");
    require(evidence_projection.receipt.canceled_weight == QRational(9, 10) &&
            evidence_projection.receipt.neutral_weight == QRational(1, 5),
            "Brain projection receipt discarded opposition or unresolved structure");

    const QMathType operator3 = QSignedLocalSpace::operator_type(3);
    const auto G = math.symbol("G", operator3, false);
    const auto H = math.symbol("H", operator3, false);
    QSignedOperatorProgram signed_program(operator3);
    signed_program.add_generator(QPolarity::Positive, G, QRational(3));
    signed_program.add_generator(QPolarity::Negative, G, QRational(2));
    signed_program.add_generator(QPolarity::Positive, H);
    signed_program.add_generator(QPolarity::Neutral, H, QRational(4));
    const auto compiled_signed = signed_program.compile(math);
    require(math.equivalent(compiled_signed.expression, math.add({G, H})),
            "signed quantum generator projection failed");
    require(compiled_signed.receipt.positive_weight == QRational(4) &&
            compiled_signed.receipt.negative_weight == QRational(2) &&
            compiled_signed.receipt.neutral_weight == QRational(4) &&
            compiled_signed.receipt.canceled_weight == QRational(2),
            "signed quantum projection receipt is wrong");

    const QSignedBasis3 basis3 = QSignedLocalSpace::basis3(math, "s");
    require(basis3.negative->type.shape == std::vector<std::size_t>({3}) &&
            basis3.neutral->type.shape == std::vector<std::size_t>({3}) &&
            basis3.positive->type.shape == std::vector<std::size_t>({3}),
            "signed three-state local basis lost dimension three");
    require(signed_program.apply(math, basis3.neutral)->type.shape == std::vector<std::size_t>({3}),
            "signed operator application lost local state dimension");

    const auto state2 = math.symbol("state2", QSignedLocalSpace::state_type(2), false);
    rejected = false;
    try { (void)signed_program.apply(math, state2); } catch (const QMathError&) { rejected = true; }
    require(rejected, "signed operator accepted mismatched local state dimension");

    QSignedExpression order_a(operator3, QSignedDomain::QuantumOperator);
    order_a.add(QPolarity::Positive, G);
    order_a.add(QPolarity::Negative, H);
    QSignedExpression order_b(operator3, QSignedDomain::QuantumOperator);
    order_b.add(QPolarity::Negative, H);
    order_b.add(QPolarity::Positive, G);
    require(order_a.canonical() == order_b.canonical(), "signed expression identity depends on insertion order");

    rejected = false;
    try { brain.add(QPolarity::Positive, QRational(-1)); } catch (const QMathError&) { rejected = true; }
    require(rejected, "signed channel accepted negative magnitude");

    std::cout << "QMath core tests passed\n";
}

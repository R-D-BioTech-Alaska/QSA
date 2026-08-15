#include "qubit/qmath.hpp"
#include "qubit/qsigned.hpp"
#include "qubit/qweyl.hpp"
#include "qubit/qweyl_algebra.hpp"
#include "qubit/qcyclotomic3.hpp"
#include "qubit/qclifford3.hpp"
#include "qubit/qmath_language.hpp"

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

    const QWeylSpace qutrit_space{3U};
    const QWeylOperator X3 = QWeylOperator::local(qutrit_space, 0U, 1, 0);
    const QWeylOperator Z3 = QWeylOperator::local(qutrit_space, 0U, 0, 1);
    require(X3.power(3).identity_exact() && Z3.power(3).identity_exact(),
            "qutrit Weyl generators do not have order three");
    require(X3.commutation_turns(Z3) == QRational(2, 3) && Z3.commutation_turns(X3) == QRational(1, 3),
            "qutrit Weyl commutation phase is wrong");
    require(!X3.commutes_with(Z3), "noncommuting qutrit Weyl generators were marked commuting");
    require(X3.multiplied(Z3).equivalent_up_to_global_phase(Z3.multiplied(X3)),
            "Weyl canonical exponents depend on multiplication order");
    const QWeylOperator XZ3 = X3.multiplied(Z3);
    require(XZ3.multiplied(XZ3.inverse()).identity_exact(), "Weyl inverse failed exact group cancellation");

    const QWeylSpace mixed_space{2U, 3U, 5U};
    const QWeylOperator mixed_x = QWeylOperator::local(mixed_space, 0U, 1, 0);
    const QWeylOperator mixed_z = QWeylOperator::local(mixed_space, 2U, 0, 1);
    require(mixed_x.commutes_with(mixed_z), "disjoint mixed-dimension Weyl operators did not commute");
    require(mixed_space.dense_dimension() == 30U, "mixed local Hilbert dimension is wrong");

    const QTernaryVector ternary_shift{QPolarity::Positive, QPolarity::Negative, QPolarity::Neutral};
    const QTernaryVector ternary_clock{QPolarity::Neutral, QPolarity::Positive, QPolarity::Negative};
    const QWeylOperator ternary_weyl = QWeylOperator::from_ternary(ternary_shift, ternary_clock);
    require(ternary_weyl.ternary_shift() == ternary_shift && ternary_weyl.ternary_clock() == ternary_clock,
            "signed ternary/Weyl exponent bridge failed");

    QSignedWeylCircuit controlled(qutrit_space);
    controlled.append(QPolarity::Positive, X3);
    controlled.append(QPolarity::Neutral, Z3);
    controlled.append(QPolarity::Negative, X3);
    auto controlled_result = controlled.compile();
    require(!controlled_result.receipt.ready && controlled_result.receipt.unresolved_steps == 1U && !controlled_result.value,
            "neutral Brain circuit control was silently treated as executable");
    controlled.resolve(1U, QPolarity::Positive);
    controlled_result = controlled.compile();
    require(controlled_result.receipt.ready && controlled_result.value.has_value(),
            "resolved signed Weyl circuit did not compile");

    QSignedWeylCircuit cancellation(qutrit_space);
    cancellation.append(QPolarity::Positive, X3);
    cancellation.append(QPolarity::Positive, Z3);
    cancellation.append(QPolarity::Negative, Z3);
    cancellation.append(QPolarity::Negative, X3);
    const auto canceled_circuit = cancellation.compile();
    require(canceled_circuit.receipt.ready && canceled_circuit.value->identity_exact(),
            "signed Weyl circuit failed exact inverse cancellation");
    require(canceled_circuit.receipt.reduced_support_terms == 0U && canceled_circuit.receipt.local_cancellations != 0U,
            "Weyl circuit receipt lost exact local cancellation");

    const QWeylSpace qubit_space{2U};
    const QWeylOperator XZ = QWeylOperator::local(qubit_space, 0U, 1, 1);
    const QWeylQubitLowering lowered = lower_weyl_qubits(XZ);
    require(lowered.operations.size() == 1U && lowered.operations[0].code == OperationCode::Y &&
            lowered.global_phase_turns == QRational(3, 4),
            "exact Weyl-to-qubit lowering lost XZ global phase");

    const QWeylMathProjection qmath_weyl = project_weyl_qmath(ternary_weyl, math);
    require(qmath_weyl.expression && qmath_weyl.expression->type.space == QMathSpace::Operator &&
            qmath_weyl.global_phase_turns == ternary_weyl.phase_turns(),
            "Weyl-to-QMath projection lost operator type or exact phase");

    QWeylAlgebra opposed(qutrit_space);
    opposed.add(QPolarity::Positive, X3, QRational(3));
    opposed.add(QPolarity::Negative, X3, QRational(2));
    opposed.add(QPolarity::Neutral, Z3, QRational(4));
    const auto opposed_projection = opposed.project();
    require(!opposed_projection.receipt.ready && opposed_projection.receipt.unresolved_basis_terms == 1U &&
            opposed_projection.receipt.canceled_weight == QRational(2) && opposed_projection.receipt.neutral_weight == QRational(4),
            "Weyl algebra lost opposed or unresolved coefficient structure");

    QWeylAlgebra resolved(qutrit_space);
    resolved.add(QPolarity::Positive, X3, QRational(3));
    resolved.add(QPolarity::Negative, X3, QRational(2));
    resolved.add(QPolarity::Positive, Z3);
    const auto resolved_projection = resolved.project();
    require(resolved_projection.receipt.ready && resolved_projection.terms.size() == 2U &&
            resolved_projection.receipt.canceled_weight == QRational(2),
            "resolved Weyl algebra projection failed exact cancellation");
    require(resolved.commuting_groups().size() == 2U && !resolved.all_commuting(),
            "Weyl algebra grouped noncommuting qutrit generators together");

    QWeylAlgebra cancel_basis(qutrit_space);
    cancel_basis.add(QPolarity::Positive, X3);
    cancel_basis.add(QPolarity::Negative, X3);
    const auto cancel_basis_projection = cancel_basis.project();
    require(cancel_basis_projection.receipt.ready && cancel_basis_projection.terms.empty() &&
            cancel_basis_projection.receipt.canceled_weight == QRational(1),
            "Weyl algebra failed complete signed basis cancellation");

    QWeylAlgebra left(qutrit_space);
    QWeylAlgebra right(qutrit_space);
    left.add(QPolarity::Positive, X3);
    right.add(QPolarity::Positive, Z3);
    const auto commutator_projection = left.commutator(right).project();
    require(commutator_projection.receipt.ready && commutator_projection.terms.size() == 2U,
            "exact Weyl commutator lost phase-distinct terms");

    QWeylAlgebra negative_left(qutrit_space);
    QWeylAlgebra negative_right(qutrit_space);
    negative_left.add(QPolarity::Negative, X3);
    negative_right.add(QPolarity::Negative, Z3);
    const auto double_negative = negative_left.multiplied(negative_right).project();
    require(double_negative.receipt.ready && double_negative.terms.size() == 1U &&
            double_negative.terms.front().coefficient == QRational(1),
            "negative Weyl coefficients did not multiply to positive structure");

    QWeylAlgebra unresolved_left(qutrit_space);
    unresolved_left.add(QPolarity::Neutral, X3);
    const auto unresolved_product = unresolved_left.multiplied(right).project();
    require(!unresolved_product.receipt.ready && unresolved_product.receipt.unresolved_basis_terms == 1U,
            "unresolved Weyl coefficient was silently collapsed during multiplication");

    QWeylAlgebra bounded_algebra(qutrit_space, QWeylAlgebraConfig{1U, 4U});
    bounded_algebra.add(QPolarity::Positive, X3);
    rejected = false;
    try { bounded_algebra.add(QPolarity::Positive, Z3); } catch (const QMathError&) { rejected = true; }
    require(rejected, "Weyl algebra basis-term cap did not fail closed");

    const QCyclotomic3 one3 = QCyclotomic3::root(0);
    const QCyclotomic3 omega3 = QCyclotomic3::root(1);
    const QCyclotomic3 omega3_sq = QCyclotomic3::root(2);
    require((one3 + omega3 + omega3_sq).is_zero(), "qutrit cyclotomic root orbit did not cancel exactly");
    require(omega3 * omega3 * omega3 == one3, "qutrit cyclotomic omega^3 identity failed");
    require(omega3.conjugate() == omega3_sq && omega3.norm() == QRational(1) && omega3.inverse() == omega3_sq,
            "qutrit cyclotomic conjugate, norm, or inverse is wrong");
    require(QCyclotomic3::from_turns(QRational(1, 3)) == omega3 &&
            QCyclotomic3::from_turns(QRational(2, 3)) == omega3_sq,
            "exact third-turn phase conversion failed");
    rejected = false;
    try { (void)QCyclotomic3::from_turns(QRational(1, 6)); } catch (const QMathError&) { rejected = true; }
    require(rejected, "non-third-turn qutrit phase did not fail closed");

    QWeyl3Algebra cyclotomic_x(qutrit_space);
    QWeyl3Algebra cyclotomic_z(qutrit_space);
    cyclotomic_x.add(X3);
    cyclotomic_z.add(Z3);
    const auto cyclotomic_commutator = cyclotomic_x.commutator(cyclotomic_z);
    require(cyclotomic_commutator.term_count() == 1U &&
            cyclotomic_commutator.coefficient(XZ3) == one3 - omega3,
            "qutrit Weyl commutator did not collapse to exact (1-omega) coefficient");

    QWeyl3Algebra phase_orbit(qutrit_space);
    phase_orbit.add(X3);
    phase_orbit.add(QWeylOperator(qutrit_space, QRational(1, 3), X3.exponents()));
    phase_orbit.add(QWeylOperator(qutrit_space, QRational(2, 3), X3.exponents()));
    require(phase_orbit.empty(), "qutrit Weyl phase orbit did not cancel exactly");

    rejected = false;
    try { (void)QWeyl3Algebra::from_projection(qutrit_space, opposed_projection); } catch (const QMathError&) { rejected = true; }
    require(rejected, "unresolved signed Weyl state entered qutrit cyclotomic algebra");
    rejected = false;
    try { (void)QWeyl3Algebra(qubit_space); } catch (const QMathError&) { rejected = true; }
    require(rejected, "qutrit cyclotomic algebra accepted non-qutrit local space");

    const QClifford3Map fourier3 = QClifford3Map::fourier(qutrit_space, 0U);
    require(fourier3.transform(X3) == Z3 && fourier3.transform(Z3) == X3.inverse(),
            "qutrit Fourier Clifford map is not exact on Weyl generators");
    require(fourier3.symplectic().inverse().composed(fourier3.symplectic()).identity_exact(),
            "qutrit Fourier symplectic inverse failed exact identity");

    const QClifford3Map phase3 = QClifford3Map::phase(qutrit_space, 0U);
    require(phase3.transform(X3) == XZ3 && phase3.transform(Z3) == Z3,
            "qutrit phase Clifford map is not exact on Weyl generators");

    QClifford3Map fourier_power = QClifford3Map::identity(qutrit_space);
    for (std::size_t i = 0U; i < 4U; ++i) fourier_power = fourier3.composed(fourier_power);
    require(fourier_power.transform(X3) == X3 && fourier_power.transform(Z3) == Z3,
            "four qutrit Fourier maps did not return the exact generator map to identity");

    const QWeylSpace qutrit_pair{3U, 3U};
    const QWeylOperator Xc = QWeylOperator::local(qutrit_pair, 0U, 1, 0);
    const QWeylOperator Xt = QWeylOperator::local(qutrit_pair, 1U, 1, 0);
    const QWeylOperator Zc = QWeylOperator::local(qutrit_pair, 0U, 0, 1);
    const QWeylOperator Zt = QWeylOperator::local(qutrit_pair, 1U, 0, 1);
    const QClifford3Map sum3 = QClifford3Map::sum(qutrit_pair, 0U, 1U);
    require(sum3.transform(Xc) == Xc.multiplied(Xt) && sum3.transform(Xt) == Xt &&
            sum3.transform(Zc) == Zc && sum3.transform(Zt) == Zc.inverse().multiplied(Zt),
            "qutrit SUM Clifford map is not exact on two-site Weyl generators");

    QClifford3Program clifford_program(qutrit_pair);
    clifford_program.append_fourier(0U);
    clifford_program.append_phase(1U);
    clifford_program.append_sum(0U, 1U);
    QClifford3CompileReceipt clifford_receipt;
    const QClifford3Map compiled_clifford = clifford_program.compile(&clifford_receipt);
    require(clifford_receipt.ready && clifford_receipt.exact && clifford_receipt.steps == 3U &&
            clifford_receipt.generator_images == 4U,
            "qutrit Clifford program receipt lost exact compilation state");
    const QClifford3Map inverse_clifford = compiled_clifford.inverse();
    const QClifford3Map clifford_identity = inverse_clifford.composed(compiled_clifford);
    require(clifford_identity.transform(Xc) == Xc && clifford_identity.transform(Xt) == Xt &&
            clifford_identity.transform(Zc) == Zc && clifford_identity.transform(Zt) == Zt,
            "qutrit Clifford inverse did not restore canonical generators exactly");

    QWeyl3Algebra clifford_algebra(qutrit_space);
    clifford_algebra.add(X3, omega3);
    const QWeyl3Algebra fourier_algebra = fourier3.transform(clifford_algebra);
    require(fourier_algebra.term_count() == 1U && fourier_algebra.coefficient(Z3) == omega3,
            "qutrit Clifford transform lost exact cyclotomic coefficient");

    rejected = false;
    try { (void)QClifford3Map::identity(qubit_space); } catch (const QMathError&) { rejected = true; }
    require(rejected, "qutrit Clifford map accepted non-qutrit local space");
    rejected = false;
    try { (void)QClifford3Map::sum(qutrit_pair, 0U, 0U); } catch (const QMathError&) { rejected = true; }
    require(rejected, "qutrit SUM accepted identical control and target");
    rejected = false;
    try { (void)QSymplectic3(1U, std::vector<std::uint8_t>{1U, 0U, 0U, 0U}); } catch (const QMathError&) { rejected = true; }
    require(rejected, "nonsymplectic qutrit matrix was accepted");
    rejected = false;
    try { (void)QSymplectic3::identity(2U, QSymplectic3Config{4U}); } catch (const QMathError&) { rejected = true; }
    require(rejected, "qutrit symplectic resource cap did not fail closed");
    QClifford3Program bounded_clifford(qutrit_space, QClifford3Config{1U, QSymplectic3Config{16U}});
    bounded_clifford.append_fourier(0U);
    rejected = false;
    try { bounded_clifford.append_phase(0U); } catch (const QMathError&) { rejected = true; }
    require(rejected, "qutrit Clifford step cap did not fail closed");

    const QMathLanguageCompilation symbolic_language = QMathLanguageCompiler::compile(f, math);
    require(symbolic_language.receipt.route == QMathLanguageRoute::Symbolic &&
            symbolic_language.receipt.type.has_value() &&
            *symbolic_language.receipt.type == f->type &&
            symbolic_language.receipt.dependencies == std::vector<std::string>({"z"}) &&
            symbolic_language.receipt.canonical == f->canonical &&
            symbolic_language.receipt.exact && symbolic_language.receipt.symbolic_ready &&
            !symbolic_language.receipt.native_lowering_ready &&
            symbolic_language.receipt.evidence.fidelity == QMathFidelity::ExactStructural &&
            symbolic_language.receipt.evidence.exact_structure() &&
            !symbolic_language.receipt.evidence.exact_math() &&
            !symbolic_language.exact_scalar.has_value(),
            "QMath language lost symbolic identity, type, dependency structure, or fidelity");

    const auto closed_scalar = math.add({
        math.rational(QRational(7, 3)),
        math.multiply({math.rational(QRational(5, 2)), math.rational(6)}),
    });
    const QMathLanguageCompilation exact_scalar_language = QMathLanguageCompiler::compile(closed_scalar, math);
    require(exact_scalar_language.receipt.evidence.fidelity == QMathFidelity::ExactAlgebraic &&
            exact_scalar_language.receipt.evidence.exact_math() &&
            exact_scalar_language.exact_scalar.has_value() &&
            *exact_scalar_language.exact_scalar == QRational(52, 3),
            "QMath language failed exact closed-rational execution or fidelity classification");
    require(QMathLanguageCompiler::compare_exact(*exact_scalar_language.exact_scalar, QRational(17), QMathComparison::Greater) &&
            QMathLanguageCompiler::compare_exact(QRational(17), QRational(17), QMathComparison::Equal) &&
            QMathLanguageCompiler::compare_exact(QRational(-3), QRational(0), QMathComparison::Less) &&
            !QMathLanguageCompiler::compare_exact(QRational(5), QRational(6), QMathComparison::GreaterEqual),
            "QMath exact rational comparison execution failed");

    const QMathEvidence bounded_numeric = QMathEvidence::bounded_numerical(1.0e-9, 1.0e-8);
    require(bounded_numeric.fidelity == QMathFidelity::BoundedNumerical &&
            bounded_numeric.bounded_numeric() && !bounded_numeric.exact_structure() &&
            bounded_numeric.error_bounds.has_value() &&
            bounded_numeric.error_bounds->absolute == 1.0e-9 &&
            bounded_numeric.error_bounds->relative == 1.0e-8,
            "QMath bounded numerical fidelity receipt is wrong");
    rejected = false;
    try { (void)QMathEvidence::bounded_numerical(-1.0, 0.0); } catch (const QMathError&) { rejected = true; }
    require(rejected, "QMath bounded numerical evidence accepted a negative error bound");
    rejected = false;
    try {
        (void)QMathEvidence::bounded_numerical(std::numeric_limits<double>::infinity(), 0.0);
    } catch (const QMathError&) {
        rejected = true;
    }
    require(rejected, "QMath bounded numerical evidence accepted a nonfinite error bound");

    QStrictOrder strict_order;
    const auto order_a_id = strict_order.add_symbol("event:A");
    const auto order_b_id = strict_order.add_symbol("event:B");
    const auto order_c_id = strict_order.add_symbol("event:C");
    const auto order_d_id = strict_order.add_symbol("event:D");
    require(strict_order.add_before(order_a_id, order_b_id) &&
            strict_order.add_before(order_a_id, order_c_id) &&
            strict_order.add_before(order_b_id, order_d_id) &&
            strict_order.add_before(order_c_id, order_d_id),
            "strict-order direct relation admission failed");
    require(strict_order.relation(order_a_id, order_d_id) == QOrderRelation::Before &&
            strict_order.relation(order_d_id, order_a_id) == QOrderRelation::After &&
            strict_order.relation(order_b_id, order_c_id) == QOrderRelation::Unresolved &&
            strict_order.relation(order_a_id, order_a_id) == QOrderRelation::Same,
            "strict-order exact closure relation is wrong");
    require(strict_order.unique_minimum() == order_a_id && strict_order.unique_maximum() == order_d_id,
            "strict-order unique extrema are wrong");
    const QStrictOrderReceipt strict_receipt = strict_order.receipt();
    require(strict_receipt.symbols == 4U && strict_receipt.direct_relations == 4U &&
            strict_receipt.closure_relations == 5U && strict_receipt.inferred_relations == 1U &&
            strict_receipt.minima == 1U && strict_receipt.maxima == 1U &&
            strict_receipt.exact && strict_receipt.acyclic,
            "strict-order receipt lost exact closure structure");
    require(!strict_order.add_before(order_a_id, order_d_id) && strict_order.direct_relation_count() == 4U,
            "strict-order admitted a redundant transitive relation");

    QStrictOrder strict_order_reordered;
    const auto reordered_d = strict_order_reordered.add_symbol("event:D");
    const auto reordered_c = strict_order_reordered.add_symbol("event:C");
    const auto reordered_b = strict_order_reordered.add_symbol("event:B");
    const auto reordered_a = strict_order_reordered.add_symbol("event:A");
    require(strict_order_reordered.add_before(reordered_c, reordered_d) &&
            strict_order_reordered.add_before(reordered_a, reordered_c) &&
            strict_order_reordered.add_before(reordered_b, reordered_d) &&
            strict_order_reordered.add_before(reordered_a, reordered_b),
            "reordered strict-order fixture admission failed");
    require(strict_order_reordered.canonical() == strict_order.canonical(),
            "strict-order canonical identity depends on insertion order");

    const std::string before_cycle = strict_order.canonical();
    rejected = false;
    try { (void)strict_order.add_before(order_d_id, order_a_id); } catch (const QMathError&) { rejected = true; }
    require(rejected && strict_order.canonical() == before_cycle,
            "strict-order cycle rejection mutated accepted state");

    QStrictOrder ambiguous_order;
    const auto ambiguous_a = ambiguous_order.add_symbol("left:A");
    const auto ambiguous_b = ambiguous_order.add_symbol("left:B");
    const auto ambiguous_c = ambiguous_order.add_symbol("right:C");
    const auto ambiguous_d = ambiguous_order.add_symbol("right:D");
    require(ambiguous_order.add_before(ambiguous_a, ambiguous_b) &&
            ambiguous_order.add_before(ambiguous_c, ambiguous_d) &&
            !ambiguous_order.unique_minimum().has_value() &&
            !ambiguous_order.unique_maximum().has_value(),
            "strict-order ambiguity did not remain explicit");

    QStrictOrder capped_order(QStrictOrderConfig{2U, 2U, 8U});
    (void)capped_order.add_symbol("cap:A");
    (void)capped_order.add_symbol("cap:B");
    rejected = false;
    try { (void)capped_order.add_symbol("cap:C"); } catch (const QMathError&) { rejected = true; }
    require(rejected, "strict-order symbol cap did not fail closed");

    const QMathLanguageCompilation order_language = QMathLanguageCompiler::compile(strict_order);
    require(order_language.receipt.route == QMathLanguageRoute::StrictOrder &&
            order_language.receipt.evidence.fidelity == QMathFidelity::ExactStructural &&
            order_language.receipt.evidence.exact_structure() && !order_language.receipt.evidence.exact_math() &&
            order_language.receipt.sites == 4U && order_language.receipt.source_terms == 4U &&
            order_language.receipt.support_terms == 5U && order_language.receipt.transform_ready &&
            order_language.strict_order.has_value() &&
            order_language.strict_order->canonical == strict_order.canonical() &&
            order_language.receipt.dependencies == std::vector<std::string>({"event:A", "event:B", "event:C", "event:D"}) &&
            std::string(qmath_language_route_name(order_language.receipt.route)) == "StrictOrder",
            "QMath language lost strict-order identity, dependencies, or exact structural evidence");

    QAffineRelationSpace affine(3U, length);
    const auto affine_a = affine.add_symbol("marker:A");
    const auto affine_b = affine.add_symbol("marker:B");
    const auto affine_c = affine.add_symbol("marker:C");
    const auto affine_d = affine.add_symbol("marker:D");
    const QAffineRelationSpace::Displacement east{QRational(1), QRational(0), QRational(0)};
    const QAffineRelationSpace::Displacement north{QRational(0), QRational(1), QRational(0)};
    const QAffineRelationSpace::Displacement east_north{QRational(1), QRational(1), QRational(0)};
    const QAffineRelationSpace::Displacement southwest{QRational(-1), QRational(-1), QRational(0)};
    require(affine.add_difference(affine_a, affine_b, east) &&
            affine.add_difference(affine_b, affine_c, north),
            "affine exact relation admission failed");
    require(affine.displacement(affine_a, affine_c) == east_north &&
            affine.displacement(affine_c, affine_a) == southwest &&
            !affine.displacement(affine_a, affine_d).has_value(),
            "affine exact composition, inversion, or disconnected-state handling failed");
    require(!affine.add_difference(affine_a, affine_c, east_north) &&
            affine.independent_constraint_count() == 2U,
            "affine redundant exact constraint changed independent state");

    const std::string affine_before_conflict = affine.canonical();
    rejected = false;
    try { (void)affine.add_difference(affine_a, affine_c, east); } catch (const QMathError&) { rejected = true; }
    require(rejected && affine.canonical() == affine_before_conflict,
            "affine contradiction rejection mutated accepted state");

    QAffineRelationSpace affine_reordered(3U, length);
    const auto affine_reordered_c = affine_reordered.add_symbol("marker:C");
    const auto affine_reordered_d = affine_reordered.add_symbol("marker:D");
    const auto affine_reordered_b = affine_reordered.add_symbol("marker:B");
    const auto affine_reordered_a = affine_reordered.add_symbol("marker:A");
    require(affine_reordered.add_difference(affine_reordered_b, affine_reordered_c, north) &&
            affine_reordered.add_difference(affine_reordered_a, affine_reordered_b, east) &&
            affine_reordered.canonical() == affine.canonical() &&
            !affine_reordered.displacement(affine_reordered_a, affine_reordered_d).has_value(),
            "affine canonical identity depends on symbol or constraint insertion order");

    QAffineRelationSpace fractional_affine(2U);
    const auto fractional_a = fractional_affine.add_symbol("fraction:A");
    const auto fractional_b = fractional_affine.add_symbol("fraction:B");
    const auto fractional_c = fractional_affine.add_symbol("fraction:C");
    const QAffineRelationSpace::Displacement half_x{QRational(1, 2), QRational(0)};
    const QAffineRelationSpace::Displacement third_y{QRational(0), QRational(1, 3)};
    const QAffineRelationSpace::Displacement composed_fraction{QRational(1, 2), QRational(1, 3)};
    require(fractional_affine.add_difference(fractional_a, fractional_b, half_x) &&
            fractional_affine.add_difference(fractional_b, fractional_c, third_y) &&
            fractional_affine.displacement(fractional_a, fractional_c) == composed_fraction,
            "affine relation lost exact rational displacement arithmetic");

    rejected = false;
    try {
        const QAffineRelationSpace::Displacement wrong_dimension{QRational(1), QRational(2)};
        (void)affine.add_difference(affine_a, affine_b, wrong_dimension);
    } catch (const QMathError&) {
        rejected = true;
    }
    require(rejected, "affine relation accepted a displacement with the wrong dimension");

    QAffineRelationSpace capped_affine(1U, {}, QAffineRelationConfig{2U, 1U, 1U});
    (void)capped_affine.add_symbol("cap-affine:A");
    (void)capped_affine.add_symbol("cap-affine:B");
    rejected = false;
    try { (void)capped_affine.add_symbol("cap-affine:C"); } catch (const QMathError&) { rejected = true; }
    require(rejected, "affine relation symbol cap did not fail closed");

    const QAffineRelationReceipt affine_receipt = affine.receipt();
    require(affine_receipt.dimensions == 3U && affine_receipt.symbols == 4U &&
            affine_receipt.independent_constraints == 2U && affine_receipt.components == 2U &&
            affine_receipt.coordinate_dimension == length && affine_receipt.exact && affine_receipt.consistent,
            "affine relation receipt lost dimensions, component state, or exactness");
    const QMathLanguageCompilation affine_language = QMathLanguageCompiler::compile(affine);
    require(affine_language.receipt.route == QMathLanguageRoute::AffineRelation &&
            affine_language.receipt.evidence.fidelity == QMathFidelity::ExactAlgebraic &&
            affine_language.receipt.evidence.exact_math() && affine_language.receipt.type.has_value() &&
            affine_language.receipt.type->space == QMathSpace::Vector &&
            affine_language.receipt.type->scalar == QMathScalar::Rational &&
            affine_language.receipt.type->shape == std::vector<std::size_t>({3U}) &&
            affine_language.receipt.type->dimension == length &&
            affine_language.receipt.sites == 4U && affine_language.receipt.source_terms == 2U &&
            affine_language.receipt.support_terms == 2U && affine_language.receipt.transform_ready &&
            affine_language.affine_relation.has_value() &&
            affine_language.affine_relation->canonical == affine.canonical() &&
            affine_language.receipt.dependencies == std::vector<std::string>({"marker:A", "marker:B", "marker:C", "marker:D"}) &&
            std::string(qmath_language_route_name(affine_language.receipt.route)) == "AffineRelation",
            "QMath language lost affine exact rational identity, physical type, or evidence");

    const QMathLanguageCompilation qubit_language = QMathLanguageCompiler::compile(XZ, math);
    require(qubit_language.receipt.route == QMathLanguageRoute::QubitWeylNative &&
            qubit_language.receipt.local_dimensions == std::vector<std::uint32_t>({2U}) &&
            qubit_language.receipt.native_lowering_ready && qubit_language.receipt.symbolic_ready &&
            qubit_language.qubit_lowering.has_value() &&
            qubit_language.qubit_lowering->operations.size() == 1U &&
            qubit_language.qubit_lowering->operations.front().code == OperationCode::Y &&
            qubit_language.receipt.global_phase_turns == QRational(3, 4) &&
            qubit_language.receipt.evidence.exact_math() &&
            qubit_language.receipt.evidence.phase_arithmetic_exact,
            "QMath language qubit route lost native lowering, exact phase, or fidelity receipt");

    const QWeylOperator mixed_weyl = mixed_x.multiplied(mixed_z);
    const QMathLanguageCompilation mixed_language = QMathLanguageCompiler::compile(mixed_weyl, math);
    require(mixed_language.receipt.route == QMathLanguageRoute::WeylStructural &&
            mixed_language.receipt.local_dimensions == std::vector<std::uint32_t>({2U, 3U, 5U}) &&
            mixed_language.receipt.symbolic_ready && !mixed_language.receipt.native_lowering_ready &&
            !mixed_language.qutrit_algebra.has_value() && mixed_language.receipt.evidence.exact_math(),
            "QMath language silently forced mixed-dimensional Weyl structure into a local backend or lost exact evidence");

    const QMathLanguageCompilation qutrit_language = QMathLanguageCompiler::compile(X3, math);
    require(qutrit_language.receipt.route == QMathLanguageRoute::QutritCyclotomicWeyl &&
            qutrit_language.qutrit_algebra.has_value() &&
            qutrit_language.qutrit_algebra->term_count() == 1U &&
            qutrit_language.receipt.transform_ready && !qutrit_language.receipt.native_lowering_ready &&
            qutrit_language.receipt.evidence.exact_math() && qutrit_language.receipt.evidence.phase_arithmetic_exact,
            "QMath language did not route exact qutrit Weyl structure through cyclotomic algebra with exact evidence");

    const QWeylOperator sixth_phase(qutrit_space, QRational(1, 6), X3.exponents());
    const QMathLanguageCompilation sixth_language = QMathLanguageCompiler::compile(sixth_phase, math);
    require(sixth_language.receipt.route == QMathLanguageRoute::WeylStructural &&
            sixth_language.receipt.global_phase_turns == QRational(1, 6) &&
            sixth_language.receipt.symbolic_ready && !sixth_language.qutrit_algebra.has_value() &&
            sixth_language.receipt.evidence.exact_math(),
            "QMath language silently coerced a non-cyclotomic qutrit phase or lost exact structural evidence");

    const QMathLanguageCompilation algebra_language = QMathLanguageCompiler::compile(cyclotomic_commutator, math);
    require(algebra_language.receipt.route == QMathLanguageRoute::QutritCyclotomicWeyl &&
            algebra_language.receipt.source_terms == cyclotomic_commutator.term_count() &&
            algebra_language.receipt.type.has_value() &&
            algebra_language.receipt.type->space == QMathSpace::Operator &&
            algebra_language.qutrit_algebra.has_value() &&
            algebra_language.qutrit_algebra->canonical() == cyclotomic_commutator.canonical() &&
            algebra_language.receipt.evidence.exact_math(),
            "QMath language did not preserve exact qutrit algebra identity or evidence");

    const QMathLanguageCompilation clifford_language = QMathLanguageCompiler::compile(fourier3, math);
    require(clifford_language.receipt.route == QMathLanguageRoute::QutritClifford &&
            clifford_language.receipt.generator_images == 2U &&
            clifford_language.receipt.transform_ready && clifford_language.qutrit_clifford.has_value() &&
            clifford_language.qutrit_clifford->canonical() == fourier3.canonical() &&
            clifford_language.receipt.evidence.exact_math(),
            "QMath language did not preserve exact qutrit Clifford transform identity or evidence");

    const QMathLanguageCompilation program_language = QMathLanguageCompiler::compile(clifford_program, math);
    const QMathLanguageCompilation program_language_repeat = QMathLanguageCompiler::compile(clifford_program, math);
    require(program_language.receipt.route == QMathLanguageRoute::QutritClifford &&
            program_language.receipt.program_steps == 3U && program_language.receipt.generator_images == 4U &&
            program_language.qutrit_clifford.has_value() &&
            program_language.receipt.canonical == compiled_clifford.canonical() &&
            program_language_repeat.receipt.canonical == program_language.receipt.canonical &&
            program_language.receipt.evidence.exact_math() &&
            std::string(qmath_language_route_name(program_language.receipt.route)) == "QutritClifford" &&
            std::string(qmath_fidelity_name(program_language.receipt.evidence.fidelity)) == "ExactAlgebraic",
            "QMath language program compilation is not deterministic or lost route/fidelity identity");

    std::cout << "QMath core tests passed\n";
}

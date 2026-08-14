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
            !symbolic_language.receipt.native_lowering_ready,
            "QMath language lost symbolic identity, type, or dependency structure");

    const QMathLanguageCompilation qubit_language = QMathLanguageCompiler::compile(XZ, math);
    require(qubit_language.receipt.route == QMathLanguageRoute::QubitWeylNative &&
            qubit_language.receipt.local_dimensions == std::vector<std::uint32_t>({2U}) &&
            qubit_language.receipt.native_lowering_ready && qubit_language.receipt.symbolic_ready &&
            qubit_language.qubit_lowering.has_value() &&
            qubit_language.qubit_lowering->operations.size() == 1U &&
            qubit_language.qubit_lowering->operations.front().code == OperationCode::Y &&
            qubit_language.receipt.global_phase_turns == QRational(3, 4),
            "QMath language qubit route lost native lowering or exact phase receipt");

    const QWeylOperator mixed_weyl = mixed_x.multiplied(mixed_z);
    const QMathLanguageCompilation mixed_language = QMathLanguageCompiler::compile(mixed_weyl, math);
    require(mixed_language.receipt.route == QMathLanguageRoute::WeylStructural &&
            mixed_language.receipt.local_dimensions == std::vector<std::uint32_t>({2U, 3U, 5U}) &&
            mixed_language.receipt.symbolic_ready && !mixed_language.receipt.native_lowering_ready &&
            !mixed_language.qutrit_algebra.has_value(),
            "QMath language silently forced mixed-dimensional Weyl structure into a local backend");

    const QMathLanguageCompilation qutrit_language = QMathLanguageCompiler::compile(X3, math);
    require(qutrit_language.receipt.route == QMathLanguageRoute::QutritCyclotomicWeyl &&
            qutrit_language.qutrit_algebra.has_value() &&
            qutrit_language.qutrit_algebra->term_count() == 1U &&
            qutrit_language.receipt.transform_ready && !qutrit_language.receipt.native_lowering_ready,
            "QMath language did not route exact qutrit Weyl structure through cyclotomic algebra");

    const QWeylOperator sixth_phase(qutrit_space, QRational(1, 6), X3.exponents());
    const QMathLanguageCompilation sixth_language = QMathLanguageCompiler::compile(sixth_phase, math);
    require(sixth_language.receipt.route == QMathLanguageRoute::WeylStructural &&
            sixth_language.receipt.global_phase_turns == QRational(1, 6) &&
            sixth_language.receipt.symbolic_ready && !sixth_language.qutrit_algebra.has_value(),
            "QMath language silently coerced a non-cyclotomic qutrit phase");

    const QMathLanguageCompilation algebra_language = QMathLanguageCompiler::compile(cyclotomic_commutator, math);
    require(algebra_language.receipt.route == QMathLanguageRoute::QutritCyclotomicWeyl &&
            algebra_language.receipt.source_terms == cyclotomic_commutator.term_count() &&
            algebra_language.receipt.type.has_value() &&
            algebra_language.receipt.type->space == QMathSpace::Operator &&
            algebra_language.qutrit_algebra.has_value() &&
            algebra_language.qutrit_algebra->canonical() == cyclotomic_commutator.canonical(),
            "QMath language did not preserve exact qutrit algebra identity");

    const QMathLanguageCompilation clifford_language = QMathLanguageCompiler::compile(fourier3, math);
    require(clifford_language.receipt.route == QMathLanguageRoute::QutritClifford &&
            clifford_language.receipt.generator_images == 2U &&
            clifford_language.receipt.transform_ready && clifford_language.qutrit_clifford.has_value() &&
            clifford_language.qutrit_clifford->canonical() == fourier3.canonical(),
            "QMath language did not preserve exact qutrit Clifford transform identity");

    const QMathLanguageCompilation program_language = QMathLanguageCompiler::compile(clifford_program, math);
    const QMathLanguageCompilation program_language_repeat = QMathLanguageCompiler::compile(clifford_program, math);
    require(program_language.receipt.route == QMathLanguageRoute::QutritClifford &&
            program_language.receipt.program_steps == 3U && program_language.receipt.generator_images == 4U &&
            program_language.qutrit_clifford.has_value() &&
            program_language.receipt.canonical == compiled_clifford.canonical() &&
            program_language_repeat.receipt.canonical == program_language.receipt.canonical &&
            std::string(qmath_language_route_name(program_language.receipt.route)) == "QutritClifford",
            "QMath language program compilation is not deterministic or lost route identity");

    std::cout << "QMath core tests passed\n";
}

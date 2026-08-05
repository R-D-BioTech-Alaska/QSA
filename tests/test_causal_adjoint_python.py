from __future__ import annotations

import math

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalRegister, CausalRuntimeError
from qsa.causal_adjoint import (
    CausalComponentWeightedAdjoint,
    CausalWeightedAdjoint,
)
from qsa.causal_support import CausalPauliSupportPlan
from qsa.causal_support_grad import CausalSupportParameterShift


QUBITS = 10_000
FIRST = 17
SECOND = 4_321
THIRD = 9_999
EXTRA = 611


def tripair_plan() -> CausalParameterizedPlan:
    operations = []
    for qubit, name in (
        (FIRST, "first"),
        (SECOND, "second"),
        (THIRD, "third"),
    ):
        operations.append(("ry", qubit, Parameter(f"semantic_{name}_theta")))
        operations.append(("rz", qubit, Parameter(f"semantic_{name}_phi")))
    operations.extend(
        (
            ("cnot", FIRST, SECOND),
            ("cnot", SECOND, THIRD),
            ("cnot", THIRD, FIRST),
        )
    )
    for layer in range(2):
        for qubit, name in (
            (FIRST, "first"),
            (SECOND, "second"),
            (THIRD, "third"),
        ):
            operations.append(
                ("ry", qubit, Parameter(f"layer_{layer}_{name}_first"))
            )
            operations.append(
                ("rz", qubit, Parameter(f"layer_{layer}_{name}_second"))
            )
        operations.extend(
            (("cnot", FIRST, SECOND), ("cnot", SECOND, THIRD))
        )
    assert len(operations) == 25
    return CausalParameterizedPlan(operations)


def brain_support() -> CausalPauliSupportPlan:
    return CausalPauliSupportPlan(
        QUBITS,
        (
            ((FIRST, "X"),),
            ((FIRST, "Y"),),
            ((FIRST, "Z"),),
            ((SECOND, "X"),),
            ((SECOND, "Y"),),
            ((SECOND, "Z"),),
            ((THIRD, "X"),),
            ((THIRD, "Y"),),
            ((THIRD, "Z"),),
            ((FIRST, "X"), (SECOND, "X")),
            ((FIRST, "Z"), (SECOND, "Z")),
            ((SECOND, "X"), (THIRD, "X")),
            ((SECOND, "Z"), (THIRD, "Z")),
            ((FIRST, "X"), (SECOND, "X"), (THIRD, "X")),
        ),
    )


def maximum_error(first, second) -> float:
    return max(abs(float(left) - float(right)) for left, right in zip(first, second))


def weighted_objective(
    root: CausalRegister,
    plan: CausalParameterizedPlan,
    support: CausalPauliSupportPlan,
    values,
    cotangent,
) -> tuple[tuple[float, ...], float]:
    with root.fork() as branch:
        branch.apply(plan, values)
        observations = support.execute(branch)
    objective = sum(
        float(weight) * float(value)
        for weight, value in zip(cotangent, observations)
    )
    return observations, objective


def test_component_adjoint_matches_parameter_shift() -> None:
    plan = tripair_plan()
    support = brain_support()
    prepare = CausalParameterizedPlan((("h", EXTRA), ("cnot", EXTRA, FIRST)))
    parameter_shift = CausalSupportParameterShift(plan, support)
    adjoint = CausalComponentWeightedAdjoint(
        plan,
        support,
        max_local_qubits=8,
    )
    values = tuple(
        0.21 * math.sin((index + 1) * 0.29)
        for index in range(plan.parameter_count)
    )
    cotangent = tuple(
        0.13 * math.cos((index + 1) * 0.47)
        for index in range(support.observable_count)
    )

    with CausalRegister(QUBITS) as root:
        root.apply(prepare, ())
        initial = root.encode_qsc()
        shifted = parameter_shift.evaluate_and_jacobian(root, values, workers=0)
        result = adjoint.evaluate(root, values, cotangent)

        assert support.term_count == 20
        assert result.local_qubit_count == 4
        assert set(result.global_qubits) == {EXTRA, FIRST, SECOND, THIRD}
        assert result.forward_sweeps == 1
        assert result.reverse_sweeps == 1
        assert result.shifted_evaluations == 0
        assert maximum_error(result.values, shifted.values) <= 2.0e-12
        assert maximum_error(result.gradient, shifted.vjp(cotangent)) <= 2.0e-12
        assert root.encode_qsc() == initial
        assert root.validate()

    adjoint.close()
    prepare.close()
    support.close()
    plan.close()


def test_batch_component_adjoint_matches_scalar() -> None:
    plan = tripair_plan()
    support = brain_support()
    prepare = CausalParameterizedPlan((("h", EXTRA), ("cnot", EXTRA, FIRST)))
    adjoint = CausalComponentWeightedAdjoint(
        plan,
        support,
        max_local_qubits=8,
    )
    rows = tuple(
        tuple(
            0.19 * math.sin((row + 1) * (index + 1) * 0.17)
            for index in range(plan.parameter_count)
        )
        for row in range(7)
    )
    cotangents = tuple(
        tuple(
            0.11 * math.cos((row + 2) * (index + 1) * 0.23)
            for index in range(support.observable_count)
        )
        for row in range(7)
    )

    with CausalRegister(QUBITS) as root:
        root.apply(prepare, ())
        initial = root.encode_qsc()
        batch = adjoint.evaluate_many(root, rows, cotangents)
        scalar = tuple(
            adjoint.evaluate(root, row, cotangent)
            for row, cotangent in zip(rows, cotangents)
        )

        assert batch.row_count == len(rows)
        assert batch.local_qubit_count == 4
        assert set(batch.global_qubits) == {EXTRA, FIRST, SECOND, THIRD}
        assert batch.forward_sweeps == len(rows)
        assert batch.reverse_sweeps == len(rows)
        assert batch.shifted_evaluations == 0
        assert batch.stored_forward_states == 0
        assert batch.native_calls == 1
        for batch_values, batch_gradient, scalar_result in zip(
            batch.values,
            batch.gradients,
            scalar,
        ):
            assert maximum_error(batch_values, scalar_result.values) <= 2.0e-12
            assert maximum_error(batch_gradient, scalar_result.gradient) <= 2.0e-12
        assert root.encode_qsc() == initial
        assert root.validate()

        try:
            adjoint.evaluate_many(root, rows, cotangents[:-1])
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched batch rows were accepted")

    adjoint.close()
    prepare.close()
    support.close()
    plan.close()


def test_repeated_parameter_accumulates_exactly() -> None:
    shared = Parameter("shared")
    plan = CausalParameterizedPlan(
        (
            ("ry", 0, shared),
            ("cnot", 0, 1),
            ("rz", 1, shared),
            ("rx", 0, Parameter("independent")),
        )
    )
    support = CausalPauliSupportPlan(
        2,
        (
            ((0, "X"),),
            ((1, "Z"),),
            ((0, "X"), (1, "X")),
        ),
    )
    adjoint = CausalWeightedAdjoint(plan, support, max_qubits=4)
    values = {"shared": 0.37, "independent": -0.29}
    cotangent = (0.41, -0.23, 0.17)
    epsilon = 1.0e-6

    with CausalRegister(2) as root:
        initial = root.encode_qsc()
        result = adjoint.evaluate(root, values, cotangent)
        observations, _objective = weighted_objective(
            root,
            plan,
            support,
            values,
            cotangent,
        )
        finite_difference = []
        for name in result.parameter_names:
            positive = dict(values)
            negative = dict(values)
            positive[name] += epsilon
            negative[name] -= epsilon
            _positive_values, positive_objective = weighted_objective(
                root,
                plan,
                support,
                positive,
                cotangent,
            )
            _negative_values, negative_objective = weighted_objective(
                root,
                plan,
                support,
                negative,
                cotangent,
            )
            finite_difference.append(
                (positive_objective - negative_objective) / (2.0 * epsilon)
            )

        assert result.parameter_names == ("shared", "independent")
        assert maximum_error(result.values, observations) <= 2.0e-12
        assert maximum_error(result.gradient, finite_difference) <= 2.0e-7
        assert root.encode_qsc() == initial

    support.close()
    plan.close()


def test_noise_and_oversized_components_fail_closed() -> None:
    noisy_plan = CausalParameterizedPlan(
        (("amplitude_damping_trajectory", 0, Parameter("gamma"), 0.25),)
    )
    noisy_support = CausalPauliSupportPlan(1, (((0, "Z"),),))
    noisy_adjoint = CausalWeightedAdjoint(noisy_plan, noisy_support, max_qubits=4)
    with CausalRegister(1) as root:
        try:
            noisy_adjoint.evaluate(root, {"gamma": 0.1}, (1.0,))
        except CausalRuntimeError:
            pass
        else:
            raise AssertionError("trajectory noise was accepted by the adjoint")
    noisy_support.close()
    noisy_plan.close()

    wide_qubits = tuple(range(100, 125))
    wide_plan = CausalParameterizedPlan(
        (("rz", wide_qubits[0], Parameter("theta")),)
    )
    wide_support = CausalPauliSupportPlan(
        QUBITS,
        (((wide_qubits[0], "Z"),),),
    )
    wide_prepare = CausalParameterizedPlan(
        (("h", wide_qubits[0]),)
        + tuple(
            ("cnot", wide_qubits[index - 1], wide_qubits[index])
            for index in range(1, len(wide_qubits))
        )
    )
    wide_adjoint = CausalComponentWeightedAdjoint(
        wide_plan,
        wide_support,
        max_local_qubits=24,
    )
    with CausalRegister(QUBITS) as root:
        root.apply(wide_prepare, ())
        try:
            wide_adjoint.evaluate(root, {"theta": 0.1}, (1.0,))
        except CausalRuntimeError:
            pass
        else:
            raise AssertionError("oversized component adjoint was accepted")

    wide_adjoint.close()
    wide_prepare.close()
    wide_support.close()
    wide_plan.close()


def main() -> None:
    test_component_adjoint_matches_parameter_shift()
    test_batch_component_adjoint_matches_scalar()
    test_repeated_parameter_accumulates_exactly()
    test_noise_and_oversized_components_fail_closed()
    print("QSA weighted adjoint tests passed.")


if __name__ == "__main__":
    main()

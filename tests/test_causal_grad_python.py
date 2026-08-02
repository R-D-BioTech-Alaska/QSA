from __future__ import annotations

import math

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalPauliPlan, CausalRegister
from qsa.causal_grad import CausalParameterShift


def close(actual: float, expected: float, tolerance: float = 2.0e-9) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{actual} differs from {expected}")


def phase41a_plan() -> CausalParameterizedPlan:
    operations = []
    for qubit in range(3):
        operations.append(("ry", qubit, Parameter(f"semantic_theta_{qubit}")))
        operations.append(("rz", qubit, Parameter(f"semantic_phi_{qubit}")))
    operations.extend(
        (
            ("cnot", 0, 1),
            ("cnot", 1, 2),
            ("cnot", 2, 0),
        )
    )
    for layer in range(2):
        for qubit in range(3):
            operations.append(
                ("ry", qubit, Parameter(f"layer_{layer}_qubit_{qubit}_first"))
            )
            operations.append(
                ("rz", qubit, Parameter(f"layer_{layer}_qubit_{qubit}_second"))
            )
        operations.extend((("cnot", 0, 1), ("cnot", 1, 2)))
    return CausalParameterizedPlan(operations)


def evaluate(
    root: CausalRegister,
    plan: CausalParameterizedPlan,
    observables: CausalPauliPlan,
    values,
):
    branch = root.fork()
    try:
        branch.apply(plan, values)
        return observables.execute(branch)
    finally:
        branch.close()


def main() -> None:
    theta = Parameter("theta")
    analytic_plan = CausalParameterizedPlan([("ry", 0, theta)])
    analytic_observables = CausalPauliPlan(1, ["Z", "X"])
    analytic_gradient = CausalParameterShift(
        analytic_plan,
        analytic_observables,
    )
    angle = 0.37
    with CausalRegister(1) as root:
        result = analytic_gradient.evaluate_and_jacobian(
            root,
            {"theta": angle},
            workers=2,
        )
        close(result.values[0], math.cos(angle))
        close(result.values[1], math.sin(angle))
        close(result.jacobian[0][0], -math.sin(angle))
        close(result.jacobian[1][0], math.cos(angle))
        close(
            result.vjp([0.4, -0.7])[0],
            -0.4 * math.sin(angle) - 0.7 * math.cos(angle),
        )
        jvp = result.jvp([1.25])
        close(jvp[0], -1.25 * math.sin(angle))
        close(jvp[1], 1.25 * math.cos(angle))
        close(root.probabilities_one()[0], 0.0)

    shared = Parameter("shared")
    repeated_plan = CausalParameterizedPlan(
        [("ry", 0, shared), ("rz", 0, shared)]
    )
    try:
        CausalParameterShift(repeated_plan, analytic_observables)
    except ValueError:
        pass
    else:
        raise AssertionError("repeated parameters must fail exact shift validation")

    plan = phase41a_plan()
    observables = CausalPauliPlan(
        3,
        [
            "XII",
            "YII",
            "ZII",
            "IXI",
            "IYI",
            "IZI",
            "IIX",
            "IIY",
            "IIZ",
            "ZZI",
            "IZZ",
            "ZIZ",
            "XXX",
            "YYY",
        ],
    )
    gradient = CausalParameterShift(plan, observables)
    values = tuple(
        0.19 * math.sin((index + 1) * 0.41)
        for index in range(plan.parameter_count)
    )

    with CausalRegister(3) as root:
        initial_qsc = root.encode_qsc()
        result = gradient.evaluate_and_jacobian(root, values, workers=4)
        assert len(result.values) == 14
        assert len(result.jacobian) == 14
        assert all(len(row) == 18 for row in result.jacobian)

        epsilon = 1.0e-6
        for parameter in range(plan.parameter_count):
            positive = list(values)
            negative = list(values)
            positive[parameter] += epsilon
            negative[parameter] -= epsilon
            positive_values = evaluate(root, plan, observables, positive)
            negative_values = evaluate(root, plan, observables, negative)
            for observable in range(len(result.values)):
                finite_difference = (
                    positive_values[observable] - negative_values[observable]
                ) / (2.0 * epsilon)
                close(
                    result.jacobian[observable][parameter],
                    finite_difference,
                    tolerance=2.0e-6,
                )
        assert root.encode_qsc() == initial_qsc
        assert root.validate()

    repeated_plan.close()
    observables.close()
    plan.close()
    analytic_observables.close()
    analytic_plan.close()
    print("QSA causal gradient tests passed.")


if __name__ == "__main__":
    main()

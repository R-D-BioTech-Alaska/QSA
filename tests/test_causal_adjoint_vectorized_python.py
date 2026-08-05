from __future__ import annotations

import math

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalRegister
from qsa.causal_adjoint import CausalWeightedAdjoint
from qsa.causal_adjoint_dense import CausalDenseBatchPrimal
from qsa.causal_adjoint_vectorized import CausalVectorizedBatchAdjoint
from qsa.causal_support import CausalPauliSupportPlan


def maximum_error(first, second) -> float:
    return max(abs(float(left) - float(right)) for left, right in zip(first, second))


def test_vectorized_primal_and_adjoint_match_direct_dense() -> None:
    shared = Parameter("shared")
    plan = CausalParameterizedPlan(
        (
            ("ry", 0, Parameter("theta_0")),
            ("rz", 0, Parameter("phi_0")),
            ("ry", 1, Parameter("theta_1")),
            ("rz", 1, Parameter("phi_1")),
            ("ry", 2, Parameter("theta_2")),
            ("rz", 2, Parameter("phi_2")),
            ("cnot", 0, 1),
            ("cnot", 1, 2),
            ("ry", 0, shared),
            ("rz", 1, shared),
            ("rx", 2, Parameter("tail")),
            ("cnot", 2, 0),
        )
    )
    support = CausalPauliSupportPlan(
        3,
        (
            ((0, "X"),),
            ((0, "Y"),),
            ((1, "Z"),),
            ((2, "X"),),
            ((0, "X"), (1, "X")),
            ((1, "Y"), (2, "Z")),
            ((0, "X"), (1, "Y"), (2, "Z")),
        ),
        imaginary_tolerance=1.0e-12,
    )
    rows = tuple(
        tuple(
            0.27 * math.sin((row + 1) * (index + 1) * 0.19)
            for index in range(plan.parameter_count)
        )
        for row in range(11)
    )
    cotangents = tuple(
        tuple(
            0.14 * math.cos((row + 2) * (index + 1) * 0.23)
            for index in range(support.observable_count)
        )
        for row in range(11)
    )
    dense_primal = CausalDenseBatchPrimal(plan, support, max_qubits=3)
    dense_adjoint = CausalWeightedAdjoint(plan, support, max_qubits=3)
    vectorized = CausalVectorizedBatchAdjoint(plan, support, max_qubits=3)

    with CausalRegister(3) as root:
        initial = root.encode_qsc()
        direct_values = dense_primal.evaluate(root, rows)
        direct_adjoint = dense_adjoint.evaluate_many(root, rows, cotangents)
        vectorized_values = vectorized.values_many(root, rows)
        vectorized_adjoint = vectorized.evaluate_many(root, rows, cotangents)

        assert vectorized_values.row_count == len(rows)
        assert vectorized_values.native_calls == 1
        assert vectorized_values.structural_branches == 0
        assert vectorized_adjoint.row_count == len(rows)
        assert vectorized_adjoint.native_calls == 1
        assert vectorized_adjoint.shifted_evaluations == 0
        for expected, observed in zip(
            direct_values.values,
            vectorized_values.values,
        ):
            assert maximum_error(expected, observed) <= 2.0e-12
        for direct_values_row, direct_gradient, vector_values_row, vector_gradient in zip(
            direct_adjoint.values,
            direct_adjoint.gradients,
            vectorized_adjoint.values,
            vectorized_adjoint.gradients,
        ):
            assert maximum_error(direct_values_row, vector_values_row) <= 2.0e-12
            assert maximum_error(direct_gradient, vector_gradient) <= 2.0e-12
        assert root.encode_qsc() == initial
        assert root.validate()

        empty_values = vectorized.values_many(root, ())
        empty_adjoint = vectorized.evaluate_many(root, (), ())
        assert empty_values.values == ()
        assert empty_adjoint.values == ()
        assert empty_adjoint.gradients == ()

        try:
            vectorized.evaluate_many(root, rows, cotangents[:-1])
        except ValueError:
            pass
        else:
            raise AssertionError("vectorized row-count mismatch was accepted")

    support.close()
    plan.close()


def main() -> None:
    test_vectorized_primal_and_adjoint_match_direct_dense()
    print("QSA vectorized batch adjoint tests passed.")


if __name__ == "__main__":
    main()

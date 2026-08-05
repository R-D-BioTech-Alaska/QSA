from __future__ import annotations

import math

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalRegister
from qsa.causal_adjoint_dense import CausalDenseBatchPrimal
from qsa.causal_support import CausalPauliSupportPlan


def maximum_error(first, second) -> float:
    return max(abs(float(left) - float(right)) for left, right in zip(first, second))


def test_fused_dense_primal_matches_ephemeral_branches() -> None:
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
            ("cnot", 2, 0),
        )
    )
    support = CausalPauliSupportPlan(
        3,
        (
            ((0, "X"),),
            ((1, "Y"),),
            ((2, "Z"),),
            ((0, "X"), (1, "X")),
            ((0, "Y"), (2, "Z")),
            ((0, "X"), (1, "Y"), (2, "Z")),
        ),
        imaginary_tolerance=1.0e-12,
    )
    rows = tuple(
        {
            name: 0.23 * math.sin((row + 1) * (index + 1) * 0.31)
            for index, name in enumerate(plan.parameter_names)
        }
        for row in range(11)
    )
    primal = CausalDenseBatchPrimal(plan, support, max_qubits=3)

    with CausalRegister(3) as root:
        initial = root.encode_qsc()
        fused = primal.evaluate(root, rows)
        scalar = []
        for row in rows:
            with root.fork() as branch:
                branch.apply(plan, row)
                scalar.append(support.execute(branch))

        assert fused.row_count == len(rows)
        assert fused.forward_sweeps == len(rows)
        assert fused.native_calls == 1
        assert fused.state_materializations == 1
        assert fused.structural_branches == 0
        for fused_values, scalar_values in zip(fused.values, scalar):
            assert maximum_error(fused_values, scalar_values) <= 2.0e-12
        assert root.encode_qsc() == initial
        assert root.validate()

        empty = primal.evaluate(root, ())
        assert empty.values == ()
        assert empty.native_calls == 0
        assert empty.state_materializations == 0

    support.close()
    plan.close()


def main() -> None:
    test_fused_dense_primal_matches_ephemeral_branches()
    print("QSA fused dense batch primal tests passed.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import math

import numpy as np

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalRegister
from qsa.causal_adjoint_vectorized import CausalVectorizedBatchAdjoint
from qsa.causal_support import CausalPauliSupportPlan


def maximum_error(first, second) -> float:
    return float(np.max(np.abs(np.asarray(first) - np.asarray(second))))


def test_numpy_buffers_match_sequence_contract() -> None:
    plan = CausalParameterizedPlan(
        (
            ("ry", 0, Parameter("theta_0")),
            ("rz", 0, Parameter("phi_0")),
            ("ry", 1, Parameter("theta_1")),
            ("rz", 1, Parameter("phi_1")),
            ("cnot", 0, 1),
            ("ry", 0, Parameter("shared")),
            ("rz", 1, Parameter("shared")),
        )
    )
    support = CausalPauliSupportPlan(
        2,
        (
            ((0, "X"),),
            ((0, "Y"),),
            ((1, "Z"),),
            ((0, "X"), (1, "X")),
        ),
        imaginary_tolerance=1.0e-12,
    )
    parameters = np.ascontiguousarray(
        [
            [
                0.21 * math.sin((row + 1) * (column + 1) * 0.17)
                for column in range(plan.parameter_count)
            ]
            for row in range(13)
        ],
        dtype=np.float64,
    )
    cotangents = np.ascontiguousarray(
        [
            [
                0.12 * math.cos((row + 2) * (column + 1) * 0.29)
                for column in range(support.observable_count)
            ]
            for row in range(13)
        ],
        dtype=np.float64,
    )
    runtime = CausalVectorizedBatchAdjoint(plan, support, max_qubits=2)

    with CausalRegister(2) as root:
        initial = root.encode_qsc()
        values = runtime.values_numpy(root, parameters)
        result = runtime.evaluate_numpy(root, parameters, cotangents)
        sequence = runtime.evaluate_many(
            root,
            tuple(tuple(float(value) for value in row) for row in parameters),
            tuple(tuple(float(value) for value in row) for row in cotangents),
        )

        assert values.dtype == np.float64
        assert values.flags.c_contiguous
        assert result.values.dtype == np.float64
        assert result.values.flags.c_contiguous
        assert result.gradients.dtype == np.float64
        assert result.gradients.flags.c_contiguous
        assert result.row_count == parameters.shape[0]
        assert result.native_calls == 1
        assert result.shifted_evaluations == 0
        assert maximum_error(values, result.values) <= 2.0e-12
        assert maximum_error(result.values, sequence.values) <= 2.0e-12
        assert maximum_error(result.gradients, sequence.gradients) <= 2.0e-12
        assert root.encode_qsc() == initial
        assert root.validate()

        empty_parameters = np.empty(
            (0, plan.parameter_count),
            dtype=np.float64,
        )
        empty_cotangents = np.empty(
            (0, support.observable_count),
            dtype=np.float64,
        )
        empty_values = runtime.values_numpy(root, empty_parameters)
        empty_result = runtime.evaluate_numpy(
            root,
            empty_parameters,
            empty_cotangents,
        )
        assert empty_values.shape == (0, support.observable_count)
        assert empty_result.values.shape == (0, support.observable_count)
        assert empty_result.gradients.shape == (0, plan.parameter_count)
        assert empty_result.native_calls == 0

        try:
            runtime.values_numpy(root, np.zeros((2, 2, 2), dtype=np.float64))
        except ValueError:
            pass
        else:
            raise AssertionError("three-dimensional parameter input was accepted")

    support.close()
    plan.close()


def main() -> None:
    test_numpy_buffers_match_sequence_contract()
    print("QSA contiguous NumPy adjoint tests passed.")


if __name__ == "__main__":
    main()

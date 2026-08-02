from __future__ import annotations

import math

from qsa import Parameter
from qsa.causal import (
    CausalParameterizedPlan,
    CausalPauliPlan,
    CausalRegister,
)


def close(actual: float, expected: float, tolerance: float = 2.0e-12) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{actual} differs from {expected}")


def main() -> None:
    theta = Parameter("theta")
    bell_plan = CausalParameterizedPlan(
        [
            ("ry", 0, theta),
            ("cnot", 0, 1),
        ]
    )
    flip_third = CausalParameterizedPlan([("x", 2)])
    closure = CausalPauliPlan(
        3,
        [
            "ZZI",
            "XXI",
            "YYI",
            "IIZ",
        ],
    )

    with CausalRegister(3) as root:
        root.apply(bell_plan, {"theta": math.pi / 2.0})
        values = closure.execute(root)
        close(values[0], 1.0)
        close(values[1], 1.0)
        close(values[2], -1.0)
        close(values[3], 1.0)
        assert root.component_count == 2
        assert root.shared_owner_count == 1

        branch = root.fork()
        try:
            assert root.shared_owner_count == 2
            assert branch.shared_owner_count == 2
            branch.apply(flip_third, [])
            assert root.shared_owner_count == 1
            assert branch.shared_owner_count == 1
            close(root.probabilities_one()[2], 0.0)
            close(branch.probabilities_one()[2], 1.0)
            root.adopt(branch)
            close(root.probabilities_one()[2], 1.0)
            assert root.validate()

            encoded = root.encode_qsc()
            with CausalRegister.decode_qsc(encoded) as restored:
                assert restored.validate()
                assert restored.component_count == root.component_count
                for basis in range(8):
                    if abs(restored.amplitude(basis) - root.amplitude(basis)) > 2.0e-12:
                        raise AssertionError("causal QSC round trip changed amplitudes")
        finally:
            branch.close()

    closure.close()
    flip_third.close()
    bell_plan.close()
    print("Python causal runtime tests passed")


if __name__ == "__main__":
    main()

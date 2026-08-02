from __future__ import annotations

import math

from qsa import Parameter
from qsa.causal import CausalParameterizedPlan, CausalPauliPlan, CausalRegister
from qsa.causal_batch import apply_many, fork_many, observe_many


def close(actual: float, expected: float, tolerance: float = 3.0e-12) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{actual} differs from {expected}")


def main() -> None:
    theta = Parameter("theta")
    phi = Parameter("phi")
    plan = CausalParameterizedPlan(
        [
            ("ry", 0, theta),
            ("cnot", 0, 1),
            ("ry", 2, phi),
        ]
    )
    closure = CausalPauliPlan(3, ["ZZI", "ZII", "XXI", "IIZ"])

    with CausalRegister(3) as root:
        branch_count = 64
        branches = fork_many(root, branch_count)
        try:
            assert len(branches) == branch_count
            assert root.shared_owner_count == branch_count + 1
            assert all(branch.shared_owner_count == branch_count + 1 for branch in branches)

            rows = [
                {
                    "theta": (index + 1) * math.pi / (2.0 * branch_count),
                    "phi": (branch_count - index) * math.pi / (3.0 * branch_count),
                }
                for index in range(branch_count)
            ]
            apply_many(plan, branches, rows, workers=4)
            assert root.shared_owner_count == 1
            assert all(branch.shared_owner_count == 1 for branch in branches)
            close(root.probabilities_one()[0], 0.0)
            close(root.probabilities_one()[1], 0.0)
            close(root.probabilities_one()[2], 0.0)

            observed = observe_many(closure, branches, workers=4)
            assert len(observed) == branch_count
            for index, values in enumerate(observed):
                current_theta = rows[index]["theta"]
                current_phi = rows[index]["phi"]
                close(values[0], 1.0)
                close(values[1], math.cos(current_theta))
                close(values[2], math.sin(current_theta))
                close(values[3], math.cos(current_phi))
                assert branches[index].validate()

            selected = 17
            root.adopt(branches[selected])
            branches[selected] = None
            selected_values = closure.execute(root)
            for actual, expected in zip(selected_values, observed[selected]):
                close(actual, expected)
        finally:
            for branch in branches:
                if branch is not None:
                    branch.close()

    closure.close()
    plan.close()
    print("Python causal batch runtime tests passed")


if __name__ == "__main__":
    main()

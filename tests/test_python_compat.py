from __future__ import annotations

from qsa import (
    GroverSearch,
    OperationPlan,
    Parameter,
    ParameterizedPlan,
    QRegister,
    QubitNativeError,
    QubitRegister,
    SymmetryState,
    __version__,
)
from qubit_native import QubitRegister as LegacyQubitRegister

def main() -> None:
    assert __version__ == "0.1.6"
    assert QRegister is QubitRegister
    assert LegacyQubitRegister is QubitRegister

    with QubitRegister(2) as state:
        assert state.native_version == "0.1.6"
        assert state.abi_version >= (1, 5, 0)

        assert state.qubit_count == 2
        assert state.qubit_count() == 2
        assert state.component_count == 2
        assert state.component_count() == 2
        assert state.estimated_bytes > 0
        assert state.estimated_bytes() > 0

        plan = OperationPlan([("h", 0), ("cnot", 0, 1)])
        assert plan.compiled_step_count(state) == 2
        state.apply_plan(plan)
        assert state.component_kind(0) == "sparse"
        assert abs(state.amplitude_bits([0, 0]) - 2**-0.5) < 1e-12
        probabilities = state.probabilities_one()
        assert len(probabilities) == 2
        assert abs(probabilities[0] - 0.5) < 1e-12
        assert abs(probabilities[1] - 0.5) < 1e-12
        packet = state.encode_qsc()
        state.z(0)
        assert state.encode_qsc() != packet
        assert state.validate()

    restored = QubitRegister.decode_qsc(packet)
    try:
        assert restored.component_count == 1
        assert abs(restored.amplitude(3) - 2**-0.5) < 1e-12
        result = restored.measure_all(seed=1234)
        assert len(result) == 2
        assert result[0] == result[1]
    finally:
        restored.close()

    phase = LegacyQubitRegister(1)
    phase.x(0).s(0).sdg(0).t(0).tdg(0)
    phase.amplitude_damping_trajectory(0, 1.0, 0.0)
    assert phase.probability_one(0) == 0.0
    phase.close()

    fused = OperationPlan([("h", 0), ("h", 0), ("ry", 0, 0.31), ("rz", 0, -0.22)])
    ensemble = [QubitRegister(1) for _ in range(24)]
    try:
        assert fused.compiled_step_count(ensemble[0]) == 2
        fused.apply_many(ensemble, workers=4)
        expected = __import__("math").sin(0.31 / 2.0) ** 2
        for member in ensemble:
            assert abs(member.probabilities_one()[0] - expected) < 1e-12
    finally:
        for member in ensemble:
            member.close()
        fused.close()

    theta = Parameter("theta")
    phi = Parameter("phi")
    parameterized = ParameterizedPlan(
        [("ry", 0, theta), ("rz", 0, phi), ("cnot", 0, 1), ("ry", 1, theta)]
    )
    parameterized_states = [QubitRegister(2) for _ in range(16)]
    try:
        assert parameterized.parameter_names == ("theta", "phi")
        parameterized.apply_many(
            parameterized_states,
            {"theta": 0.31, "phi": -0.22},
            workers=4,
        )
        reference = parameterized_states[0].encode_qsc()
        for member in parameterized_states[1:]:
            assert member.encode_qsc() == reference
    finally:
        for member in parameterized_states:
            member.close()
        parameterized.close()

    with GroverSearch(3, [5]) as grover:
        assert grover.space_size == 8
        assert grover.optimal_iterations == 2
        grover.run_optimal()
        assert abs(grover.success_probability - 0.9453125) < 1e-12
        assert grover.sample(branch_sample=0.0, index_sample=0.5) == 5
        assert grover.validate()

    count_only = GroverSearch.from_marked_count(60, 1)
    try:
        assert count_only.estimated_bytes < 512
        count_only.run_optimal()
        assert count_only.success_probability > 0.999999999
    finally:
        count_only.close()

    with QubitRegister(3) as exact_grover:
        for qubit in range(3):
            exact_grover.h(qubit)
        exact_grover.grover_iterations([5], 2)
        assert abs(abs(exact_grover.amplitude(5)) ** 2 - 0.9453125) < 1e-12

    with SymmetryState(3, [2, 3, 3]) as symmetry:
        assert symmetry.class_count == 3
        assert symmetry.membership == "ordered_ranges"
        assert abs(symmetry.class_probability(0) - 0.25) < 1e-12
        new_class = symmetry.split_class(1, 1)
        assert new_class == 2
        assert symmetry.class_count == 4
        assert symmetry.merge_equivalent() == 3
        assert symmetry.class_count == 1
        symmetry.phase(0, __import__("math").pi).reflect()
        assert symmetry.validate()
        fallback = symmetry.to_register(max_qubits=3)
        try:
            for basis in range(8):
                assert abs(fallback.amplitude(basis) - symmetry.amplitude(basis)) < 1e-12
        finally:
            fallback.close()

    with QubitRegister(6) as uniform_register:
        for qubit in range(6):
            uniform_register.h(qubit)
        discovered = SymmetryState.from_register(
            uniform_register, max_qubits=6, tolerance=1e-12
        )
        try:
            assert discovered.class_count == 1
            assert discovered.membership == "ordered_ranges"
            assert discovered.discovery_error < 1e-12
        finally:
            discovered.close()

    hamming = SymmetryState.hamming_weight(60)
    try:
        assert hamming.class_count == 61
        assert hamming.membership == "hamming_weight"
        assert hamming.class_size(30) == 118264581564861424
        assert hamming.estimated_bytes < 4096
    finally:
        hamming.close()

    count_symmetry = SymmetryState.from_counts(60, [1, 7, (1 << 60) - 8])
    try:
        count_symmetry.phase(0, __import__("math").pi).reflect()
        assert count_symmetry.estimated_bytes < 1024
        assert count_symmetry.membership == "count_only"
    finally:
        count_symmetry.close()

    labels = SymmetryState.from_labels(3, [0, 1, 1, 0, 2, 2, 2, 2])
    try:
        labels.set_amplitudes([0.25, 0.25j, 0.25])
        assert labels.membership == "explicit_labels"
        assert labels.sample(class_sample=0.0, index_sample=0.0) in (0, 3)
    finally:
        labels.close()

    try:
        phase.describe()
    except QubitNativeError:
        pass
    else:
        raise AssertionError("closed registers must reject operations")

    print("QSA Python compatibility tests passed.")

if __name__ == "__main__":
    main()

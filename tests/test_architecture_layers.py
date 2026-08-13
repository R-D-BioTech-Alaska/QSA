from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CORE_FILES = (
    "include/qubit/qcomplex.hpp",
    "include/qubit/qshared_vector.hpp",
    "include/qubit/qstate.hpp",
    "include/qubit/qplan.hpp",
    "src/qstate.cpp",
    "src/qsc.cpp",
    "src/qplan.cpp",
    "src/qread.cpp",
    "src/qdiagonal.cpp",
)

FORBIDDEN_CORE_INCLUDES = (
    '"qubit/qcausal',
    '"qubit/qcausal_',
    '"qubit/detail/qcausal',
    '"qubit/qbroker.hpp"',
    '"qubit/qestimator.hpp"',
    '"qubit/qfactor.hpp"',
    '"qubit/qfactor_broker.hpp"',
    '"qubit/qparameterized_estimator.hpp"',
    '"qubit/qtensor_adjoint_causal.hpp"',
)

STRUCTURAL_FILES = (
    "include/qubit/qrouter.hpp",
    "include/qubit/qstabilizer.hpp",
    "include/qubit/qphase_graph.hpp",
    "include/qubit/qphase_sum.hpp",
    "include/qubit/qphase_exact.hpp",
    "include/qubit/qmagic.hpp",
    "include/qubit/qsymbolic_phase.hpp",
    "include/qubit/qdefect_advisor.hpp",
    "include/qubit/qhpath_factor.hpp",
    "include/qubit/qhpath_prepared.hpp",
    "include/qubit/qhpath_basis.hpp",
    "include/qubit/qhpath_born.hpp",
    "include/qubit/qhpath_basis_born.hpp",
    "include/qubit/qamplitude_advisor.hpp",
    "include/qubit/qadaptive_amplitude.hpp",
    "include/qubit/qadaptive_prepared_amplitude.hpp",
    "include/qubit/qpauli.hpp",
    "include/qubit/qtensor.hpp",
    "include/qubit/qmps.hpp",
    "include/qubit/qfactor.hpp",
    "include/qubit/qfactor_chain.hpp",
    "include/qubit/qfactor_broker.hpp",
    "include/qubit/qqtt.hpp",
    "include/qubit/qtropical_qtt.hpp",
    "include/qubit/qpersistent_tropical_qtt.hpp",
    "include/qubit/qqtt_operator.hpp",
    "include/qubit/qqtt_field.hpp",
    "include/qubit/qqtt_poisson.hpp",
    "include/qubit/qqtt_schrodinger_poisson.hpp",
    "include/qubit/qqtt_sp_invariants.hpp",
    "include/qubit/qqtt_split_step.hpp",
    "include/qubit/qfock.hpp",
    "include/qubit/qgaussian.hpp",
    "include/qubit/qbosonic_islands.hpp",
    "src/qfactor.cpp",
)

BROKER_FILES = (
    "include/qubit/qbroker.hpp",
    "src/qbroker.cpp",
    "include/qubit/qestimator.hpp",
    "src/qestimator.cpp",
    "include/qubit/qparameterized_estimator.hpp",
    "include/qubit/qtensor_adjoint_causal.hpp",
)


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def main() -> None:
    for path in CORE_FILES:
        text = read(path)
        for token in FORBIDDEN_CORE_INCLUDES:
            assert token not in text, f"core layer {path} depends on higher-level runtime through {token}"

    for path in STRUCTURAL_FILES:
        text = read(path)
        assert '"qubit/detail/qcausal' not in text, (
            f"structural layer {path} depends on causal runtime internals"
        )
        assert '"qubit/qbroker.hpp"' not in text, (
            f"structural layer {path} depends on the execution broker"
        )
        assert '"qubit/qestimator.hpp"' not in text, (
            f"structural layer {path} depends on the estimator"
        )

    for path in BROKER_FILES:
        text = read(path)
        assert '"qubit/detail/qcausal' not in text, (
            f"execution layer {path} depends on causal runtime internals"
        )

    cmake = read("CMakeLists.txt")
    core_block = cmake.split("add_library(qstate_core STATIC", 1)[1].split(")", 1)[0]
    assert "qcausal" not in core_block, "qstate_core directly owns causal runtime sources"

    public_state = read("include/qubit/qstate.hpp")
    assert "Causal" not in public_state, "QRegister core surface contains causal runtime types"
    assert "ExactExecutionBroker" not in public_state, "QRegister core surface contains broker types"
    assert "ExactEstimator" not in public_state, "QRegister core surface contains estimator types"

    print("architecture layer tests passed")


if __name__ == "__main__":
    main()

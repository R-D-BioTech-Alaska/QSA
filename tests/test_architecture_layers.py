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
)

STRUCTURAL_FILES = (
    "include/qubit/qrouter.hpp",
    "include/qubit/qstabilizer.hpp",
    "include/qubit/qphase_graph.hpp",
    "include/qubit/qpauli.hpp",
    "include/qubit/qtensor.hpp",
)


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def main() -> None:
    for path in CORE_FILES:
        text = read(path)
        for token in FORBIDDEN_CORE_INCLUDES:
            assert token not in text, f"core layer {path} depends on causal runtime through {token}"

    for path in STRUCTURAL_FILES:
        text = read(path)
        assert '"qubit/detail/qcausal' not in text, (
            f"structural layer {path} depends on causal runtime internals"
        )

    cmake = read("CMakeLists.txt")
    core_block = cmake.split("add_library(qstate_core STATIC", 1)[1].split(")", 1)[0]
    assert "qcausal" not in core_block, "qstate_core directly owns causal runtime sources"

    public_state = read("include/qubit/qstate.hpp")
    assert "Causal" not in public_state, "QRegister core surface contains causal runtime types"

    print("architecture layer tests passed")


if __name__ == "__main__":
    main()

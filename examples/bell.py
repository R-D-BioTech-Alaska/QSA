from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from qubit_native import QubitRegister


with QubitRegister(2) as state:
    state.h(0).cnot(0, 1)
    print(state.describe())
    print("|00>:", state.amplitude(0))
    print("|11>:", state.amplitude(3))
    state.save_qsc("bell.qsc")

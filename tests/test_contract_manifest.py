from __future__ import annotations

import hashlib
import json
from pathlib import Path

import qsa
import qubit_native

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    contract = json.loads((ROOT / "compatibility/qsa_0_1_contract.json").read_text())
    c_header = (ROOT / "include/qubit/c_api.h").read_text()
    cxx_header = (ROOT / "include/qubit/qstate.hpp").read_text()

    for symbol in contract["c_symbols"]:
        assert symbol in c_header, f"missing frozen C symbol: {symbol}"
    for name in contract["cxx_names"]:
        assert name in cxx_header, f"missing frozen C++ name: {name}"

    assert qubit_native.QubitRegister is qsa.QubitRegister
    register = qsa.QubitRegister
    for name in contract["python_methods"] + contract["python_query_names"]:
        assert hasattr(register, name), f"missing frozen Python name: {name}"

    fixture = ROOT / contract["qsc_v1_fixture"]
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    assert digest == contract["qsc_v1_fixture_sha256"], "QSC v1 fixture changed"

    for record in contract.get("qsc_v1_fixtures", []):
        fixture = ROOT / record["path"]
        digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
        assert digest == record["sha256"], f"QSC v1 fixture changed: {record['path']}"

    print("QSA 0.1 compatibility manifest passed.")

if __name__ == "__main__":
    main()

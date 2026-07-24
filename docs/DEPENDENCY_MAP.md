# QSA Dependency Map

## Confirmed architectural consumers

### Qubit

QSA is the native mathematical state engine planned for Qubit nodes. The
compatibility-sensitive surfaces are the Python import, the C ABI, QSC packet
transport, component inspection, measurement, and deterministic operation
ordering.

### QELM

QELM is expected to submit quantum-channel operations, query amplitudes and
measurements, and exchange leased state fragments through QSC. High-frequency
Python-to-native gate traffic is therefore a primary performance boundary. The
0.1.1 batch-operation API addresses that boundary without removing the existing
single-gate methods.

### Brain and distributed nodes

Brain/QELM Base retains persistent intelligence and memory. Qubit nodes receive
ephemeral state work and return results. QSA must therefore preserve:

- Stateless library loading across processes
- Deterministic QSC decoding
- Explicit handle ownership
- No hidden user-data persistence
- Bounded packet validation before allocation

## Public integration surfaces

| Surface | Current contract | Migration rule |
|---|---|---|
| C++ | `qubit::QRegister` | Add methods; do not rename existing methods |
| C | `qstate_*` exports | Keep ABI-major 1 symbols permanently |
| Python | `qsa.QubitRegister` | Canonical interface |
| Legacy Python | `qubit_native.QubitRegister` | Re-export canonical class |
| Library override | `QSA_NATIVE_LIB` | Prefer this name |
| Legacy override | `QUBIT_NATIVE_LIB` | Continue accepting it |
| State transport | QSC v1 | Keep decoder and frozen fixtures |

## Repository-scan status

The QSA repository itself contains integration guidance but not the full source
of Qubit, QELM, or Brain. A live downstream source scan was attempted during
this pass, but the available GitHub code-search index returned no matches and
the isolated build container had no network DNS access. Therefore this map is
based on the QSA repository and the established project architecture, not a
claim that every downstream call site has already been enumerated.

Before publishing changes to Qubit or QELM, their checked-out source trees
should be run through `tools/scan_dependents.py` or equivalent searches for:

```text
QubitRegister
QRegister
from qsa
import qsa
qubit_native
QSA_NATIVE_LIB
QUBIT_NATIVE_LIB
encode_qsc
decode_qsc
qstate_
```

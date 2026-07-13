# Qubit State Code (QSC) v1

QSC is the binary state packet produced by `QRegister::encode_qsc()`.

All integers and IEEE-754 `float64` values are little-endian.

## Header

| Field | Type | Meaning |
|---|---:|---|
| Magic | 8 bytes | `QSC1QBT\0` |
| Major | `u16` | Format major version |
| Minor | `u16` | Format minor version |
| Qubit count | `u32` | Logical register width |
| Epsilon | `f64` | Numeric zero threshold |
| Factor tolerance | `f64` | Separability tolerance |
| Max component qubits | `u32` | Runtime component limit |
| Max dense amplitudes | `u64` | Runtime dense allocation limit |
| Max sparse entries | `u64` | Runtime sparse allocation limit |
| Component count | `u32` | Number of state components |

## Component record

Each component begins with:

| Field | Type |
|---|---:|
| Kind | `u8` |
| Member count | `u32` |
| Qubit IDs | `member_count × u32` |

Kinds:

- `0`: geometric Bloch cell
- `1`: sparse amplitude patch
- `2`: dense amplitude patch

### Cell payload

Three `f64` values: `x`, `y`, `z`.

### Sparse patch payload

- Dimension: `u64`
- Entry count: `u64`
- Repeated entries: `basis_index: u64`, `real: f64`, `imag: f64`

### Dense patch payload

- Dimension: `u64`
- Value count: `u64`
- Repeated amplitudes: `real: f64`, `imag: f64`

## Integrity trailer

The final `u64` is an FNV-1a checksum over every preceding byte.

## Network direction

QSC is intentionally self-describing enough to become a Qubit-node channel payload. A later network envelope can add:

- Node ID
- Lease ID
- Channel ID
- QELM model revision
- Operation sequence
- Deadline
- Result metadata
- Cryptographic signature

Those transport fields should remain outside the mathematical state payload so QSC can be reused locally, over a network, or in checkpoints.

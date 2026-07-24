# Security Policy

## Supported versions

Security fixes are applied to the latest QSA 0.1.x release. Older 0.1 releases
remain compatibility references but may not receive separate patches.

## Reporting a vulnerability

Do not open a public issue for an undisclosed vulnerability. Use GitHub private
vulnerability reporting from the repository Security tab when available. If it
is unavailable, contact the repository owner privately through the organization
contact channel before publishing details.

Include the affected version, platform, minimal reproduction, expected impact,
and whether untrusted QSC data or a network boundary is involved.

## QSC trust boundary

QSC v1 uses an FNV-1a checksum to detect accidental corruption. It is **not** a
cryptographic signature, authentication mechanism, or malicious-packet proof.

Before QSC is accepted across a Qubit or QELM network boundary, the transport
envelope must provide authentication, integrity protection, replay resistance,
lease and channel binding, payload-size limits, and an execution deadline.

The decoder validates format structure and configured limits, but callers must
still impose an external maximum packet size before loading untrusted data.

## Native API safety

- Handles are opaque and must be destroyed exactly once.
- A single mutable register handle must not be mutated concurrently.
- Separate handles may be processed concurrently.
- Immutable operation plans may be reused across distinct registers.
- Buffer sizes returned by the API must be respected by callers.
- `qstate_last_error()` is thread-local and applies to the calling thread.

See `docs/THREADING.md` and `docs/QSC_FORMAT.md` for operational boundaries.

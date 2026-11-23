# QSA Proof Kernel — Phase 1

QPK Phase 1 adds two independently checked claims over univariate polynomials with exact rational coefficients:

- **Polynomial identity:** coefficientwise equality of two source-bound polynomials.
- **Polynomial division:** the supplied quotient and remainder satisfy \(p=dq+r\), the divisor is nonzero, and the remainder is zero or has degree below the divisor.

The checker does not call \`QPolynomial::divmod\`. The generator may use it to propose a quotient and remainder; the checker reconstructs the product coefficient by coefficient and checks the remainder bound. Polynomial identity checking compares exact rational coefficients directly.

## Certificate format

Both claims use the line-oriented \`QPK-CERT-v1\` format. Fields have a fixed order; rational coefficients use \`QRational::canonical()\` spelling; coefficient vectors are low degree first; zero is encoded as \`0\`; variable identities are UTF-8-independent lowercase hexadecimal bytes. Unknown claim versions or rules remain \`UNVERIFIED\`. Malformed, stale, or mathematically invalid certificates are rejected.

The certificate includes its claim, theory, assumptions, rule, generator label, dependency declaration, trusted source SHA-256, and a certificate SHA-256. The source digest is domain-separated and length-framed over the source variable identities and canonical coefficient vectors. Verification also receives the original trusted polynomial inputs and recomputes that digest. A certificate cannot establish a claim about an input that the caller has not supplied or independently trusted.

The generator label is descriptive metadata and does not grant authority. The certificate digest detects accidental or adversarial byte changes when it is not recomputed; it is unkeyed and is not authentication. Re-signing a false certificate does not make the mathematical claim pass.

## Checker result

\`QpkCheckResult\` reports the exact claim, source identity, checker status and reason, certificate bytes, and checked-step count. \`PROVEN_EXACT\` applies only to the stated polynomial proposition. It does not certify floating-point results or performance improvements.

The checker enforces limits for certificate bytes, source bytes, polynomial terms, rational coefficient bytes, and variable identity bytes. Defaults cap polynomials at 129 terms (degree 128), certificates at 64 KiB, and canonical coefficient tokens at 4 KiB.

The trusted computing base for this phase includes the QPK parser and rule implementation, source binding and SHA-256 routine, and QSA's \`QInteger\`/\`QRational\` exact arithmetic. Those arithmetic types are reused rather than duplicated; independent vectors cover the SHA-256 implementation, and certificates with false quotients, remainders, malformed rationals, zero divisors, and stale sources are rejected.

QPK is opt-in and does not change the C ABI, QSC v1, runtime defaults, or Brain/QELM state. Later phases can add exact linear witnesses and selected causal-pruning checks after this base has been reviewed and extended.


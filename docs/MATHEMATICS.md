# Qubit State Algebra (QSA) — Mathematical Specification v0.1

## 1. State partition

For a register with qubit set

\[
V = \{0,1,\ldots,n-1\},
\]

QSA maintains a partition

\[
\mathcal P = \{C_1,C_2,\ldots,C_m\},
\qquad C_i \cap C_j = \varnothing,
\qquad \bigcup_i C_i = V.
\]

The full pure state is represented as

\[
|\Psi\rangle = \bigotimes_{C\in\mathcal P}|\psi_C\rangle.
\]

This is not an approximation. The partition changes whenever an operation creates or removes dependencies between components.

## 2. Geometric qubit cell

An independent pure qubit is represented by a unit Bloch vector

\[
\mathbf r=(x,y,z),\qquad x^2+y^2+z^2=1.
\]

Its canonical spinor is reconstructed as

\[
|\psi\rangle = \alpha|0\rangle+\beta|1\rangle,
\]

with

\[
\alpha=\sqrt{\frac{1+z}{2}},
\qquad
\beta=\frac{x+iy}{2\alpha}
\]

when \(\alpha\neq0\), and the south-pole case represented canonically as \(|1\rangle\).

Common single-qubit gates are rotations of \(\mathbf r\), so they do not require a complex array or matrix multiplication.

## 3. Entangled patch

A component containing \(k>1\) qubits is represented by a local amplitude function

\[
A_C:\{0,1\}^k\rightarrow\mathbb C,
\qquad
\sum_b |A_C(b)|^2=1.
\]

QSA stores this function in one of two forms:

- **Sparse:** only basis indices for which \(|A_C(b)|>\varepsilon\)
- **Dense:** a contiguous local patch when support becomes sufficiently large

The representation decision is local to the component, not global to the register.

## 4. Direct gate action

A one-qubit gate updates only pairs of amplitudes whose basis indices differ at the target bit:

\[
\begin{bmatrix}A'(b_{q=0})\\A'(b_{q=1})\end{bmatrix}
=
U
\begin{bmatrix}A(b_{q=0})\\A(b_{q=1})\end{bmatrix}.
\]

A two-qubit gate updates only quartets associated with the four local values of the selected bits. No full-system operator

\[
I\otimes\cdots\otimes U\otimes\cdots\otimes I
\]

is constructed.

## 5. Component merge

When a two-qubit gate connects qubits in different components \(C_a\) and \(C_b\), QSA first forms

\[
|\psi_{C_a\cup C_b}\rangle
=
|\psi_{C_a}\rangle\otimes|\psi_{C_b}\rangle,
\]

then applies the gate locally. Unrelated components remain untouched.

## 6. Exact singleton factor recovery

For a qubit \(q\) inside a pure patch, QSA computes its reduced density matrix

\[
\rho_q=\operatorname{Tr}_{C\setminus q}(|\psi_C\rangle\langle\psi_C|).
\]

The qubit is separable exactly when

\[
\det(\rho_q)=0
\]

within the configured floating-point tolerance. When this condition holds, QSA factors

\[
|\psi_C\rangle=|\psi_q\rangle\otimes|\psi_{C\setminus q}\rangle
\]

and restores \(q\) to a geometric cell. This is used after two-qubit operations, nonunitary trajectories, and measurement.

## 7. Measurement

For target qubit \(q\),

\[
p(1)=\sum_{b:b_q=1}|A_C(b)|^2.
\]

After sampling outcome \(r\), amplitudes inconsistent with \(r\) are removed and the remainder is renormalized. Because the measured qubit is then in a basis state, it factors from the remaining patch and becomes its own cell.

## 8. Noise by trajectories

QSA v0.1 keeps a pure-state representation. A noisy channel with Kraus operators \(K_j\) is sampled as a quantum trajectory:

\[
p_j=\langle\psi|K_j^\dagger K_j|\psi\rangle,
\qquad
|\psi'\rangle=\frac{K_j|\psi\rangle}{\sqrt{p_j}}.
\]

An ensemble of independent trajectories estimates the corresponding mixed-state dynamics without allocating a dense \(4^n\) density matrix.

## 9. Complexity

Let component widths be \(k_1,\ldots,k_m\) and sparse supports be \(s_1,\ldots,s_m\). Storage is approximately

\[
O\left(\sum_{i\in\text{cells}}1+
\sum_{i\in\text{sparse}}s_i+
\sum_{i\in\text{dense}}2^{k_i}\right),
\]

rather than automatically \(O(2^n)\).

Worst-case quantum states still require exponential information. QSA's purpose is to pay that cost only in components whose realized state actually requires it.

## 10. Invariants

The engine maintains:

1. Every logical qubit belongs to exactly one component.
2. Every component is normalized.
3. Every geometric cell has unit Bloch length.
4. Patch dimension equals \(2^{|C|}\).
5. Sparse indices are unique and inside the patch dimension.
6. The tensor product of all components is the register state.

# Bayesian adaptive compaction

`AdaptiveOperationPlan` is an opt-in exact execution path for repeated circuits whose two-qubit gates rarely recover singleton separability.

After CNOT, CZ, or an in-component SWAP, QSA can test the affected qubits for exact factorization. That test is valuable when it succeeds, but repeated unsuccessful checks can dominate dense-component workloads.

The adaptive plan keeps a Beta-Bernoulli posterior for each gate family and component-width range. A successful exact split updates the success count; an unsuccessful check updates the failure count. When the posterior success probability falls below the configured threshold, the plan can skip only the representation cleanup check.

The quantum gate is always applied. No amplitude is approximated, removed, or altered by the policy.

## Safety controls

- The feature is used only through `AdaptiveOperationPlan`.
- Warmup checks collect evidence before any skip is allowed.
- Periodic audits force an exact separability check even after a long failure streak.
- Measurement and nonunitary trajectories retain their ordinary mandatory compaction behavior.
- Metrics report checks, successful splits, skips, and forced audits.
- `reset_learning()` clears all posterior evidence.

The plan is stateful and should not be executed concurrently from multiple threads. Independent workers should use independent plans.

The reproducible benchmark target is `qstate_adaptive_compaction_benchmark`.

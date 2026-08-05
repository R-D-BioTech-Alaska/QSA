# Persistent Pullback Lifecycle Recovery v2

This draft lane contains no QSA runtime changes.

It re-evaluates the historical retained-candidate pullback failure against the current exact-repair baseline using:

1. a repeated-lifecycle regression covering same, different, zero, and negative cotangents; selected-index changes; discard; commit; generation transition; workspace refresh; and checkpoint rollback;
2. the original frozen 5,395-parameter persistent Tripair optimizer harness from QSA PR #35.

A green current-state result does not prove that PR #36 caused the historical correction. It establishes only that the current source passes the original workload and lifecycle matrix.

The safe V10 batch-native route remains the preferred integration path unless persistent execution demonstrates a complete-system advantage over it.

No GPU, automatic merge, Brain production, or final-training authority is present.

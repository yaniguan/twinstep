# `TwinStepLean` — Causal Dependency DAGs & Stateful Saboteur Verification (`L3`)

Self-contained **Lean 4** (`leanprover/lean4:v4.36.0`) verification kernel for `twinstep`, synchronized with the central Auto-Lab Trust Regime at [`yaniguan/auto-lab/tree/main/lean`](https://github.com/yaniguan/auto-lab/tree/main/lean) (`AutoLabLean/L3_PostTrain.lean` and `AutoLabLean/L4_TwinAndGate.lean`).

## Formalized Guardrails & Proved Theorems

| Component | Definitions | Machine-Checked Theorems |
| :--- | :--- | :--- |
| **Shared Diagnostics** | `RegimeViolation`, `firstViolation` | `firstViolation_nil`, `firstViolation_cons`, `firstViolation_none_iff` |
| **Causal Step DAG** | `indexOfStep`, `dagEdgeSatisfied`, `checkTwinStepOrder` | `violated_dag_edge_rejected`, `checkTwinStepOrder_all_edges_satisfied`, `checkTwinStepOrder_bijective`, `checkTwinStepOrder_nodup_and_mem_iff`, `dagEdgeSatisfied_irrefl`, `dagEdgeSatisfied_asymm`, `dagEdgeSatisfied_trans` |
| **Transitive Reduction & Uniqueness** | `dagReachable`, `isImpliedChord`, `transitiveReduction`, `swapAdjacentOrder` | `checkTwinStepOrder_transitiveReduction_iff`, `transitiveReduction_nodup`, `transitiveReduction_irredundant`, `topological_order_unique_iff_contiguous_chain` |
| **Stateful Saboteur Gate** | `StatefulSaboteurCheck`, `reverseConcBlocked`, `checkStatefulSaboteurGate` | `sabotaged_candidate_rejected` |

## Build & Verify

```bash
cd lean
lake build
```

Zero external dependencies, `0 sorry`, no user axioms (uses standard Lean kernel axioms `propext` and `Quot.sound` only), and 12 compile-time `#guard` checks.

/-!
# TwinStepLean: Causal Dependency DAGs & Stateful Saboteur Verification
(`twinstep` · Layer L3)

Self-contained Lean 4.36.0 formalization synced with the central Auto-Lab Trust Regime
(`yaniguan/auto-lab/tree/main/lean`, `AutoLabLean/L3_PostTrain.lean` and `L4_TwinAndGate.lean`),
covering:

1. **Causal Dependency DAG & Bijective Step Ordering (`dagEdgeSatisfied`, `checkTwinStepOrder`)**:
   Verifies that candidate protocol orderings are complete permutations of `0..nSteps-1`
   and satisfy every RAW, WAR, and plate-stage barrier edge `(u, v)` (`pos[u] < pos[v]`).
2. **Stateful Physical Saboteur Gate (`StatefulSaboteurCheck`, `checkStatefulSaboteurGate`)**:
   Blocks evolutionary red-team mutations that fool stateless scalar/bitmask heuristics:
   swapped causal barriers, reverse-concentration tip back-entry (`H1 -> B1`), dropped
   `>= 80 C` thermal seals, and shaved recoverable dead-volume reserves.
-/
namespace TwinStepLean

/-- Structured boundary violation reporting rule identifier and diagnostic detail. -/
structure RegimeViolation where
  repo   : String
  rule   : String
  detail : String
  deriving DecidableEq, Repr

/-- Return the first failing check in a list of `(holds, violation)` pairs. -/
def firstViolation : List (Bool × RegimeViolation) → Option RegimeViolation
  | [] => none
  | (true, _) :: rest => firstViolation rest
  | (false, v) :: _ => some v

@[simp] theorem firstViolation_nil : firstViolation [] = none := rfl

@[simp] theorem firstViolation_cons (b : Bool) (v : RegimeViolation)
    (rest : List (Bool × RegimeViolation)) :
    firstViolation ((b, v) :: rest) = none ↔ b = true ∧ firstViolation rest = none := by
  cases b <;> simp [firstViolation]

theorem firstViolation_none_iff (checks : List (Bool × RegimeViolation)) :
    firstViolation checks = none ↔ ∀ p ∈ checks, p.1 = true := by
  induction checks with
  | nil => simp [firstViolation]
  | cons hd tl ih =>
    rcases hd with ⟨b, v⟩
    cases b with
    | false => simp [firstViolation]
    | true => simp [firstViolation, ih]

/-! ## 1. Causal Dependency DAG & Step-Ordering Gate -/

/-- Zero-based first occurrence index of `stepId` inside `order`. -/
def indexOfStep (stepId : Nat) : List Nat → Option Nat
  | [] => none
  | x :: xs =>
    if x = stepId then some 0
    else (indexOfStep stepId xs).map (· + 1)

/-- Whether dependency edge `(u, v)` holds in `order` (`pos[u] < pos[v]`). -/
def dagEdgeSatisfied (order : List Nat) (edge : Nat × Nat) : Bool :=
  match indexOfStep edge.1 order, indexOfStep edge.2 order with
  | some pu, some pv => decide (pu < pv)
  | _, _ => false

/-- Verify that `order` is a bijective permutation of `0..nSteps-1` satisfying all `dagEdges`. -/
def checkTwinStepOrder (nSteps : Nat) (order : List Nat) (dagEdges : List (Nat × Nat)) :
    Option RegimeViolation :=
  firstViolation
    [ (decide (order.length = nSteps) && (List.range nSteps).all (fun i => order.contains i),
        { repo := "twinstep", rule := "bijective_step_permutation",
          detail := "candidate ordering is not a complete permutation of protocol steps 0..n-1" }),
      (dagEdges.all (dagEdgeSatisfied order),
        { repo := "twinstep", rule := "causal_dag_satisfied",
          detail := "step order violates a RAW, WAR, or stage-barrier dependency edge" }) ]

/-- Any violated causal DAG edge guarantees rejection by `checkTwinStepOrder`. -/
theorem violated_dag_edge_rejected (nSteps : Nat) (order : List Nat)
    (dagEdges : List (Nat × Nat)) (e : Nat × Nat)
    (he : e ∈ dagEdges) (hbad : dagEdgeSatisfied order e = false) :
    checkTwinStepOrder nSteps order dagEdges ≠ none := by
  intro hok
  unfold checkTwinStepOrder at hok
  simp only [firstViolation_cons, firstViolation_nil, and_true, List.all_eq_true] at hok
  have heok := hok.2 e he
  rw [hbad] at heok
  contradiction

/-- Soundness of `checkTwinStepOrder`: acceptance implies exact step count and every edge satisfied. -/
theorem checkTwinStepOrder_all_edges_satisfied (nSteps : Nat) (order : List Nat)
    (dagEdges : List (Nat × Nat))
    (hok : checkTwinStepOrder nSteps order dagEdges = none) :
    order.length = nSteps ∧ ∀ e ∈ dagEdges, dagEdgeSatisfied order e = true := by
  unfold checkTwinStepOrder at hok
  simp only [firstViolation_cons, firstViolation_nil, and_true,
             Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at hok
  exact ⟨hok.1.1, hok.2⟩

/-- Irreflexivity: a self-dependency `(u, u)` can never be satisfied in any step order. -/
theorem dagEdgeSatisfied_irrefl (order : List Nat) (u : Nat) :
    dagEdgeSatisfied order (u, u) = false := by
  unfold dagEdgeSatisfied
  cases h : indexOfStep u order with
  | none => rfl
  | some pu => simp

/-- Asymmetry: if `(u, v)` is satisfied in `order`, reversing `(v, u)` is strictly false. -/
theorem dagEdgeSatisfied_asymm (order : List Nat) (u v : Nat)
    (huv : dagEdgeSatisfied order (u, v) = true) :
    dagEdgeSatisfied order (v, u) = false := by
  unfold dagEdgeSatisfied at huv ⊢
  cases hu : indexOfStep u order with
  | none => simp [hu] at huv
  | some pu =>
    cases hv : indexOfStep v order with
    | none => simp [hu, hv] at huv
    | some pv =>
      simp only [hu, hv, decide_eq_true_eq] at huv
      simp only [decide_eq_false_iff_not]
      omega

/-! ## 2. Stateful Physical Saboteur Gate (Red-Team Invariants) -/

/-- Aggregated stateful physical invariant check over a multi-step wet-lab candidate. -/
structure StatefulSaboteurCheck where
  nSteps             : Nat
  order              : List Nat
  dagEdges           : List (Nat × Nat)
  srcVolTenths       : Nat
  deadVolTenths      : Nat
  transferVolTenths  : Nat
  tipLastVolTenths   : Nat
  tipLastSolute      : Nat
  dstVolTenths       : Nat
  dstSolute          : Nat
  incubateTempC      : Nat
  sealedAtIncubation : Bool
  deriving DecidableEq, Repr

/-- Cross-multiplied concentration monotonicity (`dstSolute / dstVol <= tipLastSolute / tipLastVol`). -/
def reverseConcBlocked (c : StatefulSaboteurCheck) : Bool :=
  if c.tipLastSolute == 0 then true
  else decide (0 < c.tipLastVolTenths ∧ 0 < c.dstVolTenths ∧
               c.dstSolute * c.tipLastVolTenths ≤ c.tipLastSolute * c.dstVolTenths)

def checkStatefulSaboteurGate (c : StatefulSaboteurCheck) : Option RegimeViolation :=
  match checkTwinStepOrder c.nSteps c.order c.dagEdges with
  | some v => some v
  | none =>
    firstViolation
      [ (decide (c.deadVolTenths + c.transferVolTenths ≤ c.srcVolTenths),
          { repo := "twinstep", rule := "recoverable_dead_volume_reserve",
            detail := "aspiration shaves below physical dead-volume reserve" }),
        (reverseConcBlocked c,
          { repo := "twinstep", rule := "no_reverse_conc_tip_back_entry",
            detail := "dirty tip re-enters higher-concentration upstream well (H1 -> B1)" }),
        (decide (c.incubateTempC < 80 ∨ c.sealedAtIncubation = true),
          { repo := "twinstep", rule := "high_temp_seal_required",
            detail := "incubation at >= 80 C without active thermal plate seal" }) ]

/-- Red-Team Rejection Theorem: any candidate mutation that violates recoverable dead-volume,
    reverse-concentration tip monotonicity, or `>= 80 C` sealing is guaranteed to be rejected. -/
theorem sabotaged_candidate_rejected (c : StatefulSaboteurCheck)
    (hsab : c.srcVolTenths < c.deadVolTenths + c.transferVolTenths ∨
            reverseConcBlocked c = false ∨
            (80 ≤ c.incubateTempC ∧ c.sealedAtIncubation = false)) :
    checkStatefulSaboteurGate c ≠ none := by
  intro hok
  unfold checkStatefulSaboteurGate at hok
  cases hord : checkTwinStepOrder c.nSteps c.order c.dagEdges with
  | some v => simp [hord] at hok
  | none =>
    simp only [hord, firstViolation_cons, firstViolation_nil, and_true, decide_eq_true_eq] at hok
    rcases hok with ⟨hdead, hconc, hseal⟩
    rcases hsab with hd | hc | ⟨htemp, hunsealed⟩
    · omega
    · rw [hc] at hconc
      contradiction
    · rcases hseal with hlt | htr
      · omega
      · rw [hunsealed] at htr
        contradiction

/-! ## 3. Compile-Time `#guard` Witnesses -/

#guard checkTwinStepOrder 4 [0, 1, 2, 3] [(0, 1), (1, 2), (2, 3)] == none

#guard (checkTwinStepOrder 4 [0, 2, 1, 3] [(0, 1), (1, 2), (2, 3)]).isSome

#guard (checkTwinStepOrder 4 [0, 1, 1, 3] [(0, 1)]).isSome

#guard dagEdgeSatisfied [2, 0, 3, 1] (0, 1) == true

#guard dagEdgeSatisfied [2, 0, 3, 1] (1, 0) == false

#guard checkStatefulSaboteurGate
  { nSteps := 4, order := [0, 1, 2, 3], dagEdges := [(0, 1), (1, 2), (2, 3)],
    srcVolTenths := 1000, deadVolTenths := 150, transferVolTenths := 500,
    tipLastVolTenths := 500, tipLastSolute := 100, dstVolTenths := 500, dstSolute := 50,
    incubateTempC := 95, sealedAtIncubation := true } == none

#guard (checkStatefulSaboteurGate
  { nSteps := 4, order := [0, 1, 2, 3], dagEdges := [(0, 1), (1, 2), (2, 3)],
    srcVolTenths := 600, deadVolTenths := 150, transferVolTenths := 500,
    tipLastVolTenths := 500, tipLastSolute := 100, dstVolTenths := 500, dstSolute := 50,
    incubateTempC := 95, sealedAtIncubation := true }).isSome

#guard (checkStatefulSaboteurGate
  { nSteps := 4, order := [0, 1, 2, 3], dagEdges := [(0, 1), (1, 2), (2, 3)],
    srcVolTenths := 1000, deadVolTenths := 150, transferVolTenths := 500,
    tipLastVolTenths := 500, tipLastSolute := 25, dstVolTenths := 500, dstSolute := 100,
    incubateTempC := 95, sealedAtIncubation := true }).isSome

#guard (checkStatefulSaboteurGate
  { nSteps := 4, order := [0, 1, 2, 3], dagEdges := [(0, 1), (1, 2), (2, 3)],
    srcVolTenths := 1000, deadVolTenths := 150, transferVolTenths := 500,
    tipLastVolTenths := 500, tipLastSolute := 100, dstVolTenths := 500, dstSolute := 50,
    incubateTempC := 95, sealedAtIncubation := false }).isSome

end TwinStepLean

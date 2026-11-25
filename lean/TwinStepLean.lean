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

/-- Transitivity: if `(u, v)` and `(v, w)` hold in `order`, then `(u, w)` holds in `order`. -/
theorem dagEdgeSatisfied_trans (order : List Nat) (u v w : Nat)
    (huv : dagEdgeSatisfied order (u, v) = true)
    (hvw : dagEdgeSatisfied order (v, w) = true) :
    dagEdgeSatisfied order (u, w) = true := by
  unfold dagEdgeSatisfied at huv hvw ⊢
  cases hu : indexOfStep u order with
  | none => simp [hu] at huv
  | some pu =>
    cases hv : indexOfStep v order with
    | none => simp [hu, hv] at huv
    | some pv =>
      cases hw : indexOfStep w order with
      | none => simp [hv, hw] at hvw
      | some pw =>
        simp only [hu, hv, hw, decide_eq_true_eq] at huv hvw ⊢
        omega

/-! ### 1b. Pigeonhole Permutation Bijectivity (`Nodup` & Range Equivalence) -/

/-- Remove the first occurrence of `a` from a list of naturals. -/
def eraseOne (a : Nat) : List Nat → List Nat
  | [] => []
  | x :: xs => if x = a then xs else x :: eraseOne a xs

private theorem eraseOne_length_of_mem {a : Nat} {l : List Nat} (h : a ∈ l) :
    (eraseOne a l).length + 1 = l.length := by
  induction l with
  | nil => cases h
  | cons x xs ih =>
    unfold eraseOne
    split
    · rfl
    · next hne =>
      cases h with
      | head => exact absurd rfl hne
      | tail _ hxs =>
        have hlen := ih hxs
        simp only [List.length_cons]
        omega

private theorem mem_of_mem_eraseOne {a y : Nat} {l : List Nat} (h : y ∈ eraseOne a l) :
    y ∈ l := by
  induction l with
  | nil => cases h
  | cons x xs ih =>
    unfold eraseOne at h
    split at h
    · exact List.Mem.tail _ h
    · cases h with
      | head => exact List.Mem.head _
      | tail _ htl => exact List.Mem.tail _ (ih htl)

private theorem mem_eraseOne_of_ne {a y : Nat} {l : List Nat} (hne : y ≠ a) (h : y ∈ l) :
    y ∈ eraseOne a l := by
  induction l with
  | nil => cases h
  | cons x xs ih =>
    unfold eraseOne
    by_cases hxa : x = a
    · simp only [hxa, ↓reduceIte]
      cases h with
      | head => exact absurd hxa hne
      | tail _ hxs => exact hxs
    · simp only [hxa, ↓reduceIte]
      cases h with
      | head => exact List.Mem.head _
      | tail _ hxs => exact List.Mem.tail _ (ih hxs)

private theorem mem_iff_eq_or_eraseOne {a y : Nat} {l : List Nat} (ha : a ∈ l) :
    y ∈ l ↔ y = a ∨ y ∈ eraseOne a l := by
  constructor
  · intro hy
    by_cases hya : y = a
    · exact Or.inl hya
    · exact Or.inr (mem_eraseOne_of_ne hya hy)
  · intro hor
    rcases hor with rfl | hmem
    · exact ha
    · exact mem_of_mem_eraseOne hmem

private theorem nodup_of_eraseOne {a : Nat} {l : List Nat}
    (ha : a ∈ l) (hna : a ∉ eraseOne a l) (hn : (eraseOne a l).Nodup) :
    l.Nodup := by
  induction l with
  | nil => exact List.Pairwise.nil
  | cons x xs ih =>
    unfold eraseOne at hna hn
    by_cases hxa : x = a
    · simp only [hxa, ↓reduceIte] at hna hn
      subst hxa
      exact List.nodup_cons.mpr ⟨hna, hn⟩
    · simp only [hxa, ↓reduceIte] at hna hn
      rw [List.nodup_cons] at hn ⊢
      rcases hn with ⟨hx_not_er, hn_er⟩
      have ha_xs : a ∈ xs := by
        cases ha with
        | head => exact absurd rfl hxa
        | tail _ hxs => exact hxs
      have hna_er : a ∉ eraseOne a xs := fun hm => hna (List.Mem.tail _ hm)
      have hxs_nodup := ih ha_xs hna_er hn_er
      refine ⟨?_, hxs_nodup⟩
      intro hx_xs
      exact hx_not_er (mem_eraseOne_of_ne hxa hx_xs)

private theorem pigeonhole_of_nodup_superset {s : List Nat} (hs : s.Nodup) :
    ∀ (l : List Nat), (∀ x ∈ s, x ∈ l) →
      s.length ≤ l.length ∧ (l.length = s.length → l.Nodup ∧ ∀ y, y ∈ l ↔ y ∈ s) := by
  induction s with
  | nil =>
    intro l _
    refine ⟨Nat.zero_le _, ?_⟩
    intro hlen
    have hl : l = [] := List.eq_nil_of_length_eq_zero hlen
    subst hl
    exact ⟨List.Pairwise.nil, fun _ => Iff.rfl⟩
  | cons a s' ih =>
    rw [List.nodup_cons] at hs
    rcases hs with ⟨ha_not_s', hs'_nodup⟩
    intro l hsub
    have ha_l : a ∈ l := hsub a (List.Mem.head _)
    have hsub' : ∀ x ∈ s', x ∈ eraseOne a l := by
      intro x hx
      have hxa : x ≠ a := fun h => ha_not_s' (h ▸ hx)
      exact mem_eraseOne_of_ne hxa (hsub x (List.Mem.tail _ hx))
    have hlen_er := eraseOne_length_of_mem ha_l
    have ⟨hge, heq_imp⟩ := ih hs'_nodup (eraseOne a l) hsub'
    simp only [List.length_cons]
    refine ⟨by omega, ?_⟩
    intro hlen_eq
    have her_eq : (eraseOne a l).length = s'.length := by omega
    have ⟨her_nodup, her_iff⟩ := heq_imp her_eq
    have ha_not_er : a ∉ eraseOne a l := fun hm => ha_not_s' ((her_iff a).mp hm)
    refine ⟨nodup_of_eraseOne ha_l ha_not_er her_nodup, ?_⟩
    intro y
    rw [mem_iff_eq_or_eraseOne ha_l, her_iff, List.mem_cons]

/-- Pigeonhole Soundness: `checkTwinStepOrder` acceptance implies `order.Nodup`
    and exact bijective membership on `0..nSteps-1`. -/
theorem checkTwinStepOrder_bijective (nSteps : Nat) (order : List Nat)
    (dagEdges : List (Nat × Nat))
    (hok : checkTwinStepOrder nSteps order dagEdges = none) :
    order.Nodup ∧ ∀ i, i ∈ order ↔ i < nSteps := by
  unfold checkTwinStepOrder at hok
  simp only [firstViolation_cons, firstViolation_nil, and_true,
             Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at hok
  rcases hok with ⟨⟨hlen, hall⟩, _⟩
  have hsub : ∀ x ∈ List.range nSteps, x ∈ order := by
    intro x hx
    have hc := hall x hx
    exact List.contains_iff_mem.mp hc
  have hlen_range : order.length = (List.range nSteps).length := by
    rw [List.length_range]
    exact hlen
  have ⟨hnodup, hiff⟩ :=
    (pigeonhole_of_nodup_superset (List.nodup_range) order hsub).2 hlen_range
  refine ⟨hnodup, fun i => ?_⟩
  rw [hiff i, List.mem_range]

/-- Combined soundness of `checkTwinStepOrder`: length, `Nodup`, range membership, and edges. -/
theorem checkTwinStepOrder_nodup_and_mem_iff (nSteps : Nat) (order : List Nat)
    (dagEdges : List (Nat × Nat))
    (hok : checkTwinStepOrder nSteps order dagEdges = none) :
    order.length = nSteps ∧ order.Nodup ∧ (∀ i, i ∈ order ↔ i < nSteps) ∧
      (∀ e ∈ dagEdges, dagEdgeSatisfied order e = true) := by
  have ⟨hlen, hedges⟩ := checkTwinStepOrder_all_edges_satisfied nSteps order dagEdges hok
  have ⟨hnodup, hmem⟩ := checkTwinStepOrder_bijective nSteps order dagEdges hok
  exact ⟨hlen, hnodup, hmem, hedges⟩

/-! ### 1c. Transitive Reduction Completeness & Irredundancy -/

/-- Fuel-bounded non-empty directed path check `u ->+ v` in `edges`. -/
def dagReachableAux (edges : List (Nat × Nat)) : Nat → Nat → Nat → Bool
  | 0, _, _ => false
  | fuel + 1, u, v =>
    edges.any (fun e =>
      decide (e.1 = u) && (decide (e.2 = v) || dagReachableAux edges fuel e.2 v))

/-- Non-empty directed path check `u ->+ v` over forward DAG edges (`e.1 < e.2`). -/
def dagReachable (edges : List (Nat × Nat)) (u v : Nat) : Bool :=
  dagReachableAux edges (v - u) u v

/-- Whether edge `e = (u, v)` is implied by a length `>= 2` path `u -> mid ->+ v`. -/
def isImpliedChord (edges : List (Nat × Nat)) (e : Nat × Nat) : Bool :=
  edges.any (fun e1 =>
    decide (e1.1 = e.1) && decide (e1.2 ≠ e.2) && dagReachable edges e1.2 e.2)

/-- Deduplicate a list of DAG edges while preserving membership and `Nodup`. -/
def dedupEdges : List (Nat × Nat) → List (Nat × Nat)
  | [] => []
  | e :: es =>
    let rest := dedupEdges es
    if e ∈ rest then rest else e :: rest

private theorem mem_dedupEdges {e : Nat × Nat} {es : List (Nat × Nat)} :
    e ∈ dedupEdges es ↔ e ∈ es := by
  induction es with
  | nil => simp [dedupEdges]
  | cons hd tl ih =>
    unfold dedupEdges
    dsimp only
    split
    · next hmem =>
      constructor
      · intro h; exact List.Mem.tail _ (ih.mp h)
      · intro h
        cases h with
        | head => exact hmem
        | tail _ htl => exact ih.mpr htl
    · next hnot =>
      simp only [List.mem_cons, ih]

private theorem nodup_dedupEdges (es : List (Nat × Nat)) : (dedupEdges es).Nodup := by
  induction es with
  | nil => exact List.Pairwise.nil
  | cons hd tl ih =>
    unfold dedupEdges
    dsimp only
    split
    · exact ih
    · next hnot => exact List.nodup_cons.mpr ⟨hnot, ih⟩

/-- Transitive reduction of forward DAG edges (`u < v`), mirroring
    `twinstep.dependencies._transitive_reduction`. -/
def transitiveReduction (_nSteps : Nat) (edges : List (Nat × Nat)) : List (Nat × Nat) :=
  (dedupEdges edges).filter (fun e => !(isImpliedChord edges e))

theorem transitiveReduction_subset (nSteps : Nat) (edges : List (Nat × Nat))
    {e : Nat × Nat} (he : e ∈ transitiveReduction nSteps edges) :
    e ∈ edges := by
  unfold transitiveReduction at he
  rw [List.mem_filter] at he
  exact mem_dedupEdges.mp he.1

theorem transitiveReduction_nodup (nSteps : Nat) (edges : List (Nat × Nat)) :
    (transitiveReduction nSteps edges).Nodup := by
  unfold transitiveReduction
  exact List.Pairwise.filter _ (nodup_dedupEdges edges)

private theorem dagReachableAux_lt_of_fwd {edges : List (Nat × Nat)}
    (hfwd : ∀ d ∈ edges, d.1 < d.2) :
    ∀ fuel u v, dagReachableAux edges fuel u v = true → u < v := by
  intro fuel
  induction fuel with
  | zero => intro _ _ h; simp [dagReachableAux] at h
  | succ k ih =>
    intro u v h
    simp only [dagReachableAux, List.any_eq_true, Bool.and_eq_true,
               Bool.or_eq_true, decide_eq_true_eq] at h
    rcases h with ⟨e, he_mem, rfl, he_rest⟩
    have hlt := hfwd e he_mem
    rcases he_rest with rfl | hrec
    · exact hlt
    · have h2 := ih e.2 v hrec
      omega

private theorem dagReachableAux_mono_edges {e1 e2 : List (Nat × Nat)}
    (hsub : ∀ d ∈ e1, d ∈ e2) :
    ∀ fuel u v, dagReachableAux e1 fuel u v = true → dagReachableAux e2 fuel u v = true := by
  intro fuel
  induction fuel with
  | zero => intro _ _ h; simp [dagReachableAux] at h
  | succ k ih =>
    intro u v h
    simp only [dagReachableAux, List.any_eq_true, Bool.and_eq_true,
               Bool.or_eq_true, decide_eq_true_eq] at h ⊢
    rcases h with ⟨e, he_mem, hu, hrest⟩
    refine ⟨e, hsub e he_mem, hu, ?_⟩
    rcases hrest with rfl | hrec
    · exact Or.inl rfl
    · exact Or.inr (ih e.2 v hrec)

private theorem dagReachableAux_mono_fuel {edges : List (Nat × Nat)} :
    ∀ {f1 f2 : Nat}, f1 ≤ f2 → ∀ u v,
      dagReachableAux edges f1 u v = true → dagReachableAux edges f2 u v = true := by
  intro f1
  induction f1 with
  | zero => intro _ _ _ _ h; simp [dagReachableAux] at h
  | succ k1 ih =>
    intro f2 hle u v h
    cases f2 with
    | zero => omega
    | succ k2 =>
      simp only [dagReachableAux, List.any_eq_true, Bool.and_eq_true,
                 Bool.or_eq_true, decide_eq_true_eq] at h ⊢
      rcases h with ⟨e, he_mem, hu, hrest⟩
      refine ⟨e, he_mem, hu, ?_⟩
      rcases hrest with rfl | hrec
      · exact Or.inl rfl
      · exact Or.inr (ih (by omega) e.2 v hrec)

private theorem dagReachableAux_satisfied_of_span {edges : List (Nat × Nat)}
    {order : List Nat}
    (hfwd : ∀ d ∈ edges, d.1 < d.2) :
    ∀ fuel a b,
      dagReachableAux edges fuel a b = true →
      (∀ d ∈ edges, d.2 - d.1 ≤ b - a → dagEdgeSatisfied order d = true) →
      dagEdgeSatisfied order (a, b) = true := by
  intro fuel
  induction fuel with
  | zero => intro _ _ h _; simp [dagReachableAux] at h
  | succ k ih =>
    intro a b h hsmall
    simp only [dagReachableAux, List.any_eq_true, Bool.and_eq_true,
               Bool.or_eq_true, decide_eq_true_eq] at h
    rcases h with ⟨e, he_mem, rfl, hrest⟩
    have he_lt := hfwd e he_mem
    rcases hrest with rfl | hrec
    · exact hsmall e he_mem (Nat.le_refl _)
    · have hmid_lt : e.2 < b := dagReachableAux_lt_of_fwd hfwd k e.2 b hrec
      have he1_ok : dagEdgeSatisfied order (e.1, e.2) = true :=
        hsmall e he_mem (by omega)
      have htail_ok : dagEdgeSatisfied order (e.2, b) = true :=
        ih e.2 b hrec (fun d hd hdspan => hsmall d hd (by omega))
      exact dagEdgeSatisfied_trans order e.1 e.2 b he1_ok htail_ok

private theorem reduced_edges_imply_all_edges (nSteps : Nat) (order : List Nat)
    (edges : List (Nat × Nat))
    (hfwd : ∀ d ∈ edges, d.1 < d.2)
    (hred : ∀ e ∈ transitiveReduction nSteps edges, dagEdgeSatisfied order e = true) :
    ∀ e ∈ edges, dagEdgeSatisfied order e = true := by
  suffices hspan : ∀ span (e : Nat × Nat),
      e ∈ edges → e.2 - e.1 ≤ span → dagEdgeSatisfied order e = true by
    intro e he
    exact hspan (e.2 - e.1) e he (Nat.le_refl _)
  intro span
  induction span with
  | zero =>
    intro e he hle
    have hlt := hfwd e he
    omega
  | succ s ih =>
    intro e he hle
    by_cases himp : isImpliedChord edges e = true
    · simp only [isImpliedChord, List.any_eq_true, Bool.and_eq_true,
                 decide_eq_true_eq] at himp
      rcases himp with ⟨e1, he1_mem, ⟨hu_eq, _⟩, hreach⟩
      have he1_lt := hfwd e1 he1_mem
      have hmid_lt : e1.2 < e.2 :=
        dagReachableAux_lt_of_fwd hfwd (e.2 - e1.2) e1.2 e.2 hreach
      have he1_sat : dagEdgeSatisfied order (e.1, e1.2) = true := by
        have h1 : dagEdgeSatisfied order e1 = true := ih e1 he1_mem (by omega)
        rcases e1 with ⟨a1, b1⟩
        simp only at hu_eq
        subst hu_eq
        exact h1
      have hreach_sat : dagEdgeSatisfied order (e1.2, e.2) = true :=
        dagReachableAux_satisfied_of_span hfwd (e.2 - e1.2) e1.2 e.2 hreach
          (fun d hd hd_le => ih d hd (by omega))
      exact dagEdgeSatisfied_trans order e.1 e1.2 e.2 he1_sat hreach_sat
    · have hfalse : isImpliedChord edges e = false :=
        Bool.eq_false_iff.mpr himp
      have he_red : e ∈ transitiveReduction nSteps edges := by
        unfold transitiveReduction
        rw [List.mem_filter]
        exact ⟨mem_dedupEdges.mpr he, by simp [hfalse]⟩
      exact hred e he_red

/-- Soundness & Completeness of `transitiveReduction`: satisfying reduced Hasse edges is
    logically equivalent to satisfying the full forward causal DAG. -/
theorem checkTwinStepOrder_transitiveReduction_iff (nSteps : Nat) (order : List Nat)
    (edges : List (Nat × Nat))
    (hfwd : ∀ e ∈ edges, e.1 < e.2) :
    checkTwinStepOrder nSteps order (transitiveReduction nSteps edges) = none ↔
      checkTwinStepOrder nSteps order edges = none := by
  unfold checkTwinStepOrder
  simp only [firstViolation_cons, firstViolation_nil, and_true, List.all_eq_true]
  constructor
  · rintro ⟨hperm, hred⟩
    exact ⟨hperm, reduced_edges_imply_all_edges nSteps order edges hfwd hred⟩
  · rintro ⟨hperm, hall⟩
    refine ⟨hperm, ?_⟩
    intro e he
    exact hall e (transitiveReduction_subset nSteps edges he)

private theorem not_mem_erase_of_nodup {α : Type} [BEq α] [LawfulBEq α]
    {x : α} {l : List α} (hn : l.Nodup) : x ∉ l.erase x := by
  induction l with
  | nil => simp
  | cons hd tl ih =>
    rw [List.nodup_cons] at hn
    rcases hn with ⟨hhd, htl⟩
    rw [List.erase_cons]
    split
    · next heq =>
      have hxeq : hd = x := beq_iff_eq.mp heq
      subst hxeq
      exact hhd
    · next hne =>
      intro hm
      cases hm with
      | head => exact absurd (beq_iff_eq.mpr rfl) hne
      | tail _ htail => exact ih htl htail

private theorem mem_of_mem_erase {α : Type} [BEq α] [LawfulBEq α]
    {x y : α} {l : List α} (h : y ∈ l.erase x) : y ∈ l := by
  induction l with
  | nil => cases h
  | cons hd tl ih =>
    rw [List.erase_cons] at h
    split at h
    · exact List.Mem.tail _ h
    · cases h with
      | head => exact List.Mem.head _
      | tail _ htl => exact List.Mem.tail _ (ih htl)

private theorem dagReachableAux_to_dagReachable {edges : List (Nat × Nat)}
    (hfwd : ∀ d ∈ edges, d.1 < d.2) :
    ∀ fuel u v, dagReachableAux edges fuel u v = true → dagReachable edges u v = true := by
  intro fuel
  induction fuel with
  | zero => intro _ _ h; simp [dagReachableAux] at h
  | succ f ih =>
    intro u v h
    simp only [dagReachableAux, List.any_eq_true, Bool.and_eq_true,
               Bool.or_eq_true, decide_eq_true_eq] at h
    rcases h with ⟨e, he_mem, rfl, hrest⟩
    have he_lt := hfwd e he_mem
    unfold dagReachable
    rcases hrest with rfl | hrec
    · cases hdiff : e.2 - e.1 with
      | zero => omega
      | succ m =>
        simp only [dagReachableAux, List.any_eq_true, Bool.and_eq_true,
                   Bool.or_eq_true, decide_eq_true_eq]
        exact ⟨e, he_mem, rfl, Or.inl rfl⟩
    · have hmid_lt : e.2 < v := dagReachableAux_lt_of_fwd hfwd f e.2 v hrec
      have hmid_reach : dagReachableAux edges (v - e.2) e.2 v = true := ih e.2 v hrec
      cases hdiff : v - e.1 with
      | zero => omega
      | succ m =>
        have hm_ge : v - e.2 ≤ m := by omega
        have hmid_m : dagReachableAux edges m e.2 v = true :=
          dagReachableAux_mono_fuel hm_ge e.2 v hmid_reach
        simp only [dagReachableAux, List.any_eq_true, Bool.and_eq_true,
                   Bool.or_eq_true, decide_eq_true_eq]
        exact ⟨e, he_mem, rfl, Or.inr hmid_m⟩

/-- Irredundancy (Minimality) of `transitiveReduction`: removing any retained edge `e`
    strictly destroys reachability from `e.1` to `e.2`. -/
theorem transitiveReduction_irredundant (nSteps : Nat) (edges : List (Nat × Nat))
    (hfwd : ∀ d ∈ edges, d.1 < d.2)
    {e : Nat × Nat} (he : e ∈ transitiveReduction nSteps edges) :
    dagReachable ((transitiveReduction nSteps edges).erase e) e.1 e.2 = false := by
  apply Bool.eq_false_iff.mpr
  intro hreach
  have hnodup := transitiveReduction_nodup nSteps edges
  have he_not_er : e ∉ (transitiveReduction nSteps edges).erase e :=
    not_mem_erase_of_nodup hnodup
  have hsub_er : ∀ d ∈ (transitiveReduction nSteps edges).erase e, d ∈ edges :=
    fun d hd => transitiveReduction_subset nSteps edges (mem_of_mem_erase hd)
  unfold dagReachable at hreach
  cases hspan : e.2 - e.1 with
  | zero =>
    rw [hspan] at hreach
    simp [dagReachableAux] at hreach
  | succ k =>
    rw [hspan] at hreach
    simp only [dagReachableAux, List.any_eq_true, Bool.and_eq_true,
               Bool.or_eq_true, decide_eq_true_eq] at hreach
    rcases hreach with ⟨e1, he1_er, hu_eq, hrest⟩
    rcases hrest with hv_eq | hrec
    · have he1_eq : e1 = e := by
        cases e1
        cases e
        simp only at hu_eq hv_eq
        subst hu_eq hv_eq
        rfl
      exact he_not_er (he1_eq ▸ he1_er)
    · have he1_edges : e1 ∈ edges := hsub_er e1 he1_er
      have hmid_lt : e1.2 < e.2 :=
        dagReachableAux_lt_of_fwd (fun d hd => hfwd d (hsub_er d hd)) k e1.2 e.2 hrec
      have hrec_edges : dagReachableAux edges k e1.2 e.2 = true :=
        dagReachableAux_mono_edges hsub_er k e1.2 e.2 hrec
      have hreach_mid : dagReachable edges e1.2 e.2 = true :=
        dagReachableAux_to_dagReachable hfwd k e1.2 e.2 hrec_edges
      have himp : isImpliedChord edges e = true := by
        simp only [isImpliedChord, List.any_eq_true, Bool.and_eq_true, decide_eq_true_eq]
        exact ⟨e1, he1_edges, ⟨hu_eq, by omega⟩, hreach_mid⟩
      unfold transitiveReduction at he
      rw [List.mem_filter] at he
      simp [himp] at he

/-! ### 1d. Topological-Sort Uniqueness Characterization -/

/-- Candidate step order swapping adjacent steps `i` and `i + 1` in `0..nSteps-1`. -/
def swapAdjacentOrder (nSteps i : Nat) : List Nat :=
  (List.range nSteps).map (fun k => if k = i then i + 1 else if k = i + 1 then i else k)

private theorem indexOfStep_some_iff_of_nodup {x : Nat} :
    ∀ {l : List Nat}, l.Nodup → ∀ {p : Nat},
      indexOfStep x l = some p ↔ ∃ (h : p < l.length), l[p] = x := by
  intro l
  induction l with
  | nil =>
    intro _ p
    simp [indexOfStep]
  | cons hd tl ih =>
    intro hnodup p
    rw [List.nodup_cons] at hnodup
    rcases hnodup with ⟨hhd_not, htl_nodup⟩
    unfold indexOfStep
    split
    · next hxeq =>
      subst hxeq
      cases p with
      | zero => simp
      | succ p' =>
        simp only [Option.some.injEq, Nat.zero_ne_add_one, false_iff]
        rintro ⟨hp, hget⟩
        simp only [List.getElem_cons_succ] at hget
        have hmem : hd ∈ tl := hget ▸ List.getElem_mem (by simp at hp; exact hp)
        exact hhd_not hmem
    · next hxne =>
      cases p with
      | zero =>
        simp only [Option.map_eq_some_iff, Nat.add_one_ne_zero, and_false, exists_false,
                   false_iff]
        rintro ⟨_, hget⟩
        simp only [List.getElem_cons_zero] at hget
        exact hxne hget
      | succ p' =>
        simp only [Option.map_eq_some_iff, Nat.add_right_cancel_iff, exists_eq_right]
        rw [ih htl_nodup]
        constructor
        · rintro ⟨hp', hget⟩
          exact ⟨Nat.succ_lt_succ hp', by simp [hget]⟩
        · rintro ⟨hp, hget⟩
          have hp' : p' < tl.length := by simp at hp; exact hp
          exact ⟨hp', by simpa using hget⟩

private theorem indexOfStep_isSome_of_mem {x : Nat} {l : List Nat} (h : x ∈ l) :
    ∃ p, indexOfStep x l = some p := by
  induction l with
  | nil => cases h
  | cons hd tl ih =>
    unfold indexOfStep
    split
    · exact ⟨0, rfl⟩
    · next hne =>
      cases h with
      | head => exact absurd rfl hne
      | tail _ htl =>
        rcases ih htl with ⟨p, hp⟩
        exact ⟨p + 1, by simp [hp]⟩

private theorem nodup_map_of_inj_on {α β : Type} {f : α → β} {l : List α}
    (hl : l.Nodup) (hinj : ∀ x ∈ l, ∀ y ∈ l, f x = f y → x = y) :
    (l.map f).Nodup := by
  induction l with
  | nil => exact List.Pairwise.nil
  | cons a tl ih =>
    rw [List.nodup_cons] at hl
    rcases hl with ⟨ha_not, htl⟩
    simp only [List.map_cons, List.nodup_cons, List.mem_map]
    refine ⟨?_, ih htl (fun x hx y hy hxy =>
      hinj x (List.Mem.tail _ hx) y (List.Mem.tail _ hy) hxy)⟩
    rintro ⟨b, hb_mem, hfb⟩
    have hba : b = a := hinj b (List.Mem.tail _ hb_mem) a (List.Mem.head _) hfb
    exact ha_not (hba ▸ hb_mem)

private theorem swapAdjacent_inj (i x y : Nat)
    (heq : (if x = i then i + 1 else if x = i + 1 then i else x) =
           (if y = i then i + 1 else if y = i + 1 then i else y)) :
    x = y := by
  split at heq
  all_goals try split at heq
  all_goals try split at heq
  all_goals try split at heq
  all_goals omega

private theorem swapAdjacent_lt (i u v : Nat) (huv : u < v) (hne : ¬(u = i ∧ v = i + 1)) :
    (if u = i then i + 1 else if u = i + 1 then i else u) <
    (if v = i then i + 1 else if v = i + 1 then i else v) := by
  split
  all_goals try split
  all_goals try split
  all_goals try split
  all_goals omega

private theorem swapAdjacentOrder_nodup (nSteps i : Nat) :
    (swapAdjacentOrder nSteps i).Nodup := by
  unfold swapAdjacentOrder
  apply nodup_map_of_inj_on List.nodup_range
  intro x _ y _ heq
  exact swapAdjacent_inj i x y heq

private theorem swapAdjacentOrder_mem_of_lt {nSteps i x : Nat}
    (hi : i + 1 < nSteps) (hx : x < nSteps) :
    x ∈ swapAdjacentOrder nSteps i := by
  unfold swapAdjacentOrder
  rw [List.mem_map]
  by_cases hxi : x = i
  · subst hxi
    exact ⟨x + 1, List.mem_range.mpr hi, by simp⟩
  · by_cases hxi1 : x = i + 1
    · subst hxi1
      exact ⟨i, List.mem_range.mpr (by omega), by simp⟩
    · refine ⟨x, List.mem_range.mpr hx, ?_⟩
      simp [hxi, hxi1]

private theorem indexOfStep_swapAdjacentOrder {nSteps i x : Nat}
    (hi : i + 1 < nSteps) (hx : x < nSteps) :
    indexOfStep x (swapAdjacentOrder nSteps i) =
      some (if x = i then i + 1 else if x = i + 1 then i else x) := by
  rw [indexOfStep_some_iff_of_nodup (swapAdjacentOrder_nodup nSteps i)]
  have hlen : (swapAdjacentOrder nSteps i).length = nSteps := by
    simp [swapAdjacentOrder]
  have hidx_lt :
      (if x = i then i + 1 else if x = i + 1 then i else x) <
        (swapAdjacentOrder nSteps i).length := by
    rw [hlen]
    by_cases h1 : x = i
    · simp [h1, hi]
    · by_cases h2 : x = i + 1
      · simp [h2]; omega
      · simp [h1, h2, hx]
  refine ⟨hidx_lt, ?_⟩
  simp only [swapAdjacentOrder, List.getElem_map, List.getElem_range]
  by_cases h1 : x = i
  · subst h1
    simp
  · by_cases h2 : x = i + 1
    · subst h2
      simp
    · simp [h1, h2]

private theorem contiguous_edge_not_implied {edges : List (Nat × Nat)}
    (hfwd : ∀ d ∈ edges, d.1 < d.2) (i : Nat) :
    isImpliedChord edges (i, i + 1) = false := by
  apply Bool.eq_false_iff.mpr
  intro himp
  simp only [isImpliedChord, List.any_eq_true, Bool.and_eq_true, decide_eq_true_eq] at himp
  rcases himp with ⟨e1, he1_mem, ⟨hu_eq, _⟩, hreach⟩
  have h1 := hfwd e1 he1_mem
  have h2 := dagReachableAux_lt_of_fwd hfwd (i + 1 - e1.2) e1.2 (i + 1) hreach
  omega

/-- Topological-Sort Uniqueness Theorem: `List.range nSteps` is the unique valid topological
    ordering accepted by `checkTwinStepOrder` iff `transitiveReduction` contains every
    consecutive step edge `(i, i + 1)` for `i + 1 < nSteps`. -/
theorem topological_order_unique_iff_contiguous_chain
    (nSteps : Nat) (edges : List (Nat × Nat))
    (hfwd : ∀ e ∈ edges, e.1 < e.2 ∧ e.2 < nSteps) :
    (∀ order, checkTwinStepOrder nSteps order edges = none → order = List.range nSteps) ↔
      (∀ i, i + 1 < nSteps → (i, i + 1) ∈ transitiveReduction nSteps edges) := by
  have hfwd_lt : ∀ e ∈ edges, e.1 < e.2 := fun e he => (hfwd e he).1
  constructor
  · intro huniq i hi
    have hno_chord : isImpliedChord edges (i, i + 1) = false :=
      contiguous_edge_not_implied hfwd_lt i
    have hin_edges : (i, i + 1) ∈ edges := by
      apply Classical.byContradiction
      intro hnot_in
      let sw := swapAdjacentOrder nSteps i
      have hsw_ok : checkTwinStepOrder nSteps sw edges = none := by
        unfold checkTwinStepOrder
        simp only [firstViolation_cons, firstViolation_nil, and_true,
                   Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true]
        refine ⟨⟨by simp [sw, swapAdjacentOrder], ?_⟩, ?_⟩
        · intro x hx
          rw [List.mem_range] at hx
          exact List.contains_iff_mem.mpr (swapAdjacentOrder_mem_of_lt hi hx)
        · intro e he
          have ⟨he_lt, he_ub⟩ := hfwd e he
          have hu_lt : e.1 < nSteps := by omega
          have hne_pair : ¬(e.1 = i ∧ e.2 = i + 1) := by
            rintro ⟨hu, hv⟩
            apply hnot_in
            cases e
            simp only at hu hv
            subst hu hv
            exact he
          unfold dagEdgeSatisfied
          rw [indexOfStep_swapAdjacentOrder hi hu_lt,
              indexOfStep_swapAdjacentOrder hi he_ub]
          simp only [decide_eq_true_eq]
          exact swapAdjacent_lt i e.1 e.2 he_lt hne_pair
      have heq := huniq sw hsw_ok
      have hi_lt_sw : i < sw.length := by simp [sw, swapAdjacentOrder]; omega
      have hget_sw : sw[i] = i + 1 := by simp [sw, swapAdjacentOrder]
      have hget_rng : (List.range nSteps)[i]'(by simp; omega) = i := by simp
      have hcongr : sw[i] = (List.range nSteps)[i]'(by simp; omega) := by simp [heq]
      omega
    unfold transitiveReduction
    rw [List.mem_filter]
    exact ⟨mem_dedupEdges.mpr hin_edges, by simp [hno_chord]⟩
  · intro hchain order hok
    have hok_red : checkTwinStepOrder nSteps order (transitiveReduction nSteps edges) = none :=
      (checkTwinStepOrder_transitiveReduction_iff nSteps order edges hfwd_lt).mpr hok
    have ⟨hlen, hnodup, hmem, hedges_red⟩ :=
      checkTwinStepOrder_nodup_and_mem_iff nSteps order (transitiveReduction nSteps edges) hok_red
    -- Every step `k < nSteps` has a unique position `posOf k < nSteps` in `order`.
    have hpos_step : ∀ k (hk : k + 1 < nSteps) p1 p2,
        indexOfStep k order = some p1 →
        indexOfStep (k + 1) order = some p2 →
        p1 < p2 := by
      intro k hk p1 p2 hp1 hp2
      have hsat := hedges_red (k, k + 1) (hchain k hk)
      unfold dagEdgeSatisfied at hsat
      simp only [hp1, hp2, decide_eq_true_eq] at hsat
      exact hsat
    have hpos_ge : ∀ k, k < nSteps → ∀ p, indexOfStep k order = some p → k ≤ p := by
      intro k
      induction k with
      | zero => intro _ _ _; exact Nat.zero_le _
      | succ k' ih =>
        intro hk p hp
        rcases indexOfStep_isSome_of_mem ((hmem k').mpr (by omega)) with ⟨p', hp'⟩
        have h1 := ih (by omega) p' hp'
        have h2 := hpos_step k' hk p' p hp' hp
        omega
    have hpos_le : ∀ d k, k < nSteps → nSteps - 1 - k = d →
        ∀ p, indexOfStep k order = some p → p ≤ k := by
      intro d
      induction d with
      | zero =>
        intro k hk hd p hp
        have ⟨hp_lt, _⟩ := (indexOfStep_some_iff_of_nodup hnodup).mp hp
        omega
      | succ d' ih =>
        intro k hk hd p hp
        have hk1 : k + 1 < nSteps := by omega
        rcases indexOfStep_isSome_of_mem ((hmem (k + 1)).mpr hk1) with ⟨p2, hp2⟩
        have hle2 := ih (k + 1) hk1 (by omega) p2 hp2
        have hlt12 := hpos_step k hk1 p p2 hp hp2
        omega
    apply List.ext_getElem
    · simp [hlen]
    · intro k hk1 hk2
      have hk_lt : k < nSteps := by simp at hk2; exact hk2
      rcases indexOfStep_isSome_of_mem ((hmem k).mpr hk_lt) with ⟨p, hp⟩
      have hge := hpos_ge k hk_lt p hp
      have hle := hpos_le (nSteps - 1 - k) k hk_lt rfl p hp
      have hpk : p = k := by omega
      subst hpk
      have ⟨_, hval⟩ := (indexOfStep_some_iff_of_nodup hnodup).mp hp
      simp [hval]

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

#guard transitiveReduction 4 [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)] == [(0, 1), (1, 2), (2, 3)]

#guard checkTwinStepOrder 4 [0, 2, 1, 3] (transitiveReduction 4 [(0, 1), (0, 2), (1, 3), (2, 3)]) == none

#guard swapAdjacentOrder 4 1 == [0, 2, 1, 3]

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

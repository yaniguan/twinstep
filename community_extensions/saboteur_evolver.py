"""Idea #15: Evolutionary Protocol Saboteur (LeanEvolve-style Red-Teaming).

Inspired by Lean4Agent and LeanEvolve (arXiv:2606.06523), this module runs a
multi-generation evolutionary mutation loop over multi-step liquid-handling
protocols (swapping causal barrier steps, injecting reverse-concentration tip
back-entry, dropping or misplacing >= 80 C seals, and shaving recoverable
dead-volume reserves) to actively discover stealth protocol corruptions that
evade weak heuristic gates while failing stateful physical and causal invariants
formalized in `TwinStepLean`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import random
from typing import Any

# Fixed-point scale: 10 tenths_uL == 1.0 uL
TENTHS_PER_UL = 10
WELL_CAP_TENTHS = 3000       # 300.0 uL capacity per assay well
DEAD_VOL_TENTHS = 150        # 15.0 uL unrecoverable dead-volume reserve
STOCK_INIT_VOL_TENTHS = 1200  # 120.0 uL initial stock volume
BUFFER_INIT_VOL_TENTHS = 2500  # 250.0 uL initial buffer volume


@dataclass(frozen=True)
class SaboteurStep:
    """Typed wet-lab protocol step with step_id for causal DAG tracking."""

    step_id: int
    op: str
    src: str = ""
    dst: str = ""
    vol_tenths: int = 0
    temp_c: int = 25
    duration_s: int = 0
    cycles: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "op": self.op,
            "src": self.src,
            "dst": self.dst,
            "vol_uL": round(self.vol_tenths / TENTHS_PER_UL, 1),
            "temp_c": self.temp_c,
            "duration_s": self.duration_s,
            "cycles": self.cycles,
        }


@dataclass(frozen=True)
class SeedProtocol:
    """Reference clean multi-step wet-lab protocol and its causal DAG edges."""

    name: str
    steps: tuple[SaboteurStep, ...]
    dag_edges: tuple[tuple[int, int], ...]
    init_vols_tenths: tuple[tuple[str, int], ...]
    init_solutes: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class GateVerdict:
    """Verification outcome from one of the three hierarchical protocol gates."""

    passed: bool
    violations: tuple[str, ...]


@dataclass(frozen=True)
class HardNegativeTwinPair:
    """Curated red-team twin pair that evades heuristic gates yet fails Lean."""

    protocol_name: str
    generation: int
    mutation_ops: tuple[str, ...]
    evaded_gates: tuple[str, ...]
    violated_lean_rules: tuple[str, ...]
    clean_steps: tuple[dict[str, Any], ...]
    sabotaged_steps: tuple[dict[str, Any], ...]


def index_of_step(step_id: int, order: list[int] | tuple[int, ...]) -> int | None:
    """Mirror of `TwinStepLean.indexOfStep`."""
    for idx, val in enumerate(order):
        if val == step_id:
            return idx
    return None


def dag_edge_satisfied(
    order: list[int] | tuple[int, ...],
    edge: tuple[int, int],
) -> bool:
    """Mirror of `TwinStepLean.dagEdgeSatisfied`."""
    pu = index_of_step(edge[0], order)
    pv = index_of_step(edge[1], order)
    if pu is None or pv is None:
        return False
    return pu < pv


def check_twin_step_order(
    n_steps: int,
    order: list[int] | tuple[int, ...],
    dag_edges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> GateVerdict:
    """Exact Python mirror of `TwinStepLean.checkTwinStepOrder`."""
    if len(order) != n_steps or any(i not in order for i in range(n_steps)):
        return GateVerdict(False, ("bijective_step_permutation",))
    for edge in dag_edges:
        if not dag_edge_satisfied(order, edge):
            return GateVerdict(False, ("causal_dag_satisfied",))
    return GateVerdict(True, ())


def build_seed_protocols() -> tuple[SeedProtocol, ...]:
    """Build clean serial-dilution + high-temp PCR / ELISA seed protocols."""
    pcr_dilution = SeedProtocol(
        name="serial_dilution_thermocycle",
        steps=(
            SaboteurStep(0, "pick_tip"),
            SaboteurStep(1, "transfer", src="BUFFER", dst="B1", vol_tenths=500),
            SaboteurStep(2, "transfer", src="BUFFER", dst="C1", vol_tenths=500),
            SaboteurStep(3, "drop_tip"),
            SaboteurStep(4, "pick_tip"),
            SaboteurStep(5, "transfer", src="A1", dst="B1", vol_tenths=500),
            SaboteurStep(6, "mix", dst="B1", vol_tenths=500, cycles=5),
            SaboteurStep(7, "transfer", src="B1", dst="C1", vol_tenths=500),
            SaboteurStep(8, "mix", dst="C1", vol_tenths=500, cycles=5),
            SaboteurStep(9, "drop_tip"),
            SaboteurStep(10, "seal"),
            SaboteurStep(11, "incubate", temp_c=95, duration_s=180),
            SaboteurStep(12, "unseal"),
            SaboteurStep(13, "read"),
        ),
        dag_edges=(
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (1, 5),
            (4, 5),
            (5, 6),
            (2, 7),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (12, 13),
        ),
        init_vols_tenths=(
            ("BUFFER", BUFFER_INIT_VOL_TENTHS),
            ("A1", STOCK_INIT_VOL_TENTHS),
            ("B1", 0),
            ("C1", 0),
        ),
        init_solutes=(
            ("BUFFER", 0),
            ("A1", 1200),
            ("B1", 0),
            ("C1", 0),
        ),
    )

    denature_assay = SeedProtocol(
        name="mastermix_denature_read",
        steps=(
            SaboteurStep(0, "pick_tip"),
            SaboteurStep(1, "transfer", src="BUFFER", dst="D1", vol_tenths=600),
            SaboteurStep(2, "transfer", src="BUFFER", dst="H1", vol_tenths=600),
            SaboteurStep(3, "drop_tip"),
            SaboteurStep(4, "pick_tip"),
            SaboteurStep(5, "transfer", src="A1", dst="D1", vol_tenths=600),
            SaboteurStep(6, "mix", dst="D1", vol_tenths=600, cycles=4),
            SaboteurStep(7, "transfer", src="D1", dst="H1", vol_tenths=600),
            SaboteurStep(8, "mix", dst="H1", vol_tenths=600, cycles=4),
            SaboteurStep(9, "drop_tip"),
            SaboteurStep(10, "seal"),
            SaboteurStep(11, "incubate", temp_c=88, duration_s=240),
            SaboteurStep(12, "unseal"),
            SaboteurStep(13, "read"),
        ),
        dag_edges=(
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (1, 5),
            (4, 5),
            (5, 6),
            (2, 7),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (12, 13),
        ),
        init_vols_tenths=(
            ("BUFFER", BUFFER_INIT_VOL_TENTHS),
            ("A1", STOCK_INIT_VOL_TENTHS),
            ("D1", 0),
            ("H1", 0),
        ),
        init_solutes=(
            ("BUFFER", 0),
            ("A1", 1200),
            ("D1", 0),
            ("H1", 0),
        ),
    )
    return (pcr_dilution, denature_assay)


def check_gen0_scalar_volume_gate(
    seed_proto: SeedProtocol,
    candidate_steps: tuple[SaboteurStep, ...],
) -> GateVerdict:
    """Gen-0 weak scalar bounds & gross-volume heuristic gate.

    Checks per-step scalar bounds and naive gross container volume (>= 0,
    ignoring recoverable dead-volume reserves, tip concentration state,
    seal timing, and causal step DAGs).
    """
    if not candidate_steps or candidate_steps[-1].op != "read":
        return GateVerdict(False, ("missing_terminal_read",))

    vols = dict(seed_proto.init_vols_tenths)
    for st in candidate_steps:
        if st.op == "transfer":
            if not (10 <= st.vol_tenths <= 10_000) or st.src == st.dst:
                return GateVerdict(False, ("scalar_transfer_bounds",))
            if vols.get(st.src, 0) < st.vol_tenths:
                return GateVerdict(False, ("gross_volume_underflow",))
            if vols.get(st.dst, 0) + st.vol_tenths > WELL_CAP_TENTHS:
                return GateVerdict(False, ("gross_volume_overflow",))
            vols[st.src] = vols.get(st.src, 0) - st.vol_tenths
            vols[st.dst] = vols.get(st.dst, 0) + st.vol_tenths
        elif st.op == "mix":
            if not (10 <= st.vol_tenths <= 10_000) or st.cycles < 1:
                return GateVerdict(False, ("scalar_mix_bounds",))
        elif st.op == "incubate":
            if not (4 <= st.temp_c <= 98) or st.duration_s <= 0:
                return GateVerdict(False, ("scalar_incubate_bounds",))
    return GateVerdict(True, ())


def check_gen1_bitmask_gate(
    seed_proto: SeedProtocol,
    candidate_steps: tuple[SaboteurStep, ...],
) -> GateVerdict:
    """Gen-1 stateless op-bitmask & initialized-well presence verifier.

    Extends Gen-0 with op-presence bitmasks (requires `pick_tip`, `transfer`,
    `mix`, `incubate`, `read`, and `seal` somewhere in the trace whenever any
    incubation uses >= 80 C) plus a coarse initialized-well bitmask.
    """
    g0 = check_gen0_scalar_volume_gate(seed_proto, candidate_steps)
    if not g0.passed:
        return g0

    seen_ops = {st.op for st in candidate_steps}
    required_ops = {"pick_tip", "transfer", "mix", "incubate", "read"}
    if not required_ops.issubset(seen_ops):
        return GateVerdict(False, ("missing_required_op_bitmask",))

    has_high_temp = any(
        st.op == "incubate" and st.temp_c >= 80 for st in candidate_steps
    )
    if has_high_temp and "seal" not in seen_ops:
        return GateVerdict(False, ("missing_seal_bitmask",))

    initialized = {k for k, v in seed_proto.init_vols_tenths if v > 0}
    tip_seen = False
    for st in candidate_steps:
        if st.op == "pick_tip":
            tip_seen = True
        elif st.op == "transfer":
            if not tip_seen or st.src not in initialized:
                return GateVerdict(False, ("uninitialized_source_bitmask",))
            initialized.add(st.dst)
        elif st.op == "mix":
            if not tip_seen or st.dst not in initialized:
                return GateVerdict(False, ("uninitialized_mix_bitmask",))

    return GateVerdict(True, ())


def check_gen2_stateful_lean_gate(
    seed_proto: SeedProtocol,
    candidate_steps: tuple[SaboteurStep, ...],
) -> GateVerdict:
    """Gen-2 stateful physical & causal DAG gate mirroring `TwinStepLean`."""
    violations: list[str] = []

    order = [st.step_id for st in candidate_steps]
    dag_verdict = check_twin_step_order(
        n_steps=len(seed_proto.steps),
        order=order,
        dag_edges=seed_proto.dag_edges,
    )
    if not dag_verdict.passed:
        violations.extend(dag_verdict.violations)

    vols = dict(seed_proto.init_vols_tenths)
    sols = dict(seed_proto.init_solutes)
    tip_mounted = False
    tip_vol = 0
    tip_sol = 0
    tip_touched_sample = False
    sealed = False

    for st in candidate_steps:
        if st.op == "pick_tip":
            if tip_mounted:
                violations.append("pick_tip_requires_empty_mount")
            tip_mounted = True
            tip_vol = 0
            tip_sol = 0
            tip_touched_sample = False
        elif st.op == "drop_tip":
            if not tip_mounted:
                violations.append("drop_tip_requires_mounted_tip")
            tip_mounted = False
            tip_vol = 0
            tip_sol = 0
            tip_touched_sample = False
        elif st.op == "seal":
            sealed = True
        elif st.op == "unseal":
            sealed = False
        elif st.op == "transfer":
            if not tip_mounted or sealed:
                violations.append("pipette_requires_mounted_tip_and_unsealed_plate")
                continue
            src_v = vols.get(st.src, 0)
            src_s = sols.get(st.src, 0)
            dst_v = vols.get(st.dst, 0)
            dst_s = sols.get(st.dst, 0)

            if DEAD_VOL_TENTHS + st.vol_tenths > src_v:
                violations.append("recoverable_dead_volume_reserve")
            if dst_v + st.vol_tenths > WELL_CAP_TENTHS:
                violations.append("no_well_overfill")

            # Tip entry into source well: check sample->buffer & reverse-conc
            if tip_touched_sample and st.src == "BUFFER":
                violations.append("no_reverse_conc_tip_back_entry")
            if tip_sol > 0 and src_v > 0 and tip_vol > 0:
                if src_s * tip_vol > tip_sol * src_v:
                    violations.append("no_reverse_conc_tip_back_entry")

            moved_sol = 0 if src_v <= 0 else (src_s * st.vol_tenths) // src_v
            vols[st.src] = max(0, src_v - st.vol_tenths)
            sols[st.src] = max(0, src_s - moved_sol)
            new_dst_v = dst_v + st.vol_tenths
            new_dst_s = dst_s + moved_sol

            # Tip entry into destination well if destination already holds solute
            if tip_sol > 0 and dst_v > 0 and tip_vol > 0:
                if dst_s * tip_vol > tip_sol * dst_v:
                    violations.append("no_reverse_conc_tip_back_entry")

            vols[st.dst] = new_dst_v
            sols[st.dst] = new_dst_s
            tip_vol = new_dst_v
            tip_sol = new_dst_s
            if new_dst_s > 0 or src_s > 0:
                tip_touched_sample = True
        elif st.op == "mix":
            if not tip_mounted or sealed:
                violations.append("mix_requires_mounted_tip_and_unsealed_plate")
                continue
            w_v = vols.get(st.dst, 0)
            w_s = sols.get(st.dst, 0)
            if tip_sol > 0 and w_v > 0 and tip_vol > 0:
                if w_s * tip_vol > tip_sol * w_v:
                    violations.append("no_reverse_conc_tip_back_entry")
            tip_vol = w_v
            tip_sol = w_s
            if w_s > 0:
                tip_touched_sample = True
        elif st.op == "incubate":
            if st.temp_c >= 80 and not sealed:
                violations.append("high_temp_seal_required")
        elif st.op == "read":
            if sealed:
                violations.append("optical_read_requires_unsealed_plate")

    dedup = tuple(dict.fromkeys(violations))
    return GateVerdict(len(dedup) == 0, dedup)


def _apply_mutation(
    seed_proto: SeedProtocol,
    steps: tuple[SaboteurStep, ...],
    mut_kind: str,
    stealth_level: int,
    rng: random.Random,
) -> tuple[SaboteurStep, ...]:
    """Apply one evolutionary sabotage operator (coarse at level 0, stealth at >=1)."""
    out = list(steps)

    if mut_kind == "swap_causal_barrier":
        if stealth_level == 0:
            # Coarse swap: swap buffer dispense (1) with sample transfer (5)
            if len(out) > 5:
                out[1], out[5] = out[5], out[1]
        else:
            # Stealth causal swap: swap upstream homogenization mix (6) with
            # downstream serial transfer (7); both wells are already initialized!
            choices = [(6, 7), (7, 8)]
            i, j = choices[rng.randrange(len(choices))]
            if len(out) > j:
                out[i], out[j] = out[j], out[i]

    elif mut_kind == "reverse_conc_back_entry":
        if stealth_level == 0:
            # Coarse corruption: delete drop_tip completely
            out = [st for st in out if st.op != "drop_tip"]
        else:
            # Stealth back-entry: after mixing dilute well (index 8), touch/mix
            # concentrated upstream well (step 6 well) before dropping tip,
            # or swap mix(C1/H1) before mix(B1/D1) after both transfers!
            if len(out) > 8 and out[6].op == "mix" and out[8].op == "mix":
                # Move dilute transfer+mix before upstream mix so dirty dilute tip
                # re-enters 2x-concentrated upstream well!
                out[6], out[7], out[8] = out[7], out[8], out[6]

    elif mut_kind == "drop_high_temp_seal":
        if stealth_level == 0:
            # Coarse seal drop: remove seal & unseal steps (caught by Gen-1 bitmask)
            out = [st for st in out if st.op not in ("seal", "unseal")]
        else:
            # Stealth seal inversion: keep `seal` in protocol bitmask, but swap
            # `seal` (10) and `incubate >= 80C` (11) so incubation runs unsealed!
            for idx in range(len(out) - 1):
                if out[idx].op == "seal" and out[idx + 1].op == "incubate":
                    out[idx], out[idx + 1] = out[idx + 1], out[idx]
                    break

    elif mut_kind == "shave_dead_volume":
        if stealth_level == 0:
            # Coarse over-aspiration exceeding gross stock volume (caught by Gen-0)
            for idx, st in enumerate(out):
                if st.op == "transfer" and st.src == "A1":
                    out[idx] = replace(st, vol_tenths=STOCK_INIT_VOL_TENTHS + 100)
                    break
        else:
            # Stealth dead-volume shave: aspirate 110.0 uL from 120.0 uL stock
            # (gross remaining = 10.0 uL >= 0 passes Gen-0 & Gen-1, but violates
            # 15.0 uL physical dead-volume reserve!)
            shaved = STOCK_INIT_VOL_TENTHS - (DEAD_VOL_TENTHS - rng.choice((30, 50, 70)))
            for idx, st in enumerate(out):
                if st.op == "transfer" and st.src == "A1":
                    out[idx] = replace(st, vol_tenths=shaved)
                    break
    else:
        raise ValueError(f"Unknown mutation operator: {mut_kind}")

    return tuple(out)


def evolve_saboteurs(
    generations: int = 10,
    pop_size: int = 15,
    seed: int = 2026,
) -> dict[str, Any]:
    """Run LeanEvolve-style evolutionary red-teaming over wet-lab protocols."""
    if generations < 1 or pop_size < 1:
        raise ValueError("generations and pop_size must be >= 1")

    rng = random.Random(seed)
    seeds = build_seed_protocols()
    operators = (
        "swap_causal_barrier",
        "reverse_conc_back_entry",
        "drop_high_temp_seal",
        "shave_dead_volume",
    )

    # Verify clean seed protocols pass all 3 gates with 0 false positives
    for sp in seeds:
        assert check_gen0_scalar_volume_gate(sp, sp.steps).passed
        assert check_gen1_bitmask_gate(sp, sp.steps).passed
        assert check_gen2_stateful_lean_gate(sp, sp.steps).passed

    # Archive of elite stealth genomes: (seed_idx, ops_tuple, stealth_level, steps, fitness)
    elites: list[tuple[int, tuple[str, ...], int, tuple[SaboteurStep, ...], float]] = []

    total_candidates = 0
    gen0_evaded_total = 0
    gen1_evaded_total = 0
    gen2_evaded_total = 0
    generation_history: list[dict[str, Any]] = []
    curated_map: dict[str, HardNegativeTwinPair] = {}

    for gen in range(generations):
        gen_g0 = 0
        gen_g1 = 0
        gen_g2 = 0
        gen_pool: list[
            tuple[int, tuple[str, ...], int, tuple[SaboteurStep, ...], float]
        ] = []

        for slot in range(pop_size):
            if gen == 0 or not elites or rng.random() < 0.20:
                sp_idx = (gen + slot) % len(seeds)
                sp = seeds[sp_idx]
                op = operators[(slot + gen) % len(operators)]
                # Generation 0 explores mostly coarse mutations; stealth rises via selection
                stealth = 1 if (gen > 0 and rng.random() < 0.5) else (1 if slot % 4 == 0 else 0)
                cand_steps = _apply_mutation(sp, sp.steps, op, stealth, rng)
                ops_used: tuple[str, ...] = (op,)
            else:
                # Tournament selection over archive elites
                k_tourn = min(3, len(elites))
                contenders = rng.sample(elites, k=k_tourn)
                parent = max(contenders, key=lambda item: item[4])
                sp_idx, parent_ops, parent_stealth, parent_steps, _ = parent
                sp = seeds[sp_idx]
                next_op = operators[rng.randrange(len(operators))]
                # Evolutionary refinement promotes stealth level and/or chains operators
                stealth = 1 if (parent_stealth == 1 or rng.random() < 0.85) else 0
                base_steps = parent_steps if len(parent_ops) < 2 else sp.steps
                cand_steps = _apply_mutation(sp, base_steps, next_op, stealth, rng)
                ops_used = (
                    parent_ops + (next_op,) if len(parent_ops) < 2 else (next_op,)
                )

            v0 = check_gen0_scalar_volume_gate(sp, cand_steps)
            v1 = check_gen1_bitmask_gate(sp, cand_steps)
            v2 = check_gen2_stateful_lean_gate(sp, cand_steps)

            # A true saboteur candidate violates at least one physical/causal rule
            is_sabotage = not v2.passed
            evades_g0 = is_sabotage and v0.passed
            evades_g1 = is_sabotage and v1.passed
            evades_g2 = is_sabotage and v2.passed

            total_candidates += 1
            gen_g0 += int(evades_g0)
            gen_g1 += int(evades_g1)
            gen_g2 += int(evades_g2)

            # Fitness rewards fooling stronger heuristic verifiers + distinct rule richness
            fitness = (
                (1.0 if is_sabotage else -5.0)
                + (1.5 if evades_g0 else 0.0)
                + (2.5 if evades_g1 else 0.0)
                + 0.25 * float(len(v2.violations))
            )
            gen_pool.append((sp_idx, ops_used, stealth, cand_steps, fitness))

            if evades_g0 and evades_g1 and not v2.passed:
                sig = json.dumps(
                    [sp.name, [st.to_dict() for st in cand_steps]], sort_keys=True
                )
                if sig not in curated_map:
                    curated_map[sig] = HardNegativeTwinPair(
                        protocol_name=sp.name,
                        generation=gen + 1,
                        mutation_ops=ops_used,
                        evaded_gates=("gen0_scalar_volume_gate", "gen1_bitmask_gate"),
                        violated_lean_rules=v2.violations,
                        clean_steps=tuple(st.to_dict() for st in sp.steps),
                        sabotaged_steps=tuple(st.to_dict() for st in cand_steps),
                    )

        gen0_evaded_total += gen_g0
        gen1_evaded_total += gen_g1
        gen2_evaded_total += gen_g2

        gen_pool.sort(key=lambda item: item[4], reverse=True)
        elites = (gen_pool[:6] + elites)[:10]

        generation_history.append(
            {
                "generation": gen + 1,
                "candidates": pop_size,
                "gen0_evasion_pct": round(100.0 * gen_g0 / pop_size, 2),
                "gen1_evasion_pct": round(100.0 * gen_g1 / pop_size, 2),
                "gen2_lean_evasion_pct": round(100.0 * gen_g2 / pop_size, 2),
            }
        )

    return {
        "idea": "Idea #15: Evolutionary Protocol Saboteur (LeanEvolve Red-Teaming)",
        "generations": generations,
        "candidates_per_generation": pop_size,
        "total_candidate_mutations": total_candidates,
        "evasion_rates_pct": {
            "gen0_scalar_volume_gate": round(
                100.0 * gen0_evaded_total / total_candidates, 2
            ),
            "gen1_bitmask_gate": round(
                100.0 * gen1_evaded_total / total_candidates, 2
            ),
            "gen2_stateful_lean_gate": round(
                100.0 * gen2_evaded_total / total_candidates, 2
            ),
        },
        "gen1_first_vs_last_evasion_pct": {
            "gen_1": generation_history[0]["gen1_evasion_pct"],
            "gen_last": generation_history[-1]["gen1_evasion_pct"],
        },
        "curated_hard_negative_count": len(curated_map),
        "curated_hard_negatives": [
            {
                "protocol_name": pair.protocol_name,
                "generation": pair.generation,
                "mutation_ops": list(pair.mutation_ops),
                "evaded_gates": list(pair.evaded_gates),
                "violated_lean_rules": list(pair.violated_lean_rules),
            }
            for pair in list(curated_map.values())[:8]
        ],
        "generation_history": generation_history,
    }


def run_benchmark(
    generations: int = 10,
    candidates_per_generation: int = 15,
    seed: int = 2026,
) -> dict[str, Any]:
    """Run 10-generation (150-candidate) evolutionary saboteur red-team benchmark."""
    return evolve_saboteurs(
        generations=generations,
        pop_size=candidates_per_generation,
        seed=seed,
    )


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2))

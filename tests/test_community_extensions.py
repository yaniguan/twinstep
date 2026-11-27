"""Tests for Idea #15: Evolutionary Protocol Saboteur & TwinStepLean."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from community_extensions import (  # noqa: E402
    build_seed_protocols,
    check_gen0_scalar_volume_gate,
    check_gen1_bitmask_gate,
    check_gen2_stateful_lean_gate,
    check_twin_step_order,
    is_topological_order_unique,
    run_benchmark,
    transitive_reduction,
)
from community_extensions.saboteur_evolver import dag_edge_satisfied  # noqa: E402


def test_clean_seed_protocols_pass_all_three_gates() -> None:
    seeds = build_seed_protocols()
    assert len(seeds) >= 2
    for sp in seeds:
        assert check_gen0_scalar_volume_gate(sp, sp.steps).passed is True
        assert check_gen1_bitmask_gate(sp, sp.steps).passed is True
        verdict = check_gen2_stateful_lean_gate(sp, sp.steps)
        assert verdict.passed is True
        assert verdict.violations == ()


def test_dag_edge_and_twin_step_order_mirror_lean() -> None:
    edges = [(0, 1), (1, 2), (2, 3)]
    assert check_twin_step_order(4, [0, 1, 2, 3], edges).passed is True

    # Violated DAG edge rejected
    swapped = check_twin_step_order(4, [0, 2, 1, 3], edges)
    assert swapped.passed is False
    assert "causal_dag_satisfied" in swapped.violations

    # Non-permutation rejected
    dup = check_twin_step_order(4, [0, 1, 1, 3], edges)
    assert dup.passed is False
    assert "bijective_step_permutation" in dup.violations

    # Irreflexivity and asymmetry
    assert dag_edge_satisfied([0, 1, 2], (1, 1)) is False
    assert dag_edge_satisfied([2, 0, 1], (0, 1)) is True
    assert dag_edge_satisfied([2, 0, 1], (1, 0)) is False


def test_stealth_sabotage_families_evade_weak_gates_but_fail_lean() -> None:
    sp = build_seed_protocols()[0]

    # 1. Dead-volume shave (110.0 uL from 120.0 uL stock with 15.0 uL dead reserve)
    shaved = list(sp.steps)
    shaved[5] = replace(shaved[5], vol_tenths=1100)
    shaved_t = tuple(shaved)
    assert check_gen0_scalar_volume_gate(sp, shaved_t).passed is True
    assert check_gen1_bitmask_gate(sp, shaved_t).passed is True
    v2_shave = check_gen2_stateful_lean_gate(sp, shaved_t)
    assert v2_shave.passed is False
    assert "recoverable_dead_volume_reserve" in v2_shave.violations

    # 2. Stealth >= 80 C seal inversion (seal kept in protocol bitmask, placed after 95 C)
    unsealed = list(sp.steps)
    unsealed[10], unsealed[11] = unsealed[11], unsealed[10]
    unsealed_t = tuple(unsealed)
    assert check_gen0_scalar_volume_gate(sp, unsealed_t).passed is True
    assert check_gen1_bitmask_gate(sp, unsealed_t).passed is True
    v2_seal = check_gen2_stateful_lean_gate(sp, unsealed_t)
    assert v2_seal.passed is False
    assert "high_temp_seal_required" in v2_seal.violations


def test_twinstep_lean_module_hygiene() -> None:
    lean_file = REPO_ROOT / "lean" / "TwinStepLean.lean"
    assert lean_file.is_file()
    src = lean_file.read_text(encoding="utf-8")
    for required in (
        "def dagEdgeSatisfied",
        "def checkTwinStepOrder",
        "def transitiveReduction",
        "theorem violated_dag_edge_rejected",
        "theorem dagEdgeSatisfied_trans",
        "theorem checkTwinStepOrder_bijective",
        "theorem checkTwinStepOrder_nodup_and_mem_iff",
        "theorem checkTwinStepOrder_transitiveReduction_iff",
        "theorem transitiveReduction_irredundant",
        "theorem topological_order_unique_iff_contiguous_chain",
        "theorem sabotaged_candidate_rejected",
    ):
        assert required in src
    assert re.search(r"\bsorry\b", src) is None
    assert re.search(r"\baxiom\b", src) is None


def test_run_benchmark_10_generations_150_candidates() -> None:
    report = run_benchmark(generations=10, candidates_per_generation=15, seed=2026)
    assert report["generations"] == 10
    assert report["total_candidate_mutations"] == 150

    rates = report["evasion_rates_pct"]
    assert rates["gen0_scalar_volume_gate"] >= rates["gen1_bitmask_gate"]
    assert rates["gen1_bitmask_gate"] > 50.0
    assert rates["gen2_stateful_lean_gate"] == 0.0
    assert report["curated_hard_negative_count"] >= 10
    assert (
        report["gen1_first_vs_last_evasion_pct"]["gen_last"]
        > report["gen1_first_vs_last_evasion_pct"]["gen_1"]
    )

    with pytest.raises(ValueError):
        run_benchmark(generations=0)

    # Issue #33: Curated hard negatives preserve clean_steps and sabotaged_steps
    assert len(report["curated_hard_negatives"]) == report["curated_hard_negative_count"]
    first_hn = report["curated_hard_negatives"][0]
    assert "clean_steps" in first_hn and "sabotaged_steps" in first_hn
    assert first_hn["clean_steps"] != first_hn["sabotaged_steps"]


def test_issue_31_transitive_reduction_and_uniqueness_mirror_parity() -> None:
    import itertools
    from twinstep.dependencies import _transitive_reduction

    # Chorded chain: (0, 2) and (0, 3) are implied by 0 -> 1 -> 2 -> 3
    chain_chords = [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)]
    red_chain = transitive_reduction(4, chain_chords)
    assert red_chain == [(0, 1), (1, 2), (2, 3)]
    assert sorted(red_chain) == list(_transitive_reduction(4, set(chain_chords)))
    assert is_topological_order_unique(4, chain_chords) is True

    valid_chain_orders = [
        list(perm)
        for perm in itertools.permutations(range(4))
        if check_twin_step_order(4, list(perm), chain_chords).passed
    ]
    assert valid_chain_orders == [[0, 1, 2, 3]]

    # Diamond DAG: 0 -> {1, 2} -> 3, plus implied chord (0, 3); missing (1, 2)
    diamond = [(0, 1), (0, 2), (1, 3), (2, 3), (0, 3)]
    red_diamond = transitive_reduction(4, diamond)
    assert sorted(red_diamond) == list(_transitive_reduction(4, set(diamond)))
    assert is_topological_order_unique(4, diamond) is False

    for perm in itertools.permutations(range(4)):
        order = list(perm)
        assert (
            check_twin_step_order(4, order, red_diamond).passed
            == check_twin_step_order(4, order, diamond).passed
        )
    valid_diamond_orders = [
        list(perm)
        for perm in itertools.permutations(range(4))
        if check_twin_step_order(4, list(perm), diamond).passed
    ]
    assert valid_diamond_orders == [[0, 1, 2, 3], [0, 2, 1, 3]]

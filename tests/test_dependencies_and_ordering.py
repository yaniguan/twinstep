"""Tests for dependency rules, violation detection, and ordering reconstruction."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

from twinstep.baseline import predict_error, predict_order
from twinstep.dependencies import (
    canonical_well,
    check_constraints,
    find_dependencies,
    find_order_violations,
    find_uninitialized_well_uses,
)
from twinstep.injectors import make_ordering_record
from twinstep.schema import (
    IncubateStep,
    MixStep,
    Protocol,
    ReadStep,
    TransferStep,
    WashStep,
    load_protocol,
)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def test_clean_examples_have_no_order_violations() -> None:
    for path in sorted(EXAMPLES_DIR.glob("*.json")):
        proto = load_protocol(path)
        assert find_order_violations(proto.steps) == []
        assert find_order_violations(proto.steps, include_uninitialized=True) == []


def test_independent_buffer_transfers_are_unconstrained() -> None:
    proto = load_protocol(EXAMPLES_DIR / "dilution_series.json")
    deps = set(find_dependencies(proto.steps))
    # Steps 0..3 dispense buffer into A1..A4 independently; no edges among them.
    for i in range(4):
        for j in range(i + 1, 4):
            assert (i, j) not in deps
    # Serial chain A1 -> A2 -> A3 -> A4 -> read must be constrained.
    assert (5, 6) in deps  # mix A1 -> transfer A1->A2
    assert (6, 7) in deps  # transfer A1->A2 -> mix A2
    assert (11, 12) in deps  # mix A4 -> read plate1


def test_detects_read_before_incubate_and_mix_before_transfer() -> None:
    steps = [
        TransferStep(source="reservoir:buffer", dest="plate1:A1", volume=100, unit="uL"),
        ReadStep(plate="plate1", mode="absorbance", wavelength=450),
        IncubateStep(plate="plate1", temperature=37, time=20, time_unit="min"),
        MixStep(well="plate1:A2", volume=50, unit="uL", cycles=3),
        TransferStep(source="reservoir:dye", dest="plate1:A2", volume=50, unit="uL"),
    ]
    violations = find_order_violations(steps)
    reasons = " | ".join(v[2] for v in violations)
    assert "reads plate1 before" in reasons
    assert "mixes plate1:A2 before" in reasons


def _swapped(steps: list[Any], i: int, j: int) -> list[Any]:
    out = list(steps)
    out[i], out[j] = out[j], out[i]
    return out


def test_read_before_mix_is_flagged_across_an_intervening_wash() -> None:
    # wash_and_read with mix 5 and read 7 swapped: ... mix B1, read, wash, mix B2.
    proto = load_protocol(EXAMPLES_DIR / "wash_and_read.json")
    steps = _swapped(list(proto.steps), 5, 7)
    violations = find_order_violations(steps)
    assert (5, 7, "step 5 reads plate1 before step 7 mixes it") in violations
    pred = predict_error(steps)
    assert pred["label"] == "swap_order"
    assert pred["step_index"] == 5


def test_mix_after_plate_wash_without_read_is_flagged() -> None:
    # wash_and_read with mix 4 and wash 6 swapped: ... wash, mix B2, mix B1, read.
    proto = load_protocol(EXAMPLES_DIR / "wash_and_read.json")
    steps = _swapped(list(proto.steps), 4, 6)
    violations = find_order_violations(steps)
    assert (4, 5, "step 4 washes plate1 before step 5 mixes it") in violations
    pred = predict_error(steps)
    assert pred["label"] == "swap_order"
    assert pred["step_index"] == 4


def test_transfer_after_plate_wash_without_read_is_flagged() -> None:
    steps = [
        TransferStep(source="reservoir:buffer", dest="plate1:A1", volume=100, unit="uL"),
        WashStep(plate="plate1", volume=300, unit="uL", cycles=3),
        TransferStep(source="reservoir:dye", dest="plate1:A1", volume=50, unit="uL"),
        ReadStep(plate="plate1", mode="absorbance", wavelength=450),
    ]
    violations = find_order_violations(steps)
    assert (1, 2, "step 1 washes plate1 before step 2 transfers on it") in violations


def test_read_wash_transfer_new_cycle_is_not_flagged() -> None:
    steps = [
        TransferStep(source="reservoir:buffer", dest="plate1:A1", volume=100, unit="uL"),
        MixStep(well="plate1:A1", volume=50, unit="uL", cycles=3),
        ReadStep(plate="plate1", mode="absorbance", wavelength=450),
        WashStep(plate="plate1", volume=300, unit="uL", cycles=3),
        TransferStep(source="reservoir:dye", dest="plate1:A1", volume=50, unit="uL"),
        MixStep(well="plate1:A1", volume=50, unit="uL", cycles=3),
        ReadStep(plate="plate1", mode="absorbance", wavelength=450),
    ]
    assert find_order_violations(steps) == []


def test_ordering_baseline_satisfies_constraints_across_shuffles() -> None:
    rng = random.Random(123)
    for path in sorted(EXAMPLES_DIR.glob("*.json")):
        proto = load_protocol(path)
        for idx in range(5):
            rec = make_ordering_record(proto, rng, record_id=f"{proto.protocol_id}:{idx}")
            pred = predict_order(rec.shuffled_steps)
            constraints = [(a, b) for a, b in rec.constraints]
            assert check_constraints(pred, constraints)


# Issue #3: Incubate swapped ahead of remaining mix/transfer steps even when prior transfers exist
def test_incubate_before_remaining_mix_flagged_despite_prior_transfers() -> None:
    proto = load_protocol(EXAMPLES_DIR / "mix_incubate_read.json")
    steps = _swapped(list(proto.steps), 4, 6)  # swap mix C1 (4) and incubate (6)
    violations = find_order_violations(steps)
    assert any("step 4 incubates before liquid handling step 5" in v[2] for v in violations)
    pred = predict_error(steps)
    assert pred["label"] == "swap_order"
    assert pred["step_index"] == 4


# Issue #4: Write-after-read (WAR) anti-dependencies (aliquot before quench / well wash)
def test_war_dependencies_constrain_aliquot_before_quench_or_wash() -> None:
    quench_steps = [
        TransferStep(source="reservoir:reaction", dest="plate1:A1", volume=100, unit="uL"),
        TransferStep(source="plate1:A1", dest="plate1:B1", volume=50, unit="uL"),
        TransferStep(source="reservoir:quench", dest="plate1:A1", volume=50, unit="uL"),
    ]
    deps = set(find_dependencies(quench_steps))
    assert (0, 1) in deps
    assert (1, 2) in deps

    wash_steps = [
        TransferStep(source="reservoir:reaction", dest="plate1:A1", volume=100, unit="uL"),
        TransferStep(source="plate1:A1", dest="plate1:B1", volume=50, unit="uL"),
        WashStep(well="plate1:A1", volume=300, cycles=2, unit="uL"),
    ]
    assert (1, 2) in set(find_dependencies(wash_steps))


# Issue #5: Canonical well normalization ('A1', 'a1', 'plate1:a1', ' plate1:A1 ')
def test_canonical_well_matching_across_bare_and_case_variants() -> None:
    assert canonical_well("a1") == "plate1:A1"
    assert canonical_well(" A1 ") == "plate1:A1"
    assert canonical_well("plate1:a1 ") == "plate1:A1"

    steps = [
        TransferStep(source="reservoir:buffer", dest=" A1 ", volume=100, unit="uL"),
        MixStep(well="plate1:a1", volume=50, cycles=3, unit="uL"),
        TransferStep(source="plate1:A1", dest="B1", volume=50, unit="uL"),
        ReadStep(plate=" plate1 ", mode="absorbance", wavelength=450),
    ]
    deps = set(find_dependencies(steps))
    assert (0, 1) in deps
    assert (1, 2) in deps
    assert (2, 3) in deps
    assert predict_error(steps)["label"] == "clean"


# Issue #8: Structural check for never-filled wells without breaking swap_order vs missing_step
def test_uninitialized_wells_structural_mode_and_baseline_disambiguation() -> None:
    steps = [
        MixStep(well="plate1:A1", volume=50, cycles=3, unit="uL"),
        ReadStep(plate="plate1", mode="absorbance", wavelength=450),
    ]
    assert find_order_violations(steps) == []
    assert len(find_order_violations(steps, include_uninitialized=True)) == 1
    assert len(find_uninitialized_well_uses(steps)) == 1
    assert predict_error(steps)["label"] == "missing_step"


# Issue #11: Multi-cycle read/wash/transfer ordering and reciprocal well transfers
def test_multi_cycle_and_reciprocal_transfers_in_predict_order() -> None:
    two_cycle = Protocol(
        protocol_id="two_cycle_kinetic",
        steps=(
            TransferStep(source="reservoir:buffer", dest="plate1:A1", volume=150, unit="uL"),
            MixStep(well="plate1:A1", volume=80, cycles=3, unit="uL"),
            ReadStep(plate="plate1", mode="absorbance", wavelength=450),
            WashStep(plate="plate1", volume=300, cycles=3, unit="uL"),
            TransferStep(source="plate1:A1", dest="plate1:A2", volume=50, unit="uL"),
            MixStep(well="plate1:A2", volume=50, cycles=3, unit="uL"),
            ReadStep(plate="plate1", mode="absorbance", wavelength=450),
        ),
    )
    rng = random.Random(99)
    for idx in range(5):
        rec = make_ordering_record(two_cycle, rng, record_id=f"tc:{idx}")
        pred_order = predict_order(rec.shuffled_steps)
        constraints = [(a, b) for a, b in rec.constraints]
        assert check_constraints(pred_order, constraints)
        orig_dicts = two_cycle.to_dicts()
        reordered = [
            two_cycle.steps[orig_dicts.index(rec.shuffled_steps[k])]
            for k in pred_order
        ]
        assert find_order_violations(reordered) == []

    reciprocal = Protocol(
        protocol_id="reciprocal_flow",
        steps=(
            TransferStep(source="reservoir:buffer", dest="plate1:A1", volume=150, unit="uL"),
            TransferStep(source="plate1:A1", dest="plate1:A2", volume=100, unit="uL"),
            TransferStep(source="plate1:A2", dest="plate1:A1", volume=50, unit="uL"),
            ReadStep(plate="plate1", mode="absorbance", wavelength=450),
        ),
    )
    rec_r = make_ordering_record(reciprocal, rng, record_id="recip:0")
    pred_r = predict_order(rec_r.shuffled_steps)
    assert check_constraints(pred_r, [(a, b) for a, b in rec_r.constraints])

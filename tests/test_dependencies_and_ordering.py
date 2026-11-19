"""Tests for dependency rules, violation detection, and ordering reconstruction."""

from __future__ import annotations

from pathlib import Path
import random

from twinstep.baseline import predict_error, predict_order
from twinstep.dependencies import check_constraints, find_dependencies, find_order_violations
from twinstep.injectors import make_ordering_record
from twinstep.schema import IncubateStep, MixStep, ReadStep, TransferStep, WashStep, load_protocol

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def test_clean_examples_have_no_order_violations() -> None:
    for path in sorted(EXAMPLES_DIR.glob("*.json")):
        proto = load_protocol(path)
        assert find_order_violations(proto.steps) == []


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


def _swapped(steps: list, i: int, j: int) -> list:
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

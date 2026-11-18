"""Tests for error injectors, determinism, and ordering dataset generation."""

from __future__ import annotations

from pathlib import Path
import random
import pytest

from twinstep.dependencies import check_constraints
from twinstep.injectors import (
    ERROR_TYPES,
    generate_dataset,
    inject_error,
    make_clean_record,
    make_ordering_record,
)
from twinstep.schema import load_protocol

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _full_protocol():
    return load_protocol(EXAMPLES_DIR / "mix_incubate_read.json")


@pytest.mark.parametrize("error_type", ERROR_TYPES)
def test_each_injector_changes_exactly_what_it_says(error_type: str) -> None:
    proto = _full_protocol()
    rng = random.Random(42)
    twin = inject_error(proto, error_type, rng)

    correct = twin.correct_steps
    corrupted = twin.corrupted_steps

    assert twin.label == error_type
    assert twin.description
    assert twin.step_index is not None
    assert corrupted != correct

    if error_type == "swap_order":
        assert len(corrupted) == len(correct)
        i, j = twin.original_value["indices"]
        assert i == twin.step_index
        assert i != j
        assert corrupted[i] == correct[j]
        assert corrupted[j] == correct[i]
        for idx in range(len(correct)):
            if idx not in (i, j):
                assert corrupted[idx] == correct[idx]

    elif error_type == "wrong_volume":
        assert len(corrupted) == len(correct)
        idx = twin.step_index
        for pos in range(len(correct)):
            if pos != idx:
                assert corrupted[pos] == correct[pos]
        orig_step = dict(correct[idx])
        mut_step = dict(corrupted[idx])
        assert mut_step["volume"] != orig_step["volume"]
        assert mut_step["volume"] in (
            round(orig_step["volume"] * 10, 6),
            round(orig_step["volume"] * 0.1, 6),
        )
        orig_step.pop("volume")
        mut_step.pop("volume")
        assert orig_step == mut_step

    elif error_type == "wrong_unit":
        assert len(corrupted) == len(correct)
        idx = twin.step_index
        for pos in range(len(correct)):
            if pos != idx:
                assert corrupted[pos] == correct[pos]
        orig_step = dict(correct[idx])
        mut_step = dict(corrupted[idx])
        assert {orig_step["unit"], mut_step["unit"]} == {"uL", "mL"}
        orig_step.pop("unit")
        mut_step.pop("unit")
        assert orig_step == mut_step

    elif error_type == "missing_step":
        idx = twin.step_index
        assert len(corrupted) == len(correct) - 1
        assert twin.new_value is None
        assert twin.original_value == correct[idx]
        assert correct[:idx] + correct[idx + 1 :] == corrupted

    elif error_type == "wrong_temperature":
        assert len(corrupted) == len(correct)
        idx = twin.step_index
        for pos in range(len(correct)):
            if pos != idx:
                assert corrupted[pos] == correct[pos]
        orig_step = dict(correct[idx])
        mut_step = dict(corrupted[idx])
        assert orig_step["type"] == "incubate"
        assert mut_step["temperature"] != orig_step["temperature"]
        orig_step.pop("temperature")
        mut_step.pop("temperature")
        assert orig_step == mut_step

    elif error_type == "wrong_time":
        assert len(corrupted) == len(correct)
        idx = twin.step_index
        for pos in range(len(correct)):
            if pos != idx:
                assert corrupted[pos] == correct[pos]
        orig_step = dict(correct[idx])
        mut_step = dict(corrupted[idx])
        assert orig_step["type"] == "incubate"
        assert (mut_step["time"], mut_step["time_unit"]) != (
            orig_step["time"],
            orig_step["time_unit"],
        )
        for key in ("time", "time_unit"):
            orig_step.pop(key)
            mut_step.pop(key)
        assert orig_step == mut_step


def test_clean_negative_and_ordering_records() -> None:
    proto = load_protocol(EXAMPLES_DIR / "dilution_series.json")
    clean = make_clean_record(proto)
    assert clean.label == "clean"
    assert clean.step_index is None
    assert clean.corrupted_steps == clean.correct_steps

    order_rec = make_ordering_record(proto, random.Random(7))
    assert order_rec.shuffled_steps != order_rec.correct_steps
    reconstructed = [order_rec.shuffled_steps[i] for i in order_rec.gold_order]
    assert reconstructed == order_rec.correct_steps
    constraints = [(a, b) for a, b in order_rec.constraints]
    assert check_constraints(order_rec.gold_order, constraints)


def test_generate_dataset_is_deterministic_with_seed() -> None:
    protocols = [load_protocol(p) for p in sorted(EXAMPLES_DIR.glob("*.json"))]
    run_a = generate_dataset(protocols, n_twins=5, seed=0)
    run_b = generate_dataset(protocols, n_twins=5, seed=0)
    run_c = generate_dataset(protocols, n_twins=5, seed=99)

    assert run_a == run_b
    assert run_a != run_c
    # 3 protocols * (5 twins + 1 clean + 5 ordering) = 33
    assert len(run_a) == 33

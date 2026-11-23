"""Tests for rule-based baseline checker and evaluation metrics."""

from __future__ import annotations

from pathlib import Path

from twinstep.baseline import predict_error, predict_record
from twinstep.evaluate import _match_predictions, evaluate_records, format_evaluation_report
from twinstep.injectors import generate_dataset
from twinstep.schema import load_protocol

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def test_baseline_zero_false_positives_on_clean_examples() -> None:
    for path in sorted(EXAMPLES_DIR.glob("*.json")):
        proto = load_protocol(path)
        pred = predict_error(proto.steps)
        assert pred["label"] == "clean"
        assert pred["step_index"] is None


def test_baseline_catches_clear_errors() -> None:
    # wrong_unit (100 mL transfer)
    pred_unit = predict_error(
        [
            {"type": "transfer", "source": "reservoir:buffer", "dest": "plate1:A1", "volume": 100, "unit": "mL"},
            {"type": "read", "plate": "plate1", "mode": "absorbance", "wavelength": 450},
        ]
    )
    assert pred_unit["label"] == "wrong_unit"
    assert pred_unit["step_index"] == 0

    # wrong_volume (1000 uL well transfer)
    pred_vol = predict_error(
        [
            {"type": "transfer", "source": "reservoir:buffer", "dest": "plate1:A1", "volume": 1000, "unit": "uL"},
            {"type": "read", "plate": "plate1", "mode": "absorbance", "wavelength": 450},
        ]
    )
    assert pred_vol["label"] == "wrong_volume"
    assert pred_vol["step_index"] == 0

    # wrong_temperature (120 C)
    pred_temp = predict_error(
        [
            {"type": "transfer", "source": "reservoir:buffer", "dest": "plate1:A1", "volume": 100, "unit": "uL"},
            {"type": "incubate", "plate": "plate1", "temperature": 120, "time": 30, "time_unit": "min"},
            {"type": "read", "plate": "plate1", "mode": "absorbance", "wavelength": 450},
        ]
    )
    assert pred_temp["label"] == "wrong_temperature"
    assert pred_temp["step_index"] == 1

    # wrong_time (300 min)
    pred_time = predict_error(
        [
            {"type": "transfer", "source": "reservoir:buffer", "dest": "plate1:A1", "volume": 100, "unit": "uL"},
            {"type": "incubate", "plate": "plate1", "temperature": 37, "time": 300, "time_unit": "min"},
            {"type": "read", "plate": "plate1", "mode": "absorbance", "wavelength": 450},
        ]
    )
    assert pred_time["label"] == "wrong_time"
    assert pred_time["step_index"] == 1


def _ed(rec_id: str, label: str, step_index: int | None) -> dict:
    return {"id": rec_id, "task": "error_detection", "label": label, "step_index": step_index}


def _ord(rec_id: str, gold_order: list[int], constraints: list[list[int]]) -> dict:
    return {"id": rec_id, "task": "ordering", "gold_order": gold_order, "constraints": constraints}


def test_evaluate_records_matches_hand_computed_metrics() -> None:
    gold = [
        _ed("a", "wrong_volume", 2),
        _ed("b", "swap_order", 1),
        _ed("c", "clean", None),
        _ed("d", "wrong_unit", 0),
        _ed("e", "wrong_temperature", 4),
        # shuffled_steps[gold_order[k]] is correct step k, so gold_order is the execution order.
        _ord("o1", [2, 0, 1], [[2, 0], [0, 1]]),
        _ord("o2", [1, 0, 2], [[1, 0], [0, 2]]),
        _ord("o3", [0, 1, 2], [[0, 1], [1, 2]]),
        _ord("o4", [1, 0], [[1, 0]]),
        _ord("o5", [2, 0, 1], [[2, 0], [0, 1]]),
    ]
    preds = [
        {"id": "a", "label": "wrong_volume", "step_index": 3},  # label right, index wrong
        {"id": "b", "label": "wrong_unit", "step_index": 1},  # index right, label wrong -> 0
        {"id": "c", "label": "clean", "step_index": None},
        # "d" has no prediction.
        {"id": "e", "label": "wrong_temperature", "step_index": 4},  # both right
        {"id": "o1", "pred_order": [2, 0, 1]},  # exact
        {"id": "o2", "pred_order": [0, 1, 2]},  # breaks (1, 0), keeps (0, 2)
        {"id": "o3", "pred_order": [0, 0, 1]},  # duplicate index -> invalid
        {"id": "o4", "pred_order": [1, 0, 2]},  # wrong length -> invalid
        {"id": "o5", "pred_order": [1, 2, 0]},  # inverse of gold_order: keeps (2, 0), breaks (0, 1)
    ]
    # Ids take priority over position, so order of predictions does not matter.
    metrics = evaluate_records(gold, list(reversed(preds)))

    ed = metrics["error_detection"]
    assert (ed["total"], ed["correct"]) == (5, 3)
    assert ed["accuracy"] == 3 / 5
    # Gold step_index present for a, b, d, e; only e has both label and index right.
    assert (ed["step_index_total"], ed["step_index_correct"]) == (4, 1)
    assert ed["step_index_accuracy"] == 1 / 4
    per_type = {k: (v["correct"], v["total"]) for k, v in ed["per_error_type"].items()}
    assert per_type == {
        "clean": (1, 1),
        "swap_order": (0, 1),
        "wrong_temperature": (1, 1),
        "wrong_unit": (0, 1),
        "wrong_volume": (1, 1),
    }
    assert ed["per_error_type"]["swap_order"]["accuracy"] == 0.0

    ordering = metrics["ordering"]
    assert ordering["total"] == 5
    assert ordering["constraints_satisfied"] == 1  # o1 only
    assert ordering["exact_match"] == 1  # o1 only; o5 (inverse permutation) is not exact
    # o1 2/2, o2 1/2, o3 0/2 (invalid), o4 0/1 (invalid), o5 1/2.
    assert (ordering["pairwise_satisfied"], ordering["pairwise_total"]) == (4, 9)
    assert ordering["constraint_accuracy"] == 1 / 5
    assert ordering["exact_accuracy"] == 1 / 5
    assert ordering["pairwise_accuracy"] == 4 / 9


def test_match_predictions_positional_fallback() -> None:
    gold = [_ed("x", "wrong_unit", 0), _ed("y", "clean", None), _ed("z", "missing_step", 1)]
    preds = [
        {"label": "wrong_unit", "step_index": 0},  # no id: matched to gold[0] by position
        {"id": "other", "label": "clean", "step_index": None},  # unknown id: not used for y
        {"id": "z", "label": "missing_step", "step_index": 1},
    ]
    pairs = _match_predictions(gold, preds)
    assert pairs[0] == (gold[0], preds[0])
    assert pairs[1] == (gold[1], None)
    assert pairs[2] == (gold[2], preds[2])

    ed = evaluate_records(gold, preds)["error_detection"]
    assert (ed["correct"], ed["total"]) == (2, 3)
    assert (ed["step_index_correct"], ed["step_index_total"]) == (2, 2)


def test_evaluate_records_reports_per_error_type_and_ordering() -> None:
    protocols = [load_protocol(p) for p in sorted(EXAMPLES_DIR.glob("*.json"))]
    records = generate_dataset(protocols, n_twins=5, seed=0)
    preds = [predict_record(rec) for rec in records]
    metrics = evaluate_records(records, preds)

    ed = metrics["error_detection"]
    assert ed["total"] == 18
    assert 0.0 <= ed["accuracy"] <= 1.0
    assert "clean" in ed["per_error_type"]
    assert ed["per_error_type"]["clean"]["accuracy"] == 1.0

    ordering = metrics["ordering"]
    assert ordering["total"] == 15
    assert ordering["constraint_accuracy"] == 1.0

    report = format_evaluation_report(metrics)
    assert "Error detection:" in report
    assert "Ordering:" in report

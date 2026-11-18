"""Tests for rule-based baseline checker and evaluation metrics."""

from __future__ import annotations

from pathlib import Path

from twinstep.baseline import predict_error, predict_record
from twinstep.evaluate import evaluate_records, format_evaluation_report
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

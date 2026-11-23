"""Comprehensive regression tests covering twinstep audit issues #3 through #26."""

from __future__ import annotations

from pathlib import Path
import random

import pytest

from twinstep import (
    evaluate_records,
    from_labfill_dict,
    from_stepbot_dict,
    inject_k_errors,
    load_protocol,
    predict_record,
    to_labfill_dict,
    to_stepbot_dict,
)
from twinstep.baseline import predict_error
from twinstep.cli import main

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"


# ---------------------------------------------------------------------------
# Issue #9: Unmixed multi-reagent wells detected before incubate / wash / read
# ---------------------------------------------------------------------------


def test_issue_9_unmixed_multi_reagent_wells_flagged_before_barriers() -> None:
    mir = load_protocol(EXAMPLES_DIR / "mix_incubate_read.json").to_dicts()
    # Drop step 4 (mix plate1:C1) -> C1 has buffer + dye unmixed before incubate at index 5
    mir_no_mix = mir[:4] + mir[5:]
    pred_mir = predict_error(mir_no_mix)
    assert pred_mir["label"] == "missing_step"
    assert pred_mir["step_index"] == 5
    assert "plate1:C1" in pred_mir["reason"]

    war = load_protocol(EXAMPLES_DIR / "wash_and_read.json").to_dicts()
    # Drop step 5 (mix plate1:B2) -> B2 has buffer + dye unmixed before wash at index 5
    war_no_mix = war[:5] + war[6:]
    pred_war = predict_error(war_no_mix)
    assert pred_war["label"] == "missing_step"
    assert pred_war["step_index"] == 5
    assert "plate1:B2" in pred_war["reason"]

    # Also test ReadStep directly on unmixed multi-fill well
    direct_read = [
        {
            "type": "transfer",
            "source": "reservoir:buf",
            "dest": "plate1:A1",
            "volume": 100,
            "unit": "uL",
        },
        {
            "type": "transfer",
            "source": "reservoir:dye",
            "dest": "plate1:A1",
            "volume": 50,
            "unit": "uL",
        },
        {"type": "read", "plate": "plate1", "mode": "absorbance", "wavelength": 450},
    ]
    pred_read = predict_error(direct_read)
    assert pred_read["label"] == "missing_step"
    assert pred_read["step_index"] == 2


# ---------------------------------------------------------------------------
# Issue #10: Small mL (< 5 mL) & sub-uL unit flips classified as wrong_unit
# ---------------------------------------------------------------------------


def test_issue_10_small_ml_and_sub_ul_unit_flips_classified_as_wrong_unit() -> None:
    # 2 uL enzyme addition flipped to 2 mL
    pred_2ml = predict_error(
        [
            {
                "type": "transfer",
                "source": "stock:enzyme",
                "dest": "plate1:A1",
                "volume": 2,
                "unit": "mL",
            },
            {"type": "read", "plate": "plate1", "mode": "absorbance", "wavelength": 450},
        ]
    )
    assert pred_2ml["label"] == "wrong_unit"
    assert pred_2ml["step_index"] == 0

    # 0.5 mL wash flipped to 0.5 uL
    pred_wash_ul = predict_error(
        [
            {
                "type": "transfer",
                "source": "reservoir:buf",
                "dest": "plate1:A1",
                "volume": 100,
                "unit": "uL",
            },
            {"type": "wash", "plate": "plate1", "volume": 0.5, "unit": "uL", "cycles": 3},
            {"type": "read", "plate": "plate1", "mode": "absorbance", "wavelength": 450},
        ]
    )
    assert pred_wash_ul["label"] == "wrong_unit"
    assert pred_wash_ul["step_index"] == 1

    # 0.5 mL bulk stock->trough transfer flipped to 0.5 uL
    pred_bulk_ul = predict_error(
        [
            {
                "type": "transfer",
                "source": "stock:buf",
                "dest": "trough:buf",
                "volume": 0.5,
                "unit": "uL",
            },
            {
                "type": "transfer",
                "source": "trough:buf",
                "dest": "plate1:A1",
                "volume": 100,
                "unit": "uL",
            },
            {"type": "read", "plate": "plate1", "mode": "absorbance", "wavelength": 450},
        ]
    )
    assert pred_bulk_ul["label"] == "wrong_unit"
    assert pred_bulk_ul["step_index"] == 0


# ---------------------------------------------------------------------------
# Issue #14: Relaxed swap-pair localization alongside strict step_index
# ---------------------------------------------------------------------------


def test_issue_14_relaxed_swap_pair_localization_in_evaluate_records() -> None:
    gold = [
        {
            "id": "sw1",
            "task": "error_detection",
            "label": "swap_order",
            "step_index": 1,
            "original_value": {"indices": [1, 4]},
        }
    ]
    preds = [{"id": "sw1", "label": "swap_order", "step_index": 4}]
    ed = evaluate_records(gold, preds)["error_detection"]
    assert ed["step_index_correct"] == 0
    assert ed["step_index_relaxed_correct"] == 1
    assert ed["step_index_relaxed_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Issue #15: Unknown tasks, duplicate IDs, and missing-gold-ID fallback
# ---------------------------------------------------------------------------


def test_issue_15_evaluate_records_guards_and_fallback() -> None:
    with pytest.raises(ValueError, match="unknown task"):
        evaluate_records([{"id": "1", "task": "bogus_task"}], [{"id": "1"}])

    with pytest.raises(ValueError, match="duplicate prediction id"):
        evaluate_records(
            [{"id": "1", "task": "error_detection", "label": "clean", "step_index": None}],
            [
                {"id": "1", "label": "clean", "step_index": None},
                {"id": "1", "label": "wrong_unit", "step_index": 0},
            ],
        )

    with pytest.raises(ValueError, match="duplicate gold record id"):
        evaluate_records(
            [
                {"id": "dup", "task": "error_detection", "label": "clean", "step_index": None},
                {"id": "dup", "task": "error_detection", "label": "clean", "step_index": None},
            ],
            [],
        )

    # Gold record without 'id' matches prediction with 'id' by position when ID is unclaimed
    metrics = evaluate_records(
        [{"task": "error_detection", "label": "wrong_unit", "step_index": 0}],
        [{"id": "pred_0", "label": "wrong_unit", "step_index": 0}],
    )
    assert metrics["error_detection"]["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Issue #17: Malformed candidate steps handled without aborting batch runs
# ---------------------------------------------------------------------------


def test_issue_17_malformed_candidate_steps_mapped_to_error_labels() -> None:
    cases = [
        (
            {"type": "transfer", "source": "reservoir:a", "dest": "plate1:A1", "volume": 0},
            "wrong_volume",
        ),
        (
            {
                "type": "transfer",
                "source": "reservoir:a",
                "dest": "plate1:A1",
                "volume": 50,
                "unit": "L",
            },
            "wrong_unit",
        ),
        (
            {"type": "incubate", "plate": "plate1", "temperature": -300, "time": 10},
            "wrong_temperature",
        ),
        (
            {"type": "incubate", "plate": "plate1", "temperature": 37, "time": -5},
            "wrong_time",
        ),
        (
            {
                "type": "incubate",
                "plate": "plate1",
                "temperature": 37,
                "time": 10,
                "time_unit": "hours",
            },
            "wrong_time",
        ),
    ]
    base_transfer = {
        "type": "transfer",
        "source": "reservoir:buf",
        "dest": "plate1:B1",
        "volume": 100,
        "unit": "uL",
    }
    for idx_case, (bad_step, expected_label) in enumerate(cases):
        rec = {
            "id": f"mal:{idx_case}",
            "task": "error_detection",
            "protocol_id": "noisy",
            "corrupted_steps": [
                base_transfer,
                bad_step,
                {"type": "read", "plate": "plate1", "mode": "absorbance", "wavelength": 450},
            ],
        }
        pred = predict_record(rec)
        assert pred["label"] == expected_label
        assert pred["step_index"] == 1


# ---------------------------------------------------------------------------
# Issue #18: Optical wavelength bounds in ReadStep baseline check
# ---------------------------------------------------------------------------


def test_issue_18_optical_wavelength_bounds_in_baseline() -> None:
    fill = {
        "type": "transfer",
        "source": "reservoir:buf",
        "dest": "plate1:A1",
        "volume": 100,
        "unit": "uL",
    }
    pred_bad_wl = predict_error(
        [
            fill,
            {"type": "read", "plate": "plate1", "mode": "absorbance", "wavelength": 50000},
        ]
    )
    assert pred_bad_wl["label"] == "wrong_wavelength"
    assert pred_bad_wl["step_index"] == 1

    pred_lumi = predict_error(
        [
            fill,
            {"type": "read", "plate": "plate1", "mode": "luminescence", "wavelength": None},
        ]
    )
    assert pred_lumi["label"] == "clean"


# ---------------------------------------------------------------------------
# Issue #21: CI configuration (Python 3.13, Windows, stdlib-only, ruff/mypy)
# ---------------------------------------------------------------------------


def test_issue_21_ci_workflow_covers_windows_py313_stdlib_and_linters() -> None:
    ci_text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "windows-latest" in ci_text
    assert '"3.13"' in ci_text
    assert "test-stdlib-only" in ci_text
    assert "find_spec('yaml') is None" in ci_text
    assert "ruff check" in ci_text
    assert "mypy src" in ci_text


# ---------------------------------------------------------------------------
# Issue #23: stepbot & labfill protocol converters and round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "example_name",
    ["dilution_series.json", "mix_incubate_read.json", "wash_and_read.json"],
)
def test_issue_23_stepbot_and_labfill_roundtrip(example_name: str) -> None:
    proto = load_protocol(EXAMPLES_DIR / example_name)

    sb_dict = to_stepbot_dict(proto)
    assert "deck" in sb_dict and "pipettes" in sb_dict and "steps" in sb_dict
    assert from_stepbot_dict(sb_dict) == proto

    lf_dict = to_labfill_dict(proto)
    assert lf_dict["unit_scale"] == "0.1_uL"
    ops = lf_dict["protocol_batch"]["ops"]
    assert any(op["op"] == "pick_up_tip" for op in ops)
    assert any(op["op"] == "drop_tip" for op in ops)
    assert from_labfill_dict(lf_dict) == proto


# ---------------------------------------------------------------------------
# Issue #25: Error-repair evaluation (exact match & topological equivalence)
# ---------------------------------------------------------------------------


def test_issue_25_error_repair_exact_equivalent_and_invalid_scoring() -> None:
    proto = load_protocol(EXAMPLES_DIR / "dilution_series.json")
    gold_steps = proto.to_dicts()
    # Swap independent buffer dispenses 0 and 1 -> distinct list, valid topological equivalence!
    commuted_steps = [gold_steps[1], gold_steps[0], *gold_steps[2:]]
    # Swap dependent mix A1 (5) and transfer A1->A2 (6) -> invalid dependency violation!
    broken_steps = list(gold_steps)
    broken_steps[5], broken_steps[6] = broken_steps[6], broken_steps[5]

    gold = [
        {"id": "r_exact", "task": "error_repair", "correct_steps": gold_steps},
        {"id": "r_equiv", "task": "error_repair", "correct_steps": gold_steps},
        {"id": "r_broken", "task": "error_repair", "correct_steps": gold_steps},
    ]
    preds = [
        {"id": "r_exact", "repaired_steps": gold_steps},
        {"id": "r_equiv", "repaired_steps": commuted_steps},
        {"id": "r_broken", "repaired_steps": broken_steps},
    ]
    rep = evaluate_records(gold, preds)["error_repair"]
    assert rep["total"] == 3
    assert rep["exact_match"] == 1
    assert rep["dependency_equivalent"] == 2
    assert rep["exact_accuracy"] == pytest.approx(1 / 3)
    assert rep["equivalent_accuracy"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Issue #26: Compound multi-fault twins (--k-errors) & F1 evaluation
# ---------------------------------------------------------------------------


def test_issue_26_inject_k_errors_and_cli_multi_fault_eval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    proto = load_protocol(EXAMPLES_DIR / "mix_incubate_read.json")
    mtwin = inject_k_errors(proto, k=2, rng=random.Random(12))
    assert len(mtwin.labels) == 2
    assert len(mtwin.step_indices) == 2
    assert mtwin.corrupted_steps != mtwin.correct_steps

    # Exact oracle F1 = 1.0, partial 1-of-2 recall = 0.5 / precision = 1.0 -> F1 = 2/3
    d = mtwin.to_dict()
    oracle_pred = {
        "id": d["id"],
        "labels": d["labels"],
        "step_indices": d["step_indices"],
    }
    partial_pred = {
        "id": d["id"],
        "labels": [d["labels"][0]],
        "step_indices": [d["step_indices"][0]],
    }
    m_oracle = evaluate_records([d], [oracle_pred])["multi_error_detection"]
    assert m_oracle["exact_accuracy"] == 1.0
    assert m_oracle["label_f1"] == 1.0
    assert m_oracle["pair_f1"] == 1.0

    m_partial = evaluate_records([d], [partial_pred])["multi_error_detection"]
    assert m_partial["exact_accuracy"] == 0.0
    assert m_partial["label_f1"] == pytest.approx(2 / 3)
    assert m_partial["pair_f1"] == pytest.approx(2 / 3)

    # End-to-end CLI `--k-errors 2` + `eval --baseline`
    out_k2 = tmp_path / "twins_k2.jsonl"
    assert (
        main(
            [
                "generate",
                str(EXAMPLES_DIR / "mix_incubate_read.json"),
                "--n",
                "3",
                "--k-errors",
                "2",
                "--ordering",
                "0",
                "-o",
                str(out_k2),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["eval", str(out_k2), "--baseline"]) == 0
    report = capsys.readouterr().out
    assert "Multi-error detection:" in report

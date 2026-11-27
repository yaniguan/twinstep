"""Regression tests for Issue #35 (protocol splits & blind-eval) and Issue #37 (prompts & score)."""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import pytest

from twinstep.cli import main
from twinstep.evaluate import evaluate_records, format_evaluation_report
from twinstep.formats import format_record
from twinstep.injectors import ERROR_TYPES, generate_dataset
from twinstep.schema import load_protocol
from twinstep.splits import BUNDLED_PROTOCOL_SPLITS, assign_protocol_split

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def test_issue_35_whole_protocol_splits_disjoint_and_complete() -> None:
    files = sorted(EXAMPLES_DIR.glob("*.json"))
    assert len(files) == 24
    protocols = [load_protocol(p) for p in files]

    train_recs = generate_dataset(protocols, n_twins=4, seed=0, split="train")
    val_recs = generate_dataset(protocols, n_twins=4, seed=0, split="val")
    test_recs = generate_dataset(protocols, n_twins=4, seed=0, split="test")

    train_pids = {r["protocol_id"] for r in train_recs}
    val_pids = {r["protocol_id"] for r in val_recs}
    test_pids = {r["protocol_id"] for r in test_recs}

    assert len(train_pids) == 8
    assert len(val_pids) == 4
    assert len(test_pids) == 12
    assert train_pids.isdisjoint(val_pids)
    assert train_pids.isdisjoint(test_pids)
    assert val_pids.isdisjoint(test_pids)
    assert (train_pids | val_pids | test_pids) == set(BUNDLED_PROTOCOL_SPLITS)

    # Custom protocol IDs assign deterministically as whole protocols
    for pid in ("custom_assay_alpha", "custom_assay_beta", "custom_assay_gamma"):
        assert assign_protocol_split(pid) == assign_protocol_split(f"{pid}:twin:0")


def test_issue_35_blind_eval_strips_reference_and_overlap_warns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pattern = str(EXAMPLES_DIR / "*.json")
    gold_test = tmp_path / "test_gold.jsonl"
    blind_test = tmp_path / "test_blind.jsonl"
    preds_test = tmp_path / "test_preds.jsonl"
    train_file = tmp_path / "train_gold.jsonl"

    assert (
        main(["generate", pattern, "--split", "test", "--n", "3", "-o", str(gold_test)]) == 0
    )
    assert (
        main(
            [
                "generate",
                pattern,
                "--split",
                "test",
                "--n",
                "3",
                "--blind-eval",
                "-o",
                str(blind_test),
            ]
        )
        == 0
    )
    assert (
        main(["generate", pattern, "--split", "train", "--n", "3", "-o", str(train_file)]) == 0
    )

    blind_rows = [json.loads(line) for line in blind_test.read_text(encoding="utf-8").splitlines()]
    assert len(blind_rows) > 0
    for row in blind_rows:
        assert "correct_steps" not in row
        assert "gold_order" not in row
        assert "label" not in row
        assert "step_index" not in row
        assert row["split"] == "test"

    assert main(["baseline", str(blind_test), "-o", str(preds_test)]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "eval",
                str(gold_test),
                str(preds_test),
                "--train-twins",
                str(train_file),
            ]
        )
        == 0
    )
    io_clean = capsys.readouterr()
    assert "WARNING" not in io_clean.err
    assert "WARNING" not in io_clean.out

    # Deliberately contaminate train_records with one test record -> must warn!
    gold_rows = [json.loads(line) for line in gold_test.read_text(encoding="utf-8").splitlines()]
    train_rows = [json.loads(line) for line in train_file.read_text(encoding="utf-8").splitlines()]
    leaked_train = [*train_rows, {**gold_rows[0], "split": "train"}]
    with pytest.warns(UserWarning, match="Protocol leakage detected"):
        leaked_metrics = evaluate_records(
            gold_rows,
            [json.loads(line) for line in preds_test.read_text(encoding="utf-8").splitlines()],
            train_records=leaked_train,
        )
    assert gold_rows[0]["protocol_id"] in leaked_metrics["split_overlap_protocols"]
    assert "WARNING:" in format_evaluation_report(leaked_metrics)


def test_issue_37_prompts_labels_normalization_and_balanced_score() -> None:
    three_classic = [
        load_protocol(EXAMPLES_DIR / name)
        for name in ("dilution_series.json", "mix_incubate_read.json", "wash_and_read.json")
    ]
    records = generate_dataset(three_classic, n_twins=5, seed=0, n_ordering=0)
    assert len(records) == 18

    # 1. Every error-detection prompt lists all 6 error labels, 'clean', and 'step_index is 0-based'
    for rec in records:
        sft_rec = format_record(rec, fmt="sft")
        prompt = sft_rec["prompt"]
        assert "step_index is 0-based" in prompt
        assert "'clean'" in prompt
        for lbl in ERROR_TYPES:
            assert f"'{lbl}'" in prompt

    multi_recs = generate_dataset(three_classic, n_twins=2, seed=0, n_ordering=0, k_errors=2)
    for mrec in multi_recs:
        mprompt = format_record(mrec, fmt="sft")["prompt"]
        assert "step_index is 0-based" in mprompt
        for lbl in ERROR_TYPES:
            assert f"'{lbl}'" in mprompt

    # 2. Three oracle answer spellings from Issue #37 evidence table
    exact_preds = [
        {"id": r["id"], "label": r["label"], "step_index": r["step_index"]} for r in records
    ]
    spaced_preds = [
        {
            "id": r["id"],
            "label": str(r["label"]).replace("_", " ").upper()
            if idx % 2 == 0
            else str(r["label"]).replace("_", "-"),
            "step_index": r["step_index"],
        }
        for idx, r in enumerate(records)
    ]
    one_based_preds = [
        {
            "id": r["id"],
            "label": r["label"],
            "step_index": (r["step_index"] + 1) if r["step_index"] is not None else None,
        }
        for r in records
    ]

    m_exact = evaluate_records(records, exact_preds)["error_detection"]
    assert (m_exact["correct"], m_exact["total"]) == (18, 18)
    assert (m_exact["step_index_correct"], m_exact["step_index_total"]) == (15, 15)
    assert m_exact["normalized_predictions"] == 0
    assert m_exact["balanced_accuracy"] == pytest.approx(1.0)

    m_spaced = evaluate_records(records, spaced_preds)["error_detection"]
    assert (m_spaced["correct"], m_spaced["total"]) == (18, 18)
    assert (m_spaced["step_index_correct"], m_spaced["step_index_total"]) == (15, 15)
    assert m_spaced["normalized_predictions"] > 0
    assert m_spaced["balanced_accuracy"] == pytest.approx(1.0)

    m_one_based = evaluate_records(records, one_based_preds)["error_detection"]
    assert (m_one_based["correct"], m_one_based["total"]) == (18, 18)
    assert (m_one_based["step_index_correct"], m_one_based["step_index_total"]) == (0, 15)

    # 3. Always-flag-error baseline gets 0.0 balanced_accuracy despite nonzero plain accuracy
    always_error_preds = [
        {
            "id": r["id"],
            "label": r["label"] if r["label"] != "clean" else "swap_order",
            "step_index": 0,
        }
        for r in records
    ]
    m_hack = evaluate_records(records, always_error_preds)["error_detection"]
    assert m_hack["accuracy"] == pytest.approx(15 / 18)
    assert m_hack["clean_accuracy"] == 0.0
    assert m_hack["balanced_accuracy"] == 0.0

    # 4. Multi-fault-only dataset prints 'no single-fault records' instead of 0/0 (0.0%)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_only_metrics = evaluate_records(multi_recs, [])
    m_report = format_evaluation_report(m_only_metrics)
    assert "no single-fault records" in m_report
    assert "0/0 (0.0%)" not in m_report

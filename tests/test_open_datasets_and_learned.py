"""Tests for open wet-lab dataset adapters, SPDX guards, and zero-leakage learned verifier."""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import pytest

from twinstep.baseline import predict_record
from twinstep.evaluate import evaluate_records
from twinstep.injectors import generate_dataset
from twinstep.learned_verifier import train_learned_verifier
from twinstep.open_datasets import (
    OPEN_PROTOCOL_CATALOG,
    DatasetLicenseError,
    download_url_cached,
    parse_bioprobench_err_records,
    parse_bioprobench_ord_records,
    parse_bioprot_json,
    parse_opentrons_json_protocol,
    parse_wlp_brat_protocol,
    sha256_bytes,
    verify_dataset_governance,
)
from twinstep.schema import load_protocol

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


def test_open_catalog_spdx_and_eval_only_guards(tmp_path: Path) -> None:
    assert set(OPEN_PROTOCOL_CATALOG) == {"wlp", "bioprobench", "bioplanner", "opentrons"}

    # Eval-only / CC-BY-NC-4.0 sources must block training use
    with pytest.raises(DatasetLicenseError):
        verify_dataset_governance("bioprobench", for_training=True)
    with pytest.raises(DatasetLicenseError):
        verify_dataset_governance("bioplanner", for_training=True)

    # Permissive sources allow training
    assert verify_dataset_governance("wlp", for_training=True).spdx_license == "MIT"
    assert verify_dataset_governance("opentrons", for_training=True).spdx_license == "Apache-2.0"

    # Cached SHA-256 verification catches corruption
    local_file = tmp_path / "cached.json"
    payload = b'{"ok": true}\n'
    local_file.write_bytes(payload)
    digest = sha256_bytes(payload)
    meta = download_url_cached(
        "https://example.org/cached.json", local_file, expected_sha256=digest
    )
    assert meta["sha256"] == digest

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download_url_cached(
            "https://example.org/cached.json", local_file, expected_sha256="0" * 64
        )


def test_parsers_for_wlp_bioprobench_bioplanner_and_opentrons() -> None:
    # 1. WLP BRAT (.txt + .ann)
    wlp_txt = (
        "Add 50 uL lysis buffer into well.\n"
        "Add 50 uL DNA sample.\n"
        "Mix 50 uL gently.\n"
        "Incubate at 37 C for 20 min.\n"
        "Wash plate with 300 uL PBS.\n"
        "Read absorbance at 450 nm.\n"
    )
    wlp_ann = (
        "T1\tAction 0 3\tAdd\n"
        "T2\tAction 35 38\tAdd\n"
        "T3\tAction 58 61\tMix\n"
        "T4\tAction 76 84\tIncubate\n"
        "T5\tAction 106 110\tWash\n"
        "T6\tAction 134 138\tRead\n"
    )
    wlp_proto = parse_wlp_brat_protocol("103", wlp_txt, wlp_ann, split="test")
    assert wlp_proto.split == "test"
    assert len(wlp_proto.steps) >= 5

    # 2. BioProBench ERR & ORD (with CC-BY-NC-4.0 eval_only guard)
    err_raw = [
        {
            "id": "TEST-ERR-1",
            "corrupted_text": "Dispense 100 mL ELISA substrate into well A1.",
            "corrected_text": "Dispense 100 uL ELISA substrate into well A1.",
            "is_correct": False,
            "type": "parameter",
            "error_description": "Milliliters instead of microliters.",
        },
        {
            "id": "TEST-ERR-2",
            "corrupted_text": "Incubate plate at 120 C for 15 min.",
            "corrected_text": "Incubate plate at 37 C for 15 min.",
            "is_correct": False,
            "type": "parameter",
            "error_description": "Excessive temperature.",
        },
        {
            "id": "TEST-ERR-3",
            "corrupted_text": "Add 50 uL sample and incubate at 37 C for 30 min.",
            "corrected_text": "Add 50 uL sample and incubate at 37 C for 30 min.",
            "is_correct": True,
            "type": "correct",
        },
    ]
    with pytest.raises(DatasetLicenseError):
        parse_bioprobench_err_records(err_raw, for_training=True)
    bpb_err = parse_bioprobench_err_records(err_raw, for_training=False)
    assert [r["label"] for r in bpb_err] == ["wrong_unit", "wrong_temperature", "clean"]

    ord_raw = [
        {
            "id": "TEST-ORD-1",
            "wrong_steps": ["Read absorbance", "Add capture reagent", "Wash unbound antibody"],
            "correct_steps": ["Add capture reagent", "Wash unbound antibody", "Read absorbance"],
            "type": "top",
        }
    ]
    bpb_ord = parse_bioprobench_ord_records(ord_raw, for_training=False)
    assert len(bpb_ord) == 1
    assert bpb_ord[0]["task"] == "ordering"

    # 3. BioPlanner / BioProt
    with pytest.raises(DatasetLicenseError):
        parse_bioprot_json({"id": "10051", "steps": ["Add 20 uL mastermix"]}, split="train")
    bp_proto = parse_bioprot_json(
        {
            "id": "10051",
            "steps": [
                "Add 20 uL PCR mastermix.",
                "Incubate at 95 C for 3 min.",
                "Read fluorescence at 520 nm.",
            ],
        },
        split="test",
    )
    assert bp_proto.protocol_id == "bioprot_10051"
    assert len(bp_proto.steps) >= 3

    # 4. Opentrons JSON v8 commands & Protocol Library protoBuild
    ot_v8 = {
        "schemaVersion": 8,
        "commands": [
            {
                "commandType": "loadLabware",
                "params": {"labwareId": "srcRes", "loadName": "nest_12_reservoir_15ml"},
            },
            {
                "commandType": "loadLabware",
                "params": {
                    "labwareId": "assayPlate",
                    "loadName": "corning_96_wellplate_360ul_flat",
                },
            },
            {
                "commandType": "loadLiquid",
                "params": {"labwareId": "srcRes", "liquidId": "buf"},
            },
            {
                "commandType": "aspirate",
                "params": {"labwareId": "srcRes", "wellName": "A1", "volume": 50},
            },
            {
                "commandType": "dispense",
                "params": {"labwareId": "assayPlate", "wellName": "B1", "volume": 50},
            },
            {"commandType": "waitForDuration", "params": {"seconds": 30}},
            {"commandType": "absorbanceReader/read", "params": {"moduleId": "reader1"}},
        ],
    }
    ot_proto = parse_opentrons_json_protocol(ot_v8, protocol_id="v8_demo", split="test")
    assert len(ot_proto.steps) == 3
    assert ot_proto.steps[0].type == "transfer"
    assert ot_proto.steps[1].type == "incubate"
    assert ot_proto.steps[2].type == "read"


def test_learned_verifier_zero_leakage_and_outperforms_rule_baseline() -> None:
    all_protocols = [load_protocol(p) for p in sorted(EXAMPLES_DIR.glob("*.json"))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_pool: list[dict[str, object]] = []
        for s in range(6):
            train_pool.extend(
                generate_dataset(
                    all_protocols, n_twins=12, seed=s, n_clean=1, n_ordering=0, split="train"
                )
            )
    test_recs = generate_dataset(all_protocols, n_twins=5, seed=0, split="test")

    # Refuses to train on test-split records
    with pytest.raises(ValueError, match="Refusing to train on non-train split"):
        train_learned_verifier(test_recs, backend="stdlib")

    verifier = train_learned_verifier(train_pool, backend="stdlib", seed=0)  # type: ignore[arg-type]

    # Refuses to evaluate on a train-split protocol_id
    with pytest.raises(ValueError, match="Data leakage blocked"):
        verifier.predict_record(train_pool[0])  # type: ignore[arg-type]

    rule_metrics = evaluate_records(test_recs, [predict_record(r) for r in test_recs])
    learned_metrics = evaluate_records(
        test_recs, [verifier.predict_record(r) for r in test_recs]
    )

    r_ed = rule_metrics["error_detection"]
    l_ed = learned_metrics["error_detection"]
    assert l_ed["accuracy"] > r_ed["accuracy"]
    assert l_ed["balanced_accuracy"] > r_ed["balanced_accuracy"]
    assert l_ed["step_index_accuracy"] > r_ed["step_index_accuracy"]


def test_committed_frontier_benchmark_results_match_heldout_test() -> None:
    results_path = REPO_ROOT / "benchmarks" / "results" / "frontier_eval_2026.json"
    if not results_path.is_file():
        pytest.skip("frontier_eval_2026.json not generated yet")
    data = json.loads(results_path.read_text(encoding="utf-8"))
    heldout = data["heldout_test_split"]
    assert heldout["split"] == "test"
    assert heldout["test_protocol_count"] == 12
    assert heldout["train_test_overlap_count"] == 0

    all_protocols = [load_protocol(p) for p in sorted(EXAMPLES_DIR.glob("*.json"))]
    test_recs = generate_dataset(all_protocols, n_twins=5, seed=0, split="test")
    rm = evaluate_records(test_recs, [predict_record(r) for r in test_recs])
    expected_ed = round(float(rm["error_detection"]["accuracy"]), 4)
    assert heldout["seed_0"]["rule_baseline"]["ed_accuracy"] == expected_ed
    assert set(data["public_corpora"]) == {"wlp", "bioprobench", "bioplanner", "opentrons"}

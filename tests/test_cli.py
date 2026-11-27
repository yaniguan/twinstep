"""End-to-end tests for the twinstep CLI."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from twinstep.cli import main

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def test_cli_generate_baseline_eval_flow(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    twins_path = tmp_path / "twins.jsonl"
    preds_path = tmp_path / "preds.jsonl"
    inputs = [str(p) for p in sorted(EXAMPLES_DIR.glob("*.json"))]

    assert main(["generate", *inputs, "--n", "5", "--seed", "0", "-o", str(twins_path)]) == 0
    assert twins_path.exists()
    twin_lines = [
        json.loads(line)
        for line in twins_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(twin_lines) == 33

    assert main(["baseline", str(twins_path), "-o", str(preds_path)]) == 0
    assert preds_path.exists()
    pred_lines = [
        json.loads(line)
        for line in preds_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(pred_lines) == 33

    capsys.readouterr()
    assert main(["eval", str(twins_path), str(preds_path), "--json"]) == 0
    out_json = json.loads(capsys.readouterr().out)
    assert out_json["error_detection"]["total"] == 18
    assert out_json["ordering"]["total"] == 15

    # Direct --baseline evaluation AND omitted preds without --baseline (Issue #20).
    assert main(["eval", str(twins_path), "--baseline"]) == 0
    out_text = capsys.readouterr().out
    assert "Error detection:" in out_text
    assert "Ordering:" in out_text

    assert main(["eval", str(twins_path)]) == 0
    out_text_no_flag = capsys.readouterr().out
    assert out_text_no_flag == out_text


# Issue #16: Literal wildcard glob expansion on Windows shells
def test_cli_expands_literal_glob_pattern(tmp_path: Path) -> None:
    out_path = tmp_path / "glob_twins.jsonl"
    pattern = str(EXAMPLES_DIR / "*.json")
    assert main(["generate", pattern, "--n", "2", "--seed", "0", "-o", str(out_path)]) == 0
    lines = [line for line in out_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 3 * (2 + 1 + 2)

    with pytest.raises(FileNotFoundError, match="no protocol files matched pattern"):
        main(["generate", str(tmp_path / "nonexistent_*.json"), "-o", str(out_path)])


# Issue #20: CLI stdout ('-o -' / omitted '-o'), stdin ('-'), and malformed JSONL errors
def test_cli_stdout_stdin_and_invalid_jsonl_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    single_proto = str(EXAMPLES_DIR / "mix_incubate_read.json")

    # Default stdout (output omitted) and explicit '-o -'
    assert main(["generate", single_proto, "--n", "1", "--clean", "0", "--ordering", "0"]) == 0
    stdout_default = capsys.readouterr().out
    rec = json.loads(stdout_default.strip())
    assert rec["protocol_id"] == "mix_incubate_read"

    assert (
        main(["generate", single_proto, "--n", "1", "--clean", "0", "--ordering", "0", "-o", "-"])
        == 0
    )
    assert capsys.readouterr().out == stdout_default

    # Pipe twins via stdin ('-') into `twinstep baseline - -o -`
    monkeypatch.setattr("sys.stdin", io.StringIO(stdout_default))
    assert main(["baseline", "-", "-o", "-"]) == 0
    pred_out = json.loads(capsys.readouterr().out.strip())
    assert pred_out["task"] == "error_detection"

    # Malformed JSONL reports path and 1-based line number
    bad_jsonl = tmp_path / "broken.jsonl"
    bad_jsonl.write_text('{"id": "ok", "label": "clean"}\n{not valid json}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"broken\.jsonl:2: invalid JSON"):
        main(["baseline", str(bad_jsonl)])


# Issue #24: DPO, SFT, and RLVR export formats and completion parsing in eval
def test_cli_export_formats_dpo_sft_rlvr_and_eval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    proto_arg = str(EXAMPLES_DIR / "mix_incubate_read.json")

    dpo_path = tmp_path / "dpo.jsonl"
    assert main(["generate", proto_arg, "--n", "2", "--format", "dpo", "-o", str(dpo_path)]) == 0
    dpo_recs = [json.loads(line) for line in dpo_path.read_text(encoding="utf-8").splitlines()]
    assert all({"prompt", "chosen", "rejected"} <= set(r) for r in dpo_recs)
    assert all(r["chosen"] != r["rejected"] for r in dpo_recs)

    sft_path = tmp_path / "sft.jsonl"
    assert main(["generate", proto_arg, "--n", "2", "--format", "sft", "-o", str(sft_path)]) == 0
    sft_recs = [json.loads(line) for line in sft_path.read_text(encoding="utf-8").splitlines()]
    assert all({"prompt", "completion"} <= set(r) for r in sft_recs)

    rlvr_path = tmp_path / "rlvr.jsonl"
    assert main(["generate", proto_arg, "--n", "2", "--format", "rlvr", "-o", str(rlvr_path)]) == 0
    rlvr_recs = [json.loads(line) for line in rlvr_path.read_text(encoding="utf-8").splitlines()]
    assert all({"prompt", "ground_truth"} <= set(r) for r in rlvr_recs)

    # Evaluate SFT model completions (wrapped in markdown json fences) against RLVR dataset!
    fenced_preds_path = tmp_path / "model_completions.jsonl"
    fenced_lines = [
        json.dumps({"id": r["id"], "completion": f"```json\n{r['completion']}\n```"})
        for r in sft_recs
    ]
    fenced_preds_path.write_text("\n".join(fenced_lines) + "\n", encoding="utf-8")

    capsys.readouterr()
    assert main(["eval", str(rlvr_path), str(fenced_preds_path), "--json"]) == 0
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["error_detection"]["accuracy"] == 1.0
    assert metrics["ordering"]["constraint_accuracy"] == 1.0


# Issue #33: Stratified difficulty filtering, saboteur export, and MPO / GDPO CLI schemas
def test_issue_33_cli_difficulty_saboteurs_and_mpo_gdpo_formats(tmp_path: Path) -> None:
    proto_arg = str(EXAMPLES_DIR / "mix_incubate_read.json")

    mpo_path = tmp_path / "mpo.jsonl"
    assert (
        main(
            [
                "generate",
                proto_arg,
                "--n",
                "4",
                "--include-saboteurs",
                "--format",
                "mpo",
                "-o",
                str(mpo_path),
            ]
        )
        == 0
    )
    mpo_recs = [json.loads(line) for line in mpo_path.read_text(encoding="utf-8").splitlines()]
    assert any(r["difficulty"] == "hard_saboteur" for r in mpo_recs)
    assert all("rejected_candidates" in r and len(r["rejected_candidates"]) >= 1 for r in mpo_recs)

    gdpo_path = tmp_path / "gdpo_sab.jsonl"
    assert (
        main(
            [
                "generate",
                proto_arg,
                "--n",
                "3",
                "--include-saboteurs",
                "--difficulty",
                "hard_saboteur",
                "--format",
                "gdpo",
                "-o",
                str(gdpo_path),
            ]
        )
        == 0
    )
    gdpo_recs = [json.loads(line) for line in gdpo_path.read_text(encoding="utf-8").splitlines()]
    assert len(gdpo_recs) >= 2
    for rec in gdpo_recs:
        assert rec["difficulty"] == "hard_saboteur"
        weights = rec["reward_weights"]
        assert set(weights) == {"detection", "localization", "repair_dag", "lean_gate"}
        assert sum(weights.values()) == pytest.approx(1.0)

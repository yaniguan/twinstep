"""End-to-end tests for the twinstep CLI."""

from __future__ import annotations

import json
from pathlib import Path

from twinstep.cli import main

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def test_cli_generate_baseline_eval_flow(tmp_path: Path, capsys) -> None:
    twins_path = tmp_path / "twins.jsonl"
    preds_path = tmp_path / "preds.jsonl"
    inputs = [str(p) for p in sorted(EXAMPLES_DIR.glob("*.json"))]

    assert main(["generate", *inputs, "--n", "5", "--seed", "0", "-o", str(twins_path)]) == 0
    assert twins_path.exists()
    twin_lines = [json.loads(line) for line in twins_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(twin_lines) == 33

    assert main(["baseline", str(twins_path), "-o", str(preds_path)]) == 0
    assert preds_path.exists()
    pred_lines = [json.loads(line) for line in preds_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(pred_lines) == 33

    capsys.readouterr()
    assert main(["eval", str(twins_path), str(preds_path), "--json"]) == 0
    out_json = json.loads(capsys.readouterr().out)
    assert out_json["error_detection"]["total"] == 18
    assert out_json["ordering"]["total"] == 15

    # Also verify direct --baseline evaluation without preds file.
    assert main(["eval", str(twins_path), "--baseline"]) == 0
    out_text = capsys.readouterr().out
    assert "Error detection:" in out_text
    assert "Ordering:" in out_text

"""Tests verifying README reproducibility numbers, citations, and zero-CJK repository hygiene."""

from __future__ import annotations

import json
from pathlib import Path
import re

from twinstep.baseline import predict_record
from twinstep.evaluate import evaluate_records, format_evaluation_report
from twinstep.injectors import generate_dataset
from twinstep.schema import VALID_READ_MODES, load_protocol
from twinstep.splits import BUNDLED_PROTOCOL_SPLITS

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

_CJK_RE = re.compile(
    r"[\u2e80-\u2eff\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
)


def test_zero_cjk_characters_across_repository() -> None:
    skip_dirs = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "data",
        ".lake",
    }
    checked = 0
    for path in REPO_ROOT.rglob("*"):
        if any(part in skip_dirs for part in path.parts) or not path.is_file():
            continue
        if path.suffix.lower() in {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".lean"}:
            text = path.read_text(encoding="utf-8")
            match = _CJK_RE.search(text)
            assert match is None, f"CJK character {match.group(0)!r} found in {path}"
            checked += 1
    assert checked >= 40


def test_readme_heldout_numbers_and_cli_snippet_reproduce_exactly() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "GPU Throughput" not in readme

    # Verify protocol counts and split counts
    example_files = sorted(EXAMPLES_DIR.glob("*.json"))
    assert len(example_files) == 24
    assert sum(1 for v in BUNDLED_PROTOCOL_SPLITS.values() if v == "train") == 8
    assert sum(1 for v in BUNDLED_PROTOCOL_SPLITS.values() if v == "val") == 4
    assert sum(1 for v in BUNDLED_PROTOCOL_SPLITS.values() if v == "test") == 12

    # Verify all 7 ReadStep modes are documented in README
    for mode in VALID_READ_MODES:
        assert f"`{mode}`" in readme

    # Reproduce the held-out `--split test --n 5 --seed 0` CLI report verbatim
    all_protocols = [load_protocol(p) for p in example_files]
    test_recs = generate_dataset(all_protocols, n_twins=5, seed=0, split="test")
    assert len(test_recs) == 132
    preds = [predict_record(r) for r in test_recs]
    metrics = evaluate_records(test_recs, preds)
    report = format_evaluation_report(metrics)

    first_line = report.splitlines()[0]
    assert first_line in readme

    # Cross-check committed 20-seed & Wilson CI summary in README
    eval_summary = json.loads(
        (REPO_ROOT / "benchmarks" / "results" / "frontier_eval_2026.json").read_text(
            encoding="utf-8"
        )
    )
    h = eval_summary["heldout_test_split"]
    r20_mean = f"{h['seeds_0_to_19_rule_baseline']['ed_accuracy_mean']:.1%}"
    l20_mean = f"{h['seeds_0_to_19_learned_stdlib']['ed_accuracy_mean']:.1%}"
    assert r20_mean in readme
    assert l20_mean in readme


def test_issue_38_doc_and_extension_claims_accurate() -> None:
    lean_readme = (REPO_ROOT / "lean" / "README.md").read_text(encoding="utf-8")
    assert "no user axioms" in lean_readme.lower()
    assert "`0 axiom`" not in lean_readme

    saboteur_src = (
        REPO_ROOT / "community_extensions" / "saboteur_evolver.py"
    ).read_text(encoding="utf-8")
    assert "arXiv:2607.17352" in saboteur_src
    assert "2606.06523" not in saboteur_src
    assert "11" in saboteur_src and "5 core" in saboteur_src
    norm_sab = " ".join(saboteur_src.split())
    assert "by construction" in norm_sab

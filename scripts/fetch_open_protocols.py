"""Fetch real public wet-lab protocol datasets into gitignored ``data/`` and run evaluations.

Downloads and SHA-256-fingerprints:
1. ``wlp``: Wet Lab Protocols BRAT ``.txt`` + ``.ann`` files (MIT)
2. ``bioprobench``: ``ERR_test.json`` & ``ORD_test.json`` (``CC-BY-NC-4.0``, ``eval_only=True``)
3. ``bioplanner``: ``BioProt`` ``protocols.io`` JSON files (``MIT``, ``eval_only=True``)
4. ``opentrons``: ``shared-data`` JSON v6-v8 & ``Opentrons/Protocols`` builds (``Apache-2.0``)

Evaluates both the stateless rule baseline and the zero-leakage learned DAG fault localizer
on the 12-protocol held-out ``test`` split (across 20 seeds + 95% Wilson CI) and all 4 public
corpora, writing compact reproducible results to ``benchmarks/results/frontier_eval_2026.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any
import warnings

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from twinstep.baseline import predict_record  # noqa: E402
from twinstep.evaluate import evaluate_records  # noqa: E402
from twinstep.injectors import generate_dataset  # noqa: E402
from twinstep.learned_verifier import train_learned_verifier  # noqa: E402
from twinstep.open_datasets import (  # noqa: E402
    OPEN_PROTOCOL_CATALOG,
    download_url_cached,
    parse_bioprobench_err_records,
    parse_bioprobench_ord_records,
    parse_bioprot_json,
    parse_opentrons_json_protocol,
    parse_wlp_brat_protocol,
    verify_dataset_governance,
)
from twinstep.schema import Protocol, load_protocol  # noqa: E402


def wilson_95_ci(k: int, n: int) -> tuple[float, float]:
    """Compute the 95% Wilson score confidence interval for ``k`` successes out of ``n``."""
    if n <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


def fetch_all_sources(data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Download sample slices for all cataloged public protocol corpora."""
    manifest: dict[str, list[dict[str, Any]]] = {}
    for source_id, entry in OPEN_PROTOCOL_CATALOG.items():
        verify_dataset_governance(source_id, for_training=False)
        src_dir = data_dir / source_id
        records: list[dict[str, Any]] = []
        for url in entry.sample_urls:
            fname = url.rsplit("/", 1)[-1]
            # Disambiguate same-named files from different schema versions
            if source_id == "opentrons" and "/fixtures/" in url:
                ver_part = url.split("/fixtures/", 1)[1].split("/", 1)[0]
                fname = f"v{ver_part}_{fname}"
            meta = download_url_cached(url, src_dir / fname)
            records.append(meta)
        manifest[source_id] = records
    return manifest


def _compact_ed_ord(metrics: dict[str, Any]) -> dict[str, Any]:
    ed = metrics["error_detection"]
    ord_m = metrics["ordering"]
    ci_lo, ci_hi = wilson_95_ci(int(ed["correct"]), int(ed["total"]))
    return {
        "protocol_count": metrics.get("protocol_count", 0),
        "ed_correct": ed["correct"],
        "ed_total": ed["total"],
        "ed_accuracy": round(float(ed["accuracy"]), 4),
        "ed_wilson_95_ci": [ci_lo, ci_hi],
        "ed_balanced_accuracy": round(float(ed["balanced_accuracy"]), 4),
        "step_index_correct": ed["step_index_correct"],
        "step_index_total": ed["step_index_total"],
        "step_index_accuracy": round(float(ed["step_index_accuracy"]), 4),
        "ord_constraints_satisfied": ord_m["constraints_satisfied"],
        "ord_total": ord_m["total"],
        "ord_constraint_accuracy": round(float(ord_m["constraint_accuracy"]), 4),
        "ord_pairwise_accuracy": round(float(ord_m["pairwise_accuracy"]), 4),
    }


def run_full_evaluation(data_dir: Path, *, fetch: bool = True) -> dict[str, Any]:
    """Run held-out test split (20 seeds) and 4-corpus public benchmark evaluations."""
    examples_dir = REPO_ROOT / "examples"
    all_protocols = [load_protocol(p) for p in sorted(examples_dir.glob("*.json"))]

    # Train learned verifiers strictly on the 8 train-split protocols across multiple seeds
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_pool: list[dict[str, Any]] = []
        for t_seed in range(6):
            train_pool.extend(
                generate_dataset(
                    all_protocols, n_twins=12, seed=t_seed, n_clean=1, n_ordering=0, split="train"
                )
            )

    stdlib_verifier = train_learned_verifier(train_pool, backend="stdlib", seed=0)
    optional_verifiers: dict[str, Any] = {"learned_stdlib": stdlib_verifier}
    if importlib.util.find_spec("sklearn") is not None:
        optional_verifiers["learned_sklearn"] = train_learned_verifier(
            train_pool, backend="sklearn", seed=0
        )
    if importlib.util.find_spec("lightgbm") is not None:
        optional_verifiers["learned_lightgbm"] = train_learned_verifier(
            train_pool, backend="lightgbm", seed=0
        )

    # 1. Held-out 12-protocol test split: seed 0 + 20-seed distribution (seeds 0..19)
    seed_ed_accs: list[float] = []
    seed_bal_accs: list[float] = []
    seed_step_accs: list[float] = []
    seed_ord_accs: list[float] = []
    learned_seed_ed_accs: list[float] = []
    learned_seed_bal_accs: list[float] = []
    learned_seed_step_accs: list[float] = []

    seed0_rule_metrics: dict[str, Any] = {}
    seed0_model_metrics: dict[str, dict[str, Any]] = {}

    for seed in range(20):
        test_recs = generate_dataset(all_protocols, n_twins=5, seed=seed, split="test")
        rule_preds = [predict_record(r) for r in test_recs]
        rm = evaluate_records(test_recs, rule_preds, train_records=train_pool)
        assert rm["split_overlap_protocols"] == []

        seed_ed_accs.append(float(rm["error_detection"]["accuracy"]))
        seed_bal_accs.append(float(rm["error_detection"]["balanced_accuracy"]))
        seed_step_accs.append(float(rm["error_detection"]["step_index_accuracy"]))
        seed_ord_accs.append(float(rm["ordering"]["constraint_accuracy"]))

        l_preds = [stdlib_verifier.predict_record(r) for r in test_recs]
        lm = evaluate_records(test_recs, l_preds, train_records=train_pool)
        learned_seed_ed_accs.append(float(lm["error_detection"]["accuracy"]))
        learned_seed_bal_accs.append(float(lm["error_detection"]["balanced_accuracy"]))
        learned_seed_step_accs.append(float(lm["error_detection"]["step_index_accuracy"]))

        if seed == 0:
            seed0_rule_metrics = _compact_ed_ord(rm)
            for m_name, ver in optional_verifiers.items():
                mpreds = [ver.predict_record(r) for r in test_recs]
                mm = evaluate_records(test_recs, mpreds, train_records=train_pool)
                seed0_model_metrics[m_name] = _compact_ed_ord(mm)

    heldout_summary = {
        "split": "test",
        "test_protocol_count": 12,
        "train_protocol_count": 8,
        "val_protocol_count": 4,
        "train_test_overlap_count": 0,
        "seed_0": {
            "rule_baseline": seed0_rule_metrics,
            **seed0_model_metrics,
        },
        "seeds_0_to_19_rule_baseline": {
            "ed_accuracy_mean": round(statistics.mean(seed_ed_accs), 4),
            "ed_accuracy_std": round(statistics.stdev(seed_ed_accs), 4),
            "ed_accuracy_min": round(min(seed_ed_accs), 4),
            "ed_accuracy_max": round(max(seed_ed_accs), 4),
            "ed_balanced_mean": round(statistics.mean(seed_bal_accs), 4),
            "ed_balanced_std": round(statistics.stdev(seed_bal_accs), 4),
            "step_index_mean": round(statistics.mean(seed_step_accs), 4),
            "step_index_std": round(statistics.stdev(seed_step_accs), 4),
            "ord_constraint_mean": round(statistics.mean(seed_ord_accs), 4),
            "ord_constraint_std": round(statistics.stdev(seed_ord_accs), 4),
        },
        "seeds_0_to_19_learned_stdlib": {
            "ed_accuracy_mean": round(statistics.mean(learned_seed_ed_accs), 4),
            "ed_accuracy_std": round(statistics.stdev(learned_seed_ed_accs), 4),
            "ed_balanced_mean": round(statistics.mean(learned_seed_bal_accs), 4),
            "ed_balanced_std": round(statistics.stdev(learned_seed_bal_accs), 4),
            "step_index_mean": round(statistics.mean(learned_seed_step_accs), 4),
            "step_index_std": round(statistics.stdev(learned_seed_step_accs), 4),
        },
    }

    manifest = fetch_all_sources(data_dir) if fetch else {}

    # 2. Evaluate on the 4 real downloaded public datasets
    public_eval: dict[str, Any] = {}

    # 2a. WLP
    wlp_dir = data_dir / "wlp"
    if wlp_dir.is_dir():
        wlp_protos: list[Protocol] = []
        for txt_path in sorted(wlp_dir.glob("*.txt")):
            ann_path = txt_path.with_suffix(".ann")
            if ann_path.is_file():
                wlp_protos.append(
                    parse_wlp_brat_protocol(
                        txt_path.stem,
                        txt_path.read_text(encoding="utf-8"),
                        ann_path.read_text(encoding="utf-8"),
                        split="test",
                    )
                )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wlp_recs = generate_dataset(wlp_protos, n_twins=5, seed=0, split="test")
        public_eval["wlp"] = {
            "spdx_license": OPEN_PROTOCOL_CATALOG["wlp"].spdx_license,
            "eval_only": OPEN_PROTOCOL_CATALOG["wlp"].eval_only,
            "protocols_parsed": len(wlp_protos),
            "rule_baseline": _compact_ed_ord(
                evaluate_records(wlp_recs, [predict_record(r) for r in wlp_recs])
            ),
            "learned_stdlib": _compact_ed_ord(
                evaluate_records(wlp_recs, [stdlib_verifier.predict_record(r) for r in wlp_recs])
            ),
        }

    # 2b. BioProBench (ERR_test.json + ORD_test.json, CC-BY-NC-4.0, eval_only=True)
    bpb_dir = data_dir / "bioprobench"
    if (bpb_dir / "ERR_test.json").is_file() and (bpb_dir / "ORD_test.json").is_file():
        err_raw = json.loads((bpb_dir / "ERR_test.json").read_text(encoding="utf-8"))
        ord_raw = json.loads((bpb_dir / "ORD_test.json").read_text(encoding="utf-8"))
        bpb_err_recs = parse_bioprobench_err_records(err_raw, max_records=160, for_training=False)
        bpb_ord_recs = parse_bioprobench_ord_records(ord_raw, max_records=100, for_training=False)
        bpb_all = [*bpb_err_recs, *bpb_ord_recs]
        public_eval["bioprobench"] = {
            "spdx_license": OPEN_PROTOCOL_CATALOG["bioprobench"].spdx_license,
            "eval_only": OPEN_PROTOCOL_CATALOG["bioprobench"].eval_only,
            "raw_err_count": len(err_raw),
            "raw_ord_count": len(ord_raw),
            "lowered_err_records": len(bpb_err_recs),
            "lowered_ord_records": len(bpb_ord_recs),
            "rule_baseline": _compact_ed_ord(
                evaluate_records(bpb_all, [predict_record(r) for r in bpb_all])
            ),
            "learned_stdlib": _compact_ed_ord(
                evaluate_records(bpb_all, [stdlib_verifier.predict_record(r) for r in bpb_all])
            ),
        }

    # 2c. BioPlanner (BioProt)
    bp_dir = data_dir / "bioplanner"
    if bp_dir.is_dir():
        bp_protos = [
            parse_bioprot_json(json.loads(p.read_text(encoding="utf-8")), split="test")
            for p in sorted(bp_dir.glob("*.json"))
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bp_recs = generate_dataset(bp_protos, n_twins=5, seed=0, split="test")
        public_eval["bioplanner"] = {
            "spdx_license": OPEN_PROTOCOL_CATALOG["bioplanner"].spdx_license,
            "eval_only": OPEN_PROTOCOL_CATALOG["bioplanner"].eval_only,
            "protocols_parsed": len(bp_protos),
            "rule_baseline": _compact_ed_ord(
                evaluate_records(bp_recs, [predict_record(r) for r in bp_recs])
            ),
            "learned_stdlib": _compact_ed_ord(
                evaluate_records(bp_recs, [stdlib_verifier.predict_record(r) for r in bp_recs])
            ),
        }

    # 2d. Opentrons JSON v6-v8 + Opentrons/Protocols
    ot_dir = data_dir / "opentrons"
    if ot_dir.is_dir():
        ot_protos = [
            parse_opentrons_json_protocol(
                json.loads(p.read_text(encoding="utf-8")),
                protocol_id=p.stem,
                split="test",
            )
            for p in sorted(ot_dir.glob("*.json"))
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ot_recs = generate_dataset(ot_protos, n_twins=5, seed=0, split="test")
        public_eval["opentrons"] = {
            "spdx_license": OPEN_PROTOCOL_CATALOG["opentrons"].spdx_license,
            "eval_only": OPEN_PROTOCOL_CATALOG["opentrons"].eval_only,
            "protocols_parsed": len(ot_protos),
            "rule_baseline": _compact_ed_ord(
                evaluate_records(ot_recs, [predict_record(r) for r in ot_recs])
            ),
            "learned_stdlib": _compact_ed_ord(
                evaluate_records(ot_recs, [stdlib_verifier.predict_record(r) for r in ot_recs])
            ),
        }

    return {
        "schema_version": "twinstep-frontier-eval/v1",
        "heldout_test_split": heldout_summary,
        "public_corpora": public_eval,
        "downloaded_files": {
            src: [
                {"file": Path(m["path"]).name, "bytes": m["bytes"], "sha256": m["sha256"]}
                for m in items
            ]
            for src, items in manifest.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and benchmark open protocol datasets.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "open_protocols",
        help="Gitignored local download cache directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "frontier_eval_2026.json",
        help="Output path for reproducible JSON evaluation metrics.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip network fetch and evaluate already-cached files.",
    )
    args = parser.parse_args(argv)

    summary = run_full_evaluation(args.data_dir, fetch=not args.offline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote frontier benchmark summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

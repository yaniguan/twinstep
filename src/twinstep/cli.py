"""Command-line interface for twinstep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from twinstep.baseline import predict_record
from twinstep.evaluate import evaluate_records, format_evaluation_report
from twinstep.injectors import generate_dataset
from twinstep.schema import load_protocol


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
    return records


def _write_jsonl(records: Sequence[dict[str, Any]], output: Path | None) -> None:
    lines = [json.dumps(rec, sort_keys=False) for rec in records]
    payload = ("\n".join(lines) + "\n") if lines else ""
    if output is None:
        sys.stdout.write(payload)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")


def _cmd_generate(args: argparse.Namespace) -> int:
    protocols = [load_protocol(p) for p in args.protocols]
    records = generate_dataset(
        protocols,
        n_twins=args.n,
        seed=args.seed,
        n_clean=args.clean,
        n_ordering=args.ordering,
    )
    out_path = Path(args.output) if args.output and args.output != "-" else None
    _write_jsonl(records, out_path)
    if out_path is not None:
        print(f"Wrote {len(records)} records from {len(protocols)} protocols to {out_path}")
    return 0


def _cmd_baseline(args: argparse.Namespace) -> int:
    records = _read_jsonl(Path(args.twins))
    preds = [predict_record(rec) for rec in records]
    out_path = Path(args.output) if args.output and args.output != "-" else None
    _write_jsonl(preds, out_path)
    if out_path is not None:
        print(f"Wrote {len(preds)} baseline predictions to {out_path}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    gold = _read_jsonl(Path(args.twins))
    if args.baseline or args.preds is None:
        preds = [predict_record(rec) for rec in gold]
    else:
        preds = _read_jsonl(Path(args.preds))

    metrics = evaluate_records(gold, preds)
    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print(format_evaluation_report(metrics))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level twinstep argument parser."""
    parser = argparse.ArgumentParser(
        prog="twinstep",
        description="Seeded-error twin generator and evaluator for liquid-handling steps.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate error-detection and ordering JSONL twins.")
    gen.add_argument("protocols", nargs="+", help="Input protocol JSON or YAML files.")
    gen.add_argument("--n", type=int, default=5, help="Corrupted twins per protocol (default: 5).")
    gen.add_argument("--clean", type=int, default=1, help="Emit one clean negative per protocol when > 0 (default: 1); 0 disables.",
    )
    gen.add_argument(
        "--ordering",
        type=int,
        default=None,
        help="Ordering examples per protocol (default: matches --n).",
    )
    gen.add_argument("--seed", type=int, default=0, help="Deterministic RNG seed (default: 0).")
    gen.add_argument("-o", "--output", default=None, help="Output JSONL path (default: stdout).")
    gen.set_defaults(func=_cmd_generate)

    base = sub.add_parser("baseline", help="Run the rule-based checker and ordering baseline.")
    base.add_argument("twins", help="Input twins JSONL file.")
    base.add_argument("-o", "--output", default=None, help="Output predictions JSONL path (default: stdout).")
    base.set_defaults(func=_cmd_baseline)

    ev = sub.add_parser("eval", help="Score a predictions JSONL file (or the baseline) against twins.")
    ev.add_argument("twins", help="Ground-truth twins JSONL file.")
    ev.add_argument(
        "preds",
        nargs="?",
        default=None,
        help="Predictions JSONL file (omit or pass --baseline to score the baseline).",
    )
    ev.add_argument(
        "--baseline",
        action="store_true",
        help="Evaluate the built-in rule-based baseline directly.",
    )
    ev.add_argument("--json", action="store_true", help="Print metrics as JSON.")
    ev.set_defaults(func=_cmd_eval)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

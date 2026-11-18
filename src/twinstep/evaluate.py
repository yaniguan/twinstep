"""Evaluation metrics for error-detection and step-ordering predictions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from twinstep.dependencies import check_constraints


def _match_predictions(
    gold_records: Sequence[dict[str, Any]],
    pred_records: Sequence[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    by_id: dict[str, dict[str, Any]] = {}
    for pred in pred_records:
        rec_id = pred.get("id")
        if rec_id is not None:
            by_id[str(rec_id)] = pred

    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for idx, gold in enumerate(gold_records):
        rec_id = gold.get("id")
        if rec_id is not None and str(rec_id) in by_id:
            pairs.append((gold, by_id[str(rec_id)]))
        elif idx < len(pred_records) and pred_records[idx].get("id") is None:
            pairs.append((gold, pred_records[idx]))
        else:
            pairs.append((gold, None))
    return pairs


def evaluate_records(
    gold_records: Sequence[dict[str, Any]],
    pred_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compute overall and per-error-type accuracy plus ordering constraint accuracy."""
    matched = _match_predictions(gold_records, pred_records)

    ed_total = 0
    ed_correct = 0
    step_idx_total = 0
    step_idx_correct = 0
    per_type_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    ord_total = 0
    ord_constraints_ok = 0
    ord_exact_ok = 0
    pair_ok = 0
    pair_total = 0

    for gold, pred in matched:
        task = gold.get("task", "error_detection")
        if task == "error_detection":
            ed_total += 1
            gold_label = str(gold["label"])
            per_type_counts[gold_label][1] += 1
            pred_label = pred.get("label") if pred is not None else None
            if pred_label == gold_label:
                ed_correct += 1
                per_type_counts[gold_label][0] += 1

            gold_idx = gold.get("step_index")
            if gold_idx is not None:
                step_idx_total += 1
                if pred is not None and pred.get("step_index") == gold_idx and pred_label == gold_label:
                    step_idx_correct += 1

        elif task == "ordering":
            ord_total += 1
            gold_order = gold["gold_order"]
            constraints = [tuple(pair) for pair in gold.get("constraints", [])]
            pred_order = pred.get("pred_order") if pred is not None else None
            if (
                isinstance(pred_order, list)
                and len(pred_order) == len(gold_order)
                and set(pred_order) == set(range(len(gold_order)))
            ):
                if pred_order == gold_order:
                    ord_exact_ok += 1
                if check_constraints(pred_order, constraints):
                    ord_constraints_ok += 1
                pos = {step_idx: rank for rank, step_idx in enumerate(pred_order)}
                for a, b in constraints:
                    pair_total += 1
                    if pos[a] < pos[b]:
                        pair_ok += 1
            else:
                pair_total += len(constraints)

    per_error_type = {
        err_type: {
            "correct": counts[0],
            "total": counts[1],
            "accuracy": (counts[0] / counts[1]) if counts[1] else 0.0,
        }
        for err_type, counts in sorted(per_type_counts.items())
    }

    return {
        "error_detection": {
            "total": ed_total,
            "correct": ed_correct,
            "accuracy": (ed_correct / ed_total) if ed_total else 0.0,
            "step_index_correct": step_idx_correct,
            "step_index_total": step_idx_total,
            "step_index_accuracy": (step_idx_correct / step_idx_total) if step_idx_total else 0.0,
            "per_error_type": per_error_type,
        },
        "ordering": {
            "total": ord_total,
            "constraints_satisfied": ord_constraints_ok,
            "constraint_accuracy": (ord_constraints_ok / ord_total) if ord_total else 0.0,
            "exact_match": ord_exact_ok,
            "exact_accuracy": (ord_exact_ok / ord_total) if ord_total else 0.0,
            "pairwise_satisfied": pair_ok,
            "pairwise_total": pair_total,
            "pairwise_accuracy": (pair_ok / pair_total) if pair_total else 0.0,
        },
    }


def format_evaluation_report(metrics: dict[str, Any]) -> str:
    """Format evaluation metrics as a compact human-readable report."""
    lines: list[str] = []
    ed = metrics["error_detection"]
    lines.append(
        f"Error detection: {ed['correct']}/{ed['total']} ({ed['accuracy']:.1%}) "
        f"| step_index: {ed['step_index_correct']}/{ed['step_index_total']} "
        f"({ed['step_index_accuracy']:.1%})"
    )
    for err_type, stats in ed["per_error_type"].items():
        lines.append(
            f"  - {err_type:18s}: {stats['correct']:3d}/{stats['total']:<3d} ({stats['accuracy']:.1%})"
        )

    ordering = metrics["ordering"]
    if ordering["total"] > 0:
        lines.append(
            f"Ordering: constraints {ordering['constraints_satisfied']}/{ordering['total']} "
            f"({ordering['constraint_accuracy']:.1%}) | exact {ordering['exact_match']}/{ordering['total']} "
            f"({ordering['exact_accuracy']:.1%}) | pairwise {ordering['pairwise_satisfied']}/{ordering['pairwise_total']} "
            f"({ordering['pairwise_accuracy']:.1%})"
        )
    return "\n".join(lines)

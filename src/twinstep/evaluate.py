"""Evaluation metrics for error-detection, repair, multi-fault, and step-ordering tasks."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
import re
from typing import Any
import warnings

from twinstep.dependencies import check_constraints, find_dependencies
from twinstep.formats import parse_completion_payload
from twinstep.schema import Step, steps_from_list
from twinstep.splits import VALID_SPLITS, detect_protocol_split_overlaps

_SUPPORTED_TASKS = {
    "error_detection",
    "ordering",
    "error_repair",
    "multi_error_detection",
}


def normalize_error_label(raw_label: Any) -> tuple[str | None, bool]:
    """Canonicalize label casing, whitespace, and hyphens, returning ``(label, changed)``."""
    if raw_label is None or not isinstance(raw_label, str):
        return None, False
    cleaned = re.sub(r"[\s\-]+", "_", raw_label.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        return "", cleaned != raw_label
    return cleaned, cleaned != raw_label


def _normalize_gold(raw_gold: dict[str, Any]) -> dict[str, Any]:
    if "ground_truth" in raw_gold and isinstance(raw_gold["ground_truth"], dict):
        merged = dict(raw_gold["ground_truth"])
        for key in ("id", "protocol_id", "split", "task", "corrupted_steps", "shuffled_steps"):
            if key in raw_gold and raw_gold[key] is not None:
                merged.setdefault(key, raw_gold[key])
        return merged
    return raw_gold


def _normalize_pred(raw_pred: dict[str, Any] | None) -> dict[str, Any] | None:
    if raw_pred is None:
        return None
    for text_key in ("completion", "response", "output"):
        if isinstance(raw_pred.get(text_key), str) and not any(
            k in raw_pred for k in ("label", "pred_order", "repaired_steps", "labels")
        ):
            parsed = parse_completion_payload(raw_pred[text_key])
            if parsed is not None:
                merged = dict(raw_pred)
                merged.update(parsed)
                return merged
    return raw_pred


def _match_predictions(
    gold_records: Sequence[dict[str, Any]],
    pred_records: Sequence[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    seen_gold_ids: set[str] = set()
    for gold in gold_records:
        gid = gold.get("id")
        if gid is not None:
            gkey = str(gid)
            if gkey in seen_gold_ids:
                raise ValueError(f"duplicate gold record id: {gkey!r}")
            seen_gold_ids.add(gkey)

    by_id: dict[str, dict[str, Any]] = {}
    for pred in pred_records:
        rec_id = pred.get("id")
        if rec_id is not None:
            pkey = str(rec_id)
            if pkey in by_id:
                raise ValueError(f"duplicate prediction id: {pkey!r}")
            by_id[pkey] = pred

    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for idx, gold in enumerate(gold_records):
        rec_id = gold.get("id")
        if rec_id is not None:
            gkey = str(rec_id)
            if gkey in by_id:
                pairs.append((gold, by_id[gkey]))
            elif idx < len(pred_records) and pred_records[idx].get("id") is None:
                pairs.append((gold, pred_records[idx]))
            else:
                pairs.append((gold, None))
        else:
            if idx < len(pred_records):
                cand = pred_records[idx]
                cand_id = cand.get("id")
                if cand_id is None or str(cand_id) not in seen_gold_ids:
                    pairs.append((gold, cand))
                else:
                    pairs.append((gold, None))
            else:
                pairs.append((gold, None))
    return pairs


def _score_repair(
    correct_raw: Sequence[dict[str, Any]],
    repaired_raw: Any,
) -> tuple[bool, bool]:
    """Return (exact_match, dependency_equivalent) for candidate repaired_steps."""
    if not isinstance(repaired_raw, (list, tuple)):
        return False, False
    try:
        gold_steps = steps_from_list(correct_raw)
        cand_steps = steps_from_list(repaired_raw)
    except ValueError:
        return False, False

    if cand_steps == gold_steps:
        return True, True
    if len(cand_steps) != len(gold_steps) or Counter(cand_steps) != Counter(gold_steps):
        return False, False

    # Build order-preserving bijection for identical repeated steps: gold_steps[i] -> cand pos.
    buckets: dict[Step, list[int]] = defaultdict(list)
    for c_idx, c_step in enumerate(cand_steps):
        buckets[c_step].append(c_idx)
    read_ptr: dict[Step, int] = defaultdict(int)
    gold_to_cand_pos: list[int] = []
    for g_step in gold_steps:
        ptr = read_ptr[g_step]
        gold_to_cand_pos.append(buckets[g_step][ptr])
        read_ptr[g_step] = ptr + 1

    constraints = find_dependencies(gold_steps)
    equiv = all(gold_to_cand_pos[a] < gold_to_cand_pos[b] for a, b in constraints)
    return False, equiv


def _multiset_prf(gold_items: Sequence[Any], pred_items: Sequence[Any]) -> tuple[int, int, int]:
    """Return (tp, pred_count, gold_count) using multiset intersection."""
    gc = Counter(gold_items)
    pc = Counter(pred_items)
    tp = sum((gc & pc).values())
    return tp, sum(pc.values()), sum(gc.values())


def _safe_f1(tp: int, pred_n: int, gold_n: int) -> tuple[float, float, float]:
    if pred_n == 0 and gold_n == 0:
        return 1.0, 1.0, 1.0
    prec = (tp / pred_n) if pred_n > 0 else 0.0
    rec = (tp / gold_n) if gold_n > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _compute_core_metrics(
    normalized_pairs: Sequence[tuple[dict[str, Any], dict[str, Any] | None]],
) -> dict[str, Any]:
    ed_total = 0
    ed_correct = 0
    clean_total = 0
    clean_correct = 0
    err_total = 0
    err_correct = 0
    normalized_preds = 0
    step_idx_total = 0
    step_idx_correct = 0
    step_idx_relaxed_correct = 0
    per_type_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    ord_total = 0
    ord_constraints_ok = 0
    ord_exact_ok = 0
    pair_ok = 0
    pair_total = 0

    rep_total = 0
    rep_exact_ok = 0
    rep_equiv_ok = 0

    multi_total = 0
    multi_exact_ok = 0
    multi_lbl_tp = 0
    multi_lbl_pred = 0
    multi_lbl_gold = 0
    multi_pair_tp = 0
    multi_pair_pred = 0
    multi_pair_gold = 0

    protocol_ids: set[str] = set()

    for gold, pred in normalized_pairs:
        if gold.get("protocol_id") is not None:
            protocol_ids.add(str(gold["protocol_id"]))
        task = gold.get("task", "error_detection")
        if task not in _SUPPORTED_TASKS:
            raise ValueError(
                f"unknown task {task!r} in gold record {gold.get('id')!r}; "
                f"expected one of {sorted(_SUPPORTED_TASKS)}"
            )

        if task == "error_detection":
            ed_total += 1
            gold_label_norm, _ = normalize_error_label(str(gold["label"]))
            gold_label = gold_label_norm if gold_label_norm is not None else str(gold["label"])
            per_type_counts[gold_label][1] += 1
            if gold_label == "clean":
                clean_total += 1
            else:
                err_total += 1

            raw_pred_label = pred.get("label") if pred is not None else None
            pred_label, was_norm = normalize_error_label(raw_pred_label)
            if was_norm:
                normalized_preds += 1

            if pred_label == gold_label:
                ed_correct += 1
                per_type_counts[gold_label][0] += 1
                if gold_label == "clean":
                    clean_correct += 1
                else:
                    err_correct += 1

            gold_idx = gold.get("step_index")
            if gold_idx is not None:
                step_idx_total += 1
                if pred is not None and pred_label == gold_label:
                    pred_idx = pred.get("step_index")
                    if pred_idx == gold_idx:
                        step_idx_correct += 1
                        step_idx_relaxed_correct += 1
                    elif (
                        gold_label == "swap_order"
                        and isinstance(gold.get("original_value"), dict)
                        and isinstance(gold["original_value"].get("indices"), list)
                        and pred_idx in gold["original_value"]["indices"]
                    ):
                        step_idx_relaxed_correct += 1

            if (
                pred is not None
                and "repaired_steps" in pred
                and isinstance(gold.get("correct_steps"), list)
            ):
                rep_total += 1
                ex_ok, eq_ok = _score_repair(gold["correct_steps"], pred.get("repaired_steps"))
                if ex_ok:
                    rep_exact_ok += 1
                if eq_ok:
                    rep_equiv_ok += 1

        elif task == "error_repair":
            rep_total += 1
            correct_steps = gold.get("correct_steps")
            if not isinstance(correct_steps, list):
                raise ValueError("error_repair gold record requires 'correct_steps' list")
            cand_repair = pred.get("repaired_steps") if pred is not None else None
            ex_ok, eq_ok = _score_repair(correct_steps, cand_repair)
            if ex_ok:
                rep_exact_ok += 1
            if eq_ok:
                rep_equiv_ok += 1

        elif task == "multi_error_detection":
            multi_total += 1
            gold_labels = [
                normalize_error_label(str(x))[0] or str(x) for x in gold.get("labels", [])
            ]
            gold_indices = [int(x) for x in gold.get("step_indices", [])]
            rec_was_norm = False
            if pred is not None:
                if isinstance(pred.get("labels"), list):
                    pred_labels = []
                    for item_lbl in pred["labels"]:
                        nlbl, wnorm = normalize_error_label(str(item_lbl))
                        pred_labels.append(nlbl if nlbl is not None else str(item_lbl))
                        rec_was_norm = rec_was_norm or wnorm
                elif pred.get("label") is not None:
                    nlbl, wnorm = normalize_error_label(pred.get("label"))
                    rec_was_norm = rec_was_norm or wnorm
                    pred_labels = [nlbl] if (nlbl is not None and nlbl != "clean") else []
                else:
                    pred_labels = []
                if rec_was_norm:
                    normalized_preds += 1
                if isinstance(pred.get("step_indices"), list):
                    pred_indices = [
                        int(x)
                        for x in pred["step_indices"]
                        if isinstance(x, int) and not isinstance(x, bool)
                    ]
                elif isinstance(pred.get("step_index"), int) and not isinstance(
                    pred.get("step_index"), bool
                ):
                    pred_indices = [int(pred["step_index"])]
                else:
                    pred_indices = []
            else:
                pred_labels = []
                pred_indices = []

            if Counter(pred_labels) == Counter(gold_labels):
                multi_exact_ok += 1
            ltp, lpn, lgn = _multiset_prf(gold_labels, pred_labels)
            multi_lbl_tp += ltp
            multi_lbl_pred += lpn
            multi_lbl_gold += lgn

            gold_pairs = list(zip(gold_labels, gold_indices))
            pred_pairs = list(zip(pred_labels, pred_indices))
            ptp, ppn, pgn = _multiset_prf(gold_pairs, pred_pairs)
            multi_pair_tp += ptp
            multi_pair_pred += ppn
            multi_pair_gold += pgn

        elif task == "ordering":
            ord_total += 1
            gold_order = gold["gold_order"]
            constraints = [(int(pair[0]), int(pair[1])) for pair in gold.get("constraints", [])]
            pred_order = pred.get("pred_order") if pred is not None else None
            if (
                isinstance(pred_order, list)
                and len(pred_order) == len(gold_order)
                and all(isinstance(x, int) and not isinstance(x, bool) for x in pred_order)
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

    acc_clean = (clean_correct / clean_total) if clean_total > 0 else 0.0
    acc_err = (err_correct / err_total) if err_total > 0 else 0.0
    balanced_acc = (
        (2.0 * acc_clean * acc_err / (acc_clean + acc_err))
        if (clean_total > 0 and err_total > 0 and (acc_clean + acc_err) > 0.0)
        else 0.0
    )

    lbl_p, lbl_r, lbl_f1 = _safe_f1(multi_lbl_tp, multi_lbl_pred, multi_lbl_gold)
    pair_p, pair_r, pair_f1 = _safe_f1(multi_pair_tp, multi_pair_pred, multi_pair_gold)

    return {
        "protocol_count": len(protocol_ids),
        "error_detection": {
            "total": ed_total,
            "correct": ed_correct,
            "accuracy": (ed_correct / ed_total) if ed_total else 0.0,
            "clean_correct": clean_correct,
            "clean_total": clean_total,
            "clean_accuracy": acc_clean,
            "error_correct": err_correct,
            "error_total": err_total,
            "error_accuracy": acc_err,
            "balanced_accuracy": balanced_acc,
            "normalized_predictions": normalized_preds,
            "step_index_correct": step_idx_correct,
            "step_index_relaxed_correct": step_idx_relaxed_correct,
            "step_index_total": step_idx_total,
            "step_index_accuracy": (step_idx_correct / step_idx_total) if step_idx_total else 0.0,
            "step_index_relaxed_accuracy": (
                (step_idx_relaxed_correct / step_idx_total) if step_idx_total else 0.0
            ),
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
        "error_repair": {
            "total": rep_total,
            "exact_match": rep_exact_ok,
            "exact_accuracy": (rep_exact_ok / rep_total) if rep_total else 0.0,
            "dependency_equivalent": rep_equiv_ok,
            "equivalent_accuracy": (rep_equiv_ok / rep_total) if rep_total else 0.0,
        },
        "multi_error_detection": {
            "total": multi_total,
            "exact_match": multi_exact_ok,
            "exact_accuracy": (multi_exact_ok / multi_total) if multi_total else 0.0,
            "label_precision": lbl_p,
            "label_recall": lbl_r,
            "label_f1": lbl_f1,
            "pair_precision": pair_p,
            "pair_recall": pair_r,
            "pair_f1": pair_f1,
            "normalized_predictions": normalized_preds,
        },
    }


def evaluate_records(
    gold_records: Sequence[dict[str, Any]],
    pred_records: Sequence[dict[str, Any]],
    *,
    train_records: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute error-detection, step-ordering, error-repair, multi-fault, and per-split metrics."""
    matched = _match_predictions(gold_records, pred_records)
    normalized_pairs = [
        (_normalize_gold(raw_gold), _normalize_pred(raw_pred))
        for raw_gold, raw_pred in matched
    ]

    norm_golds = [g for g, _ in normalized_pairs]
    norm_train = [_normalize_gold(t) for t in train_records] if train_records is not None else None
    overlaps = detect_protocol_split_overlaps(norm_golds, train_records=norm_train)
    if overlaps:
        warnings.warn(
            f"Protocol leakage detected: protocol_id(s) {overlaps} appear in both train and test",
            UserWarning,
            stacklevel=2,
        )

    result = _compute_core_metrics(normalized_pairs)
    result["split_overlap_protocols"] = overlaps

    split_buckets: dict[str, list[tuple[dict[str, Any], dict[str, Any] | None]]] = {}
    for g, p in normalized_pairs:
        sp = g.get("split")
        if isinstance(sp, str) and sp in VALID_SPLITS:
            split_buckets.setdefault(sp, []).append((g, p))

    by_split: dict[str, dict[str, Any]] = {}
    for sp_name in VALID_SPLITS:
        if sp_name in split_buckets:
            by_split[sp_name] = _compute_core_metrics(split_buckets[sp_name])
    result["by_split"] = by_split
    return result


def format_evaluation_report(metrics: dict[str, Any]) -> str:
    """Format evaluation metrics as a compact human-readable report."""
    lines: list[str] = []
    overlaps = metrics.get("split_overlap_protocols") or []
    if overlaps:
        lines.append(
            f"WARNING: protocol_id overlap between train and test splits: {', '.join(overlaps)}"
        )

    ed = metrics["error_detection"]
    if ed["total"] == 0:
        lines.append("Error detection: no single-fault records")
    else:
        norm_suffix = (
            f" | normalized: {ed['normalized_predictions']}"
            if ed.get("normalized_predictions", 0) > 0
            else ""
        )
        lines.append(
            f"Error detection: {ed['correct']}/{ed['total']} ({ed['accuracy']:.1%}) "
            f"| balanced: {ed.get('balanced_accuracy', 0.0):.1%} "
            f"| step_index: {ed['step_index_correct']}/{ed['step_index_total']} "
            f"({ed['step_index_accuracy']:.1%}){norm_suffix}"
        )
        for err_type, stats in ed["per_error_type"].items():
            acc_str = f"{stats['accuracy']:.1%}"
            lines.append(
                f"  - {err_type:18s}: {stats['correct']:3d}/{stats['total']:<3d} ({acc_str})"
            )

    ordering = metrics["ordering"]
    if ordering["total"] > 0:
        c_part = (
            f"constraints {ordering['constraints_satisfied']}/{ordering['total']} "
            f"({ordering['constraint_accuracy']:.1%})"
        )
        e_part = (
            f"exact {ordering['exact_match']}/{ordering['total']} "
            f"({ordering['exact_accuracy']:.1%})"
        )
        p_part = (
            f"pairwise {ordering['pairwise_satisfied']}/{ordering['pairwise_total']} "
            f"({ordering['pairwise_accuracy']:.1%})"
        )
        lines.append(f"Ordering: {c_part} | {e_part} | {p_part}")

    repair = metrics.get("error_repair", {})
    if repair.get("total", 0) > 0:
        rx = f"exact {repair['exact_match']}/{repair['total']} ({repair['exact_accuracy']:.1%})"
        rq = (
            f"equivalent {repair['dependency_equivalent']}/{repair['total']} "
            f"({repair['equivalent_accuracy']:.1%})"
        )
        lines.append(f"Error repair: {rx} | {rq}")

    multi = metrics.get("multi_error_detection", {})
    if multi.get("total", 0) > 0:
        lines.append(
            f"Multi-error detection: exact {multi['exact_match']}/{multi['total']} "
            f"({multi['exact_accuracy']:.1%}) | label F1 {multi['label_f1']:.1%} "
            f"| pair F1 {multi['pair_f1']:.1%}"
        )

    by_split = metrics.get("by_split") or {}
    if len(by_split) >= 2:
        lines.append("Per-split breakdown:")
        for sp_name in VALID_SPLITS:
            if sp_name not in by_split:
                continue
            sm = by_split[sp_name]
            sed = sm["error_detection"]
            sord = sm["ordering"]
            ed_str = (
                f"{sed['correct']}/{sed['total']} ({sed['accuracy']:.1%}, "
                f"bal {sed['balanced_accuracy']:.1%})"
                if sed["total"] > 0
                else "no single-fault records"
            )
            ord_str = (
                f" | ord {sord['constraints_satisfied']}/{sord['total']} "
                f"({sord['constraint_accuracy']:.1%})"
                if sord["total"] > 0
                else ""
            )
            lines.append(
                f"  [{sp_name:5s}] ({sm['protocol_count']} protocols) ED {ed_str}{ord_str}"
            )
    return "\n".join(lines)

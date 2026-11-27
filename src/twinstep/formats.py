"""DPO, MPO, GDPO, SFT, and RLVR prompt-completion formatters for post-training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import json
from pathlib import Path
import re
from typing import Any

from twinstep.dependencies import find_dependencies
from twinstep.injectors import (
    ERROR_TYPES,
    _swap_candidates,
    classify_difficulty,
    filter_records_by_difficulty,
    generate_dataset,
)
from twinstep.schema import (
    IncubateStep,
    MixStep,
    Protocol,
    TransferStep,
    WashStep,
    steps_from_list,
    steps_to_dicts,
)
from twinstep.splits import assign_protocol_split, strip_for_blind_eval

EXPORT_FORMATS: tuple[str, ...] = ("raw", "dpo", "mpo", "gdpo", "sft", "rlvr")

_CONFUSABLE_LABELS: dict[str, str] = {
    "wrong_volume": "wrong_unit",
    "wrong_unit": "wrong_volume",
    "swap_order": "missing_step",
    "missing_step": "swap_order",
    "wrong_temperature": "wrong_time",
    "wrong_time": "wrong_temperature",
}

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _build_prompt(record: dict[str, Any]) -> str:
    task = record.get("task", "error_detection")
    if task == "ordering":
        steps_json = json.dumps(record["shuffled_steps"], indent=2)
        return (
            "Order the following shuffled liquid-handling protocol steps so all liquid-flow "
            "and plate-barrier dependencies are satisfied. Respond with JSON containing "
            "'pred_order' (0-based indices into shuffled_steps).\n\n"
            f"Shuffled steps:\n{steps_json}"
        )
    steps_json = json.dumps(record["corrupted_steps"], indent=2)
    allowed_fault_labels = ", ".join(f"'{lbl}'" for lbl in ERROR_TYPES)
    if task == "multi_error_detection":
        return (
            "Inspect the candidate liquid-handling protocol steps and identify all injected "
            f"errors. Allowed error labels in 'labels': {allowed_fault_labels} (or [] if clean). "
            "step_index is 0-based for every integer in 'step_indices'. Respond with JSON "
            "containing 'labels', 'step_indices', and 'repaired_steps'.\n\n"
            f"Candidate steps:\n{steps_json}"
        )
    return (
        "Inspect the candidate liquid-handling protocol steps, identify any error label and "
        "step_index, and provide 'repaired_steps' as JSON. "
        f"Allowed 'label' values: 'clean', {allowed_fault_labels}. "
        "step_index is 0-based (0 .. len(corrupted_steps) - 1 for mutated/swapped steps, "
        "0 .. len(corrupted_steps) insertion index for 'missing_step', or null when "
        "label is 'clean').\n\n"
        f"Candidate steps:\n{steps_json}"
    )


def _non_commuting_corruption(
    correct_dicts: Sequence[dict[str, Any]],
) -> tuple[str, int, list[dict[str, Any]]]:
    """Construct a genuinely invalid corruption for clean-protocol false-alarm negatives."""
    steps = steps_from_list(correct_dicts)
    swap_cands = _swap_candidates(steps)
    if swap_cands:
        i, j = swap_cands[0]
        swapped = list(steps)
        swapped[i], swapped[j] = swapped[j], swapped[i]
        return "swap_order", i, steps_to_dicts(swapped)

    for idx, step in enumerate(steps):
        if isinstance(step, (TransferStep, MixStep, WashStep)):
            flipped_unit: Any = "mL" if step.unit == "uL" else "uL"
            mut = list(steps)
            mut[idx] = replace(step, unit=flipped_unit)
            return "wrong_unit", idx, steps_to_dicts(mut)

    for idx, step in enumerate(steps):
        if isinstance(step, IncubateStep):
            mut = list(steps)
            mut[idx] = replace(step, temperature=step.temperature + 85.0)
            return "wrong_temperature", idx, steps_to_dicts(mut)

    if len(correct_dicts) >= 2:
        drop_idx = len(correct_dicts) - 1
        return "missing_step", drop_idx, [dict(d) for d in correct_dicts[:-1]]

    fallback = [dict(d) for d in correct_dicts]
    if fallback and fallback[0].get("type") == "read":
        fallback[0]["wavelength"] = 1200
    return "wrong_volume", 0, fallback


def _reward_weights_for_difficulty(difficulty: str) -> dict[str, float]:
    if difficulty in ("hard", "hard_saboteur"):
        return {
            "detection": 0.20,
            "localization": 0.25,
            "repair_dag": 0.25,
            "lean_gate": 0.30,
        }
    if difficulty == "medium":
        return {
            "detection": 0.25,
            "localization": 0.30,
            "repair_dag": 0.25,
            "lean_gate": 0.20,
        }
    return {
        "detection": 0.30,
        "localization": 0.25,
        "repair_dag": 0.25,
        "lean_gate": 0.20,
    }


def _mpo_weights_for_difficulty(difficulty: str) -> dict[str, float]:
    if difficulty in ("hard", "hard_saboteur"):
        return {"preference_weight": 0.75, "bco_quality_weight": 0.15, "sft_anchor_weight": 0.10}
    if difficulty == "medium":
        return {"preference_weight": 0.65, "bco_quality_weight": 0.20, "sft_anchor_weight": 0.15}
    return {"preference_weight": 0.55, "bco_quality_weight": 0.20, "sft_anchor_weight": 0.25}


def _build_structured_preferences(
    record: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return ``(chosen_payload, graded_rejected_candidates)`` with non-strawman negatives."""
    task = str(record.get("task", "error_detection"))
    rec_hash = sum(ord(ch) for ch in str(record.get("id", "0")))

    if task == "ordering":
        shuffled = list(record["shuffled_steps"])
        n = len(shuffled)
        gold_order = [int(x) for x in record["gold_order"]]
        chosen = {
            "pred_order": gold_order,
            "ordered_steps": record["correct_steps"],
        }
        constraints = [(int(p[0]), int(p[1])) for p in record.get("constraints", [])]
        candidates: list[dict[str, Any]] = []
        if constraints:
            u, v = constraints[0]
            near_miss = list(gold_order)
            pu = near_miss.index(u)
            pv = near_miss.index(v)
            near_miss[pu], near_miss[pv] = near_miss[pv], near_miss[pu]
            candidates.append(
                {
                    "payload": {
                        "pred_order": near_miss,
                        "ordered_steps": [shuffled[idx] for idx in near_miss],
                    },
                    "failure_mode": "single_edge_inversion",
                    "margin": 0.45,
                    "reward_components": {
                        "detection": 1.0,
                        "localization": 0.5,
                        "repair_dag": 0.0,
                        "lean_gate": 0.0,
                    },
                }
            )
            identity_ord = list(range(n))
            if identity_ord != near_miss and identity_ord != gold_order:
                candidates.append(
                    {
                        "payload": {
                            "pred_order": identity_ord,
                            "ordered_steps": shuffled,
                        },
                        "failure_mode": "unreordered_shuffle",
                        "margin": 0.85,
                        "reward_components": {
                            "detection": 0.0,
                            "localization": 0.0,
                            "repair_dag": 0.0,
                            "lean_gate": 0.0,
                        },
                    }
                )
        else:
            dup_ord = [0] * n
            candidates.append(
                {
                    "payload": {
                        "pred_order": dup_ord,
                        "ordered_steps": [shuffled[0] for _ in range(n)],
                    },
                    "failure_mode": "non_bijective_permutation",
                    "margin": 0.90,
                    "reward_components": {
                        "detection": 0.0,
                        "localization": 0.0,
                        "repair_dag": 0.0,
                        "lean_gate": 0.0,
                    },
                }
            )
        return chosen, candidates

    if task == "multi_error_detection":
        labels = [str(x) for x in record.get("labels", [])]
        step_indices = [int(x) for x in record.get("step_indices", [])]
        chosen = {
            "labels": labels,
            "step_indices": step_indices,
            "repaired_steps": record["correct_steps"],
        }
        if labels:
            multi_cands: list[dict[str, Any]] = []
            if len(labels) >= 2:
                multi_cands.append(
                    {
                        "payload": {
                            "labels": labels[:-1],
                            "step_indices": step_indices[:-1],
                            "repaired_steps": record["corrupted_steps"],
                        },
                        "failure_mode": "partial_fault_recall",
                        "margin": 0.40,
                        "reward_components": {
                            "detection": 0.5,
                            "localization": 0.5,
                            "repair_dag": 0.0,
                            "lean_gate": 0.0,
                        },
                    }
                )
            alt_last = _CONFUSABLE_LABELS.get(labels[-1], "wrong_volume")
            multi_cands.append(
                {
                    "payload": {
                        "labels": [*labels[:-1], alt_last],
                        "step_indices": step_indices,
                        "repaired_steps": record["corrupted_steps"],
                    },
                    "failure_mode": "confusable_fault_class",
                    "margin": 0.55,
                    "reward_components": {
                        "detection": 0.5 if len(labels) >= 2 else 0.0,
                        "localization": 0.5,
                        "repair_dag": 0.0,
                        "lean_gate": 0.0,
                    },
                }
            )
            multi_cands.append(
                {
                    "payload": {
                        "labels": [],
                        "step_indices": [],
                        "repaired_steps": record["corrupted_steps"],
                    },
                    "failure_mode": "missed_detection",
                    "margin": 0.95,
                    "reward_components": {
                        "detection": 0.0,
                        "localization": 0.0,
                        "repair_dag": 0.0,
                        "lean_gate": 0.0,
                    },
                }
            )
            return chosen, multi_cands

        fa_lbl, fa_idx, broken = _non_commuting_corruption(record["correct_steps"])
        return chosen, [
            {
                "payload": {
                    "labels": [fa_lbl],
                    "step_indices": [fa_idx],
                    "repaired_steps": broken,
                },
                "failure_mode": "false_alarm_corruption",
                "margin": 0.75,
                "reward_components": {
                    "detection": 0.0,
                    "localization": 0.0,
                    "repair_dag": 0.0,
                    "lean_gate": 0.0,
                },
            }
        ]

    label = str(record["label"])
    gold_idx = record.get("step_index")
    chosen = {
        "label": label,
        "step_index": gold_idx,
        "repaired_steps": record["correct_steps"],
    }
    if label == "clean":
        fa_lbl, fa_idx, broken = _non_commuting_corruption(record["correct_steps"])
        alt_lbl = _CONFUSABLE_LABELS.get(fa_lbl, "wrong_volume")
        return chosen, [
            {
                "payload": {
                    "label": fa_lbl,
                    "step_index": fa_idx,
                    "repaired_steps": broken,
                },
                "failure_mode": "false_alarm_corruption",
                "margin": 0.70,
                "reward_components": {
                    "detection": 0.0,
                    "localization": 0.0,
                    "repair_dag": 0.0,
                    "lean_gate": 0.0,
                },
            },
            {
                "payload": {
                    "label": alt_lbl,
                    "step_index": fa_idx,
                    "repaired_steps": broken,
                },
                "failure_mode": "false_alarm_confusable",
                "margin": 0.85,
                "reward_components": {
                    "detection": 0.0,
                    "localization": 0.0,
                    "repair_dag": 0.0,
                    "lean_gate": 0.0,
                },
            },
        ]

    corrupted_steps = record["corrupted_steps"]
    n_corr = max(1, len(corrupted_steps))
    base_idx = int(gold_idx) if isinstance(gold_idx, int) else 0
    forbidden = {base_idx}
    orig_val = record.get("original_value")
    if label == "swap_order" and isinstance(orig_val, dict):
        raw_indices = orig_val.get("indices")
        if isinstance(raw_indices, list):
            forbidden.update(int(x) for x in raw_indices if isinstance(x, int))

    wrong_idx: int | None = None
    for cand_pos in range(n_corr):
        shifted = (base_idx + 1 + cand_pos) % n_corr
        if shifted not in forbidden:
            wrong_idx = shifted
            break

    confusable = _CONFUSABLE_LABELS.get(label, "wrong_volume")
    cand_conf = {
        "payload": {
            "label": confusable,
            "step_index": base_idx,
            "repaired_steps": corrupted_steps,
        },
        "failure_mode": "confusable_fault_class",
        "margin": 0.50,
        "reward_components": {
            "detection": 0.0,
            "localization": 0.5,
            "repair_dag": 0.0,
            "lean_gate": 0.0,
        },
    }
    cand_miss = {
        "payload": {
            "label": "clean",
            "step_index": None,
            "repaired_steps": corrupted_steps,
        },
        "failure_mode": "missed_detection",
        "margin": 0.90,
        "reward_components": {
            "detection": 0.0,
            "localization": 0.0,
            "repair_dag": 0.0,
            "lean_gate": 0.0,
        },
    }

    if wrong_idx is not None:
        cand_loc = {
            "payload": {
                "label": label,
                "step_index": wrong_idx,
                "repaired_steps": corrupted_steps,
            },
            "failure_mode": "wrong_localization",
            "margin": 0.40,
            "reward_components": {
                "detection": 1.0,
                "localization": 0.0,
                "repair_dag": 0.0,
                "lean_gate": 0.0,
            },
        }
        ordered_hard = (
            [cand_loc, cand_conf, cand_miss]
            if rec_hash % 2 == 0
            else [cand_conf, cand_loc, cand_miss]
        )
        return chosen, ordered_hard

    return chosen, [cand_conf, cand_miss]


def _build_chosen_rejected(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    chosen, rejected_candidates = _build_structured_preferences(record)
    primary_rej: dict[str, Any] = dict(rejected_candidates[0]["payload"])
    return chosen, primary_rej


def _build_ground_truth(record: dict[str, Any], task: str, difficulty: str) -> dict[str, Any]:
    gt: dict[str, Any] = {"task": task, "difficulty": difficulty}
    for key in (
        "split",
        "label",
        "step_index",
        "original_value",
        "labels",
        "step_indices",
        "correct_steps",
        "gold_order",
        "constraints",
        "is_saboteur",
        "violated_lean_rules",
    ):
        if key in record:
            gt[key] = record[key]
    if "correct_steps" in record and "constraints" not in gt:
        try:
            parsed = steps_from_list(record["correct_steps"])
            gt["dag_edges"] = [[u, v] for u, v in find_dependencies(parsed)]
        except ValueError:
            pass
    return gt


def format_record(
    record: dict[str, Any],
    fmt: str = "raw",
    *,
    blind_eval: bool = False,
) -> dict[str, Any]:
    """Serialize a record into ``raw``, ``dpo``, ``mpo``, ``gdpo``, ``sft``, or ``rlvr``."""
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"unsupported export format {fmt!r}; expected one of {EXPORT_FORMATS}")

    difficulty = str(record.get("difficulty") or classify_difficulty(record))
    proto_id = str(record.get("protocol_id") or "")
    split = str(record.get("split") or assign_protocol_split(proto_id))
    if fmt == "raw":
        raw_copy = dict(record)
        raw_copy["split"] = split
        raw_copy["difficulty"] = difficulty
        return strip_for_blind_eval(raw_copy) if blind_eval else raw_copy

    prompt = _build_prompt(record)
    task = str(record.get("task", "error_detection"))
    if blind_eval:
        blind_out: dict[str, Any] = {
            "id": record.get("id"),
            "protocol_id": record.get("protocol_id"),
            "split": split,
            "task": task,
            "difficulty": difficulty,
            "format": fmt,
            "prompt": prompt,
        }
        if "corrupted_steps" in record:
            blind_out["corrupted_steps"] = record["corrupted_steps"]
        if "shuffled_steps" in record:
            blind_out["shuffled_steps"] = record["shuffled_steps"]
        return blind_out

    chosen_obj, rejected_candidates = _build_structured_preferences(record)
    rejected_obj = rejected_candidates[0]["payload"]

    if fmt == "dpo":
        return {
            "id": record.get("id"),
            "protocol_id": record.get("protocol_id"),
            "split": split,
            "task": task,
            "difficulty": difficulty,
            "format": "dpo",
            "prompt": prompt,
            "chosen": json.dumps(chosen_obj, sort_keys=False),
            "rejected": json.dumps(rejected_obj, sort_keys=False),
        }

    if fmt == "mpo":
        return {
            "id": record.get("id"),
            "protocol_id": record.get("protocol_id"),
            "split": split,
            "task": task,
            "difficulty": difficulty,
            "format": "mpo",
            "prompt": prompt,
            "chosen": json.dumps(chosen_obj, sort_keys=False),
            "rejected": json.dumps(rejected_obj, sort_keys=False),
            "rejected_candidates": [
                {
                    "completion": json.dumps(cand["payload"], sort_keys=False),
                    "failure_mode": cand["failure_mode"],
                    "margin": cand["margin"],
                    "reward_components": dict(cand["reward_components"]),
                }
                for cand in rejected_candidates
            ],
            "mpo_weights": _mpo_weights_for_difficulty(difficulty),
        }

    if fmt == "gdpo":
        chosen_rewards = {
            "detection": 1.0,
            "localization": 1.0,
            "repair_dag": 1.0,
            "lean_gate": 1.0,
        }
        return {
            "id": record.get("id"),
            "protocol_id": record.get("protocol_id"),
            "split": split,
            "task": task,
            "difficulty": difficulty,
            "format": "gdpo",
            "prompt": prompt,
            "chosen": json.dumps(chosen_obj, sort_keys=False),
            "rejected": json.dumps(rejected_obj, sort_keys=False),
            "chosen_rewards": chosen_rewards,
            "rejected_rewards": dict(rejected_candidates[0]["reward_components"]),
            "reward_weights": _reward_weights_for_difficulty(difficulty),
            "rejected_candidates": [
                {
                    "completion": json.dumps(cand["payload"], sort_keys=False),
                    "failure_mode": cand["failure_mode"],
                    "margin": cand["margin"],
                    "reward_components": dict(cand["reward_components"]),
                }
                for cand in rejected_candidates
            ],
            "ground_truth": _build_ground_truth({**record, "split": split}, task, difficulty),
        }

    if fmt == "sft":
        return {
            "id": record.get("id"),
            "protocol_id": record.get("protocol_id"),
            "split": split,
            "task": task,
            "difficulty": difficulty,
            "format": "sft",
            "prompt": prompt,
            "completion": json.dumps(chosen_obj, sort_keys=False),
        }

    # fmt == "rlvr"
    out: dict[str, Any] = {
        "id": record.get("id"),
        "protocol_id": record.get("protocol_id"),
        "split": split,
        "task": task,
        "difficulty": difficulty,
        "format": "rlvr",
        "prompt": prompt,
        "ground_truth": _build_ground_truth({**record, "split": split}, task, difficulty),
    }
    if "corrupted_steps" in record:
        out["corrupted_steps"] = record["corrupted_steps"]
    if "shuffled_steps" in record:
        out["shuffled_steps"] = record["shuffled_steps"]
    return out


def format_records(
    records: Sequence[dict[str, Any]],
    fmt: str = "raw",
    *,
    blind_eval: bool = False,
) -> list[dict[str, Any]]:
    """Serialize a batch of records into the requested export format."""
    return [format_record(rec, fmt=fmt, blind_eval=blind_eval) for rec in records]


def export_contrastive_dataset(
    protocols: Sequence[Protocol],
    *,
    fmt: str = "dpo",
    n_twins: int = 6,
    seed: int = 0,
    n_clean: int = 1,
    n_ordering: int = 2,
    include_compound: bool = True,
    include_saboteurs: bool = True,
    difficulty: str = "all",
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Build a difficulty-stratified (easy / medium / hard_saboteur) contrastive dataset."""
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"unsupported export format {fmt!r}; expected one of {EXPORT_FORMATS}")

    base_records = generate_dataset(
        protocols,
        n_twins=n_twins,
        seed=seed,
        n_clean=n_clean,
        n_ordering=n_ordering,
        k_errors=1,
        difficulty="all",
        include_saboteurs=include_saboteurs,
    )
    combined = list(base_records)
    if include_compound:
        eligible = [p for p in protocols if len(p.steps) >= 2]
        if eligible:
            compound_records = generate_dataset(
                eligible,
                n_twins=max(1, n_twins // 2),
                seed=seed + 101,
                n_clean=0,
                n_ordering=0,
                k_errors=2,
                difficulty="all",
                include_saboteurs=False,
            )
            combined.extend(compound_records)

    filtered = filter_records_by_difficulty(combined, difficulty=difficulty)
    formatted = format_records(filtered, fmt=fmt)

    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(item, sort_keys=False) for item in formatted]
        payload = ("\n".join(lines) + "\n") if lines else ""
        out_file.write_text(payload, encoding="utf-8", newline="\n")

    return formatted


def parse_completion_payload(text: str) -> dict[str, Any] | None:
    """Extract a structured prediction dict from a raw JSON or Markdown-fenced completion."""
    stripped = text.strip()
    if not stripped:
        return None
    fence_match = _FENCE_RE.search(stripped)
    candidate = fence_match.group(1) if fence_match is not None else stripped
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(stripped[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None

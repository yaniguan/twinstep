"""DPO, SFT, and RLVR prompt-completion formatters for post-training pipelines."""

from __future__ import annotations

from collections.abc import Sequence
import json
import re
from typing import Any

EXPORT_FORMATS: tuple[str, ...] = ("raw", "dpo", "sft", "rlvr")

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
    if task == "multi_error_detection":
        return (
            "Inspect the candidate liquid-handling protocol steps and identify all injected "
            "errors. Respond with JSON containing 'labels', 'step_indices', and "
            "'repaired_steps'.\n\n"
            f"Candidate steps:\n{steps_json}"
        )
    return (
        "Inspect the candidate liquid-handling protocol steps, identify any error label and "
        "step_index (or 'clean' with null step_index), and provide 'repaired_steps' as JSON.\n\n"
        f"Candidate steps:\n{steps_json}"
    )


def _build_chosen_rejected(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    task = record.get("task", "error_detection")
    if task == "ordering":
        n = len(record["shuffled_steps"])
        chosen = {
            "pred_order": record["gold_order"],
            "ordered_steps": record["correct_steps"],
        }
        rejected = {
            "pred_order": list(range(n)),
            "ordered_steps": record["shuffled_steps"],
        }
        return chosen, rejected

    if task == "multi_error_detection":
        labels = record.get("labels", [])
        chosen = {
            "labels": labels,
            "step_indices": record.get("step_indices", []),
            "repaired_steps": record["correct_steps"],
        }
        if labels:
            rejected = {
                "labels": [],
                "step_indices": [],
                "repaired_steps": record["corrupted_steps"],
            }
        else:
            rejected = {
                "labels": ["swap_order"],
                "step_indices": [0],
                "repaired_steps": list(reversed(record["correct_steps"])),
            }
        return chosen, rejected

    label = record["label"]
    chosen = {
        "label": label,
        "step_index": record.get("step_index"),
        "repaired_steps": record["correct_steps"],
    }
    if label == "clean":
        rejected = {
            "label": "swap_order",
            "step_index": 0,
            "repaired_steps": list(reversed(record["correct_steps"])),
        }
    else:
        rejected = {
            "label": "clean",
            "step_index": None,
            "repaired_steps": record["corrupted_steps"],
        }
    return chosen, rejected


def format_record(record: dict[str, Any], fmt: str = "raw") -> dict[str, Any]:
    """Serialize a generated record into ``raw``, ``dpo``, ``sft``, or ``rlvr`` format."""
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"unsupported export format {fmt!r}; expected one of {EXPORT_FORMATS}")
    if fmt == "raw":
        return dict(record)

    prompt = _build_prompt(record)
    chosen_obj, rejected_obj = _build_chosen_rejected(record)
    task = record.get("task", "error_detection")

    if fmt == "dpo":
        return {
            "id": record.get("id"),
            "protocol_id": record.get("protocol_id"),
            "task": task,
            "format": "dpo",
            "prompt": prompt,
            "chosen": json.dumps(chosen_obj, sort_keys=False),
            "rejected": json.dumps(rejected_obj, sort_keys=False),
        }

    if fmt == "sft":
        return {
            "id": record.get("id"),
            "protocol_id": record.get("protocol_id"),
            "task": task,
            "format": "sft",
            "prompt": prompt,
            "completion": json.dumps(chosen_obj, sort_keys=False),
        }

    # fmt == "rlvr"
    gt: dict[str, Any] = {"task": task}
    for key in (
        "label",
        "step_index",
        "original_value",
        "labels",
        "step_indices",
        "correct_steps",
        "gold_order",
        "constraints",
    ):
        if key in record:
            gt[key] = record[key]
    out: dict[str, Any] = {
        "id": record.get("id"),
        "protocol_id": record.get("protocol_id"),
        "task": task,
        "format": "rlvr",
        "prompt": prompt,
        "ground_truth": gt,
    }
    if "corrupted_steps" in record:
        out["corrupted_steps"] = record["corrupted_steps"]
    if "shuffled_steps" in record:
        out["shuffled_steps"] = record["shuffled_steps"]
    return out


def format_records(records: Sequence[dict[str, Any]], fmt: str = "raw") -> list[dict[str, Any]]:
    """Serialize a batch of records into the requested export format."""
    return [format_record(rec, fmt=fmt) for rec in records]


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

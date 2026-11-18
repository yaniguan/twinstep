"""Rule-based checker and topological ordering baseline (no reference peeking)."""

from __future__ import annotations

import heapq
from typing import Any, Sequence

from twinstep.dependencies import find_order_violations, is_well_ref, step_plates
from twinstep.schema import (
    IncubateStep,
    MixStep,
    ReadStep,
    Step,
    TransferStep,
    WashStep,
    step_from_dict,
)


def _to_steps(raw_steps: Sequence[Step | dict[str, Any]]) -> list[Step]:
    return [step_from_dict(s) if isinstance(s, dict) else s for s in raw_steps]


def _volume_in_ul(volume: float, unit: str) -> float:
    return volume * 1000.0 if unit == "mL" else volume


def predict_error(raw_steps: Sequence[Step | dict[str, Any]]) -> dict[str, Any]:
    """Predict whether a step sequence contains an error using sanity and dependency rules."""
    steps = _to_steps(raw_steps)

    # 1. Dependency order violations.
    violations = find_order_violations(steps)
    if violations:
        idx, _, reason = violations[0]
        return {"label": "swap_order", "step_index": idx, "reason": reason}

    # 2. Volume and unit sanity checks.
    for idx, step in enumerate(steps):
        if isinstance(step, (TransferStep, MixStep)):
            vol_ul = _volume_in_ul(step.volume, step.unit)
            if step.unit == "mL" and step.volume >= 5.0:
                return {
                    "label": "wrong_unit",
                    "step_index": idx,
                    "reason": f"step {idx} {step.type} uses {step.volume:g} mL (expected uL)",
                }
            if vol_ul < 1.0 or vol_ul > 300.0:
                return {
                    "label": "wrong_volume",
                    "step_index": idx,
                    "reason": f"step {idx} {step.type} volume {vol_ul:g} uL outside [1, 300] uL",
                }
        elif isinstance(step, WashStep):
            vol_ul = _volume_in_ul(step.volume, step.unit)
            if step.unit == "mL" and step.volume >= 10.0:
                return {
                    "label": "wrong_unit",
                    "step_index": idx,
                    "reason": f"step {idx} wash uses {step.volume:g} mL (expected uL)",
                }
            if step.unit == "uL" and step.volume <= 5.0:
                return {
                    "label": "wrong_unit",
                    "step_index": idx,
                    "reason": f"step {idx} wash uses {step.volume:g} uL (expected mL)",
                }
            if vol_ul < 50.0 or vol_ul > 2000.0:
                return {
                    "label": "wrong_volume",
                    "step_index": idx,
                    "reason": f"step {idx} wash volume {vol_ul:g} uL outside [50, 2000] uL",
                }

    # 3. Temperature and time sanity checks.
    for idx, step in enumerate(steps):
        if isinstance(step, IncubateStep):
            if step.temperature < 4.0 or step.temperature > 95.0:
                return {
                    "label": "wrong_temperature",
                    "step_index": idx,
                    "reason": (
                        f"step {idx} incubate temperature {step.temperature:g} C "
                        "outside [4, 95] C"
                    ),
                }
            if step.time_unit == "s" and (step.time < 5.0 or step.time > 600.0):
                return {
                    "label": "wrong_time",
                    "step_index": idx,
                    "reason": f"step {idx} incubate duration {step.time:g} s outside [5, 600] s",
                }
            if step.time_unit == "min" and (step.time < 1.0 or step.time > 180.0):
                return {
                    "label": "wrong_time",
                    "step_index": idx,
                    "reason": (
                        f"step {idx} incubate duration {step.time:g} min outside [1, 180] min"
                    ),
                }

    # 4. Missing step heuristics: uninitialized/unmixed wells or missing final read.
    fill_counts: dict[str, int] = {}
    mixed_since_fill: dict[str, bool] = {}
    for idx, step in enumerate(steps):
        if isinstance(step, TransferStep):
            if is_well_ref(step.source):
                if fill_counts.get(step.source, 0) == 0:
                    return {
                        "label": "missing_step",
                        "step_index": idx,
                        "reason": f"step {idx} transfers from uninitialized well {step.source}",
                    }
                if fill_counts.get(step.source, 0) >= 2 and not mixed_since_fill.get(step.source, False):
                    return {
                        "label": "missing_step",
                        "step_index": idx,
                        "reason": f"step {idx} transfers from unmixed well {step.source}",
                    }
            if is_well_ref(step.dest):
                fill_counts[step.dest] = fill_counts.get(step.dest, 0) + 1
                mixed_since_fill[step.dest] = False
        elif isinstance(step, MixStep):
            if is_well_ref(step.well):
                if fill_counts.get(step.well, 0) == 0:
                    return {
                        "label": "missing_step",
                        "step_index": idx,
                        "reason": f"step {idx} mixes uninitialized well {step.well}",
                    }
                mixed_since_fill[step.well] = True

    has_read = any(isinstance(step, ReadStep) for step in steps)
    if not has_read and len(steps) >= 2:
        return {
            "label": "missing_step",
            "step_index": len(steps),
            "reason": "protocol is missing a final read step",
        }

    return {
        "label": "clean",
        "step_index": None,
        "reason": "all sanity and dependency checks passed",
    }


def _stage_rank(step: Step) -> int:
    if isinstance(step, TransferStep):
        return 0
    if isinstance(step, MixStep):
        return 1
    if isinstance(step, IncubateStep):
        return 2
    if isinstance(step, WashStep):
        return 3
    if isinstance(step, ReadStep):
        return 4
    return 5


def predict_order(raw_steps: Sequence[Step | dict[str, Any]]) -> list[int]:
    """Reconstruct a dependency-valid execution order over shuffled steps."""
    steps = _to_steps(raw_steps)
    n = len(steps)
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    indegree = [0] * n

    def add_edge(u: int, v: int) -> None:
        if u != v and v not in adj[u]:
            adj[u].add(v)
            indegree[v] += 1

    for i, a in enumerate(steps):
        for j, b in enumerate(steps):
            if i == j:
                continue
            # Well liquid flow: transfer(dest=W) -> mix(W) -> transfer(source=W).
            if isinstance(a, TransferStep) and isinstance(b, MixStep) and a.dest == b.well:
                add_edge(i, j)
            elif isinstance(a, TransferStep) and isinstance(b, TransferStep) and a.dest == b.source:
                add_edge(i, j)
            elif isinstance(a, MixStep) and isinstance(b, TransferStep) and a.well == b.source:
                add_edge(i, j)
            # Plate barriers: liquid handling -> incubate -> wash -> read.
            pa = step_plates(a)
            pb = step_plates(b)
            same_plate = bool(pa & pb) or (
                isinstance(a, IncubateStep) and a.plate is None and bool(pb)
            ) or (
                isinstance(b, IncubateStep) and b.plate is None and bool(pa)
            )
            if same_plate and _stage_rank(a) < _stage_rank(b):
                if isinstance(b, (IncubateStep, WashStep, ReadStep)) or isinstance(
                    a, (IncubateStep, WashStep)
                ):
                    add_edge(i, j)

    heap: list[tuple[int, int]] = []
    for idx in range(n):
        if indegree[idx] == 0:
            heapq.heappush(heap, (_stage_rank(steps[idx]), idx))

    order: list[int] = []
    visited = [False] * n
    while heap:
        _, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        order.append(u)
        for v in sorted(adj[u]):
            indegree[v] -= 1
            if indegree[v] == 0:
                heapq.heappush(heap, (_stage_rank(steps[v]), v))

    if len(order) < n:
        remaining = [i for i in range(n) if not visited[i]]
        remaining.sort(key=lambda idx: (indegree[idx], _stage_rank(steps[idx]), idx))
        order.extend(remaining)
    return order


def predict_record(record: dict[str, Any]) -> dict[str, Any]:
    """Generate a prediction dict for either an error_detection or ordering record."""
    task = record.get("task", "error_detection")
    if task == "error_detection":
        pred = predict_error(record["corrupted_steps"])
        return {
            "id": record.get("id"),
            "task": "error_detection",
            "protocol_id": record.get("protocol_id"),
            "label": pred["label"],
            "step_index": pred["step_index"],
            "reason": pred["reason"],
        }
    if task == "ordering":
        pred_order = predict_order(record["shuffled_steps"])
        return {
            "id": record.get("id"),
            "task": "ordering",
            "protocol_id": record.get("protocol_id"),
            "pred_order": pred_order,
        }
    raise ValueError(f"unknown record task: {task!r}")

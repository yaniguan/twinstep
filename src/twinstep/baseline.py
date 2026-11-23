"""Rule-based checker and topological ordering baseline (no reference peeking)."""

from __future__ import annotations

from collections.abc import Sequence
import heapq
from typing import Any

from twinstep.dependencies import (
    canonical_well,
    find_order_violations,
    is_well_ref,
    plate_of,
    step_plates,
)
from twinstep.schema import (
    IncubateStep,
    MixStep,
    ReadStep,
    Step,
    TransferStep,
    WashStep,
    step_from_dict,
)

_MIN_WAVELENGTH_NM = 190
_MAX_WAVELENGTH_NM = 1100


def _classify_schema_error(raw_item: Any, idx: int, exc: ValueError) -> dict[str, Any]:
    """Map a schema-invalid candidate step dict to the closest error category at ``idx``."""
    msg = str(exc)
    if isinstance(raw_item, dict):
        stype = raw_item.get("type")
        if stype in ("transfer", "mix", "wash"):
            unit = raw_item.get("unit", "uL")
            if unit not in ("uL", "mL"):
                return {
                    "label": "wrong_unit",
                    "step_index": idx,
                    "reason": f"step {idx} has invalid volume unit {unit!r} ({msg})",
                }
        if stype == "incubate":
            t_unit = raw_item.get("time_unit", raw_item.get("unit", "min"))
            if t_unit not in ("min", "s"):
                return {
                    "label": "wrong_time",
                    "step_index": idx,
                    "reason": f"step {idx} has invalid time_unit {t_unit!r} ({msg})",
                }
    lower_msg = msg.lower()
    if "temperature" in lower_msg:
        label = "wrong_temperature"
    elif "time_unit" in lower_msg or "time" in lower_msg:
        label = "wrong_time"
    elif "unit" in lower_msg:
        label = "wrong_unit"
    elif "volume" in lower_msg or "cycles" in lower_msg:
        label = "wrong_volume"
    elif "wavelength" in lower_msg:
        label = "wrong_wavelength"
    else:
        label = "swap_order"
    return {
        "label": label,
        "step_index": idx,
        "reason": f"step {idx} failed schema validation: {msg}",
    }


def _volume_in_ul(volume: float, unit: str) -> float:
    return volume * 1000.0 if unit == "mL" else volume


def _check_single_step_bounds(idx: int, step: Step) -> dict[str, Any] | None:
    if isinstance(step, (TransferStep, MixStep)):
        vol_ul = _volume_in_ul(step.volume, step.unit)
        touches_well = (
            isinstance(step, MixStep)
            or is_well_ref(step.source)
            or is_well_ref(step.dest)
        )
        if step.unit == "mL" and ((touches_well and vol_ul > 300.0) or step.volume >= 5.0):
            return {
                "label": "wrong_unit",
                "step_index": idx,
                "reason": f"step {idx} {step.type} uses {step.volume:g} mL (expected uL)",
            }
        if (
            isinstance(step, TransferStep)
            and not touches_well
            and step.unit == "uL"
            and step.volume <= 10.0
        ):
            return {
                "label": "wrong_unit",
                "step_index": idx,
                "reason": f"step {idx} bulk transfer uses {step.volume:g} uL (expected mL)",
            }
        if touches_well and (vol_ul < 1.0 or vol_ul > 300.0):
            return {
                "label": "wrong_volume",
                "step_index": idx,
                "reason": f"step {idx} {step.type} volume {vol_ul:g} uL outside [1, 300] uL",
            }
        if not touches_well and (vol_ul < 1.0 or vol_ul > 50000.0):
            return {
                "label": "wrong_volume",
                "step_index": idx,
                "reason": f"step {idx} transfer volume {vol_ul:g} uL outside [1, 50000] uL",
            }
    elif isinstance(step, WashStep):
        vol_ul = _volume_in_ul(step.volume, step.unit)
        if step.unit == "mL" and vol_ul > 2000.0:
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
    elif isinstance(step, IncubateStep):
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
    elif isinstance(step, ReadStep):
        if step.wavelength is not None and (
            step.wavelength < _MIN_WAVELENGTH_NM or step.wavelength > _MAX_WAVELENGTH_NM
        ):
            return {
                "label": "wrong_wavelength",
                "step_index": idx,
                "reason": (
                    f"step {idx} read wavelength {step.wavelength} nm outside "
                    f"[{_MIN_WAVELENGTH_NM}, {_MAX_WAVELENGTH_NM}] nm"
                ),
            }
    return None


def _unmixed_well_on_plates(
    plates: set[str],
    fill_counts: dict[str, int],
    mixed_since_fill: dict[str, bool],
) -> str | None:
    for well in sorted(fill_counts):
        if fill_counts[well] < 2 or mixed_since_fill.get(well, False):
            continue
        w_plate = plate_of(well)
        if not plates or (w_plate is not None and w_plate in plates):
            return well
    return None


def predict_error(raw_steps: Sequence[Step | dict[str, Any]]) -> dict[str, Any]:
    """Predict whether a step sequence contains an error using sanity and dependency rules."""
    steps: list[Step] = []
    for idx, item in enumerate(raw_steps):
        if isinstance(item, dict):
            try:
                steps.append(step_from_dict(item))
            except ValueError as exc:
                return _classify_schema_error(item, idx, exc)
        else:
            steps.append(item)

    # 1. Dependency order violations.
    violations = find_order_violations(steps)
    if violations:
        idx, _, reason = violations[0]
        return {"label": "swap_order", "step_index": idx, "reason": reason}

    # 2 & 3. Volume, unit, temperature, time, and wavelength sanity checks.
    for idx, step in enumerate(steps):
        if isinstance(step, (TransferStep, MixStep, WashStep)):
            hit = _check_single_step_bounds(idx, step)
            if hit is not None:
                return hit
    for idx, step in enumerate(steps):
        if isinstance(step, (IncubateStep, ReadStep)):
            hit = _check_single_step_bounds(idx, step)
            if hit is not None:
                return hit

    # 4. Missing step heuristics: uninitialized/unmixed wells or missing final read.
    fill_counts: dict[str, int] = {}
    mixed_since_fill: dict[str, bool] = {}
    for idx, step in enumerate(steps):
        if isinstance(step, TransferStep):
            if is_well_ref(step.source):
                src_w = canonical_well(step.source)
                if fill_counts.get(src_w, 0) == 0:
                    return {
                        "label": "missing_step",
                        "step_index": idx,
                        "reason": f"step {idx} transfers from uninitialized well {step.source}",
                    }
                if fill_counts.get(src_w, 0) >= 2 and not mixed_since_fill.get(src_w, False):
                    return {
                        "label": "missing_step",
                        "step_index": idx,
                        "reason": f"step {idx} transfers from unmixed well {step.source}",
                    }
            if is_well_ref(step.dest):
                dst_w = canonical_well(step.dest)
                fill_counts[dst_w] = fill_counts.get(dst_w, 0) + 1
                mixed_since_fill[dst_w] = False
        elif isinstance(step, MixStep):
            if is_well_ref(step.well):
                w = canonical_well(step.well)
                if fill_counts.get(w, 0) == 0:
                    return {
                        "label": "missing_step",
                        "step_index": idx,
                        "reason": f"step {idx} mixes uninitialized well {step.well}",
                    }
                mixed_since_fill[w] = True
        elif isinstance(step, IncubateStep):
            unmixed = _unmixed_well_on_plates(step_plates(step), fill_counts, mixed_since_fill)
            if unmixed is not None:
                return {
                    "label": "missing_step",
                    "step_index": idx,
                    "reason": f"step {idx} incubates before mixing multi-reagent well {unmixed}",
                }
        elif isinstance(step, WashStep):
            if step.well is not None:
                w = canonical_well(step.well)
                if fill_counts.get(w, 0) >= 2 and not mixed_since_fill.get(w, False):
                    return {
                        "label": "missing_step",
                        "step_index": idx,
                        "reason": f"step {idx} washes unmixed multi-reagent well {w}",
                    }
                fill_counts[w] = 0
                mixed_since_fill[w] = False
            elif step.plate is not None:
                unmixed = _unmixed_well_on_plates({step.plate}, fill_counts, mixed_since_fill)
                if unmixed is not None:
                    return {
                        "label": "missing_step",
                        "step_index": idx,
                        "reason": (
                            f"step {idx} washes {step.plate} before mixing "
                            f"multi-reagent well {unmixed}"
                        ),
                    }
                for w in list(fill_counts):
                    if plate_of(w) == step.plate:
                        fill_counts[w] = 0
                        mixed_since_fill[w] = False
        elif isinstance(step, ReadStep):
            unmixed = _unmixed_well_on_plates({step.plate}, fill_counts, mixed_since_fill)
            if unmixed is not None:
                return {
                    "label": "missing_step",
                    "step_index": idx,
                    "reason": (
                        f"step {idx} reads {step.plate} before mixing "
                        f"multi-reagent well {unmixed}"
                    ),
                }

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


def predict_multi_errors(raw_steps: Sequence[Step | dict[str, Any]]) -> dict[str, Any]:
    """Scan all candidate steps to collect multiple fault labels and step indices."""
    found: list[tuple[int, str, str]] = []
    seen_indices: set[int] = set()
    valid_steps: list[Step | None] = []

    for idx, item in enumerate(raw_steps):
        if isinstance(item, dict):
            try:
                valid_steps.append(step_from_dict(item))
            except ValueError as exc:
                hit = _classify_schema_error(item, idx, exc)
                found.append((idx, hit["label"], hit["reason"]))
                seen_indices.add(idx)
                valid_steps.append(None)
        else:
            valid_steps.append(item)

    for idx, step in enumerate(valid_steps):
        if step is None or idx in seen_indices:
            continue
        hit = _check_single_step_bounds(idx, step)
        if hit is not None:
            found.append((idx, hit["label"], hit["reason"]))
            seen_indices.add(idx)

    clean_pairs = [(idx, s) for idx, s in enumerate(valid_steps) if s is not None]
    if len(clean_pairs) == len(valid_steps):
        only_steps = [s for _, s in clean_pairs]
        for v_idx, _, reason in find_order_violations(only_steps):
            if v_idx not in seen_indices:
                found.append((v_idx, "swap_order", reason))
                seen_indices.add(v_idx)

    found.sort(key=lambda item: item[0])
    return {
        "labels": [lbl for _, lbl, _ in found],
        "step_indices": [pos for pos, _, _ in found],
        "reasons": [rsn for _, _, rsn in found],
    }


def _stage_rank(step: Step | None) -> int:
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


def _assign_plate_cycles(steps: Sequence[Step | None]) -> list[int]:
    """Assign each step to a 0-based cycle segment when multiple read/wash barriers exist."""
    n = len(steps)
    cycles = [0] * n

    plate_reads: dict[str, list[int]] = {}
    plate_washes: dict[str, list[int]] = {}
    plate_incubates: dict[str, list[int]] = {}
    for idx, step in enumerate(steps):
        if isinstance(step, ReadStep):
            plate_reads.setdefault(step.plate, []).append(idx)
        elif isinstance(step, WashStep) and step.plate is not None and step.well is None:
            plate_washes.setdefault(step.plate, []).append(idx)
        elif isinstance(step, IncubateStep) and step.plate is not None:
            plate_incubates.setdefault(step.plate, []).append(idx)

    all_plates = set(plate_reads) | set(plate_washes) | set(plate_incubates)
    multi_plates = {
        p
        for p in all_plates
        if max(
            len(plate_reads.get(p, [])),
            len(plate_washes.get(p, [])),
            len(plate_incubates.get(p, [])),
        )
        >= 2
    }
    if not multi_plates:
        return cycles

    # Estimate causal liquid-flow depth per well from external-source transfers (depth 0).
    well_depth: dict[str, int] = {}
    for step in steps:
        if (
            isinstance(step, TransferStep)
            and not is_well_ref(step.source)
            and is_well_ref(step.dest)
        ):
            well_depth.setdefault(canonical_well(step.dest), 0)
    changed = True
    while changed:
        changed = False
        for step in steps:
            if (
                isinstance(step, TransferStep)
                and is_well_ref(step.source)
                and is_well_ref(step.dest)
            ):
                s_w = canonical_well(step.source)
                d_w = canonical_well(step.dest)
                if s_w in well_depth and d_w not in well_depth:
                    well_depth[d_w] = well_depth[s_w] + 1
                    changed = True

    for plate in sorted(multi_plates):
        reads = plate_reads.get(plate, [])
        washes = plate_washes.get(plate, [])
        incubates = plate_incubates.get(plate, [])
        num_cycles = max(len(reads), len(washes), len(incubates))

        for c_idx, step_idx in enumerate(reads):
            # Spread reads toward the end when fewer reads than washes; otherwise 1 per cycle.
            offset = num_cycles - len(reads) if len(reads) < num_cycles else 0
            cycles[step_idx] = min(num_cycles - 1, offset + c_idx)
        for c_idx, step_idx in enumerate(washes):
            cycles[step_idx] = min(num_cycles - 1, c_idx)
        for c_idx, step_idx in enumerate(incubates):
            cycles[step_idx] = min(num_cycles - 1, c_idx)

        transfers_by_well: dict[str, list[int]] = {}
        mixes_by_well: dict[str, list[int]] = {}
        for idx, step in enumerate(steps):
            if step is None or plate not in step_plates(step):
                continue
            if isinstance(step, TransferStep):
                key_w = (
                    canonical_well(step.dest)
                    if is_well_ref(step.dest)
                    else canonical_well(step.source)
                )
                transfers_by_well.setdefault(key_w, []).append(idx)
            elif isinstance(step, MixStep):
                mixes_by_well.setdefault(canonical_well(step.well), []).append(idx)

        max_depth = max(well_depth.values(), default=0)
        for well, idxs in transfers_by_well.items():
            if max_depth >= 1 and len(idxs) == 1:
                src_step = steps[idxs[0]]
                assert isinstance(src_step, TransferStep)
                if is_well_ref(src_step.source):
                    d = well_depth.get(canonical_well(src_step.source), 0) + 1
                else:
                    d = 0
                cycles[idxs[0]] = min(num_cycles - 1, d)
            else:
                for pos, step_idx in enumerate(idxs):
                    cycles[step_idx] = min(num_cycles - 1, (pos * num_cycles) // max(1, len(idxs)))

        for well, idxs in mixes_by_well.items():
            if max_depth >= 1 and len(idxs) == 1 and well in well_depth:
                cycles[idxs[0]] = min(num_cycles - 1, well_depth[well])
            else:
                for pos, step_idx in enumerate(idxs):
                    cycles[step_idx] = min(num_cycles - 1, (pos * num_cycles) // max(1, len(idxs)))

    return cycles


def predict_order(raw_steps: Sequence[Step | dict[str, Any]]) -> list[int]:
    """Reconstruct a dependency-valid execution order over shuffled steps.

    Single-cycle protocols use canonical stage ordering
    (``transfer -> mix -> incubate -> wash -> read``) plus well liquid-flow edges.
    When multiple ``ReadStep``/``WashStep``/``IncubateStep`` barriers appear on the same
    plate, steps are partitioned into cycle segments first, and cycle-forming reciprocal
    well transfers are oriented using external reservoir fills to prevent graph 2-cycles.
    """
    steps: list[Step | None] = []
    for item in raw_steps:
        if isinstance(item, dict):
            try:
                steps.append(step_from_dict(item))
            except ValueError:
                steps.append(None)
        else:
            steps.append(item)

    n = len(steps)
    cycles = _assign_plate_cycles(steps)
    has_multi_cycles = any(c > 0 for c in cycles)

    external_filled_wells: set[str] = {
        canonical_well(s.dest)
        for s in steps
        if isinstance(s, TransferStep) and not is_well_ref(s.source) and is_well_ref(s.dest)
    }

    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    indegree = [0] * n

    def _can_reach(start: int, target: int) -> bool:
        stack = [start]
        seen = {start}
        while stack:
            cur = stack.pop()
            if cur == target:
                return True
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    def add_edge(u: int, v: int) -> None:
        if u != v and v not in adj[u] and not _can_reach(v, u):
            adj[u].add(v)
            indegree[v] += 1

    # 1. Inter-cycle ordering edges when multiple cycles exist on a shared plate.
    if has_multi_cycles:
        for i in range(n):
            si = steps[i]
            if si is None:
                continue
            pi = step_plates(si)
            for j in range(n):
                sj = steps[j]
                if i == j or sj is None:
                    continue
                if cycles[i] < cycles[j] and bool(pi & step_plates(sj)):
                    add_edge(i, j)

    # 2. Well liquid flow and intra-cycle stage barriers.
    # Add external-fill transfers before inter-well transfers so reciprocal A1<->A2 pairs
    # orient cleanly from externally initialized wells outward.
    pair_indices = sorted(
        [(i, j) for i in range(n) for j in range(n) if i != j],
        key=lambda pair: (
            0
            if (
                isinstance(steps[pair[0]], TransferStep)
                and is_well_ref(steps[pair[0]].source)  # type: ignore[union-attr]
                and canonical_well(steps[pair[0]].source) in external_filled_wells  # type: ignore[union-attr]
            )
            else 1
        ),
    )

    for i, j in pair_indices:
        a = steps[i]
        b = steps[j]
        if a is None or b is None:
            continue
        if cycles[i] != cycles[j]:
            continue

        if (
            isinstance(a, TransferStep)
            and isinstance(b, MixStep)
            and is_well_ref(a.dest)
            and canonical_well(a.dest) == canonical_well(b.well)
        ):
            add_edge(i, j)
        elif (
            isinstance(a, TransferStep)
            and isinstance(b, TransferStep)
            and is_well_ref(a.dest)
            and canonical_well(a.dest) == canonical_well(b.source)
        ):
            add_edge(i, j)
        elif (
            isinstance(a, MixStep)
            and isinstance(b, TransferStep)
            and is_well_ref(a.well)
            and canonical_well(a.well) == canonical_well(b.source)
        ):
            add_edge(i, j)

        pa = step_plates(a)
        pb = step_plates(b)
        same_plate = (
            bool(pa & pb)
            or (isinstance(a, IncubateStep) and a.plate is None and bool(pb))
            or (isinstance(b, IncubateStep) and b.plate is None and bool(pa))
        )
        if same_plate:
            has_later_cycle = any(
                cycles[k] > cycles[i] and bool(pa & step_plates(steps[k]))
                for k in range(n)
                if steps[k] is not None
            )
            if (
                has_multi_cycles
                and isinstance(a, ReadStep)
                and isinstance(b, WashStep)
                and b.well is None
                and has_later_cycle
            ):
                add_edge(i, j)
            elif _stage_rank(a) < _stage_rank(b):
                if (
                    has_multi_cycles
                    and isinstance(a, WashStep)
                    and a.well is None
                    and isinstance(b, ReadStep)
                    and has_later_cycle
                ):
                    continue
                if isinstance(b, (IncubateStep, WashStep, ReadStep)) or isinstance(
                    a, (IncubateStep, WashStep)
                ):
                    add_edge(i, j)

    heap: list[tuple[int, int, int]] = []
    for idx in range(n):
        if indegree[idx] == 0:
            heapq.heappush(heap, (cycles[idx], _stage_rank(steps[idx]), idx))

    order: list[int] = []
    visited = [False] * n
    while heap:
        _, _, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        order.append(u)
        for v in sorted(adj[u]):
            indegree[v] -= 1
            if indegree[v] == 0:
                heapq.heappush(heap, (cycles[v], _stage_rank(steps[v]), v))

    if len(order) < n:
        remaining = [i for i in range(n) if not visited[i]]
        remaining.sort(key=lambda idx: (cycles[idx], indegree[idx], _stage_rank(steps[idx]), idx))
        order.extend(remaining)
    return order


def predict_record(record: dict[str, Any]) -> dict[str, Any]:
    """Generate a prediction dict for error_detection, multi_error_detection, or ordering."""
    if "ground_truth" in record and isinstance(record["ground_truth"], dict):
        gt = record["ground_truth"]
        task = str(record.get("task") or gt.get("task") or "error_detection")
    else:
        task = str(record.get("task", "error_detection"))

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
    if task == "multi_error_detection":
        mpred = predict_multi_errors(record["corrupted_steps"])
        return {
            "id": record.get("id"),
            "task": "multi_error_detection",
            "protocol_id": record.get("protocol_id"),
            "labels": mpred["labels"],
            "step_indices": mpred["step_indices"],
            "reasons": mpred["reasons"],
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

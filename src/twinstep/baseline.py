"""Rule-based checker and topological ordering baseline (no reference peeking)."""

from __future__ import annotations

from collections.abc import Sequence
import heapq
from typing import Any

from twinstep.dependencies import (
    canonical_well,
    find_order_violations,
    infer_default_plate,
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
    canonical_plate,
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
    default_plate: str = "plate1",
    excluded_wells: set[str] | None = None,
) -> str | None:
    for well in sorted(fill_counts):
        if excluded_wells is not None and well in excluded_wells:
            continue
        if fill_counts[well] < 2 or mixed_since_fill.get(well, False):
            continue
        w_plate = plate_of(well, default_plate)
        if not plates or (w_plate is not None and w_plate in plates):
            return well
    return None


def _wells_of_step(step: Step, default_plate: str) -> set[str]:
    wells: set[str] = set()
    if isinstance(step, TransferStep):
        if is_well_ref(step.source):
            wells.add(canonical_well(step.source, default_plate))
        if is_well_ref(step.dest):
            wells.add(canonical_well(step.dest, default_plate))
    elif isinstance(step, MixStep) and is_well_ref(step.well):
        wells.add(canonical_well(step.well, default_plate))
    elif isinstance(step, WashStep) and step.well is not None:
        wells.add(canonical_well(step.well, default_plate))
    return wells


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

    default_plate = infer_default_plate(steps)

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
                src_w = canonical_well(step.source, default_plate)
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
                dst_w = canonical_well(step.dest, default_plate)
                fill_counts[dst_w] = fill_counts.get(dst_w, 0) + 1
                mixed_since_fill[dst_w] = False
        elif isinstance(step, MixStep):
            if is_well_ref(step.well):
                w = canonical_well(step.well, default_plate)
                if fill_counts.get(w, 0) == 0:
                    return {
                        "label": "missing_step",
                        "step_index": idx,
                        "reason": f"step {idx} mixes uninitialized well {step.well}",
                    }
                mixed_since_fill[w] = True
        elif isinstance(step, IncubateStep):
            unmixed = _unmixed_well_on_plates(
                step_plates(step, default_plate),
                fill_counts,
                mixed_since_fill,
                default_plate,
            )
            if unmixed is not None:
                return {
                    "label": "missing_step",
                    "step_index": idx,
                    "reason": f"step {idx} incubates before mixing multi-reagent well {unmixed}",
                }
        elif isinstance(step, WashStep):
            if step.well is not None:
                w = canonical_well(step.well, default_plate)
                if fill_counts.get(w, 0) >= 2 and not mixed_since_fill.get(w, False):
                    return {
                        "label": "missing_step",
                        "step_index": idx,
                        "reason": f"step {idx} washes unmixed multi-reagent well {w}",
                    }
                fill_counts[w] = 0
                mixed_since_fill[w] = False
            elif step.plate is not None:
                p_norm = canonical_plate(step.plate)
                unmixed = _unmixed_well_on_plates(
                    {p_norm}, fill_counts, mixed_since_fill, default_plate
                )
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
                    if plate_of(w, default_plate) == p_norm:
                        fill_counts[w] = 0
                        mixed_since_fill[w] = False
        elif isinstance(step, ReadStep):
            p_norm = canonical_plate(step.plate)
            unmixed = _unmixed_well_on_plates(
                {p_norm}, fill_counts, mixed_since_fill, default_plate
            )
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
        bound_hit = _check_single_step_bounds(idx, step)
        if bound_hit is not None:
            found.append((idx, bound_hit["label"], bound_hit["reason"]))
            seen_indices.add(idx)

    clean_pairs = [(idx, s) for idx, s in enumerate(valid_steps) if s is not None]
    if clean_pairs:
        only_steps = [s for _, s in clean_pairs]
        default_plate = infer_default_plate(only_steps)
        swapped_wells: set[str] = set()
        for sub_v_idx, sub_other_idx, reason in find_order_violations(only_steps):
            orig_idx = clean_pairs[sub_v_idx][0]
            swapped_wells.update(_wells_of_step(only_steps[sub_v_idx], default_plate))
            if 0 <= sub_other_idx < len(only_steps):
                swapped_wells.update(_wells_of_step(only_steps[sub_other_idx], default_plate))
            if orig_idx not in seen_indices:
                found.append((orig_idx, "swap_order", reason))
                seen_indices.add(orig_idx)

        ever_filled = {
            canonical_well(s.dest, default_plate)
            for _, s in clean_pairs
            if isinstance(s, TransferStep) and is_well_ref(s.dest)
        }
        fill_counts: dict[str, int] = {}
        mixed_since_fill: dict[str, bool] = {}
        for orig_idx, step in clean_pairs:
            if isinstance(step, TransferStep):
                if is_well_ref(step.source):
                    src_w = canonical_well(step.source, default_plate)
                    if (
                        fill_counts.get(src_w, 0) == 0
                        and src_w not in ever_filled
                        and src_w not in swapped_wells
                        and orig_idx not in seen_indices
                    ):
                        found.append(
                            (
                                orig_idx,
                                "missing_step",
                                f"step {orig_idx} transfers from uninitialized well {step.source}",
                            )
                        )
                        seen_indices.add(orig_idx)
                    elif (
                        fill_counts.get(src_w, 0) >= 2
                        and not mixed_since_fill.get(src_w, False)
                        and src_w not in swapped_wells
                        and orig_idx not in seen_indices
                    ):
                        found.append(
                            (
                                orig_idx,
                                "missing_step",
                                f"step {orig_idx} transfers from unmixed well {step.source}",
                            )
                        )
                        seen_indices.add(orig_idx)
                        mixed_since_fill[src_w] = True
                if is_well_ref(step.dest):
                    dst_w = canonical_well(step.dest, default_plate)
                    fill_counts[dst_w] = fill_counts.get(dst_w, 0) + 1
                    mixed_since_fill[dst_w] = False
            elif isinstance(step, MixStep):
                if is_well_ref(step.well):
                    w = canonical_well(step.well, default_plate)
                    if (
                        fill_counts.get(w, 0) == 0
                        and w not in ever_filled
                        and w not in swapped_wells
                        and orig_idx not in seen_indices
                    ):
                        found.append(
                            (
                                orig_idx,
                                "missing_step",
                                f"step {orig_idx} mixes uninitialized well {step.well}",
                            )
                        )
                        seen_indices.add(orig_idx)
                    mixed_since_fill[w] = True
            elif isinstance(step, IncubateStep):
                unmixed = _unmixed_well_on_plates(
                    step_plates(step, default_plate),
                    fill_counts,
                    mixed_since_fill,
                    default_plate,
                    excluded_wells=swapped_wells,
                )
                if unmixed is not None and orig_idx not in seen_indices:
                    found.append(
                        (
                            orig_idx,
                            "missing_step",
                            f"step {orig_idx} incubates before mixing multi-reagent well {unmixed}",
                        )
                    )
                    seen_indices.add(orig_idx)
                    mixed_since_fill[unmixed] = True
            elif isinstance(step, WashStep):
                if step.well is not None:
                    w = canonical_well(step.well, default_plate)
                    if (
                        fill_counts.get(w, 0) >= 2
                        and not mixed_since_fill.get(w, False)
                        and w not in swapped_wells
                        and orig_idx not in seen_indices
                    ):
                        found.append(
                            (
                                orig_idx,
                                "missing_step",
                                f"step {orig_idx} washes unmixed multi-reagent well {w}",
                            )
                        )
                        seen_indices.add(orig_idx)
                    fill_counts[w] = 0
                    mixed_since_fill[w] = False
                elif step.plate is not None:
                    p_norm = canonical_plate(step.plate)
                    unmixed = _unmixed_well_on_plates(
                        {p_norm},
                        fill_counts,
                        mixed_since_fill,
                        default_plate,
                        excluded_wells=swapped_wells,
                    )
                    if unmixed is not None and orig_idx not in seen_indices:
                        found.append(
                            (
                                orig_idx,
                                "missing_step",
                                f"step {orig_idx} washes {step.plate} before mixing "
                                f"multi-reagent well {unmixed}",
                            )
                        )
                        seen_indices.add(orig_idx)
                    for w in list(fill_counts):
                        if plate_of(w, default_plate) == p_norm:
                            fill_counts[w] = 0
                            mixed_since_fill[w] = False
            elif isinstance(step, ReadStep):
                p_norm = canonical_plate(step.plate)
                unmixed = _unmixed_well_on_plates(
                    {p_norm},
                    fill_counts,
                    mixed_since_fill,
                    default_plate,
                    excluded_wells=swapped_wells,
                )
                if unmixed is not None and orig_idx not in seen_indices:
                    found.append(
                        (
                            orig_idx,
                            "missing_step",
                            f"step {orig_idx} reads {step.plate} before mixing "
                            f"multi-reagent well {unmixed}",
                        )
                    )
                    seen_indices.add(orig_idx)
                    mixed_since_fill[unmixed] = True

    has_any_read = any(isinstance(s, ReadStep) for _, s in clean_pairs) or any(
        isinstance(item, dict) and item.get("type") == "read" for item in raw_steps
    )
    if not has_any_read and len(raw_steps) >= 2 and len(raw_steps) not in seen_indices:
        found.append((len(raw_steps), "missing_step", "protocol is missing a final read step"))
        seen_indices.add(len(raw_steps))

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


def _transfer_role_key(step: TransferStep, orig_idx: int) -> tuple[int, float, int]:
    src_low = step.source.lower()
    if any(
        tok in src_low
        for tok in ("buf", "coat", "block", "sample", "water", "pbs", "media", "diluent")
    ):
        role = 0
    elif any(
        tok in src_low
        for tok in (
            "substrate",
            "dye",
            "detect",
            "conjugate",
            "stop",
            "quench",
            "elut",
            "develop",
            "secondary",
        )
    ):
        role = 2
    else:
        role = 1
    return (role, -_volume_in_ul(step.volume, step.unit), orig_idx)


def _transfer_idx_key(steps: Sequence[Step | None], s_idx: int) -> tuple[int, float, int]:
    st = steps[s_idx]
    if isinstance(st, TransferStep):
        return _transfer_role_key(st, s_idx)
    return (1, 0.0, s_idx)


def _mix_idx_key(steps: Sequence[Step | None], m_idx: int) -> tuple[float, int, int]:
    st = steps[m_idx]
    if isinstance(st, MixStep):
        return (_volume_in_ul(st.volume, st.unit), st.cycles, m_idx)
    return (0.0, 0, m_idx)


def _assign_plate_cycles(
    steps: Sequence[Step | None],
    default_plate: str = "plate1",
) -> dict[str, list[int]]:
    """Assign each step per-plate 0-based cycle coordinates across barrier segments."""
    n = len(steps)
    all_plates: set[str] = set()
    for step in steps:
        if step is not None:
            all_plates.update(step_plates(step, default_plate))
    if not all_plates and any(isinstance(s, IncubateStep) for s in steps):
        all_plates.add(canonical_plate(default_plate))

    plate_reads: dict[str, list[int]] = {}
    plate_washes: dict[str, list[int]] = {}
    plate_incubates: dict[str, list[int]] = {}
    for idx, step in enumerate(steps):
        if isinstance(step, ReadStep):
            plate_reads.setdefault(canonical_plate(step.plate), []).append(idx)
        elif isinstance(step, WashStep) and step.plate is not None and step.well is None:
            plate_washes.setdefault(canonical_plate(step.plate), []).append(idx)
        elif isinstance(step, IncubateStep):
            if step.plate is not None:
                plate_incubates.setdefault(canonical_plate(step.plate), []).append(idx)
            else:
                for p in all_plates:
                    plate_incubates.setdefault(p, []).append(idx)

    plate_cycles: dict[str, list[int]] = {p: [0] * n for p in all_plates}
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
        return plate_cycles

    # Estimate causal liquid-flow depth per well from external-source transfers (depth 0).
    well_depth: dict[str, int] = {}
    for step in steps:
        if (
            isinstance(step, TransferStep)
            and not is_well_ref(step.source)
            and is_well_ref(step.dest)
        ):
            well_depth.setdefault(canonical_well(step.dest, default_plate), 0)
    changed = True
    while changed:
        changed = False
        for step in steps:
            if (
                isinstance(step, TransferStep)
                and is_well_ref(step.source)
                and is_well_ref(step.dest)
            ):
                s_w = canonical_well(step.source, default_plate)
                d_w = canonical_well(step.dest, default_plate)
                if s_w in well_depth and d_w not in well_depth:
                    well_depth[d_w] = well_depth[s_w] + 1
                    changed = True

    max_depth = max(well_depth.values(), default=0)

    for plate in sorted(multi_plates):
        p_cyc = plate_cycles[plate]
        reads = plate_reads.get(plate, [])
        washes = plate_washes.get(plate, [])
        incubates = plate_incubates.get(plate, [])
        num_cycles = max(len(reads), len(washes), len(incubates))

        for c_idx, step_idx in enumerate(reads):
            offset = num_cycles - len(reads) if len(reads) < num_cycles else 0
            p_cyc[step_idx] = min(num_cycles - 1, offset + c_idx)
        for c_idx, step_idx in enumerate(washes):
            p_cyc[step_idx] = min(num_cycles - 1, c_idx)
        for c_idx, step_idx in enumerate(incubates):
            inc_offset = (
                num_cycles - len(incubates)
                if (len(incubates) < num_cycles and len(washes) > 0)
                else 0
            )
            p_cyc[step_idx] = min(num_cycles - 1, inc_offset + c_idx)

        incoming_by_well: dict[str, list[int]] = {}
        outgoing_only: list[int] = []
        mixes_by_well: dict[str, list[int]] = {}
        for idx, step in enumerate(steps):
            if step is None or plate not in step_plates(step, default_plate):
                continue
            if isinstance(step, TransferStep):
                dst_p = (
                    plate_of(step.dest, default_plate) if is_well_ref(step.dest) else None
                )
                src_p = (
                    plate_of(step.source, default_plate) if is_well_ref(step.source) else None
                )
                if dst_p == plate:
                    dst_w = canonical_well(step.dest, default_plate)
                    incoming_by_well.setdefault(dst_w, []).append(idx)
                elif src_p == plate:
                    outgoing_only.append(idx)
            elif isinstance(step, MixStep):
                mixes_by_well.setdefault(canonical_well(step.well, default_plate), []).append(idx)

        well_max_cycle: dict[str, int] = {}
        for well, idxs in incoming_by_well.items():
            if len(idxs) == 1:
                src_step = steps[idxs[0]]
                assert isinstance(src_step, TransferStep)
                if max_depth >= 1 and is_well_ref(src_step.source):
                    d = well_depth.get(canonical_well(src_step.source, default_plate), 0) + 1
                else:
                    d = 0
                cyc_val = min(num_cycles - 1, d)
                p_cyc[idxs[0]] = cyc_val
                well_max_cycle[well] = max(well_max_cycle.get(well, 0), cyc_val)
            else:
                ordered_idxs = sorted(
                    idxs,
                    key=lambda s_idx: _transfer_idx_key(steps, s_idx),
                )
                for pos, step_idx in enumerate(ordered_idxs):
                    cyc_val = min(
                        num_cycles - 1, (pos * num_cycles) // max(1, len(ordered_idxs))
                    )
                    p_cyc[step_idx] = cyc_val
                    well_max_cycle[well] = max(well_max_cycle.get(well, 0), cyc_val)

        for step_idx in outgoing_only:
            out_step = steps[step_idx]
            assert isinstance(out_step, TransferStep)
            src_w = canonical_well(out_step.source, default_plate)
            if src_w in well_max_cycle:
                p_cyc[step_idx] = well_max_cycle[src_w]
            else:
                p_cyc[step_idx] = min(num_cycles - 1, well_depth.get(src_w, 0) + 1)

        for well, idxs in mixes_by_well.items():
            if (
                max_depth >= 1
                and len(idxs) == 1
                and well in well_depth
                and len(incoming_by_well.get(well, [])) <= 1
            ):
                p_cyc[idxs[0]] = min(num_cycles - 1, well_depth[well])
            else:
                ordered_mixes = sorted(
                    idxs,
                    key=lambda m_idx: _mix_idx_key(steps, m_idx),
                )
                for pos, step_idx in enumerate(ordered_mixes):
                    p_cyc[step_idx] = min(
                        num_cycles - 1, (pos * num_cycles) // max(1, len(ordered_mixes))
                    )

    return plate_cycles


def predict_order(raw_steps: Sequence[Step | dict[str, Any]]) -> list[int]:
    """Reconstruct a dependency-valid execution order over shuffled steps.

    Single-cycle protocols use canonical stage ordering
    (``transfer -> mix -> incubate -> wash -> read``) plus well liquid-flow edges.
    When multiple ``ReadStep``/``WashStep``/``IncubateStep`` barriers appear on the same
    plate, steps are partitioned into per-plate cycle segments first, and cycle-forming
    reciprocal well transfers are oriented using external reservoir fills.
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
    default_plate = infer_default_plate(steps)
    plate_cycles = _assign_plate_cycles(steps, default_plate)
    all_plates = set(plate_cycles)
    multi_plates = {
        p for p, cyc_list in plate_cycles.items() if any(c > 0 for c in cyc_list)
    }

    def _effective_plates(step: Step | None) -> set[str]:
        if step is None:
            return set()
        if isinstance(step, IncubateStep) and step.plate is None:
            return set(all_plates)
        return step_plates(step, default_plate)

    external_filled_wells: set[str] = {
        canonical_well(s.dest, default_plate)
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

    # 1. Inter-cycle ordering edges on shared multi-cycle plates only.
    for p in sorted(multi_plates):
        p_cyc = plate_cycles[p]
        touching = [idx for idx in range(n) if p in _effective_plates(steps[idx])]
        for i in touching:
            for j in touching:
                if i != j and p_cyc[i] < p_cyc[j]:
                    add_edge(i, j)

    # 1b. Multi-mix sub-stages within the same (well, cycle): e.g. homogenize before quench.
    well_cycle_transfers: dict[tuple[str, int], list[int]] = {}
    well_cycle_mixes: dict[tuple[str, int], list[int]] = {}
    for idx, step in enumerate(steps):
        if isinstance(step, TransferStep) and is_well_ref(step.dest):
            w = canonical_well(step.dest, default_plate)
            pw = plate_of(step.dest, default_plate)
            c = plate_cycles[pw][idx] if pw is not None and pw in plate_cycles else 0
            well_cycle_transfers.setdefault((w, c), []).append(idx)
        elif isinstance(step, MixStep) and is_well_ref(step.well):
            w = canonical_well(step.well, default_plate)
            pw = plate_of(step.well, default_plate)
            c = plate_cycles[pw][idx] if pw is not None and pw in plate_cycles else 0
            well_cycle_mixes.setdefault((w, c), []).append(idx)

    multi_mix_keys: set[tuple[str, int]] = set()
    for wc_key, mix_idxs in well_cycle_mixes.items():
        if len(mix_idxs) < 2:
            continue
        multi_mix_keys.add(wc_key)
        sorted_mixes = sorted(
            mix_idxs,
            key=lambda m_idx: _mix_idx_key(steps, m_idx),
        )
        for m_pos in range(len(sorted_mixes) - 1):
            add_edge(sorted_mixes[m_pos], sorted_mixes[m_pos + 1])

        t_idxs = well_cycle_transfers.get(wc_key, [])
        if t_idxs:
            sorted_transfers = sorted(
                t_idxs,
                key=lambda t_idx: _transfer_idx_key(steps, t_idx),
            )
            num_m = len(sorted_mixes)
            for t_pos, t_idx in enumerate(sorted_transfers):
                # Place quench/additive transfers after upstream mix; early fills before mix 0.
                stage = min(
                    num_m - 1,
                    max(0, num_m - (len(sorted_transfers) - t_pos)),
                )
                add_edge(t_idx, sorted_mixes[stage])
                if stage > 0:
                    add_edge(sorted_mixes[stage - 1], t_idx)

    # 2. Well liquid flow and intra-cycle stage barriers.
    pair_indices = sorted(
        [(i, j) for i in range(n) for j in range(n) if i != j],
        key=lambda pair: (
            0
            if (
                isinstance(steps[pair[0]], TransferStep)
                and is_well_ref(steps[pair[0]].source)  # type: ignore[union-attr]
                and canonical_well(
                    steps[pair[0]].source,  # type: ignore[union-attr]
                    default_plate,
                )
                in external_filled_wells
            )
            else 1
        ),
    )

    for i, j in pair_indices:
        a = steps[i]
        b = steps[j]
        if a is None or b is None:
            continue

        if (
            isinstance(a, TransferStep)
            and isinstance(b, MixStep)
            and is_well_ref(a.dest)
            and canonical_well(a.dest, default_plate) == canonical_well(b.well, default_plate)
        ):
            w = canonical_well(a.dest, default_plate)
            pw = plate_of(a.dest, default_plate)
            ca = plate_cycles[pw][i] if pw is not None and pw in plate_cycles else 0
            cb = plate_cycles[pw][j] if pw is not None and pw in plate_cycles else 0
            if ca == cb and (w, ca) not in multi_mix_keys:
                add_edge(i, j)
        elif (
            isinstance(a, TransferStep)
            and isinstance(b, TransferStep)
            and is_well_ref(a.dest)
            and canonical_well(a.dest, default_plate) == canonical_well(b.source, default_plate)
        ):
            pw = plate_of(a.dest, default_plate)
            ca = plate_cycles[pw][i] if pw is not None and pw in plate_cycles else 0
            cb = plate_cycles[pw][j] if pw is not None and pw in plate_cycles else 0
            if ca == cb:
                add_edge(i, j)
        elif (
            isinstance(a, MixStep)
            and isinstance(b, TransferStep)
            and is_well_ref(a.well)
            and canonical_well(a.well, default_plate) == canonical_well(b.source, default_plate)
        ):
            pw = plate_of(a.well, default_plate)
            ca = plate_cycles[pw][i] if pw is not None and pw in plate_cycles else 0
            cb = plate_cycles[pw][j] if pw is not None and pw in plate_cycles else 0
            if ca == cb:
                add_edge(i, j)

        shared_plates = _effective_plates(a) & _effective_plates(b)
        for p in sorted(shared_plates):
            p_cyc = plate_cycles[p]
            if p_cyc[i] != p_cyc[j]:
                continue
            p_is_multi = p in multi_plates
            has_later_cycle = p_is_multi and any(
                p_cyc[k] > p_cyc[i] and p in _effective_plates(sk)
                for k, sk in enumerate(steps)
                if sk is not None
            )
            if (
                p_is_multi
                and isinstance(a, ReadStep)
                and isinstance(b, WashStep)
                and b.well is None
                and has_later_cycle
            ):
                add_edge(i, j)
            elif _stage_rank(a) < _stage_rank(b):
                if (
                    p_is_multi
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

    def _step_cycle_rank(idx: int) -> int:
        eff = _effective_plates(steps[idx])
        if not eff:
            return 0
        return max(plate_cycles[p][idx] for p in eff if p in plate_cycles)

    heap: list[tuple[int, int, int]] = []
    for idx in range(n):
        if indegree[idx] == 0:
            heapq.heappush(heap, (_step_cycle_rank(idx), _stage_rank(steps[idx]), idx))

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
                heapq.heappush(heap, (_step_cycle_rank(v), _stage_rank(steps[v]), v))

    if len(order) < n:
        remaining = [i for i in range(n) if not visited[i]]
        remaining.sort(
            key=lambda idx: (
                _step_cycle_rank(idx),
                indegree[idx],
                _stage_rank(steps[idx]),
                idx,
            )
        )
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

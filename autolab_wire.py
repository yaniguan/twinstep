"""Auto-Lab 15-repository wire contract node for twinstep (autolab-wire/v1)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import copy
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

REPO_ID = "twinstep"
LAYER = "L3"
REPO_DIR = Path(__file__).resolve().parent
SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from twinstep.dependencies import check_constraints, find_dependencies  # noqa: E402
from twinstep.formats import format_record  # noqa: E402
from twinstep.injectors import (  # noqa: E402
    TwinRecord,
    _swap_candidates,
    classify_difficulty,
    generate_dataset,
    inject_error,
)
from twinstep.interop import from_stepbot_dict, from_stepbot_yaml  # noqa: E402
from twinstep.schema import (  # noqa: E402
    IncubateStep,
    MixStep,
    Protocol,
    ReadStep,
    Step,
    TransferStep,
    WashStep,
    steps_from_list,
    steps_to_dicts,
)

_WIRE_LENGTH_FORMATS: tuple[str, ...] = ("raw", "dpo", "sft", "rlvr")


def _safe_positive_float(value: Any, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        return default
    return parsed


def _safe_temp_c(value: Any, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= -273.15:
        return default
    return parsed


def _resolve_candidate_order(
    candidate: Any,
    steps: Sequence[Step],
) -> list[int] | None:
    """Resolve an integer-index or step-name ordering into a bijective 0..N-1 permutation."""
    if not isinstance(candidate, (list, tuple)) or len(candidate) != len(steps):
        return None

    n_steps = len(steps)
    if all(isinstance(item, int) and not isinstance(item, bool) for item in candidate):
        indices = [int(item) for item in candidate]
        if set(indices) != set(range(n_steps)):
            return None
        return indices

    raw_names: list[str] = []
    for item in candidate:
        if isinstance(item, str):
            raw_names.append(item.strip().lower())
        elif isinstance(item, dict) and isinstance(item.get("type"), str):
            raw_names.append(str(item["type"]).strip().lower())
        else:
            return None

    used: set[int] = set()
    resolved: list[int] = []
    for step_name in raw_names:
        matched_idx: int | None = None
        for idx, step in enumerate(steps):
            if idx not in used and step.type == step_name:
                matched_idx = idx
                break
        if matched_idx is None:
            return None
        used.add(matched_idx)
        resolved.append(matched_idx)
    return resolved


def _estimate_token_lengths(records: Sequence[dict[str, Any]]) -> list[int]:
    """Compute realistic post-training sequence lengths in [48, 240] for labfill bin packing."""
    lengths: list[int] = []
    for idx, rec in enumerate(records):
        fmt = _WIRE_LENGTH_FORMATS[idx % len(_WIRE_LENGTH_FORMATS)]
        formatted = format_record(rec, fmt=fmt)
        serialized = json.dumps(formatted, sort_keys=True)
        approx_tokens = round(len(serialized) / 4.2)
        lengths.append(max(48, min(240, int(approx_tokens))))

    fallback_defaults = (96, 144, 180, 72)
    while len(lengths) < 4:
        lengths.append(fallback_defaults[len(lengths)])
    return lengths


def _extract_protocol_yaml_dict(out: dict[str, Any]) -> dict[str, Any] | None:
    direct = out.get("protocol_yaml_dict")
    if isinstance(direct, dict):
        return direct
    for container_key in ("stepbot", "rehearsal", "lab_rehearsal"):
        sub = out.get(container_key)
        if isinstance(sub, dict) and isinstance(sub.get("protocol_yaml_dict"), dict):
            return sub["protocol_yaml_dict"]
    return None


def _lower_wire_protocol(out: dict[str, Any]) -> Protocol:
    """Lower upstream stepbot/rehearsal YAML/dict/step payloads or synthesize 4-step fallback."""
    explicit_id = out.get("protocol_id")

    yaml_payload = out.get("stepbot_yaml", out.get("protocol_yaml"))
    if isinstance(yaml_payload, (str, Path)) and str(yaml_payload).strip():
        proto = from_stepbot_yaml(yaml_payload)
        if explicit_id:
            return Protocol(protocol_id=str(explicit_id), steps=proto.steps)
        return proto

    sb_dict = out.get("stepbot_protocol")
    if isinstance(sb_dict, dict):
        proto = from_stepbot_dict(sb_dict)
        if explicit_id:
            return Protocol(protocol_id=str(explicit_id), steps=proto.steps)
        return proto

    yaml_dict = _extract_protocol_yaml_dict(out)
    if yaml_dict is not None:
        proto = from_stepbot_dict(yaml_dict)
        if explicit_id:
            return Protocol(protocol_id=str(explicit_id), steps=proto.steps)
        return proto

    raw_steps = out.get("protocol_steps")
    if isinstance(raw_steps, (list, tuple)) and raw_steps:
        pid = str(explicit_id or "autolab_wire_protocol")
        try:
            typed_steps = tuple(steps_from_list(raw_steps))
            return Protocol(protocol_id=pid, steps=typed_steps)
        except ValueError:
            return from_stepbot_dict({"protocol_id": pid, "steps": list(raw_steps)})

    raw_spans = out.get("parsed_spans")
    parsed_spans = raw_spans if isinstance(raw_spans, dict) else {}

    volume_ul = _safe_positive_float(parsed_spans.get("volume_ul", 100.0), 100.0)
    temp_c = _safe_temp_c(parsed_spans.get("temp_c", 37.0), 37.0)
    duration_min = _safe_positive_float(parsed_spans.get("duration_min", 15.0), 15.0)
    mix_volume_ul = round(volume_ul * 0.5, 4) if volume_ul >= 2.0 else volume_ul

    protocol_id = str(explicit_id or "autolab_wire_protocol")
    return Protocol(
        protocol_id=protocol_id,
        steps=(
            TransferStep(
                source="reservoir:reagent",
                dest="plate1:A1",
                volume=volume_ul,
                unit="uL",
            ),
            MixStep(
                well="plate1:A1",
                volume=mix_volume_ul,
                cycles=5,
                unit="uL",
            ),
            IncubateStep(
                plate="plate1",
                temperature=temp_c,
                time=duration_min,
                time_unit="min",
            ),
            ReadStep(
                plate="plate1",
                mode="absorbance",
                wavelength=450,
            ),
        ),
    )


def _synthesize_artifact_twin(
    protocol: Protocol,
    attrib: dict[str, Any],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Synthesize a targeted error-detection twin conditioned on L4 attribution telemetry."""
    steps = list(protocol.steps)
    n = len(steps)
    correct_dicts = steps_to_dicts(steps)
    rng = random.Random(seed)

    raw_blamed = attrib.get("blamed_step_index", attrib.get("step_index"))
    blamed_idx = (
        max(0, min(n - 1, int(raw_blamed)))
        if isinstance(raw_blamed, int) and not isinstance(raw_blamed, bool)
        else None
    )
    family = str(attrib.get("fault_family") or attrib.get("error_type") or "").strip().lower()

    inc_idxs = [i for i, s in enumerate(steps) if isinstance(s, IncubateStep)]
    vol_idxs = [
        i for i, s in enumerate(steps) if isinstance(s, (TransferStep, MixStep, WashStep))
    ]

    if any(tok in family for tok in ("thermal", "temp")) and inc_idxs:
        t_idx = blamed_idx if blamed_idx in inc_idxs else inc_idxs[0]
        inc_step = steps[t_idx]
        assert isinstance(inc_step, IncubateStep)
        obs_temp = _safe_temp_c(attrib.get("observed_temp_c"), inc_step.temperature + 45.0)
        if obs_temp == inc_step.temperature:
            obs_temp = inc_step.temperature + 45.0
        mutated = list(steps)
        mutated[t_idx] = replace(inc_step, temperature=obs_temp)
        rec = TwinRecord(
            id=f"{protocol.protocol_id}:artifact:0",
            protocol_id=protocol.protocol_id,
            correct_steps=correct_dicts,
            corrupted_steps=steps_to_dicts(mutated),
            label="wrong_temperature",
            step_index=t_idx,
            original_value=inc_step.temperature,
            new_value=obs_temp,
            description=(
                f"Execution artifact thermal excursion at step {t_idx}: "
                f"observed {obs_temp:g} C vs target {inc_step.temperature:g} C."
            ),
        ).to_dict()
        rec["difficulty"] = classify_difficulty(rec)
        return rec

    if any(tok in family for tok in ("time", "timing", "duration", "delay")) and inc_idxs:
        t_idx = blamed_idx if blamed_idx in inc_idxs else inc_idxs[0]
        inc_step = steps[t_idx]
        assert isinstance(inc_step, IncubateStep)
        obs_dur = _safe_positive_float(
            attrib.get("observed_duration_min", attrib.get("observed_time")),
            max(0.5, round(inc_step.time * 0.1, 4)),
        )
        if obs_dur == inc_step.time:
            obs_dur = round(inc_step.time * 10.0, 4)
        mutated = list(steps)
        mutated[t_idx] = replace(inc_step, time=obs_dur)
        rec = TwinRecord(
            id=f"{protocol.protocol_id}:artifact:0",
            protocol_id=protocol.protocol_id,
            correct_steps=correct_dicts,
            corrupted_steps=steps_to_dicts(mutated),
            label="wrong_time",
            step_index=t_idx,
            original_value=f"{inc_step.time:g} {inc_step.time_unit}",
            new_value=f"{obs_dur:g} {inc_step.time_unit}",
            description=(
                f"Execution artifact timing drift at step {t_idx}: "
                f"observed {obs_dur:g} {inc_step.time_unit}."
            ),
        ).to_dict()
        rec["difficulty"] = classify_difficulty(rec)
        return rec

    if "unit" in family and vol_idxs:
        v_idx = blamed_idx if blamed_idx in vol_idxs else vol_idxs[0]
        v_step = steps[v_idx]
        assert isinstance(v_step, (TransferStep, MixStep, WashStep))
        new_u: Any = "mL" if v_step.unit == "uL" else "uL"
        mutated = list(steps)
        mutated[v_idx] = replace(v_step, unit=new_u)
        rec = TwinRecord(
            id=f"{protocol.protocol_id}:artifact:0",
            protocol_id=protocol.protocol_id,
            correct_steps=correct_dicts,
            corrupted_steps=steps_to_dicts(mutated),
            label="wrong_unit",
            step_index=v_idx,
            original_value=v_step.unit,
            new_value=new_u,
            description=f"Execution artifact unit mismatch at step {v_idx} ({new_u}).",
        ).to_dict()
        rec["difficulty"] = classify_difficulty(rec)
        return rec

    if any(tok in family for tok in ("dead_volume", "aspirat", "vol", "dispense")) and vol_idxs:
        v_idx = blamed_idx if blamed_idx in vol_idxs else vol_idxs[0]
        v_step = steps[v_idx]
        assert isinstance(v_step, (TransferStep, MixStep, WashStep))
        obs_vol = _safe_positive_float(
            attrib.get("observed_volume_ul", attrib.get("observed_volume")),
            round(v_step.volume * 0.5, 4) if v_step.volume > 2.0 else v_step.volume * 10.0,
        )
        if obs_vol == v_step.volume:
            obs_vol = round(v_step.volume * 0.5, 4)
        mutated = list(steps)
        mutated[v_idx] = replace(v_step, volume=obs_vol)
        rec = TwinRecord(
            id=f"{protocol.protocol_id}:artifact:0",
            protocol_id=protocol.protocol_id,
            correct_steps=correct_dicts,
            corrupted_steps=steps_to_dicts(mutated),
            label="wrong_volume",
            step_index=v_idx,
            original_value=v_step.volume,
            new_value=obs_vol,
            description=(
                f"Execution artifact volume deviation at step {v_idx}: "
                f"observed {obs_vol:g} {v_step.unit} vs target {v_step.volume:g} {v_step.unit}."
            ),
        ).to_dict()
        rec["difficulty"] = classify_difficulty(rec)
        return rec

    if any(tok in family for tok in ("skip", "missing", "drop", "unmixed")) and n >= 2:
        mix_idxs = [i for i, s in enumerate(steps) if isinstance(s, MixStep)]
        d_idx = blamed_idx if blamed_idx is not None else (mix_idxs[0] if mix_idxs else 0)
        mutated = list(steps)
        removed = mutated.pop(d_idx)
        rec = TwinRecord(
            id=f"{protocol.protocol_id}:artifact:0",
            protocol_id=protocol.protocol_id,
            correct_steps=correct_dicts,
            corrupted_steps=steps_to_dicts(mutated),
            label="missing_step",
            step_index=d_idx,
            original_value=correct_dicts[d_idx],
            new_value=None,
            description=f"Execution artifact omitted step {d_idx} ({removed.type}).",
        ).to_dict()
        rec["difficulty"] = classify_difficulty(rec)
        return rec

    swap_cands = _swap_candidates(steps)
    if swap_cands and any(tok in family for tok in ("premature", "order", "swap", "barrier")):
        matching = [
            pair for pair in swap_cands if blamed_idx in pair
        ] if blamed_idx is not None else []
        i, j = matching[0] if matching else swap_cands[0]
        mutated = list(steps)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        rec = TwinRecord(
            id=f"{protocol.protocol_id}:artifact:0",
            protocol_id=protocol.protocol_id,
            correct_steps=correct_dicts,
            corrupted_steps=steps_to_dicts(mutated),
            label="swap_order",
            step_index=i,
            original_value={"indices": [i, j], "steps": [correct_dicts[i], correct_dicts[j]]},
            new_value={"indices": [j, i], "steps": [correct_dicts[j], correct_dicts[i]]},
            description=f"Execution artifact swapped step {i} and step {j}.",
        ).to_dict()
        rec["difficulty"] = classify_difficulty(rec)
        return rec

    fallback_type = (
        "wrong_volume"
        if vol_idxs
        else ("wrong_temperature" if inc_idxs else "missing_step")
    )
    fallback_twin = inject_error(
        protocol,
        fallback_type,
        rng,
        record_id=f"{protocol.protocol_id}:artifact:0",
    ).to_dict()
    fallback_twin["difficulty"] = classify_difficulty(fallback_twin)
    return fallback_twin


def _build_contrastive_pairs(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for rec in records:
        dpo_rec = format_record(rec, fmt="dpo")
        mpo_rec = format_record(rec, fmt="mpo")
        gdpo_rec = format_record(rec, fmt="gdpo")
        rlvr_rec = format_record(rec, fmt="rlvr")
        pairs.append(
            {
                **dpo_rec,
                "ground_truth": rlvr_rec["ground_truth"],
                "dpo": dpo_rec,
                "mpo": mpo_rec,
                "gdpo": gdpo_rec,
                "rlvr": rlvr_rec,
            }
        )
    return pairs


def process_packet(
    packet: dict[str, Any],
    *,
    synthesize_from_execution_artifact: bool = False,
    synthesize_on_artifact: bool = False,
) -> dict[str, Any]:
    """Process an autolab-wire/v1 packet through Layer L3 twinstep DAG & twin synthesis."""
    out: dict[str, Any] = copy.deepcopy(packet)
    out.setdefault("schema_version", "autolab-wire/v1")
    out.setdefault("halted", False)
    out.setdefault("halt_reason", None)
    out.setdefault("trace", [])
    out.setdefault("bottom_reasoning_traces", [])

    attrib_raw = out.get("attribution")
    if not isinstance(attrib_raw, dict):
        attrib_raw = out.get("execution_telemetry")
    attrib = attrib_raw if isinstance(attrib_raw, dict) else {}
    attrib_outcome = str(attrib.get("outcome") or "").strip().upper()
    is_quarantine = (
        out.get("halt_reason") == "L4_ATTRIB_QUARANTINE"
        or attrib_outcome in ("QUARANTINE", "L4_ATTRIB_QUARANTINE")
    )
    synth_on_artifact = bool(
        synthesize_from_execution_artifact
        or synthesize_on_artifact
        or out.get("synthesize_from_execution_artifact")
        or out.get("synthesize_on_artifact")
    )

    was_upstream_halted = bool(out["halted"])
    if was_upstream_halted and (is_quarantine or not synth_on_artifact):
        return out

    protocol = _lower_wire_protocol(out)
    dag_edges = find_dependencies(protocol.steps)
    step_count = len(protocol.steps)
    dag_edges_list = [[int(u), int(v)] for u, v in dag_edges]

    seed = int(out.get("seed", 0))
    n_twins = max(4, int(out.get("n_twins", 6)))
    base_records = generate_dataset(
        [protocol],
        n_twins=n_twins,
        seed=seed,
        n_clean=0,
        n_ordering=0,
    )

    has_artifact_context = bool(attrib) or synth_on_artifact
    targeted_twin: dict[str, Any] | None = None
    if has_artifact_context and not is_quarantine:
        targeted_twin = _synthesize_artifact_twin(protocol, attrib, seed=seed)
        targeted_key = json.dumps(targeted_twin["corrupted_steps"], sort_keys=True)
        dedup_rest = [
            rec
            for rec in base_records
            if json.dumps(rec["corrupted_steps"], sort_keys=True) != targeted_key
        ]
        twin_records = [targeted_twin, *dedup_rest[: max(0, n_twins - 1)]]
    else:
        twin_records = base_records

    fault_types = list(dict.fromkeys(str(rec["label"]) for rec in twin_records))
    token_lengths_list = _estimate_token_lengths(twin_records)

    candidate_step_order = out.get("candidate_step_order")
    if not was_upstream_halted and candidate_step_order is not None:
        resolved_order = _resolve_candidate_order(candidate_step_order, protocol.steps)
        if resolved_order is None or not check_constraints(resolved_order, dag_edges):
            out["halted"] = True
            out["halt_reason"] = "L3_TWINSTEP_DAG_BLOCK"

    error_twins: dict[str, Any] = {
        "step_count": step_count,
        "dag_edges": dag_edges_list,
        "corrupted_twin_count": len(twin_records),
        "fault_types": fault_types,
        "token_lengths": token_lengths_list,
    }
    if has_artifact_context:
        if is_quarantine:
            error_twins["admit_to_memory"] = False
            error_twins["attribution_outcome"] = attrib_outcome or "L4_ATTRIB_QUARANTINE"
            error_twins["contrastive_pairs"] = []
        else:
            assert targeted_twin is not None
            error_twins["admit_to_memory"] = True
            error_twins["attribution_outcome"] = attrib_outcome or "EXECUTION_ARTIFACT"
            error_twins["blamed_step_index"] = targeted_twin["step_index"]
            error_twins["targeted_fault_type"] = targeted_twin["label"]
            error_twins["contrastive_pairs"] = _build_contrastive_pairs(twin_records)

    out["error_twins"] = error_twins

    status = "BLOCK" if out["halted"] else "PASS"
    if out["halted"] and not was_upstream_halted:
        summary = (
            f"Blocked candidate_step_order={candidate_step_order!r}: "
            f"violates causal DAG edges {dag_edges_list}."
        )
    else:
        summary = (
            f"Verified {step_count}-step DAG ({len(dag_edges_list)} edges) and synthesized "
            f"{len(twin_records)} seeded error twins ({len(token_lengths_list)} packed lengths)."
        )

    out["trace"].append(
        {
            "repo": REPO_ID,
            "layer": LAYER,
            "status": status,
            "summary": summary,
        }
    )
    out["bottom_reasoning_traces"].append(
        {
            "repo": REPO_ID,
            "invariant": "violated_dag_edge_rejected",
            "first_principles": (
                "Liquid-handling RAW, WAR, and plate-stage barrier edges form a strict partial "
                "order; any candidate step ordering that executes an optical read or downstream "
                "consumer before its preparation or incubation barrier is rejected."
            ),
        }
    )
    return out


__all__ = ["LAYER", "REPO_DIR", "REPO_ID", "process_packet"]

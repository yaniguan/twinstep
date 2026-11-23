"""Converters between twinstep Protocols, stepbot Opentrons dicts, and labfill batches."""

from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Any

from twinstep.dependencies import canonical_well, is_well_ref
from twinstep.schema import (
    IncubateStep,
    MixStep,
    Protocol,
    ReadStep,
    Step,
    TransferStep,
    WashStep,
    step_from_dict,
)

_IDENT_CLEAN_RE = re.compile(r"[^A-Za-z0-9_]")
_INCUBATE_COMMENT_RE = re.compile(
    r"^Incubate(?: on (?P<plate>[A-Za-z0-9_]+))? at (?P<temp>[-+]?\d+(?:\.\d+)?) C "
    r"for (?P<time>\d+(?:\.\d+)?) (?P<unit>min|s)$"
)
_WASH_COMMENT_RE = re.compile(
    r"^Wash (?P<kind>plate|well) (?P<target>[^:]+(?::[A-Za-z0-9]+)?): "
    r"(?P<cycles>\d+) cycles x (?P<vol>\d+(?:\.\d+)?) (?P<unit>uL|mL)$"
)
_READ_COMMENT_RE = re.compile(
    r"^Read (?P<plate>[A-Za-z0-9_]+) \((?P<mode>[A-Za-z0-9_-]+)"
    r"(?: at (?P<wl>\d+) nm)?\)$"
)


def _as_protocol(protocol: Protocol | Sequence[Step | dict[str, Any]]) -> Protocol:
    if isinstance(protocol, Protocol):
        return protocol
    steps = tuple(step_from_dict(s) if isinstance(s, dict) else s for s in protocol)
    return Protocol(protocol_id="twinstep_protocol", steps=steps)


def _vol_to_ul(volume: float, unit: str) -> float:
    return volume * 1000.0 if unit == "mL" else volume


def _sanitize_ident(raw: str, prefix: str = "lw") -> str:
    cleaned = _IDENT_CLEAN_RE.sub("_", raw.strip())
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"{prefix}_{cleaned}" if cleaned else prefix
    if cleaned in {"protocol", "metadata", "liquids", "pipette_left", "pipette_right"}:
        cleaned = f"{prefix}_{cleaned}"
    return cleaned


def to_stepbot_dict(
    protocol: Protocol | Sequence[Step | dict[str, Any]],
    *,
    plate_labware: str = "corning_96_wellplate_360ul_flat",
    reservoir_labware: str = "nest_12_reservoir_15ml",
    pipette_model: str = "p300_single_gen2",
    tip_rack_labware: str = "opentrons_96_tiprack_300ul",
    reservoir_volume_ul: float = 10000.0,
    new_tip: str = "always",
) -> dict[str, Any]:
    """Export a ``twinstep`` Protocol into a ``stepbot`` YAML/JSON protocol mapping."""
    proto = _as_protocol(protocol)
    plates: dict[str, str] = {}
    reagents: dict[str, str] = {}

    def map_loc(loc: str) -> str:
        norm = canonical_well(loc)
        if not is_well_ref(loc):
            if norm not in reagents:
                col = len(reagents) + 1
                if col > 12:
                    raise ValueError("more than 12 external reagents; exceeds 12-well reservoir")
                reagents[norm] = f"A{col}"
            return f"reservoir:{reagents[norm]}"
        raw_plate, well_coord = norm.split(":", 1)
        if raw_plate not in plates:
            plates[raw_plate] = _sanitize_ident(raw_plate, "plate")
        return f"{plates[raw_plate]}:{well_coord}"

    steps_out: list[dict[str, Any]] = []
    for step in proto.steps:
        if isinstance(step, TransferStep):
            vol_ul = _vol_to_ul(step.volume, step.unit)
            steps_out.append(
                {
                    "transfer": {
                        "src": map_loc(step.source),
                        "dst": map_loc(step.dest),
                        "volume": vol_ul,
                        "new_tip": new_tip,
                    }
                }
            )
        elif isinstance(step, MixStep):
            vol_ul = _vol_to_ul(step.volume, step.unit)
            steps_out.append(
                {
                    "mix": {
                        "wells": map_loc(step.well),
                        "volume": vol_ul,
                        "reps": step.cycles,
                        "new_tip": new_tip,
                    }
                }
            )
        elif isinstance(step, IncubateStep):
            plate_part = ""
            if step.plate is not None:
                p_alias = plates.setdefault(step.plate, _sanitize_ident(step.plate, "plate"))
                plate_part = f" on {p_alias}"
            steps_out.append(
                {
                    "comment": (
                        f"Incubate{plate_part} at {step.temperature:g} C "
                        f"for {step.time:g} {step.time_unit}"
                    )
                }
            )
            if step.time_unit == "min":
                steps_out.append({"delay": {"minutes": step.time}})
            else:
                steps_out.append({"delay": {"seconds": step.time}})
        elif isinstance(step, WashStep):
            if step.well is not None:
                mapped_w = map_loc(step.well)
                steps_out.append(
                    {
                        "comment": (
                            f"Wash well {mapped_w}: "
                            f"{step.cycles} cycles x {step.volume:g} {step.unit}"
                        )
                    }
                )
            else:
                assert step.plate is not None
                p_alias = plates.setdefault(step.plate, _sanitize_ident(step.plate, "plate"))
                steps_out.append(
                    {
                        "comment": (
                            f"Wash plate {p_alias}: "
                            f"{step.cycles} cycles x {step.volume:g} {step.unit}"
                        )
                    }
                )
        elif isinstance(step, ReadStep):
            p_alias = plates.setdefault(step.plate, _sanitize_ident(step.plate, "plate"))
            wl_part = f" at {step.wavelength} nm" if step.wavelength is not None else ""
            steps_out.append({"comment": f"Read {p_alias} ({step.mode}{wl_part})"})

    if not plates:
        plates["plate1"] = "plate1"

    deck: dict[str, dict[str, Any]] = {}
    slot = 1
    for alias in plates.values():
        deck[alias] = {"labware": plate_labware, "slot": slot}
        slot += 1
    if reagents:
        deck["reservoir"] = {"labware": reservoir_labware, "slot": slot}
        slot += 1
    deck["tips"] = {"labware": tip_rack_labware, "slot": slot}

    liquids: dict[str, dict[str, Any]] = {}
    for norm_reagent, r_well in reagents.items():
        liq_key = _IDENT_CLEAN_RE.sub("_", norm_reagent)
        liquids[liq_key] = {
            "wells": f"reservoir:{r_well}",
            "volume": reservoir_volume_ul,
            "description": norm_reagent,
        }

    out: dict[str, Any] = {
        "metadata": {"name": proto.protocol_id},
        "deck": deck,
        "pipettes": {"left": {"model": pipette_model, "tip_racks": ["tips"]}},
        "steps": steps_out,
    }
    if liquids:
        out["liquids"] = liquids
    return out


def from_stepbot_dict(stepbot_data: dict[str, Any]) -> Protocol:
    """Reconstruct a ``twinstep`` Protocol from a ``to_stepbot_dict`` mapping."""
    meta = stepbot_data.get("metadata") or {}
    proto_id = str(meta.get("name") or "stepbot_protocol")

    well_to_reagent: dict[str, str] = {}
    for liq_name, entry in (stepbot_data.get("liquids") or {}).items():
        if isinstance(entry, dict) and isinstance(entry.get("wells"), str):
            desc = str(entry.get("description") or f"reservoir:{liq_name}")
            well_to_reagent[entry["wells"]] = desc

    def unmap_loc(ref: str) -> str:
        return well_to_reagent.get(ref, ref)

    raw_steps = stepbot_data.get("steps") or []
    out_steps: list[Step] = []
    idx = 0
    while idx < len(raw_steps):
        item = raw_steps[idx]
        if "transfer" in item:
            tp = item["transfer"]
            out_steps.append(
                TransferStep(
                    source=unmap_loc(str(tp["src"])),
                    dest=unmap_loc(str(tp["dst"])),
                    volume=float(tp["volume"]),
                    unit="uL",
                )
            )
            idx += 1
        elif "mix" in item:
            mp = item["mix"]
            well_raw = mp.get("wells", mp.get("well"))
            out_steps.append(
                MixStep(
                    well=unmap_loc(str(well_raw)),
                    volume=float(mp["volume"]),
                    cycles=int(mp["reps"]),
                    unit="uL",
                )
            )
            idx += 1
        elif "comment" in item:
            text = str(item["comment"])
            inc_m = _INCUBATE_COMMENT_RE.match(text)
            if inc_m is not None:
                out_steps.append(
                    IncubateStep(
                        plate=inc_m.group("plate"),
                        temperature=float(inc_m.group("temp")),
                        time=float(inc_m.group("time")),
                        time_unit=inc_m.group("unit"),  # type: ignore[arg-type]
                    )
                )
                if idx + 1 < len(raw_steps) and "delay" in raw_steps[idx + 1]:
                    idx += 2
                else:
                    idx += 1
                continue
            wash_m = _WASH_COMMENT_RE.match(text)
            if wash_m is not None:
                kind = wash_m.group("kind")
                target = wash_m.group("target")
                out_steps.append(
                    WashStep(
                        plate=target if kind == "plate" else None,
                        well=target if kind == "well" else None,
                        cycles=int(wash_m.group("cycles")),
                        volume=float(wash_m.group("vol")),
                        unit=wash_m.group("unit"),  # type: ignore[arg-type]
                    )
                )
                idx += 1
                continue
            read_m = _READ_COMMENT_RE.match(text)
            if read_m is not None:
                wl_str = read_m.group("wl")
                out_steps.append(
                    ReadStep(
                        plate=read_m.group("plate"),
                        mode=read_m.group("mode"),
                        wavelength=int(wl_str) if wl_str is not None else None,
                    )
                )
                idx += 1
                continue
            idx += 1
        else:
            idx += 1

    return Protocol(protocol_id=proto_id, steps=tuple(out_steps))


def to_labfill_dict(
    protocol: Protocol | Sequence[Step | dict[str, Any]],
    *,
    well_capacity_ul: float = 360.0,
    reservoir_volume_ul: float = 10000.0,
    reservoir_capacity_ul: float = 15000.0,
) -> dict[str, Any]:
    """Export a ``twinstep`` Protocol into ``labfill`` (LabStateBatch, ProtocolBatch) dicts.

    Volumes are quantized to integer ``0.1 uL`` units (``volume_01ul``), and every
    ``TransferStep`` / ``MixStep`` is wrapped with explicit ``pick_up_tip`` and ``drop_tip``
    operations for tensorized tip-state checking in ``labfill``.
    """
    proto = _as_protocol(protocol)
    well_names: list[str] = []
    well_index: dict[str, int] = {}

    def get_well_idx(loc: str) -> int:
        norm = canonical_well(loc)
        if norm not in well_index:
            well_index[norm] = len(well_names)
            well_names.append(norm)
        return well_index[norm]

    ops: list[dict[str, Any]] = []
    for step_idx, step in enumerate(proto.steps):
        if isinstance(step, TransferStep):
            src_i = get_well_idx(step.source)
            dst_i = get_well_idx(step.dest)
            vol_01ul = round(_vol_to_ul(step.volume, step.unit) * 10.0)
            ops.extend(
                [
                    {"op": "pick_up_tip", "source_step": step_idx},
                    {
                        "op": "transfer",
                        "src_well": src_i,
                        "dst_well": dst_i,
                        "src_name": step.source,
                        "dst_name": step.dest,
                        "volume_01ul": vol_01ul,
                        "source_step": step_idx,
                    },
                    {"op": "drop_tip", "source_step": step_idx},
                ]
            )
        elif isinstance(step, MixStep):
            w_i = get_well_idx(step.well)
            vol_01ul = round(_vol_to_ul(step.volume, step.unit) * 10.0)
            ops.extend(
                [
                    {"op": "pick_up_tip", "source_step": step_idx},
                    {
                        "op": "mix",
                        "well": w_i,
                        "well_name": step.well,
                        "volume_01ul": vol_01ul,
                        "cycles": step.cycles,
                        "source_step": step_idx,
                    },
                    {"op": "drop_tip", "source_step": step_idx},
                ]
            )
        elif isinstance(step, IncubateStep):
            duration_s = round(step.time * 60.0 if step.time_unit == "min" else step.time)
            ops.append(
                {
                    "op": "incubate",
                    "plate": step.plate,
                    "temperature_c": step.temperature,
                    "duration_s": duration_s,
                    "time": step.time,
                    "time_unit": step.time_unit,
                    "source_step": step_idx,
                }
            )
        elif isinstance(step, WashStep):
            vol_01ul = round(_vol_to_ul(step.volume, step.unit) * 10.0)
            wash_w_i: int | None = get_well_idx(step.well) if step.well is not None else None
            ops.append(
                {
                    "op": "wash",
                    "plate": step.plate,
                    "well": wash_w_i,
                    "well_name": step.well,
                    "volume_01ul": vol_01ul,
                    "cycles": step.cycles,
                    "source_step": step_idx,
                }
            )
        elif isinstance(step, ReadStep):
            ops.append(
                {
                    "op": "read",
                    "plate": step.plate,
                    "mode": step.mode,
                    "wavelength": step.wavelength,
                    "source_step": step_idx,
                }
            )

    init_vol_01ul: list[int] = []
    cap_01ul: list[int] = []
    for name in well_names:
        if is_well_ref(name):
            init_vol_01ul.append(0)
            cap_01ul.append(round(well_capacity_ul * 10.0))
        else:
            init_vol_01ul.append(round(reservoir_volume_ul * 10.0))
            cap_01ul.append(round(reservoir_capacity_ul * 10.0))

    return {
        "protocol_id": proto.protocol_id,
        "unit_scale": "0.1_uL",
        "lab_state": {
            "wells": well_names,
            "initial_volume_01ul": init_vol_01ul,
            "capacity_01ul": cap_01ul,
            "tips_available": 96,
        },
        "protocol_batch": {
            "ops": ops,
        },
    }


def from_labfill_dict(labfill_data: dict[str, Any]) -> Protocol:
    """Reconstruct a ``twinstep`` Protocol from a ``to_labfill_dict`` representation."""
    proto_id = str(labfill_data.get("protocol_id") or "labfill_protocol")
    wells = list(labfill_data["lab_state"]["wells"])
    ops = labfill_data["protocol_batch"]["ops"]

    steps: list[Step] = []
    for op_entry in ops:
        op = op_entry.get("op")
        if op in ("pick_up_tip", "drop_tip"):
            continue
        if op == "transfer":
            src = str(op_entry.get("src_name") or wells[int(op_entry["src_well"])])
            dst = str(op_entry.get("dst_name") or wells[int(op_entry["dst_well"])])
            vol_ul = float(op_entry["volume_01ul"]) / 10.0
            steps.append(TransferStep(source=src, dest=dst, volume=vol_ul, unit="uL"))
        elif op == "mix":
            well = str(op_entry.get("well_name") or wells[int(op_entry["well"])])
            vol_ul = float(op_entry["volume_01ul"]) / 10.0
            steps.append(
                MixStep(well=well, volume=vol_ul, cycles=int(op_entry["cycles"]), unit="uL")
            )
        elif op == "incubate":
            t_unit = str(op_entry.get("time_unit", "s"))
            t_val = float(op_entry.get("time", op_entry["duration_s"]))
            steps.append(
                IncubateStep(
                    plate=op_entry.get("plate"),
                    temperature=float(op_entry["temperature_c"]),
                    time=t_val,
                    time_unit=t_unit,  # type: ignore[arg-type]
                )
            )
        elif op == "wash":
            vol_ul = float(op_entry["volume_01ul"]) / 10.0
            w_idx = op_entry.get("well")
            w_name = op_entry.get("well_name") or (wells[int(w_idx)] if w_idx is not None else None)
            steps.append(
                WashStep(
                    plate=op_entry.get("plate"),
                    well=w_name,
                    volume=vol_ul,
                    cycles=int(op_entry["cycles"]),
                    unit="uL",
                )
            )
        elif op == "read":
            steps.append(
                ReadStep(
                    plate=str(op_entry["plate"]),
                    mode=str(op_entry["mode"]),
                    wavelength=op_entry.get("wavelength"),
                )
            )
    return Protocol(protocol_id=proto_id, steps=tuple(steps))

"""Typed, deterministic error injectors and dataset generator for protocol steps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
import json
import random
from typing import Any
import warnings

from twinstep.baseline import _assign_plate_cycles, predict_error, predict_record
from twinstep.dependencies import (
    check_constraints,
    find_dependencies,
    infer_default_plate,
    is_well_ref,
)
from twinstep.schema import (
    IncubateStep,
    MixStep,
    Protocol,
    ReadStep,
    Step,
    TransferStep,
    WashStep,
    step_from_dict,
    steps_to_dicts,
)

ERROR_TYPES: tuple[str, ...] = (
    "swap_order",
    "wrong_volume",
    "wrong_unit",
    "missing_step",
    "wrong_temperature",
    "wrong_time",
)

DIFFICULTY_FILTERS: tuple[str, ...] = (
    "all",
    "easy",
    "medium",
    "hard",
    "hard_saboteur",
)

# Draws per twin before an error type is treated as out of distinct outputs.
MAX_DRAW_ATTEMPTS = 50


@dataclass(frozen=True)
class TwinRecord:
    """One labeled error-detection example (corrupted twin or clean negative).

    ``step_index`` coordinate conventions:
    - Single-step value mutations (``wrong_volume``, ``wrong_unit``,
      ``wrong_temperature``, ``wrong_time``): 0-based index of the mutated step in
      ``corrupted_steps`` (``0 .. len(corrupted_steps) - 1``).
    - ``swap_order``: smaller swapped index ``i`` from ``original_value["indices"] == [i, j]``.
    - ``missing_step``: 0-based insertion index in ``0 .. len(corrupted_steps)`` where the
      deleted step originally stood (``corrupted_steps[:idx] + [removed] + corrupted_steps[idx:]``
      reconstructs ``correct_steps``).
    - ``clean``: ``None``.
    """

    id: str
    protocol_id: str
    correct_steps: list[dict[str, Any]]
    corrupted_steps: list[dict[str, Any]]
    label: str
    step_index: int | None
    original_value: Any
    new_value: Any
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": "error_detection",
            "protocol_id": self.protocol_id,
            "correct_steps": self.correct_steps,
            "corrupted_steps": self.corrupted_steps,
            "label": self.label,
            "error_type": self.label,
            "step_index": self.step_index,
            "original_value": self.original_value,
            "new_value": self.new_value,
            "description": self.description,
        }


@dataclass(frozen=True)
class MultiTwinRecord:
    """Compound multi-fault twin containing ``k`` non-overlapping step mutations."""

    id: str
    protocol_id: str
    correct_steps: list[dict[str, Any]]
    corrupted_steps: list[dict[str, Any]]
    labels: list[str]
    step_indices: list[int]
    descriptions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": "multi_error_detection",
            "protocol_id": self.protocol_id,
            "k_errors": len(self.labels),
            "correct_steps": self.correct_steps,
            "corrupted_steps": self.corrupted_steps,
            "labels": list(self.labels),
            "step_indices": list(self.step_indices),
            "descriptions": list(self.descriptions),
        }


@dataclass(frozen=True)
class OrderingRecord:
    """One step-ordering example with shuffled steps, gold order, and constraints."""

    id: str
    protocol_id: str
    correct_steps: list[dict[str, Any]]
    shuffled_steps: list[dict[str, Any]]
    gold_order: list[int]
    constraints: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": "ordering",
            "protocol_id": self.protocol_id,
            "correct_steps": self.correct_steps,
            "shuffled_steps": self.shuffled_steps,
            "gold_order": self.gold_order,
            "constraints": self.constraints,
        }


@lru_cache(maxsize=256)
def _cached_swap_candidates(steps_tuple: tuple[Step, ...]) -> tuple[tuple[int, int], ...]:
    constraints = find_dependencies(steps_tuple)
    n = len(steps_tuple)
    # Compute transitive closure so any ancestor/descendant dependency swap is eligible.
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for u, v in constraints:
        adj[u].add(v)
    reachable: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n - 1, -1, -1):
        for nxt in adj[i]:
            reachable[i].add(nxt)
            reachable[i].update(reachable[nxt])

    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in sorted(reachable[i]):
            if steps_tuple[i] != steps_tuple[j]:
                pairs.append((i, j))
    return tuple(pairs)


def _swap_candidates(steps: Sequence[Step]) -> list[tuple[int, int]]:
    return list(_cached_swap_candidates(tuple(steps)))


def _volume_step_indices(steps: Sequence[Step]) -> list[int]:
    return [
        idx
        for idx, step in enumerate(steps)
        if isinstance(step, (TransferStep, MixStep, WashStep))
    ]


def _incubate_indices(steps: Sequence[Step]) -> list[int]:
    return [idx for idx, step in enumerate(steps) if isinstance(step, IncubateStep)]


def applicable_error_types(steps: Sequence[Step]) -> list[str]:
    """Return the subset of ERROR_TYPES that can be injected into the given steps."""
    applicable: list[str] = []
    if _swap_candidates(steps):
        applicable.append("swap_order")
    if _volume_step_indices(steps):
        applicable.append("wrong_volume")
        applicable.append("wrong_unit")
    if len(steps) >= 2:
        applicable.append("missing_step")
    if _incubate_indices(steps):
        applicable.append("wrong_temperature")
        applicable.append("wrong_time")
    return applicable


def _clean_float(value: float) -> float:
    rounded = round(value, 6)
    return float(int(rounded)) if rounded == int(rounded) else rounded


def _mutate_volume(target: TransferStep | MixStep | WashStep, rng: random.Random) -> float:
    vol_candidates = [
        _clean_float(target.volume * f)
        for f in (10.0, 0.1)
        if _clean_float(target.volume * f) > 0 and _clean_float(target.volume * f) != target.volume
    ]
    if vol_candidates:
        return rng.choice(vol_candidates)
    raw_scaled = target.volume * 10.0
    if raw_scaled > 0 and raw_scaled != target.volume:
        return raw_scaled
    return 10.0 if target.volume != 10.0 else 100.0


def _mutate_temperature(target: IncubateStep, rng: random.Random) -> float:
    candidates = [
        _clean_float(target.temperature + delta)
        for delta in (-40.0, -25.0, 25.0, 60.0, 85.0)
        if _clean_float(target.temperature + delta) != target.temperature
        and _clean_float(target.temperature + delta) > -273.15
    ]
    if not candidates:
        fallback = _clean_float(max(target.temperature + 25.0, 0.0))
        if fallback == target.temperature or fallback <= -273.15:
            fallback = 37.0 if target.temperature != 37.0 else 60.0
        candidates = [fallback]
    return rng.choice(candidates)


def _mutate_time(target: IncubateStep, rng: random.Random) -> tuple[float, str]:
    mode = rng.choice(("unit", "scale"))
    if mode == "unit":
        new_unit = "s" if target.time_unit == "min" else "min"
        return target.time, new_unit
    scale_candidates = [
        _clean_float(target.time * f)
        for f in (10.0, 0.1)
        if _clean_float(target.time * f) > 0 and _clean_float(target.time * f) != target.time
    ]
    if scale_candidates:
        return rng.choice(scale_candidates), target.time_unit
    raw_up = target.time * 10.0
    if raw_up > 0 and raw_up != target.time:
        return raw_up, target.time_unit
    return max(target.time, 1.0), ("s" if target.time_unit == "min" else "min")


def inject_error(
    protocol: Protocol,
    error_type: str,
    rng: random.Random,
    record_id: str = "",
) -> TwinRecord:
    """Inject a single typed error into a protocol and return a verified TwinRecord."""
    if error_type not in ERROR_TYPES:
        raise ValueError(f"unknown error_type: {error_type!r}")

    steps = list(protocol.steps)
    correct_dicts = steps_to_dicts(steps)
    mutated = list(steps)

    if error_type == "swap_order":
        candidates = _swap_candidates(steps)
        if not candidates:
            raise ValueError(f"protocol {protocol.protocol_id!r} has no swappable dependencies")
        i, j = rng.choice(candidates)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        step_index = i
        original_value: Any = {"indices": [i, j], "steps": [correct_dicts[i], correct_dicts[j]]}
        new_value: Any = {"indices": [j, i], "steps": [correct_dicts[j], correct_dicts[i]]}
        description = (
            f"Swapped step {i} ({steps[i].type}) and step {j} ({steps[j].type}), "
            "violating dependency order."
        )

    elif error_type == "wrong_volume":
        indices = _volume_step_indices(steps)
        if not indices:
            raise ValueError(f"protocol {protocol.protocol_id!r} has no volume steps")
        step_index = rng.choice(indices)
        target = steps[step_index]
        assert isinstance(target, (TransferStep, MixStep, WashStep))
        new_vol = _mutate_volume(target, rng)
        mutated[step_index] = replace(target, volume=new_vol)
        original_value = target.volume
        new_value = new_vol
        description = (
            f"Scaled step {step_index} ({target.type}) volume from "
            f"{target.volume:g} {target.unit} to {new_vol:g} {target.unit}."
        )

    elif error_type == "wrong_unit":
        indices = _volume_step_indices(steps)
        if not indices:
            raise ValueError(f"protocol {protocol.protocol_id!r} has no unit-bearing steps")
        step_index = rng.choice(indices)
        target = steps[step_index]
        assert isinstance(target, (TransferStep, MixStep, WashStep))
        new_unit: Any = "mL" if target.unit == "uL" else "uL"
        mutated[step_index] = replace(target, unit=new_unit)
        original_value = target.unit
        new_value = new_unit
        description = (
            f"Changed step {step_index} ({target.type}) volume unit from "
            f"{target.unit} to {new_unit} with unchanged number {target.volume:g}."
        )

    elif error_type == "missing_step":
        if len(steps) < 2:
            raise ValueError(f"protocol {protocol.protocol_id!r} needs at least 2 steps")
        step_index = rng.randrange(len(steps))
        removed = steps[step_index]
        mutated.pop(step_index)
        original_value = correct_dicts[step_index]
        new_value = None
        description = f"Dropped step {step_index} ({removed.type}) from protocol."

    elif error_type == "wrong_temperature":
        indices = _incubate_indices(steps)
        if not indices:
            raise ValueError(f"protocol {protocol.protocol_id!r} has no incubate steps")
        step_index = rng.choice(indices)
        target = steps[step_index]
        assert isinstance(target, IncubateStep)
        new_temp = _mutate_temperature(target, rng)
        mutated[step_index] = replace(target, temperature=new_temp)
        original_value = target.temperature
        new_value = new_temp
        description = (
            f"Changed step {step_index} (incubate) temperature from "
            f"{target.temperature:g} C to {new_temp:g} C."
        )

    elif error_type == "wrong_time":
        indices = _incubate_indices(steps)
        if not indices:
            raise ValueError(f"protocol {protocol.protocol_id!r} has no incubate steps")
        step_index = rng.choice(indices)
        target = steps[step_index]
        assert isinstance(target, IncubateStep)
        new_time, new_t_unit = _mutate_time(target, rng)
        mutated[step_index] = replace(target, time=new_time, time_unit=new_t_unit)  # type: ignore[arg-type]
        original_value = f"{target.time:g} {target.time_unit}"
        new_value = f"{new_time:g} {new_t_unit}"
        description = (
            f"Changed step {step_index} (incubate) duration from "
            f"{original_value} to {new_value}."
        )

    corrupted_dicts = steps_to_dicts(mutated)
    if corrupted_dicts == correct_dicts:
        raise RuntimeError(f"injector {error_type!r} produced identical protocol")

    return TwinRecord(
        id=record_id or f"{protocol.protocol_id}:{error_type}",
        protocol_id=protocol.protocol_id,
        correct_steps=correct_dicts,
        corrupted_steps=corrupted_dicts,
        label=error_type,
        step_index=step_index,
        original_value=original_value,
        new_value=new_value,
        description=description,
    )


def _swap_survives_deletion(
    steps: Sequence[Step],
    swap_pairs: Sequence[tuple[int, int]],
    deleted_idx: int | None,
) -> bool:
    surviving_orig = [idx for idx in range(len(steps)) if idx != deleted_idx]
    surviving_steps = [steps[idx] for idx in surviving_orig]
    surviving_deps = find_dependencies(surviving_steps)
    if not surviving_deps:
        return False
    surviving_swap_set = set(_swap_candidates(surviving_steps))
    perm = list(range(len(surviving_steps)))
    for i, j in swap_pairs:
        if i == deleted_idx or j == deleted_idx:
            return False
        fi = i - 1 if (deleted_idx is not None and i > deleted_idx) else i
        fj = j - 1 if (deleted_idx is not None and j > deleted_idx) else j
        if (fi, fj) not in surviving_swap_set:
            return False
        single_perm = list(range(len(surviving_steps)))
        single_perm[fi], single_perm[fj] = single_perm[fj], single_perm[fi]
        if check_constraints(single_perm, surviving_deps):
            return False
        perm[fi], perm[fj] = perm[fj], perm[fi]
    return not check_constraints(perm, surviving_deps)


def inject_k_errors(
    protocol: Protocol,
    k: int,
    rng: random.Random,
    record_id: str = "",
) -> MultiTwinRecord:
    """Inject ``k`` non-overlapping step mutations into ``protocol``."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    steps = list(protocol.steps)
    n = len(steps)
    if n < k:
        raise ValueError(
            f"protocol {protocol.protocol_id!r} has {n} steps, fewer than k={k} requested errors"
        )

    correct_dicts = steps_to_dicts(steps)
    swap_all = _swap_candidates(steps)

    for _ in range(MAX_DRAW_ATTEMPTS * 2):
        mutated = list(steps)
        used: set[int] = set()
        primary_orig: set[int] = set()
        swap_pairs: list[tuple[int, int]] = []
        raw_edits: list[dict[str, Any]] = []
        deleted_idx: int | None = None
        ok = True

        for fault_num in range(k):
            forbidden_primary = {deleted_idx + 1} if deleted_idx is not None else set()
            choices: list[str] = []
            avail_vol = [
                i
                for i in _volume_step_indices(steps)
                if i not in used and i not in forbidden_primary
            ]
            avail_inc = [
                i
                for i in _incubate_indices(steps)
                if i not in used and i not in forbidden_primary
            ]
            avail_swaps = [
                (a, b)
                for a, b in swap_all
                if a not in used
                and b not in used
                and a not in forbidden_primary
                and _swap_survives_deletion(steps, [*swap_pairs, (a, b)], deleted_idx)
            ]
            avail_del = [
                i
                for i in range(n)
                if i not in used
                and (i + 1) not in primary_orig
                and (not swap_pairs or _swap_survives_deletion(steps, swap_pairs, i))
            ]

            if avail_vol:
                choices.extend(("wrong_volume", "wrong_unit"))
            if avail_inc:
                choices.extend(("wrong_temperature", "wrong_time"))
            if avail_swaps and (n - len(used) - 2 >= (k - fault_num - 1)):
                choices.append("swap_order")
            if deleted_idx is None and avail_del and n - 1 >= max(1, k):
                choices.append("missing_step")

            if not choices:
                ok = False
                break

            err_type = rng.choice(choices)
            if err_type == "swap_order":
                i, j = rng.choice(avail_swaps)
                mutated[i], mutated[j] = mutated[j], mutated[i]
                used.update((i, j))
                primary_orig.add(i)
                swap_pairs.append((i, j))
                raw_edits.append({"type": err_type, "i": i, "j": j})
            elif err_type == "wrong_volume":
                idx = rng.choice(avail_vol)
                t_vol = steps[idx]
                assert isinstance(t_vol, (TransferStep, MixStep, WashStep))
                nv = _mutate_volume(t_vol, rng)
                mutated[idx] = replace(t_vol, volume=nv)
                used.add(idx)
                primary_orig.add(idx)
                raw_edits.append({"type": err_type, "idx": idx, "nv": nv})
            elif err_type == "wrong_unit":
                idx = rng.choice(avail_vol)
                t_unit = steps[idx]
                assert isinstance(t_unit, (TransferStep, MixStep, WashStep))
                nu: Any = "mL" if t_unit.unit == "uL" else "uL"
                mutated[idx] = replace(t_unit, unit=nu)
                used.add(idx)
                primary_orig.add(idx)
                raw_edits.append({"type": err_type, "idx": idx, "nu": nu})
            elif err_type == "wrong_temperature":
                idx = rng.choice(avail_inc)
                t_inc = steps[idx]
                assert isinstance(t_inc, IncubateStep)
                nt = _mutate_temperature(t_inc, rng)
                mutated[idx] = replace(t_inc, temperature=nt)
                used.add(idx)
                primary_orig.add(idx)
                raw_edits.append({"type": err_type, "idx": idx, "nt": nt})
            elif err_type == "wrong_time":
                idx = rng.choice(avail_inc)
                t_inc = steps[idx]
                assert isinstance(t_inc, IncubateStep)
                ntm, ntu = _mutate_time(t_inc, rng)
                mutated[idx] = replace(t_inc, time=ntm, time_unit=ntu)  # type: ignore[arg-type]
                used.add(idx)
                primary_orig.add(idx)
                raw_edits.append({"type": err_type, "idx": idx, "ntm": ntm, "ntu": ntu})
            elif err_type == "missing_step":
                idx = rng.choice(avail_del)
                deleted_idx = idx
                used.add(idx)
                raw_edits.append({"type": err_type, "idx": idx})

        if not ok or len(raw_edits) != k:
            continue

        if swap_pairs and not _swap_survives_deletion(steps, swap_pairs, deleted_idx):
            continue

        surviving_ref_steps = [
            steps[idx] for idx in range(n) if idx != deleted_idx
        ]
        if deleted_idx is not None:
            mutated.pop(deleted_idx)

        def _post_idx(orig: int) -> int:
            return orig - 1 if (deleted_idx is not None and orig > deleted_idx) else orig

        records_meta: list[tuple[int, str, str]] = []
        valid_invariants = True
        for edit in raw_edits:
            etype = str(edit["type"])
            if etype == "swap_order":
                oi = int(edit["i"])
                oj = int(edit["j"])
                fi = _post_idx(oi)
                fj = _post_idx(oj)
                if mutated[fi] == mutated[fj]:
                    valid_invariants = False
                    break
                desc = (
                    f"Swapped step {fi} ({steps[oi].type}) and "
                    f"step {fj} ({steps[oj].type}), violating dependency order."
                )
                records_meta.append((fi, etype, desc))
            elif etype == "wrong_volume":
                oidx = int(edit["idx"])
                fpos = _post_idx(oidx)
                ref_step = surviving_ref_steps[fpos]
                mut_step = mutated[fpos]
                assert isinstance(ref_step, (TransferStep, MixStep, WashStep))
                assert isinstance(mut_step, (TransferStep, MixStep, WashStep))
                if mut_step.volume == ref_step.volume:
                    valid_invariants = False
                    break
                nv_val = float(edit["nv"])
                desc = (
                    f"Scaled step {fpos} ({ref_step.type}) volume "
                    f"to {nv_val:g} {ref_step.unit}."
                )
                records_meta.append((fpos, etype, desc))
            elif etype == "wrong_unit":
                oidx = int(edit["idx"])
                fpos = _post_idx(oidx)
                ref_step = surviving_ref_steps[fpos]
                mut_step = mutated[fpos]
                assert isinstance(ref_step, (TransferStep, MixStep, WashStep))
                assert isinstance(mut_step, (TransferStep, MixStep, WashStep))
                if mut_step.unit == ref_step.unit:
                    valid_invariants = False
                    break
                nu_val = str(edit["nu"])
                desc = f"Changed step {fpos} ({ref_step.type}) unit to {nu_val}."
                records_meta.append((fpos, etype, desc))
            elif etype == "wrong_temperature":
                oidx = int(edit["idx"])
                fpos = _post_idx(oidx)
                ref_step = surviving_ref_steps[fpos]
                mut_step = mutated[fpos]
                assert isinstance(ref_step, IncubateStep)
                assert isinstance(mut_step, IncubateStep)
                if mut_step.temperature == ref_step.temperature:
                    valid_invariants = False
                    break
                nt_val = float(edit["nt"])
                desc = f"Changed step {fpos} (incubate) temperature to {nt_val:g} C."
                records_meta.append((fpos, etype, desc))
            elif etype == "wrong_time":
                oidx = int(edit["idx"])
                fpos = _post_idx(oidx)
                ref_step = surviving_ref_steps[fpos]
                mut_step = mutated[fpos]
                assert isinstance(ref_step, IncubateStep)
                assert isinstance(mut_step, IncubateStep)
                if (mut_step.time, mut_step.time_unit) == (
                    ref_step.time,
                    ref_step.time_unit,
                ):
                    valid_invariants = False
                    break
                ntm_val = float(edit["ntm"])
                ntu_val = str(edit["ntu"])
                desc = (
                    f"Changed step {fpos} (incubate) duration "
                    f"to {ntm_val:g} {ntu_val}."
                )
                records_meta.append((fpos, etype, desc))
            elif etype == "missing_step":
                d_idx = int(edit["idx"])
                if len(mutated) != n - 1:
                    valid_invariants = False
                    break
                desc = f"Dropped step {d_idx} ({steps[d_idx].type}) from protocol."
                records_meta.append((d_idx, etype, desc))

        if not valid_invariants:
            continue

        if len({item[0] for item in records_meta}) != k:
            continue

        corrupted_dicts = steps_to_dicts(mutated)
        if corrupted_dicts == correct_dicts:
            continue

        records_meta.sort(key=lambda item: item[0])
        return MultiTwinRecord(
            id=record_id or f"{protocol.protocol_id}:multi:{k}",
            protocol_id=protocol.protocol_id,
            correct_steps=correct_dicts,
            corrupted_steps=corrupted_dicts,
            labels=[item[1] for item in records_meta],
            step_indices=[item[0] for item in records_meta],
            descriptions=[item[2] for item in records_meta],
        )

    raise ValueError(
        f"cannot inject {k} non-overlapping errors into protocol {protocol.protocol_id!r}"
    )


def make_clean_record(protocol: Protocol, record_id: str = "") -> TwinRecord:
    """Create a clean negative record matching the input protocol."""
    correct_dicts = protocol.to_dicts()
    return TwinRecord(
        id=record_id or f"{protocol.protocol_id}:clean",
        protocol_id=protocol.protocol_id,
        correct_steps=correct_dicts,
        corrupted_steps=[dict(step) for step in correct_dicts],
        label="clean",
        step_index=None,
        original_value=None,
        new_value=None,
        description="No error injected (clean negative).",
    )


def _has_distinct_steps(steps: Sequence[Step]) -> bool:
    return len(set(steps)) >= 2


def _canonicalize_identical_step_perm(steps: Sequence[Step], perm: list[int]) -> list[int]:
    """Ensure identical steps at original positions a < b map to shuffled indices s_a < s_b."""
    n = len(perm)
    inv = [0] * n
    for s_idx, o_idx in enumerate(perm):
        inv[o_idx] = s_idx
    groups: dict[Step, list[int]] = {}
    for o_idx, step in enumerate(steps):
        groups.setdefault(step, []).append(o_idx)
    canon_perm = list(perm)
    for orig_group in groups.values():
        if len(orig_group) > 1:
            shuffled_slots = sorted(inv[o] for o in orig_group)
            for o_idx, s_slot in zip(orig_group, shuffled_slots):
                canon_perm[s_slot] = o_idx
    return canon_perm


def make_ordering_record(
    protocol: Protocol,
    rng: random.Random,
    record_id: str = "",
    *,
    precomputed_constraints: Sequence[tuple[int, int]] | None = None,
    precomputed_correct_dicts: list[dict[str, Any]] | None = None,
) -> OrderingRecord:
    """Create an ordering record with shuffled steps, gold_order, and dependency constraints."""
    steps = list(protocol.steps)
    n = len(steps)
    if n < 2:
        raise ValueError(f"protocol {protocol.protocol_id!r} needs at least 2 steps for ordering")
    if not _has_distinct_steps(steps):
        raise ValueError(
            f"protocol {protocol.protocol_id!r} has only identical repeated steps; "
            "no distinct step permutation exists"
        )

    gold_constraints = (
        list(precomputed_constraints)
        if precomputed_constraints is not None
        else find_dependencies(steps)
    )
    correct_dicts = (
        [dict(d) for d in precomputed_correct_dicts]
        if precomputed_correct_dicts is not None
        else steps_to_dicts(steps)
    )

    perm = list(range(n))
    for _ in range(50):
        rng.shuffle(perm)
        perm = _canonicalize_identical_step_perm(steps, perm)
        shuffled_dicts = [correct_dicts[i] for i in perm]
        if shuffled_dicts == correct_dicts:
            continue
        inv_perm = [0] * n
        for shuffled_idx, orig_idx in enumerate(perm):
            inv_perm[orig_idx] = shuffled_idx
        shuffled_constraints = [[inv_perm[a], inv_perm[b]] for a, b in gold_constraints]
        if not shuffled_constraints or not check_constraints(
            list(range(n)), [(a, b) for a, b in shuffled_constraints]
        ):
            break
    else:
        # Fallback deterministic reversal; if palindromic, pick a guaranteed non-identity swap.
        perm = _canonicalize_identical_step_perm(steps, list(reversed(range(n))))
        shuffled_dicts = [correct_dicts[i] for i in perm]
        if shuffled_dicts == correct_dicts:
            best_perm: list[int] | None = None
            for i in range(n):
                for j in range(i + 1, n):
                    if correct_dicts[i] == correct_dicts[j]:
                        continue
                    cand = list(range(n))
                    cand[i], cand[j] = cand[j], cand[i]
                    cand = _canonicalize_identical_step_perm(steps, cand)
                    cand_inv = [0] * n
                    for s_idx, o_idx in enumerate(cand):
                        cand_inv[o_idx] = s_idx
                    cand_constraints = [(cand_inv[a], cand_inv[b]) for a, b in gold_constraints]
                    if best_perm is None:
                        best_perm = cand
                    if cand_constraints and not check_constraints(list(range(n)), cand_constraints):
                        best_perm = cand
                        break
                if best_perm is not None and gold_constraints:
                    test_inv = [0] * n
                    for s_idx, o_idx in enumerate(best_perm):
                        test_inv[o_idx] = s_idx
                    if not check_constraints(
                        list(range(n)), [(test_inv[a], test_inv[b]) for a, b in gold_constraints]
                    ):
                        break
            assert best_perm is not None
            perm = best_perm
            shuffled_dicts = [correct_dicts[i] for i in perm]
        inv_perm = [0] * n
        for shuffled_idx, orig_idx in enumerate(perm):
            inv_perm[orig_idx] = shuffled_idx
        shuffled_constraints = [[inv_perm[a], inv_perm[b]] for a, b in gold_constraints]

    # gold_order[k] = index in shuffled_steps corresponding to correct_steps[k].
    gold_order = list(inv_perm)
    return OrderingRecord(
        id=record_id or f"{protocol.protocol_id}:order",
        protocol_id=protocol.protocol_id,
        correct_steps=correct_dicts,
        shuffled_steps=shuffled_dicts,
        gold_order=gold_order,
        constraints=shuffled_constraints,
    )


def _steps_key(steps: Sequence[dict[str, Any]]) -> str:
    return json.dumps(list(steps), sort_keys=True)


def classify_difficulty(record: dict[str, Any]) -> str:
    """Classify a generated twin or ordering record into calibrated difficulty tiers."""
    if record.get("is_saboteur") or record.get("difficulty") == "hard_saboteur":
        return "hard_saboteur"

    task = str(record.get("task", "error_detection"))
    if task == "multi_error_detection":
        labels = record.get("labels", [])
        return "hard" if len(labels) >= 2 else "easy"

    if task == "ordering":
        raw_shuffled = record.get("shuffled_steps", [])
        parsed_steps: list[Step | None] = []
        for item in raw_shuffled:
            if isinstance(item, dict):
                try:
                    parsed_steps.append(step_from_dict(item))
                except ValueError:
                    parsed_steps.append(None)
        default_plate = infer_default_plate(parsed_steps)
        p_cycles = _assign_plate_cycles(parsed_steps, default_plate)
        if any(any(c > 0 for c in cyc_list) for cyc_list in p_cycles.values()):
            return "hard"
        pred_ord = predict_record(record)
        if pred_ord.get("pred_order") == record.get("gold_order"):
            return "easy"
        return "medium"

    gold_label = str(record.get("label", "clean"))
    if gold_label == "clean":
        return "easy"
    pred = predict_record(record)
    if pred.get("label") == gold_label and pred.get("step_index") == record.get("step_index"):
        return "easy"
    return "medium"


def filter_records_by_difficulty(
    records: Sequence[dict[str, Any]],
    difficulty: str = "all",
) -> list[dict[str, Any]]:
    """Filter annotated dataset records by difficulty tier."""
    if difficulty not in DIFFICULTY_FILTERS:
        raise ValueError(
            f"unsupported difficulty {difficulty!r}; expected one of {DIFFICULTY_FILTERS}"
        )
    if difficulty == "all":
        return list(records)
    out: list[dict[str, Any]] = []
    for rec in records:
        tier = str(rec.get("difficulty") or classify_difficulty(rec))
        if difficulty == "hard" and tier in ("hard", "hard_saboteur"):
            out.append(rec)
        elif tier == difficulty:
            out.append(rec)
    return out


def _lower_saboteur_steps(raw_steps: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    lowered: list[dict[str, Any]] = []
    for st in raw_steps:
        op = str(st.get("op", ""))
        if op == "transfer":
            src_raw = str(st.get("src", "BUFFER"))
            src = (
                "reservoir:buffer"
                if src_raw.upper() == "BUFFER"
                else f"plate1:{src_raw.upper()}"
            )
            dst = f"plate1:{str(st.get('dst', 'A1')).upper()}"
            vol = max(1.0, float(st.get("vol_uL", 50.0)))
            lowered.append(
                TransferStep(source=src, dest=dst, volume=vol, unit="uL").to_dict()
            )
        elif op == "mix":
            dst = f"plate1:{str(st.get('dst', 'A1')).upper()}"
            vol = max(1.0, float(st.get("vol_uL", 50.0)))
            cycles = max(1, int(st.get("cycles", 3)))
            lowered.append(
                MixStep(well=dst, volume=vol, cycles=cycles, unit="uL").to_dict()
            )
        elif op == "incubate":
            temp = float(st.get("temp_c", 37.0))
            dur = max(5.0, float(st.get("duration_s", 60.0)))
            lowered.append(
                IncubateStep(
                    plate="plate1",
                    temperature=temp,
                    time=dur,
                    time_unit="s",
                ).to_dict()
            )
        elif op == "read":
            lowered.append(
                ReadStep(plate="plate1", mode="absorbance", wavelength=450).to_dict()
            )
    return lowered


def saboteur_pairs_to_records(
    pairs: Sequence[dict[str, Any]],
    *,
    id_prefix: str = "saboteur",
) -> list[dict[str, Any]]:
    """Adapt curated ``saboteur_evolver`` hard-negative pairs into ``twinstep`` records."""
    adapted: list[dict[str, Any]] = []
    for idx, pair in enumerate(pairs):
        clean_raw = pair.get("clean_steps")
        sab_raw = pair.get("sabotaged_steps")
        if not isinstance(clean_raw, (list, tuple)) or not isinstance(sab_raw, (list, tuple)):
            continue
        clean_dicts = _lower_saboteur_steps(clean_raw)
        sab_dicts = _lower_saboteur_steps(sab_raw)
        if not clean_dicts or not sab_dicts:
            continue

        violated_rules = [str(r) for r in pair.get("violated_lean_rules", [])]
        evaded_gates = [str(g) for g in pair.get("evaded_gates", [])]
        mut_ops = [str(m) for m in pair.get("mutation_ops", [])]
        proto_name = str(pair.get("protocol_name", "saboteur_protocol"))

        label = "wrong_volume"
        step_index = 0
        orig_val: Any = None
        new_val: Any = None

        if sab_dicts == clean_dicts:
            # Stateful seal-inversion mutation: reflect subtle unsealed thermal excursion in-range.
            inc_positions = [
                pos for pos, item in enumerate(sab_dicts) if item.get("type") == "incubate"
            ]
            target_pos = inc_positions[0] if inc_positions else 0
            mut_item = dict(sab_dicts[target_pos])
            if mut_item.get("type") == "incubate":
                orig_temp = float(mut_item["temperature"])
                new_temp = round(orig_temp - 6.5 if orig_temp >= 85.0 else 86.0, 2)
                mut_item["temperature"] = new_temp
                sab_dicts[target_pos] = mut_item
                label = "wrong_temperature"
                step_index = target_pos
                orig_val = orig_temp
                new_val = new_temp
            else:
                orig_vol = float(mut_item.get("volume", 50.0))
                mut_item["volume"] = round(min(280.0, orig_vol + 45.0), 2)
                sab_dicts[target_pos] = mut_item
                label = "wrong_volume"
                step_index = target_pos
                orig_val = orig_vol
                new_val = mut_item["volume"]
        else:
            diff_indices = [
                pos
                for pos in range(min(len(clean_dicts), len(sab_dicts)))
                if clean_dicts[pos] != sab_dicts[pos]
            ]
            if len(sab_dicts) < len(clean_dicts):
                label = "missing_step"
                step_index = diff_indices[0] if diff_indices else len(sab_dicts)
                orig_val = clean_dicts[step_index]
                new_val = None
            else:
                step_index = diff_indices[0] if diff_indices else 0
                c_item = clean_dicts[step_index]
                s_item = sab_dicts[step_index]
                if (
                    c_item.get("type") != s_item.get("type")
                    or c_item.get("dest") != s_item.get("dest")
                    or c_item.get("well") != s_item.get("well")
                ):
                    label = "swap_order"
                    second_idx = diff_indices[-1] if len(diff_indices) >= 2 else step_index + 1
                    second_idx = min(len(clean_dicts) - 1, second_idx)
                    orig_val = {
                        "indices": [step_index, second_idx],
                        "steps": [clean_dicts[step_index], clean_dicts[second_idx]],
                    }
                    new_val = {
                        "indices": [second_idx, step_index],
                        "steps": [sab_dicts[step_index], sab_dicts[second_idx]],
                    }
                elif c_item.get("volume") != s_item.get("volume"):
                    label = "wrong_volume"
                    orig_val = c_item.get("volume")
                    new_val = s_item.get("volume")
                else:
                    label = "wrong_temperature"
                    orig_val = c_item.get("temperature")
                    new_val = s_item.get("temperature")

        rec = TwinRecord(
            id=f"{proto_name}:{id_prefix}:{idx}",
            protocol_id=proto_name,
            correct_steps=clean_dicts,
            corrupted_steps=sab_dicts,
            label=label,
            step_index=step_index,
            original_value=orig_val,
            new_value=new_val,
            description=(
                f"Evolutionary saboteur ({', '.join(mut_ops) or 'stealth'}): "
                f"evades {', '.join(evaded_gates)} while violating {', '.join(violated_rules)}."
            ),
        ).to_dict()
        rec["difficulty"] = "hard_saboteur"
        rec["is_saboteur"] = True
        rec["mutation_ops"] = mut_ops
        rec["evaded_gates"] = evaded_gates
        rec["violated_lean_rules"] = violated_rules
        rec["clean_saboteur_steps"] = [dict(x) for x in clean_raw]
        rec["sabotaged_saboteur_steps"] = [dict(x) for x in sab_raw]
        adapted.append(rec)
    return adapted


def synthesize_saboteur_records(
    protocols: Sequence[Protocol],
    *,
    seed: int = 0,
    max_per_protocol: int = 2,
) -> list[dict[str, Any]]:
    """Synthesize stealth hard-negative saboteur twins that evade stateless bounds gates."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []

    for proto in protocols:
        steps = list(proto.steps)
        correct_dicts = steps_to_dicts(steps)
        emitted = 0

        # 1. Dead-volume / stoichiometry shave inside [15, 280] uL (evades stateless baseline).
        transfer_idxs = [
            i
            for i, s in enumerate(steps)
            if isinstance(s, TransferStep)
            and s.unit == "uL"
            and (is_well_ref(s.source) or is_well_ref(s.dest))
        ]
        if transfer_idxs and emitted < max_per_protocol:
            t_idx = transfer_idxs[rng.randrange(len(transfer_idxs))]
            t_step = steps[t_idx]
            assert isinstance(t_step, TransferStep)
            cand_vol = _clean_float(
                110.0
                if abs(t_step.volume - 110.0) >= 5.0
                else min(260.0, max(15.0, t_step.volume * 0.75))
            )
            if cand_vol != t_step.volume:
                sab_steps = list(steps)
                sab_steps[t_idx] = replace(t_step, volume=cand_vol)
                sab_dicts = steps_to_dicts(sab_steps)
                if predict_error(sab_dicts)["label"] == "clean":
                    rec = TwinRecord(
                        id=f"{proto.protocol_id}:saboteur:{emitted}",
                        protocol_id=proto.protocol_id,
                        correct_steps=correct_dicts,
                        corrupted_steps=sab_dicts,
                        label="wrong_volume",
                        step_index=t_idx,
                        original_value=t_step.volume,
                        new_value=cand_vol,
                        description=(
                            f"Stealth dead-volume saboteur at step {t_idx}: "
                            f"shifted transfer from {t_step.volume:g} uL to {cand_vol:g} uL."
                        ),
                    ).to_dict()
                    rec["difficulty"] = "hard_saboteur"
                    rec["is_saboteur"] = True
                    rec["mutation_ops"] = ["shave_dead_volume"]
                    rec["evaded_gates"] = ["gen0_scalar_volume_gate", "gen1_bitmask_gate"]
                    rec["violated_lean_rules"] = ["recoverable_dead_volume_reserve"]
                    out.append(rec)
                    emitted += 1

        # 2. High-temperature unsealed thermal drift inside [4, 95] C (evades stateless baseline).
        inc_idxs = _incubate_indices(steps)
        if inc_idxs and emitted < max_per_protocol:
            i_idx = inc_idxs[rng.randrange(len(inc_idxs))]
            i_step = steps[i_idx]
            assert isinstance(i_step, IncubateStep)
            cand_temp = 88.0 if abs(i_step.temperature - 88.0) >= 2.0 else 82.0
            sab_steps = list(steps)
            sab_steps[i_idx] = replace(i_step, temperature=cand_temp)
            sab_dicts = steps_to_dicts(sab_steps)
            if predict_error(sab_dicts)["label"] == "clean":
                rec = TwinRecord(
                    id=f"{proto.protocol_id}:saboteur:{emitted}",
                    protocol_id=proto.protocol_id,
                    correct_steps=correct_dicts,
                    corrupted_steps=sab_dicts,
                    label="wrong_temperature",
                    step_index=i_idx,
                    original_value=i_step.temperature,
                    new_value=cand_temp,
                    description=(
                        f"Stealth high-temperature saboteur at step {i_idx}: "
                        f"unsealed {cand_temp:g} C thermal excursion."
                    ),
                ).to_dict()
                rec["difficulty"] = "hard_saboteur"
                rec["is_saboteur"] = True
                rec["mutation_ops"] = ["drop_high_temp_seal"]
                rec["evaded_gates"] = ["gen0_scalar_volume_gate", "gen1_bitmask_gate"]
                rec["violated_lean_rules"] = ["high_temp_seal_required"]
                out.append(rec)
                emitted += 1

    # 3. Also load curated HardNegativeTwinPair evolutions when community_extensions is present.
    import importlib.util
    from pathlib import Path

    evolver_path = (
        Path(__file__).resolve().parents[2] / "community_extensions" / "saboteur_evolver.py"
    )
    if evolver_path.is_file():
        spec = importlib.util.spec_from_file_location("_twinstep_saboteur_evolver", evolver_path)
        if spec is not None and spec.loader is not None:
            mod: Any = importlib.util.module_from_spec(spec)
            import sys

            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            evo_report = mod.evolve_saboteurs(generations=4, pop_size=10, seed=seed)
            curated = evo_report.get("curated_hard_negatives", [])
            if isinstance(curated, list):
                out.extend(saboteur_pairs_to_records(curated[:4], id_prefix="evolved"))

    return out


def generate_dataset(
    protocols: Sequence[Protocol],
    n_twins: int = 5,
    seed: int = 0,
    n_clean: int = 1,
    n_ordering: int | None = None,
    *,
    k_errors: int = 1,
    difficulty: str = "all",
    include_saboteurs: bool = False,
) -> list[dict[str, Any]]:
    """Generate deterministic error-detection twins, clean negatives, and ordering records.

    Twins within a protocol have distinct ``corrupted_steps``. An error type that yields
    no new output after ``MAX_DRAW_ATTEMPTS`` draws is dropped with a warning, so fewer than
    ``n_twins`` records may be returned for small protocols. Any ``n_clean > 0`` emits a
    single clean record per protocol, since extra clean copies would be identical.
    """
    if n_twins < 0 or n_clean < 0:
        raise ValueError("n_twins and n_clean must be non-negative")
    if k_errors < 1:
        raise ValueError("k_errors must be >= 1")
    if difficulty not in DIFFICULTY_FILTERS:
        raise ValueError(
            f"unsupported difficulty {difficulty!r}; expected one of {DIFFICULTY_FILTERS}"
        )
    ordering_count = n_twins if n_ordering is None else n_ordering
    if ordering_count < 0:
        raise ValueError("n_ordering must be non-negative")

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []

    for protocol in protocols:
        gold_constraints = find_dependencies(protocol.steps)
        correct_dicts = protocol.to_dicts()
        applicable = applicable_error_types(protocol.steps)
        if n_twins > 0 and not applicable:
            raise ValueError(f"no error injectors apply to protocol {protocol.protocol_id!r}")

        seen: set[str] = set()
        if k_errors == 1:
            active = list(applicable)
            twins: list[TwinRecord] = []
            while len(twins) < n_twins and active:
                batch = list(active)
                rng.shuffle(batch)
                for err_type in batch:
                    if len(twins) >= n_twins:
                        break
                    for _ in range(MAX_DRAW_ATTEMPTS):
                        twin = inject_error(
                            protocol,
                            err_type,
                            rng,
                            record_id=f"{protocol.protocol_id}:twin:{len(twins)}",
                        )
                        key = _steps_key(twin.corrupted_steps)
                        if key not in seen:
                            seen.add(key)
                            twins.append(twin)
                            break
                    else:
                        active.remove(err_type)
                        warnings.warn(
                            f"protocol {protocol.protocol_id!r}: no new distinct {err_type!r} twin "
                            f"after {MAX_DRAW_ATTEMPTS} draws; no longer drawing this type",
                            stacklevel=2,
                        )
            if len(twins) < n_twins:
                warnings.warn(
                    f"protocol {protocol.protocol_id!r}: only {len(twins)} distinct twins "
                    f"available (requested {n_twins})",
                    stacklevel=2,
                )
            for twin in twins:
                t_dict = twin.to_dict()
                t_dict["difficulty"] = classify_difficulty(t_dict)
                records.append(t_dict)
        else:
            multi_twins: list[MultiTwinRecord] = []
            while len(multi_twins) < n_twins:
                found_new = False
                for _ in range(MAX_DRAW_ATTEMPTS):
                    mtwin = inject_k_errors(
                        protocol,
                        k_errors,
                        rng,
                        record_id=f"{protocol.protocol_id}:multi:{len(multi_twins)}",
                    )
                    key = _steps_key(mtwin.corrupted_steps)
                    if key not in seen:
                        seen.add(key)
                        multi_twins.append(mtwin)
                        found_new = True
                        break
                if not found_new:
                    break
            if len(multi_twins) < n_twins:
                warnings.warn(
                    f"protocol {protocol.protocol_id!r}: only {len(multi_twins)} distinct "
                    f"{k_errors}-error twins available (requested {n_twins})",
                    stacklevel=2,
                )
            for mtwin in multi_twins:
                m_dict = mtwin.to_dict()
                m_dict["difficulty"] = classify_difficulty(m_dict)
                records.append(m_dict)

        # Clean negatives are identical copies of the protocol, so emit at most one.
        if n_clean > 0:
            if k_errors == 1:
                clean = make_clean_record(
                    protocol,
                    record_id=f"{protocol.protocol_id}:clean:0",
                )
                c_dict = clean.to_dict()
                c_dict["difficulty"] = classify_difficulty(c_dict)
                records.append(c_dict)
            else:
                cm_dict = MultiTwinRecord(
                    id=f"{protocol.protocol_id}:clean:0",
                    protocol_id=protocol.protocol_id,
                    correct_steps=correct_dicts,
                    corrupted_steps=[dict(s) for s in correct_dicts],
                    labels=[],
                    step_indices=[],
                    descriptions=["No error injected (clean negative)."],
                ).to_dict()
                cm_dict["difficulty"] = classify_difficulty(cm_dict)
                records.append(cm_dict)

        if ordering_count > 0 and len(protocol.steps) >= 2:
            if not _has_distinct_steps(protocol.steps):
                warnings.warn(
                    f"protocol {protocol.protocol_id!r}: skipping ordering records because "
                    "all steps are identical",
                    stacklevel=2,
                )
            else:
                for idx in range(ordering_count):
                    order_rec = make_ordering_record(
                        protocol,
                        rng,
                        record_id=f"{protocol.protocol_id}:order:{idx}",
                        precomputed_constraints=gold_constraints,
                        precomputed_correct_dicts=correct_dicts,
                    )
                    o_dict = order_rec.to_dict()
                    o_dict["difficulty"] = classify_difficulty(o_dict)
                    records.append(o_dict)

    if include_saboteurs:
        records.extend(synthesize_saboteur_records(protocols, seed=seed))

    return filter_records_by_difficulty(records, difficulty=difficulty)

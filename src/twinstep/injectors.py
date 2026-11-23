"""Typed, deterministic error injectors and dataset generator for protocol steps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
import json
import random
from typing import Any
import warnings

from twinstep.dependencies import check_constraints, find_dependencies
from twinstep.schema import (
    IncubateStep,
    MixStep,
    Protocol,
    Step,
    TransferStep,
    WashStep,
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

    for _ in range(MAX_DRAW_ATTEMPTS):
        mutated = list(steps)
        used: set[int] = set()
        records_meta: list[tuple[int, str, str]] = []
        deleted_idx: int | None = None
        ok = True

        for fault_num in range(k):
            choices: list[str] = []
            avail_vol = [i for i in _volume_step_indices(steps) if i not in used]
            avail_inc = [i for i in _incubate_indices(steps) if i not in used]
            avail_swaps = [(a, b) for a, b in swap_all if a not in used and b not in used]
            avail_del = [i for i in range(n) if i not in used]

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
                desc = f"Swapped step {i} ({steps[i].type}) and step {j} ({steps[j].type})."
                records_meta.append((i, err_type, desc))
            elif err_type == "wrong_volume":
                idx = rng.choice(avail_vol)
                t_vol = steps[idx]
                assert isinstance(t_vol, (TransferStep, MixStep, WashStep))
                nv = _mutate_volume(t_vol, rng)
                mutated[idx] = replace(t_vol, volume=nv)
                used.add(idx)
                desc = f"Scaled step {idx} ({t_vol.type}) volume to {nv:g} {t_vol.unit}."
                records_meta.append((idx, err_type, desc))
            elif err_type == "wrong_unit":
                idx = rng.choice(avail_vol)
                t_unit = steps[idx]
                assert isinstance(t_unit, (TransferStep, MixStep, WashStep))
                nu: Any = "mL" if t_unit.unit == "uL" else "uL"
                mutated[idx] = replace(t_unit, unit=nu)
                used.add(idx)
                desc = f"Changed step {idx} ({t_unit.type}) unit to {nu}."
                records_meta.append((idx, err_type, desc))
            elif err_type == "wrong_temperature":
                idx = rng.choice(avail_inc)
                t_inc = steps[idx]
                assert isinstance(t_inc, IncubateStep)
                nt = _mutate_temperature(t_inc, rng)
                mutated[idx] = replace(t_inc, temperature=nt)
                used.add(idx)
                desc = f"Changed step {idx} (incubate) temperature to {nt:g} C."
                records_meta.append((idx, err_type, desc))
            elif err_type == "wrong_time":
                idx = rng.choice(avail_inc)
                t_inc = steps[idx]
                assert isinstance(t_inc, IncubateStep)
                ntm, ntu = _mutate_time(t_inc, rng)
                mutated[idx] = replace(t_inc, time=ntm, time_unit=ntu)  # type: ignore[arg-type]
                used.add(idx)
                desc = f"Changed step {idx} (incubate) duration to {ntm:g} {ntu}."
                records_meta.append((idx, err_type, desc))
            elif err_type == "missing_step":
                idx = rng.choice(avail_del)
                deleted_idx = idx
                used.add(idx)
                desc = f"Dropped step {idx} ({steps[idx].type}) from protocol."
                records_meta.append((idx, err_type, desc))

        if not ok or len(records_meta) != k:
            continue

        if deleted_idx is not None:
            mutated.pop(deleted_idx)
            adjusted: list[tuple[int, str, str]] = []
            for pos, lbl, dsc in records_meta:
                if lbl != "missing_step" and pos > deleted_idx:
                    adjusted.append((pos - 1, lbl, dsc))
                else:
                    adjusted.append((pos, lbl, dsc))
            records_meta = adjusted

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


def generate_dataset(
    protocols: Sequence[Protocol],
    n_twins: int = 5,
    seed: int = 0,
    n_clean: int = 1,
    n_ordering: int | None = None,
    *,
    k_errors: int = 1,
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
            records.extend(twin.to_dict() for twin in twins)
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
            records.extend(mtwin.to_dict() for mtwin in multi_twins)

        # Clean negatives are identical copies of the protocol, so emit at most one.
        if n_clean > 0:
            if k_errors == 1:
                clean = make_clean_record(
                    protocol,
                    record_id=f"{protocol.protocol_id}:clean:0",
                )
                records.append(clean.to_dict())
            else:
                records.append(
                    MultiTwinRecord(
                        id=f"{protocol.protocol_id}:clean:0",
                        protocol_id=protocol.protocol_id,
                        correct_steps=correct_dicts,
                        corrupted_steps=[dict(s) for s in correct_dicts],
                        labels=[],
                        step_indices=[],
                        descriptions=["No error injected (clean negative)."],
                    ).to_dict()
                )

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
                    records.append(order_rec.to_dict())

    return records

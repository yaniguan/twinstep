"""Typed, deterministic error injectors and dataset generator for protocol steps."""

from __future__ import annotations

from dataclasses import dataclass, replace
import random
from typing import Any, Sequence

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


@dataclass(frozen=True)
class TwinRecord:
    """One labeled error-detection example (corrupted twin or clean negative)."""

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


def _swap_candidates(steps: Sequence[Step]) -> list[tuple[int, int]]:
    constraints = find_dependencies(steps)
    n = len(steps)
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
            if steps[i] != steps[j]:
                pairs.append((i, j))
    return pairs


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
        factor = rng.choice((10.0, 0.1))
        new_vol = _clean_float(target.volume * factor)
        if new_vol == target.volume or new_vol <= 0:
            new_vol = _clean_float(target.volume * 10.0)
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
        new_unit = "mL" if target.unit == "uL" else "uL"
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
        candidates = [
            _clean_float(target.temperature + delta)
            for delta in (-40.0, -25.0, 25.0, 60.0, 85.0)
            if _clean_float(target.temperature + delta) != target.temperature
            and target.temperature + delta > -273.15
        ]
        new_temp = rng.choice(candidates)
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
        mode = rng.choice(("unit", "scale"))
        if mode == "unit":
            new_time = target.time
            new_unit = "s" if target.time_unit == "min" else "min"
        else:
            factor = 10.0 if target.time < 60 else 0.1
            new_time = _clean_float(target.time * factor)
            new_unit = target.time_unit
        mutated[step_index] = replace(target, time=new_time, time_unit=new_unit)
        original_value = f"{target.time:g} {target.time_unit}"
        new_value = f"{new_time:g} {new_unit}"
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


def make_ordering_record(
    protocol: Protocol,
    rng: random.Random,
    record_id: str = "",
) -> OrderingRecord:
    """Create an ordering record with shuffled steps, gold_order, and dependency constraints."""
    steps = list(protocol.steps)
    n = len(steps)
    if n < 2:
        raise ValueError(f"protocol {protocol.protocol_id!r} needs at least 2 steps for ordering")

    gold_constraints = find_dependencies(steps)
    correct_dicts = steps_to_dicts(steps)

    perm = list(range(n))
    for _ in range(50):
        rng.shuffle(perm)
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
        # Fallback deterministic reversal when random shuffles match identity or constraints.
        perm = list(reversed(range(n)))
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


def generate_dataset(
    protocols: Sequence[Protocol],
    n_twins: int = 5,
    seed: int = 0,
    n_clean: int = 1,
    n_ordering: int | None = None,
) -> list[dict[str, Any]]:
    """Generate deterministic error-detection twins, clean negatives, and ordering records."""
    if n_twins < 0 or n_clean < 0:
        raise ValueError("n_twins and n_clean must be non-negative")
    ordering_count = n_twins if n_ordering is None else n_ordering
    if ordering_count < 0:
        raise ValueError("n_ordering must be non-negative")

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []

    for protocol in protocols:
        applicable = applicable_error_types(protocol.steps)
        if n_twins > 0 and not applicable:
            raise ValueError(f"no error injectors apply to protocol {protocol.protocol_id!r}")

        # Cycle through shuffled applicable error types so all supported classes are represented.
        chosen_types: list[str] = []
        while len(chosen_types) < n_twins:
            batch = list(applicable)
            rng.shuffle(batch)
            chosen_types.extend(batch)
        chosen_types = chosen_types[:n_twins]

        for idx, err_type in enumerate(chosen_types):
            twin = inject_error(
                protocol,
                err_type,
                rng,
                record_id=f"{protocol.protocol_id}:twin:{idx}",
            )
            records.append(twin.to_dict())

        for idx in range(n_clean):
            clean = make_clean_record(
                protocol,
                record_id=f"{protocol.protocol_id}:clean:{idx}",
            )
            records.append(clean.to_dict())

        if len(protocol.steps) >= 2:
            for idx in range(ordering_count):
                order_rec = make_ordering_record(
                    protocol,
                    rng,
                    record_id=f"{protocol.protocol_id}:order:{idx}",
                )
                records.append(order_rec.to_dict())

    return records

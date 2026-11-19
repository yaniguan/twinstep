"""Dependency graph and ordering constraint utilities for liquid-handling steps."""

from __future__ import annotations

import re
from typing import Sequence

from twinstep.schema import IncubateStep, MixStep, ReadStep, Step, TransferStep, WashStep

_EXTERNAL_PREFIXES = ("reservoir", "stock", "trough", "waste")
_BARE_WELL_RE = re.compile(r"^[A-Ha-h](?:[1-9]|1[0-2])$")


def is_well_ref(location: str) -> bool:
    """Return True when a location names an assay well rather than an external reservoir."""
    prefix = location.split(":", 1)[0].lower()
    if prefix in _EXTERNAL_PREFIXES:
        return False
    return ":" in location or bool(_BARE_WELL_RE.match(location))


def plate_of(location: str) -> str | None:
    """Extract the plate identifier associated with a well or plate location."""
    if ":" in location:
        prefix = location.split(":", 1)[0]
        if prefix.lower() in _EXTERNAL_PREFIXES:
            return None
        return prefix
    if _BARE_WELL_RE.match(location):
        return "plate1"
    if location.lower() in _EXTERNAL_PREFIXES:
        return None
    return location


def step_plates(step: Step) -> set[str]:
    """Return plate identifiers touched by a step (empty if global)."""
    if isinstance(step, TransferStep):
        plates = {p for p in (plate_of(step.source), plate_of(step.dest)) if p is not None}
        return plates
    if isinstance(step, MixStep):
        p = plate_of(step.well)
        return {p} if p is not None else set()
    if isinstance(step, WashStep):
        plates: set[str] = set()
        if step.plate is not None:
            plates.add(step.plate)
        if step.well is not None:
            p = plate_of(step.well)
            if p is not None:
                plates.add(p)
        return plates
    if isinstance(step, IncubateStep):
        return {step.plate} if step.plate is not None else set()
    if isinstance(step, ReadStep):
        return {step.plate}
    return set()


def _plates_overlap(a: Step, b: Step) -> bool:
    pa = step_plates(a)
    pb = step_plates(b)
    if isinstance(a, IncubateStep) and a.plate is None:
        return bool(pb) or isinstance(b, IncubateStep)
    if isinstance(b, IncubateStep) and b.plate is None:
        return bool(pa) or isinstance(a, IncubateStep)
    return bool(pa & pb)


def _is_barrier(step: Step) -> bool:
    if isinstance(step, (IncubateStep, ReadStep)):
        return True
    if isinstance(step, WashStep) and step.plate is not None and step.well is None:
        return True
    return False


def _has_direct_dependency(earlier: Step, later: Step) -> bool:
    # Well-level liquid flow dependencies.
    if isinstance(earlier, TransferStep):
        if isinstance(later, TransferStep) and earlier.dest == later.source:
            return True
        if isinstance(later, MixStep) and earlier.dest == later.well:
            return True
        if isinstance(later, WashStep) and later.well is not None and earlier.dest == later.well:
            return True

    if isinstance(earlier, MixStep):
        if isinstance(later, TransferStep) and earlier.well == later.source:
            return True
        if isinstance(later, WashStep) and later.well is not None and earlier.well == later.well:
            return True

    if isinstance(earlier, WashStep) and earlier.well is not None:
        if isinstance(later, TransferStep) and earlier.well == later.dest:
            return True
        if isinstance(later, MixStep) and earlier.well == later.well:
            return True

    # Plate-level stage barriers (incubate, read, whole-plate wash).
    if (_is_barrier(earlier) or _is_barrier(later)) and _plates_overlap(earlier, later):
        return True

    return False


def _transitive_reduction(n: int, edges: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Remove edges (i, k) already implied by a longer directed path i -> ... -> k."""
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)

    reachable: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n - 1, -1, -1):
        for nxt in adj[i]:
            reachable[i].add(nxt)
            reachable[i].update(reachable[nxt])

    reduced: list[tuple[int, int]] = []
    for u, v in sorted(edges):
        implied = any(v in reachable[mid] for mid in adj[u] if mid != v)
        if not implied:
            reduced.append((u, v))
    return reduced


def find_dependencies(steps: Sequence[Step]) -> list[tuple[int, int]]:
    """Return minimal (i, j) dependency constraints with i < j for a valid step sequence."""
    edges: set[tuple[int, int]] = set()
    n = len(steps)
    for i in range(n):
        for j in range(i + 1, n):
            if _has_direct_dependency(steps[i], steps[j]):
                edges.add((i, j))
    return _transitive_reduction(n, edges)


def check_constraints(order: Sequence[int], constraints: Sequence[tuple[int, int]]) -> bool:
    """Return True if an ordering (indices into original steps) satisfies all (a, b) constraints."""
    pos = {step_idx: rank for rank, step_idx in enumerate(order)}
    return all(pos[a] < pos[b] for a, b in constraints)


def find_order_violations(steps: Sequence[Step]) -> list[tuple[int, int, str]]:
    """Detect ordering anomalies in an arbitrary step sequence without a reference protocol."""
    violations: list[tuple[int, int, str]] = []
    n = len(steps)

    # Track first destination transfer index per well.
    first_fill: dict[str, int] = {}
    for idx, step in enumerate(steps):
        if isinstance(step, TransferStep) and is_well_ref(step.dest):
            first_fill.setdefault(step.dest, idx)

    # 1. Well used (mixed or transferred from) before it is first filled.
    for idx, step in enumerate(steps):
        if isinstance(step, MixStep) and is_well_ref(step.well):
            fill_idx = first_fill.get(step.well)
            if fill_idx is not None and idx < fill_idx:
                violations.append(
                    (idx, fill_idx, f"step {idx} mixes {step.well} before step {fill_idx} fills it")
                )
        elif isinstance(step, TransferStep) and is_well_ref(step.source):
            fill_idx = first_fill.get(step.source)
            if fill_idx is not None and idx < fill_idx:
                violations.append(
                    (
                        idx,
                        fill_idx,
                        f"step {idx} transfers from {step.source} before step {fill_idx} fills it",
                    )
                )

    # 2. Transfer from well W before a mix of well W (with no refill between them).
    for i in range(n):
        step_i = steps[i]
        if not (isinstance(step_i, TransferStep) and is_well_ref(step_i.source)):
            continue
        well = step_i.source
        for j in range(i + 1, n):
            step_j = steps[j]
            if isinstance(step_j, TransferStep) and step_j.dest == well:
                break
            if isinstance(step_j, MixStep) and step_j.well == well:
                violations.append(
                    (
                        i,
                        j,
                        f"step {i} transfers from {well} before step {j} mixes {well}",
                    )
                )
                break

    # 2b. Mix of well W followed by a refill transfer into W without a subsequent mix of W.
    for i, step_i in enumerate(steps):
        if not (isinstance(step_i, MixStep) and is_well_ref(step_i.well)):
            continue
        well = step_i.well
        refill_idx: int | None = None
        remixed = False
        for j in range(i + 1, n):
            step_j = steps[j]
            if isinstance(step_j, WashStep) and (step_j.well == well or step_j.plate == plate_of(well)):
                break
            if isinstance(step_j, TransferStep) and step_j.dest == well and refill_idx is None:
                refill_idx = j
            elif isinstance(step_j, MixStep) and step_j.well == well and refill_idx is not None:
                remixed = True
                break
        if refill_idx is not None and not remixed:
            violations.append(
                (
                    i,
                    refill_idx,
                    f"step {i} mixes {well} before step {refill_idx} transfers into {well}",
                )
            )

    # 3. Read before incubate, or read before any preparation step on that plate.
    for i, step_i in enumerate(steps):
        if not isinstance(step_i, ReadStep):
            continue
        plate = step_i.plate
        prior_prep = any(
            not isinstance(steps[k], ReadStep) and plate in step_plates(steps[k])
            for k in range(i)
        )
        washed = False
        for j in range(i + 1, n):
            step_j = steps[j]
            if isinstance(step_j, WashStep) and plate in step_plates(step_j):
                # A wash alone does not end the cycle; keep scanning for a later mix/incubate.
                if not prior_prep:
                    violations.append(
                        (i, j, f"step {i} reads {plate} before step {j} prepares it")
                    )
                    break
                washed = True
                continue
            if washed and isinstance(step_j, TransferStep) and plate in step_plates(step_j):
                # A transfer after the wash starts a genuine new cycle on this plate.
                break
            if isinstance(step_j, IncubateStep) and (step_j.plate is None or step_j.plate == plate):
                violations.append(
                    (i, j, f"step {i} reads {plate} before step {j} incubates it")
                )
                break
            if isinstance(step_j, MixStep) and plate in step_plates(step_j):
                violations.append(
                    (i, j, f"step {i} reads {plate} before step {j} mixes it")
                )
                break
            if not prior_prep and plate in step_plates(step_j):
                violations.append(
                    (i, j, f"step {i} reads {plate} before step {j} prepares it")
                )
                break

    # 3b. Mix or transfer on plate P after a whole-plate wash of P with no read in between.
    # If P was already read before the wash, a transfer starts a new cycle (see rule 3).
    for i, step_i in enumerate(steps):
        if not (isinstance(step_i, WashStep) and step_i.plate is not None and step_i.well is None):
            continue
        plate = step_i.plate
        read_before = any(
            isinstance(steps[k], ReadStep) and steps[k].plate == plate for k in range(i)
        )
        for j in range(i + 1, n):
            step_j = steps[j]
            if isinstance(step_j, ReadStep) and step_j.plate == plate:
                break
            if isinstance(step_j, TransferStep) and plate in step_plates(step_j):
                if not read_before:
                    violations.append(
                        (i, j, f"step {i} washes {plate} before step {j} transfers on it")
                    )
                break
            if isinstance(step_j, MixStep) and plate in step_plates(step_j):
                violations.append(
                    (i, j, f"step {i} washes {plate} before step {j} mixes it")
                )
                break

    # 4. Incubate before any liquid handling on its target plate(s).
    for i, step_i in enumerate(steps):
        if not isinstance(step_i, IncubateStep):
            continue
        target_plates = step_plates(step_i)
        has_prior = any(
            isinstance(steps[k], (TransferStep, MixStep, WashStep))
            and (not target_plates or bool(target_plates & step_plates(steps[k])))
            for k in range(i)
        )
        if not has_prior:
            for j in range(i + 1, n):
                step_j = steps[j]
                if isinstance(step_j, (TransferStep, MixStep)) and (
                    not target_plates or bool(target_plates & step_plates(step_j))
                ):
                    violations.append(
                        (i, j, f"step {i} incubates before liquid handling step {j}")
                    )
                    break

    return violations

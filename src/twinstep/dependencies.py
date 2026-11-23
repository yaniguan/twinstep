"""Dependency graph and ordering constraint utilities for liquid-handling steps."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
import re

from twinstep.schema import (
    IncubateStep,
    MixStep,
    ReadStep,
    Step,
    TransferStep,
    WashStep,
    canonical_location,
)

_EXTERNAL_PREFIXES = ("reservoir", "stock", "trough", "waste")
_BARE_WELL_RE = re.compile(r"^[A-Ha-h](?:[1-9]|1[0-2])$")


def is_well_ref(location: str) -> bool:
    """Return True when a location names an assay well rather than an external reservoir."""
    loc = location.strip()
    prefix = loc.split(":", 1)[0].strip().lower()
    if prefix in _EXTERNAL_PREFIXES:
        return False
    return ":" in loc or bool(_BARE_WELL_RE.match(loc))


def canonical_well(location: str) -> str:
    """Canonicalize a well or container location ('a1' -> 'plate1:A1')."""
    return canonical_location(location)


def plate_of(location: str) -> str | None:
    """Extract the plate identifier associated with a well or plate location."""
    loc = location.strip()
    if ":" in loc:
        prefix = loc.split(":", 1)[0].strip()
        if prefix.lower() in _EXTERNAL_PREFIXES:
            return None
        return prefix
    if _BARE_WELL_RE.match(loc):
        return "plate1"
    if loc.lower() in _EXTERNAL_PREFIXES:
        return None
    return loc


def step_plates(step: Step) -> set[str]:
    """Return plate identifiers touched by a step (empty if global)."""
    if isinstance(step, TransferStep):
        return {p for p in (plate_of(step.source), plate_of(step.dest)) if p is not None}
    if isinstance(step, MixStep):
        p = plate_of(step.well)
        return {p} if p is not None else set()
    if isinstance(step, WashStep):
        plates: set[str] = set()
        if step.plate is not None:
            plates.add(step.plate.strip())
        if step.well is not None:
            p = plate_of(step.well)
            if p is not None:
                plates.add(p)
        return plates
    if isinstance(step, IncubateStep):
        return {step.plate.strip()} if step.plate is not None else set()
    if isinstance(step, ReadStep):
        return {step.plate.strip()}
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
    # Well-level RAW, WAW/flow, and WAR (anti-dependency) constraints.
    if isinstance(earlier, TransferStep):
        e_dst = canonical_well(earlier.dest)
        e_src = canonical_well(earlier.source)
        if isinstance(later, TransferStep):
            l_src = canonical_well(later.source)
            l_dst = canonical_well(later.dest)
            if is_well_ref(earlier.dest) and e_dst == l_src:
                return True
            if is_well_ref(earlier.source) and is_well_ref(later.dest) and e_src == l_dst:
                return True
        if isinstance(later, MixStep) and is_well_ref(earlier.dest):
            if e_dst == canonical_well(later.well):
                return True
        if isinstance(later, WashStep) and later.well is not None:
            l_well = canonical_well(later.well)
            if is_well_ref(earlier.dest) and e_dst == l_well:
                return True
            if is_well_ref(earlier.source) and e_src == l_well:
                return True

    if isinstance(earlier, MixStep) and is_well_ref(earlier.well):
        e_well = canonical_well(earlier.well)
        if isinstance(later, TransferStep):
            if e_well == canonical_well(later.source):
                return True
        if isinstance(later, WashStep) and later.well is not None:
            if e_well == canonical_well(later.well):
                return True

    if isinstance(earlier, WashStep) and earlier.well is not None:
        e_well = canonical_well(earlier.well)
        if isinstance(later, TransferStep) and e_well == canonical_well(later.dest):
            return True
        if isinstance(later, MixStep) and e_well == canonical_well(later.well):
            return True

    # Plate-level stage barriers (incubate, read, whole-plate wash).
    if (_is_barrier(earlier) or _is_barrier(later)) and _plates_overlap(earlier, later):
        return True

    return False


def _transitive_reduction(
    n: int, edges: set[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
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
    return tuple(reduced)


@lru_cache(maxsize=256)
def _cached_dependencies(steps_tuple: tuple[Step, ...]) -> tuple[tuple[int, int], ...]:
    edges: set[tuple[int, int]] = set()
    n = len(steps_tuple)
    for i in range(n):
        for j in range(i + 1, n):
            if _has_direct_dependency(steps_tuple[i], steps_tuple[j]):
                edges.add((i, j))
    return _transitive_reduction(n, edges)


def find_dependencies(steps: Sequence[Step]) -> list[tuple[int, int]]:
    """Return minimal (i, j) dependency constraints with i < j for a valid step sequence."""
    return list(_cached_dependencies(tuple(steps)))


def check_constraints(order: Sequence[int], constraints: Sequence[tuple[int, int]]) -> bool:
    """Return True if an ordering (indices into original steps) satisfies all (a, b) constraints."""
    pos = {step_idx: rank for rank, step_idx in enumerate(order)}
    return all(pos[a] < pos[b] for a, b in constraints)


def find_uninitialized_well_uses(steps: Sequence[Step]) -> list[tuple[int, str, str]]:
    """Return (step_idx, canonical_well, reason) for steps accessing wells never filled."""
    filled: set[str] = set()
    issues: list[tuple[int, str, str]] = []
    for idx, step in enumerate(steps):
        if isinstance(step, TransferStep):
            if is_well_ref(step.source):
                w = canonical_well(step.source)
                if w not in filled:
                    issues.append(
                        (idx, w, f"step {idx} transfers from uninitialized well {step.source}")
                    )
            if is_well_ref(step.dest):
                filled.add(canonical_well(step.dest))
        elif isinstance(step, MixStep) and is_well_ref(step.well):
            w = canonical_well(step.well)
            if w not in filled:
                issues.append((idx, w, f"step {idx} mixes uninitialized well {step.well}"))
    return issues


def find_order_violations(
    steps: Sequence[Step],
    *,
    include_uninitialized: bool = False,
) -> list[tuple[int, int, str]]:
    """Detect ordering anomalies in an arbitrary step sequence without a reference protocol.

    By default (`include_uninitialized=False`), accesses to wells that are *never* filled
    anywhere in `steps` are omitted so `baseline.predict_error` can distinguish `swap_order`
    from `missing_step`. Pass `include_uninitialized=True` to flag never-filled wells too.
    """
    violations: list[tuple[int, int, str]] = []
    n = len(steps)

    # Track first destination transfer index per canonical well.
    first_fill: dict[str, int] = {}
    for idx, step in enumerate(steps):
        if isinstance(step, TransferStep) and is_well_ref(step.dest):
            first_fill.setdefault(canonical_well(step.dest), idx)

    # 1. Well used (mixed or transferred from) before it is first filled (or never filled).
    for idx, step in enumerate(steps):
        if isinstance(step, MixStep) and is_well_ref(step.well):
            w = canonical_well(step.well)
            fill_idx = first_fill.get(w)
            if fill_idx is not None and idx < fill_idx:
                violations.append(
                    (idx, fill_idx, f"step {idx} mixes {step.well} before step {fill_idx} fills it")
                )
            elif fill_idx is None and include_uninitialized:
                violations.append(
                    (idx, -1, f"step {idx} mixes {step.well} which is never filled")
                )
        elif isinstance(step, TransferStep) and is_well_ref(step.source):
            w = canonical_well(step.source)
            fill_idx = first_fill.get(w)
            if fill_idx is not None and idx < fill_idx:
                violations.append(
                    (
                        idx,
                        fill_idx,
                        f"step {idx} transfers from {step.source} before step {fill_idx} fills it",
                    )
                )
            elif fill_idx is None and include_uninitialized:
                violations.append(
                    (idx, -1, f"step {idx} transfers from {step.source} which is never filled")
                )

    # 2. Transfer from well W before a mix of well W (with no refill between them).
    for i in range(n):
        step_i = steps[i]
        if not (isinstance(step_i, TransferStep) and is_well_ref(step_i.source)):
            continue
        well = canonical_well(step_i.source)
        for j in range(i + 1, n):
            step_j = steps[j]
            if isinstance(step_j, TransferStep) and canonical_well(step_j.dest) == well:
                break
            if isinstance(step_j, MixStep) and canonical_well(step_j.well) == well:
                violations.append(
                    (
                        i,
                        j,
                        f"step {i} transfers from {step_i.source} "
                        f"before step {j} mixes {step_j.well}",
                    )
                )
                break

    # 2b. Mix of well W followed by a refill transfer into W without a subsequent mix of W.
    for i, step_i in enumerate(steps):
        if not (isinstance(step_i, MixStep) and is_well_ref(step_i.well)):
            continue
        well = canonical_well(step_i.well)
        w_plate = plate_of(step_i.well)
        refill_idx: int | None = None
        remixed = False
        for j in range(i + 1, n):
            step_j = steps[j]
            if isinstance(step_j, WashStep) and (
                (step_j.well is not None and canonical_well(step_j.well) == well)
                or (step_j.plate is not None and step_j.plate == w_plate)
            ):
                break
            if (
                isinstance(step_j, TransferStep)
                and canonical_well(step_j.dest) == well
                and refill_idx is None
            ):
                refill_idx = j
            elif (
                isinstance(step_j, MixStep)
                and canonical_well(step_j.well) == well
                and refill_idx is not None
            ):
                remixed = True
                break
        if refill_idx is not None and not remixed:
            violations.append(
                (
                    i,
                    refill_idx,
                    f"step {i} mixes {step_i.well} before step {refill_idx} transfers into {well}",
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
            isinstance(s_k, ReadStep) and s_k.plate == plate
            for s_k in steps[:i]
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

    # 4. Incubate before remaining unseparated liquid handling on its target plate(s).
    for i, step_i in enumerate(steps):
        if not isinstance(step_i, IncubateStep):
            continue
        target_plates = step_plates(step_i)
        has_prior = any(
            isinstance(steps[k], (TransferStep, MixStep, WashStep))
            and (not target_plates or bool(target_plates & step_plates(steps[k])))
            for k in range(i)
        )
        for j in range(i + 1, n):
            step_j = steps[j]
            sj_plates = step_plates(step_j)
            touches_target = not target_plates or bool(target_plates & sj_plates)
            if has_prior and touches_target and isinstance(step_j, (ReadStep, WashStep)):
                # A ReadStep or WashStep after prior prep closes the incubation cycle.
                break
            if isinstance(step_j, (TransferStep, MixStep)) and touches_target:
                violations.append(
                    (i, j, f"step {i} incubates before liquid handling step {j}")
                )
                break

    return violations

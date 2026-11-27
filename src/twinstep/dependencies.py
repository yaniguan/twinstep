"""Dependency graph and ordering constraint utilities for liquid-handling steps."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from twinstep.schema import (
    IncubateStep,
    MixStep,
    ReadStep,
    Step,
    TransferStep,
    WashStep,
    canonical_location,
    canonical_plate,
    is_bare_well,
)

_EXTERNAL_PREFIXES = ("reservoir", "stock", "trough", "waste")


def _explicit_well_plate(location: str) -> str | None:
    loc = location.strip()
    if ":" not in loc:
        return None
    prefix = canonical_plate(loc.split(":", 1)[0])
    if prefix in _EXTERNAL_PREFIXES:
        return None
    return prefix


def infer_default_plate(steps: Sequence[Step | None]) -> str:
    """Bind bare wells to the unique explicit plate in ``steps`` (else ``'plate1'``)."""
    explicit: set[str] = set()
    for step in steps:
        if step is None:
            continue
        if isinstance(step, TransferStep):
            for loc in (step.source, step.dest):
                p = _explicit_well_plate(loc)
                if p is not None:
                    explicit.add(p)
        elif isinstance(step, MixStep):
            p = _explicit_well_plate(step.well)
            if p is not None:
                explicit.add(p)
        elif isinstance(step, WashStep):
            if step.plate is not None:
                explicit.add(canonical_plate(step.plate))
            if step.well is not None:
                p = _explicit_well_plate(step.well)
                if p is not None:
                    explicit.add(p)
        elif isinstance(step, IncubateStep):
            if step.plate is not None:
                explicit.add(canonical_plate(step.plate))
        elif isinstance(step, ReadStep):
            explicit.add(canonical_plate(step.plate))
    if len(explicit) == 1:
        return next(iter(explicit))
    return "plate1"


def is_well_ref(location: str) -> bool:
    """Return True when a location names an assay well rather than an external reservoir."""
    loc = location.strip()
    prefix = loc.split(":", 1)[0].strip().lower()
    if prefix in _EXTERNAL_PREFIXES:
        return False
    return ":" in loc or is_bare_well(loc)


def canonical_well(location: str, default_plate: str = "plate1") -> str:
    """Canonicalize a well or container location ('a01' -> '<default_plate>:A1')."""
    return canonical_location(location, default_plate=default_plate)


def plate_of(location: str, default_plate: str = "plate1") -> str | None:
    """Extract the canonical plate identifier associated with a well or plate location."""
    loc = location.strip()
    if ":" in loc:
        prefix = canonical_plate(loc.split(":", 1)[0])
        if prefix in _EXTERNAL_PREFIXES:
            return None
        return prefix
    if is_bare_well(loc):
        return canonical_plate(default_plate)
    if loc.lower() in _EXTERNAL_PREFIXES:
        return None
    return canonical_plate(loc)


def step_plates(step: Step, default_plate: str = "plate1") -> set[str]:
    """Return canonical plate identifiers touched by a step (empty if global)."""
    if isinstance(step, TransferStep):
        return {
            p
            for p in (
                plate_of(step.source, default_plate),
                plate_of(step.dest, default_plate),
            )
            if p is not None
        }
    if isinstance(step, MixStep):
        p = plate_of(step.well, default_plate)
        return {p} if p is not None else set()
    if isinstance(step, WashStep):
        plates: set[str] = set()
        if step.plate is not None:
            plates.add(canonical_plate(step.plate))
        if step.well is not None:
            p = plate_of(step.well, default_plate)
            if p is not None:
                plates.add(p)
        return plates
    if isinstance(step, IncubateStep):
        return {canonical_plate(step.plate)} if step.plate is not None else set()
    if isinstance(step, ReadStep):
        return {canonical_plate(step.plate)}
    return set()


def _plates_overlap(a: Step, b: Step, default_plate: str = "plate1") -> bool:
    pa = step_plates(a, default_plate)
    pb = step_plates(b, default_plate)
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


def _has_direct_dependency(
    earlier: Step, later: Step, default_plate: str = "plate1"
) -> bool:
    # Well-level RAW, WAW/flow, and WAR (anti-dependency) constraints.
    if isinstance(earlier, TransferStep):
        e_dst = canonical_well(earlier.dest, default_plate)
        e_src = canonical_well(earlier.source, default_plate)
        if isinstance(later, TransferStep):
            l_src = canonical_well(later.source, default_plate)
            l_dst = canonical_well(later.dest, default_plate)
            if is_well_ref(earlier.dest) and e_dst == l_src:
                return True
            if is_well_ref(earlier.source) and is_well_ref(later.dest) and e_src == l_dst:
                return True
        if isinstance(later, MixStep) and is_well_ref(earlier.dest):
            if e_dst == canonical_well(later.well, default_plate):
                return True
        if isinstance(later, WashStep) and later.well is not None:
            l_well = canonical_well(later.well, default_plate)
            if is_well_ref(earlier.dest) and e_dst == l_well:
                return True
            if is_well_ref(earlier.source) and e_src == l_well:
                return True

    if isinstance(earlier, MixStep) and is_well_ref(earlier.well):
        e_well = canonical_well(earlier.well, default_plate)
        if isinstance(later, TransferStep):
            if is_well_ref(later.source) and e_well == canonical_well(
                later.source, default_plate
            ):
                return True
            if is_well_ref(later.dest) and e_well == canonical_well(
                later.dest, default_plate
            ):
                return True
        if isinstance(later, MixStep) and is_well_ref(later.well):
            if e_well == canonical_well(later.well, default_plate) and earlier != later:
                return True
        if isinstance(later, WashStep) and later.well is not None:
            if e_well == canonical_well(later.well, default_plate):
                return True

    if isinstance(earlier, WashStep) and earlier.well is not None:
        e_well = canonical_well(earlier.well, default_plate)
        if isinstance(later, TransferStep) and e_well == canonical_well(
            later.dest, default_plate
        ):
            return True
        if isinstance(later, MixStep) and e_well == canonical_well(
            later.well, default_plate
        ):
            return True

    # Plate-level stage barriers (incubate, read, whole-plate wash).
    if (_is_barrier(earlier) or _is_barrier(later)) and _plates_overlap(
        earlier, later, default_plate
    ):
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
    default_plate = infer_default_plate(steps_tuple)
    for i in range(n):
        for j in range(i + 1, n):
            if _has_direct_dependency(steps_tuple[i], steps_tuple[j], default_plate):
                edges.add((i, j))
    return _transitive_reduction(n, edges)


def find_dependencies(steps: Sequence[Step]) -> list[tuple[int, int]]:
    """Return minimal (i, j) dependency constraints with i < j for a valid step sequence."""
    return list(_cached_dependencies(tuple(steps)))


def check_constraints(order: Sequence[int], constraints: Sequence[tuple[int, int]]) -> bool:
    """Return True if an ordering (indices into original steps) satisfies all (a, b) constraints."""
    if len(order) != len(set(order)):
        return False
    pos = {step_idx: rank for rank, step_idx in enumerate(order)}
    return all(a in pos and b in pos and pos[a] < pos[b] for a, b in constraints)


def find_uninitialized_well_uses(steps: Sequence[Step]) -> list[tuple[int, str, str]]:
    """Return (step_idx, canonical_well, reason) for steps accessing wells never filled."""
    default_plate = infer_default_plate(steps)
    filled: set[str] = set()
    issues: list[tuple[int, str, str]] = []
    for idx, step in enumerate(steps):
        if isinstance(step, TransferStep):
            if is_well_ref(step.source):
                w = canonical_well(step.source, default_plate)
                if w not in filled:
                    issues.append(
                        (idx, w, f"step {idx} transfers from uninitialized well {step.source}")
                    )
            if is_well_ref(step.dest):
                filled.add(canonical_well(step.dest, default_plate))
        elif isinstance(step, MixStep) and is_well_ref(step.well):
            w = canonical_well(step.well, default_plate)
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
    default_plate = infer_default_plate(steps)

    # Track first destination transfer index per canonical well.
    first_fill: dict[str, int] = {}
    for idx, step in enumerate(steps):
        if isinstance(step, TransferStep) and is_well_ref(step.dest):
            first_fill.setdefault(canonical_well(step.dest, default_plate), idx)

    # 1. Well used (mixed or transferred from) before it is first filled (or never filled).
    for idx, step in enumerate(steps):
        if isinstance(step, MixStep) and is_well_ref(step.well):
            w = canonical_well(step.well, default_plate)
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
            w = canonical_well(step.source, default_plate)
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
        well = canonical_well(step_i.source, default_plate)
        for j in range(i + 1, n):
            step_j = steps[j]
            if (
                isinstance(step_j, TransferStep)
                and canonical_well(step_j.dest, default_plate) == well
            ):
                break
            if isinstance(step_j, MixStep) and canonical_well(step_j.well, default_plate) == well:
                violations.append(
                    (
                        i,
                        j,
                        f"step {i} transfers from {step_i.source} "
                        f"before step {j} mixes {step_j.well}",
                    )
                )
                break

    # 2b. Mix of well W followed by a refill transfer into W within the same stage
    # without a subsequent mix of W.
    for i, step_i in enumerate(steps):
        if not (isinstance(step_i, MixStep) and is_well_ref(step_i.well)):
            continue
        well = canonical_well(step_i.well, default_plate)
        w_plate = plate_of(step_i.well, default_plate)
        refill_idx: int | None = None
        remixed = False
        for j in range(i + 1, n):
            step_j = steps[j]
            is_stage_barrier = (
                (
                    isinstance(step_j, WashStep)
                    and (
                        (
                            step_j.well is not None
                            and canonical_well(step_j.well, default_plate) == well
                        )
                        or (
                            step_j.plate is not None
                            and canonical_plate(step_j.plate) == w_plate
                        )
                    )
                )
                or (
                    isinstance(step_j, IncubateStep)
                    and (step_j.plate is None or canonical_plate(step_j.plate) == w_plate)
                )
                or (isinstance(step_j, ReadStep) and canonical_plate(step_j.plate) == w_plate)
            )
            if is_stage_barrier:
                break
            if (
                isinstance(step_j, TransferStep)
                and canonical_well(step_j.dest, default_plate) == well
                and refill_idx is None
            ):
                refill_idx = j
            elif (
                isinstance(step_j, MixStep)
                and canonical_well(step_j.well, default_plate) == well
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
        plate = canonical_plate(step_i.plate)
        prior_prep = any(
            not isinstance(steps[k], ReadStep) and plate in step_plates(steps[k], default_plate)
            for k in range(i)
        )
        washed = False
        for j in range(i + 1, n):
            step_j = steps[j]
            if isinstance(step_j, WashStep) and plate in step_plates(step_j, default_plate):
                # A wash alone does not end the cycle; keep scanning for a later mix/incubate.
                if not prior_prep:
                    violations.append(
                        (i, j, f"step {i} reads {plate} before step {j} prepares it")
                    )
                    break
                washed = True
                continue
            if (
                washed
                and isinstance(step_j, TransferStep)
                and plate in step_plates(step_j, default_plate)
            ):
                # A transfer after the wash starts a genuine new cycle on this plate.
                break
            if isinstance(step_j, IncubateStep) and (
                step_j.plate is None or canonical_plate(step_j.plate) == plate
            ):
                violations.append(
                    (i, j, f"step {i} reads {plate} before step {j} incubates it")
                )
                break
            if isinstance(step_j, MixStep) and plate in step_plates(step_j, default_plate):
                violations.append(
                    (i, j, f"step {i} reads {plate} before step {j} mixes it")
                )
                break
            if isinstance(step_j, TransferStep) and plate in step_plates(step_j, default_plate):
                has_later_read = any(
                    isinstance(s_m, ReadStep) and canonical_plate(s_m.plate) == plate
                    for s_m in steps[j + 1 :]
                )
                if not prior_prep or not has_later_read:
                    violations.append(
                        (i, j, f"step {i} reads {plate} before step {j} prepares it")
                    )
                    break
            if not prior_prep and plate in step_plates(step_j, default_plate):
                violations.append(
                    (i, j, f"step {i} reads {plate} before step {j} prepares it")
                )
                break

    # 3b. Well-flow wash check: flag (a) mixing a pre-wash well after a whole-plate wash
    # without a post-wash refill, or (b) transferring onto a plate whose pre-wash well writes
    # were never incubated or read prior to the wash (prematurely swapped wash).
    for i, step_i in enumerate(steps):
        if not (isinstance(step_i, WashStep) and step_i.plate is not None and step_i.well is None):
            continue
        plate = canonical_plate(step_i.plate)
        pre_wash_wells: set[str] = set()
        unconsumed_wells: set[str] = set()
        for k in range(i):
            s_k = steps[k]
            if isinstance(s_k, TransferStep) and is_well_ref(s_k.dest):
                if plate_of(s_k.dest, default_plate) == plate:
                    w_k = canonical_well(s_k.dest, default_plate)
                    pre_wash_wells.add(w_k)
                    unconsumed_wells.add(w_k)
            elif isinstance(s_k, ReadStep) and canonical_plate(s_k.plate) == plate:
                unconsumed_wells.clear()
            elif isinstance(s_k, IncubateStep) and (
                s_k.plate is None or canonical_plate(s_k.plate) == plate
            ):
                unconsumed_wells.clear()
            elif (
                isinstance(s_k, WashStep)
                and s_k.plate is not None
                and s_k.well is None
                and canonical_plate(s_k.plate) == plate
            ):
                unconsumed_wells.clear()

        post_wash_filled: set[str] = set()
        for j in range(i + 1, n):
            step_j = steps[j]
            if isinstance(step_j, ReadStep) and canonical_plate(step_j.plate) == plate:
                break
            if isinstance(step_j, IncubateStep) and (
                step_j.plate is None or canonical_plate(step_j.plate) == plate
            ):
                break
            if isinstance(step_j, TransferStep) and plate in step_plates(step_j, default_plate):
                if unconsumed_wells:
                    violations.append(
                        (i, j, f"step {i} washes {plate} before step {j} transfers on it")
                    )
                    break
                if is_well_ref(step_j.dest) and plate_of(step_j.dest, default_plate) == plate:
                    post_wash_filled.add(canonical_well(step_j.dest, default_plate))
            elif (
                isinstance(step_j, MixStep)
                and is_well_ref(step_j.well)
                and plate_of(step_j.well, default_plate) == plate
            ):
                mw = canonical_well(step_j.well, default_plate)
                if mw in pre_wash_wells and mw not in post_wash_filled:
                    violations.append(
                        (i, j, f"step {i} washes {plate} before step {j} mixes it")
                    )
                    break

    # 4. Well-flow incubate check: allow post-incubation reagent additions (e.g. stop solution
    # or multi-stage viability reagents + mix) on already-initialized wells, while flagging
    # empty-plate incubations, unrefilled post-incubation mixes, or unincubated parallel wells.
    for i, step_i in enumerate(steps):
        if not isinstance(step_i, IncubateStep):
            continue
        target_plates = step_plates(step_i, default_plate)
        wells_before_inc: set[str] = set()
        has_prior = False
        for k in range(i):
            s_k = steps[k]
            sk_plates = step_plates(s_k, default_plate)
            overlaps_k = not target_plates or bool(target_plates & sk_plates)
            if isinstance(s_k, (TransferStep, MixStep, WashStep)) and overlaps_k:
                has_prior = True
            if isinstance(s_k, TransferStep) and is_well_ref(s_k.dest):
                dp = plate_of(s_k.dest, default_plate)
                if not target_plates or (dp is not None and dp in target_plates):
                    wells_before_inc.add(canonical_well(s_k.dest, default_plate))

        post_inc_filled: set[str] = set()
        for j in range(i + 1, n):
            step_j = steps[j]
            sj_plates = step_plates(step_j, default_plate)
            touches_target = (not target_plates and bool(sj_plates)) or bool(
                target_plates & sj_plates
            )
            if has_prior and touches_target and isinstance(
                step_j, (ReadStep, WashStep, IncubateStep)
            ):
                break
            if isinstance(step_j, TransferStep) and touches_target:
                if not wells_before_inc and not has_prior:
                    violations.append(
                        (i, j, f"step {i} incubates before liquid handling step {j}")
                    )
                    break
                if is_well_ref(step_j.dest):
                    dp = plate_of(step_j.dest, default_plate)
                    if not target_plates or (dp is not None and dp in target_plates):
                        dst_w = canonical_well(step_j.dest, default_plate)
                        if (
                            not is_well_ref(step_j.source)
                            and dst_w not in wells_before_inc
                            and dst_w not in post_inc_filled
                        ):
                            has_later_inc = any(
                                isinstance(s_m, IncubateStep)
                                and (s_m.plate is None or canonical_plate(s_m.plate) == dp)
                                for s_m in steps[j + 1 :]
                            )
                            if not has_later_inc:
                                violations.append(
                                    (i, j, f"step {i} incubates before liquid handling step {j}")
                                )
                                break
                        post_inc_filled.add(dst_w)
            elif isinstance(step_j, MixStep) and touches_target and is_well_ref(step_j.well):
                mw = canonical_well(step_j.well, default_plate)
                if mw not in post_inc_filled:
                    violations.append(
                        (i, j, f"step {i} incubates before liquid handling step {j}")
                    )
                    break

    return violations

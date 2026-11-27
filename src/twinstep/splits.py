"""Deterministic whole-protocol train/val/test splits and blind-eval sanitization."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
from typing import Any, Literal

SplitName = Literal["train", "val", "test"]

VALID_SPLITS: tuple[str, ...] = ("train", "val", "test")
SPLIT_FILTERS: tuple[str, ...] = ("all", "train", "val", "test")

# Canonical whole-protocol split table for the 24 MIT-licensed bundled protocols
# (8 train, 4 val, 12 held-out test).
BUNDLED_PROTOCOL_SPLITS: dict[str, SplitName] = {
    # --- train (8) ---
    "dilution_series": "train",
    "mix_incubate_read": "train",
    "wash_and_read": "train",
    "bca_protein_assay": "train",
    "qpcr_sybr_setup": "train",
    "plasmid_miniprep_elution": "train",
    "mtt_viability_assay": "train",
    "direct_elisa_coating": "train",
    # --- val (4) ---
    "bradford_coomassie_assay": "val",
    "crispr_rnp_cleavage": "val",
    "kinetic_wash_reprobe": "val",
    "turbidity_od600_growth": "val",
    # --- test (12) ---
    "sandwich_elisa": "test",
    "rt_qpcr_mastermix": "test",
    "spri_magnetic_bead_cleanup": "test",
    "ic50_384well_serial_dilution": "test",
    "gibson_assembly_reaction": "test",
    "western_blot_wash_probe": "test",
    "cell_viability_luminescent": "test",
    "lal_endotoxin_chromogenic": "test",
    "picogreen_dna_quant": "test",
    "enzyme_quench_kinetics": "test",
    "high_content_cell_imaging": "test",
    "competitive_immunoassay_384": "test",
}

_BLIND_STRIP_KEYS: frozenset[str] = frozenset(
    {
        "correct_steps",
        "gold_order",
        "constraints",
        "label",
        "error_type",
        "labels",
        "step_index",
        "step_indices",
        "original_value",
        "new_value",
        "description",
        "descriptions",
        "ground_truth",
        "chosen",
        "rejected",
        "rejected_candidates",
        "chosen_rewards",
        "rejected_rewards",
        "completion",
        "clean_saboteur_steps",
        "violated_lean_rules",
        "evaded_gates",
        "mutation_ops",
    }
)


def assign_protocol_split(protocol_id: str, explicit_split: str | None = None) -> SplitName:
    """Assign a whole ``protocol_id`` deterministically to ``train``, ``val``, or ``test``.

    Bundled protocols follow ``BUNDLED_PROTOCOL_SPLITS`` (8 train / 4 val / 12 test).
    Custom protocols honor ``explicit_split`` when provided, and otherwise hash
    ``protocol_id`` via SHA-256 into rough 33% / 17% / 50% whole-protocol buckets.
    """
    if explicit_split is not None:
        norm = explicit_split.strip().lower()
        if norm == "train":
            return "train"
        if norm == "val":
            return "val"
        if norm == "test":
            return "test"
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {explicit_split!r}")

    key = protocol_id.strip()
    if key in BUNDLED_PROTOCOL_SPLITS:
        return BUNDLED_PROTOCOL_SPLITS[key]
    # Strip trailing suffixes if a record ID prefix was passed.
    base_key = key.split(":", 1)[0]
    if base_key in BUNDLED_PROTOCOL_SPLITS:
        return BUNDLED_PROTOCOL_SPLITS[base_key]

    digest = hashlib.sha256(base_key.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % 24
    if bucket < 8:
        return "train"
    if bucket < 12:
        return "val"
    return "test"


def strip_for_blind_eval(record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``record`` with reference protocols and ground-truth labels removed."""
    return {k: v for k, v in record.items() if k not in _BLIND_STRIP_KEYS}


def detect_protocol_split_overlaps(
    records: Sequence[dict[str, Any]],
    train_records: Sequence[dict[str, Any]] | None = None,
) -> list[str]:
    """Return sorted ``protocol_id`` values that appear in both ``train`` and ``test``."""
    train_pids: set[str] = set()
    test_pids: set[str] = set()

    if train_records is not None:
        for rec in train_records:
            pid = rec.get("protocol_id")
            if pid is not None:
                train_pids.add(str(pid))
        for rec in records:
            pid = rec.get("protocol_id")
            split = str(rec.get("split") or "test")
            if pid is not None and (train_records is not None or split == "test"):
                if split in ("test", "val") or rec.get("split") is None:
                    test_pids.add(str(pid))

    by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for rec in records:
        pid = rec.get("protocol_id")
        sp = rec.get("split")
        if pid is not None and isinstance(sp, str) and sp in by_split:
            by_split[sp].add(str(pid))

    overlaps = (train_pids & test_pids) | (by_split["train"] & by_split["test"])
    return sorted(overlaps)


__all__ = [
    "BUNDLED_PROTOCOL_SPLITS",
    "SPLIT_FILTERS",
    "SplitName",
    "VALID_SPLITS",
    "assign_protocol_split",
    "detect_protocol_split_overlaps",
    "strip_for_blind_eval",
]

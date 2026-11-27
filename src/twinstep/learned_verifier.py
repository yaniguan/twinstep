"""Zero-leakage graph-feature learned fault localizer for liquid-handling protocols.

Trains strictly on ``train``-split protocols (refusing any overlap with ``val`` or ``test``
``protocol_id``s) using reference-free relational, well-bipartite symmetry, stoichiometry,
and causal-DAG features extracted solely from candidate ``corrupted_steps``.

Supports a pure Python 3.11+ standard-library multinomial logistic + step-ranking backend
(``backend="stdlib"``) as well as optional ``scikit-learn`` and ``lightgbm`` backends when
installed in an external research environment.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import importlib
import math
import random
import statistics
from typing import Any

from twinstep.baseline import (
    DEFAULT_MAX_INCUBATE_MIN,
    _check_single_step_bounds,
    _volume_in_ul,
    predict_error,
    predict_order,
)
from twinstep.dependencies import (
    canonical_well,
    find_dependencies,
    find_order_violations,
    infer_default_plate,
    is_well_ref,
    step_plates,
)
from twinstep.schema import (
    IncubateStep,
    MixStep,
    ReadStep,
    Step,
    TransferStep,
    WashStep,
    step_from_dict,
)

ALL_VERIFIER_LABELS: tuple[str, ...] = (
    "clean",
    "swap_order",
    "wrong_volume",
    "wrong_unit",
    "missing_step",
    "wrong_temperature",
    "wrong_time",
)

_CANONICAL_TEMPS_C: tuple[float, ...] = (4.0, 20.0, 25.0, 30.0, 37.0, 42.0, 50.0, 60.0, 72.0, 95.0)
_CANONICAL_MIN_DURATIONS: frozenset[float] = frozenset(
    {
        2.0,
        3.0,
        5.0,
        6.0,
        10.0,
        15.0,
        16.0,
        20.0,
        30.0,
        45.0,
        60.0,
        90.0,
        120.0,
        720.0,
        960.0,
        1440.0,
    }
)


def _softmax(logits: Sequence[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _extract_protocol_and_step_features(
    raw_steps: Sequence[Step | dict[str, Any]],
) -> tuple[list[float], dict[str, list[tuple[int, float]]]]:
    """Extract reference-free protocol-level features and per-fault-class step anomaly scores.

    Inspects ONLY the candidate ``raw_steps`` (never ``correct_steps`` or ``protocol_id``).
    """
    parsed: list[Step | None] = []
    schema_fault: tuple[int, str] | None = None
    for idx, item in enumerate(raw_steps):
        if isinstance(item, dict):
            try:
                parsed.append(step_from_dict(item))
            except ValueError:
                rule_hit = predict_error(raw_steps)
                schema_fault = (idx, str(rule_hit["label"]))
                parsed.append(None)
        else:
            parsed.append(item)

    valid_steps = [s for s in parsed if s is not None]
    n = len(raw_steps)
    default_plate = infer_default_plate(valid_steps) if valid_steps else "plate1"

    rule_pred = predict_error(raw_steps)
    rule_label = str(rule_pred["label"])
    rule_idx = rule_pred["step_index"]

    violations = find_order_violations(valid_steps) if valid_steps else []
    deps = find_dependencies(valid_steps) if valid_steps else []

    # Group volumes by (step.type, source_prefix_or_kind) and per destination well
    reagent_vols: dict[str, list[tuple[int, float]]] = defaultdict(list)
    type_vols: dict[str, list[tuple[int, float]]] = defaultdict(list)
    well_vols: dict[str, list[tuple[int, float]]] = defaultdict(list)
    reagent_to_wells: dict[str, list[tuple[int, str]]] = defaultdict(list)
    all_assay_wells_order: list[str] = []
    mix_wells: dict[str, list[int]] = defaultdict(list)

    for idx, st in enumerate(parsed):
        if isinstance(st, TransferStep):
            vul = _volume_in_ul(st.volume, st.unit)
            type_vols["transfer"].append((idx, vul))
            if not is_well_ref(st.source):
                reagent_vols[st.source.lower()].append((idx, vul))
            if is_well_ref(st.dest):
                dw = canonical_well(st.dest, default_plate)
                well_vols[dw].append((idx, vul))
                if dw not in all_assay_wells_order:
                    all_assay_wells_order.append(dw)
                if not is_well_ref(st.source):
                    reagent_to_wells[st.source.lower()].append((idx, dw))
            if is_well_ref(st.source):
                sw = canonical_well(st.source, default_plate)
                well_vols[sw].append((idx, vul))
                if sw not in all_assay_wells_order:
                    all_assay_wells_order.append(sw)
        elif isinstance(st, MixStep):
            vul = _volume_in_ul(st.volume, st.unit)
            type_vols["mix"].append((idx, vul))
            if is_well_ref(st.well):
                mw = canonical_well(st.well, default_plate)
                well_vols[mw].append((idx, vul))
                mix_wells[mw].append(idx)
                if mw not in all_assay_wells_order:
                    all_assay_wells_order.append(mw)
        elif isinstance(st, WashStep):
            vul = _volume_in_ul(st.volume, st.unit)
            type_vols["wash"].append((idx, vul))

    # Candidate step localization scores per fault family
    cand_scores: dict[str, list[tuple[int, float]]] = {lbl: [] for lbl in ALL_VERIFIER_LABELS}

    max_vol_peer_log_ratio = 0.0
    max_vol_well_log_ratio = 0.0
    max_unit_flag = 0.0
    max_temp_flag = 0.0
    max_temp_noncanon_dist = 0.0
    max_time_bound_flag = 0.0
    max_time_sec_flip_score = 0.0
    max_time_scale_score = 0.0

    for idx, st in enumerate(parsed):
        if st is None:
            if schema_fault is not None and schema_fault[0] == idx:
                cand_scores[schema_fault[1]].append((idx, 10.0))
            continue

        bound_hit = _check_single_step_bounds(idx, st, max_incubate_min=DEFAULT_MAX_INCUBATE_MIN)
        if bound_hit is not None and bound_hit["label"] in cand_scores:
            b_lbl = str(bound_hit["label"])
            cand_scores[b_lbl].append((idx, 8.0))
            if b_lbl == "wrong_unit":
                max_unit_flag = 1.0
            elif b_lbl == "wrong_volume":
                max_vol_peer_log_ratio = max(max_vol_peer_log_ratio, 1.5)
            elif b_lbl == "wrong_temperature":
                max_temp_flag = 1.0
            elif b_lbl == "wrong_time":
                max_time_bound_flag = 1.0

        if isinstance(st, (TransferStep, MixStep, WashStep)):
            vul = _volume_in_ul(st.volume, st.unit)
            if st.unit == "mL":
                cand_scores["wrong_unit"].append((idx, 4.0))
                max_unit_flag = max(max_unit_flag, 0.8)

            # Compare against same-reagent peers (if >= 2) or same-type peers
            peers: list[float] = []
            if isinstance(st, TransferStep) and not is_well_ref(st.source):
                peers = [v for j, v in reagent_vols[st.source.lower()] if j != idx]
            if not peers:
                peers = [v for j, v in type_vols[st.type] if j != idx]

            peer_ratio = 0.0
            if peers and vul > 0:
                med_peer = statistics.median(peers)
                if med_peer > 0:
                    peer_ratio = abs(math.log10(vul / med_peer))
            well_ratio = 0.0
            w_other: list[float] = []
            if isinstance(st, TransferStep) and is_well_ref(st.dest):
                dw = canonical_well(st.dest, default_plate)
                w_other = [v for j, v in well_vols[dw] if j != idx]
            elif isinstance(st, MixStep) and is_well_ref(st.well):
                mw = canonical_well(st.well, default_plate)
                w_other = [v for j, v in well_vols[mw] if j != idx]
            if w_other and vul > 0:
                med_w = statistics.median(w_other)
                if med_w > 0:
                    well_ratio = abs(math.log10(vul / med_w))

            # Subtle decimal/factor-of-10 anomalies (e.g. 25 uL -> 2.5 uL, 20 uL -> 2 uL, 180 -> 18)
            is_fractional = 1.0 if abs(vul - round(vul)) > 1e-4 else 0.0
            vol_anom = 1.8 * peer_ratio + 1.4 * well_ratio + 0.6 * is_fractional
            if bound_hit is not None and bound_hit["label"] == "wrong_volume":
                vol_anom += 6.0
            cand_scores["wrong_volume"].append((idx, vol_anom))
            max_vol_peer_log_ratio = max(max_vol_peer_log_ratio, peer_ratio)
            max_vol_well_log_ratio = max(max_vol_well_log_ratio, well_ratio)

        elif isinstance(st, IncubateStep):
            t_dist = min(abs(st.temperature - tc) for tc in _CANONICAL_TEMPS_C)
            is_temp_bound = (
                max_temp_flag > 0
                and bound_hit is not None
                and bound_hit["label"] == "wrong_temperature"
            )
            temp_anom = (t_dist / 10.0) + (6.0 if is_temp_bound else 0.0)
            cand_scores["wrong_temperature"].append((idx, temp_anom))
            max_temp_noncanon_dist = max(max_temp_noncanon_dist, t_dist / 25.0)

            sec_flip = 0.0
            scale_anom = 0.0
            if st.time_unit == "s":
                # Unit-flipped minute incubations (e.g. 15 min -> 15 s, 30 min -> 30 s)
                if st.time in _CANONICAL_MIN_DURATIONS or st.time < 120.0:
                    sec_flip = 1.2 if st.time in _CANONICAL_MIN_DURATIONS else 0.8
            else:
                # 0.1x or 10x scaled minute durations (e.g. 15 -> 1.5, 30 -> 3.0, 60 -> 600)
                if abs(st.time - round(st.time)) > 1e-4:
                    scale_anom = 1.4
                elif st.time not in _CANONICAL_MIN_DURATIONS:
                    if st.time in {1.0, 4.5, 9.0, 96.0, 144.0, 150.0, 200.0, 300.0, 450.0, 600.0}:
                        scale_anom = 1.25
                    elif st.time < 2.0 or (180.0 < st.time < 700.0):
                        scale_anom = 0.9
            time_anom = 2.0 * sec_flip + 1.8 * scale_anom
            if bound_hit is not None and bound_hit["label"] == "wrong_time":
                time_anom += 6.0
            cand_scores["wrong_time"].append((idx, time_anom))
            max_time_sec_flip_score = max(max_time_sec_flip_score, sec_flip)
            max_time_scale_score = max(max_time_scale_score, scale_anom)

    if violations:
        v_idx = violations[0][0]
        cand_scores["swap_order"].append((v_idx, 9.0))

    # Bipartite (reagent, well) and mix-symmetry deficit detector for `missing_step`
    bipartite_deficit = 0.0
    missing_mix_symmetry = 0.0
    if len(all_assay_wells_order) >= 2:
        well_set = set(all_assay_wells_order)
        for r_name, rw_list in reagent_to_wells.items():
            covered_wells = {w for _, w in rw_list}
            # Replicate reagent dispensed to k >= 1 wells but missing 1 peer well!
            if 1 <= len(covered_wells) < len(well_set) and (len(well_set) - len(covered_wells)) == 1:  # noqa: E501
                # Check whether the missing well has similar downstream usage (e.g. mixed or read)
                missing_w = next(w for w in all_assay_wells_order if w not in covered_wells)
                if len(well_vols[missing_w]) >= 1:
                    bipartite_deficit = 1.0
                    # Estimate insertion index from well order among the reagent's transfers
                    missing_rank = all_assay_wells_order.index(missing_w)
                    sorted_rw = sorted(rw_list, key=lambda item: item[0])
                    ins_idx = sorted_rw[0][0]
                    for pos_idx, w_here in sorted_rw:
                        if all_assay_wells_order.index(w_here) < missing_rank:
                            ins_idx = pos_idx + 1
                    cand_scores["missing_step"].append((ins_idx, 4.5))

        mixed_well_set = set(mix_wells.keys())
        multi_fill_wells = [w for w in all_assay_wells_order if len(well_vols[w]) >= 2]
        if 1 <= len(mixed_well_set) < len(multi_fill_wells):
            missing_mix_symmetry = 1.0
            missing_mw = next(w for w in multi_fill_wells if w not in mixed_well_set)
            mw_rank = multi_fill_wells.index(missing_mw)
            existing_mix_idxs = sorted(i for idxs in mix_wells.values() for i in idxs)
            ins_m = existing_mix_idxs[0] + mw_rank if existing_mix_idxs else n - 1
            cand_scores["missing_step"].append((min(ins_m, n), 4.8))

    if rule_label == "missing_step" and isinstance(rule_idx, int):
        cand_scores["missing_step"].append((rule_idx, 7.5))

    # Also detect consecutive barrier gaps (e.g. dropped wash or incubate between two stages)
    consec_reads_or_washes = 0.0
    for idx in range(len(valid_steps) - 1):
        a_s, b_s = valid_steps[idx], valid_steps[idx + 1]
        if isinstance(a_s, WashStep) and isinstance(b_s, WashStep):
            if step_plates(a_s, default_plate) & step_plates(b_s, default_plate):
                consec_reads_or_washes = 1.0
                cand_scores["missing_step"].append((idx + 1, 4.0))
        if isinstance(a_s, IncubateStep) and isinstance(b_s, IncubateStep):
            if step_plates(a_s, default_plate) & step_plates(b_s, default_plate):
                consec_reads_or_washes = 1.0
                cand_scores["missing_step"].append((idx + 1, 4.0))

    has_read = 1.0 if any(isinstance(s, ReadStep) for s in valid_steps) else 0.0
    has_inc = 1.0 if any(isinstance(s, IncubateStep) for s in valid_steps) else 0.0
    has_wash = 1.0 if any(isinstance(s, WashStep) for s in valid_steps) else 0.0

    rule_one_hot = [1.0 if rule_label == lbl else 0.0 for lbl in ALL_VERIFIER_LABELS]

    features: list[float] = [
        1.0,  # bias
        *rule_one_hot,  # 7 rule-baseline class indicators
        1.0 if violations else 0.0,
        min(float(len(violations)), 4.0) / 4.0,
        min(max_vol_peer_log_ratio, 2.5),
        min(max_vol_well_log_ratio, 2.5),
        max_unit_flag,
        max_temp_flag,
        min(max_temp_noncanon_dist, 2.5),
        max_time_bound_flag,
        max_time_sec_flip_score,
        max_time_scale_score,
        bipartite_deficit,
        missing_mix_symmetry,
        consec_reads_or_washes,
        1.0 - has_read,
        has_inc,
        has_wash,
        min(float(n), 25.0) / 25.0,
        min(float(len(deps)), 30.0) / 30.0,
    ]
    return features, cand_scores


def _select_best_step_index(
    pred_label: str,
    raw_steps: Sequence[Step | dict[str, Any]],
    cand_scores: dict[str, list[tuple[int, float]]],
) -> int | None:
    if pred_label == "clean":
        return None
    scored = cand_scores.get(pred_label, [])
    if scored:
        best_by_idx: dict[int, float] = {}
        for idx, sc in scored:
            if idx not in best_by_idx or sc > best_by_idx[idx]:
                best_by_idx[idx] = sc
        best_idx = max(sorted(best_by_idx), key=lambda k: best_by_idx[k])
        return best_idx

    # Fallback by compatible step type
    for idx, item in enumerate(raw_steps):
        stype = item.get("type") if isinstance(item, dict) else item.type
        if pred_label in ("wrong_temperature", "wrong_time") and stype == "incubate":
            return idx
        if pred_label in ("wrong_volume", "wrong_unit") and stype in ("transfer", "mix", "wash"):
            return idx
    return 0


@dataclass
class LearnedProtocolVerifier:
    """Zero-leakage graph-feature classifier and step-level fault localizer."""

    backend: str
    weights: list[list[float]]
    train_protocol_ids: frozenset[str]
    external_model: Any = None

    def predict_error_steps(self, raw_steps: Sequence[Step | dict[str, Any]]) -> dict[str, Any]:
        """Predict ``label``, ``step_index``, and class probabilities from candidate steps."""
        feats, cand_scores = _extract_protocol_and_step_features(raw_steps)
        if self.backend == "stdlib" or self.external_model is None:
            logits = [
                sum(w_j * x_j for w_j, x_j in zip(w_row, feats)) for w_row in self.weights
            ]
            probs = _softmax(logits)
            best_cls = max(range(len(ALL_VERIFIER_LABELS)), key=lambda c: probs[c])
            pred_label = ALL_VERIFIER_LABELS[best_cls]
            conf = probs[best_cls]
        else:
            pred_arr = self.external_model.predict([feats])
            best_cls = int(pred_arr[0])
            pred_label = ALL_VERIFIER_LABELS[best_cls]
            conf = 1.0

        step_idx = _select_best_step_index(pred_label, raw_steps, cand_scores)
        return {
            "label": pred_label,
            "step_index": step_idx,
            "confidence": round(float(conf), 4),
        }

    def predict_record(
        self,
        record: dict[str, Any],
        *,
        enforce_zero_leakage: bool = True,
    ) -> dict[str, Any]:
        """Predict on an ``error_detection`` or ``ordering`` record with leakage guard."""
        pid = str(record.get("protocol_id") or "")
        if enforce_zero_leakage and pid and pid in self.train_protocol_ids:
            raise ValueError(
                f"Data leakage blocked: test record protocol_id {pid!r} was seen in train split!"
            )

        if "ground_truth" in record and isinstance(record["ground_truth"], dict):
            gt = record["ground_truth"]
            task = str(record.get("task") or gt.get("task") or "error_detection")
        else:
            task = str(record.get("task", "error_detection"))

        if task == "ordering":
            return {
                "id": record.get("id"),
                "task": "ordering",
                "protocol_id": record.get("protocol_id"),
                "pred_order": predict_order(record["shuffled_steps"]),
            }

        pred = self.predict_error_steps(record["corrupted_steps"])
        return {
            "id": record.get("id"),
            "task": "error_detection",
            "protocol_id": record.get("protocol_id"),
            "label": pred["label"],
            "step_index": pred["step_index"],
            "confidence": pred["confidence"],
        }


def train_learned_verifier(
    train_records: Sequence[dict[str, Any]],
    *,
    backend: str = "stdlib",
    epochs: int = 180,
    lr: float = 0.25,
    l2: float = 1e-3,
    seed: int = 0,
) -> LearnedProtocolVerifier:
    """Train a ``LearnedProtocolVerifier`` strictly on ``train``-split error-detection records."""
    if backend not in ("stdlib", "sklearn", "lightgbm"):
        raise ValueError(
            f"unsupported backend {backend!r}; expected 'stdlib', 'sklearn', or 'lightgbm'"
        )

    ed_records = [
        r for r in train_records if str(r.get("task", "error_detection")) == "error_detection"
    ]
    if not ed_records:
        raise ValueError("train_learned_verifier requires at least one error_detection record")

    non_train = [
        r.get("protocol_id")
        for r in ed_records
        if r.get("split") not in (None, "train")
    ]
    if non_train:
        raise ValueError(
            f"Refusing to train on non-train split records (found protocol_ids={non_train[:5]!r})"
        )

    train_pids = frozenset(
        str(r["protocol_id"]) for r in ed_records if r.get("protocol_id") is not None
    )
    label_to_idx = {lbl: idx for idx, lbl in enumerate(ALL_VERIFIER_LABELS)}

    xs: list[list[float]] = []
    ys: list[int] = []
    for rec in ed_records:
        lbl = str(rec["label"])
        if lbl not in label_to_idx:
            continue
        feats, _ = _extract_protocol_and_step_features(rec["corrupted_steps"])
        xs.append(feats)
        ys.append(label_to_idx[lbl])

    if not xs:
        raise ValueError("no supported training labels found in train_records")

    n_classes = len(ALL_VERIFIER_LABELS)
    n_feats = len(xs[0])

    # Inverse-sqrt class balancing so rare 'clean' records keep high specificity (balanced score)
    counts = Counter(ys)
    mean_count = len(ys) / max(1, len(counts))
    sample_weights = [math.sqrt(mean_count / max(1, counts[y])) for y in ys]

    weights: list[list[float]] = [[0.0] * n_feats for _ in range(n_classes)]
    ext_model: Any = None

    if backend == "stdlib":
        rng = random.Random(seed)
        indices = list(range(len(xs)))
        for ep in range(epochs):
            rng.shuffle(indices)
            step_lr = lr / (1.0 + 0.015 * ep)
            grads = [[0.0] * n_feats for _ in range(n_classes)]
            total_w = sum(sample_weights)
            for i in indices:
                x = xs[i]
                y = ys[i]
                sw = sample_weights[i] / total_w
                logits = [sum(w_j * x_j for w_j, x_j in zip(w_row, x)) for w_row in weights]
                probs = _softmax(logits)
                for c in range(n_classes):
                    diff = (probs[c] - (1.0 if c == y else 0.0)) * sw
                    g_row = grads[c]
                    for j in range(n_feats):
                        g_row[j] += diff * x[j]
            for c in range(n_classes):
                for j in range(n_feats):
                    reg = l2 * weights[c][j] if j > 0 else 0.0
                    weights[c][j] -= step_lr * (grads[c][j] + reg)
    elif backend == "sklearn":
        sk_linear: Any = importlib.import_module("sklearn.linear_model")
        clf = sk_linear.LogisticRegression(
            max_iter=500, class_weight="balanced", random_state=seed
        )
        clf.fit(xs, ys)
        ext_model = clf
    elif backend == "lightgbm":
        lgb: Any = importlib.import_module("lightgbm")
        clf = lgb.LGBMClassifier(
            n_estimators=80,
            learning_rate=0.08,
            num_leaves=15,
            class_weight="balanced",
            random_state=seed,
            verbose=-1,
        )
        clf.fit(xs, ys)
        ext_model = clf

    return LearnedProtocolVerifier(
        backend=backend,
        weights=weights,
        train_protocol_ids=train_pids,
        external_model=ext_model,
    )


__all__ = [
    "ALL_VERIFIER_LABELS",
    "LearnedProtocolVerifier",
    "train_learned_verifier",
]

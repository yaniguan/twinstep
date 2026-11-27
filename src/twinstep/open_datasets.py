"""Stdlib-only catalog, license guards, and parsers for public wet-lab protocol datasets.

Supports four real open protocol corpora:
1. ``wlp``: Wet Lab Protocols (NAACL 2018 / W-NUT 2020, BRAT ``.txt`` + ``.ann``, MIT).
2. ``bioprobench``: BioProBench ``ERR`` & ``ORD`` tasks (``CC-BY-NC-4.0``, ``eval_only=True``).
3. ``bioplanner``: BioPlanner / ``BioProt`` protocols.io corpus (JSON, ``eval_only=True``).
4. ``opentrons``: Opentrons ``shared-data`` JSON v6-v8 & ``Opentrons/Protocols`` (``Apache-2.0``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any
import urllib.request

from twinstep.dependencies import check_constraints
from twinstep.injectors import OrderingRecord, TwinRecord
from twinstep.schema import (
    IncubateStep,
    MixStep,
    Protocol,
    ReadStep,
    Step,
    TimeUnit,
    TransferStep,
    VolumeUnit,
    WashStep,
)


class DatasetLicenseError(PermissionError):
    """Raised when an evaluation-only or non-commercial dataset is requested for training."""


@dataclass(frozen=True)
class OpenProtocolSource:
    """Metadata and SPDX governance entry for a public protocol benchmark."""

    source_id: str
    name: str
    spdx_license: str
    eval_only: bool
    training_allowed: bool
    homepage: str
    sample_urls: tuple[str, ...]


OPEN_PROTOCOL_CATALOG: dict[str, OpenProtocolSource] = {
    "wlp": OpenProtocolSource(
        source_id="wlp",
        name="Wet Lab Protocols (WLP BRAT Corpus)",
        spdx_license="MIT",
        eval_only=False,
        training_allowed=True,
        homepage="https://github.com/chaitanya2334/WLP-Dataset",
        sample_urls=(
            "https://raw.githubusercontent.com/chaitanya2334/WLP-Dataset/master/test/protocol_103.txt",
            "https://raw.githubusercontent.com/chaitanya2334/WLP-Dataset/master/test/protocol_103.ann",
            "https://raw.githubusercontent.com/chaitanya2334/WLP-Dataset/master/test/protocol_106.txt",
            "https://raw.githubusercontent.com/chaitanya2334/WLP-Dataset/master/test/protocol_106.ann",
            "https://raw.githubusercontent.com/chaitanya2334/WLP-Dataset/master/test/protocol_122.txt",
            "https://raw.githubusercontent.com/chaitanya2334/WLP-Dataset/master/test/protocol_122.ann",
            "https://raw.githubusercontent.com/chaitanya2334/WLP-Dataset/master/test/protocol_132.txt",
            "https://raw.githubusercontent.com/chaitanya2334/WLP-Dataset/master/test/protocol_132.ann",
            "https://raw.githubusercontent.com/chaitanya2334/WLP-Dataset/master/test/protocol_156.txt",
            "https://raw.githubusercontent.com/chaitanya2334/WLP-Dataset/master/test/protocol_156.ann",
            "https://raw.githubusercontent.com/chaitanya2334/WLP-Dataset/master/test/protocol_161.txt",
            "https://raw.githubusercontent.com/chaitanya2334/WLP-Dataset/master/test/protocol_161.ann",
        ),
    ),
    "bioprobench": OpenProtocolSource(
        source_id="bioprobench",
        name="BioProBench (ERR & ORD Protocol Reasoning Benchmark)",
        spdx_license="CC-BY-NC-4.0",
        eval_only=True,
        training_allowed=False,
        homepage="https://github.com/YuyangSunshine/bioprobench",
        sample_urls=(
            "https://raw.githubusercontent.com/YuyangSunshine/bioprobench/main/Data/ERR_test.json",
            "https://raw.githubusercontent.com/YuyangSunshine/bioprobench/main/Data/ORD_test.json",
        ),
    ),
    "bioplanner": OpenProtocolSource(
        source_id="bioplanner",
        name="BioPlanner / BioProt (protocols.io Evaluation Suite)",
        spdx_license="MIT",
        eval_only=True,
        training_allowed=False,
        homepage="https://github.com/bioplanner/bioplanner",
        sample_urls=(
            "https://raw.githubusercontent.com/bioplanner/bioplanner/main/bioprot/10051.json",
            "https://raw.githubusercontent.com/bioplanner/bioplanner/main/bioprot/1007.json",
            "https://raw.githubusercontent.com/bioplanner/bioplanner/main/bioprot/10157.json",
            "https://raw.githubusercontent.com/bioplanner/bioplanner/main/bioprot/10279.json",
            "https://raw.githubusercontent.com/bioplanner/bioplanner/main/bioprot/10606.json",
            "https://raw.githubusercontent.com/bioplanner/bioplanner/main/bioprot/1073.json",
        ),
    ),
    "opentrons": OpenProtocolSource(
        source_id="opentrons",
        name="Opentrons shared-data JSON v6-v8 & Protocol Library",
        spdx_license="Apache-2.0",
        eval_only=False,
        training_allowed=True,
        homepage="https://github.com/Opentrons/opentrons",
        sample_urls=(
            "https://raw.githubusercontent.com/Opentrons/opentrons/edge/shared-data/protocol/fixtures/6/simpleV6.json",
            "https://raw.githubusercontent.com/Opentrons/opentrons/edge/shared-data/protocol/fixtures/6/tempAndMagModuleCommands.json",
            "https://raw.githubusercontent.com/Opentrons/opentrons/edge/shared-data/protocol/fixtures/6/heaterShakerCommands.json",
            "https://raw.githubusercontent.com/Opentrons/opentrons/edge/shared-data/protocol/fixtures/6/transferSettings.json",
            "https://raw.githubusercontent.com/Opentrons/opentrons/edge/shared-data/protocol/fixtures/7/simpleV7.json",
            "https://raw.githubusercontent.com/Opentrons/opentrons/edge/shared-data/protocol/fixtures/7/moveLabware.json",
            "https://raw.githubusercontent.com/Opentrons/opentrons/edge/shared-data/protocol/fixtures/8/simpleV8.json",
            "https://raw.githubusercontent.com/Opentrons/opentrons/edge/shared-data/protocol/fixtures/8/simpleFlexV8.json",
            "https://raw.githubusercontent.com/Opentrons/Protocols/develop/protoBuilds/00222e/00222e.ot2.apiv2.py.json",
            "https://raw.githubusercontent.com/Opentrons/Protocols/develop/protoBuilds/002af3/002af3.ot2.apiv2.py.json",
        ),
    ),
}

_VOL_RE = re.compile(
    r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>uL|ul|\u03bcL|\u03bcl|\u00b5L|\u00b5l|mL|ml)\b"
)
_TEMP_RE = re.compile(r"(?P<temp>-?\d+(?:\.\d+)?)\s*(?:\^\s*)?(?:\u00b0\s*C|deg\s*C|C)\b")
_TIME_RE = re.compile(
    r"(?P<time>\d+(?:\.\d+)?)\s*(?P<unit>hours?|hrs?|hr|h|minutes?|mins?|min|seconds?|secs?|sec|s)\b",
    re.IGNORECASE,
)
_WL_RE = re.compile(r"(?P<wl>\d{3,4})\s*nm\b", re.IGNORECASE)


def verify_dataset_governance(source_id: str, *, for_training: bool = False) -> OpenProtocolSource:
    """Validate dataset catalog entry and block training use when ``eval_only=True``."""
    if source_id not in OPEN_PROTOCOL_CATALOG:
        raise ValueError(
            f"unknown open dataset {source_id!r}; expected one of {sorted(OPEN_PROTOCOL_CATALOG)}"
        )
    entry = OPEN_PROTOCOL_CATALOG[source_id]
    if entry.eval_only and entry.training_allowed:
        raise ValueError(f"inconsistent catalog flags on {source_id!r}")
    if for_training and (entry.eval_only or not entry.training_allowed):
        raise DatasetLicenseError(
            f"dataset {source_id!r} ({entry.spdx_license}, eval_only={entry.eval_only}) "
            "cannot be used for model training"
        )
    return entry


def sha256_bytes(payload: bytes) -> str:
    """Compute lowercase hexadecimal SHA-256 digest of ``payload``."""
    return hashlib.sha256(payload).hexdigest()


def download_url_cached(
    url: str,
    dest_path: Path,
    *,
    expected_sha256: str | None = None,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Download ``url`` to ``dest_path`` (if absent) and verify SHA-256."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.is_file():
        data = dest_path.read_bytes()
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "twinstep-open-datasets/0.1"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read()
        dest_path.write_bytes(data)

    digest = sha256_bytes(data)
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError(
            f"SHA-256 mismatch for {dest_path}: expected {expected_sha256}, got {digest}"
        )
    return {
        "url": url,
        "path": str(dest_path),
        "bytes": len(data),
        "sha256": digest,
    }


def _extract_vol_ul_or_ml(text: str, default_ul: float = 100.0) -> tuple[float, VolumeUnit]:
    m = _VOL_RE.search(text)
    if m is None:
        return default_ul, "uL"
    val = float(m.group("val"))
    if val <= 0:
        return default_ul, "uL"
    raw_u = m.group("unit").lower()
    if raw_u == "ml":
        if val <= 0.25:
            return round(val * 1000.0, 3), "uL"
        return min(val, 2.0), "mL"
    return min(max(val, 1.0), 300.0), "uL"


def _extract_temp_c(text: str, default_c: float = 37.0) -> float:
    m = _TEMP_RE.search(text)
    if m is None:
        low = text.lower()
        if "ice" in low:
            return 4.0
        if "room temp" in low or "rt" in low.split():
            return 25.0
        return default_c
    t = float(m.group("temp"))
    return min(max(t, 4.0), 95.0)


def _extract_time(text: str, default_min: float = 15.0) -> tuple[float, TimeUnit]:
    if "overnight" in text.lower():
        return 960.0, "min"
    m = _TIME_RE.search(text)
    if m is None:
        return default_min, "min"
    val = float(m.group("time"))
    if val <= 0:
        return default_min, "min"
    u_low = m.group("unit").lower()
    if u_low.startswith("h"):
        return min(val * 60.0, 4320.0), "min"
    if u_low.startswith("s"):
        if val > 600.0:
            return round(val / 60.0, 2), "min"
        return max(val, 5.0), "s"
    return min(max(val, 1.0), 4320.0), "min"


def _sanitize_slug(raw: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    return slug[:28] or "reagent"


def _sentences_to_steps(
    protocol_id: str, sentences: Sequence[str], split: str = "test"
) -> Protocol:
    """Lower ordered wet-lab step sentences into a validated multi-step ``Protocol``."""
    steps: list[Step] = []
    well_filled = False
    stage_fills = 0
    stage_mixed = False

    def _flush_unmixed_before_barrier() -> None:
        nonlocal stage_fills, stage_mixed
        if stage_fills >= 2 and not stage_mixed:
            steps.append(MixStep(well="plate1:A1", volume=50.0, cycles=4, unit="uL"))
        stage_fills = 0
        stage_mixed = False

    for idx, raw_line in enumerate(sentences):
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()

        if any(w in low for w in ("wash", "rinse")):
            if not well_filled:
                steps.append(
                    TransferStep(
                        source="reservoir:sample",
                        dest="plate1:A1",
                        volume=100.0,
                        unit="uL",
                    )
                )
                steps.append(
                    IncubateStep(plate="plate1", temperature=25.0, time=10.0, time_unit="min")
                )
                well_filled = True
                stage_fills = 0
                stage_mixed = False
            elif stage_fills > 0 and not isinstance(steps[-1], (IncubateStep, ReadStep, WashStep)):
                _flush_unmixed_before_barrier()
                steps.append(
                    IncubateStep(plate="plate1", temperature=25.0, time=10.0, time_unit="min")
                )
            vol_w, _ = _extract_vol_ul_or_ml(line, default_ul=300.0)
            vol_wash = min(max(vol_w if vol_w >= 50.0 else 300.0, 50.0), 1000.0)
            steps.append(WashStep(plate="plate1", volume=vol_wash, cycles=3, unit="uL"))
            stage_fills = 0
            stage_mixed = False
        elif any(w in low for w in ("incubat", "denatur", "anneal", "thermocyc", "overnight")):
            if not well_filled:
                steps.append(
                    TransferStep(
                        source="reservoir:reaction_mix",
                        dest="plate1:A1",
                        volume=100.0,
                        unit="uL",
                    )
                )
                well_filled = True
                stage_fills = 1
            _flush_unmixed_before_barrier()
            temp_c = _extract_temp_c(line, default_c=37.0)
            dur, t_unit = _extract_time(line, default_min=15.0)
            steps.append(
                IncubateStep(
                    plate="plate1",
                    temperature=temp_c,
                    time=dur,
                    time_unit=t_unit,
                )
            )
        elif any(w in low for w in ("mix", "vortex", "resuspend", "homogeniz", "invert")):
            if not well_filled:
                steps.append(
                    TransferStep(
                        source="reservoir:buffer",
                        dest="plate1:A1",
                        volume=100.0,
                        unit="uL",
                    )
                )
                well_filled = True
                stage_fills = 1
            elif stage_fills == 0 and isinstance(steps[-1], (IncubateStep, WashStep)):
                steps.append(
                    TransferStep(
                        source=f"reservoir:reagent_{idx}",
                        dest="plate1:A1",
                        volume=50.0,
                        unit="uL",
                    )
                )
                stage_fills = 1
            vol_m, _ = _extract_vol_ul_or_ml(line, default_ul=50.0)
            steps.append(
                MixStep(
                    well="plate1:A1",
                    volume=min(max(vol_m, 10.0), 200.0),
                    cycles=5,
                    unit="uL",
                )
            )
            stage_mixed = True
        elif any(
            w in low
            for w in ("read", "absorbance", "fluorescen", "luminescen", "od600", "spectrophotom")
        ):
            if not well_filled:
                steps.append(
                    TransferStep(
                        source="reservoir:sample",
                        dest="plate1:A1",
                        volume=100.0,
                        unit="uL",
                    )
                )
                well_filled = True
                stage_fills = 1
            _flush_unmixed_before_barrier()
            wl_m = _WL_RE.search(line)
            wl = int(wl_m.group("wl")) if wl_m is not None else 450
            wl = min(max(wl, 220), 900)
            mode = "fluorescence" if "fluores" in low else "absorbance"
            steps.append(ReadStep(plate="plate1", mode=mode, wavelength=wl))
        elif any(
            w in low
            for w in ("add", "transfer", "pipet", "dispens", "aliquot", "inoculat", "dilut")
        ) or _VOL_RE.search(line):
            vol_t, unit_t = _extract_vol_ul_or_ml(line, default_ul=50.0)
            if unit_t == "mL":
                vol_t = min(vol_t * 100.0, 200.0)
                unit_t = "uL"
            stop_words = {
                "add",
                "the",
                "into",
                "with",
                "well",
                "plate",
                "from",
                "transfer",
                "pipette",
            }
            words = [
                w for w in re.findall(r"[A-Za-z]{3,}", line) if w.lower() not in stop_words
            ]
            reagent_slug = _sanitize_slug("_".join(words[:2]) if words else f"reagent_{idx}")
            if stage_fills >= 1 and stage_mixed:
                steps.append(
                    IncubateStep(plate="plate1", temperature=25.0, time=5.0, time_unit="min")
                )
                stage_fills = 0
                stage_mixed = False
            steps.append(
                TransferStep(
                    source=f"reservoir:{reagent_slug}",
                    dest="plate1:A1",
                    volume=vol_t,
                    unit="uL",
                )
            )
            well_filled = True
            stage_fills += 1
            stage_mixed = False

        if len(steps) >= 14:
            break

    if not well_filled:
        steps.append(
            TransferStep(source="reservoir:sample", dest="plate1:A1", volume=100.0, unit="uL")
        )
        stage_fills = 1
    _flush_unmixed_before_barrier()
    if not any(isinstance(s, ReadStep) for s in steps):
        steps.append(ReadStep(plate="plate1", mode="absorbance", wavelength=450))
    if len(steps) < 3:
        steps.insert(
            1, IncubateStep(plate="plate1", temperature=25.0, time=10.0, time_unit="min")
        )
    return Protocol(protocol_id=protocol_id, steps=tuple(steps), split=split)


def parse_wlp_brat_protocol(
    protocol_id: str,
    txt_text: str,
    ann_text: str,
    *,
    split: str = "test",
) -> Protocol:
    """Parse a WLP BRAT standoff pair (``.txt`` + ``.ann``) into a ``twinstep.Protocol``."""
    action_offsets: list[tuple[int, str]] = []
    for raw_line in ann_text.splitlines():
        if not raw_line.startswith("T") or "\t" not in raw_line:
            continue
        parts = raw_line.split("\t")
        if len(parts) < 3:
            continue
        meta = parts[1].split()
        if len(meta) >= 3 and meta[0] == "Action":
            try:
                start_char = int(meta[1])
            except ValueError:
                continue
            action_offsets.append((start_char, parts[2].strip()))

    sentences = [ln.strip() for ln in txt_text.splitlines() if ln.strip()]
    if action_offsets and not sentences:
        sentences = [verb for _, verb in sorted(action_offsets)]
    return _sentences_to_steps(f"wlp_{protocol_id}", sentences, split=split)


def parse_bioprot_json(data: dict[str, Any], *, split: str = "test") -> Protocol:
    """Parse a BioPlanner ``BioProt`` JSON document into a ``twinstep.Protocol``."""
    verify_dataset_governance("bioplanner", for_training=(split == "train"))
    pid = f"bioprot_{data.get('id', 'unknown')}"
    raw_steps = data.get("steps")
    if isinstance(raw_steps, str):
        lines = [ln.strip() for ln in raw_steps.splitlines() if ln.strip()]
    elif isinstance(raw_steps, list):
        lines = [str(item) for item in raw_steps]
    else:
        lines = [str(data.get("original description") or "")]
    return _sentences_to_steps(pid, lines, split=split)


def parse_opentrons_json_protocol(
    data: dict[str, Any],
    protocol_id: str = "opentrons_protocol",
    *,
    split: str = "test",
) -> Protocol:
    """Parse an Opentrons JSON v6-v8 command fixture or Protocol Library build into ``Protocol``."""
    # Support Opentrons/Protocols protoBuild JSON (`content` Python API v2 source).
    if "commands" not in data and isinstance(data.get("content"), str):
        lines = [ln.strip() for ln in str(data["content"]).splitlines() if ln.strip()]
        return _sentences_to_steps(f"ot_lib_{protocol_id}", lines, split=split)

    commands = data.get("commands")
    if not isinstance(commands, list):
        raise ValueError("Opentrons JSON protocol requires a 'commands' list or 'content' source")

    labware_roles: dict[str, str] = {}
    liquid_loaded_labware: set[str] = set()
    for cmd in commands:
        if not isinstance(cmd, dict):
            continue
        ctype = str(cmd.get("commandType") or "")
        raw_params = cmd.get("params")
        params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
        if ctype == "loadLabware":
            lw_id = str(params.get("labwareId") or "")
            load_name = str(params.get("loadName") or lw_id).lower()
            if "trash" in load_name or "trash" in lw_id.lower():
                labware_roles[lw_id] = "waste"
            elif "reservoir" in load_name or "trough" in load_name or "source" in lw_id.lower():
                labware_roles[lw_id] = "reservoir"
            else:
                labware_roles[lw_id] = "plate1"
        elif ctype == "loadLiquid":
            lw_id = str(params.get("labwareId") or "")
            if lw_id:
                liquid_loaded_labware.add(lw_id)

    steps: list[Step] = []
    pending_aspirate: tuple[str, str, float] | None = None
    filled_wells: set[str] = set()
    stage_fills: dict[str, int] = {}
    stage_mixed: dict[str, bool] = {}

    def _map_lw_well(lw_id: str, well_name: str, is_source: bool) -> str:
        role = labware_roles.get(lw_id, "plate1")
        w_clean = well_name.strip() or "A1"
        if role == "waste":
            return f"waste:{w_clean.lower()}"
        is_unfilled_src = (
            is_source
            and lw_id in liquid_loaded_labware
            and f"plate1:{w_clean.upper()}" not in filled_wells
        )
        if role == "reservoir" or is_unfilled_src:
            return f"reservoir:{_sanitize_slug(lw_id)}_{w_clean.lower()}"
        return f"plate1:{w_clean.upper()}"

    def _flush_stage_mixes() -> None:
        for w_key, cnt in list(stage_fills.items()):
            if cnt >= 2 and not stage_mixed.get(w_key, False):
                steps.append(MixStep(well=w_key, volume=20.0, cycles=3, unit="uL"))
            stage_fills[w_key] = 0
            stage_mixed[w_key] = False

    for cmd in commands:
        if not isinstance(cmd, dict):
            continue
        ctype = str(cmd.get("commandType") or "")
        raw_params = cmd.get("params")
        params = raw_params if isinstance(raw_params, dict) else {}

        if ctype == "aspirate":
            lw_id = str(params.get("labwareId") or "source")
            w_name = str(params.get("wellName") or "A1")
            vol = float(params.get("volume") or 20.0)
            pending_aspirate = (lw_id, w_name, min(max(vol, 1.0), 300.0))
        elif ctype == "dispense" and pending_aspirate is not None:
            src_lw, src_w, asp_vol = pending_aspirate
            dst_lw = str(params.get("labwareId") or "plate1")
            dst_w = str(params.get("wellName") or "A1")
            disp_vol = min(max(float(params.get("volume") or asp_vol), 1.0), 300.0)
            pending_aspirate = None

            if src_lw == dst_lw and src_w == dst_w:
                target_w = f"plate1:{dst_w.upper()}"
                if target_w not in filled_wells:
                    steps.append(
                        TransferStep(
                            source=f"reservoir:{_sanitize_slug(src_lw)}",
                            dest=target_w,
                            volume=disp_vol,
                            unit="uL",
                        )
                    )
                    filled_wells.add(target_w)
                    stage_fills[target_w] = 1
                steps.append(MixStep(well=target_w, volume=disp_vol, cycles=3, unit="uL"))
                stage_mixed[target_w] = True
            else:
                src_ref = _map_lw_well(src_lw, src_w, is_source=True)
                dst_ref = _map_lw_well(dst_lw, dst_w, is_source=False)
                if src_ref == dst_ref:
                    src_ref = f"reservoir:{_sanitize_slug(src_lw)}"
                if dst_ref.startswith("plate1:") and stage_mixed.get(dst_ref, False):
                    steps.append(
                        IncubateStep(plate="plate1", temperature=25.0, time=1.0, time_unit="min")
                    )
                    stage_fills[dst_ref] = 0
                    stage_mixed[dst_ref] = False
                steps.append(
                    TransferStep(source=src_ref, dest=dst_ref, volume=disp_vol, unit="uL")
                )
                if dst_ref.startswith("plate1:"):
                    filled_wells.add(dst_ref)
                    stage_fills[dst_ref] = stage_fills.get(dst_ref, 0) + 1
                    stage_mixed[dst_ref] = False
        elif ctype in ("delay", "waitForDuration"):
            if filled_wells:
                _flush_stage_mixes()
                secs = float(params.get("seconds") or (float(params.get("minutes") or 1.0) * 60.0))
                if secs > 600.0:
                    steps.append(
                        IncubateStep(
                            plate="plate1",
                            temperature=25.0,
                            time=min(round(secs / 60.0, 2), 4320.0),
                            time_unit="min",
                        )
                    )
                else:
                    steps.append(
                        IncubateStep(
                            plate="plate1",
                            temperature=25.0,
                            time=max(secs, 5.0),
                            time_unit="s",
                        )
                    )
        elif ctype in (
            "temperatureModule/setTargetTemperature",
            "heaterShaker/setTargetTemperature",
        ):
            if filled_wells:
                _flush_stage_mixes()
                celsius = min(max(float(params.get("celsius") or 37.0), 4.0), 95.0)
                steps.append(
                    IncubateStep(
                        plate="plate1",
                        temperature=celsius,
                        time=5.0,
                        time_unit="min",
                    )
                )
        elif "absorbanceReader/read" in ctype:
            if not filled_wells:
                steps.append(
                    TransferStep(
                        source="reservoir:sample", dest="plate1:A1", volume=100.0, unit="uL"
                    )
                )
                filled_wells.add("plate1:A1")
            _flush_stage_mixes()
            steps.append(ReadStep(plate="plate1", mode="absorbance", wavelength=450))

    if not filled_wells:
        steps.append(
            TransferStep(source="reservoir:reagent", dest="plate1:A1", volume=50.0, unit="uL")
        )
        steps.append(MixStep(well="plate1:A1", volume=30.0, cycles=3, unit="uL"))
    else:
        _flush_stage_mixes()

    if not any(isinstance(s, ReadStep) for s in steps):
        steps.append(ReadStep(plate="plate1", mode="absorbance", wavelength=450))
    if len(steps) < 2:
        steps.insert(0, TransferStep("reservoir:buf", "plate1:A1", 50.0, "uL"))

    return Protocol(protocol_id=protocol_id, steps=tuple(steps), split=split)


def parse_bioprobench_err_records(
    raw_items: Sequence[dict[str, Any]],
    *,
    max_records: int = 200,
    for_training: bool = False,
) -> list[dict[str, Any]]:
    """Convert quantitative liquid-handling BioProBench ``ERR`` items into ``TwinRecord`` dicts.

    Enforces ``CC-BY-NC-4.0`` ``eval_only=True`` governance (raises ``DatasetLicenseError``
    when ``for_training=True``).
    """
    verify_dataset_governance("bioprobench", for_training=for_training)
    records: list[dict[str, Any]] = []

    for item in raw_items:
        if len(records) >= max_records:
            break
        rec_id = str(item.get("id") or f"bioprobench_err_{len(records)}")
        corrupted_txt = str(item.get("corrupted_text") or "")
        corrected_txt = str(item.get("corrected_text") or corrupted_txt)
        is_correct = bool(item.get("is_correct", False))

        corr_vol = _VOL_RE.search(corrupted_txt)
        gold_vol = _VOL_RE.search(corrected_txt)
        corr_temp = _TEMP_RE.search(corrupted_txt)
        gold_temp = _TEMP_RE.search(corrected_txt)
        corr_time = _TIME_RE.search(corrupted_txt)
        gold_time = _TIME_RE.search(corrected_txt)

        if is_correct:
            v_val, _ = _extract_vol_ul_or_ml(corrected_txt, default_ul=100.0)
            t_val = _extract_temp_c(corrected_txt, default_c=37.0)
            d_val, d_unit = _extract_time(corrected_txt, default_min=15.0)
            clean_steps = [
                TransferStep("reservoir:reagent", "plate1:A1", v_val, "uL").to_dict(),
                IncubateStep(
                    temperature=t_val,
                    time=d_val,
                    time_unit=d_unit,
                    plate="plate1",
                ).to_dict(),
                ReadStep("plate1", "absorbance", 450).to_dict(),
            ]
            records.append(
                TwinRecord(
                    id=rec_id,
                    protocol_id=f"bioprobench_{rec_id}",
                    correct_steps=clean_steps,
                    corrupted_steps=[dict(s) for s in clean_steps],
                    label="clean",
                    step_index=None,
                    original_value=None,
                    new_value=None,
                    description="BioProBench clean protocol step.",
                    split="test",
                ).to_dict()
            )
            continue

        # Quantitative parameter differences (volume / unit / temperature / duration)
        if corr_vol is not None and gold_vol is not None and corr_vol.group(0) != gold_vol.group(0):
            cv = float(corr_vol.group("val"))
            gv = float(gold_vol.group("val"))
            cu: Any = "mL" if corr_vol.group("unit").lower() == "ml" else "uL"
            gu: Any = "mL" if gold_vol.group("unit").lower() == "ml" else "uL"
            if cv <= 0 or gv <= 0:
                continue
            label = "wrong_unit" if (cu != gu and abs(cv - gv) < 1e-6) else "wrong_volume"
            gold_s = [
                TransferStep("reservoir:reagent", "plate1:A1", min(gv, 250.0), gu).to_dict(),
                ReadStep("plate1", "absorbance", 450).to_dict(),
            ]
            corr_s = [
                TransferStep("reservoir:reagent", "plate1:A1", cv, cu).to_dict(),
                ReadStep("plate1", "absorbance", 450).to_dict(),
            ]
            if corr_s != gold_s:
                records.append(
                    TwinRecord(
                        id=rec_id,
                        protocol_id=f"bioprobench_{rec_id}",
                        correct_steps=gold_s,
                        corrupted_steps=corr_s,
                        label=label,
                        step_index=0,
                        original_value=gv if label == "wrong_volume" else gu,
                        new_value=cv if label == "wrong_volume" else cu,
                        description=str(
                            item.get("error_description") or "BioProBench volume/unit fault."
                        ),
                        split="test",
                    ).to_dict()
                )
        elif (
            corr_temp is not None
            and gold_temp is not None
            and corr_temp.group("temp") != gold_temp.group("temp")
        ):
            ct = float(corr_temp.group("temp"))
            gt = float(gold_temp.group("temp"))
            if ct <= -273.15 or gt <= -273.15 or ct == gt:
                continue
            gold_s = [
                TransferStep("reservoir:reagent", "plate1:A1", 100.0, "uL").to_dict(),
                IncubateStep(temperature=gt, time=15.0, time_unit="min", plate="plate1").to_dict(),
                ReadStep("plate1", "absorbance", 450).to_dict(),
            ]
            corr_s = [
                TransferStep("reservoir:reagent", "plate1:A1", 100.0, "uL").to_dict(),
                IncubateStep(temperature=ct, time=15.0, time_unit="min", plate="plate1").to_dict(),
                ReadStep("plate1", "absorbance", 450).to_dict(),
            ]
            records.append(
                TwinRecord(
                    id=rec_id,
                    protocol_id=f"bioprobench_{rec_id}",
                    correct_steps=gold_s,
                    corrupted_steps=corr_s,
                    label="wrong_temperature",
                    step_index=1,
                    original_value=gt,
                    new_value=ct,
                    description=str(
                        item.get("error_description") or "BioProBench temperature fault."
                    ),
                    split="test",
                ).to_dict()
            )
        elif (
            corr_time is not None
            and gold_time is not None
            and corr_time.group(0).lower() != gold_time.group(0).lower()
        ):
            cd, cu_t = _extract_time(corr_time.group(0), default_min=15.0)
            gd, gu_t = _extract_time(gold_time.group(0), default_min=15.0)
            if (cd, cu_t) == (gd, gu_t):
                continue
            gold_s = [
                TransferStep("reservoir:reagent", "plate1:A1", 100.0, "uL").to_dict(),
                IncubateStep(
                    temperature=37.0, time=gd, time_unit=gu_t, plate="plate1"
                ).to_dict(),
                ReadStep("plate1", "absorbance", 450).to_dict(),
            ]
            corr_s = [
                TransferStep("reservoir:reagent", "plate1:A1", 100.0, "uL").to_dict(),
                IncubateStep(
                    temperature=37.0, time=cd, time_unit=cu_t, plate="plate1"
                ).to_dict(),
                ReadStep("plate1", "absorbance", 450).to_dict(),
            ]
            records.append(
                TwinRecord(
                    id=rec_id,
                    protocol_id=f"bioprobench_{rec_id}",
                    correct_steps=gold_s,
                    corrupted_steps=corr_s,
                    label="wrong_time",
                    step_index=1,
                    original_value=f"{gd:g} {gu_t}",
                    new_value=f"{cd:g} {cu_t}",
                    description=str(item.get("error_description") or "BioProBench duration fault."),
                    split="test",
                ).to_dict()
            )

    return records


def parse_bioprobench_ord_records(
    raw_items: Sequence[dict[str, Any]],
    *,
    max_records: int = 100,
    for_training: bool = False,
) -> list[dict[str, Any]]:
    """Convert BioProBench ``ORD`` items into ``twinstep`` ``OrderingRecord`` dicts."""
    verify_dataset_governance("bioprobench", for_training=for_training)
    out: list[dict[str, Any]] = []

    for item in raw_items:
        if len(out) >= max_records:
            break
        wrong = item.get("wrong_steps")
        correct = item.get("correct_steps")
        if not isinstance(wrong, list) or not isinstance(correct, list):
            continue
        if len(wrong) != len(correct) or len(correct) < 2 or len(set(correct)) != len(correct):
            continue
        if set(wrong) != set(correct):
            continue

        # Map each distinct step text to a distinct typed liquid-handling step in gold order.
        n = len(correct)
        typed_gold: list[dict[str, Any]] = []
        for idx, text in enumerate(correct):
            low = str(text).lower()
            if idx == n - 1 and any(
                w in low for w in ("detect", "read", "analy", "measure", "quant", "image", "elisa")
            ):
                typed_gold.append(ReadStep("plate1", "absorbance", 450).to_dict())
            elif "wash" in low and idx > 0:
                typed_gold.append(
                    WashStep(plate="plate1", volume=300.0, cycles=idx + 1, unit="uL").to_dict()
                )
            elif any(w in low for w in ("incubat", "cultur", "proliferat")) and idx > 0:
                typed_gold.append(
                    IncubateStep(
                        plate="plate1",
                        temperature=37.0,
                        time=float(max(5, (n - idx) * 15)),
                        time_unit="min",
                    ).to_dict()
                )
            else:
                slug = _sanitize_slug(str(text))
                typed_gold.append(
                    TransferStep(
                        source=f"reservoir:{slug}_{idx}",
                        dest="plate1:A1",
                        volume=float(20 + idx * 10),
                        unit="uL",
                    ).to_dict()
                )

        text_to_gold_idx = {str(txt): i for i, txt in enumerate(correct)}
        shuffled_dicts = [typed_gold[text_to_gold_idx[str(w)]] for w in wrong]
        gold_to_shuffled = {text_to_gold_idx[str(w)]: s_i for s_i, w in enumerate(wrong)}
        gold_order = [gold_to_shuffled[i] for i in range(n)]
        constraints = [[gold_order[i], gold_order[i + 1]] for i in range(n - 1)]
        if check_constraints(list(range(n)), [(c[0], c[1]) for c in constraints]):
            continue

        rec_id = str(item.get("id") or f"bioprobench_ord_{len(out)}")
        out.append(
            OrderingRecord(
                id=rec_id,
                protocol_id=f"bioprobench_{rec_id}",
                correct_steps=typed_gold,
                shuffled_steps=shuffled_dicts,
                gold_order=gold_order,
                constraints=constraints,
                split="test",
            ).to_dict()
        )
    return out


__all__ = [
    "DatasetLicenseError",
    "OPEN_PROTOCOL_CATALOG",
    "OpenProtocolSource",
    "download_url_cached",
    "parse_bioprobench_err_records",
    "parse_bioprobench_ord_records",
    "parse_bioprot_json",
    "parse_opentrons_json_protocol",
    "parse_wlp_brat_protocol",
    "sha256_bytes",
    "verify_dataset_governance",
]

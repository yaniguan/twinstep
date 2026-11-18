"""Validated dataclass schema and JSON/YAML loaders for generic liquid-handling steps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Literal, Sequence, Union

VolumeUnit = Literal["uL", "mL"]
TimeUnit = Literal["min", "s"]

VALID_VOLUME_UNITS = ("uL", "mL")
VALID_TIME_UNITS = ("min", "s")


def _require_nonempty_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    val = float(value)
    if not math.isfinite(val) or val <= 0:
        raise ValueError(f"{field} must be a positive finite number")
    return val


def _require_finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{field} must be finite")
    return val


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True)
class TransferStep:
    """Liquid transfer from a source location to a destination location."""

    source: str
    dest: str
    volume: float
    unit: VolumeUnit = "uL"
    type: Literal["transfer"] = "transfer"

    def __post_init__(self) -> None:
        _require_nonempty_str(self.source, "source")
        _require_nonempty_str(self.dest, "dest")
        object.__setattr__(self, "volume", _require_positive_number(self.volume, "volume"))
        if self.unit not in VALID_VOLUME_UNITS:
            raise ValueError(f"unit must be one of {VALID_VOLUME_UNITS}, got {self.unit!r}")
        if self.type != "transfer":
            raise ValueError(f"invalid type for TransferStep: {self.type!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "source": self.source,
            "dest": self.dest,
            "volume": self.volume,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class MixStep:
    """Aspirate/dispense mixing within a single well."""

    well: str
    volume: float
    cycles: int
    unit: VolumeUnit = "uL"
    type: Literal["mix"] = "mix"

    def __post_init__(self) -> None:
        _require_nonempty_str(self.well, "well")
        object.__setattr__(self, "volume", _require_positive_number(self.volume, "volume"))
        _require_positive_int(self.cycles, "cycles")
        if self.unit not in VALID_VOLUME_UNITS:
            raise ValueError(f"unit must be one of {VALID_VOLUME_UNITS}, got {self.unit!r}")
        if self.type != "mix":
            raise ValueError(f"invalid type for MixStep: {self.type!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "well": self.well,
            "volume": self.volume,
            "unit": self.unit,
            "cycles": self.cycles,
        }


@dataclass(frozen=True)
class IncubateStep:
    """Timed temperature incubation step."""

    temperature: float
    time: float
    time_unit: TimeUnit = "min"
    plate: str | None = None
    type: Literal["incubate"] = "incubate"

    def __post_init__(self) -> None:
        temp = _require_finite_number(self.temperature, "temperature")
        if temp <= -273.15:
            raise ValueError("temperature must be above -273.15 C")
        object.__setattr__(self, "temperature", temp)
        object.__setattr__(self, "time", _require_positive_number(self.time, "time"))
        if self.time_unit not in VALID_TIME_UNITS:
            raise ValueError(f"time_unit must be one of {VALID_TIME_UNITS}, got {self.time_unit!r}")
        if self.plate is not None:
            _require_nonempty_str(self.plate, "plate")
        if self.type != "incubate":
            raise ValueError(f"invalid type for IncubateStep: {self.type!r}")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": self.type,
            "temperature": self.temperature,
            "time": self.time,
            "time_unit": self.time_unit,
        }
        if self.plate is not None:
            data["plate"] = self.plate
        return data


@dataclass(frozen=True)
class WashStep:
    """Plate or well wash step with repeated volume cycles."""

    volume: float
    cycles: int
    well: str | None = None
    plate: str | None = None
    unit: VolumeUnit = "uL"
    type: Literal["wash"] = "wash"

    def __post_init__(self) -> None:
        if self.well is None and self.plate is None:
            raise ValueError("wash step requires at least one of 'well' or 'plate'")
        if self.well is not None:
            _require_nonempty_str(self.well, "well")
        if self.plate is not None:
            _require_nonempty_str(self.plate, "plate")
        object.__setattr__(self, "volume", _require_positive_number(self.volume, "volume"))
        _require_positive_int(self.cycles, "cycles")
        if self.unit not in VALID_VOLUME_UNITS:
            raise ValueError(f"unit must be one of {VALID_VOLUME_UNITS}, got {self.unit!r}")
        if self.type != "wash":
            raise ValueError(f"invalid type for WashStep: {self.type!r}")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.type}
        if self.plate is not None:
            data["plate"] = self.plate
        if self.well is not None:
            data["well"] = self.well
        data["volume"] = self.volume
        data["unit"] = self.unit
        data["cycles"] = self.cycles
        return data


@dataclass(frozen=True)
class ReadStep:
    """Plate reader measurement step."""

    plate: str
    mode: str
    wavelength: int
    type: Literal["read"] = "read"

    def __post_init__(self) -> None:
        _require_nonempty_str(self.plate, "plate")
        _require_nonempty_str(self.mode, "mode")
        _require_positive_int(self.wavelength, "wavelength")
        if self.type != "read":
            raise ValueError(f"invalid type for ReadStep: {self.type!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "plate": self.plate,
            "mode": self.mode,
            "wavelength": self.wavelength,
        }


Step = Union[TransferStep, MixStep, IncubateStep, WashStep, ReadStep]


@dataclass(frozen=True)
class Protocol:
    """Validated named sequence of liquid-handling steps."""

    protocol_id: str
    steps: tuple[Step, ...]

    def __post_init__(self) -> None:
        _require_nonempty_str(self.protocol_id, "protocol_id")
        if not self.steps:
            raise ValueError("protocol must contain at least one step")

    def to_dicts(self) -> list[dict[str, Any]]:
        return [step.to_dict() for step in self.steps]


def step_from_dict(data: dict[str, Any]) -> Step:
    """Parse and validate a single step dictionary into its dataclass."""
    if not isinstance(data, dict):
        raise ValueError("each step must be a mapping")
    step_type = data.get("type")
    if step_type == "transfer":
        return TransferStep(
            source=data.get("source"),  # type: ignore[arg-type]
            dest=data.get("dest"),  # type: ignore[arg-type]
            volume=data.get("volume"),  # type: ignore[arg-type]
            unit=data.get("unit", "uL"),
        )
    if step_type == "mix":
        return MixStep(
            well=data.get("well"),  # type: ignore[arg-type]
            volume=data.get("volume"),  # type: ignore[arg-type]
            cycles=data.get("cycles"),  # type: ignore[arg-type]
            unit=data.get("unit", "uL"),
        )
    if step_type == "incubate":
        return IncubateStep(
            temperature=data.get("temperature"),  # type: ignore[arg-type]
            time=data.get("time"),  # type: ignore[arg-type]
            time_unit=data.get("time_unit", data.get("unit", "min")),
            plate=data.get("plate"),
        )
    if step_type == "wash":
        return WashStep(
            well=data.get("well"),
            plate=data.get("plate"),
            volume=data.get("volume"),  # type: ignore[arg-type]
            cycles=data.get("cycles"),  # type: ignore[arg-type]
            unit=data.get("unit", "uL"),
        )
    if step_type == "read":
        return ReadStep(
            plate=data.get("plate"),  # type: ignore[arg-type]
            mode=data.get("mode"),  # type: ignore[arg-type]
            wavelength=data.get("wavelength"),  # type: ignore[arg-type]
        )
    raise ValueError(f"unknown step type: {step_type!r}")


def steps_from_list(raw_steps: Sequence[dict[str, Any]]) -> list[Step]:
    """Validate a sequence of raw step dictionaries."""
    if not isinstance(raw_steps, (list, tuple)):
        raise ValueError("steps must be a list of step mappings")
    return [step_from_dict(item) for item in raw_steps]


def steps_to_dicts(steps: Sequence[Step | dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialize steps (dataclasses or dicts) to plain JSON-safe dictionaries."""
    out: list[dict[str, Any]] = []
    for step in steps:
        if isinstance(step, dict):
            out.append(step_from_dict(step).to_dict())
        else:
            out.append(step.to_dict())
    return out


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if text in ("null", "~", ""):
        return None
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    return text


def _parse_simple_yaml(text: str) -> Any:
    """Minimal YAML parser for protocol lists and mappings when PyYAML is unavailable."""
    lines = [
        line.rstrip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError("empty YAML document")

    if lines[0].lstrip().startswith("- "):
        return _parse_yaml_list(lines, 0)[0]

    mapping: dict[str, Any] = {}
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if ":" not in line:
            raise ValueError(f"invalid YAML line: {line!r}")
        key, rest = line.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        if rest:
            mapping[key] = _parse_scalar(rest)
            idx += 1
        else:
            items, idx = _parse_yaml_list(lines, idx + 1)
            mapping[key] = items
    return mapping


def _parse_yaml_list(lines: list[str], start: int) -> tuple[list[dict[str, Any]], int]:
    items: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    idx = start
    while idx < len(lines):
        stripped = lines[idx].lstrip()
        if stripped.startswith("- "):
            if current is not None:
                items.append(current)
            current = {}
            rest = stripped[2:].strip()
            if rest:
                key, val = rest.split(":", 1)
                current[key.strip()] = _parse_scalar(val)
            idx += 1
        elif current is not None and (lines[idx].startswith(" ") or lines[idx].startswith("\t")):
            key, val = stripped.split(":", 1)
            current[key.strip()] = _parse_scalar(val)
            idx += 1
        else:
            break
    if current is not None:
        items.append(current)
    return items, idx


def _load_raw_document(path: Path) -> Any:
    content = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(content)
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError:
            return _parse_simple_yaml(content)
        return yaml.safe_load(content)
    raise ValueError(f"unsupported protocol file extension: {path.suffix!r}")


def load_protocol(path: str | Path) -> Protocol:
    """Load and validate a protocol from a JSON or YAML file."""
    file_path = Path(path)
    raw = _load_raw_document(file_path)
    if isinstance(raw, list):
        protocol_id = file_path.stem
        raw_steps = raw
    elif isinstance(raw, dict):
        protocol_id = str(raw.get("protocol_id") or file_path.stem)
        raw_steps = raw.get("steps")
    else:
        raise ValueError("protocol file must contain a list of steps or a mapping with 'steps'")

    steps = tuple(steps_from_list(raw_steps))
    return Protocol(protocol_id=protocol_id, steps=steps)


__all__ = [
    "IncubateStep",
    "MixStep",
    "Protocol",
    "ReadStep",
    "Step",
    "TransferStep",
    "WashStep",
    "asdict",
    "load_protocol",
    "step_from_dict",
    "steps_from_list",
    "steps_to_dicts",
]

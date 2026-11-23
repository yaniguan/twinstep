"""Validated dataclass schema and JSON/YAML loaders for generic liquid-handling steps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Literal, Union

VolumeUnit = Literal["uL", "mL"]
TimeUnit = Literal["min", "s"]

VALID_VOLUME_UNITS = ("uL", "mL")
VALID_TIME_UNITS = ("min", "s")

_EXTERNAL_PREFIXES = ("reservoir", "stock", "trough", "waste")
_BARE_WELL_RE = re.compile(r"^[A-Ha-h](?:[1-9]|1[0-2])$")
_INT_RE = re.compile(r"^[+-]?\d+$")


def _require_nonempty_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


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


def canonical_location(location: str) -> str:
    """Normalize well/plate/reservoir strings so 'a1', 'A1', and 'plate1:A1' match."""
    loc = location.strip()
    if ":" in loc:
        prefix, rest = loc.split(":", 1)
        prefix = prefix.strip()
        rest = rest.strip()
        if prefix.lower() in _EXTERNAL_PREFIXES:
            return f"{prefix.lower()}:{rest.lower()}"
        return f"{prefix}:{rest.upper()}"
    if _BARE_WELL_RE.match(loc):
        return f"plate1:{loc.upper()}"
    if loc.lower() in _EXTERNAL_PREFIXES:
        return loc.lower()
    return loc


def _well_plate_or_none(well_str: str) -> str | None:
    loc = well_str.strip()
    if ":" in loc:
        prefix, _ = loc.split(":", 1)
        prefix = prefix.strip()
        if prefix.lower() in _EXTERNAL_PREFIXES:
            return None
        return prefix
    if _BARE_WELL_RE.match(loc):
        return "plate1"
    return None


@dataclass(frozen=True)
class TransferStep:
    """Liquid transfer from a source location to a destination location."""

    source: str
    dest: str
    volume: float
    unit: VolumeUnit = "uL"
    type: Literal["transfer"] = "transfer"

    def __post_init__(self) -> None:
        src = _require_nonempty_str(self.source, "source")
        dst = _require_nonempty_str(self.dest, "dest")
        object.__setattr__(self, "source", src)
        object.__setattr__(self, "dest", dst)
        if canonical_location(src) == canonical_location(dst):
            raise ValueError(
                f"transfer source and dest must be distinct locations, got {src!r} and {dst!r}"
            )
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
        object.__setattr__(self, "well", _require_nonempty_str(self.well, "well"))
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
            object.__setattr__(self, "plate", _require_nonempty_str(self.plate, "plate"))
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
        clean_well: str | None = None
        clean_plate: str | None = None
        if self.well is not None:
            clean_well = _require_nonempty_str(self.well, "well")
            object.__setattr__(self, "well", clean_well)
            w_plate = _well_plate_or_none(clean_well)
            if w_plate is None:
                raise ValueError(
                    f"wash 'well' must name an assay well (e.g. 'plate1:A1'), got {clean_well!r}"
                )
        if self.plate is not None:
            clean_plate = _require_nonempty_str(self.plate, "plate")
            object.__setattr__(self, "plate", clean_plate)
        if clean_well is not None and clean_plate is not None:
            w_plate = _well_plate_or_none(clean_well)
            if w_plate != clean_plate:
                raise ValueError(
                    f"wash step plate {clean_plate!r} conflicts with well {clean_well!r}"
                )
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
    wavelength: int | None = None
    type: Literal["read"] = "read"

    def __post_init__(self) -> None:
        clean_plate = _require_nonempty_str(self.plate, "plate")
        clean_mode = _require_nonempty_str(self.mode, "mode")
        object.__setattr__(self, "plate", clean_plate)
        object.__setattr__(self, "mode", clean_mode)
        if self.wavelength is None:
            if clean_mode.lower() != "luminescence":
                raise ValueError(
                    f"wavelength is required for read mode {clean_mode!r} "
                    "(wavelength=None is supported only for 'luminescence')"
                )
        else:
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
        object.__setattr__(
            self, "protocol_id", _require_nonempty_str(self.protocol_id, "protocol_id")
        )
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
            time_unit=data.get("time_unit", data.get("unit", "min")),  # type: ignore[arg-type]
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
            wavelength=data.get("wavelength"),
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


def _strip_inline_comment(raw: str) -> str:
    """Strip unquoted '# ...' YAML inline comments while preserving '#' inside quotes."""
    in_single = False
    in_double = False
    escaped = False
    for idx, ch in enumerate(raw):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_double:
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            if idx == 0 or raw[idx - 1].isspace():
                return raw[:idx].rstrip()
    return raw.rstrip()


def _parse_scalar(raw: str) -> Any:
    text = _strip_inline_comment(raw).strip()
    if text in ("null", "~", ""):
        return None
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    if len(text) >= 2 and (
        (text.startswith('"') and text.endswith('"'))
        or (text.startswith("'") and text.endswith("'"))
    ):
        quote = text[0]
        inner = text[1:-1]
        if quote == '"':
            return inner.replace('\\"', '"').replace("\\\\", "\\")
        return inner.replace("''", "'")
    if _INT_RE.fullmatch(text):
        return int(text)
    try:
        val = float(text)
        if math.isfinite(val):
            return val
    except ValueError:
        pass
    return text


def _line_indent(line: str) -> int:
    expanded = line.expandtabs(2)
    return len(expanded) - len(expanded.lstrip(" "))


def _split_yaml_key_val(text: str) -> tuple[str, str]:
    in_single = False
    in_double = False
    for idx, ch in enumerate(text):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == ":" and not in_single and not in_double:
            if idx + 1 == len(text) or text[idx + 1].isspace():
                return text[:idx].strip(), text[idx + 1 :]
    raise ValueError(f"invalid YAML key-value line: {text!r}")


def _parse_simple_yaml(text: str) -> Any:
    """Minimal YAML parser for protocol lists and mappings when PyYAML is unavailable."""
    cleaned_lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        uncommented = _strip_inline_comment(raw_line)
        if not uncommented.strip():
            continue
        cleaned_lines.append((_line_indent(uncommented), uncommented.strip()))

    if not cleaned_lines:
        raise ValueError("empty YAML document")

    first_indent, first_content = cleaned_lines[0]
    if first_content == "-" or first_content.startswith("- "):
        items, next_idx = _parse_yaml_list(cleaned_lines, 0, first_indent)
        if next_idx < len(cleaned_lines):
            raise ValueError(f"unexpected YAML content at: {cleaned_lines[next_idx][1]!r}")
        return items

    mapping: dict[str, Any] = {}
    idx = 0
    root_indent = first_indent
    while idx < len(cleaned_lines):
        indent, content = cleaned_lines[idx]
        if indent != root_indent:
            raise ValueError(
                f"unexpected YAML indentation ({indent} != {root_indent}): {content!r}"
            )
        key, raw_rest = _split_yaml_key_val(content)
        rest = raw_rest.strip()
        if rest:
            mapping[key] = _parse_scalar(rest)
            idx += 1
        else:
            if idx + 1 >= len(cleaned_lines) or cleaned_lines[idx + 1][0] < indent:
                mapping[key] = None
                idx += 1
            else:
                child_indent, child_first = cleaned_lines[idx + 1]
                if child_first == "-" or child_first.startswith("- "):
                    items, idx = _parse_yaml_list(cleaned_lines, idx + 1, child_indent)
                    mapping[key] = items
                elif child_indent > indent:
                    sub_map: dict[str, Any] = {}
                    idx += 1
                    while idx < len(cleaned_lines) and cleaned_lines[idx][0] == child_indent:
                        s_key, s_rest = _split_yaml_key_val(cleaned_lines[idx][1])
                        sub_map[s_key] = _parse_scalar(s_rest)
                        idx += 1
                    mapping[key] = sub_map
                else:
                    raise ValueError(f"expected indented block after '{key}:'")
    return mapping


def _parse_yaml_list(
    lines: list[tuple[int, str]],
    start: int,
    list_indent: int,
) -> tuple[list[dict[str, Any]], int]:
    items: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    item_key_indent: int | None = None
    idx = start
    while idx < len(lines):
        indent, content = lines[idx]
        if indent < list_indent:
            break
        if indent == list_indent and (content == "-" or content.startswith("- ")):
            if current is not None:
                items.append(current)
            current = {}
            item_key_indent = None
            rest = content[1:].strip()
            if rest:
                key, val = _split_yaml_key_val(rest)
                current[key] = _parse_scalar(val)
            idx += 1
        elif current is not None and indent > list_indent:
            if item_key_indent is None:
                item_key_indent = indent
            elif indent != item_key_indent:
                raise ValueError(
                    f"inconsistent YAML mapping indentation "
                    f"({indent} != {item_key_indent}): {content!r}"
                )
            key, val = _split_yaml_key_val(content)
            current[key] = _parse_scalar(val)
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
            import yaml
        except ImportError:
            return _parse_simple_yaml(content)
        if yaml is None:
            return _parse_simple_yaml(content)
        return yaml.safe_load(content)
    raise ValueError(f"unsupported protocol file extension: {path.suffix!r}")


def load_protocol(path: str | Path) -> Protocol:
    """Load and validate a protocol from a JSON or YAML file."""
    file_path = Path(path)
    raw = _load_raw_document(file_path)
    raw_steps: Any
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
    "canonical_location",
    "load_protocol",
    "step_from_dict",
    "steps_from_list",
    "steps_to_dicts",
]

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
ReadMode = Literal[
    "absorbance",
    "fluorescence",
    "luminescence",
    "imaging",
    "turbidity",
    "od600",
    "plate_reader",
]

VALID_VOLUME_UNITS = ("uL", "mL")
VALID_TIME_UNITS = ("min", "s")
VALID_READ_MODES: tuple[str, ...] = (
    "absorbance",
    "fluorescence",
    "luminescence",
    "imaging",
    "turbidity",
    "od600",
    "plate_reader",
)
_OPTIONAL_WAVELENGTH_MODES = frozenset({"luminescence", "imaging", "od600", "plate_reader"})

_EXTERNAL_PREFIXES = ("reservoir", "stock", "trough", "waste")
# Supports 96-well and 384-well coordinates (rows A-P, columns 1-24, optional 0-padding 01-09).
_BARE_WELL_RE = re.compile(r"^([A-Pa-p])(0?[1-9]|1[0-9]|2[0-4])$")
_INT_RE = re.compile(r"^[+-]?\d(?:_?\d)*$")
_FLOAT_RE = re.compile(
    r"^[+-]?(?:\d(?:_?\d)*(?:\.\d(?:_?\d)*)?(?:[eE][+-]?\d(?:_?\d)*)?"
    r"|\.\d(?:_?\d)*(?:[eE][+-]?\d(?:_?\d)*)?)$"
)
_DOUBLE_ESCAPE_MAP: dict[str, str] = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "0": "\0",
    "b": "\b",
    "f": "\f",
}


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


def canonical_plate(plate: str) -> str:
    """Canonicalize a plate identifier case-insensitively."""
    return plate.strip().lower()


def is_bare_well(location: str) -> bool:
    """Return True when ``location`` is a bare 96- or 384-well coordinate (e.g. A1, A01, P24)."""
    return _BARE_WELL_RE.match(location.strip()) is not None


def _normalize_well_coord(coord: str) -> str:
    m = _BARE_WELL_RE.match(coord)
    if m is not None:
        return f"{m.group(1).upper()}{int(m.group(2))}"
    return coord.upper()


def canonical_location(location: str, default_plate: str = "plate1") -> str:
    """Normalize well/plate/reservoir strings so 'a1', 'A01', and 'Plate1:A1' match."""
    loc = location.strip()
    if ":" in loc:
        prefix, rest = loc.split(":", 1)
        prefix_norm = canonical_plate(prefix)
        rest = rest.strip()
        if prefix_norm in _EXTERNAL_PREFIXES:
            return f"{prefix_norm}:{rest.lower()}"
        return f"{prefix_norm}:{_normalize_well_coord(rest)}"
    if _BARE_WELL_RE.match(loc):
        return f"{canonical_plate(default_plate)}:{_normalize_well_coord(loc)}"
    if loc.lower() in _EXTERNAL_PREFIXES:
        return loc.lower()
    return canonical_plate(loc)


def _well_plate_or_none(
    well_str: str, default_plate: str | None = "plate1"
) -> str | None:
    loc = well_str.strip()
    if ":" in loc:
        prefix, _ = loc.split(":", 1)
        prefix_norm = canonical_plate(prefix)
        if prefix_norm in _EXTERNAL_PREFIXES:
            return None
        return prefix_norm
    if _BARE_WELL_RE.match(loc):
        return canonical_plate(default_plate) if default_plate is not None else None
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
            w_plate = _well_plate_or_none(clean_well, default_plate=clean_plate)
            if w_plate != canonical_plate(clean_plate):
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
        clean_mode = _require_nonempty_str(self.mode, "mode").lower()
        if clean_mode not in VALID_READ_MODES:
            raise ValueError(
                f"mode must be one of {VALID_READ_MODES}, got {self.mode!r}"
            )
        object.__setattr__(self, "plate", clean_plate)
        object.__setattr__(self, "mode", clean_mode)
        if self.wavelength is None:
            if clean_mode not in _OPTIONAL_WAVELENGTH_MODES:
                raise ValueError(
                    f"wavelength is required for read mode {clean_mode!r} "
                    f"(wavelength=None is supported only for {sorted(_OPTIONAL_WAVELENGTH_MODES)})"
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
    split: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "protocol_id", _require_nonempty_str(self.protocol_id, "protocol_id")
        )
        if not self.steps:
            raise ValueError("protocol must contain at least one step")
        if self.split is not None:
            clean_split = _require_nonempty_str(self.split, "split").lower()
            if clean_split not in ("train", "val", "test"):
                raise ValueError(
                    f"split must be one of ('train', 'val', 'test'), got {self.split!r}"
                )
            object.__setattr__(self, "split", clean_split)

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


def _unescape_double_quoted(inner: str) -> str:
    out: list[str] = []
    idx = 0
    n = len(inner)
    while idx < n:
        ch = inner[idx]
        if ch == "\\" and idx + 1 < n:
            nxt = inner[idx + 1]
            if nxt in _DOUBLE_ESCAPE_MAP:
                out.append(_DOUBLE_ESCAPE_MAP[nxt])
                idx += 2
                continue
            if nxt == "x" and idx + 3 < n:
                hex2 = inner[idx + 2 : idx + 4]
                if all(c in "0123456789abcdefABCDEF" for c in hex2):
                    out.append(chr(int(hex2, 16)))
                    idx += 4
                    continue
            if nxt == "u" and idx + 5 < n:
                hex4 = inner[idx + 2 : idx + 6]
                if all(c in "0123456789abcdefABCDEF" for c in hex4):
                    out.append(chr(int(hex4, 16)))
                    idx += 6
                    continue
        out.append(ch)
        idx += 1
    return "".join(out)


def _try_unquote_yaml_str(text: str) -> str | None:
    if len(text) < 2:
        return None
    if text[0] == '"' and text[-1] == '"':
        inner = text[1:-1]
        escaped = False
        for ch in inner:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
            elif ch == '"':
                return None
        if escaped:
            return None
        return _unescape_double_quoted(inner)
    if text[0] == "'" and text[-1] == "'":
        inner = text[1:-1]
        idx = 0
        while idx < len(inner):
            if inner[idx] == "'":
                if idx + 1 < len(inner) and inner[idx + 1] == "'":
                    idx += 2
                    continue
                return None
            idx += 1
        return inner.replace("''", "'")
    return None


def _split_flow_items(inner: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    in_single = False
    in_double = False
    escaped = False
    depth = 0
    for ch in inner:
        if escaped:
            escaped = False
            buf.append(ch)
            continue
        if ch == "\\" and in_double:
            escaped = True
            buf.append(ch)
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            buf.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
        elif not in_single and not in_double:
            if ch in "[{":
                depth += 1
                buf.append(ch)
            elif ch in "]}":
                depth -= 1
                buf.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(buf))
                buf = []
            else:
                buf.append(ch)
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts


def _parse_scalar(raw: str) -> Any:
    text = _strip_inline_comment(raw).strip()
    unquoted = _try_unquote_yaml_str(text)
    if unquoted is not None:
        return unquoted
    if text in ("null", "Null", "NULL", "~", ""):
        return None
    if text in ("true", "True", "TRUE"):
        return True
    if text in ("false", "False", "FALSE"):
        return False
    if text in (".inf", "+.inf", ".Inf", "+.Inf", ".INF", "+.INF"):
        return float("inf")
    if text in ("-.inf", "-.Inf", "-.INF"):
        return float("-inf")
    if text in (".nan", ".NaN", ".NAN"):
        return float("nan")
    if len(text) >= 2 and text[0] == "[" and text[-1] == "]":
        inner_list = text[1:-1].strip()
        if not inner_list:
            return []
        return [
            _parse_scalar(item)
            for item in _split_flow_items(inner_list)
            if item.strip()
        ]
    if len(text) >= 2 and text[0] == "{" and text[-1] == "}":
        inner_map = text[1:-1].strip()
        if not inner_map:
            return {}
        flow_map: dict[str, Any] = {}
        for item in _split_flow_items(inner_map):
            if not item.strip():
                continue
            k, v = _split_yaml_key_val(item.strip())
            flow_map[k] = _parse_scalar(v)
        return flow_map
    if _INT_RE.fullmatch(text):
        return int(text.replace("_", ""))
    if _FLOAT_RE.fullmatch(text):
        try:
            return float(text.replace("_", ""))
        except ValueError:
            pass
    return text


def _line_indent(line: str) -> int:
    expanded = line.expandtabs(2)
    return len(expanded) - len(expanded.lstrip(" "))


def _find_yaml_key_colon(text: str) -> int | None:
    in_single = False
    in_double = False
    escaped = False
    depth = 0
    for idx, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_double:
            escaped = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_single and not in_double
        elif not in_single and not in_double:
            if ch in "[{":
                depth += 1
            elif ch in "]}":
                depth = max(0, depth - 1)
            elif ch == ":" and depth == 0:
                if idx + 1 == len(text) or text[idx + 1].isspace():
                    return idx
    return None


def _split_yaml_key_val(text: str) -> tuple[str, str]:
    colon_idx = _find_yaml_key_colon(text)
    if colon_idx is None:
        raise ValueError(f"invalid YAML key-value line: {text!r}")
    raw_key = text[:colon_idx].strip()
    if not raw_key:
        raise ValueError(f"empty YAML mapping key in line: {text!r}")
    unquoted_key = _try_unquote_yaml_str(raw_key)
    key = unquoted_key if unquoted_key is not None else raw_key
    return key, text[colon_idx + 1 :]


def _parse_simple_yaml(text: str) -> Any:
    """Minimal YAML parser for protocol lists and mappings when PyYAML is unavailable."""
    cleaned_lines: list[tuple[int, str]] = []
    doc_started = False
    for raw_line in text.splitlines():
        uncommented = _strip_inline_comment(raw_line)
        stripped = uncommented.strip()
        if not stripped:
            continue
        indent = _line_indent(uncommented)
        if indent == 0 and stripped == "---":
            if doc_started or cleaned_lines:
                raise ValueError("multiple YAML documents are not supported")
            doc_started = True
            continue
        if indent == 0 and stripped == "...":
            break
        cleaned_lines.append((indent, stripped))

    if not cleaned_lines:
        raise ValueError("empty YAML document")

    first_indent = cleaned_lines[0][0]
    node, next_idx = _parse_yaml_node(cleaned_lines, 0, first_indent)
    if next_idx < len(cleaned_lines):
        raise ValueError(f"unexpected YAML content at: {cleaned_lines[next_idx][1]!r}")
    return node


def _parse_yaml_node(
    lines: list[tuple[int, str]],
    start: int,
    node_indent: int,
) -> tuple[Any, int]:
    _, content = lines[start]
    if content == "-" or content.startswith("- "):
        return _parse_yaml_list(lines, start, node_indent)
    return _parse_yaml_mapping(lines, start, node_indent)


def _parse_yaml_mapping(
    lines: list[tuple[int, str]],
    start: int,
    map_indent: int,
) -> tuple[dict[str, Any], int]:
    mapping: dict[str, Any] = {}
    idx = start
    while idx < len(lines):
        indent, content = lines[idx]
        if indent < map_indent:
            break
        if indent != map_indent:
            raise ValueError(
                f"unexpected YAML indentation ({indent} != {map_indent}): {content!r}"
            )
        if content == "-" or content.startswith("- "):
            break
        key, raw_rest = _split_yaml_key_val(content)
        rest = raw_rest.strip()
        if rest:
            mapping[key] = _parse_scalar(rest)
            idx += 1
        else:
            if idx + 1 >= len(lines) or lines[idx + 1][0] < indent:
                mapping[key] = None
                idx += 1
            else:
                child_indent, child_first = lines[idx + 1]
                if child_indent == indent and (
                    child_first == "-" or child_first.startswith("- ")
                ):
                    items, idx = _parse_yaml_list(lines, idx + 1, child_indent)
                    mapping[key] = items
                elif child_indent > indent:
                    child_val, idx = _parse_yaml_node(lines, idx + 1, child_indent)
                    mapping[key] = child_val
                else:
                    mapping[key] = None
                    idx += 1
    return mapping, idx


def _parse_yaml_list(
    lines: list[tuple[int, str]],
    start: int,
    list_indent: int,
) -> tuple[list[Any], int]:
    items: list[Any] = []
    idx = start
    while idx < len(lines):
        indent, content = lines[idx]
        if indent < list_indent:
            break
        if indent != list_indent or not (content == "-" or content.startswith("- ")):
            break
        after_dash = content[1:]
        rest = after_dash.strip()
        if not rest:
            idx += 1
            if idx < len(lines) and lines[idx][0] > list_indent:
                elem, idx = _parse_yaml_node(lines, idx, lines[idx][0])
                items.append(elem)
            else:
                items.append(None)
            continue

        if _find_yaml_key_colon(rest) is None:
            items.append(_parse_scalar(rest))
            idx += 1
            continue

        spaces_after_dash = len(after_dash) - len(after_dash.lstrip(" "))
        first_key_indent = list_indent + 1 + spaces_after_dash
        key, raw_val = _split_yaml_key_val(rest)
        val_str = raw_val.strip()
        item_map: dict[str, Any] = {}
        idx += 1
        if val_str:
            item_map[key] = _parse_scalar(val_str)
        elif idx < len(lines) and lines[idx][0] > first_key_indent:
            child_val, idx = _parse_yaml_node(lines, idx, lines[idx][0])
            item_map[key] = child_val
        elif (
            idx < len(lines)
            and lines[idx][0] == first_key_indent
            and (lines[idx][1] == "-" or lines[idx][1].startswith("- "))
        ):
            child_val, idx = _parse_yaml_list(lines, idx, lines[idx][0])
            item_map[key] = child_val
        else:
            item_map[key] = None

        if idx < len(lines) and lines[idx][0] > list_indent:
            sib_indent = lines[idx][0]
            if sib_indent != first_key_indent:
                raise ValueError(
                    f"inconsistent YAML mapping indentation "
                    f"({sib_indent} != {first_key_indent}): {lines[idx][1]!r}"
                )
            more_map, idx = _parse_yaml_mapping(lines, idx, sib_indent)
            item_map.update(more_map)
        items.append(item_map)
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
    raw_split: str | None = None
    if isinstance(raw, list):
        protocol_id = file_path.stem
        raw_steps = raw
    elif isinstance(raw, dict):
        protocol_id = str(raw.get("protocol_id") or file_path.stem)
        raw_steps = raw.get("steps")
        if raw.get("split") is not None:
            raw_split = str(raw["split"])
    else:
        raise ValueError("protocol file must contain a list of steps or a mapping with 'steps'")

    steps = tuple(steps_from_list(raw_steps))
    return Protocol(protocol_id=protocol_id, steps=steps, split=raw_split)


__all__ = [
    "IncubateStep",
    "MixStep",
    "Protocol",
    "ReadMode",
    "ReadStep",
    "Step",
    "TransferStep",
    "VALID_READ_MODES",
    "WashStep",
    "asdict",
    "canonical_location",
    "canonical_plate",
    "is_bare_well",
    "load_protocol",
    "step_from_dict",
    "steps_from_list",
    "steps_to_dicts",
]

"""Tests for protocol step schema validation and JSON/YAML loading."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pytest

from twinstep.schema import (
    IncubateStep,
    MixStep,
    Protocol,
    ReadStep,
    TransferStep,
    WashStep,
    _parse_simple_yaml,
    load_protocol,
    step_from_dict,
    steps_to_dicts,
)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def test_step_round_trip_all_types() -> None:
    steps = [
        TransferStep(source="reservoir:buffer", dest="plate1:A1", volume=100, unit="uL"),
        MixStep(well="plate1:A1", volume=80, cycles=5, unit="uL"),
        IncubateStep(temperature=37, time=30, time_unit="min", plate="plate1"),
        WashStep(plate="plate1", volume=300, cycles=3, unit="uL"),
        ReadStep(plate="plate1", mode="absorbance", wavelength=450),
        ReadStep(plate="plate1", mode="luminescence", wavelength=None),
    ]
    serialized = steps_to_dicts(steps)
    reparsed = [step_from_dict(item) for item in serialized]
    assert reparsed == steps


@pytest.mark.parametrize(
    "bad_step",
    [
        {"type": "transfer", "source": "a", "dest": "b", "volume": 0, "unit": "uL"},
        {"type": "transfer", "source": "a", "dest": "b", "volume": 10, "unit": "L"},
        {"type": "transfer", "source": "", "dest": "b", "volume": 10, "unit": "uL"},
        {"type": "transfer", "source": "plate1:A1", "dest": "plate1:A1", "volume": 50},
        {"type": "transfer", "source": "A1", "dest": "plate1:a1", "volume": 50},
        {"type": "mix", "well": "A1", "volume": 50, "cycles": 0},
        {"type": "mix", "well": "A1", "volume": True, "cycles": 3},
        {"type": "incubate", "temperature": -300, "time": 10, "time_unit": "min"},
        {"type": "incubate", "temperature": 37, "time": 10, "time_unit": "hr"},
        {"type": "wash", "volume": 200, "cycles": 2},
        {"type": "wash", "plate": "plate1", "well": "plate2:A1", "volume": 200, "cycles": 2},
        {"type": "wash", "well": "reservoir:water", "volume": 200, "cycles": 2},
        {"type": "read", "plate": "plate1", "mode": "absorbance", "wavelength": 0},
        {"type": "read", "plate": "plate1", "mode": "absorbance", "wavelength": None},
        {"type": "read", "plate": "plate1", "mode": "banana", "wavelength": 450},
        {"type": "unknown", "foo": 1},
    ],
)
def test_invalid_step_rejected(bad_step: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        step_from_dict(bad_step)


def test_load_example_protocols() -> None:
    files = sorted(EXAMPLES_DIR.glob("*.json"))
    assert len(files) == 24
    for path in files:
        proto = load_protocol(path)
        assert isinstance(proto, Protocol)
        assert proto.protocol_id == path.stem
        assert len(proto.steps) >= 5


def test_load_yaml_protocol(tmp_path: Path) -> None:
    yaml_path = tmp_path / "sample.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "protocol_id: yaml_demo",
                "steps:",
                "  - type: transfer",
                "    source: reservoir:water",
                "    dest: plate1:A1",
                "    volume: 100",
                "    unit: uL",
                "  - type: incubate",
                "    plate: plate1",
                "    temperature: 25",
                "    time: 10",
                "    time_unit: min",
                "  - type: read",
                "    plate: plate1",
                "    mode: absorbance",
                "    wavelength: 450",
            ]
        ),
        encoding="utf-8",
    )
    proto = load_protocol(yaml_path)
    assert proto.protocol_id == "yaml_demo"
    assert len(proto.steps) == 3
    assert isinstance(proto.steps[0], TransferStep)
    assert isinstance(proto.steps[1], IncubateStep)
    assert isinstance(proto.steps[2], ReadStep)


def test_parse_simple_yaml_inline_comments_scientific_and_colons() -> None:
    yaml_text = "\n".join(
        [
            "# Leading comment",
            "protocol_id: 'demo#1'  # inline comment after quoted hash",
            "steps:  # trailing comment on mapping key",
            "  - type: transfer  # step type",
            "    source: reservoir:buffer",
            "    dest: plate1:A1",
            "    volume: 1e2  # scientific notation 100.0",
            "    unit: uL",
            "  - type: incubate",
            "    plate: plate1",
            "    temperature: +25.5",
            "    time: 5e-1  # 0.5 min",
            "    time_unit: min",
        ]
    )
    parsed = _parse_simple_yaml(yaml_text)
    assert parsed["protocol_id"] == "demo#1"
    assert len(parsed["steps"]) == 2
    assert parsed["steps"][0]["volume"] == 100.0
    assert parsed["steps"][1]["temperature"] == 25.5
    assert parsed["steps"][1]["time"] == 0.5


def test_load_protocol_yaml_fallback_when_pyyaml_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "yaml", None)

    mapping_yaml = tmp_path / "map_proto.yaml"
    mapping_yaml.write_text(
        "\n".join(
            [
                "steps:  # protocol steps",
                "  - type: transfer",
                "    source: reservoir:water",
                "    dest: plate1:A1",
                "    volume: 1.5e2  # 150 uL",
                "    unit: uL",
                "  - type: read",
                "    plate: plate1",
                "    mode: luminescence",
                "protocol_id: after_steps_id",
            ]
        ),
        encoding="utf-8",
    )
    p1 = load_protocol(mapping_yaml)
    assert p1.protocol_id == "after_steps_id"
    assert isinstance(p1.steps[0], TransferStep) and p1.steps[0].volume == 150.0
    assert isinstance(p1.steps[1], ReadStep) and p1.steps[1].wavelength is None

    list_yaml = tmp_path / "list_proto.yml"
    list_yaml.write_text(
        "\n".join(
            [
                "- type: transfer",
                "  source: reservoir:water",
                "  dest: plate1:B1",
                "  volume: 50  # uL",
                "- type: mix",
                "  well: plate1:B1",
                "  volume: 40",
                "  cycles: 3",
            ]
        ),
        encoding="utf-8",
    )
    p2 = load_protocol(list_yaml)
    assert p2.protocol_id == "list_proto"
    assert len(p2.steps) == 2


# Issue #29: PyYAML parity on document markers, quoted/escaped colon keys, underscores, & overflow
def test_issue_29_parse_simple_yaml_pyyaml_parity_and_stdlib_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        import yaml as pyyaml
    except ImportError:
        pyyaml = None

    doc_markers = (
        "---\n"
        '"protocol_id": "demo: 1"\n'
        '"steps":\n'
        '  - "type": transfer\n'
        '    "source": "reservoir:buf"\n'
        '    "dest": "plate1:A1"\n'
        "    volume: 1_000\n"
        "    unit: uL\n"
        "  - type: read\n"
        "    plate: plate1\n"
        "    mode: absorbance\n"
        "    wavelength: 1_000\n"
        "...\n"
    )
    parsed_doc = _parse_simple_yaml(doc_markers)
    assert list(parsed_doc.keys()) == ["protocol_id", "steps"]
    assert parsed_doc["protocol_id"] == "demo: 1"
    assert parsed_doc["steps"][1]["wavelength"] == 1000
    assert isinstance(parsed_doc["steps"][1]["wavelength"], int)
    if pyyaml is not None:
        assert parsed_doc == pyyaml.safe_load(doc_markers)

    escaped_fixture = (
        '---\n'
        '"key\\"with: colon": 42\n'
        '"multiline": "line1\\nline2\\t\\"quoted\\\\"\n'
        '"single_esc": \'plate:\'\'1\'\'\'\n'
        "upper_true: TRUE\n"
        "upper_false: FALSE\n"
        "upper_null: NULL\n"
        "sci_small: 1.0e-3\n"
        "sci_short: 1e-3\n"
        "...\n"
    )
    parsed_esc = _parse_simple_yaml(escaped_fixture)
    assert parsed_esc['key"with: colon'] == 42
    assert parsed_esc["multiline"] == 'line1\nline2\t"quoted\\'
    assert parsed_esc["single_esc"] == "plate:'1'"
    assert parsed_esc["upper_true"] is True
    assert parsed_esc["upper_false"] is False
    assert parsed_esc["upper_null"] is None
    assert parsed_esc["sci_small"] == pytest.approx(0.001)
    assert parsed_esc["sci_short"] == pytest.approx(0.001)
    if pyyaml is not None:
        # Compare all keys except `1e-3` where YAML 1.2 parses float vs PyYAML 1.1 string
        sub_check = {k: v for k, v in parsed_esc.items() if k != "sci_short"}
        pyyaml_loaded = pyyaml.safe_load(escaped_fixture)
        pyyaml_loaded.pop("sci_short", None)
        assert sub_check == pyyaml_loaded

    # Overflow scientific notation propagates non-finite float to schema validator
    overflow_step = _parse_simple_yaml(
        "type: transfer\nsource: reservoir:a\ndest: plate1:A1\nvolume: 1.0e500\n"
    )
    with pytest.raises(ValueError, match="positive finite number"):
        step_from_dict(overflow_step)

    # Full stdlib-only (`yaml=None`) load_protocol round-trip with `---`/`...` and `1_000`
    monkeypatch.setitem(sys.modules, "yaml", None)
    doc_file = tmp_path / "doc_markers.yaml"
    doc_file.write_text(doc_markers, encoding="utf-8")
    loaded = load_protocol(doc_file)
    assert loaded.protocol_id == "demo: 1"
    assert len(loaded.steps) == 2
    assert isinstance(loaded.steps[1], ReadStep) and loaded.steps[1].wavelength == 1000

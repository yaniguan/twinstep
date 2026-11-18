"""Tests for protocol step schema validation and JSON/YAML loading."""

from __future__ import annotations

from pathlib import Path
import pytest

from twinstep.schema import (
    IncubateStep,
    MixStep,
    Protocol,
    ReadStep,
    TransferStep,
    WashStep,
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
        {"type": "mix", "well": "A1", "volume": 50, "cycles": 0},
        {"type": "mix", "well": "A1", "volume": True, "cycles": 3},
        {"type": "incubate", "temperature": -300, "time": 10, "time_unit": "min"},
        {"type": "incubate", "temperature": 37, "time": 10, "time_unit": "hr"},
        {"type": "wash", "volume": 200, "cycles": 2},
        {"type": "read", "plate": "plate1", "mode": "absorbance", "wavelength": 0},
        {"type": "unknown", "foo": 1},
    ],
)
def test_invalid_step_rejected(bad_step: dict) -> None:
    with pytest.raises(ValueError):
        step_from_dict(bad_step)


def test_load_example_protocols() -> None:
    files = sorted(EXAMPLES_DIR.glob("*.json"))
    assert len(files) == 3
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

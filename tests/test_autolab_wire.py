"""Tests for the Auto-Lab 15-repository wire contract node (autolab_wire.py)."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import autolab_wire  # noqa: E402
from twinstep.injectors import ERROR_TYPES  # noqa: E402


def test_wire_constants_and_valid_dag_twin_synthesis() -> None:
    assert autolab_wire.REPO_ID == "twinstep"
    assert autolab_wire.LAYER == "L3"
    assert autolab_wire.REPO_DIR == REPO_ROOT

    packet: dict[str, Any] = {
        "parsed_spans": {
            "volume_ul": 120.0,
            "temp_c": 37.0,
            "duration_min": 20.0,
        },
        "seed": 42,
    }
    snapshot = {"parsed_spans": dict(packet["parsed_spans"]), "seed": 42}

    out = autolab_wire.process_packet(packet)
    assert packet == snapshot

    assert out["schema_version"] == "autolab-wire/v1"
    assert out["halted"] is False
    assert out["halt_reason"] is None

    twins = out["error_twins"]
    assert twins["step_count"] == 4
    assert twins["dag_edges"] == [[0, 1], [1, 2], [2, 3]]
    assert twins["corrupted_twin_count"] == 6
    assert set(twins["fault_types"]) == set(ERROR_TYPES)

    token_lengths = twins["token_lengths"]
    assert len(token_lengths) >= 4
    assert all(isinstance(length, int) and 48 <= length <= 240 for length in token_lengths)

    assert len(out["trace"]) == 1
    assert out["trace"][0]["repo"] == "twinstep"
    assert out["trace"][0]["layer"] == "L3"
    assert out["trace"][0]["status"] == "PASS"

    assert len(out["bottom_reasoning_traces"]) == 1
    assert out["bottom_reasoning_traces"][0]["repo"] == "twinstep"
    assert out["bottom_reasoning_traces"][0]["invariant"] == "violated_dag_edge_rejected"

    # Deterministic repeatability and explicit valid order acceptance
    assert autolab_wire.process_packet(packet) == out

    valid_idx_out = autolab_wire.process_packet(
        {"parsed_spans": packet["parsed_spans"], "candidate_step_order": [0, 1, 2, 3]}
    )
    assert valid_idx_out["halted"] is False

    valid_name_out = autolab_wire.process_packet(
        {
            "parsed_spans": packet["parsed_spans"],
            "candidate_step_order": ["transfer", "mix", "incubate", "read"],
        }
    )
    assert valid_name_out["halted"] is False


def test_violated_step_ordering_read_before_incubate_blocks() -> None:
    # Step indices: 0=transfer, 1=mix, 2=incubate, 3=read -> [0, 1, 3, 2] puts read before incubate
    blocked_idx = autolab_wire.process_packet(
        {
            "parsed_spans": {"volume_ul": 100.0, "temp_c": 37.0, "duration_min": 15.0},
            "candidate_step_order": [0, 1, 3, 2],
        }
    )
    assert blocked_idx["halted"] is True
    assert blocked_idx["halt_reason"] == "L3_TWINSTEP_DAG_BLOCK"
    assert blocked_idx["trace"][-1]["repo"] == "twinstep"
    assert blocked_idx["trace"][-1]["status"] == "BLOCK"
    assert (
        blocked_idx["bottom_reasoning_traces"][-1]["invariant"] == "violated_dag_edge_rejected"
    )

    # Step-name representation: read before incubate
    blocked_names = autolab_wire.process_packet(
        {
            "parsed_spans": {"volume_ul": 100.0, "temp_c": 37.0, "duration_min": 15.0},
            "candidate_step_order": ["transfer", "mix", "read", "incubate"],
        }
    )
    assert blocked_names["halted"] is True
    assert blocked_names["halt_reason"] == "L3_TWINSTEP_DAG_BLOCK"
    assert blocked_names["trace"][-1]["status"] == "BLOCK"

    # Non-bijective duplicate step order is also rejected
    blocked_dup = autolab_wire.process_packet({"candidate_step_order": [0, 1, 1, 3]})
    assert blocked_dup["halted"] is True
    assert blocked_dup["halt_reason"] == "L3_TWINSTEP_DAG_BLOCK"


def test_halted_packet_passthrough() -> None:
    upstream_halted: dict[str, Any] = {
        "schema_version": "autolab-wire/v1",
        "halted": True,
        "halt_reason": "L4_ATTRIB_QUARANTINE",
        "trace": [
            {"repo": "lab-attrib", "layer": "L4", "status": "QUARANTINE", "summary": "ghost"}
        ],
        "bottom_reasoning_traces": [
            {"repo": "lab-attrib", "invariant": "corrupted_run_never_poisons_memory"}
        ],
    }
    out = autolab_wire.process_packet(upstream_halted)

    assert out == upstream_halted
    assert out is not upstream_halted
    assert out["trace"] is not upstream_halted["trace"]
    assert "error_twins" not in out


def test_issue_32_stepbot_yaml_and_execution_artifact_contrastive_twins(
    tmp_path: Path,
) -> None:
    import json
    from twinstep.interop import to_stepbot_dict
    from twinstep.schema import load_protocol

    mir = load_protocol(REPO_ROOT / "examples" / "mix_incubate_read.json")
    sb_dict = to_stepbot_dict(mir)

    pkt: dict[str, Any] = {
        "protocol_id": mir.protocol_id,
        "stepbot_protocol": sb_dict,
        "candidate_step_order": list(range(len(mir.steps))),
        "attribution": {
            "outcome": "EXECUTION_ARTIFACT",
            "blamed_step_index": 6,
            "fault_family": "thermal_excursion",
            "observed_temp_c": 82.0,
        },
    }
    out = autolab_wire.process_packet(pkt)
    assert out["halted"] is False
    assert out["halt_reason"] is None
    twins = out["error_twins"]
    assert twins["step_count"] == 8
    assert twins["admit_to_memory"] is True
    assert twins["attribution_outcome"] == "EXECUTION_ARTIFACT"
    assert twins["blamed_step_index"] == 6
    assert twins["targeted_fault_type"] == "wrong_temperature"
    pairs = twins["contrastive_pairs"]
    assert len(pairs) >= 6
    first_chosen = json.loads(pairs[0]["dpo"]["chosen"])
    assert first_chosen["label"] == "wrong_temperature"
    assert first_chosen["step_index"] == 6
    assert pairs[0]["rlvr"]["ground_truth"]["step_index"] == 6

    # Lowering rehearsal / stepbot protocol_yaml_dict with multi-well transfer + standalone delay
    rehearsal_pkt: dict[str, Any] = {
        "rehearsal": {
            "protocol_yaml_dict": {
                "metadata": {"name": "native_elisa"},
                "steps": [
                    {
                        "transfer": {
                            "src": "reservoir:A1",
                            "dst": ["plate1:A1", "plate1:A2"],
                            "volume": 100,
                        }
                    },
                    {"mix": {"wells": ["plate1:A1", "plate1:A2"], "volume": 50, "reps": 3}},
                    {"delay": {"minutes": 10}},
                    {"read": {"plate": "plate1", "mode": "absorbance", "wavelength": 450}},
                ],
            }
        },
        "candidate_step_order": [0, 1, 2, 3, 4, 5],
    }
    out_reh = autolab_wire.process_packet(rehearsal_pkt)
    assert out_reh["halted"] is False
    assert out_reh["error_twins"]["step_count"] == 6
    assert set(out_reh["error_twins"]) == {
        "step_count",
        "dag_edges",
        "corrupted_twin_count",
        "fault_types",
        "token_lengths",
    }

    # Halted execution artifact: passthrough by default, synthesized when requested,
    # while L4_ATTRIB_QUARANTINE never admits twins to memory.
    halted_artifact: dict[str, Any] = {
        "halted": True,
        "halt_reason": "L4_EXECUTION_ARTIFACT",
        "protocol_steps": mir.to_dicts(),
        "attribution": {
            "outcome": "CERTIFIED_NEGATIVE",
            "blamed_step_index": 2,
            "fault_family": "dead_volume",
            "observed_volume_ul": 8.0,
        },
    }
    assert "error_twins" not in autolab_wire.process_packet(halted_artifact)

    synth_out = autolab_wire.process_packet(
        halted_artifact, synthesize_from_execution_artifact=True
    )
    assert synth_out["halted"] is True
    assert synth_out["halt_reason"] == "L4_EXECUTION_ARTIFACT"
    assert synth_out["error_twins"]["admit_to_memory"] is True
    assert synth_out["error_twins"]["targeted_fault_type"] == "wrong_volume"
    assert len(synth_out["error_twins"]["contrastive_pairs"]) >= 4

    yaml_path = tmp_path / "mir_stepbot.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "---",
                "metadata:",
                "  name: yaml_wire_proto",
                "steps:",
                "  - transfer:",
                "      src: reservoir:A1",
                "      dst: [plate1:A1, plate1:A2]",
                "      volume: 100",
                "  - delay:",
                "      minutes: 5",
                "  - read:",
                "      plate: plate1",
                "      mode: absorbance",
                "      wavelength: 450",
            ]
        ),
        encoding="utf-8",
    )
    out_yaml = autolab_wire.process_packet({"stepbot_yaml": yaml_path})
    assert out_yaml["halted"] is False
    assert out_yaml["error_twins"]["step_count"] == 4

    quarantine_non_halted = autolab_wire.process_packet(
        {
            "halted": False,
            "attribution": {"outcome": "L4_ATTRIB_QUARANTINE"},
        }
    )
    assert quarantine_non_halted["error_twins"]["admit_to_memory"] is False
    assert quarantine_non_halted["error_twins"]["contrastive_pairs"] == []

# twinstep

`twinstep` is a seeded-error twin generator and evaluator for generic liquid-handling protocol steps. Given valid JSON or YAML protocols (`transfer`, `mix`, `incubate`, `wash`, `read`), it deterministically injects typed protocol errors and generates labeled JSONL records (`raw`, `dpo`, `sft`, or `rlvr`) for protocol reasoning benchmarks:

1. **Error detection & compound multi-fault debugging**: corrupted step sequences (`--k-errors 1` or `--k-errors k`) paired with their reference steps, `label`/`labels`, `step_index`/`step_indices`, `original_value`, `new_value`, and plain-English descriptions.
2. **Step ordering**: shuffled step sequences paired with `gold_order` and minimal dependency `constraints` (RAW, WAR, and plate barriers) between steps where order matters.
3. **Error repair**: exact canonical and dependency-equivalent scoring for repaired protocols (`repaired_steps`).

## Install

```bash
pip install -e .
```

For development (running tests, linters, and full YAML support via PyYAML):

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+. The runtime package uses the Python standard library only (`dependencies = []`).

Part of **[Robot AI Auto-Lab (`auto-lab`)](https://github.com/yaniguan/auto-lab)** ([Project Studio](https://yaniguan.github.io/auto-lab/)) — **Layer `L3 · Post-Training RLVR, Error Twins & GPU Throughput`**.

## System Profile: Object · Data · Model · Training · History · Unique Properties

Full code-level architecture, dependency/injector rules, benchmark tables, and history are documented in **[`docs/ARCHITECTURE_AND_HISTORY.md`](docs/ARCHITECTURE_AND_HISTORY.md)**:

- **Object**: Deterministic, zero-runtime-dependency generator, reference-free baseline, and evaluator for liquid-handling protocol **error detection**, **compound multi-fault localization**, **error repair**, and **step ordering**.
- **Data**: Strictly validated 5-step JSON/YAML schema (`transfer`, `mix`, `incubate`, `wash`, `read`) with canonical well normalization (`A1` / `plate1:a1` -> `plate1:A1`), luminescence wavelength-optional support, and 3 MIT-licensed example protocols (`dilution_series.json`, `wash_and_read.json`, `mix_incubate_read.json`).
- **Model**: Cached transitive-reduced well liquid-flow (RAW + WAR) + plate-barrier dependency DAG (`twinstep.dependencies`), 6 seeded single/multi-fault injectors (`swap_order`, `wrong_volume`, `wrong_unit`, `missing_step`, `wrong_temperature`, `wrong_time`), and a reference-blind rule + cycle-aware `heapq` topological baseline (`twinstep.baseline`).
- **Training**: Exports `raw`, `dpo`, `sft`, and `rlvr` datasets for [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain) (no internal neural weights; CPU baseline achieves `83.3%` error detection and `100.0%` ordering constraint satisfaction on `--n 5 --seed 0`).
- **History**: Initial release `8a17ea1` on `main` (`0.1.0`, 29 tests) -> `d598c32` (`review-fixes`, twin deduplication, wash-cycle violation rules, and hand-verified evaluation tests) -> WAR dependencies, multi-cycle topological ordering, `dpo`/`sft`/`rlvr` post-training exports, `error_repair` & `--k-errors` evaluation, and `stepbot`/`labfill` converters (`67` tests).
- **Unique Properties**: Pure Python 3.11+ stdlib portability (with built-in comment/scientific-notation YAML fallback), partial-order constraint evaluation that respects independent step commutativity, and direct interoperability (`twinstep.interop`) with [`stepbot`](https://github.com/yaniguan/stepbot), [`labfill`](https://github.com/yaniguan/labfill), [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain), [`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal), [`lab-verify`](https://github.com/yaniguan/lab-verify), and [`YLM`](https://github.com/yaniguan/YLM) LabLean.

## Repository Layout

```text
twinstep/
├── autolab_wire.py              # Layer L3 Auto-Lab 15-repo wire node (autolab-wire/v1)
├── examples/                    # Bundled valid liquid-handling protocols (JSON)
│   ├── dilution_series.json
│   ├── mix_incubate_read.json
│   └── wash_and_read.json
├── community_extensions/        # Evolutionary protocol saboteur red-teaming (Idea #15)
├── lean/                        # Lean 4.36.0 causal DAG & stateful saboteur proofs (TwinStepLean)
├── src/twinstep/
│   ├── __init__.py              # Public API exports
│   ├── schema.py                # Frozen step dataclasses & JSON + stdlib YAML loaders
│   ├── dependencies.py          # RAW/WAR liquid-flow + barrier DAG, reduction, violations
│   ├── injectors.py             # 6 seeded single/multi-fault injectors & dataset generator
│   ├── baseline.py              # Reference-blind rule checker & cycle-aware ordering baseline
│   ├── evaluate.py              # Error-detection, repair, multi-fault F1, and ordering evaluators
│   ├── formats.py               # `raw`, `dpo`, `sft`, and `rlvr` prompt-completion serializers
│   ├── interop.py               # Bidirectional `stepbot` and `labfill` (0.1 uL) converters
│   └── cli.py                   # `twinstep generate | baseline | eval` CLI
├── docs/
│   └── ARCHITECTURE_AND_HISTORY.md
└── tests/                       # Schema, dependency, injector, baseline/eval, CLI, wire, and audit tests
```

## Quickstart

Generate twins and ordering examples from the three included protocols in `examples/` (glob patterns expand cross-platform on POSIX and Windows shells):

```bash
twinstep generate "examples/*.json" --n 5 --seed 0 -o twins.jsonl
twinstep baseline twins.jsonl -o preds.jsonl
twinstep eval twins.jsonl preds.jsonl
```

Evaluate the built-in rule-based baseline directly, or export preference/RLVR datasets:

```bash
twinstep eval twins.jsonl --baseline
twinstep generate "examples/*.json" --n 5 --format dpo -o twins_dpo.jsonl
twinstep generate "examples/*.json" --n 5 --format rlvr -o twins_rlvr.jsonl
twinstep generate "examples/*.json" --n 5 --k-errors 2 -o twins_k2.jsonl
```

## Example Output

Re-measured with Python 3.11 (`--n 5 --seed 0`):

```text
$ twinstep generate "examples/*.json" --n 5 --seed 0 -o twins.jsonl
Wrote 33 records from 3 protocols to twins.jsonl

$ twinstep baseline twins.jsonl -o preds.jsonl
Wrote 33 baseline predictions to preds.jsonl

$ twinstep eval twins.jsonl preds.jsonl
Error detection: 15/18 (83.3%) | step_index: 10/15 (66.7%)
  - clean             :   3/3   (100.0%)
  - missing_step      :   2/3   (66.7%)
  - swap_order        :   4/4   (100.0%)
  - wrong_time        :   0/1   (0.0%)
  - wrong_unit        :   3/3   (100.0%)
  - wrong_volume      :   3/4   (75.0%)
Ordering: constraints 15/15 (100.0%) | exact 0/15 (0.0%) | pairwise 130/130 (100.0%)
```

Larger sample covering all six error types (`--n 20 --clean 4 --seed 0`):

```text
$ twinstep generate "examples/*.json" --n 20 --clean 4 --seed 0 -o twins20.jsonl
UserWarning: protocol 'mix_incubate_read': no new distinct 'wrong_time' twin after 50 draws; no longer drawing this type
Wrote 123 records from 3 protocols to twins20.jsonl

$ twinstep eval twins20.jsonl --baseline
Error detection: 42/63 (66.7%) | step_index: 32/60 (53.3%)
  - clean             :   3/3   (100.0%)
  - missing_step      :   4/14  (28.6%)
  - swap_order        :  14/14  (100.0%)
  - wrong_temperature :   2/3   (66.7%)
  - wrong_time        :   1/3   (33.3%)
  - wrong_unit        :  13/13  (100.0%)
  - wrong_volume      :   5/13  (38.5%)
Ordering: constraints 60/60 (100.0%) | exact 0/60 (0.0%) | pairwise 520/520 (100.0%)
```

Twins within a protocol always have distinct `corrupted_steps`. `mix_incubate_read` has a single 30 min incubation, so `wrong_time` can only produce `30 s`, `300 min`, or `3 min`; once all three are drawn the type is retired with a warning and the remaining slots go to other error types. `--clean` emits one clean negative per protocol for any value above 0, because extra copies would be identical.

## How It Works

### Protocol Schema

`twinstep.schema` validates protocols as typed frozen dataclasses:

- `transfer`: `source`, `dest` (`source != dest`), `volume`, `unit` (`uL` or `mL`)
- `mix`: `well`, `volume`, `unit` (`uL` or `mL`), `cycles`
- `incubate`: `temperature` (> -273.15 C), `time`, `time_unit` (`min` or `s`), optional `plate`
- `wash`: consistent `plate` and/or `well`, `volume`, `unit` (`uL` or `mL`), `cycles`
- `read`: `plate`, `mode` (`absorbance`, `fluorescence`, or `luminescence`), `wavelength` (nm; optional when `mode == "luminescence"`)

### Error Injectors and `step_index` Semantics

Each injector in `twinstep.injectors` is seeded with `--seed` and guaranteed to produce a sequence distinct from the original:

- `swap_order`: swaps two dependency-ordered steps `(i, j)` (`step_index = i`; `evaluate_records` reports both strict `i` match and relaxed `{i, j}` pair localization).
- `wrong_volume`: scales a step volume by `10x` or `0.1x` (`step_index` in `0..N-1`).
- `wrong_unit`: flips `uL` and `mL` while keeping the numeric volume unchanged (`step_index` in `0..N-1`).
- `missing_step`: drops one step (`step_index` is the 0-based insertion index in `0..len(corrupted_steps)` where the deleted step stood).
- `wrong_temperature`: shifts an `incubate` temperature (`step_index` in `0..N-1`).
- `wrong_time`: flips `min` and `s` or scales duration by `10x` / `0.1x` (`step_index` in `0..N-1`).

### Rule-Based Baseline and Evaluation

`twinstep.baseline` inspects only the candidate `corrupted_steps` (or `shuffled_steps`) without access to `correct_steps` or `gold_order`:

- Flags dependency violations (`read` before `incubate`/`mix` including across an intervening wash; `incubate` before remaining `mix`/`transfer` within the same cycle; `mix` or `transfer` on a plate after a whole-plate wash without a prior `read`; using a well before filling it; mixing before a later addition without re-mixing).
- Checks per-step-type volume bounds (`[1, 300] uL` for well `transfer`/`mix`, `[50, 2000] uL` for `wash`), unit mismatches (including `< 5 mL` well transfers and `0.5 uL` washes), temperature bounds (`[4, 95] C`), duration bounds (`[5, 600] s` or `[1, 180] min`), and optical wavelength bounds (`[190, 1100] nm`).
- Flags uninitialized wells, unmixed multi-reagent wells prior to `transfer`/`incubate`/`wash`/`read`, or a missing `read` step as `missing_step`.
- Reconstructs step ordering via cycle-partitioned topological sort over well liquid-flow edges and plate-stage barriers.

## 15-Repository Auto-Lab Wire Contract (autolab-wire/v1)

`autolab_wire.py` (`REPO_ID = "twinstep"`, `LAYER = "L3"`) implements `process_packet(packet: dict) -> dict` for the 15-repository [Robot AI Auto-Lab](https://github.com/yaniguan/auto-lab) wire pipeline:

- **Upstream links**: Consumes protocol span parameters (`parsed_spans`: `volume_ul`, `temp_c`, `duration_min`) alongside upstream pre-execution and post-run state from [`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal) (`L4`), [`stepbot`](https://github.com/yaniguan/stepbot) (`L1`), and [`lab-attrib`](https://github.com/yaniguan/lab-attrib) (`L4` causal blame & SHA-256 certified ledger). Halted upstream packets (`halted == True`) pass through unchanged.
- **Causal DAG & twin synthesis**: Constructs a validated `twinstep` `Protocol` (`transfer -> mix -> incubate -> read`), extracts transitive-reduced dependency edges via `twinstep.dependencies.find_dependencies`, enforces the Lean `violated_dag_edge_rejected` invariant on any `candidate_step_order` (halting with `halt_reason = "L3_TWINSTEP_DAG_BLOCK"` if violated), and synthesizes seeded corrupted twins via `twinstep.injectors.generate_dataset`.
- **Downstream handoff**: Emits `out["error_twins"]` (`step_count`, `dag_edges`, `corrupted_twin_count`, `fault_types`, and `token_lengths` in `48..240`) directly to downstream [`labfill`](https://github.com/yaniguan/labfill) (`L3` BF16 sequence bin packing) and [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain) (`L3` anti-hack RLVR).

## Limits

- **Ordering metrics**: many step orders are valid, so `exact` (same order as the original file) is strict and low for the baseline. `constraints` and `pairwise` count only the dependency pairs that must hold.
- **Stateless single-protocol checks**: the rule-based baseline cannot detect subtle in-range errors (such as `100 uL -> 10 uL`, `37 C -> 62 C`, `30 min -> 3 min`, or dropping an independent buffer dispense) without a reference specification.
- **Multi-cycle ordering without well-flow cues**: `predict_order` partitions multi-barrier protocols into cycle segments and resolves causal well-flow chains across cycles, but distinct external reagent additions targeting the same well in different cycles cannot be ordered relative to each other from unordered `shuffled_steps` alone.

## License

MIT License. Copyright (c) 2026 Yani Guan.

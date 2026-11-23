# twinstep

`twinstep` is a seeded-error twin generator and evaluator for generic liquid-handling protocol steps. Given valid JSON or YAML protocols (`transfer`, `mix`, `incubate`, `wash`, `read`), it deterministically injects typed protocol errors and generates labeled JSONL pairs for two tasks:

1. **Error detection**: corrupted step sequences paired with their reference steps, `label` (`error_type` or `clean`), `step_index`, `original_value`, `new_value`, and a plain-English `description`.
2. **Step ordering**: shuffled step sequences paired with `gold_order` and minimal dependency `constraints` between steps where order matters.

## Install

```bash
pip install -e .
```

For development (running tests and full YAML support via PyYAML):

```bash
pip install -e .[dev]
```

Requires Python 3.11+. The runtime package uses the Python standard library only.

Part of **[Robot AI Auto-Lab (`auto-lab`)](https://github.com/yaniguan/auto-lab)** ([Project Studio](https://yaniguan.github.io/auto-lab/)) — **Layer `L3 · Post-Training RLVR, Error Twins & GPU Throughput`**.

## System Profile: Object · Data · Model · Training · History · Unique Properties

Full code-level architecture, dependency/injector rules, benchmark tables (`main` vs open PR `origin/review-fixes`), and history are documented in **[`docs/ARCHITECTURE_AND_HISTORY.md`](docs/ARCHITECTURE_AND_HISTORY.md)**:

- **Object**: Deterministic, zero-runtime-dependency generator, reference-free baseline, and evaluator for liquid-handling protocol **error detection** and **step ordering**.
- **Data**: Strictly validated 5-step JSON/YAML schema (`transfer`, `mix`, `incubate`, `wash`, `read`) with 3 MIT-licensed example protocols (`dilution_series.json`, `wash_and_read.json`, `mix_incubate_read.json`).
- **Model**: Transitive-reduced well liquid-flow + plate-barrier dependency DAG (`twinstep.dependencies`), 6 seeded single-fault injectors (`swap_order`, `wrong_volume`, `wrong_unit`, `missing_step`, `wrong_temperature`, `wrong_time`), and a reference-blind rule + `heapq` topological baseline (`twinstep.baseline`).
- **Training**: Generates clean SFT/DPO preference pairs and verifiable RLVR ordering/localization rewards for [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain) (no internal neural weights; CPU baseline achieves `72.2%` error detection on `main` / `77.8%` on `origin/review-fixes` and `100.0%` ordering constraint satisfaction on `--n 5 --seed 0`).
- **History**: Initial release `8a17ea1` on `main` (`0.1.0`, 29 tests) and open PR `origin/review-fixes` (`d598c32`, twin deduplication, wash-cycle violation rules, and hand-verified evaluation tests).
- **Unique Properties**: Pure Python 3.11+ stdlib portability (with built-in YAML fallback), partial-order constraint evaluation that respects independent step commutativity, and direct interoperability with [`labfill`](https://github.com/yaniguan/labfill), [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain), [`stepbot`](https://github.com/yaniguan/stepbot), [`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal), [`lab-verify`](https://github.com/yaniguan/lab-verify), and [`YLM`](https://github.com/yaniguan/YLM) LabLean.

## Repository Layout

```text
twinstep/
├── examples/                    # Bundled valid liquid-handling protocols (JSON)
│   ├── dilution_series.json
│   ├── mix_incubate_read.json
│   └── wash_and_read.json
├── src/twinstep/
│   ├── __init__.py              # Public API exports
│   ├── schema.py                # Frozen step dataclasses & JSON/YAML loaders
│   ├── dependencies.py          # Liquid-flow + barrier DAG, transitive reduction, violations
│   ├── injectors.py             # 6 seeded error injectors & JSONL dataset generator
│   ├── baseline.py              # Reference-blind rule checker & topological order baseline
│   ├── evaluate.py              # Error-detection, step_index, and ordering evaluators
│   └── cli.py                   # `twinstep generate | baseline | eval` CLI
├── docs/
│   └── ARCHITECTURE_AND_HISTORY.md
└── tests/                       # Schema, dependency, injector, baseline/eval, and CLI tests
```

## Quickstart

Generate twins and ordering examples from the three included protocols in `examples/`:

```bash
twinstep generate examples/*.json --n 5 --seed 0 -o twins.jsonl
twinstep baseline twins.jsonl -o preds.jsonl
twinstep eval twins.jsonl preds.jsonl
```

You can also evaluate the built-in rule-based baseline directly without writing an intermediate predictions file:

```bash
twinstep eval twins.jsonl --baseline
```

## Example Output

Measured on a CPU-only Linux box, 2026-11-18.

Running the quickstart commands (`--n 5 --seed 0`):

```text
$ twinstep generate examples/*.json --n 5 --seed 0 -o twins.jsonl
Wrote 33 records from 3 protocols to twins.jsonl

$ twinstep baseline twins.jsonl -o preds.jsonl
Wrote 33 baseline predictions to preds.jsonl

$ twinstep eval twins.jsonl preds.jsonl
Error detection: 13/18 (72.2%) | step_index: 8/15 (53.3%)
  - clean             :   3/3   (100.0%)
  - missing_step      :   1/4   (25.0%)
  - swap_order        :   4/4   (100.0%)
  - wrong_time        :   0/1   (0.0%)
  - wrong_unit        :   3/3   (100.0%)
  - wrong_volume      :   2/3   (66.7%)
Ordering: constraints 15/15 (100.0%) | exact 0/15 (0.0%) | pairwise 130/130 (100.0%)
```

Larger sample covering all six error types (`--n 20 --clean 4 --seed 0`):

```text
$ twinstep generate examples/*.json --n 20 --clean 4 --seed 0 -o twins20.jsonl
Wrote 132 records from 3 protocols to twins20.jsonl

$ twinstep eval twins20.jsonl --baseline
Error detection: 48/72 (66.7%) | step_index: 34/60 (56.7%)
  - clean             :  12/12  (100.0%)
  - missing_step      :   1/14  (7.1%)
  - swap_order        :  10/13  (76.9%)
  - wrong_temperature :   1/3   (33.3%)
  - wrong_time        :   3/4   (75.0%)
  - wrong_unit        :  13/13  (100.0%)
  - wrong_volume      :   8/13  (61.5%)
Ordering: constraints 60/60 (100.0%) | exact 1/60 (1.7%) | pairwise 520/520 (100.0%)
```

## How It Works

### Protocol Schema

`twinstep.schema` validates protocols as typed frozen dataclasses:

- `transfer`: `source`, `dest`, `volume`, `unit` (`uL` or `mL`)
- `mix`: `well`, `volume`, `unit` (`uL` or `mL`), `cycles`
- `incubate`: `temperature` (°C), `time`, `time_unit` (`min` or `s`), optional `plate`
- `wash`: `plate` and/or `well`, `volume`, `unit` (`uL` or `mL`), `cycles`
- `read`: `plate`, `mode` (e.g. `absorbance`, `fluorescence`), `wavelength` (nm)

Three example protocols live in `examples/`:
- `examples/dilution_series.json`: 2-fold water/buffer + dye serial dilution across `plate1:A1`–`A4` followed by an absorbance read.
- `examples/wash_and_read.json`: buffer/dye transfers, mixing, plate wash, and absorbance read.
- `examples/mix_incubate_read.json`: buffer/dye transfers, mixing, 37 °C incubation, and fluorescence read.

### Error Injectors

Each injector in `twinstep.injectors` is seeded with `--seed` and guaranteed to produce a sequence distinct from the original:

- `swap_order`: swaps two dependency-ordered steps (e.g., `read` before `incubate`, or `transfer` after a downstream `mix`).
- `wrong_volume`: scales a step volume by `10x` or `0.1x`.
- `wrong_unit`: flips `uL` and `mL` while keeping the numeric volume unchanged.
- `missing_step`: drops one step from the sequence.
- `wrong_temperature`: shifts an `incubate` temperature.
- `wrong_time`: flips `min` and `s` or scales duration by `10x` / `0.1x`.

### Rule-Based Baseline and Evaluation

`twinstep.baseline` inspects only the candidate `corrupted_steps` (or `shuffled_steps`) without access to `correct_steps` or `gold_order`:

- Flags dependency violations (`read` before `incubate`/`mix`, mixing or transferring from a well before filling it, mixing before a later addition without re-mixing).
- Checks per-step-type volume bounds (`[1, 300] uL` for well `transfer`/`mix`, `[50, 2000] uL` for `wash`), unit mismatches, temperature bounds (`[4, 95] C`), and duration bounds (`[5, 600] s` or `[1, 180] min`).
- Flags uninitialized/unmixed wells or a missing `read` step as `missing_step`.
- Reconstructs step ordering via topological sort over well liquid-flow edges and plate-stage barriers (`transfer` -> `mix` -> `incubate` -> `wash` -> `read`).

## Limits

- **Ordering metrics**: many step orders are valid, so `exact` (same order as the original file) is strict and low for the baseline. `constraints` and `pairwise` count only the dependency pairs that must hold.
- **Stateless single-protocol checks**: the rule-based baseline cannot detect subtle in-range errors (such as `100 uL -> 10 uL`, `37 C -> 62 C`, `30 min -> 30 s`, or dropping an independent buffer dispense) without a reference specification.
- **Dependency assumptions**: ordering constraints model single-plate stage barriers and well-level liquid flow (`reservoir:`/`stock:`/`trough:`/`waste` prefixes are treated as external sources/sinks). Multi-stage protocols that wash, dispense, and wash again are simplified by the stage-ranked ordering baseline.
- **Generic scope**: schemas and examples cover generic liquid handling only (water, buffer, dye, dilution series, plate reads).

## License

MIT License. Copyright (c) 2026 Yani Guan.

# twinstep

`twinstep` is a zero-runtime-dependency (`dependencies = []`) seeded-error twin generator, verifier benchmark, and evaluator for liquid-handling protocol steps. Given valid JSON or YAML protocols (`transfer`, `mix`, `incubate`, `wash`, `read`), it deterministically synthesizes whole-protocol `train` / `val` / `test` splits (`raw`, `dpo`, `mpo`, `gdpo`, `sft`, or `rlvr`) for protocol reasoning and fault-localization evaluation:

1. **Error detection & compound multi-fault debugging**: corrupted step sequences (`--k-errors 1` or `--k-errors k`) paired with ground-truth `label`/`labels`, `step_index`/`step_indices` (`0`-based), `original_value`, `new_value`, and plain-English descriptions (or stripped via `--blind-eval` for reference-blind evaluation).
2. **Step ordering**: shuffled step sequences paired with `gold_order` and minimal transitive-reduced dependency `constraints` (RAW, WAR, and plate barriers).
3. **Error repair**: exact canonical and dependency-equivalent scoring for repaired protocols (`repaired_steps`).
4. **Public wet-lab benchmark adapters & zero-leakage learned localizer**: stdlib-only parsers and SPDX/`eval_only` license guards (`twinstep.open_datasets`) for **WLP**, **BioProBench** (`ERR` & `ORD`), **BioPlanner (`BioProt`)**, and **Opentrons** JSON v6-v8 / Protocol Library, alongside a reference-free graph-feature fault localizer (`twinstep.learned_verifier`) trained strictly on `train`-split protocols.

## Install

```bash
pip install -e .
```

For development (running tests, linters, and optional YAML support via PyYAML):

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+. The core runtime package uses the Python standard library only (`dependencies = []`) and runs entirely on CPU.

Part of **[Robot AI Auto-Lab (`auto-lab`)](https://github.com/yaniguan/auto-lab)** ([Project Studio](https://yaniguan.github.io/auto-lab/)) — **Layer `L3 · Protocol Error Twins, Dependency DAGs & Verifier Benchmarks`**.

## System Profile: Object · Data · Model · Evaluation · History · Unique Properties

Full code-level architecture, dependency/injector rules, benchmark tables, and history are documented in **[`docs/ARCHITECTURE_AND_HISTORY.md`](docs/ARCHITECTURE_AND_HISTORY.md)**:

- **Object**: Deterministic, stdlib-only twin generator, reference-blind rule checker, zero-leakage graph-feature fault localizer, and evaluator for liquid-handling protocol **error detection**, **compound multi-fault localization**, **error repair**, and **step ordering**.
- **Data**: Validated 5-step JSON/YAML schema (`transfer`, `mix`, `incubate`, `wash`, `read`) supporting 96- and 384-well plates (`A1..P24` and zero-padded `A01..P24`), 7 validated optical/imaging read modes, and **24 MIT-licensed bundled protocols** partitioned whole-protocol into **8 `train`**, **4 `val`**, and **12 held-out `test`** assays, plus adapters for 4 public wet-lab corpora (`wlp`, `bioprobench`, `bioplanner`, `opentrons`).
- **Model**: Cached transitive-reduced well liquid-flow (RAW + WAR) + plate-barrier dependency DAG (`twinstep.dependencies`), 6 seeded single/multi-fault injectors (`swap_order`, `wrong_volume`, `wrong_unit`, `missing_step`, `wrong_temperature`, `wrong_time`), a per-well dataflow rule checker + DAG topological ordering baseline (`twinstep.baseline`), and a 26-feature relational/bipartite-symmetry multinomial logistic fault localizer (`twinstep.learned_verifier`).
- **Post-Training Exports & Scope**: Exports `raw`, `dpo`, `mpo`, `gdpo`, `sft`, and `rlvr` JSONL datasets with explicit allowed-label lists and `0`-based `step_index` instructions. `twinstep` contains no GPU training loop and measures no GPU throughput; evaluating whether exported preference pairs improve neural LLM weights requires an external GPU post-training run (e.g. in [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain)).
- **History**: Initial release `8a17ea1` (`0.1.0`, 29 tests) -> `d598c32` (twin deduplication & wash-cycle rules) -> WAR dependencies, multi-fault/repair evaluation, `stepbot`/`labfill` converters, `TwinStepLean` proofs -> multi-stage well-flow rules, 24-protocol `train`/`val`/`test` split isolation, ProcessBench balanced scoring, 4 public protocol corpora, and zero-leakage learned fault localizer (`96` tests).
- **Interoperability**: `twinstep.interop` provides bidirectional converters for **two** sibling formats — [`stepbot`](https://github.com/yaniguan/stepbot) YAML/dicts and [`labfill`](https://github.com/yaniguan/labfill) `0.1 uL` batches — while `autolab_wire.py` exchanges `autolab-wire/v1` JSON packets across the Auto-Lab pipeline.

## Repository Layout

```text
twinstep/
├── autolab_wire.py              # Layer L3 Auto-Lab wire node (autolab-wire/v1)
├── benchmarks/results/          # Reproducible held-out (20-seed) & public-corpus JSON metrics
│   └── frontier_eval_2026.json
├── community_extensions/        # Evolutionary protocol saboteur red-teaming (arXiv:2607.17352)
├── examples/                    # 24 MIT-licensed protocols (8 train / 4 val / 12 held-out test)
├── lean/                        # Lean 4.36.0 causal DAG & saboteur proofs (TwinStepLean)
├── scripts/
│   └── fetch_open_protocols.py  # Downloader, SHA-256 verifier & 4-corpus benchmark runner
├── src/twinstep/
│   ├── __init__.py              # Public API exports
│   ├── schema.py                # Frozen step dataclasses, 96/384-well & stdlib YAML loaders
│   ├── splits.py                # Whole-protocol train/val/test splits & --blind-eval stripper
│   ├── dependencies.py          # RAW/WAR well-flow + barrier DAG, reduction & order checks
│   ├── injectors.py             # 6 seeded single/multi-fault injectors & dataset generator
│   ├── baseline.py              # Reference-blind well-flow checker & DAG topological baseline
│   ├── learned_verifier.py      # Zero-leakage graph-feature fault localizer (stdlib/sklearn/lgbm)
│   ├── open_datasets.py         # WLP, BioProBench, BioPlanner & Opentrons parsers + SPDX guards
│   ├── evaluate.py              # Accuracy, ProcessBench balanced score, repair & split leak checks
│   ├── formats.py               # `raw`, `dpo`, `mpo`, `gdpo`, `sft`, and `rlvr` serializers
│   ├── interop.py               # Bidirectional `stepbot` and `labfill` (0.1 uL) converters
│   └── cli.py                   # `twinstep generate | baseline | eval` CLI
├── docs/
│   └── ARCHITECTURE_AND_HISTORY.md
└── tests/                       # 96 unit, CLI, split-leakage, open-dataset & reproducibility tests
```

## Quickstart & Held-Out Test Split Evaluation

Generate held-out `test`-split twins (`12` unseen protocols, zero overlap with the `8` `train` and `4` `val` protocols) and score the stateless rule baseline:

```bash
twinstep generate "examples/*.json" --split test --n 5 --seed 0 -o test_twins.jsonl
twinstep generate "examples/*.json" --split test --n 5 --seed 0 --blind-eval -o test_blind.jsonl
twinstep baseline test_blind.jsonl -o test_preds.jsonl
twinstep eval test_twins.jsonl test_preds.jsonl
```

Measured output on the **12 held-out `test` protocols** (`--split test --n 5 --seed 0`, `132` records = `72` error-detection + `60` ordering):

```text
$ twinstep generate "examples/*.json" --split test --n 5 --seed 0 -o test_twins.jsonl
Wrote 132 records from 12 protocols to test_twins.jsonl

$ twinstep eval test_twins.jsonl --baseline
Error detection: 41/72 (56.9%) | balanced: 65.2% | step_index: 29/60 (48.3%)
  - clean             :  12/12  (100.0%)
  - missing_step      :   1/9   (11.1%)
  - swap_order        :   7/10  (70.0%)
  - wrong_temperature :   5/12  (41.7%)
  - wrong_time        :   2/10  (20.0%)
  - wrong_unit        :   9/9   (100.0%)
  - wrong_volume      :   5/10  (50.0%)
Ordering: constraints 60/60 (100.0%) | exact 2/60 (3.3%) | pairwise 880/880 (100.0%)
```

### Multi-Seed Held-Out Test Split & Public Wet-Lab Corpora (`benchmarks/results/frontier_eval_2026.json`)

Reproduce the full 20-seed held-out evaluation (`seeds 0..19`) and public-corpus downloads with:

```bash
python scripts/fetch_open_protocols.py
```

#### 1. Held-Out `test` Split (`12` unseen protocols; `train` = `8` protocols, `0` protocol overlap)

| Verifier (evaluated strictly on `test`) | Seed `0` ED Acc (95% Wilson CI) | Seed `0` Balanced Score | Seed `0` `step_index` | 20-Seed ED `mean +- std` (range) | 20-Seed Balanced `mean +- std` | 20-Seed `step_index` `mean +- std` |
|---|---|---|---|---|---|---|
| Stateless Rule Baseline (`twinstep.baseline`) | `41/72` (`56.9%`, `[45.4%, 67.7%]`) | `65.2%` | `29/60` (`48.3%`) | `57.7% +- 4.5%` (`48.6%–63.9%`) | `65.8% +- 5.0%` | `44.2% +- 4.9%` |
| Graph-Feature Verifier (`learned_stdlib`, trained on `8` `train` protocols) | `58/72` (`80.6%`, `[70.0%, 88.1%]`) | `86.8%` | `44/60` (`73.3%`) | `81.3% +- 4.7%` | `87.3% +- 3.7%` | `70.7% +- 5.6%` |
| Graph-Feature Verifier (`learned_sklearn`, optional) | `59/72` (`81.9%`, `[71.5%, 89.1%]`) | `87.9%` | `45/60` (`75.0%`) | — | — | — |
| Graph-Feature Verifier (`learned_lightgbm`, optional) | `52/72` (`72.2%`, `[61.0%, 81.2%]`) | `76.1%` | `41/60` (`68.3%`) | — | — | — |

#### 2. External Public Wet-Lab Protocol Benchmarks (Zero-Shot Transfer)

| Public Corpus | License & Policy | Evaluated Slice | Rule ED Acc | Learned (`stdlib`) ED Acc | Ordering `constraints` (Pairwise) |
|---|---|---|---|---|---|
| **WLP** (`chaitanya2334/WLP-Dataset`) | `MIT` (`training_allowed=True`) | `6` BRAT `.txt`+`.ann` protocols (`36` ED, `30` Ord) | `19/36` (`52.8%`) | `22/36` (`61.1%`) | `5/30` (`16.7%`, pairwise `61.0%`) |
| **BioProBench** (`YuyangSunshine/bioprobench`) | `CC-BY-NC-4.0` (`eval_only=True`) | `160` `ERR` + `100` `ORD` test items | `122/160` (`76.2%`, bal `0.0%`) | `120/160` (`75.0%`, bal `10.0%`) | `6/100` (`6.0%`, pairwise `50.6%`) |
| **BioPlanner / BioProt** (`bioplanner/bioplanner`) | `MIT` (`eval_only=True`) | `6` `protocols.io` JSON protocols (`36` ED, `30` Ord) | `25/36` (`69.4%`) | `25/36` (`69.4%`) | `15/30` (`50.0%`, pairwise `65.9%`) |
| **Opentrons** (`shared-data` v6-v8 + `Opentrons/Protocols`) | `Apache-2.0` (`training_allowed=True`) | `10` JSON protocols (`60` ED, `50` Ord) | `41/60` (`68.3%`) | `42/60` (`70.0%`) | `40/50` (`80.0%`, pairwise `75.2%`) |

## How It Works

### Protocol Schema (`twinstep.schema`)

- `transfer`: `source`, `dest` (`source != dest`), `volume`, `unit` (`uL` or `mL`); supports 96- and 384-well bare or plate-prefixed coordinates (`A1..P24` and `A01..P24`).
- `mix`: `well`, `volume`, `unit` (`uL` or `mL`), `cycles`.
- `incubate`: `temperature` (> `-273.15` C), `time`, `time_unit` (`min` or `s`), optional `plate`.
- `wash`: consistent `plate` and/or `well`, `volume`, `unit` (`uL` or `mL`), `cycles`.
- `read`: `plate`, `mode` (validated against `absorbance`, `fluorescence`, `luminescence`, `imaging`, `turbidity`, `od600`, `plate_reader`), `wavelength` (nm; optional for `luminescence`, `imaging`, `od600`, `plate_reader`).

### Error Injectors, Prompts, and Balanced Scoring

- **6 fault injectors** (`twinstep.injectors`): `swap_order`, `wrong_volume`, `wrong_unit`, `missing_step`, `wrong_temperature`, and `wrong_time`.
- **Self-contained prompts** (`twinstep.formats`): every `error_detection` and `multi_error_detection` prompt explicitly lists all allowed error labels and specifies `step_index is 0-based`.
- **Label normalization & ProcessBench balanced score** (`twinstep.evaluate`): `evaluate_records` normalizes case, spaces, and hyphens in predicted labels (reporting `normalized_predictions`), checks for `train`/`test` `protocol_id` overlap (`UserWarning`), and reports the ProcessBench harmonic mean `2 * acc_clean * acc_err / (acc_clean + acc_err)` (arXiv:2412.06559) so always-error predictors score `0.0%`.

### Rule Checker & Ordering Scope (`twinstep.baseline`)

- Uses per-well liquid dataflow across `IncubateStep`, `WashStep`, and `ReadStep` barriers so multi-stage assays (e.g. ELISA overnight `960 min` coating + blocking + TMB + acid stop, or `1440 min` 24 h compound incubation + viability reagent + mix) pass with zero false positives (`max_incubate_min = 4320.0` min / `72 h` by default, configurable).
- Reconstructs step orderings via inferred causal DAG edges + Kahn's topological sort with stable input-position tie-breaking. While this achieves `100.0%` constraint satisfaction on the structured bundled protocols, open-world imported protocols (`WLP`, `BioProBench` `ORD`, `BioPlanner`) show `6.0%–80.0%` constraint satisfaction when ordering cues depend on free-text biological semantics.

### Evolutionary Saboteur Extension (`community_extensions/saboteur_evolver.py`)

- Inspired by **LeanEvolve** (`arXiv:2607.17352`), `saboteur_evolver.py` evolves candidate mutations against weak stateless gates (`Gen-0` scalar bounds and `Gen-1` op-presence bitmasks) using an `11`-rule stateful Python oracle (`check_gen2_stateful_lean_gate`) that extends the `5` core rules mechanized in `lean/TwinStepLean.lean` (`no user axioms`). Because saboteurs are defined as candidates failing `Gen-2`, `0.0%` `Gen-2` evasion holds by construction; the empirical metric is `Gen-0`/`Gen-1` evasion growth across generations.

## License

MIT License. Copyright (c) 2026 Yani Guan.

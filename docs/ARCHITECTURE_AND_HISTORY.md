# `twinstep` — Architecture, Benchmarks, and Engineering History

> **Auto-Lab Ecosystem Role:** Layer `L3 · Protocol Error Twins, Dependency DAGs & Verifier Benchmarks` in [Robot AI Auto-Lab (`auto-lab`)](https://github.com/yaniguan/auto-lab) ([Project Studio](https://yaniguan.github.io/auto-lab/)).

---

## 1. Object

### Problem Statement
Benchmarking and post-training protocol verifiers require (1) held-out multi-stage assays that cannot be solved by memorizing 3 small seed templates, (2) partial-order step-sequencing evaluation that respects commuting operations, and (3) balanced metrics that penalize false alarms on clean multi-stage protocols.

`twinstep` is a **zero-runtime-dependency** (Python 3.11+ standard library, CPU-only) seeded-error twin generator, reference-blind verifier suite, open-corpus adapter, and evaluator for liquid-handling protocols (`transfer`, `mix`, `incubate`, `wash`, `read`).

### I/O Contract & Scope Boundaries

| Command / Module | Inputs | Outputs |
|---|---|---|
| `twinstep generate` (`injectors.py`, `splits.py`) | Protocol `.json`/`.yaml` files, `--split`, `--blind-eval`, `--n`, `--seed`, `--format` | JSONL stream/file of `TwinRecord`, `MultiTwinRecord`, and `OrderingRecord` dicts |
| `twinstep baseline` (`baseline.py`) | `twins.jsonl` (inspects **only** `corrupted_steps` / `shuffled_steps`) | JSONL predictions (`label`, `step_index`, `reason` or `pred_order`) |
| `twinstep.learned_verifier` | `train`-split `TwinRecord`s (enforces zero `train`/`test` `protocol_id` overlap) | Reference-free 26-feature multinomial logistic / sklearn / LightGBM fault localizer |
| `twinstep eval` (`evaluate.py`) | Ground-truth `twins.jsonl` + `preds.jsonl` (`--split`, `--train-twins`) | Plain accuracy, ProcessBench balanced score, `step_index`, `normalized_predictions`, per-split breakdown & leakage warnings |
| `twinstep.open_datasets` | Cached/downloaded `WLP`, `BioProBench`, `BioPlanner`, `Opentrons` files | Validated `Protocol` / `TwinRecord` / `OrderingRecord` instances with SPDX & `eval_only` guards |

**Scope Boundaries:**
- `twinstep` contains no GPU code and measures no GPU throughput; evaluating whether exported `dpo`/`mpo`/`gdpo`/`sft`/`rlvr` datasets improve neural LLM checkpoints requires an external GPU post-training run.
- `twinstep.interop` implements **two** format converters (`stepbot` YAML/dicts and `labfill` `0.1 uL` batches); broader ecosystem coordination uses JSON packets via `autolab_wire.py`.

---

## 2. Data & Whole-Protocol Splits

### Validated Protocol Step Schema (`src/twinstep/schema.py`)

| Step Type | Dataclass | Validation Rules |
|---|---|---|
| `transfer` | `TransferStep` | Distinct `source` and `dest`; 96/384-well coordinates (`A1..P24` and `A01..P24`); finite `volume > 0`; `unit` in `("uL", "mL")`. |
| `mix` | `MixStep` | Non-empty `well`; finite `volume > 0`; integer `cycles >= 1`; `unit` in `("uL", "mL")`. |
| `incubate` | `IncubateStep` | Finite `temperature > -273.15` C; finite `time > 0`; `time_unit` in `("min", "s")`; optional `plate`. |
| `wash` | `WashStep` | Consistent `well` and/or `plate`; finite `volume > 0`; integer `cycles >= 1`. |
| `read` | `ReadStep` | Non-empty `plate`; `mode` in `("absorbance", "fluorescence", "luminescence", "imaging", "turbidity", "od600", "plate_reader")`; integer `wavelength >= 1` nm (optional for luminescence/imaging/od600/plate_reader). |

### 24 Bundled MIT Protocols (`examples/`, `src/twinstep/splits.py`)
Whole protocols (never individual twins) are partitioned into disjoint splits:
- **`train` (`8` protocols)**: `dilution_series`, `mix_incubate_read`, `wash_and_read`, `bca_protein_assay`, `qpcr_sybr_setup`, `plasmid_miniprep_elution`, `mtt_viability_assay`, `direct_elisa_coating`.
- **`val` (`4` protocols)**: `bradford_coomassie_assay`, `crispr_rnp_cleavage`, `kinetic_wash_reprobe`, `turbidity_od600_growth`.
- **`test` (`12` held-out protocols)**: `sandwich_elisa`, `rt_qpcr_mastermix`, `spri_magnetic_bead_cleanup`, `ic50_384well_serial_dilution`, `gibson_assembly_reaction`, `western_blot_wash_probe`, `cell_viability_luminescent`, `lal_endotoxin_chromogenic`, `picogreen_dna_quant`, `enzyme_quench_kinetics`, `high_content_cell_imaging`, `competitive_immunoassay_384`.

---

## 3. Verifier Models & Formal Guarantees

1. **Per-Well Liquid Dataflow & DAG Baseline (`dependencies.py`, `baseline.py`)**:
   - Tracks well initialization and stage-local multi-reagent mixing across `IncubateStep`, `WashStep`, and `ReadStep` barriers so multi-stage assays (overnight `960 min` coating, `1440 min` compound treatment, post-incubation acid stop solutions) produce zero false alarms (`DEFAULT_MAX_INCUBATE_MIN = 4320.0`).
   - Reconstructs step orderings via causal well-flow + cycle-segmented DAG edges and Kahn's topological sort with stable input-position tie-breaking.
2. **Zero-Leakage Graph-Feature Fault Localizer (`learned_verifier.py`)**:
   - Extracts 26 reference-free relational features (same-reagent peer volume log-ratios, per-well volume ratios, bipartite `(reagent, well)` matrix-completion deficits, mix-symmetry deficits, canonical temperature offsets, and second/minute duration anomalies) and ranks candidate fault step indices without inspecting `correct_steps` or `protocol_id`.
3. **Lean 4 Formalization (`lean/TwinStepLean.lean`) & Saboteur Red-Teaming (`community_extensions/saboteur_evolver.py`)**:
   - `TwinStepLean` (`leanprover/lean4:v4.36.0`, `0 sorry`, no user axioms — uses standard Lean kernel `propext` and `Quot.sound` only) proves bijective permutation soundness, Hasse transitive-reduction equivalence/minimality, contiguous-chain topological uniqueness, and 5 core physical/causal rules.
   - `saboteur_evolver.py` (inspired by LeanEvolve, `arXiv:2607.17352`) evaluates 11 Python rules (the 5 Lean rules + 6 trace-hygiene rules) as the ground-truth oracle (`0.0%` Gen-2 evasion holds by construction) to measure Gen-0/Gen-1 heuristic evasion.

---

## 4. Reproducible Held-Out & Open-Corpus Benchmarks

All numbers below are generated by `python scripts/fetch_open_protocols.py` and stored in `benchmarks/results/frontier_eval_2026.json` (`96` passing `pytest` tests verify reproducibility):

- **Held-Out `test` Split (`12` protocols, `n_twins=5`, `20` seeds `0..19`, `0` train/test overlap)**:
  - **Rule Baseline**: Seed `0` ED `41/72` (`56.9%`, 95% Wilson CI `[45.4%, 67.7%]`), ProcessBench balanced `65.2%`, `step_index` `29/60` (`48.3%`); 20-seed mean `57.7% +- 4.5%` (balanced `65.8% +- 5.0%`, `step_index` `44.2% +- 4.9%`).
  - **Learned Verifier (`stdlib`)**: Seed `0` ED `58/72` (`80.6%`, 95% Wilson CI `[70.0%, 88.1%]`), ProcessBench balanced `86.8%`, `step_index` `44/60` (`73.3%`); 20-seed mean `81.3% +- 4.7%` (balanced `87.3% +- 3.7%`, `step_index` `70.7% +- 5.6%`).
- **Open Wet-Lab Corpora**:
  - **WLP** (`MIT`, `6` BRAT protocols): Rule `52.8%` vs Learned `61.1%` ED; `16.7%` ordering constraints (`61.0%` pairwise).
  - **BioProBench** (`CC-BY-NC-4.0`, `eval_only=True`, `160` `ERR` + `100` `ORD`): Rule `76.2%` (bal `0.0%`) vs Learned `75.0%` (bal `10.0%`); `6.0%` ordering constraints (`50.6%` pairwise).
  - **BioPlanner / BioProt** (`MIT`, `eval_only=True`, `6` JSON protocols): `69.4%` ED; `50.0%` ordering constraints (`65.9%` pairwise).
  - **Opentrons** (`Apache-2.0`, `10` JSON v6-v8 & library builds): Rule `68.3%` vs Learned `70.0%` ED; `80.0%` ordering constraints (`75.2%` pairwise).

---

## 5. Engineering History

| Milestone | Summary of Technical Changes |
|---|---|
| `8a17ea1` (`0.1.0`) | Initial 5-step schema, transitive-reduced dependency DAG, 6 error injectors, rule baseline, CLI, 3 example protocols, 29 tests. |
| `d598c32` | Per-protocol twin deduplication (`MAX_DRAW_ATTEMPTS = 50`), wash-cycle scanning, hand-computed metric tests. |
| Issues `#3–#33` | WAR anti-dependencies, multi-fault `--k-errors` & `error_repair` scoring, `dpo`/`mpo`/`gdpo`/`sft`/`rlvr` exports, `stepbot`/`labfill` converters, `TwinStepLean` Lean 4.36.0 proofs, `autolab_wire.py`. |
| Issues `#35–#38` | Per-well dataflow barrier rules & DAG topological ordering (`4320 min` bound, `A01..P24` 96/384-well coordinates, validated read modes); 24 MIT protocols with `8/4/12` whole-protocol `train`/`val`/`test` split isolation & `--blind-eval`; prompt label/0-based specs, label normalization & ProcessBench balanced score; `WLP`, `BioProBench`, `BioPlanner`, and `Opentrons` open-corpus adapters; zero-leakage graph-feature fault localizer (`96` tests). |

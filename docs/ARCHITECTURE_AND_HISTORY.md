# `twinstep` — Architecture, Benchmarks, and Engineering History

> **Auto-Lab Ecosystem Role:** Layer `L3 · Post-Training RLVR, Error Twins & GPU Throughput` in [Robot AI Auto-Lab (`auto-lab`)](https://github.com/yaniguan/auto-lab) ([Project Studio](https://yaniguan.github.io/auto-lab/)).

---

## 1. Object

### Problem Statement
Post-training and benchmarking language models on laboratory protocols require reliable negative examples (corrupted protocols with exact fault localization) and partial-order step-sequencing tasks where independent steps are not falsely penalized. Public text corpora often carry restrictive non-commercial licenses or noisy, unverified labels.

`twinstep` is a **zero-runtime-dependency** (Python 3.11+ standard library) seeded-error twin generator, reference-free baseline checker, and evaluator for generic liquid-handling protocols (`transfer`, `mix`, `incubate`, `wash`, `read`). From any valid JSON or YAML protocol, it deterministically synthesizes JSONL benchmarks for two tasks:
1. **Error Detection (`task = "error_detection"`)**: Pairs `corrupted_steps` with `correct_steps`, ground-truth `label` (`swap_order`, `wrong_volume`, `wrong_unit`, `missing_step`, `wrong_temperature`, `wrong_time`, or `clean`), `step_index`, `original_value`, `new_value`, and human-readable `description`.
2. **Step Ordering (`task = "ordering"`)**: Pairs `shuffled_steps` with `gold_order` and minimal transitive-reduced dependency `constraints` (`[a, b]` indices into `shuffled_steps`).

### Role in the Auto-Lab `L0`–`L6` Hierarchy
Within the [Auto-Lab](https://github.com/yaniguan/auto-lab) Brain/Body architecture:
- **Layer (`L3`)**: Supplies commercially clean (MIT) contrastive preference pairs (`correct_steps` vs. `corrupted_steps`), fault-localization targets, and constraint-based step-ordering rewards for SFT, DPO, and RLVR in [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain), complementing the tensor-batched state-machine twins in [`labfill`](https://github.com/yaniguan/labfill).

### I/O Contract & Scope Boundaries

| Command / Module | Inputs | Outputs |
|---|---|---|
| `twinstep generate` (`injectors.py`) | Protocol `.json`/`.yaml` files, `--n`, `--clean`, `--ordering`, `--seed` | JSONL stream/file of `TwinRecord` and `OrderingRecord` dicts |
| `twinstep baseline` (`baseline.py`) | `twins.jsonl` (inspects **only** `corrupted_steps` / `shuffled_steps`) | JSONL predictions (`label`, `step_index`, `reason` or `pred_order`) |
| `twinstep eval` (`evaluate.py`) | Ground-truth `twins.jsonl` + `preds.jsonl` (or `--baseline`) | Text or `--json` metrics (overall/per-type accuracy, `step_index`, `constraints`, `exact`, `pairwise`) |

**Scope Boundaries:**
- Covers generic liquid-handling primitives (`water`, `buffer`, `dye`, serial dilutions, plate washes, incubations, absorbance/fluorescence reads); does not model 3D robot kinematics, tip-rack geometry, or reagent chemistry.
- The built-in `baseline` intentionally operates without a reference specification, establishing an honest floor for what single-protocol sanity and flow rules can catch versus subtle in-range parameter shifts (`100 uL -> 10 uL`, `37 C -> 62 C`, `30 min -> 30 s`).

---

## 2. Data

### Validated Protocol Step Schema (`src/twinstep/schema.py`)
Protocols are parsed into `Protocol(protocol_id, steps)` containing five frozen, strictly validated dataclasses:

| Step Type | Dataclass | Required & Optional Fields | Validation Rules |
|---|---|---|---|
| `transfer` | `TransferStep` | `source: str`, `dest: str`, `volume: float`, `unit: "uL" \| "mL" = "uL"` | Non-empty strings; finite `volume > 0` (`bool` rejected); valid `VolumeUnit`. |
| `mix` | `MixStep` | `well: str`, `volume: float`, `cycles: int`, `unit: "uL" \| "mL" = "uL"` | Non-empty `well`; finite `volume > 0`; integer `cycles >= 1` (`bool` rejected). |
| `incubate` | `IncubateStep` | `temperature: float`, `time: float`, `time_unit: "min" \| "s" = "min"`, `plate: str \| None = None` | Finite `temperature > -273.15` °C; finite `time > 0`; valid `TimeUnit` (also accepts `unit` alias on load). |
| `wash` | `WashStep` | `volume: float`, `cycles: int`, `well: str \| None = None`, `plate: str \| None = None`, `unit: "uL" \| "mL" = "uL"` | At least one of `well` or `plate` non-empty; finite `volume > 0`; `cycles >= 1`. |
| `read` | `ReadStep` | `plate: str`, `mode: str`, `wavelength: int` | Non-empty `plate` and `mode`; integer `wavelength >= 1` nm. |

JSON files load via `json.loads`; `.yaml`/`.yml` files use `yaml.safe_load` when `PyYAML` is installed and fall back to a built-in subset parser (`_parse_simple_yaml`).

### Bundled Example Protocols (`examples/`, MIT License)
1. **`examples/dilution_series.json` (`13` steps)**: Dispenses `100 uL` `reservoir:buffer` into `plate1:A1..A4` (4 independent transfers), adds `100 uL` `reservoir:dye` to `plate1:A1`, performs three 2-fold `100 uL` serial transfers + 5-cycle `100 uL` mixes (`A1 -> A2 -> A3 -> A4`), and reads `plate1` absorbance at `450 nm`.
2. **`examples/wash_and_read.json` (`8` steps)**: Dispenses `150 uL` buffer and `50 uL` dye into `plate1:B1` and `B2`, mixes both wells (`100 uL`, 4 cycles), runs a whole-plate wash (`plate1`, `300 uL`, 3 cycles), and reads absorbance at `595 nm`.
3. **`examples/mix_incubate_read.json` (`8` steps)**: Dispenses `180 uL` buffer and `20 uL` dye into `plate1:C1` and `C2`, mixes both wells (`100 uL`, 6 cycles), incubates `plate1` at `37 °C` for `30 min`, and reads fluorescence at `520 nm`.

---

## 3. Model

### 3.1 Liquid-Flow & Stage-Barrier Dependency Graph (`src/twinstep/dependencies.py`)
- **Location Semantics**: Prefixes `reservoir`, `stock`, `trough`, and `waste` denote unconstrained external sources/sinks (`is_well_ref == False`, `plate_of == None`). `plate:well` strings resolve to `plate`; bare well coordinates matching `^[A-Ha-h](?:[1-9]|1[0-2])$` resolve to default `"plate1"`.
- **Direct Dependency Edges (`_has_direct_dependency`)**: Places directed edge `i -> j` (`i < j`) when:
  - Well-level liquid flow links `TransferStep.dest` to a later `TransferStep.source`, `MixStep.well`, or well-targeted `WashStep.well`; `MixStep.well` to a later `TransferStep.source` or `WashStep.well`; or `WashStep.well` to a later `TransferStep.dest` or `MixStep.well`.
  - Either step is a plate-level stage barrier (`IncubateStep`, `ReadStep`, or whole-plate `WashStep(plate != None, well == None)`) and their target plates overlap (global `IncubateStep(plate=None)` overlaps any plate-touching step or other `IncubateStep`).
- **Transitive Reduction (`_transitive_reduction`)**: Computes reverse-order reachability sets and prunes any chord `(u, v)` implied by `u -> mid -> ... -> v`, yielding minimal non-redundant ordering `constraints`.
- **Reference-Free Anomaly Detection (`find_order_violations`)**: Scans candidate step lists for (1) mixing or aspirating from a well prior to its first fill, (2) aspirating from `W` before a later mix of `W` without an intervening refill, (2b) mixing `W` followed by a refill into `W` with no remix before wash/end, (3) `read` before `incubate`/`mix`/plate preparation *(extended on open PR `origin/review-fixes` `d598c32` to keep scanning across intervening washes and add rule `3b` flagging `mix`/`transfer` after a whole-plate wash with no intervening `read`)*, and (4) `incubate` before any liquid handling on its target plate.

### 3.2 Seeded Single-Fault Injectors & Ordering Shuffler (`src/twinstep/injectors.py`)
`inject_error(protocol, error_type, rng)` applies one deterministic mutation and asserts `corrupted_steps != correct_steps`:
- `swap_order`: Samples `(i, j)` from the transitive closure of `find_dependencies(steps)` with `steps[i] != steps[j]` and swaps them (`step_index = i`).
- `wrong_volume`: Multiplies a `transfer`/`mix`/`wash` volume by `10.0` or `0.1`.
- `wrong_unit`: Flips `uL <-> mL` while preserving the numeric scalar (`1000x` physical error).
- `missing_step`: Deletes one step at `rng.randrange(len(steps))` (`len(steps) >= 2`).
- `wrong_temperature`: Shifts an `IncubateStep` temperature by `delta in (-40, -25, +25, +60, +85) °C` (keeping `> -273.15 °C`).
- `wrong_time`: Either flips `min <-> s` (`mode="unit"`) or scales duration by `10.0` (`time < 60`) / `0.1` (`time >= 60`) (`mode="scale"`).

`make_ordering_record` draws up to 50 random permutations until `shuffled_steps != correct_steps` and `[0, 1, ..., n-1]` violates at least one mapped constraint (falling back to index reversal), emitting `gold_order = inv_perm` and `constraints` indexed in `shuffled_steps`.

### 3.3 Rule-Based Checker & Priority-Heap Topological Baseline (`src/twinstep/baseline.py`)
Without peeking at `correct_steps` or `gold_order`:
- **`predict_error`**: Checks (1) `find_order_violations` -> `swap_order`; (2) `transfer`/`mix` with `unit == "mL"` and `volume >= 5.0` -> `wrong_unit`, or `vol_ul` outside `[1, 300] uL` -> `wrong_volume`; `wash` with `mL >= 10.0` or `uL <= 5.0` -> `wrong_unit`, or `vol_ul` outside `[50, 2000] uL` -> `wrong_volume`; (3) `incubate` outside `[4, 95] °C` -> `wrong_temperature`, or outside `[5, 600] s` / `[1, 180] min` -> `wrong_time`; (4) aspirating/mixing an uninitialized well, aspirating from a multi-fill unmixed well, or lacking any `ReadStep` (`len >= 2`) -> `missing_step`; otherwise `"clean"`.
- **`predict_order`**: Constructs a directed graph from pairwise well liquid-flow edges and same-plate stage-rank ordering (`transfer=0 < mix=1 < incubate=2 < wash=3 < read=4`) and runs Kahn's topological sort using a `heapq` min-heap keyed by `(stage_rank, shuffled_idx)` (with an indegree-sorted fallback if cyclic).

---

## 4. Training

### Supervision & Evaluation Role in Post-Training
`twinstep` contains **no trainable neural weights** and requires no GPU. Instead, it serves two training roles in Layer `L3`:
1. **Contrastive & SFT Dataset Synthesis**: `TwinRecord` supplies aligned `(correct_steps, corrupted_steps, label, step_index, description)` tuples directly usable as DPO preference pairs, SFT critique/repair examples, or token/step-classification targets.
2. **Verifiable Ordering & Detection Rewards (RLVR)**: `evaluate_records` (`src/twinstep/evaluate.py`) scores model outputs by `id` (with positional fallback when `pred["id"] is None`):
   - **Error Detection**: Overall `accuracy`, per-class recall (`per_error_type`), and strict joint `step_index_accuracy` (crediting localization only when **both** `pred["label"] == gold["label"]` and `pred["step_index"] == gold["step_index"]`).
   - **Step Ordering**: Validates permutation bijectivity (`len == n` and `set(pred_order) == range(n)`), then computes `constraint_accuracy` (all DAG constraints satisfied), `pairwise_accuracy` (fraction of individual `[a, b]` edges satisfied), and `exact_accuracy` (exact match to `gold_order`).

### Measured CPU Results (`main` vs. Open PR `origin/review-fixes`)
All measurements were recorded on CPU (2026-11-18); local test suite verification on 2026-11-22 passed `29/29` tests in `0.37s` (`pytest 8.4.2`, Python `3.13.13`).

| Benchmark Setting (`seed = 0`, 3 example protocols) | Branch / Commit | Records (`ED` / `Ord`) | Error Detection Acc | Joint `step_index` Acc | Ordering `constraints` | Ordering `pairwise` | Ordering `exact` |
|---|---|---:|---:|---:|---:|---:|---:|
| `--n 5 --clean 1` | `main` (`8a17ea1`) | `33` (`18` / `15`) | `13/18` (`72.2%`) | `8/15` (`53.3%`) | `15/15` (`100.0%`) | `130/130` (`100.0%`) | `0/15` (`0.0%`) |
| `--n 20 --clean 4` | `main` (`8a17ea1`) | `132` (`72` / `60`) | `48/72` (`66.7%`) | `34/60` (`56.7%`) | `60/60` (`100.0%`) | `520/520` (`100.0%`) | `1/60` (`1.7%`) |
| `--n 5 --clean 1` | `origin/review-fixes` (`d598c32`) | `33` (`18` / `15`) | `14/18` (`77.8%`) | `9/15` (`60.0%`) | `15/15` (`100.0%`) | `130/130` (`100.0%`) | `0/15` (`0.0%`) |
| `--n 20 --clean 4` | `origin/review-fixes` (`d598c32`) | `123` (`63` / `60`) | `43/63` (`68.3%`) | `33/60` (`55.0%`) | `60/60` (`100.0%`) | `520/520` (`100.0%`) | `0/60` (`0.0%`) |

Per-class behavior highlights the gap that learned models are meant to close: the stateless baseline achieves `100%` on `clean` and `wrong_unit` (and `100%` on `swap_order` once `d598c32` wash rules are applied), `53.8%–75.0%` on `wrong_volume`, `33.3%–50.0%` on `wrong_temperature`, `0.0%–75.0%` on `wrong_time`, and `7.1%–33.3%` on `missing_step` (since dropping an independent buffer transfer leaves remaining steps locally plausible).

---

## 5. History

| Commit | Branch / Ref | Date | Summary of Technical Changes |
|---|---|---|---|
| `8a17ea1` | `main` (`origin/main`) | 2026-11-18 | Initial release (`0.1.0`): validated 5-step schema & JSON/YAML loader (`schema.py`), liquid-flow & barrier dependency graph with transitive reduction (`dependencies.py`), 6 seeded error injectors & dataset generator (`injectors.py`), rule-based checker & heap topological baseline (`baseline.py`), multi-metric evaluator (`evaluate.py`), `twinstep` CLI (`cli.py`), 3 example protocols, 29 unit/CLI tests, and Python 3.11/3.12 CI. |
| `d598c32` | `origin/review-fixes` (open PR) | 2026-11-18 | Enforced per-protocol uniqueness of `corrupted_steps` in `generate_dataset` (`MAX_DRAW_ATTEMPTS = 50` with `UserWarning` retirement when small protocols exhaust discrete mutations) and capped `clean` output to at most 1 identical copy per protocol; refined `find_order_violations` wash-cycle scanning (`3` and `3b`); added hand-computed metric and `_match_predictions` fallback tests; updated README tables. |

---

## 6. Unique Properties

### Key Invariants & Design Guarantees
1. **Pure-Stdlib Runtime Portability**: Runs out of the box on Python 3.11+ with `dependencies = []`, including a fallback YAML parser when `PyYAML` is absent.
2. **Exact Single-Edit Provenance**: Every `TwinRecord` guarantees `corrupted_steps != correct_steps` and records exact `step_index`, `original_value`, and `new_value` metadata.
3. **Partial-Order Fairness**: `find_dependencies` applies transitive reduction and omits edges between independent well operations (verified in `test_independent_buffer_transfers_are_unconstrained`), so `constraint_accuracy` and `pairwise_accuracy` evaluate true physical precedence rather than arbitrary file order.
4. **Strict Reference Blindness in Baseline**: `predict_record` never reads `correct_steps`, `gold_order`, or `constraints`, preventing data leakage in baseline comparisons.

### Known Limits
- **Single-Stage Barrier Simplification**: `predict_order` assigns fixed stage ranks (`transfer < mix < incubate < wash < read`), which simplifies multi-wash or multi-read ELISA-style cycles unless segmented per cycle.
- **In-Range Blindness Without Reference**: Without protocol-level intent (e.g., expected dilution ratio or target assay temperature), stateless rules cannot flag in-range corruptions such as `180 uL -> 18 uL` or `37 °C -> 62 °C`.

### Ecosystem Interfaces
- **Peer `L3` Post-Training Stack**: Complements [`labfill`](https://github.com/yaniguan/labfill) (tensor-batched state-machine twins & single-edit repair) and supplies JSONL supervision/evaluation suites for [`labbench-posttrain`](https://github.com/yaniguan/labbench-posttrain).
- **Downstream Execution & Verification (`L2` / `L4` / Body)**: Shares step-level JSON/YAML liquid-handling representations with [`stepbot`](https://github.com/yaniguan/stepbot) (Opentrons Python API v2 compiler), [`lab-rehearsal`](https://github.com/yaniguan/lab-rehearsal) (deck simulation & memory recall), [`lab-verify`](https://github.com/yaniguan/lab-verify), and [`YLM`](https://github.com/yaniguan/YLM) (`lean/LabLean`).

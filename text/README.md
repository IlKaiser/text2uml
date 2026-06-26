# `text/` — Linguistic-Complexity Metrics

Implements the linguistic-complexity metrics from the ICSOC2026 paper, computes
them on every `dataset/*/description.md`, and renders a **complexity profile**
heatmap that compares the datasets against one another.

The paper's thesis: with the prompt held fixed, the **linguistic complexity of
the input text** drives model behaviour. This package quantifies that complexity
so it can be inspected per dataset and, optionally, related to the recorded LLM
F1 scores in `dataset/evaluation_results_llm.csv`.

---

## Metrics

| Metric | Type | What it measures | Direction |
|---|---|---|---|
| `mdd` | exact | Mean Dependency Distance — mean `\|pos(head)−pos(dep)\|` over dependency arcs | higher = more complex |
| `subordination_index` | exact | `Csub / Ctotal` — share of subordinate clauses | higher = more complex |
| `parse_tree_depth` | exact | Mean per-sentence dependency-tree depth | higher = more complex |
| `parse_tree_depth_max` | exact | Maximum dependency-tree depth across sentences | higher = more complex |
| `flesch_reading_ease` | exact | Flesch Reading Ease | **higher = simpler** (inverted) |
| `rst_depth_proxy` | proxy | Rhetorical cue-phrase density per sentence (stands in for RST tree richness) | higher = more complex |
| `context_dependence_proxy` | proxy | Share of referring expressions with no local antecedent (stands in for CDR) | higher = more complex |
| `inferential_load_proxy` | proxy | Inference triggers per sentence (stands in for ILI) | higher = more complex |

The last three are **deterministic surface proxies** for theoretical constructs
(RST depth, Context Dependence Ratio, Inferential Load Index) that would
otherwise need a discourse parser or an LLM. Each class docstring in
[`metrics/discourse.py`](metrics/discourse.py) states exactly which surface
signal it uses. Everything runs offline — no API keys.

`flesch_reading_ease` is the one metric where a **higher** value means **less**
complexity; the others increase with complexity. This matters for the overall
ordering described below.

---

## How metrics are computed (pipeline)

1. [`pipeline.py`](pipeline.py) walks `dataset/<Name>/description.md` (sorted),
   reads each description as UTF-8.
2. Each text is parsed **once** with spaCy (`en_core_web_sm`, loaded as a shared
   singleton in [`metrics/base.py`](metrics/base.py)).
3. Every metric in the registry ([`metrics/__init__.py`](metrics/__init__.py)) is
   computed on that parse. Registration order = column order in the CSV.
4. A failed parse or a single failing metric yields `nan` for the affected
   values (the batch continues); empty text yields all-`nan` with an `error`.
5. One row per dataset is written to `text/output/complexity_metrics.csv` with
   `sub_folder_name`, `n_tokens`, `n_sentences`, every metric, and `error`.

---

## The complexity-profile heatmap

[`plots.py → plot_complexity_profile`](plots.py) renders the only figure
produced by default. Three things happen to the raw metrics table before it is
drawn.

### 1. Z-scoring (drives cell **colour**)

Each metric is standardised **independently, across datasets**:

```python
z = (data - data.mean()) / data.std(ddof=0)
```

For dataset *i* and metric *m*:

```
z[i, m] = (x[i, m] − mean_m) / std_m
```

- `mean_m`, `std_m` are taken **column-wise** — per metric, over all datasets.
- `ddof=0` ⇒ **population** standard deviation (divide by N, not N−1).
- Result: every metric column has mean 0 and std 1, so heterogeneous scales
  (Flesch ≈ 50 vs parse depth ≈ 3) become directly comparable in one image.

The heatmap uses `cmap="RdBu_r", center=0`, so:

- **red** cell  → this dataset is **above** the metric's average,
- **blue** cell → **below** the metric's average,
- white/pale    → near the average.

The colour bar is labelled `z-score`.

### 2. Cell annotations (the **raw** values)

Colour encodes the z-score, but the number printed inside each cell is the
**actual metric value** (`fmt=".2f"`), reordered to match the row order. So a
reader sees both the relative position (colour) and the real measurement
(e.g. `flesch_reading_ease = 62.50`, `parse_tree_depth = 3.00`).

### 3. Row ordering (increasing overall complexity)

Rows are **not** alphabetical. An overall complexity score per dataset is the
mean z-score across all metrics, with `flesch_reading_ease` **sign-flipped**
(because higher Flesch = simpler):

```python
oriented = z.copy()
oriented["flesch_reading_ease"] = -oriented["flesch_reading_ease"]
complexity = oriented.mean(axis=1)
order = complexity.sort_values().index   # ascending
```

Datasets are sorted by that score, so the **simplest dataset is at the top** and
the **most complex at the bottom**. This orientation affects ordering only — the
displayed z-scores and annotations are unchanged.

> Note: the correlation helpers (`aggregate_results`, `plot_metric_vs_f1`,
> `plot_correlation_summary`) still exist in `plots.py` but are **not** called by
> `generate_all_plots`; the default run produces the profile heatmap only. Re-add
> a call in `generate_all_plots` to bring the F1-correlation figures back.

---

## Setup

Uses the `kul2` environment. Install dependencies and the spaCy model:

```sh
pip install -r text/requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('cmudict')"   # optional, sharpens Flesch syllables
```

---

## Run

From the **repository root** (run as a module so package imports resolve):

```sh
python -m text.run               # compute metrics + generate the heatmap
python -m text.run --no-plots    # write metrics CSV only
python -m text.run --plots-only  # re-plot from an existing CSV (no recompute)
python -m text.run --self-check  # run metrics on the paper's example sentences
python -m text.run -v            # add debug logging
```

`--self-check` runs the suite on three sentences from the paper
(elaboration / contrast / concession) and asserts every value is finite — a fast
sanity check that needs only spaCy, not pandas/matplotlib.

---

## Outputs (`text/output/`)

- `complexity_metrics.csv` — one row per dataset: all metrics + `n_tokens` /
  `n_sentences` + `error`.
- `complexity_profile_heatmap.{svg,png}` — the z-scored profile per dataset,
  annotated with raw values, sorted by increasing overall complexity.

Figure formats are set by `figure_formats` in [`config.py`](config.py)
(`["svg", "png"]` by default).

---

## Configuration ([`config.py`](config.py))

`TextConfig` is a frozen dataclass; the rest of the package reads everything from
it. Key fields:

| Field | Default | Meaning |
|---|---|---|
| `dataset_dir` | `<repo>/dataset` | where `*/description.md` are read from |
| `description_filename` | `description.md` | per-dataset input file |
| `results_csv` | `dataset/evaluation_results_llm.csv` | LLM F1 results (used only by the optional correlation helpers) |
| `output_dir` | `text/output` | where CSV + figures are written |
| `metrics_csv_name` | `complexity_metrics.csv` | metrics output file |
| `spacy_model` | `en_core_web_sm` | parser model |
| `figure_formats` | `["svg", "png"]` | formats written per figure |

---

## Layout

```
text/
├── config.py          # frozen-dataclass paths/settings
├── metrics/
│   ├── base.py        # Metric protocol, shared spaCy loader
│   ├── syntactic.py   # MDD, subordination, parse-tree depth (+ max)
│   ├── readability.py # Flesch + syllable counting (CMU dict / heuristic)
│   ├── discourse.py   # RST / CDR / ILI surface proxies
│   └── __init__.py    # registry + compute_all
├── pipeline.py        # walk dataset -> metrics CSV
├── plots.py           # complexity-profile heatmap (+ unused correlation helpers)
├── run.py             # CLI entrypoint (python -m text.run)
└── output/            # generated CSV + figures
```

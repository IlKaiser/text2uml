# L3 Description Signal-to-Noise Ratio Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** For each case's real (L3) `description.md`, measure how much of the description's content is actually reflected in the gold `plantuml.txt` diagram ("signal") versus narrative/business-rule elaboration that never surfaces in the diagram ("noise"), producing one signal-to-noise value per case that can be correlated against F1.

**Architecture:** A new standalone module, `experiments/levels/snr.py`, following the same shape as `experiments/levels/model_complexity.py`: reuse `src/eval.py`'s existing Lark parser to extract the gold component list, reuse spaCy (already the project's sentence/token tool via `text.metrics.base.get_nlp`) to split the L3 description into sentences, then one LLM call per case to classify each sentence as SIGNAL or NOISE against the gold component list. No new environment is required — the `kul` conda env already has the Lark grammar parser, spaCy, and `langchain_openai` together.

**Tech stack:** `langchain_openai.ChatOpenAI` (`gpt-4o-mini`, matching `src/eval.py`'s `check_class_llm`), spaCy (`en_core_web_sm`, matching `text/metrics/base.py`), pandas, matplotlib (for the correlation scatter plot).

## Global Constraints

- Runs entirely in the `kul` conda env (`/Users/marcocalamo/anaconda3/envs/kul/bin/python`) — no `kul3`/Anthropic dependency.
- Computed only against the L3 (`description.md` / `plantuml.txt`) pair — not the zero/one/two/four levels.
- Every corpus-wide run recomputes **all** cases in one pass (no `only=[...]` partial-scope parameter) — this sidesteps the exact `write_f1_csv` truncation-bug class fixed earlier in this project; a merge-on-write function is unnecessary complexity here since a full recompute is cheap (~170 cases × 1 LLM call).
- Same `skip_folders` exclusion as the rest of the levels pipeline (`LevelsConfig.skip_folders`: `AlphaInsurance`, `GasStation` few-shot examples).
- Gold components include inheritance edges (`ClassA <|-- ClassB`), in addition to classes, attributes, and associations — broader than the weighted-F1 weight (which excludes inheritance) because SNR asks "does this content reach the diagram at all," not "how much does this case's model weigh."
- Token counts use spaCy's per-sentence token length (`len(span)`), matching the existing `n_tokens=len(doc)` convention in `text/metrics/__init__.py`.
- If the LLM call fails or its output can't be parsed, fall back to a substring heuristic (sentence is SIGNAL if it contains any gold class or attribute name, case-insensitive) — mirrors the existing non-LLM fallback pattern in `check_class`/`check_attributes`.

---

## Components

### 1. Gold component extraction

Function `gold_components(gold_path: Path, parser) -> GoldComponents` in `experiments/levels/snr.py`.

Reuses `ev.parse_path(str(gold_path), parser)` (from `load_evaluator(cfg)`, same as `model_complexity.py`) which returns `(classes, relationships, attributes, inheritance)`:

- `classes: List[str]` — used as-is.
- `attributes: List[str]` — rendered as `"ClassName.attr_name"` from the `(class, attr)` tuples (attr already includes `:Type`, e.g. `"Name:String"` → render as `"CardHolder.Name:String"`).
- `associations: List[str]` — each relationship dict has exactly two keys (class names, possibly with an embedded role name as `"ClassName, roleName"`); render as `"ClassA -- ClassB"` using only the class-name portion before any comma.
- `inheritance: List[str]` — rendered as `"Child <|-- Parent"` from the inheritance pair list.

`GoldComponents` is a small frozen dataclass with these four `Tuple[str, ...]` fields plus an `all_names() -> Tuple[str, ...]` helper (flat list of every class/attribute name substring, used by the fallback heuristic).

### 2. Sentence splitting

Function `split_sentences(text: str) -> List[Sentence]` where `Sentence` is a frozen dataclass `(text: str, n_tokens: int)`.

Uses `text.metrics.base.get_nlp()` and iterates `doc.sents`, using `len(span)` as the token count per sentence (spaCy token count, consistent with the rest of the metrics pipeline). Sentences that are empty after `.strip()` are dropped.

### 3. Sentence classification

Function `classify_sentences(sentences: List[Sentence], gold: GoldComponents) -> List[str]` returning one label (`"SIGNAL"` or `"NOISE"`) per sentence, same order as input.

Prompt (via `ChatOpenAI(model="gpt-4o-mini")` + `PromptTemplate` + `StrOutputParser`, matching `check_class_llm`'s call shape): give the model the numbered sentence list and the four gold component lists (classes, attributes, associations, inheritance), ask it to output one line per sentence number in the form `N: SIGNAL` or `N: NOISE`, where SIGNAL means the sentence introduces, describes, or gives a cardinality/relationship for at least one of the listed components, and NOISE means it's narrative elaboration, an example, a business rule, or a process description that doesn't correspond to any listed component.

Parsing: a small regex `^(\d+):\s*(SIGNAL|NOISE)` per line, building an index→label map; any sentence index missing from the LLM's response defaults to NOISE (conservative — better to undercount signal than silently crash). If the call raises an exception (rate limit, network, malformed response with zero parseable lines), fall back entirely to the substring heuristic: label SIGNAL if any `gold.all_names()` entry (case-insensitive) appears in the sentence text, else NOISE.

### 4. Per-case orchestration

Function `compute_case_snr(case: str, description_path: Path, gold_path: Path, parser) -> dict`:

1. `gold = gold_components(gold_path, parser)`
2. `sentences = split_sentences(description_path.read_text(encoding="utf-8"))`
3. `labels = classify_sentences(sentences, gold)`
4. Aggregate:
   - `n_sentences = len(sentences)`
   - `n_signal = sum(1 for l in labels if l == "SIGNAL")`
   - `n_noise = n_sentences - n_signal`
   - `signal_tokens = sum(s.n_tokens for s, l in zip(sentences, labels) if l == "SIGNAL")`
   - `noise_tokens = sum(s.n_tokens for s, l in zip(sentences, labels) if l == "NOISE")`
   - `snr = signal_tokens / noise_tokens if noise_tokens else float("inf")`
   - `signal_ratio = signal_tokens / (signal_tokens + noise_tokens) if (signal_tokens + noise_tokens) else float("nan")`
5. Return a flat dict with `sub_folder_name`, all of the above, plus `n_classes`, `n_attributes`, `n_associations`, `n_inheritance` (gold component counts, for context/debugging).

### 5. Corpus-wide driver + CSV

Function `compute_all(cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> pd.DataFrame`:

- `ev = load_evaluator(cfg)`; `parser = ev.init_parser(str(cfg.grammar_path))` (same pattern as `model_complexity.compute_gold_complexity`).
- Iterates `sorted(p for p in cfg.dataset_dir.iterdir() if p.is_dir())`, skipping `cfg.skip_folders`.
- Skips (with a `logger.debug`) any case missing `description.md` or `plantuml.txt`.
- Wraps each case's `compute_case_snr` call in `try/except Exception` (`logger.warning`, skip the case) so one bad case can't abort the batch — matches `compute_gold_complexity`'s per-case isolation.
- Returns a `pd.DataFrame` of all rows.

Function `write_snr_csv(df: pd.DataFrame, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> Path` writes unconditionally to `cfg.output_dir / "levels_snr.csv"` (plain overwrite — safe here per the Global Constraints no-partial-scope rule).

### 6. Correlation plot

New function `plot_snr_vs_f1(snr_df: pd.DataFrame, f1_df: pd.DataFrame, cfg: LevelsConfig = DEFAULT_LEVELS_CONFIG) -> float` in `experiments/levels/snr.py` (kept alongside the computation rather than added to the already-broad `plots.py`, since it's a one-off diagnostic scatter rather than part of the standard multi-level plot suite):

- Joins `snr_df` (`signal_ratio`) with `f1_df` filtered to `level == "three"` (`f1_global`) on `sub_folder_name`.
- Computes Pearson `r` and `p` via `scipy.stats.pearsonr` (already a project dependency, used in the earlier L3 metric-correlation analysis).
- Scatter plot: x = `signal_ratio`, y = `f1_global`, with an OLS trend line (`numpy.polyfit`, degree 1) and the `r`/`p` annotated in the title.
- Saves to `cfg.output_dir / "levels_snr_vs_f1.{png,svg}"` via the same `_save`-style pattern as `plots.py`.
- Returns the Pearson `r` (so a caller/test can assert on it without re-reading the saved figure).

---

## Error Handling

- Missing `description.md` or `plantuml.txt`: case is skipped at the `compute_all` level (debug log), not an error.
- Gold parse failure (malformed `plantuml.txt`): case is skipped with a warning, matching `compute_gold_complexity`.
- LLM call failure or unparseable response: falls back to the substring heuristic (never raises out of `classify_sentences`).
- Empty description (zero sentences after splitting): returns a row with all counts at 0 and `snr`/`signal_ratio` as `nan` — excluded from the correlation plot's join (pandas drops NaN rows in `pearsonr` input) rather than crashing the plot.

## Testing

- `tests/experiments/levels/test_snr.py`:
  - `gold_components` against a small in-memory parsed structure (or a tiny literal `plantuml.txt` fixture) — verify classes/attributes/associations/inheritance render correctly, including the comma-role-name stripping case.
  - `split_sentences` on a short literal string — verify sentence count and token counts.
  - `classify_sentences` with the LLM call monkeypatched (no real API calls in tests) — verify label parsing from a canned response string, and verify the fallback heuristic triggers when the monkeypatched call raises.
  - `compute_case_snr` end-to-end on a tiny fixture case (small literal description + gold), with `classify_sentences` monkeypatched to a fixed label sequence — verify the aggregate counts and `snr`/`signal_ratio` arithmetic.
  - `write_snr_csv` — verify it writes the expected columns (no merge-behavior test needed here, since Global Constraints rule out partial-scope writes for this module).
  - `plot_snr_vs_f1` — verify it returns a Pearson r consistent with a small hand-constructed pair of DataFrames (e.g., perfectly correlated toy data → r ≈ 1.0), and that the output files are created, using `tmp_path` for `cfg.output_dir`.

No test should call the real OpenAI API — every test exercising `classify_sentences` or anything that calls it monkeypatches the LLM call itself (matching how `text/rewrite`'s tests monkeypatch `rewrite_to_minimize`/`rewrite_to_shape`).

## Out of Scope

- Levels zero/one/two/four are not scored for SNR (L3 only, per explicit scope decision).
- No CLI `main()`/argparse entry point is required unless useful for manual reruns — this can be driven the same way `model_complexity.py` currently is (direct Python invocation), not wired into `case_pipeline.py`'s orchestration.
- No merge-on-write CSV logic (see Global Constraints).

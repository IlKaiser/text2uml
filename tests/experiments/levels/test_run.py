from __future__ import annotations

import pandas as pd

from experiments.levels.run import _resolve_plot_scope


def test_resolve_plot_scope_defaults_to_corpus_when_one_model(tmp_path):
    csv = tmp_path / "levels_f1.csv"
    df = pd.DataFrame({"model": ["claude-sonnet-4-6"] * 3, "f1_global": [0.1, 0.2, 0.3]})
    df.to_csv(csv, index=False)

    plot_df, subdir, suffix = _resolve_plot_scope(df, "claude-sonnet-4-6", csv)

    assert subdir == "corpus"
    assert suffix == ""
    assert len(plot_df) == 3


def test_resolve_plot_scope_ignores_model_arg_when_not_given(tmp_path):
    csv = tmp_path / "levels_f1.csv"
    df = pd.DataFrame({"model": ["claude-sonnet-4-6", "gpt-4o-mini"], "f1_global": [0.1, 0.2]})
    df.to_csv(csv, index=False)

    plot_df, subdir, suffix = _resolve_plot_scope(df, None, csv)

    assert subdir == "corpus"
    assert suffix == ""
    assert len(plot_df) == 2


def test_resolve_plot_scope_detects_other_models_from_csv_not_just_df(tmp_path):
    """Bug history: when generate+evaluate+plot run together for one model,
    `df` (from evaluate_all) only ever holds that model's freshly-scored
    rows. The old guard checked `df` alone, so it never saw that the
    persisted CSV already had other models on disk and silently overwrote
    the shared corpus/ plots with single-model data. Fixed by checking the
    full CSV's model column too."""
    csv = tmp_path / "levels_f1.csv"
    pd.DataFrame({
        "model": ["claude-sonnet-4-6", "claude-sonnet-4-6", "gemma4:e4b"],
        "f1_global": [0.4, 0.5, 0.6],
    }).to_csv(csv, index=False)

    # `df` here simulates evaluate_all's fresh output: only the new model's rows.
    fresh_df = pd.DataFrame({"model": ["gpt-4o-mini"] * 2, "f1_global": [0.7, 0.8]})

    plot_df, subdir, suffix = _resolve_plot_scope(fresh_df, "gpt-4o-mini", csv)

    assert subdir == "corpus_gpt-4o-mini"
    assert suffix == " — gpt-4o-mini"
    assert len(plot_df) == 2


def test_resolve_plot_scope_sanitizes_colon_in_model_name(tmp_path):
    """Local Ollama-style tags like 'gemma4:e4b' contain ':', which must be
    stripped from the subdir name the same way evaluate.py's
    _safe_model_name sanitizes result filenames."""
    csv = tmp_path / "levels_f1.csv"
    pd.DataFrame({
        "model": ["claude-sonnet-4-6", "gemma4:e4b"],
        "f1_global": [0.4, 0.6],
    }).to_csv(csv, index=False)

    fresh_df = pd.DataFrame({"model": ["gemma4:e4b"], "f1_global": [0.6]})

    plot_df, subdir, suffix = _resolve_plot_scope(fresh_df, "gemma4:e4b", csv)

    assert subdir == "corpus_gemma4_e4b"
    assert suffix == " — gemma4:e4b"
    assert len(plot_df) == 1

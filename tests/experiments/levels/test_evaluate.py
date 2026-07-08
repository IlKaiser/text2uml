from __future__ import annotations

from dataclasses import replace

import pandas as pd

from experiments.levels.config import DEFAULT_LEVELS_CONFIG
from experiments.levels.evaluate import write_f1_csv


def test_write_f1_csv_merges_scoped_writes_instead_of_truncating(tmp_path):
    """Bug history: write_f1_csv used to overwrite the whole file
    unconditionally, so calling evaluate_all(only=[...]) for a subset of
    cases and passing the result straight to write_f1_csv silently deleted
    every other case's rows. This happened twice in real usage before being
    fixed -- lock in the merge behavior."""
    cfg = replace(DEFAULT_LEVELS_CONFIG, output_dir=tmp_path)

    full_corpus = pd.DataFrame([
        {"sub_folder_name": "A", "level": "three", "level_rank": 3, "f1_global": 0.5},
        {"sub_folder_name": "B", "level": "three", "level_rank": 3, "f1_global": 0.6},
        {"sub_folder_name": "C", "level": "three", "level_rank": 3, "f1_global": 0.7},
    ])
    write_f1_csv(full_corpus, cfg)

    # Simulate a later scoped re-evaluation of just case "B" (e.g. after
    # regenerating one case's description) at an updated score.
    scoped_update = pd.DataFrame([
        {"sub_folder_name": "B", "level": "three", "level_rank": 3, "f1_global": 0.99},
    ])
    write_f1_csv(scoped_update, cfg)

    result = pd.read_csv(cfg.f1_csv)
    assert set(result["sub_folder_name"]) == {"A", "B", "C"}
    assert result.set_index("sub_folder_name").loc["A", "f1_global"] == 0.5
    assert result.set_index("sub_folder_name").loc["B", "f1_global"] == 0.99
    assert result.set_index("sub_folder_name").loc["C", "f1_global"] == 0.7


def test_write_f1_csv_can_add_a_new_case_not_previously_present(tmp_path):
    cfg = replace(DEFAULT_LEVELS_CONFIG, output_dir=tmp_path)
    write_f1_csv(pd.DataFrame([{"sub_folder_name": "A", "level": "three", "level_rank": 3, "f1_global": 0.5}]), cfg)
    write_f1_csv(pd.DataFrame([{"sub_folder_name": "B", "level": "three", "level_rank": 3, "f1_global": 0.6}]), cfg)

    result = pd.read_csv(cfg.f1_csv)
    assert set(result["sub_folder_name"]) == {"A", "B"}


def test_write_f1_csv_does_not_clobber_a_different_models_rows_for_the_same_case(tmp_path):
    """Bug history: keying the merge on sub_folder_name alone meant scoring a
    second model for a case already in the CSV silently deleted the first
    model's row for that same case (and every other level of it), since the
    anti-join never looked at the model or level columns. Only surfaced once
    a second model was evaluated -- lock in the fix."""
    cfg = replace(DEFAULT_LEVELS_CONFIG, output_dir=tmp_path)

    write_f1_csv(pd.DataFrame([
        {"sub_folder_name": "A", "level": "three", "level_rank": 3, "model": "claude-sonnet-4-6", "f1_global": 0.5},
    ]), cfg)
    write_f1_csv(pd.DataFrame([
        {"sub_folder_name": "A", "level": "three", "level_rank": 3, "model": "gpt-4o-mini", "f1_global": 0.3},
    ]), cfg)

    result = pd.read_csv(cfg.f1_csv)
    assert set(result["model"]) == {"claude-sonnet-4-6", "gpt-4o-mini"}
    by_model = result.set_index("model")
    assert by_model.loc["claude-sonnet-4-6", "f1_global"] == 0.5
    assert by_model.loc["gpt-4o-mini", "f1_global"] == 0.3

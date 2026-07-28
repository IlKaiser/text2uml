from __future__ import annotations

import pandas as pd
import pytest

from experiments.levels.config import DEFAULT_LEVELS_CONFIG
from experiments.levels.model_complexity import compute_gold_complexity, weighted_f1


def test_compute_gold_complexity_matches_known_gasstation_kul_counts():
    """dataset/GasStation_KUL/plantuml.txt has 7 classes, 12 attributes, 7
    associations, 0 inheritance edges -- hand-verified against the file."""
    df = compute_gold_complexity(DEFAULT_LEVELS_CONFIG)
    row = df[df["sub_folder_name"] == "GasStation_KUL"].iloc[0]
    assert row["n_classes"] == 7
    assert row["n_attributes"] == 12
    assert row["n_associations"] == 7
    assert row["n_inheritance"] == 0
    assert row["total_size"] == 7 + 12 + 7


def test_weighted_f1_gives_more_influence_to_larger_gold_models():
    f1_df = pd.DataFrame([
        {"sub_folder_name": "Small", "level": "three", "level_rank": 3, "f1_global": 1.0},
        {"sub_folder_name": "Big", "level": "three", "level_rank": 3, "f1_global": 0.0},
    ])
    complexity_df = pd.DataFrame([
        {"sub_folder_name": "Small", "total_size": 1},
        {"sub_folder_name": "Big", "total_size": 99},
    ])
    result = weighted_f1(f1_df, complexity_df)
    row = result.iloc[0]
    assert row["f1_mean"] == pytest.approx(0.5)  # unweighted: (1.0 + 0.0) / 2
    # weighted: dominated by "Big" (weight 99 vs 1), which scored 0.0.
    assert row["f1_weighted_mean"] == pytest.approx(1.0 * 1 / 100)
    assert row["n_cases"] == 2
    assert row["total_weight"] == 100


def test_weighted_f1_raises_on_no_overlap():
    f1_df = pd.DataFrame([{"sub_folder_name": "A", "level": "three", "level_rank": 3, "f1_global": 0.5}])
    complexity_df = pd.DataFrame([{"sub_folder_name": "B", "total_size": 10}])
    with pytest.raises(ValueError):
        weighted_f1(f1_df, complexity_df)

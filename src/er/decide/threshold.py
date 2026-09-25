"""Global-threshold baseline — the alternative the per-entity expected-F0.5 rule has to beat on OOF."""

from __future__ import annotations

import numpy as np
import polars as pl

from er.metrics import macro_f05


def predict(df: pl.DataFrame, threshold: float, prob_col: str = "p") -> pl.DataFrame:
    return df.filter(pl.col(prob_col) >= threshold).select("s1_id", "target_id")


def tune(df: pl.DataFrame, truth: pl.DataFrame, prob_col: str = "p",
        grid: np.ndarray | None = None) -> tuple[float, float, list[dict]]:
    """Grid-search the single cutoff that maximizes macro F0.5 on this (OOF) frame."""
    grid = np.arange(0.05, 1.0, 0.05) if grid is None else grid
    truth_subset = truth.filter(pl.col("s1_id").is_in(df["s1_id"].unique().implode()))
    curve = []
    best_t, best_f05 = float(grid[0]), -1.0
    for t in grid:
        f05 = macro_f05(truth_subset, predict(df, float(t), prob_col))
        curve.append({"threshold": float(t), "macro_f05": f05})
        if f05 > best_f05:
            best_t, best_f05 = float(t), f05
    return best_t, best_f05, curve

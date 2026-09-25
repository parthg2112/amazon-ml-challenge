"""Precision/recall/F0.5 curves, singleton accuracy, and the staged before/after metrics the gate report needs."""

from __future__ import annotations

import numpy as np
import polars as pl

from er.metrics import macro_f05, row_table


def pr_curve(df: pl.DataFrame, truth: pl.DataFrame, prob_col: str = "p", grid: np.ndarray | None = None,
            country_col: str = "q_country") -> pl.DataFrame:
    """Precision/recall/F0.5 at each threshold in `grid`, overall and per country."""
    grid = np.arange(0.05, 1.0, 0.05) if grid is None else grid
    countries: list[str | None] = [None]
    if country_col in df.columns:
        countries += sorted(df[country_col].drop_nulls().unique().to_list())
    rows = []
    for country in countries:
        sub = df if country is None else df.filter(pl.col(country_col) == country)
        if sub.height == 0:
            continue
        truth_sub = truth.filter(pl.col("s1_id").is_in(sub["s1_id"].unique().implode()))
        for t in grid:
            preds = sub.filter(pl.col(prob_col) >= t).select("s1_id", "target_id")
            rt = row_table(truth_sub, preds)
            tp, n_pred, n_true = int(rt["tp"].sum()), int(rt["n_pred"].sum()), int(rt["n_true"].sum())
            rows.append({
                "country": country or "overall", "threshold": round(float(t), 2),
                "precision": tp / n_pred if n_pred else None,
                "recall": tp / n_true if n_true else None,
                "macro_f05": macro_f05(truth_sub, preds),
            })
    return pl.DataFrame(rows)


def singleton_accuracy(preds: pl.DataFrame, truth: pl.DataFrame) -> float | None:
    """Correct empty predictions / true singletons (S1 with no true links)."""
    rt = row_table(truth, preds)
    singles = rt.filter(pl.col("n_true") == 0)
    return float((singles["n_pred"] == 0).mean()) if singles.height else None


def stage_metrics(preds: pl.DataFrame, truth: pl.DataFrame) -> dict:
    rt = row_table(truth, preds)
    return {
        "macro_f05": macro_f05(truth, preds),
        "singleton_accuracy": singleton_accuracy(preds, truth),
        "n_predicted_pairs": int(rt["n_pred"].sum()),
    }

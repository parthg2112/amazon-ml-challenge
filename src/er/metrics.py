"""Exact leaderboard metric (macro F0.5 per S1) and blocking-quality metrics.

Blocking metrics are built from per-S1 row tables so they can be accumulated over query chunks.
"""

from __future__ import annotations

import polars as pl

SRCS = ("S2", "S3")


def row_f05(n_true: pl.Expr, n_pred: pl.Expr, tp: pl.Expr) -> pl.Expr:
    """1.25·TP / (|P| + 0.25·|T|); empty/empty = 1 (PROBLEM_STATEMENT §8)."""
    return (
        pl.when((n_true == 0) & (n_pred == 0))
        .then(1.0)
        .otherwise(1.25 * tp / (n_pred + 0.25 * n_true))
    )


def _truth_long(truth: pl.DataFrame) -> pl.DataFrame:
    return truth.select("s1_id", "truth").explode("truth", empty_as_null=True).drop_nulls("truth").rename(
        {"truth": "target_id"}
    )


def row_table(truth: pl.DataFrame, pairs: pl.DataFrame) -> pl.DataFrame:
    """One row per S1 in `truth`: totals and per-source counts of truth, predictions, and hits."""
    pairs = pairs.select("s1_id", "target_id").unique().with_columns(pl.col("target_id").str.slice(0, 2).alias("src"))
    tlong = _truth_long(truth).with_columns(pl.col("target_id").str.slice(0, 2).alias("src"))
    hits = pairs.join(tlong, on=["s1_id", "target_id", "src"], how="inner")

    def per_src(df: pl.DataFrame, name: str) -> pl.DataFrame:
        return df.group_by("s1_id").agg(
            pl.len().alias(name), *[(pl.col("src") == s).sum().alias(f"{name}_{s}") for s in SRCS]
        )

    out = truth.select("s1_id")
    for df, name in ((tlong, "n_true"), (pairs, "n_pred"), (hits, "tp")):
        out = out.join(per_src(df, name), on="s1_id", how="left")
    return out.with_columns(pl.exclude("s1_id").fill_null(0).cast(pl.Int32))


def macro_f05(truth: pl.DataFrame, predictions: pl.DataFrame) -> float:
    """Mean row F0.5 over every S1 in `truth`; S1s absent from predictions count as empty."""
    if truth.height == 0:
        raise ValueError("empty evaluation set")
    rows = row_table(truth, predictions)
    return rows.select(row_f05(pl.col("n_true"), pl.col("n_pred"), pl.col("tp")).mean()).item()


def summarize(rows: pl.DataFrame, n_targets: int, slices: pl.DataFrame | None = None, brief: bool = False) -> dict:
    rows = rows.with_columns(
        row_f05(pl.col("n_true"), pl.col("tp"), pl.col("tp")).alias("oracle"),
        row_f05(pl.col("n_true"), pl.col("n_pred"), pl.col("tp")).alias("predict_all"),
    )
    n = rows.height
    nonempty = rows.filter(pl.col("n_true") > 0)
    true_links = int(rows["n_true"].sum())
    total_cand = int(rows["n_pred"].sum())
    res = {
        "s1_rows": n,
        "oracle_ceiling_f05": rows["oracle"].mean(),
        "pair_recall": rows["tp"].sum() / true_links if true_links else None,
        "full_coverage": (nonempty["tp"] == nonempty["n_true"]).mean() if nonempty.height else None,
        "mean_candidates": total_cand / n if n else 0.0,
    }
    if brief:
        return res
    sizes = rows["n_pred"]
    res.update({
        "predict_all_f05": rows["predict_all"].mean(),
        "true_links": true_links,
        "candidate_pairs": total_cand,
        "candidate_precision": rows["tp"].sum() / total_cand if total_cand else None,
        "singleton_rows": int((rows["n_true"] == 0).sum()),
        "singletons_with_candidates": int(((rows["n_true"] == 0) & (rows["n_pred"] > 0)).sum()),
        "candidates_p50": float(sizes.quantile(0.5)),
        "candidates_p95": float(sizes.quantile(0.95)),
        "candidates_p99": float(sizes.quantile(0.99)),
        "candidates_max": int(sizes.max()),
        "reduction_ratio": 1 - total_cand / (n * n_targets) if n and n_targets else None,
        "pair_recall_by_source": {
            s: rows[f"tp_{s}"].sum() / rows[f"n_true_{s}"].sum() if rows[f"n_true_{s}"].sum() else None
            for s in SRCS
        },
        "mean_candidates_by_source": {s: rows[f"n_pred_{s}"].sum() / n for s in SRCS},
    })
    if slices is not None:
        joined = rows.join(slices, on="s1_id", how="left")
        res["slices"] = {
            col: {str(k[0]): summarize(g, n_targets, brief=True) for k, g in joined.group_by(col)}
            for col in slices.columns if col != "s1_id"
        }
    return res


def blocking_metrics(truth: pl.DataFrame, candidates: pl.DataFrame, n_targets: int,
                     slices: pl.DataFrame | None = None) -> dict:
    return summarize(row_table(truth, candidates), n_targets, slices)

"""One S2/S3 record -> at most one S1.

Primary rule: the highest calibrated probability wins a contested record. The comparison ("a proper
assignment method") is scarcity-weighted: it divides each claim's probability by how many candidates
that S1 has to fall back on, so an S1 with few alternatives can outbid a stronger claimant that has
plenty of other good options. Both are O(n log n) — cheap enough to run and compare on every subset.
"""

from __future__ import annotations

import polars as pl


def _winner(df: pl.DataFrame, priority_col: str) -> pl.DataFrame:
    ranked = df.with_columns(
        pl.col(priority_col).rank("ordinal", descending=True).over("target_id").alias("_rank")
    )
    return ranked.filter(pl.col("_rank") == 1).drop("_rank")


def resolve_max_probability(df: pl.DataFrame, prob_col: str = "p") -> pl.DataFrame:
    """Primary rule: each contested target goes to its highest-probability claimant."""
    return _winner(df.sort("s1_id"), prob_col)  # stable pre-sort makes exact ties deterministic


def resolve_scarcity_weighted(df: pl.DataFrame, prob_col: str = "p") -> pl.DataFrame:
    """Comparison rule: priority = probability / that S1's own candidate count."""
    df = df.with_columns(pl.len().over("s1_id").alias("_n_cands"))
    df = df.with_columns((pl.col(prob_col) / pl.col("_n_cands")).alias("_priority"))
    return _winner(df.sort("s1_id"), "_priority").drop("_n_cands", "_priority")


def conflict_stats(df: pl.DataFrame) -> dict:
    sizes = df.group_by("target_id").agg(pl.len().alias("n_claimants"))
    return {
        "targets_claimed": sizes.height,
        "contested_targets": int((sizes["n_claimants"] > 1).sum()),
        "max_claimants": int(sizes["n_claimants"].max()) if sizes.height else 0,
    }

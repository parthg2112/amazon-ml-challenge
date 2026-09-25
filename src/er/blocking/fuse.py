"""Channel union → weighted RRF ranking within (S1, source)."""

from __future__ import annotations

import polars as pl

RRF_K = 60


def add_ranks(df: pl.DataFrame) -> pl.DataFrame:
    """Rank within (s1_id, source); ties share the best rank."""
    return df.with_columns(pl.col("target_id").str.slice(0, 2).alias("src")).with_columns(
        pl.col("score").rank("min", descending=True).over("s1_id", "src").cast(pl.Int32).alias("rank")
    )


def fuse(cands: pl.DataFrame, weights: dict[str, float]) -> pl.DataFrame:
    w = pl.col("channel").replace_strict(weights, default=1.0, return_dtype=pl.Float64)
    fused = (
        cands.with_columns((w / (RRF_K + pl.col("rank"))).alias("rrf"))
        .group_by("s1_id", "target_id", "src")
        .agg(pl.col("rrf").sum(), pl.col("channel").unique().sort().alias("channels"))
    )
    return fused.with_columns(
        pl.col("rrf").rank("ordinal", descending=True).over("s1_id", "src").cast(pl.Int32).alias("frank")
    )

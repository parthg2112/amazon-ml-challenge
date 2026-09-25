"""Deterministic entity-level split, byte-compatible with the retired Perl Validation.pm."""

from __future__ import annotations

import hashlib

import polars as pl

SEED = "er-v1-2026-09-25"
VALIDATION_PERCENT = 20
DEV_PERCENT = 10  # share of validation used for fast iteration
RANKER_BASIS_POINTS = 200  # 2% of all S1 (train fold only) to fit the stage-2 candidate ranker


def stable_bucket(entity_id: str, seed: str, domain: str, modulus: int) -> int:
    digest = hashlib.sha256(f"{domain}\x1f{seed}\x1f{entity_id}".encode()).hexdigest()
    return int(digest[:8], 16) % modulus


def assign_split(s1_ids: pl.Series, seed: str = SEED) -> pl.DataFrame:
    ids = s1_ids.to_list()
    val = [stable_bucket(i, seed, "validation", 100) < VALIDATION_PERCENT for i in ids]
    dev = [stable_bucket(i, seed, "benchmark", 10_000) < DEV_PERCENT * 100 for i in ids]
    rk = [stable_bucket(i, seed, "ranker", 10_000) < RANKER_BASIS_POINTS for i in ids]
    return pl.DataFrame({"s1_id": ids, "is_val": val, "is_dev": dev, "is_rtrain": rk}).with_columns(
        pl.when(~pl.col("is_val"))
        .then(pl.lit("train"))
        .when(pl.col("is_dev"))
        .then(pl.lit("dev"))
        .otherwise(pl.lit("val"))
        .alias("fold")
    )


def load_split() -> pl.DataFrame:
    from er import io

    path = io.DATA / "split.parquet"
    if path.exists():
        df = pl.read_parquet(path)
        if "is_rtrain" in df.columns:
            return df
    df = assign_split(io.load_truth()["s1_id"])
    df.write_parquet(path)
    return df


def subset_ids(split_df: pl.DataFrame, subset: str) -> pl.Series:
    """subset: train | rtrain (2% ranker sample of train fold) | rtrain<N> (N% of train fold, ⊇ rtrain)
    | dev (10% of validation) | val (20% holdout)."""
    if subset == "rtrain":
        return split_df.filter(~pl.col("is_val") & pl.col("is_rtrain"))["s1_id"]
    if subset.startswith("rtrain"):
        bp = int(subset[6:]) * 100
        ids = split_df.filter(~pl.col("is_val"))["s1_id"].to_list()
        return pl.Series("s1_id", [i for i in ids if stable_bucket(i, SEED, "ranker", 10_000) < bp])
    if subset == "train":
        return split_df.filter(~pl.col("is_val"))["s1_id"]
    if subset == "dev":
        return split_df.filter(pl.col("is_val") & pl.col("is_dev"))["s1_id"]
    if subset == "val":
        return split_df.filter(pl.col("is_val"))["s1_id"]
    raise ValueError(subset)

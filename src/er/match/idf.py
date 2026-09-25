"""Word-level IDF over the target pool: an exact (unhashed) vocabulary, built once from S2+S3.

Used to weight name/address token overlap so common words ("services", "the") count for less than
rare ones (a distinctive surname, a rare street name) — country-agnostic, since it is keyed by the
token string itself and estimated fresh from whichever pool it is built on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class WordIdf:
    table: dict[str, float]
    default: float  # assigned to tokens never seen in the pool (rarer than anything observed)
    rare_cutoff: float  # idf value at the 75th percentile; tokens above this count as "rare"

    def lookup(self, exploded: pl.DataFrame, tok_col: str) -> pl.Series:
        return exploded[tok_col].replace_strict(self.table, default=self.default, return_dtype=pl.Float32)


def build_word_idf(pool: pl.DataFrame, col: str) -> WordIdf:
    n = pool.height
    df = (
        pool.lazy()
        .select(pl.col(col).str.split(" ").alias("tok"))
        .explode("tok", empty_as_null=True)
        .drop_nulls("tok")
        .filter(pl.col("tok") != "")
        .group_by("tok")
        .len(name="df")
        .collect()
    )
    idf = df.with_columns(
        (math.log(1 + n) - (pl.col("df") + 1).log() + 1).cast(pl.Float32).alias("idf")
    )
    table = dict(zip(idf["tok"].to_list(), idf["idf"].to_list()))
    default = float(math.log(1 + n) + 1)  # token unseen in pool: df=0, rarer than any observed token
    cutoff = float(idf["idf"].quantile(0.75)) if idf.height else default
    return WordIdf(table=table, default=default, rare_cutoff=cutoff)


def token_overlap_features(df: pl.DataFrame, q_col: str, t_col: str, idf: WordIdf, prefix: str) -> pl.DataFrame:
    """Token-set overlap/extras between two space-joined token strings, plain-count and IDF-weighted."""
    row_id = "_row"
    work = df.with_row_index(row_id).select(
        row_id,
        pl.col(q_col).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias("_q"),
        pl.col(t_col).str.split(" ").list.eval(pl.element().filter(pl.element() != "")).alias("_t"),
    ).with_columns(
        pl.col("_q").list.set_intersection(pl.col("_t")).alias("_inter"),
        pl.col("_q").list.set_difference(pl.col("_t")).alias("_extra_q"),
        pl.col("_t").list.set_difference(pl.col("_q")).alias("_extra_q_t"),
    )

    def idf_sum(list_col: str, only_rare: bool = False) -> pl.Series:
        exploded = work.select(row_id, pl.col(list_col).alias("tok")).explode("tok", empty_as_null=True) \
            .drop_nulls("tok")
        exploded = exploded.with_columns(idf.lookup(exploded, "tok").alias("idf"))
        if only_rare:
            exploded = exploded.filter(pl.col("idf") > idf.rare_cutoff)
        agg = exploded.group_by(row_id).agg(pl.len().alias("n"), pl.col("idf").sum().alias("sum"))
        out = work.select(row_id).join(agg, on=row_id, how="left")
        return out

    inter_idf = idf_sum("_inter")
    rare_inter = idf_sum("_inter", only_rare=True)
    extra_q_idf = idf_sum("_extra_q")
    extra_t_idf = idf_sum("_extra_q_t")

    return df.with_columns(
        work["_inter"].list.len().alias(f"{prefix}_overlap_n"),
        inter_idf["sum"].fill_null(0.0).alias(f"{prefix}_overlap_idf"),
        rare_inter["n"].fill_null(0).alias(f"{prefix}_rare_overlap_n"),
        work["_extra_q"].list.len().alias(f"{prefix}_extra_q_n"),
        extra_q_idf["sum"].fill_null(0.0).alias(f"{prefix}_extra_q_idf"),
        work["_extra_q_t"].list.len().alias(f"{prefix}_extra_t_n"),
        extra_t_idf["sum"].fill_null(0.0).alias(f"{prefix}_extra_t_idf"),
    )

"""Per-S1 expected-F0.5-optimal candidate-set size, given calibrated match probabilities.

For a fixed predicted-set size k, the set maximizing expected F0.5 is always the top-k probabilities
(a classical F-measure result), so the only choice left is k itself (including k=0, predict nothing).

We use the standard "plug-in" estimator: E[TP | top-k] = sum of the k selected probabilities — exact,
by linearity of expectation, independent of any correlation between candidates. E[T], the number of true
links in this S1's whole candidate pool, is estimated the same way as the sum of ALL its probabilities.
Plugging these into the metric formula (`1.25*TP / (P + 0.25*T)`) is not literally E[F0.5] — a ratio of
correlated random sums has no closed form without a full Poisson-binomial DP over the joint distribution
— but it is the standard practical estimator for expected precision/recall/F-measure at a cutoff, and
it is exact whenever probabilities are well separated near 0 or 1, which calibration makes more likely.

k=0 is handled exactly rather than by the same plug-in formula (which degenerates at k=0): the metric
defines empty/empty as F0.5=1, so E[F0.5 | predict nothing] = P(no true link exists in the pool at all)
= prod(1 - p_i), computed exactly from the independent-Bernoulli assumption already implicit above.
"""

from __future__ import annotations

import polars as pl

_EPS = 1e-6


def add_candidate_ranks(df: pl.DataFrame, prob_col: str = "p") -> pl.DataFrame:
    """Sort each S1's candidates (across both sources) by probability descending; add cumulative stats."""
    df = df.sort(["s1_id", prob_col], descending=[False, True])
    safe_p = pl.col(prob_col).clip(_EPS, 1 - _EPS)
    return df.with_columns(
        pl.int_range(1, pl.len() + 1).over("s1_id").alias("_k"),
        pl.col(prob_col).cum_sum().over("s1_id").alias("_cum_p"),
        pl.col(prob_col).sum().over("s1_id").alias("_total_p"),
        (1 - safe_p).log().sum().over("s1_id").exp().alias("_p_empty"),
    )


def add_expected_f05(df: pl.DataFrame, prob_col: str = "p") -> pl.DataFrame:
    df = add_candidate_ranks(df, prob_col)
    return df.with_columns(
        (1.25 * pl.col("_cum_p") / (pl.col("_k") + 0.25 * pl.col("_total_p"))).alias("_ef05_prefix")
    )


def choose_k(df: pl.DataFrame, prob_col: str = "p") -> pl.DataFrame:
    """One row per S1: chosen_k (0 = predict nothing) and its expected F0.5."""
    df = add_expected_f05(df, prob_col)
    best_prefix = (
        df.sort(["s1_id", "_ef05_prefix"], descending=[False, True])
        .group_by("s1_id", maintain_order=True)
        .agg(pl.col("_k").first().alias("_best_k"), pl.col("_ef05_prefix").first().alias("_best_val"),
             pl.col("_p_empty").first().alias("_p_empty"))
    )
    return best_prefix.with_columns(
        pl.when(pl.col("_p_empty") >= pl.col("_best_val")).then(0).otherwise(pl.col("_best_k"))
        .alias("chosen_k"),
        pl.max_horizontal("_p_empty", "_best_val").alias("expected_f05"),
    ).select("s1_id", "chosen_k", "expected_f05")


def select(df: pl.DataFrame, prob_col: str = "p") -> pl.DataFrame:
    """The final predicted (s1_id, target_id) pairs: each S1's top `chosen_k` candidates."""
    ranked = add_candidate_ranks(df, prob_col)
    chosen = choose_k(df, prob_col).select("s1_id", "chosen_k")
    return ranked.join(chosen, on="s1_id", how="left").filter(pl.col("_k") <= pl.col("chosen_k")) \
        .select("s1_id", "target_id", "src", prob_col)

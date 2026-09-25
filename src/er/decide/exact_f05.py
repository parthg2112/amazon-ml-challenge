"""Exact expected-F0.5 set size per S1 (Jansche 2007; Dembczynski et al. 2011, GFM), vectorized.

Under independent Bernoulli candidates, for predicting the top-k: TP ~ PoissonBinomial(p_1..p_k),
U (true links left unpredicted) ~ PoissonBinomial(p_k+1..p_n), independent, and
E[F0.5] = sum_{a,b} P(TP=a) P(U=b) * 1.25a / (k + 0.25(a+b)); k=0 scores P(TP+U=0).
Unlike the plug-in ratio in `expected_f05.py`, this is the exact expectation.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def _pmf_prefixes(P: np.ndarray) -> list[np.ndarray]:
    """pmfs[k] = Poisson-binomial pmf of the count over columns [0, k), shape (S, k+1)."""
    S, n = P.shape
    pmf = np.ones((S, 1))
    out = [pmf]
    for j in range(n):
        p = P[:, j:j + 1]
        nxt = np.zeros((S, pmf.shape[1] + 1))
        nxt[:, :-1] += pmf * (1 - p)
        nxt[:, 1:] += pmf * p
        pmf = nxt
        out.append(pmf)
    return out


def pmf_all(P: np.ndarray) -> np.ndarray:
    """Poisson-binomial pmf over every candidate of each row, shape (S, n+1)."""
    return _pmf_prefixes(P)[-1]


def expected_f05_matrix(P: np.ndarray, tilt: np.ndarray | None = None) -> np.ndarray:
    """P: (S, n) probabilities sorted descending per row (pad with 0). Returns E[F0.5] for k=0..n, (S, n+1).

    `tilt` (optional, indexed by the total true-link count T = TP + U) reweights the count posterior:
    the expectation becomes sum(A W tilt B) / sum(A tilt B). It exists because the independence
    assumption above is wrong in a specific, correctable way — a record's candidates share its name and
    address, so the implied count distribution is far flatter than the truth. `tilt=None` leaves the
    plain independent expectation untouched. See `er.decide.linkcount`.
    """
    S, n = P.shape
    pre = _pmf_prefixes(P)
    suf = _pmf_prefixes(P[:, ::-1])  # suf[m] = pmf over the last m columns
    out = np.empty((S, n + 1))
    if tilt is None:
        out[:, 0] = np.prod(1 - P, axis=1)
    else:
        w = np.zeros(n + 1)
        w[:min(len(tilt), n + 1)] = tilt[:n + 1]
        full = pre[n]  # pmf of T over all candidates
        denom0 = np.maximum(full @ w, 1e-300)
        out[:, 0] = full[:, 0] * w[0] / denom0
    for k in range(1, n + 1):
        A, B = pre[k], suf[n - k]  # (S, k+1), (S, n-k+1)
        a = np.arange(k + 1)[:, None]
        b = np.arange(n - k + 1)[None, :]
        W = 1.25 * a / (k + 0.25 * (a + b))  # (k+1, n-k+1)
        if tilt is None:
            out[:, k] = np.einsum("sa,ab,sb->s", A, W, B, optimize=True)
        else:
            T = np.zeros((k + 1, n - k + 1))
            T[:] = w[(a + b).clip(max=n)]
            num = np.einsum("sa,ab,sb->s", A, W * T, B, optimize=True)
            den = np.einsum("sa,ab,sb->s", A, T, B, optimize=True)
            out[:, k] = num / np.maximum(den, 1e-300)
    return out


def select(df: pl.DataFrame, prob_col: str = "p") -> pl.DataFrame:
    """Top-k per S1 with k maximizing exact E[F0.5] (k=0 allowed). Returns the selected rows."""
    ranked = df.sort(["s1_id", prob_col, "target_id"], descending=[False, True, False]).with_columns(
        pl.int_range(1, pl.len() + 1).over("s1_id").alias("_k"))
    n = int(ranked["_k"].max()) if ranked.height else 0
    wide = ranked.pivot(on="_k", index="s1_id", values=prob_col, aggregate_function="first")
    cols = [str(i) for i in range(1, n + 1)]
    for c in cols:
        if c not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0).alias(c))
    P = np.clip(wide.select(cols).fill_null(0.0).to_numpy().astype(np.float64), 0.0, 1.0)
    best_k = np.argmax(expected_f05_matrix(P), axis=1)
    chosen = pl.DataFrame({"s1_id": wide["s1_id"], "_chosen": best_k})
    return ranked.join(chosen, on="s1_id").filter(pl.col("_k") <= pl.col("_chosen")).drop("_k", "_chosen")

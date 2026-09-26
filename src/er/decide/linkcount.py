"""Step 5, unexplored area 4: the link-count structure as an explicit, label-free prior.

The true link count per S1 is strongly structured and, on the training truth, essentially
country-invariant: India averages 3.4645 links per S1 and the US 3.4591, with singleton rates 5.588%
and 5.583%. Per source the counts are bounded (S2 in 0..5, S3 in 0..6) and the total peaks at 3-4.
The decision layer used none of this, and the frozen policy under-links on an unseen country
(France ~3.08 links per record against the ~3.46 prior).

Both uses here are label-free, so they apply to France at test time:

  * `prior_matched_threshold` moves the operating point until the predicted link rate matches the
    prior. This is black-box label-shift correction (Lipton et al. 2018) applied to the count.
  * `select_tilted` addresses the documented reason the exact expected-F0.5 rule lost to a plain
    threshold: the Poisson-binomial treats a record's candidates as independent, which they are not
    (they share its name and address), so the implied count distribution comes out far too flat.
    Tilting the count posterior onto the empirical prior corrects the distribution without touching
    the per-candidate probabilities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from er import io
from er.decide.exact_f05 import expected_f05_matrix, pmf_all

PRIOR_PATH = io.REPORTS / "decide" / "link_prior.json"


@dataclass(frozen=True)
class LinkPrior:
    """Empirical P(#links) over the whole S1 population, in total and per source."""

    total: np.ndarray  # total[t] = P(T = t), t = 0..len-1
    per_src: dict[str, np.ndarray]

    @property
    def mean_total(self) -> float:
        return float(np.arange(len(self.total)) @ self.total)

    def mean_src(self, src: str) -> float:
        a = self.per_src[src]
        return float(np.arange(len(a)) @ a)

    @property
    def caps(self) -> dict[str, int]:
        """Largest count ever observed per source — a hard ceiling, not a tuned parameter."""
        return {s: int(np.max(np.flatnonzero(a > 0))) for s, a in self.per_src.items()}

    def to_json(self) -> dict:
        return {"total": self.total.tolist(), "per_src": {s: a.tolist() for s, a in self.per_src.items()},
                "mean_total": self.mean_total, "caps": self.caps}


def fit_prior(truth: pl.DataFrame) -> LinkPrior:
    """Count distribution from the ground truth. `truth`: s1_id + truth (list of target ids)."""
    n = truth.height
    counts = truth.select(
        pl.col("truth").list.len().alias("t"),
        *[pl.col("truth").list.eval(pl.element().str.starts_with(s)).list.sum().alias(s)
          for s in io.TARGET_SOURCES],
    )

    def hist(col: str) -> np.ndarray:
        c = counts[col].to_numpy()
        return np.bincount(c, minlength=int(c.max()) + 1).astype(np.float64) / n

    return LinkPrior(total=hist("t"), per_src={s: hist(s) for s in io.TARGET_SOURCES})


def load_prior(path: Path = PRIOR_PATH, truth: pl.DataFrame | None = None) -> LinkPrior:
    """Cached prior; fits from truth on first use. Train-only — never fitted on anything scored."""
    if path.exists():
        d = json.loads(path.read_text())
        return LinkPrior(total=np.array(d["total"]),
                         per_src={s: np.array(a) for s, a in d["per_src"].items()})
    prior = fit_prior(io.load_truth() if truth is None else truth)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prior.to_json(), indent=2))
    return prior


# ---------------------------------------------------------------- operating point


def prior_matched_threshold(scored: pl.DataFrame, n_s1: int, target_mean: float,
                           prob_col: str = "p") -> float:
    """The cutoff at which predicted links per S1 equals `target_mean`. No labels involved.

    `n_s1` is the whole S1 universe, including records with no candidates at all — otherwise the rate
    is computed against the wrong denominator and the threshold comes out too low.
    """
    want = round(target_mean * n_s1)
    if want <= 0:
        return 1.0
    p = np.sort(scored[prob_col].to_numpy())[::-1]
    if want >= len(p):
        return float(p[-1])
    return float(p[want - 1])


def apply_caps(scored: pl.DataFrame, prior: LinkPrior, prob_col: str = "p") -> pl.DataFrame:
    """Drop anything past the largest per-source count the truth has ever shown."""
    caps = prior.caps
    keep = pl.col(prob_col).rank("ordinal", descending=True).over("s1_id", "src") <= pl.col("src").replace_strict(
        caps, default=max(caps.values()), return_dtype=pl.Int32)
    return scored.filter(keep)


def select_by_expected_count(df: pl.DataFrame, prior: LinkPrior, blend: float = 0.0,
                            prob_col: str = "p") -> pl.DataFrame:
    """Top-k per S1 with k from the expected count: (1-blend)*sum(p) shrunk toward the prior mean."""
    per_s1 = df.group_by("s1_id").agg(pl.col(prob_col).sum().alias("_sum"))
    k = ((1 - blend) * pl.col("_sum") + blend * prior.mean_total).round().cast(pl.Int32).clip(lower_bound=0)
    per_s1 = per_s1.with_columns(k.alias("_k"))
    ranked = df.sort(["s1_id", prob_col, "target_id"], descending=[False, True, False]).with_columns(
        pl.int_range(1, pl.len() + 1).over("s1_id").alias("_rank"))
    return ranked.join(per_s1.select("s1_id", "_k"), on="s1_id", how="left").filter(
        pl.col("_rank") <= pl.col("_k")).drop("_rank", "_k")


# ---------------------------------------------------------------- prior-tilted exact F0.5


def tilt_weights(prior: np.ndarray, implied: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """w[t] = prior(t) / implied(t): how much the model's own count distribution must move.

    w == 1 everywhere means the model already reproduces the prior and the tilt is a no-op.
    """
    n = max(len(prior), len(implied))
    pr = np.zeros(n)
    pr[:len(prior)] = prior
    im = np.zeros(n)
    im[:len(implied)] = implied
    return np.where(im > floor, pr / np.maximum(im, floor), 0.0)


def _matrix(df: pl.DataFrame, prob_col: str) -> tuple[pl.DataFrame, np.ndarray, pl.DataFrame]:
    """Per-S1 probability matrix, rows padded with 0, sorted descending."""
    ranked = df.sort(["s1_id", prob_col, "target_id"], descending=[False, True, False]).with_columns(
        pl.int_range(1, pl.len() + 1).over("s1_id").alias("_k"))
    n = int(ranked["_k"].max()) if ranked.height else 0
    wide = ranked.pivot(on="_k", index="s1_id", values=prob_col, aggregate_function="first")
    cols = [str(i) for i in range(1, n + 1)]
    for c in cols:
        if c not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0).alias(c))
    P = np.clip(wide.select(cols).fill_null(0.0).to_numpy().astype(np.float64), 0.0, 1.0)
    return ranked, P, wide


def implied_count_pmf(P: np.ndarray) -> np.ndarray:
    """The count distribution the independent model implies, averaged over S1."""
    return pmf_all(P).mean(axis=0)


def _moments(pmf: np.ndarray) -> tuple[float, float]:
    t = np.arange(len(pmf))
    mean = float(t @ pmf)
    return mean, float((t ** 2 @ pmf) - mean ** 2)


def count_diagnostics(scored: pl.DataFrame, prior: LinkPrior, n_s1: int | None = None,
                     prob_col: str = "p") -> dict:
    """Is the count prior worth using? Compare the model's implied count spread against the truth.

    Simulation at the model's operating quality (see docs/step4_5_deep_dive.md) says the tilted rule
    only beats a tuned threshold when the implied variance is inflated relative to the true prior:
    matched (2.88 vs 2.91) loses by 0.0008, inflated (5.05 vs 2.91) wins by 0.0050. So this ratio,
    which costs one pass over the OOF, decides whether the rest of area 4 is worth any GPU time.
    """
    _ranked, P, _wide = _matrix(scored, prob_col)
    implied = implied_count_pmf(P)
    im_mean, im_var = _moments(implied)
    pr_mean, pr_var = _moments(prior.total)
    n = scored["s1_id"].n_unique() if n_s1 is None else n_s1
    return {
        "implied_mean": im_mean, "implied_var": im_var,
        "prior_mean": pr_mean, "prior_var": pr_var,
        "var_ratio": im_var / pr_var if pr_var else None,
        "mean_ratio": im_mean / pr_mean if pr_mean else None,
        "scored_mass_per_s1": float(scored[prob_col].sum()) / max(n, 1),
        "tilt_recommended": bool(pr_var and im_var / pr_var > 1.25),
    }


def match_expected_count(df: pl.DataFrame, target: float, group_col: str | None = None,
                         prob_col: str = "p", tol: float = 0.0) -> pl.DataFrame:
    """KPM: per group, scale odds by alpha so sum(p) per S1 equals `target` (the country-invariant prior).
    Monotone, so it moves only value-based rules. India proxy: +0.0027 alone, +0.0004 on top of band + EXR.
    Groups within `tol` (relative) are left alone: in-distribution drift of 1-1.5% costs -0.0001 if matched."""
    def one(g: pl.DataFrame) -> pl.DataFrame:
        drift = float(g[prob_col].sum()) / max(g["s1_id"].n_unique(), 1) / target - 1
        return g if abs(drift) <= tol else _scale_to(g, target, prob_col)

    if group_col is None:
        return one(df)
    return pl.concat([one(g) for _, g in df.group_by(group_col, maintain_order=True)])


def _scale_to(df: pl.DataFrame, target: float, prob_col: str) -> pl.DataFrame:
    q = np.clip(df[prob_col].to_numpy().astype(np.float64), 1e-6, 1 - 1e-6)
    odds, n = q / (1 - q), max(df["s1_id"].n_unique(), 1)
    lo, hi = -4.0, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        a = np.exp(mid)
        lo, hi = (mid, hi) if (a * odds / (1 + a * odds)).sum() / n < target else (lo, mid)
    a = np.exp((lo + hi) / 2)
    return df.with_columns(pl.Series(prob_col, (a * odds / (1 + a * odds)).astype(np.float32)))


def select_tilted(df: pl.DataFrame, prior: LinkPrior, prob_col: str = "p",
                 strength: float = 1.0, group_col: str | None = None) -> pl.DataFrame:
    """Exact expected-F0.5 selection with the count posterior tilted onto the empirical prior.

    `strength` interpolates in log space: 0 reproduces `exact_f05.select`, 1 applies the full tilt.
    `group_col` computes the tilt per group (e.g. country label) from that group's own implied counts.
    """
    if group_col is not None:
        return pl.concat([select_tilted(g.drop(group_col), prior, prob_col, strength)
                          for _, g in df.group_by(group_col, maintain_order=True)])
    if df.height == 0:
        return df.select("s1_id", "target_id")
    ranked, P, wide = _matrix(df, prob_col)
    w = tilt_weights(prior.total, implied_count_pmf(P))
    if strength != 1.0:
        w = np.where(w > 0, w ** strength, 0.0)
    best_k = np.argmax(expected_f05_matrix(P, tilt=w), axis=1)
    chosen = pl.DataFrame({"s1_id": wide["s1_id"], "_chosen": best_k})
    return ranked.join(chosen, on="s1_id").filter(pl.col("_k") <= pl.col("_chosen")).drop("_k", "_chosen")

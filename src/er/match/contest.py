"""Contest-relative band-combiner features: how a claimant's street compares with its rivals' for the same target.

Sibling families (same name, house and town, different street) are 3x denser in France. On train (name key + house)
contests with one owner among the claimants, the best `addr_content` match (street + city words) is the owner
99.65% (US) / 98.6% (India) of the time and wrong 0.16% / 0.23%; the street segment alone (`street_line`) is right
88.6% / 56.5% and wrong 6.9% / 3.5%, the first comma segment wrong 17.5% / 10.7%. So the address margin leads; the
street-only margin is an extra column for same-town siblings, whose city words tie. Rivals are every S1 whose ranker
shortlist holds the target, over the full universe the bundle's competitor features use, so the fit (OOF rows) and
test (all S1) see the same rival density.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from er import io

CONTEST_COLS = ["contest_addr_margin", "contest_best_addr", "contest_street_margin", "contest_same_name_rivals"]


def _sim(claims: pl.DataFrame, q: str, t: str) -> np.ndarray:
    return cpdist(claims[q].fill_null("").to_list(), claims[t].fill_null("").to_list(), scorer=fuzz.token_set_ratio,
                  workers=-1, dtype=np.float32) / 100


def _margin(df: pl.DataFrame, sim: str) -> tuple[pl.Expr, pl.Expr]:
    """(own similarity minus the best rival's, 0 without a rival) and (unique best among the claimants)."""
    best = pl.col(sim).max().over("target_id")
    n_best = (pl.col(sim) == best).sum().over("target_id")
    second = pl.col(sim).sort(descending=True).get(1, null_on_oob=True).over("target_id")
    rival_best = pl.when((pl.col(sim) == best) & (n_best == 1)).then(second.fill_null(pl.col(sim))).otherwise(best)
    return (pl.col(sim) - rival_best).cast(pl.Float32), ((pl.col(sim) == best) & (n_best == 1)).cast(pl.Float32)


def contest_from_claims(claims: pl.DataFrame) -> pl.DataFrame:
    """claims: one row per (s1_id, target_id) with q_/t_addr_content, q_/t_addr_street and q_name_min_key."""
    df = claims.with_columns(pl.Series("_addr", _sim(claims, "q_addr_content", "t_addr_content")),
                             pl.Series("_street", _sim(claims, "q_addr_street", "t_addr_street")))
    addr_margin, best_addr = _margin(df, "_addr")
    street_margin, _ = _margin(df, "_street")
    key = pl.col("q_name_min_key").fill_null("")
    same_name = pl.when(key != "").then(pl.len().over("target_id", "q_name_min_key") - 1).otherwise(0)
    return df.select("s1_id", "target_id", addr_margin.alias("contest_addr_margin"),
                     best_addr.alias("contest_best_addr"), street_margin.alias("contest_street_margin"),
                     same_name.log1p().cast(pl.Float32).alias("contest_same_name_rivals"))


def universe_claims(targets: pl.Series, config: str, split: str, k: int, scope: str) -> pl.DataFrame:
    """Every (s1_id, target_id) in the universe's pass-1 shortlists whose target is in `targets`."""
    from er.match.dataset import Universe

    if split == "train" and scope != "full":
        raise ValueError(f"contest features need a full-scope bundle (test rivals are all S1), got scope {scope!r}")
    uni = Universe(config, split, k, "full")
    want = targets.unique().implode()
    return pl.concat([pl.read_parquet(p, columns=["s1_id", "target_id"]).filter(pl.col("target_id").is_in(want))
                      for p in uni.shortlist_paths()]).unique(["s1_id", "target_id"])


def contest_features(pairs: pl.DataFrame, split: str, config: str, k: int, scope: str) -> pl.DataFrame:
    """CONTEST_COLS for `pairs` (s1_id, target_id); rivals from the full universe, streets from `load_street`."""
    from er.normalize.pipeline import NormConfig, load_norm, load_street

    nc = NormConfig(**tomllib.loads(Path(config).read_text(encoding="utf-8")).get("norm", {}))
    claims = universe_claims(pairs["target_id"], config, split, k, scope)
    streets = pl.concat([load_street(split, s, nc) for s in io.SOURCES])
    addr = pl.concat([load_norm(split, s, nc).select("entity_id", "addr_content") for s in io.SOURCES]).join(
        streets, on="entity_id", how="left")
    names = load_norm(split, "S1", nc).select(pl.col("entity_id").alias("s1_id"), pl.col("name_min_key")
                                              .alias("q_name_min_key"))
    side = {"addr_content": "addr_content", "addr_street": "addr_street"}
    claims = (claims.join(addr.rename({"entity_id": "s1_id", **{k: f"q_{v}" for k, v in side.items()}}), on="s1_id",
                          how="left")
              .join(addr.rename({"entity_id": "target_id", **{k: f"t_{v}" for k, v in side.items()}}), on="target_id",
                    how="left")
              .join(names, on="s1_id", how="left"))
    return pairs.select("s1_id", "target_id").join(contest_from_claims(claims), on=["s1_id", "target_id"], how="left")

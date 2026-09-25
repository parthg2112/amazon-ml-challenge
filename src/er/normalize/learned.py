"""N4: mine alias tables from TRAIN-fold true pairs only (validation stays unseen)."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from er import io
from er.normalize.pipeline import NormConfig, load_norm
from er.normalize.profiles import profile_for
from er.normalize.text import normalize_text
from er.split import subset_ids


def _train_pairs(cfg: NormConfig, split_df: pl.DataFrame) -> pl.DataFrame:
    ids = subset_ids(split_df, "train")
    truth = io.load_truth().filter(pl.col("s1_id").is_in(ids.implode()))
    links = io.truth_long(truth)
    s1 = load_norm("train", "S1", cfg).select(
        pl.col("entity_id").alias("s1_id"), pl.col("name_full").alias("q_name"), pl.col("addr_region").alias("q_region")
    )
    tg = pl.concat([
        load_norm("train", s, cfg).select(
            pl.col("entity_id").alias("target_id"), pl.col("name_full").alias("t_name"), "name_native", "addr_region"
        )
        for s in io.TARGET_SOURCES
    ])
    return links.join(s1, on="s1_id").join(tg, on="target_id")


def mine_name_tokens(pairs: pl.DataFrame, min_support: int, min_precision: float) -> dict[str, str]:
    """Positional alignment of transliterated native-script names with equal token counts."""
    aligned = (
        pairs.filter(pl.col("name_native"))
        .select(pl.col("t_name").str.split(" ").alias("t"), pl.col("q_name").str.split(" ").alias("q"))
        .filter(pl.col("t").list.len() == pl.col("q").list.len())
        .explode("t", "q")
        .filter(pl.col("t") != pl.col("q"))
    )
    counts = aligned.group_by("t", "q").len("n")
    totals = aligned.group_by("t").len("tot")
    best = (
        counts.join(totals, on="t").sort("n", descending=True).group_by("t", maintain_order=True).first()
        .filter((pl.col("n") >= min_support) & (pl.col("n") / pl.col("tot") >= min_precision))
    )
    return dict(zip(best["t"], best["q"]))


def mine_regions(cfg: NormConfig, split_df: pl.DataFrame, pairs: pl.DataFrame,
                 min_support: int, min_precision: float) -> dict[str, str]:
    """Unrecognized last address segment of a target ↔ recognized region of its true S1."""
    unknown = pairs.filter(pl.col("addr_region") == "").select("target_id", "q_region").filter(pl.col("q_region") != "")
    raw = pl.concat([io.load_source("train", s).select("entity_id", "business_address", "country")
                     for s in io.TARGET_SOURCES])
    seg = unknown.join(raw, left_on="target_id", right_on="entity_id").with_columns(
        pl.col("business_address").str.split(",").list.last().alias("seg")
    )
    seg = seg.with_columns(
        pl.struct("seg", "country").map_elements(
            lambda r: " ".join(t for t in normalize_text(r["seg"], cfg.translit,
                                                         profile_for(r["country"]).apostrophe_splits).split()
                               if not t.isdigit()),
            return_dtype=pl.Utf8,
        ).alias("key")
    ).filter(pl.col("key") != "")
    counts = seg.group_by("key", "q_region").len("n")
    totals = seg.group_by("key").len("tot")
    best = (
        counts.join(totals, on="key").sort("n", descending=True).group_by("key", maintain_order=True).first()
        .filter((pl.col("n") >= min_support) & (pl.col("n") / pl.col("tot") >= min_precision))
    )
    return dict(zip(best["key"], best["q_region"]))


def mine(cfg: NormConfig, out: Path, min_support: int = 20, min_precision: float = 0.9) -> dict:
    split_df = pl.read_parquet(io.DATA / "split.parquet")
    pairs = _train_pairs(cfg, split_df)
    tables = {
        "name_tokens": mine_name_tokens(pairs, min_support, min_precision),
        "regions": mine_regions(cfg, split_df, pairs, min_support, min_precision),
        "meta": {"source_variant": cfg.variant, "min_support": min_support, "min_precision": min_precision,
                 "fold": "train"},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tables, ensure_ascii=False, indent=1), encoding="utf-8")
    return tables

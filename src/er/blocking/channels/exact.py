"""B1 exact-key channels via Polars hash joins, with a block-size cap."""

from __future__ import annotations

import polars as pl


def _keys(df: pl.DataFrame, col: str, id_name: str) -> pl.DataFrame:
    k = df.select(pl.col("entity_id").alias(id_name), pl.col(col).alias("key"))
    if k.schema["key"] == pl.List(pl.Utf8):
        k = k.explode("key")
    return k.filter(pl.col("key").is_not_null() & (pl.col("key") != "")).unique()


_TARGET_CACHE: dict[tuple, tuple[pl.DataFrame, pl.DataFrame]] = {}


def _target_side(targets: pl.DataFrame, target_col: str, max_block: int, cache_key: tuple | None):
    if cache_key is not None and cache_key in _TARGET_CACHE:
        return _TARGET_CACHE[cache_key]
    tk = _keys(targets, target_col, "target_id")
    sizes = tk.group_by("key").len("block")
    out = (tk.join(sizes.filter(pl.col("block") <= max_block), on="key"), sizes)
    if cache_key is not None:
        _TARGET_CACHE[cache_key] = out
    return out


def exact_channel(queries: pl.DataFrame, targets: pl.DataFrame, query_col: str, target_col: str,
                  max_block: int, cache_key: tuple | None = None) -> tuple[pl.DataFrame, dict]:
    """cache_key: identifies an unchanging target pool so its key table is built once per process."""
    qk = _keys(queries, query_col, "s1_id")
    tk, sizes = _target_side(targets, target_col, max_block, cache_key)
    pairs = qk.join(tk, on="key").select(
        "s1_id", "target_id", (1.0 / pl.col("block")).alias("score")
    ).group_by("s1_id", "target_id").agg(pl.col("score").max())
    matched = qk.join(sizes, on="key", how="left")
    stats = {
        "query_keys": qk.height,
        "keys_over_cap": int(matched.filter(pl.col("block") > max_block).height),
    }
    return pairs, stats


def house_token_keys(df: pl.DataFrame, df_table: pl.DataFrame, rarest: int | None) -> pl.DataFrame:
    """house|token keys. rarest=None → every content token (targets); else the n rarest known tokens (queries)."""
    toks = (
        df.filter(pl.col("addr_house") != "")
        .select("entity_id", "addr_house", pl.col("addr_content").str.split(" ").alias("tok"))
        .explode("tok")
        .filter(pl.col("tok").str.len_chars() > 1)
        .unique()
    )
    if rarest is not None:
        toks = (
            toks.join(df_table, on="tok", how="inner")
            .sort("df", "tok")
            .group_by("entity_id", maintain_order=True)
            .head(rarest)
        )
    return toks.select(
        "entity_id", (pl.col("addr_house") + "|" + pl.col("tok")).alias("house_tok")
    ).group_by("entity_id").agg("house_tok")


def token_df(targets: pl.DataFrame) -> pl.DataFrame:
    return (
        targets.select(pl.col("addr_content").str.split(" ").alias("tok"))
        .explode("tok")
        .filter(pl.col("tok").str.len_chars() > 1)
        .group_by("tok")
        .len("df")
    )

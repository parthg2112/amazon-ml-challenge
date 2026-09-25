"""Stage-2 set-aware model on stage-1 OOF probabilities (roadmap Phase E2, targets the recall-only loss).

Half the remaining v2 loss is S1s with correct predictions but missed true links: true links cluster, so
once an S1 has confident matches, candidates resembling THOSE records are likely true too. Stage 2 sees
each pair's stage-1 p in the context of its S1 (rank, gap to best, confident count, opposite-source best)
plus coherence: dense cosine between this target and the S1's top-1 target. Stacking is honest because
stage-1 OOF p for fold f comes from models that never saw fold f, and stage 2 uses the same folds.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from er import io
from er.blocking.channels.dense import embed_source

STACK_FEATURES = [
    "p", "p_rank_s1", "p_rank_src", "p_gap_top", "p_ratio_top", "s1_max_p", "s1_second_p", "s1_sum_p",
    "s1_n_ge50", "s1_n_ge72", "s1_n_ge90", "src_max_p", "opp_src_max_p", "coh_top1", "coh_top_opp", "is_s3",
]


def _row_maps(split: str) -> dict[str, pl.DataFrame]:
    return {src: io.load_source(split, src).select(pl.col("entity_id").alias("target_id")).with_row_index("row")
            for src in io.TARGET_SOURCES}


def _cosine(ids_a: pl.Series, ids_b: pl.Series, split: str, model: str, maps: dict) -> np.ndarray:
    """Dense cosine between two aligned target-id columns (either source), batched over the mmap'd embeddings."""
    out = np.zeros(ids_a.len(), dtype=np.float32)
    frame = pl.DataFrame({"a": ids_a, "b": ids_b}).with_row_index("i")
    emb = {src: embed_source(split, src, model=model) for src in io.TARGET_SOURCES}
    for sa in io.TARGET_SOURCES:
        for sb in io.TARGET_SOURCES:
            part = frame.filter(pl.col("a").str.starts_with(sa) & pl.col("b").str.starts_with(sb))
            if part.height == 0:
                continue
            ra = part.select(pl.col("a").alias("target_id")).join(maps[sa], on="target_id", how="left",
                                                                    maintain_order="left")["row"].to_numpy()
            rb = part.select(pl.col("b").alias("target_id")).join(maps[sb], on="target_id", how="left",
                                                                    maintain_order="left")["row"].to_numpy()
            idx = part["i"].to_numpy()
            for s in range(0, len(idx), 200_000):
                va = np.asarray(emb[sa][ra[s:s + 200_000]], dtype=np.float32)
                vb = np.asarray(emb[sb][rb[s:s + 200_000]], dtype=np.float32)
                out[idx[s:s + 200_000]] = np.einsum("ij,ij->i", va, vb)
    return out


def build(scored: pl.DataFrame, split: str, model: str) -> pl.DataFrame:
    """scored: s1_id, target_id, src, p (stage-1). Returns scored + STACK_FEATURES."""
    df = scored.with_columns(
        pl.col("p").rank("ordinal", descending=True).over("s1_id").alias("p_rank_s1"),
        pl.col("p").rank("ordinal", descending=True).over("s1_id", "src").alias("p_rank_src"),
        pl.col("p").max().over("s1_id").alias("s1_max_p"),
        pl.col("p").sum().over("s1_id").alias("s1_sum_p"),
        (pl.col("p") >= 0.5).sum().over("s1_id").alias("s1_n_ge50"),
        (pl.col("p") >= 0.72).sum().over("s1_id").alias("s1_n_ge72"),
        (pl.col("p") >= 0.9).sum().over("s1_id").alias("s1_n_ge90"),
        pl.col("p").max().over("s1_id", "src").alias("src_max_p"),
        (pl.col("src") == "S3").cast(pl.Int8).alias("is_s3"),
    )
    second = df.sort(["s1_id", "p"], descending=[False, True]).group_by("s1_id", maintain_order=True).agg(
        pl.col("p").get(1, null_on_oob=True).alias("s1_second_p"),
        pl.col("target_id").first().alias("_top1"),
    )
    opp = df.group_by("s1_id", "src").agg(pl.col("p").max().alias("_m"), pl.col("target_id").sort_by("p").last()
                                          .alias("_top_src"))
    opp = opp.with_columns(pl.when(pl.col("src") == "S2").then(pl.lit("S3")).otherwise(pl.lit("S2")).alias("src"))
    df = (df.join(second, on="s1_id", how="left")
          .join(opp.rename({"_m": "opp_src_max_p", "_top_src": "_top_opp"}), on=["s1_id", "src"], how="left")
          .with_columns(pl.col("opp_src_max_p").fill_null(0.0), pl.col("s1_second_p").fill_null(0.0),
                        (pl.col("s1_max_p") - pl.col("p")).alias("p_gap_top"),
                        (pl.col("p") / pl.col("s1_max_p").clip(1e-6)).alias("p_ratio_top")))
    maps = _row_maps(split)
    coh1 = _cosine(df["target_id"], df["_top1"], split, model, maps)
    top_opp = df["_top_opp"].fill_null(df["target_id"])
    coh2 = _cosine(df["target_id"], top_opp, split, model, maps)
    return df.with_columns(
        pl.Series("coh_top1", np.where(df["target_id"] == df["_top1"], 1.0, coh1).astype(np.float32)),
        pl.Series("coh_top_opp", np.where(df["_top_opp"].is_null().to_numpy(), 0.0, coh2).astype(np.float32)),
    ).drop("_top1", "_top_opp")

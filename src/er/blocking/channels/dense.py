"""B7 dense multilingual retrieval on GPU: fp16 embeddings + exact matmul top-k per source."""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from er import io

MODEL = "intfloat/multilingual-e5-small"  # MIT, 118M params, 384-d


def _texts(df: pl.DataFrame) -> list[str]:
    # e5 convention: symmetric retrieval uses the "query: " prefix on both sides.
    return df.select(
        pl.concat_str([pl.lit("query: "), pl.col("business_name").fill_null(""), pl.lit(" | "),
                       pl.col("business_address").fill_null("")])
    ).to_series().to_list()


def embed_source(split: str, src: str, batch: int = 1024, force: bool = False) -> np.ndarray:
    """Embeds raw text (the model handles scripts/accents itself); cached as float16 .npy."""
    path = io.ARTIFACTS / "cache" / "dense" / f"{MODEL.split('/')[-1]}_{split}_{src}.npy"
    if path.exists() and not force:
        return np.load(path, mmap_mode="r")
    import torch
    from sentence_transformers import SentenceTransformer

    path.parent.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL, device="cuda", model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = 64
    texts = _texts(io.load_source(split, src))
    t0 = time.perf_counter()
    emb = model.encode(texts, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=True).astype(np.float16)
    np.save(path, emb)
    print(f"embedded {split}_{src}: {len(texts):,} in {time.perf_counter() - t0:.0f}s")
    return emb


def dense_channel(queries: pl.DataFrame, targets: dict[str, pl.DataFrame], k: int,
                  query_rows: np.ndarray, split: str = "train", qbatch: int = 256) -> pl.DataFrame:
    """query_rows: row positions of `queries` within the full S1 parquet (to index cached embeddings)."""
    import torch

    q_all = embed_source(split, "S1")
    q = torch.from_numpy(np.ascontiguousarray(q_all[query_rows])).cuda()
    out = []
    for src, tdf in targets.items():
        t = torch.from_numpy(np.array(embed_source(split, src))).cuda()
        rows, cols, vals = [], [], []
        for i in range(0, q.shape[0], qbatch):
            sims = q[i:i + qbatch] @ t.T
            v, idx = torch.topk(sims, k, dim=1)
            rows.append(np.repeat(np.arange(i, i + v.shape[0]), k))
            cols.append(idx.flatten().cpu().numpy())
            vals.append(v.flatten().float().cpu().numpy())
        del t
        torch.cuda.empty_cache()
        out.append(pl.DataFrame({
            "s1_id": queries["entity_id"].gather(np.concatenate(rows)),
            "target_id": tdf["entity_id"].gather(np.concatenate(cols)),
            "score": np.concatenate(vals),
        }))
    return pl.concat(out)

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


def embed_source(split: str, src: str, batch: int = 1024, force: bool = False, model: str = MODEL) -> np.ndarray:
    """Embeds raw text (the model handles scripts/accents itself); cached as float16 .npy per model."""
    path = io.ARTIFACTS / "cache" / "dense" / f"{model.replace(chr(92), '/').rstrip('/').split('/')[-1]}_{split}_{src}.npy"
    if path.exists() and not force:
        return np.load(path, mmap_mode="r")
    import torch
    from sentence_transformers import SentenceTransformer

    path.parent.mkdir(parents=True, exist_ok=True)
    enc = SentenceTransformer(model, device="cuda", model_kwargs={"torch_dtype": torch.float16})
    enc.max_seq_length = 64
    texts = _texts(io.load_source(split, src))
    t0 = time.perf_counter()
    emb = enc.encode(texts, batch_size=batch, normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=True).astype(np.float16)
    tmp = path.with_name(path.stem + ".partial.npy")
    np.save(tmp, emb)
    tmp.replace(path)
    print(f"embedded {split}_{src}: {len(texts):,} in {time.perf_counter() - t0:.0f}s")
    return emb


_GPU: dict[tuple, object] = {}  # (split, src) → (resident fp16 targets, block slices)


def _resident(split: str, src: str, tkey: np.ndarray | None, model: str = MODEL):
    """One split's target embeddings on the GPU (≈7.6 GB for S2+S3), rows sorted by partition key so every
    block is a zero-copy slice. The other split is evicted first."""
    import torch

    key = (split, src, tkey is not None, model)
    if key not in _GPU:
        for k in [k for k in _GPU if k[0] != split or k[1] == src]:
            del _GPU[k]
        torch.cuda.empty_cache()
        emb = embed_source(split, src, model=model)
        if tkey is None:
            order, blocks = np.arange(len(emb)), {"*": (0, len(emb))}
        else:
            order = np.argsort(tkey, kind="stable")
            keys, starts = np.unique(tkey[order], return_index=True)
            ends = list(starts[1:]) + [len(order)]
            blocks = {str(r): (int(a), int(b)) for r, a, b in zip(keys, starts, ends)}
        _GPU[key] = (torch.from_numpy(np.asarray(emb[order])).cuda(), order, blocks)
    return _GPU[key]


def _topk(q, t, k: int, qbatch: int):
    import torch

    k = min(k, t.shape[0])
    vals, idx = [], []
    for i in range(0, q.shape[0], qbatch):
        v, j = torch.topk(q[i:i + qbatch] @ t.T, k, dim=1)
        vals.append(v.float())
        idx.append(j)
    return torch.cat(vals), torch.cat(idx)


def dense_channel(queries: pl.DataFrame, targets: dict[str, pl.DataFrame], k: int,
                  query_rows: np.ndarray, split: str = "train", partition: str | None = None,
                  qbatch: int = 256, model: str = MODEL) -> pl.DataFrame:
    """query_rows: row positions of `queries` within the full S1 parquet (to index cached embeddings).

    partition: keyed query r searches target blocks {r, unknown}; unkeyed queries search all targets.
    """
    import torch

    order_q = np.argsort(query_rows, kind="stable")
    q_all = torch.from_numpy(np.ascontiguousarray(embed_source(split, "S1", model=model)[query_rows[order_q]])).cuda()
    qids = queries["entity_id"].gather(order_q)
    qkey = queries[partition].fill_null("").to_numpy().astype(str)[order_q] if partition else None
    out = []
    with torch.inference_mode():
        for src, tdf in targets.items():
            tkey = tdf[partition].fill_null("").to_numpy().astype(str) if partition else None
            t, t_order, blocks = _resident(split, src, tkey, model)
            tids = tdf["entity_id"].gather(t_order)
            groups = [(np.arange(len(qids)), [(0, t.shape[0])])] if qkey is None else []
            if qkey is not None:
                for r in np.unique(qkey):
                    qi = np.flatnonzero(qkey == r)
                    if r == "":
                        spans = [(0, t.shape[0])]
                    else:
                        spans = [blocks[b] for b in (r, "") if b in blocks]
                    groups.append((qi, spans))
            for qi, spans in groups:
                if not spans:
                    continue
                qt = q_all[torch.from_numpy(qi).cuda()]
                for a, b in spans:
                    v, j = _topk(qt, t[a:b], k, qbatch)
                    out.append(pl.DataFrame({
                        "s1_id": qids.gather(np.repeat(qi, v.shape[1])),
                        "target_id": tids.gather((j + a).flatten().cpu().numpy()),
                        "score": v.flatten().cpu().numpy(),
                    }))
    df = pl.concat(out)
    if partition:
        df = df.sort("score", descending=True).group_by("s1_id", "target_id", maintain_order=True).first()
        df = df.sort("score", descending=True).group_by(
            "s1_id", pl.col("target_id").str.slice(0, 2).alias("_src"), maintain_order=True).head(k).drop("_src")
    return df

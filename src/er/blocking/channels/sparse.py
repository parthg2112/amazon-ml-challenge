"""B2–B5 sparse TF-IDF cosine top-k: parallel hashing + multi-threaded sparse_dot_topn."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from er import io

N_FEATURES = 2**21
THREADS = os.cpu_count() or 12


@dataclass(frozen=True)
class FieldSpec:
    col: str
    analyzer: str = "char_wb"  # char_wb | word
    ngram: tuple[int, int] = (3, 3)
    weight: float = 1.0


def _vectorizer(f: FieldSpec) -> HashingVectorizer:
    return HashingVectorizer(
        analyzer=f.analyzer, ngram_range=tuple(f.ngram), n_features=N_FEATURES,
        alternate_sign=False, norm=None, binary=True, lowercase=False, dtype=np.float32,
        token_pattern=r"\S+",
    )


def _hash_chunk(texts: list[str], f: FieldSpec) -> sp.csr_matrix:
    return _vectorizer(f).transform(texts)


def hash_texts(texts: list[str], f: FieldSpec) -> sp.csr_matrix:
    step = max(1, len(texts) // (THREADS * 4) + 1)
    chunks = [texts[i:i + step] for i in range(0, len(texts), step)]
    parts = Parallel(n_jobs=THREADS, backend="loky")(delayed(_hash_chunk)(c, f) for c in chunks)
    return sp.vstack(parts).tocsr()


def _cache_key(variant: str, f: FieldSpec, src: str) -> str:
    h = hashlib.sha1(json.dumps(asdict(f), sort_keys=True).encode()).hexdigest()[:10]
    return f"{variant}_{f.col}_{h}_{src}"


def target_counts(targets: dict[str, pl.DataFrame], f: FieldSpec, variant: str) -> dict[str, sp.csr_matrix]:
    """Binary hashed counts per target source, cached on disk (targets never change per variant)."""
    out = {}
    cache = io.ARTIFACTS / "cache" / "hashed"
    cache.mkdir(parents=True, exist_ok=True)
    for src, df in targets.items():
        path = cache / f"{_cache_key(variant, f, src)}.npz"
        if path.exists():
            out[src] = sp.load_npz(path).tocsr()
        else:
            out[src] = hash_texts(df[f.col].fill_null("").to_list(), f)
            sp.save_npz(path, out[src], compressed=False)
    return out


def idf_weights(target_mats: dict[str, sp.csr_matrix], max_df: float) -> np.ndarray:
    """Smooth IDF on the whole target pool; features above max_df share are zeroed (cost + noise control)."""
    n = sum(m.shape[0] for m in target_mats.values())
    df = np.zeros(N_FEATURES, dtype=np.int64)
    for m in target_mats.values():
        df += np.bincount(m.indices, minlength=N_FEATURES)
    idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
    idf[df > max_df * n] = 0.0
    return idf


def weighted(m: sp.csr_matrix, idf: np.ndarray, weight: float) -> sp.csr_matrix:
    m = m @ sp.diags(idf)
    m = normalize(m, norm="l2", copy=False)
    if weight != 1.0:
        m = m * weight
    return m.tocsr()


def sparse_topk(qmat: sp.csr_matrix, tmat: sp.csr_matrix, k: int, threshold: float) -> tuple[np.ndarray, ...]:
    tmat_t = tmat.T.tocsr()
    res = sp_matmul_topn(qmat, tmat_t, top_n=k, threshold=threshold, sort=True, n_threads=THREADS)
    res = res.tocoo()
    return res.row, res.col, res.data


def _topk_frame(qids: pl.Series, tids: pl.Series, qmat: sp.csr_matrix, tmat: sp.csr_matrix, k: int,
                threshold: float) -> pl.DataFrame:
    if qmat.shape[0] == 0 or tmat.shape[0] == 0:
        return pl.DataFrame(schema={"s1_id": pl.Utf8, "target_id": pl.Utf8, "score": pl.Float32})
    rows, cols, vals = sparse_topk(qmat, tmat, k, threshold)
    return pl.DataFrame({"s1_id": qids.gather(rows), "target_id": tids.gather(cols),
                         "score": vals.astype(np.float32)})


def _partitioned(qids: pl.Series, qkey: np.ndarray, qmat: sp.csr_matrix, tids: pl.Series, tkey: np.ndarray,
                 tmat: sp.csr_matrix, k: int, threshold: float) -> pl.DataFrame:
    """Query with key r searches targets with key r plus unkeyed targets; unkeyed queries search everything."""
    parts = []
    t_unknown = np.flatnonzero(tkey == "")
    q_known = np.flatnonzero(qkey != "")
    q_unknown = np.flatnonzero(qkey == "")
    parts.append(_topk_frame(qids.gather(q_unknown), tids, qmat[q_unknown], tmat, k, threshold))
    parts.append(_topk_frame(qids.gather(q_known), tids.gather(t_unknown), qmat[q_known], tmat[t_unknown], k,
                             threshold))
    t_order = np.argsort(tkey, kind="stable")
    t_sorted = tkey[t_order]
    for r in np.unique(qkey[q_known]):
        qi = q_known[qkey[q_known] == r]
        lo, hi = np.searchsorted(t_sorted, r, "left"), np.searchsorted(t_sorted, r, "right")
        ti = t_order[lo:hi]
        parts.append(_topk_frame(qids.gather(qi), tids.gather(ti), qmat[qi], tmat[ti], k, threshold))
    df = pl.concat(parts)
    return df.sort("score", descending=True).group_by("s1_id", maintain_order=True).head(k)


_PREPARED: dict[tuple, tuple[list[np.ndarray], dict[str, sp.csr_matrix]]] = {}


def _prepare_targets(targets: dict[str, pl.DataFrame], fields: list[FieldSpec], max_df: float,
                     variant: str) -> tuple[list[np.ndarray], dict[str, sp.csr_matrix]]:
    """IDF + weighted/normalized target matrices, built once per process and reused across query chunks."""
    key = (variant, tuple(fields), max_df)
    if key not in _PREPARED:
        idfs, t_parts = [], {src: [] for src in targets}
        for f in fields:
            tcounts = target_counts(targets, f, variant)
            idf = idf_weights(tcounts, max_df)
            idfs.append(idf)
            for src, m in tcounts.items():
                t_parts[src].append(weighted(m, idf, f.weight))
            del tcounts
        tmats = {src: normalize(sp.hstack(p).tocsr()) if len(p) > 1 else p[0] for src, p in t_parts.items()}
        _PREPARED[key] = (idfs, tmats)
    return _PREPARED[key]


def sparse_channel(queries: pl.DataFrame, targets: dict[str, pl.DataFrame], fields: list[FieldSpec],
                   k: int, max_df: float, threshold: float, variant: str,
                   partition: str | None = None) -> pl.DataFrame:
    idfs, tmats = _prepare_targets(targets, fields, max_df, variant)
    q_parts = [weighted(hash_texts(queries[f.col].fill_null("").to_list(), f), idf, f.weight)
               for f, idf in zip(fields, idfs)]
    qmat = normalize(sp.hstack(q_parts).tocsr()) if len(fields) > 1 else q_parts[0]
    qids = queries["entity_id"]
    out = []
    for src, tmat in tmats.items():
        tids = targets[src]["entity_id"]
        if partition:
            out.append(_partitioned(qids, queries[partition].fill_null("").to_numpy().astype(str), qmat, tids,
                                    targets[src][partition].fill_null("").to_numpy().astype(str), tmat, k,
                                    threshold))
        else:
            out.append(_topk_frame(qids, tids, qmat, tmat, k, threshold))
    return pl.concat(out)

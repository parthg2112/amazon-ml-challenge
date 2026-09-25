"""B2–B5 sparse TF-IDF cosine top-k: parallel hashing + multi-threaded sparse_dot_topn."""

from __future__ import annotations

import hashlib
import json
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
THREADS = io.THREADS


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
            tmp = path.with_name(path.stem + ".partial.npz")  # rename only when complete: no truncated caches
            sp.save_npz(tmp, out[src], compressed=False)
            tmp.replace(path)
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


def _topk_block(qids: pl.Series, qmat: sp.csr_matrix, block: tuple[pl.Series, sp.csr_matrix], k: int,
                threshold: float) -> pl.DataFrame:
    tids, tmat_t = block
    if qmat.shape[0] == 0 or tmat_t.shape[1] == 0:
        return pl.DataFrame(schema={"s1_id": pl.Utf8, "target_id": pl.Utf8, "score": pl.Float32})
    res = sp_matmul_topn(qmat, tmat_t, top_n=k, threshold=threshold, sort=True, n_threads=THREADS).tocoo()
    return pl.DataFrame({"s1_id": qids.gather(res.row), "target_id": tids.gather(res.col),
                         "score": res.data.astype(np.float32)})


Blocks = dict[str, tuple[pl.Series, sp.csr_matrix]]  # block key → (target ids, transposed features×targets)
_PREPARED: dict[tuple, tuple[list[np.ndarray], dict[str, Blocks]]] = {}
ALL = "*"


def _blocks(tids: pl.Series, tmat: sp.csr_matrix, tkey: np.ndarray | None) -> Blocks:
    """Pre-transposed target blocks, built once: the per-call transpose and row slicing were the old hot spot."""
    if tkey is None:
        return {ALL: (tids, tmat.T.tocsr())}
    order = np.argsort(tkey, kind="stable")
    keys, starts = np.unique(tkey[order], return_index=True)
    ends = list(starts[1:]) + [len(order)]
    return {str(r): (tids.gather(order[a:b]), tmat[order[a:b]].T.tocsr()) for r, a, b in zip(keys, starts, ends)}


def _prepare_targets(targets: dict[str, pl.DataFrame], fields: list[FieldSpec], max_df: float, variant: str,
                     partition: str | None) -> tuple[list[np.ndarray], dict[str, Blocks]]:
    """IDF + weighted/normalized target blocks, built once per process and reused across query chunks."""
    key = (variant, tuple(fields), max_df, partition)
    if key not in _PREPARED:
        idfs, t_parts = [], {src: [] for src in targets}
        for f in fields:
            tcounts = target_counts(targets, f, variant)
            idf = idf_weights(tcounts, max_df)
            idfs.append(idf)
            for src, m in tcounts.items():
                t_parts[src].append(weighted(m, idf, f.weight))
            del tcounts
        blocks = {}
        for src, parts in list(t_parts.items()):
            tmat = normalize(sp.hstack(parts).tocsr()) if len(parts) > 1 else parts[0]
            del t_parts[src]
            tkey = targets[src][partition].fill_null("").to_numpy().astype(str) if partition else None
            blocks[src] = _blocks(targets[src]["entity_id"], tmat, tkey)
        _PREPARED[key] = (idfs, blocks)
    return _PREPARED[key]


def _search(qids: pl.Series, qmat: sp.csr_matrix, blocks: Blocks, qkey: np.ndarray | None, k: int,
            threshold: float) -> pl.DataFrame:
    """Keyed query r searches blocks {r, unknown}; unkeyed queries search every block. Exact top-k per query."""
    if qkey is None:
        return _topk_block(qids, qmat, blocks[ALL], k, threshold)
    parts = []
    known = qkey != ""
    q_unknown = np.flatnonzero(~known)
    if len(q_unknown):
        sub_ids, sub = qids.gather(q_unknown), qmat[q_unknown]
        parts += [_topk_block(sub_ids, sub, b, k, threshold) for b in blocks.values()]
    q_known = np.flatnonzero(known)
    if "" in blocks and len(q_known):
        parts.append(_topk_block(qids.gather(q_known), qmat[q_known], blocks[""], k, threshold))
    for r in np.unique(qkey[q_known]):
        if r in blocks:
            qi = q_known[qkey[q_known] == r]
            parts.append(_topk_block(qids.gather(qi), qmat[qi], blocks[r], k, threshold))
    df = pl.concat(parts)
    return df.sort("score", descending=True).group_by("s1_id", maintain_order=True).head(k)


def sparse_channel(queries: pl.DataFrame, targets: dict[str, pl.DataFrame], fields: list[FieldSpec],
                   k: int, max_df: float, threshold: float, variant: str,
                   partition: str | None = None) -> pl.DataFrame:
    idfs, blocks = _prepare_targets(targets, fields, max_df, variant, partition)
    q_parts = [weighted(hash_texts(queries[f.col].fill_null("").to_list(), f), idf, f.weight)
               for f, idf in zip(fields, idfs)]
    qmat = normalize(sp.hstack(q_parts).tocsr()) if len(fields) > 1 else q_parts[0]
    qkey = queries[partition].fill_null("").to_numpy().astype(str) if partition else None
    return pl.concat([_search(queries["entity_id"], qmat, b, qkey, k, threshold) for b in blocks.values()])

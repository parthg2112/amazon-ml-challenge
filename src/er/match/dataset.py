"""Two-pass Step 4 dataset over a full S1 universe (review M1, M3; docs/step4_5_review.md).

Pass 1 caches each chunk's ranker shortlist plus the context score for EVERY S1 in the universe, so
cross-S1 features (claimants per record, margin to the best competitor) see all competitors. Pass 2
builds full pair features only where needed: val S1 for training, every test S1 for scoring.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
import tomllib
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from er import io
from er.blocking.channels.dense import MODEL, embed_source
from er.blocking.runner import CHUNK, Context, _chunks, _match_pool, precut, release_caches, train_ranker
from er.match import features
from er.match.idf import build_word_idf
from er.split import load_split, subset_ids

SHORTLIST_VERSION = 1  # bump when pass-1 logic changes
FEATURE_VERSION = 1  # bump when pass-2 features change
CACHE = io.ARTIFACTS / "cache" / "match"


def _h(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def universe_chunks(split: str, cfg: dict) -> list[pl.Series]:
    """Chunk order reuses cached channel outputs: val chunks, then the ranker sample, then the rest."""
    size = cfg.get("chunk", CHUNK)
    sp = load_split()
    if split == "test":
        from er.normalize.pipeline import NormConfig, load_norm

        return list(_chunks(load_norm("test", "S1", NormConfig(**cfg.get("norm", {})))["entity_id"], size))
    val = subset_ids(sp, "val")
    rtrain = subset_ids(sp, cfg["ranker"].get("train_subset", "rtrain")).sort()
    rest = subset_ids(sp, "train").filter(~subset_ids(sp, "train").is_in(rtrain.implode())).sort()
    return list(_chunks(val, size)) + list(_chunks(rtrain, size)) + list(_chunks(rest, size))


def _dense_model(cfg: dict) -> str:
    return next((c.get("model", MODEL) for c in cfg["channels"] if c["type"] == "dense"), MODEL)


class Universe:
    def __init__(self, config_path: str, split: str, k: int = 10):
        self.config_path, self.split, self.k = config_path, split, k
        self.cfg = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
        self.chunks = universe_chunks(split, self.cfg)
        self.key = _h(SHORTLIST_VERSION, self.cfg, k, split)
        self.dir = CACHE / f"{split}_{self.key}"
        self._ranker = None
        self._ctx: Context | None = None
        self._pools = None

    # ---------- pass 1 ----------
    def _shortlist_path(self, i: int, chunk: pl.Series) -> Path:
        return self.dir / f"shortlist_{i:03d}_{_h(chunk.to_list())}.parquet"

    def _ensure_ranker(self):
        if self._ranker is None:
            train_ctx = Context(self.cfg, "train") if self.split == "train" else None
            self._ranker = train_ranker(self.cfg, train_ctx)
            if self.split == "train":
                self._ctx = train_ctx
            else:
                del train_ctx
                release_caches()
        return self._ranker

    def _norm(self):
        from er.normalize.pipeline import NormConfig

        return NormConfig(**self.cfg.get("norm", {}))

    def _context(self) -> Context:
        if self._ctx is None:
            self._ctx = Context(self.cfg, self.split)
        return self._ctx

    def _slim_pools(self):
        if self._pools is None:
            from er.normalize.pipeline import load_norm

            norm = self._norm()
            cols = ["entity_id", "name_core", "addr_content"]
            self._pools = (load_norm(self.split, "S1", norm).select(cols),
                           pl.concat([load_norm(self.split, s, norm).select(cols) for s in io.TARGET_SOURCES]))
        return self._pools

    def shortlists(self) -> None:
        """Build any missing pass-1 chunk files (resumable; atomic writes)."""
        from er.blocking import ranker

        self.dir.mkdir(parents=True, exist_ok=True)
        todo = [(i, c) for i, c in enumerate(self.chunks) if not self._shortlist_path(i, c).exists()]
        if not todo:
            return
        model, feats, channels, _ = self._ensure_ranker()
        ctx = self._context()
        s1p, tp = self._slim_pools()
        for n, (i, chunk) in enumerate(todo):
            t0 = time.perf_counter()
            cands, queries = ctx.collect(chunk, f"{self.split}-shortlist", {})
            ranked = ranker.rank_by_model(ranker.build(precut(cands, self.cfg), channels, queries, ctx.all_targets),
                                          model, feats)
            keep = ranked.filter(pl.col("frank") <= self.k).select("s1_id", "target_id", "src", "p", "frank")
            filtered = cands.join(keep.select("s1_id", "target_id"), on=["s1_id", "target_id"], how="semi")
            pairs = ranker.provenance(filtered, channels).join(keep, on=["s1_id", "target_id", "src"], how="left")
            j = pairs.select("s1_id", "target_id").join(
                s1p.rename({"entity_id": "s1_id", "name_core": "qn", "addr_content": "qa"}), on="s1_id", how="left"
            ).join(tp.rename({"entity_id": "target_id", "name_core": "tn", "addr_content": "ta"}), on="target_id",
                   how="left")
            sn = cpdist(j["qn"].fill_null("").to_list(), j["tn"].fill_null("").to_list(),
                        scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
            sa = cpdist(j["qa"].fill_null("").to_list(), j["ta"].fill_null("").to_list(),
                        scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
            pairs = pairs.with_columns(pl.Series(features.CTX_SCORE, (sn + sa) / 2))
            path = self._shortlist_path(i, chunk)
            tmp = path.with_suffix(".partial")
            pairs.write_parquet(tmp)
            tmp.replace(path)
            print(f"  shortlist {self.split} chunk {i} ({n + 1}/{len(todo)} new): {pairs.height:,} pairs "
                  f"in {time.perf_counter() - t0:.0f}s", flush=True)
            del cands, queries, ranked, pairs, j
            gc.collect()

    def shortlist(self, i: int) -> pl.DataFrame:
        return pl.read_parquet(self._shortlist_path(i, self.chunks[i]))

    def global_context(self) -> pl.DataFrame:
        # keyed by the exact chunk set: a context over a partial universe must never be reused
        path = self.dir / f"global_context_{_h([_h(c.to_list()) for c in self.chunks])}.parquet"
        if not path.exists():
            slim = pl.concat([
                pl.read_parquet(self._shortlist_path(i, c), columns=["s1_id", "target_id", features.CTX_SCORE])
                for i, c in enumerate(self.chunks)
            ])
            ctx = features.global_context(slim)
            tmp = path.with_suffix(".partial")
            ctx.write_parquet(tmp)
            tmp.replace(path)
        return pl.read_parquet(path)

    # ---------- pass 2 ----------
    def _feature_inputs(self):
        """Full-column pools, IDF fitted on THIS split's target pool (M3), dense embeddings + row maps."""
        if not hasattr(self, "_fi"):
            s1, targets = _match_pool(self.split, self._norm())
            name_idf = build_word_idf(targets, "name_core")
            addr_idf = build_word_idf(targets, "addr_content")
            model = _dense_model(self.cfg)
            emb = {src: embed_source(self.split, src, model=model) for src in io.SOURCES}
            rows = {src: io.load_source(self.split, src).select("entity_id").with_row_index("row")
                    for src in io.SOURCES}
            self._fi = (s1, targets, name_idf, addr_idf, emb, rows)
        return self._fi

    def _dense_cos(self, pairs: pl.DataFrame) -> np.ndarray:
        _, _, _, _, emb, rows = self._feature_inputs()
        q = pairs.select("s1_id").join(rows["S1"].rename({"entity_id": "s1_id"}), on="s1_id", how="left",
                                       maintain_order="left")["row"].to_numpy()
        out = np.zeros(pairs.height, dtype=np.float32)
        for src in io.TARGET_SOURCES:
            mask = (pairs["src"] == src).to_numpy()
            if not mask.any():
                continue
            t = pairs.filter(pl.Series(mask)).select("target_id").join(
                rows[src].rename({"entity_id": "target_id"}), on="target_id", how="left", maintain_order="left"
            )["row"].to_numpy()
            qi, idx = q[mask], np.flatnonzero(mask)
            for a in range(0, len(idx), 100_000):
                qa = np.asarray(emb["S1"][qi[a:a + 100_000]], dtype=np.float32)
                ta = np.asarray(emb[src][t[a:a + 100_000]], dtype=np.float32)
                out[idx[a:a + 100_000]] = np.einsum("ij,ij->i", qa, ta)
        return out

    def features(self, i: int, s1_filter: pl.Series | None = None, gctx: pl.DataFrame | None = None) -> pl.DataFrame:
        s1, targets, name_idf, addr_idf, _, _ = self._feature_inputs()
        pairs = self.shortlist(i)
        if s1_filter is not None:
            pairs = pairs.filter(pl.col("s1_id").is_in(s1_filter.implode()))
        if pairs.height == 0:
            return pairs
        pairs = pairs.drop(features.CTX_SCORE).with_columns(pl.Series("dense_cos", self._dense_cos(pairs)))
        queries = s1.filter(pl.col("entity_id").is_in(pairs["s1_id"].unique().implode()))
        df = features.build_local(pairs, queries, targets, name_idf, addr_idf).drop(features.CTX_SCORE)
        gctx = self.global_context() if gctx is None else gctx
        return df.join(gctx, on=["s1_id", "target_id"], how="left")

    def extra_features(self) -> list[str]:
        cols = pl.read_parquet_schema(self._shortlist_path(0, self.chunks[0]))
        base = [c for c in cols if c.startswith(("rank_", "score_")) or c in ("n_channels", "best_rank", "p", "frank")]
        return base + ["dense_cos"]

    def val_features(self) -> pl.DataFrame:
        """Pass-2 features for val S1 (the only S1 clean for both ranker p and dense scores), cached per chunk."""
        val = subset_ids(load_split(), "val")
        gctx = self.global_context()
        parts = []
        for i, chunk in enumerate(self.chunks):
            if not chunk.is_in(val.implode()).any():
                continue
            path = self.dir / f"features_v{FEATURE_VERSION}_{i:03d}.parquet"
            if not path.exists():
                t0 = time.perf_counter()
                df = self.features(i, val, gctx)
                df = df.select(["s1_id", "target_id", "src", "q_country"] + features.NUMERIC_FEATURES
                               + self.extra_features())
                tmp = path.with_suffix(".partial")
                df.write_parquet(tmp)
                tmp.replace(path)
                print(f"  features chunk {i}: {df.height:,} rows in {time.perf_counter() - t0:.0f}s", flush=True)
            parts.append(pl.read_parquet(path))
        return pl.concat(parts, how="diagonal_relaxed")

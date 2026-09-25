"""Unexplored area 1: a Ditto-style cross-encoder over serialized pairs (Li et al. 2020, VLDB).

Every existing name/address feature compares two strings through a fixed function — token overlap,
edit distance, a bi-encoder cosine. A cross-encoder reads both records in one pass and can attend
across them, which is why it beats bi-encoders on entity matching (Peeters & Bizer 2021). The
fine-tuned bi-encoder already bought +0.0013 in blocking, and this is the strictly stronger version
of that idea.

Two correctness rules this module enforces, because getting either wrong silently inflates OOF:

  * **Training pairs come from the train fold only** (and never the ranker's own sample). The Step-4
    model is cross-validated on val, so a cross-encoder that had seen val pairs would leak straight
    into the feature it produces.
  * **Hard negatives, not random ones.** The negatives are the top-ranked non-matches from the same
    shortlist the classifier will face. Random negatives are trivially separable and teach nothing
    about the pairs that actually decide the score.

Scoring the full shortlist is the expensive part, so `score_pairs` can be restricted to an uncertain
band: pairs the GBDT already calls confidently are left alone and the transformer runs only where the
decision is actually in doubt.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from er import io
from er.split import load_split, subset_ids

MODEL = "intfloat/multilingual-e5-small"  # MIT; mDeBERTa-v3-base is the larger drop-in
OUT = io.ARTIFACTS / "models" / "cross_encoder"
SCORES = io.ARTIFACTS / "match" / "ce"
CE_FEATURE = "ce_prob"


def serialize(df: pl.DataFrame, name_col: str, addr_col: str) -> pl.Expr:
    """Ditto's column-tagged serialization: the tags survive tokenization and mark field boundaries."""
    return pl.concat_str([
        pl.lit("[COL] name [VAL] "), pl.col(name_col).fill_null(""),
        pl.lit(" [COL] addr [VAL] "), pl.col(addr_col).fill_null(""),
    ])


def pair_texts(pairs: pl.DataFrame, split: str) -> tuple[list[str], list[str]]:
    """Raw (un-normalized) text on both sides: the transformer handles scripts and accents itself."""
    s1 = io.load_source(split, "S1").rename({"entity_id": "s1_id"})
    tg = pl.concat([io.load_source(split, s) for s in io.TARGET_SOURCES]).rename({"entity_id": "target_id"})
    left = pairs.select("s1_id").join(s1, on="s1_id", how="left", maintain_order="left")
    right = pairs.select("target_id").join(tg, on="target_id", how="left", maintain_order="left")
    ser = serialize(left, "business_name", "business_address")
    ser2 = serialize(right, "business_name", "business_address")
    return left.select(ser).to_series().to_list(), right.select(ser2).to_series().to_list()


def training_pairs(config_path: str, k: int = 10, n_s1: int = 60_000, neg_per_pos: int = 3,
                  seed: int = 0) -> pl.DataFrame:
    """Labelled shortlist pairs from train-fold S1 only, keeping the hardest negatives per record."""
    from er.blocking import ranker
    from er.match.dataset import Universe

    split = load_split()
    uni = Universe(config_path, "train", k, scope="full")
    uni.shortlists()
    excluded = subset_ids(split, "rtrain10").implode()  # the ranker's own sample: its `p` is overfit
    val = subset_ids(split, "val").implode()
    frames = []
    for i in range(len(uni.chunks)):
        sl = uni.shortlist(i).select("s1_id", "target_id", "src", "p")
        sl = sl.filter(~pl.col("s1_id").is_in(val) & ~pl.col("s1_id").is_in(excluded))
        if sl.height:
            frames.append(sl)
    pairs = pl.concat(frames)
    keep = pairs["s1_id"].unique().sample(n=min(n_s1, pairs["s1_id"].n_unique()), seed=seed)
    pairs = pairs.filter(pl.col("s1_id").is_in(keep.implode()))
    y = ranker.label(pairs, io.load_truth())
    pairs = pairs.with_columns(pl.Series("label", y))
    pos = pairs.filter(pl.col("label") == 1)
    neg = (pairs.filter(pl.col("label") == 0)
           .sort(["s1_id", "p"], descending=[False, True])
           .group_by("s1_id", maintain_order=True).head(neg_per_pos))
    return pl.concat([pos, neg]).sample(fraction=1.0, shuffle=True, seed=seed)


def finetune(name: str = "ce1", config_path: str = "configs/blocking/b_v9.toml", k: int = 10,
             n_s1: int = 60_000, batch: int = 128, epochs: int = 1, lr: float = 2e-5,
             model: str = MODEL, max_len: int = 96) -> Path:
    import torch
    from sentence_transformers import InputExample
    from sentence_transformers.cross_encoder import CrossEncoder
    from torch.utils.data import DataLoader

    t0 = time.perf_counter()
    pairs = training_pairs(config_path, k, n_s1)
    left, right = pair_texts(pairs, "train")
    labels = pairs["label"].to_list()
    print(f"{len(labels):,} training pairs, {sum(labels):,} positive", flush=True)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    ce = CrossEncoder(model, num_labels=1, max_length=max_len, device=device)
    examples = [InputExample(texts=[a, b], label=float(c)) for a, b, c in zip(left, right, labels)]
    loader = DataLoader(examples, shuffle=True, batch_size=batch)
    out = OUT / name
    ce.fit(train_dataloader=loader, epochs=epochs, warmup_steps=int(0.05 * len(loader) * epochs),
           optimizer_params={"lr": lr}, use_amp=(device == "cuda"), show_progress_bar=True)
    ce.save(str(out))
    meta = {"name": name, "model": model, "pairs": len(labels), "positives": int(sum(labels)),
            "n_s1": n_s1, "epochs": epochs, "lr": lr, "max_len": max_len, "device": device,
            "seconds": round(time.perf_counter() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"cross-encoder {name} in {meta['seconds']}s -> {out}", flush=True)
    return out


def score_pairs(pairs: pl.DataFrame, split: str, name: str = "ce1", batch: int = 256,
               band: tuple[float, float] | None = None, prob_col: str = "p") -> pl.DataFrame:
    """Cross-encoder probability per pair. `band` restricts scoring to an uncertain range of `prob_col`.

    Returns s1_id, target_id, ce_prob, ce_scored. Outside the band `ce_prob` is null and `ce_scored`
    is 0, so the GBDT can learn "not scored" as its own case instead of reading a fabricated value.
    """
    from sentence_transformers.cross_encoder import CrossEncoder

    todo = pairs
    if band is not None:
        lo, hi = band
        todo = pairs.filter((pl.col(prob_col) > lo) & (pl.col(prob_col) < hi))
    if todo.height == 0:
        return pairs.select("s1_id", "target_id").with_columns(
            pl.lit(None, dtype=pl.Float32).alias(CE_FEATURE), pl.lit(0, dtype=pl.Int8).alias("ce_scored"))
    ce = CrossEncoder(str(OUT / name))
    left, right = pair_texts(todo, split)
    scores = ce.predict(list(zip(left, right)), batch_size=batch, show_progress_bar=True,
                        apply_softmax=False, convert_to_numpy=True)
    scores = 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype=np.float64)))  # single logit -> probability
    scored = todo.select("s1_id", "target_id").with_columns(
        pl.Series(CE_FEATURE, scores.astype(np.float32)), pl.lit(1, dtype=pl.Int8).alias("ce_scored"))
    return pairs.select("s1_id", "target_id").join(scored, on=["s1_id", "target_id"], how="left") \
        .with_columns(pl.col("ce_scored").fill_null(0).cast(pl.Int8))


def score_cached(pairs: pl.DataFrame, split: str, name: str = "ce1", **kw) -> pl.DataFrame:
    """Scoring is the expensive half; cache it per (split, model) so re-fits reuse it."""
    path = SCORES / f"{name}_{split}.parquet"
    if path.exists():
        have = pl.read_parquet(path)
        missing = pairs.join(have, on=["s1_id", "target_id"], how="anti")
        if missing.height == 0:
            return pairs.select("s1_id", "target_id").join(have, on=["s1_id", "target_id"], how="left")
        have = pl.concat([have, score_pairs(missing, split, name, **kw)], how="diagonal_relaxed")
    else:
        have = score_pairs(pairs, split, name, **kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    have.write_parquet(tmp)
    tmp.replace(path)
    return pairs.select("s1_id", "target_id").join(have, on=["s1_id", "target_id"], how="left")

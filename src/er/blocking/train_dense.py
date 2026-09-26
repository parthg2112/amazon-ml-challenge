"""Contrastive fine-tune of the dense retriever on train-fold pairs (never val, never the ranker's sample)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl

from er import io
from er.blocking.channels.dense import MODEL, _texts
from er.split import load_split, subset_ids

OUT = io.ARTIFACTS / "models"


def training_links(n_pairs: int, ranker_subset: str, seed: int = 0) -> pl.DataFrame:
    """The (s1_id, target_id) links the fine-tune sees: deterministic, so later stages can exclude those S1."""
    split = load_split()
    excluded = subset_ids(split, ranker_subset).implode()
    ids = subset_ids(split, "train").filter(~subset_ids(split, "train").is_in(excluded))
    links = io.truth_long(io.load_truth().filter(pl.col("s1_id").is_in(ids.implode())))
    links = links.sample(fraction=1.0, shuffle=True, seed=seed).unique("s1_id", keep="first", maintain_order=True)
    return links.head(n_pairs)


def training_pairs(n_pairs: int, ranker_subset: str, seed: int = 0) -> pl.DataFrame:
    """One random true target per S1 (MNRL treats other in-batch rows as negatives, so no S1 may repeat)."""
    links = training_links(n_pairs, ranker_subset, seed)
    s1 = io.load_source("train", "S1").rename({"entity_id": "s1_id"})
    tg = pl.concat([io.load_source("train", s) for s in io.TARGET_SOURCES]).rename({"entity_id": "target_id"})
    q = links.select("s1_id").join(s1, on="s1_id", how="left", maintain_order="left")
    t = links.select("target_id").join(tg, on="target_id", how="left", maintain_order="left")
    return pl.DataFrame({"anchor": _texts(q), "positive": _texts(t)})


def finetune(name: str, n_pairs: int = 600_000, batch: int = 512, epochs: int = 1, lr: float = 5e-5,
             ranker_subset: str = "rtrain10") -> Path:
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, losses
    from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments
    from sentence_transformers.trainer import SentenceTransformerTrainer

    out = OUT / name
    t0 = time.perf_counter()
    pairs = training_pairs(n_pairs, ranker_subset)
    print(f"{pairs.height:,} pairs in {time.perf_counter() - t0:.0f}s", flush=True)
    model = SentenceTransformer(MODEL, device="cuda")
    model.max_seq_length = 64
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out / "ckpt"), num_train_epochs=epochs, per_device_train_batch_size=batch,
        learning_rate=lr, warmup_ratio=0.05, fp16=True, batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=100, save_strategy="no", report_to=[], seed=0, dataloader_num_workers=0,
    )
    loss = losses.MultipleNegativesRankingLoss(model, scale=20.0)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=Dataset.from_polars(pairs), loss=loss)
    trainer.train()
    model.save(str(out))
    print(f"fine-tuned {name} in {time.perf_counter() - t0:.0f}s -> {out}", flush=True)
    del model, trainer
    torch.cuda.empty_cache()
    return out

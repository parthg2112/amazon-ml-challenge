"""Cross-encoder pair model (review item 1): raw `name | address` text of both records, jointly encoded.

Trained on non-val train S1 (rtrain10 channel caches: hard negatives from blocking), so its val scores are
clean features for the Step 4 model. Init from the fine-tuned bi-encoder (e5s_ft1), mean pooling + linear head.
"""

from __future__ import annotations

import gc
import time
import tomllib
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer

from er import io
from er.blocking.runner import CHUNK, Context, _chunks, precut
from er.split import load_split, subset_ids

MAX_LEN = 96


def training_pairs(config_path: str, n_chunks: int, countries: list[str] | None = None, hard: int = 6,
                   rand: int = 2, seed: int = 0) -> pl.DataFrame:
    """All true links + per S1 the `hard` best-ranked negatives and `rand` random ones, from cached channels."""
    cfg = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    ctx = Context(cfg, "train")
    ids = subset_ids(load_split(), cfg["ranker"].get("train_subset", "rtrain"))
    truth = io.load_truth().explode("truth", empty_as_null=True).drop_nulls("truth").select(
        "s1_id", pl.col("truth").alias("target_id"), pl.lit(1).alias("y"))
    out = []
    for i, chunk in enumerate(_chunks(ids.sort(), cfg.get("chunk", CHUNK))):
        if i >= n_chunks:
            break
        cands, queries = ctx.collect(chunk, "ce-pairs", {})
        c = precut(cands, cfg).group_by("s1_id", "target_id").agg(pl.col("rank").min().alias("best_rank"))
        c = c.join(queries.select(pl.col("entity_id").alias("s1_id"), "country"), on="s1_id")
        if countries:
            c = c.filter(pl.col("country").is_in(countries))
        c = c.join(truth, on=["s1_id", "target_id"], how="left").with_columns(pl.col("y").fill_null(0))
        neg = c.filter(pl.col("y") == 0).with_columns(
            pl.col("best_rank").rank("ordinal").over("s1_id").alias("r"),
            pl.int_range(pl.len()).shuffle(seed=seed + i).over("s1_id").alias("rr"))
        neg = neg.filter((pl.col("r") <= hard) | (pl.col("rr") < rand))
        # positives the blocker missed are still true links: keep them so the model learns what a match is
        pos = truth.join(c.select("s1_id", "country").unique(), on="s1_id")
        out.append(pl.concat([pos.select("s1_id", "target_id", "y", "country"),
                              neg.select("s1_id", "target_id", "y", "country")]))
        del cands, queries, c, neg
        gc.collect()
    return pl.concat(out).unique(["s1_id", "target_id"])


def _texts(split: str) -> pl.DataFrame:
    return pl.concat([io.load_source(split, s) for s in io.SOURCES]).select(
        "entity_id", (pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null(""))
        .alias("text"))


def attach_text(pairs: pl.DataFrame, split: str, texts: pl.DataFrame | None = None) -> pl.DataFrame:
    t = _texts(split) if texts is None else texts
    return (pairs.join(t.rename({"entity_id": "s1_id", "text": "a"}), on="s1_id", how="left", maintain_order="left")
            .join(t.rename({"entity_id": "target_id", "text": "b"}), on="target_id", how="left",
                  maintain_order="left"))


class CrossEncoder(torch.nn.Module):
    def __init__(self, init: str):
        super().__init__()
        self.enc = AutoModel.from_pretrained(init)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1)).squeeze(-1)


def _batches(a: list[str], b: list[str], tok, bs: int):
    for s in range(0, len(a), bs):
        yield s, tok(a[s:s + bs], b[s:s + bs], truncation=True, max_length=MAX_LEN, padding=True,
                     return_tensors="pt")


def train(pairs: pl.DataFrame, out_dir: Path, init: str = "artifacts/models/e5s_ft1", epochs: int = 1,
          bs: int = 128, lr: float = 3e-5, seed: int = 0) -> dict:
    """pairs: a, b, y. Saves encoder + head to out_dir."""
    torch.manual_seed(seed)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(init)
    model = CrossEncoder(init).to(dev)
    df = pairs.sample(fraction=1.0, shuffle=True, seed=seed)
    a, b, y = df["a"].to_list(), df["b"].to_list(), torch.tensor(df["y"].to_numpy(), dtype=torch.float32)
    steps = epochs * ((len(a) + bs - 1) // bs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.06,
                                                anneal_strategy="linear")
    scaler = torch.amp.GradScaler()
    lossf = torch.nn.BCEWithLogitsLoss()
    t0, step = time.perf_counter(), 0
    model.train()
    for ep in range(epochs):
        for s, enc in _batches(a, b, tok, bs):
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(**enc)
            loss = lossf(logit.float(), y[s:s + bs].to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 500 == 0:
                print(f"  ce step {step}/{steps} loss {loss.item():.4f} {time.perf_counter() - t0:.0f}s", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.enc.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    torch.save(model.head.state_dict(), out_dir / "head.pt")
    return {"pairs": len(a), "steps": steps, "train_s": round(time.perf_counter() - t0)}


def load(model_dir: Path):
    model = CrossEncoder(str(model_dir))
    model.head.load_state_dict(torch.load(model_dir / "head.pt"))
    return model.to("cuda").eval().half(), AutoTokenizer.from_pretrained(str(model_dir))


@torch.inference_mode()
def score(pairs: pl.DataFrame, model, tok, bs: int = 1024) -> np.ndarray:
    """pairs: a, b. Returns logits (float32). Sorted by length internally to cut padding."""
    a, b = pairs["a"].to_list(), pairs["b"].to_list()
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)], kind="stable")
    a, b = [a[i] for i in order], [b[i] for i in order]
    out = np.empty(len(a), dtype=np.float32)
    for s, enc in _batches(a, b, tok, bs):
        enc = {k: v.to("cuda") for k, v in enc.items()}
        out[s:s + bs] = model(**enc).float().cpu().numpy()
    res = np.empty_like(out)
    res[order] = out
    return res

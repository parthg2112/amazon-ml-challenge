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
                   rand: int = 2, seed: int = 0, chunks: list[pl.Series] | None = None,
                   exclude: pl.Series | None = None) -> pl.DataFrame:
    """All true links + per S1 the `hard` best-ranked negatives and `rand` random ones, from cached channels.
    Default chunks: the ranker's subset. `chunks` (e.g. full-scope Universe chunks, whose channels are cached)
    widens the pool; `exclude` drops S1 that must stay unseen (a scorer's OOF S1)."""
    cfg = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    ctx = Context(cfg, "train")
    if chunks is None:
        ids = subset_ids(load_split(), cfg["ranker"].get("train_subset", "rtrain"))
        chunks = list(_chunks(ids.sort(), cfg.get("chunk", CHUNK)))
    truth = io.load_truth().explode("truth", empty_as_null=True).drop_nulls("truth").select(
        "s1_id", pl.col("truth").alias("target_id"), pl.lit(1).alias("y"))
    out = []
    for i, chunk in enumerate(chunks):
        if i >= n_chunks:
            break
        cands, queries = ctx.collect(chunk, "ce-pairs", {})
        c = precut(cands, cfg).group_by("s1_id", "target_id").agg(pl.col("rank").min().alias("best_rank"))
        c = c.join(queries.select(pl.col("entity_id").alias("s1_id"), "country"), on="s1_id")
        if exclude is not None:
            c = c.filter(~pl.col("s1_id").is_in(exclude.implode()))
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
          bs: int = 128, lr: float = 3e-5, seed: int = 0, reserve_gb: float = 0.0) -> dict:
    """pairs: a, b, y. Saves encoder + head to out_dir. `reserve_gb` holds disk for the final save (a growing
    pagefile filled the disk during two multi-hour runs, which then failed at save)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hold = out_dir / ".reserve"
    if reserve_gb:
        with hold.open("wb") as f:
            f.truncate(int(reserve_gb * 1e9))
            f.seek(int(reserve_gb * 1e9) - 1)
            f.write(b"\0")
    torch.manual_seed(seed)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(init)
    model = CrossEncoder(init)
    if (Path(init) / "head.pt").exists():  # continuing from a trained CE (domain adaptation): keep its head
        model.head.load_state_dict(torch.load(Path(init) / "head.pt"))
    model = model.to(dev)
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
    hold.unlink(missing_ok=True)
    model.enc.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    torch.save(model.head.state_dict(), out_dir / "head.pt")
    return {"pairs": len(a), "steps": steps, "train_s": round(time.perf_counter() - t0)}


def pseudo_pairs(name: str, parts: int, n_s1: int | None = None) -> tuple[pl.DataFrame, int]:
    """Pseudo-labels for test S1 whose country label has no training labels (open set, no label named): the
    scorer's confident winners (p >= 0.95) vs confident rejects (p <= 0.02, top 4) in the same shortlists."""
    from er.decide import exclusivity

    norm = pl.col("country").str.strip_chars().str.to_lowercase()
    seen = io.load_source("train", "S1").select(norm.unique())["country"]
    q = io.load_source("test", "S1").filter(~norm.is_in(seen.implode())).select(pl.col("entity_id").alias("s1_id"))
    t = pl.read_parquet(io.ARTIFACTS / "match" / f"test_scored_{name}_p{parts}.parquet").join(q, on="s1_id", how="semi")
    win = exclusivity.resolve_max_probability(t.select("s1_id", "target_id", "src", "p"))
    pos = win.filter(pl.col("p") >= 0.95).select("s1_id", "target_id", pl.lit(1).alias("y"))
    s1s = pos.select("s1_id").unique().sort("s1_id")
    if n_s1 is not None:
        s1s = s1s.sample(n=min(n_s1, s1s.height), seed=0)
    neg = (t.join(s1s, on="s1_id", how="semi").filter(pl.col("p") <= 0.02)
           .with_columns(pl.col("p").rank("ordinal", descending=True).over("s1_id").alias("r")).filter(pl.col("r") <= 4)
           .select("s1_id", "target_id", pl.lit(0).alias("y")))
    return pl.concat([pos.join(s1s, on="s1_id", how="semi"), neg]), q.height


def adapt(name: str, parts: int, base: str, out: str, n_s1: int = 80_000,
          config_path: str = "configs/blocking/b_v9.toml") -> dict:
    """Domain adaptation for unlabeled test countries: fine-tune `base` on their pseudo-labels mixed 1:1 with labeled
    pairs of S1 the scorer's OOF never saw. India rehearsal: +0.0029."""
    from er.match.dataset import Universe

    pseudo, n_unlab = pseudo_pairs(name, parts, n_s1)
    uni = Universe(config_path, "train", 10, "full")
    oof = pl.read_parquet(io.ARTIFACTS / "match" / f"oof_{name}.parquet", columns=["s1_id"])["s1_id"].unique()
    lab = training_pairs(config_path, 8, chunks=uni.chunks[18:26], exclude=oof)  # rtrain10 chunks of the full scope
    lab = lab.sample(n=min(lab.height, pseudo.height), seed=0)
    mix = pl.concat([attach_text(pseudo, "test").select("a", "b", "y"),
                     attach_text(lab.select("s1_id", "target_id", "y"), "train").select("a", "b", "y")])
    res = train(mix, io.ARTIFACTS / "models" / out, init=str(io.ARTIFACTS / "models" / base), lr=1e-5)
    return {"unlabeled_s1": n_unlab, "pseudo_pairs": pseudo.height, "labeled_pairs": lab.height, **res}


def train_mix(name: str, parts: int, out: str, n_labeled: int = 4_200_000, init: str = "intfloat/multilingual-e5-base",
              lr: float = 2e-5, config_path: str = "configs/blocking/b_v9.toml") -> dict:
    """CE4F: a CE from scratch on `n_labeled` pairs from every non-val full-scope chunk (never the scorer's OOF S1)
    plus all pseudo-labels of unlabeled test countries. 442k val 0.9900 with raw features (CE3-adapted: 0.9889)."""
    from er.match.dataset import Universe

    pseudo, n_unlab = pseudo_pairs(name, parts)
    uni = Universe(config_path, "train", 10, "full")
    oof = pl.read_parquet(io.ARTIFACTS / "match" / f"oof_{name}.parquet", columns=["s1_id"])["s1_id"].unique()
    chunks = uni.chunks[18:]  # rtrain10 + train-fold chunks; val chunks come first
    lab = training_pairs(config_path, len(chunks), chunks=chunks, exclude=oof)
    lab = lab.sample(n=min(lab.height, n_labeled), seed=0)
    mix = pl.concat([attach_text(lab.select("s1_id", "target_id", "y"), "train").select("a", "b", "y"),
                     attach_text(pseudo, "test").select("a", "b", "y")]).with_columns(pl.col("y").cast(pl.Int32))
    del lab
    res = train(mix, io.ARTIFACTS / "models" / out, init=init, lr=lr, reserve_gb=1.3)
    return {"unlabeled_s1": n_unlab, "pseudo_pairs": pseudo.height, "labeled_pairs": mix.height - pseudo.height, **res}


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

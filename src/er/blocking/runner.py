"""Run a blocking config (configs/blocking/*.toml): channels → fusion/ranker → metrics or test outputs.

Queries are processed in chunks so validation (442k S1) and test (1.73M S1) stay memory-bounded.
"""

from __future__ import annotations

import json
import time
import tomllib
from collections import defaultdict
from pathlib import Path

import polars as pl
import psutil

from er import io
from er.blocking.channels.exact import exact_channel, house_token_keys, token_df
from er.blocking.channels.sparse import FieldSpec, sparse_channel
from er.blocking.fuse import add_ranks, fuse
from er.metrics import row_table, summarize
from er.normalize.pipeline import NormConfig, load_norm
from er.split import load_split, subset_ids

EXPERIMENTS = io.REPORTS / "experiments"
CHUNK = 50_000


def _cardinality_bucket() -> pl.Expr:
    n = pl.col("truth").list.len()
    return (pl.when(n == 0).then(pl.lit("0")).when(n == 1).then(pl.lit("1")).when(n <= 3).then(pl.lit("2-3"))
            .when(n <= 5).then(pl.lit("4-5")).otherwise(pl.lit("6+")))


def run_channel(ch: dict, queries: pl.DataFrame, targets: dict[str, pl.DataFrame], variant: str,
                split: str) -> tuple[pl.DataFrame, dict]:
    kind = ch["type"]
    stats: dict = {}
    if kind == "exact":
        parts = []
        for src, t in targets.items():
            p, stats[src] = exact_channel(queries, t, ch["query_col"], ch.get("target_col", ch["query_col"]),
                                          ch.get("max_block", 200))
            parts.append(p)
        df = pl.concat(parts)
    elif kind == "house_token":
        parts = []
        for src, t in targets.items():
            dft = token_df(t)
            qk = house_token_keys(queries, dft, ch.get("rarest", 2)).rename({"house_tok": "k"})
            tk = house_token_keys(t, dft, None).rename({"house_tok": "k"})
            p, stats[src] = exact_channel(qk, tk, "k", "k", ch.get("max_block", 200))
            parts.append(p)
        df = pl.concat(parts)
    elif kind == "sparse":
        fields = [FieldSpec(col=f["col"], analyzer=f.get("analyzer", "char_wb"),
                            ngram=tuple(f.get("ngram", (3, 3))), weight=f.get("weight", 1.0))
                  for f in ch["fields"]]
        df = sparse_channel(queries, targets, fields, ch.get("k", 50), ch.get("max_df", 0.01),
                            ch.get("threshold", 0.0), variant if split == "train" else f"{split}-{variant}",
                            ch.get("partition"))
    elif kind == "dense":
        from er.blocking.channels.dense import dense_channel

        df = dense_channel(queries, targets, ch.get("k", 50), queries["row"].to_numpy(), split)
    else:
        raise ValueError(kind)
    return add_ranks(df).with_columns(pl.lit(ch["id"]).alias("channel")), stats


class Context:
    """Normalized S1 + target pools for one data split, loaded once."""

    def __init__(self, cfg: dict, split: str):
        self.cfg, self.split = cfg, split
        from er.blocking.ranker import PAIR_COLS

        self.norm = NormConfig(**cfg.get("norm", {}))
        cols = {"entity_id", "country", *PAIR_COLS}
        for ch in cfg["channels"]:
            cols |= {ch.get("query_col"), ch.get("target_col"), ch.get("partition")}
            cols |= {f["col"] for f in ch.get("fields", [])}
        cols = [c for c in cols if c]
        self.s1 = load_norm(split, "S1", self.norm).select(cols).with_row_index("row")
        self.targets = {src: load_norm(split, src, self.norm).select(cols) for src in io.TARGET_SOURCES}
        self.all_targets = pl.concat([t.select("entity_id", *PAIR_COLS) for t in self.targets.values()])
        self.n_targets = self.all_targets.height

    def collect(self, ids: pl.Series, label: str, timings: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
        queries = self.s1.filter(pl.col("entity_id").is_in(ids.implode()))
        frames = []
        for ch in self.cfg["channels"]:
            t0 = time.perf_counter()
            df, _ = run_channel(ch, queries, self.targets, self.norm.variant, self.split)
            timings[ch["id"]] = timings.get(ch["id"], 0.0) + time.perf_counter() - t0
            frames.append(df)
        cands = pl.concat(frames, how="diagonal_relaxed").select(
            "s1_id", "target_id", "src", "channel", "score", "rank")
        print(f"  [{label}] {queries.height:,} queries → {cands.height:,} channel pairs")
        return cands, queries


def train_ranker(cfg: dict):
    """Fit the stage-2 ranker on the train-fold rtrain sample against the TRAIN target pool."""
    from er.blocking import ranker

    ctx = Context(cfg, "train")
    channels = [ch["id"] for ch in cfg["channels"]]
    feats = ranker.feature_names(channels)
    ids = subset_ids(load_split(), cfg["ranker"].get("train_subset", "rtrain"))
    cands, queries = ctx.collect(ids, "ranker-train", {})
    df = ranker.build(cands, channels, queries, ctx.all_targets)
    y = ranker.label(df, io.load_truth().filter(pl.col("s1_id").is_in(ids.implode())))
    model = ranker.fit(df, y, feats)
    return model, feats, channels, {"train_pairs": len(y), "pos_rate": float(y.mean())}


def _chunks(ids: pl.Series, size: int):
    for i in range(0, ids.len(), size):
        yield ids.slice(i, size)


def run(config_path: str, subset: str | None = None) -> dict:
    cfg = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    subset = subset or cfg.get("subset", "dev")
    ctx = Context(cfg, "train")
    run_id = f"{cfg['name']}__{ctx.norm.variant}__{subset}"
    out_dir = EXPERIMENTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    budgets = cfg.get("budgets", [10, 20, 30, 50])
    weights = {ch["id"]: ch.get("weight", 1.0) for ch in cfg["channels"]}
    use_ranker = cfg.get("ranker", {}).get("enabled", False)
    t_start = time.perf_counter()
    ranker_meta: dict = {}
    if use_ranker:
        from er.blocking import ranker

        model, feats, channels, ranker_meta = train_ranker(cfg)
        ranker_meta["train_s"] = round(time.perf_counter() - t_start, 1)

    ids = subset_ids(load_split(), subset)
    truth_all = io.load_truth()
    rows: dict[str, list[pl.DataFrame]] = defaultdict(list)
    unique = defaultdict(int)
    timings: dict = {}
    for chunk in _chunks(ids, cfg.get("chunk", CHUNK)):
        cands, queries = ctx.collect(chunk, subset, timings)
        truth = truth_all.filter(pl.col("s1_id").is_in(chunk.implode()))
        fused = fuse(cands, weights)
        for k in budgets:
            rows[f"rrf@{k}"].append(row_table(truth, fused.filter(pl.col("frank") <= k)))
        rows["union"].append(row_table(truth, fused))
        for ch in weights:
            rows[f"channel:{ch}"].append(row_table(truth, cands.filter(pl.col("channel") == ch)))
        tl = truth.explode("truth", empty_as_null=True).drop_nulls("truth").rename({"truth": "target_id"})
        solo = tl.join(fused, on=["s1_id", "target_id"]).filter(pl.col("channels").list.len() == 1)
        for ch, n in solo.group_by(pl.col("channels").list.first()).len().iter_rows():
            unique[ch] += n
        if use_ranker:
            ranked = ranker.rank_by_model(ranker.build(cands, channels, queries, ctx.all_targets), model, feats)
            for k in cfg["ranker"].get("budgets", budgets):
                rows[f"ranker@{k}"].append(row_table(truth, ranked.filter(pl.col("frank") <= k)))

    truth = truth_all.filter(pl.col("s1_id").is_in(ids.implode()))
    slices = truth.select("s1_id", _cardinality_bucket().alias("cardinality")).join(
        ctx.s1.select(pl.col("entity_id").alias("s1_id"), "country"), on="s1_id")
    total_links = int(truth["truth"].list.len().sum())

    def agg(key: str, with_slices: bool = False) -> dict:
        return summarize(pl.concat(rows[key]), ctx.n_targets, slices if with_slices else None)

    metrics = {
        "budgets": {str(k): agg(f"rrf@{k}", k == max(budgets)) for k in budgets},
        "union": agg("union", True),
        "channels": {
            ch: {**agg(f"channel:{ch}"), "unique_links": unique[ch],
                 "unique_link_share": unique[ch] / total_links if total_links else 0.0}
            for ch in weights
        },
    }
    if use_ranker:
        rb = cfg["ranker"].get("budgets", budgets)
        metrics["ranked"] = {str(k): agg(f"ranker@{k}", k in (10, max(rb))) for k in rb}
        metrics["ranker"] = ranker_meta
    metrics.update({
        "run_id": run_id, "config": cfg, "subset": subset, "norm_variant": ctx.norm.variant,
        "timings_s": {k: round(v, 1) for k, v in timings.items()},
        "total_s": round(time.perf_counter() - t_start, 1),
        "peak_rss_gb": round(getattr(psutil.Process().memory_info(), "peak_wset", 0) / 1e9, 1),
    })
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def infer(config_path: str, k: int, out_dir: Path) -> dict:
    """Test-set blocking: ranker trained on train, applied to test S1 vs test S2/S3 → candidate_pairs.tsv."""
    from er.blocking import ranker

    cfg = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    model, feats, channels, meta = train_ranker(cfg)
    ctx = Context(cfg, "test")
    ids = ctx.s1["entity_id"]
    parts, timings = [], {}
    for chunk in _chunks(ids, cfg.get("chunk", CHUNK)):
        cands, queries = ctx.collect(chunk, "test", timings)
        ranked = ranker.rank_by_model(ranker.build(cands, channels, queries, ctx.all_targets), model, feats)
        parts.append(ranked.filter(pl.col("frank") <= k).select("s1_id", "target_id", "p"))
    sel = pl.concat(parts)
    out_dir.mkdir(parents=True, exist_ok=True)
    sel.write_parquet(io.ARTIFACTS / "candidates" / f"test_{Path(config_path).stem}_k{k}.parquet", mkdir=True)
    lists = sel.sort("s1_id", "p", descending=[False, True]).group_by("s1_id", maintain_order=True).agg(
        pl.col("target_id").str.join(",").alias("candidate_entity_ids"))
    full = ids.to_frame("source1_entity_id").join(
        lists.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left"
    ).with_columns(pl.col("candidate_entity_ids").fill_null(""))
    full.write_csv(out_dir / "candidate_pairs.tsv", separator="\t", quote_style="never")
    # Placeholder until Step 4: all-empty matches (valid format, every S1 present).
    full.select("source1_entity_id", pl.lit("").alias("matched_entity_ids")).write_csv(
        out_dir / "matching_results.tsv", separator="\t", quote_style="never")
    stats = {
        "test_s1": ids.len(), "candidate_pairs": sel.height, "per_source_k": k,
        "s1_without_candidates": int((full["candidate_entity_ids"] == "").sum()),
        "by_country": ctx.s1.select(pl.col("entity_id").alias("s1_id"), "country").join(
            sel.group_by("s1_id").len(), on="s1_id", how="left").group_by("country").agg(
            pl.col("len").fill_null(0).mean().alias("mean_candidates")).to_dicts(),
        "timings_s": {kk: round(v, 1) for kk, v in timings.items()},
        "total_s": round(time.perf_counter() - t0, 1), "ranker": meta,
    }
    (io.REPORTS / "test_blocking.json").write_text(json.dumps(stats, indent=2, default=str))
    return stats


def _match_pool(split: str, norm_cfg: NormConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Full-column normalized S1/target pools for match features — `Context` only loads the blocking
    ranker's narrower `PAIR_COLS`, which is missing several columns `er.match.features` needs."""
    from er.match.features import JOIN_COLS

    cols = ["entity_id", *JOIN_COLS]
    s1 = load_norm(split, "S1", norm_cfg).select(cols)
    targets = pl.concat([load_norm(split, src, norm_cfg).select(cols) for src in io.TARGET_SOURCES])
    return s1, targets


def collect_match_features(config_path: str, subset: str, k: int) -> tuple[pl.DataFrame, list[str]]:
    """Step 4 input: blocking ranker's per-S1 shortlist, with channel provenance and match features.

    Chunked like `run`/`infer` so this scales to the full train fold. Returns (features_df, extra_feature_cols)
    where extra_feature_cols are the blocking-stage columns (channel provenance + ranker probability) to
    add to `er.match.features.NUMERIC_FEATURES`.
    """
    from er.blocking import ranker
    from er.match import features
    from er.match.idf import build_word_idf

    cfg = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    model, feats, channels, _meta = train_ranker(cfg)
    ctx = Context(cfg, "train")
    match_s1, match_targets = _match_pool("train", ctx.norm)
    name_idf = build_word_idf(match_targets, "name_core")
    addr_idf = build_word_idf(match_targets, "addr_content")

    ids = subset_ids(load_split(), subset)
    parts, timings = [], {}
    extra: list[str] | None = None
    for chunk in _chunks(ids, cfg.get("chunk", CHUNK)):
        cands, queries = ctx.collect(chunk, subset, timings)
        ranked = ranker.rank_by_model(ranker.build(cands, channels, queries, ctx.all_targets), model, feats)
        keep = ranked.filter(pl.col("frank") <= k).select("s1_id", "target_id", "src", "p", "frank")
        filtered = cands.join(keep.select("s1_id", "target_id"), on=["s1_id", "target_id"], how="semi")
        pairs = ranker.provenance(filtered, channels).join(keep, on=["s1_id", "target_id", "src"], how="left")
        if extra is None:
            # Computed from the blocking-stage frame, before features.build() adds its own rank_by_*/
            # rank_within_s1_src columns — those would otherwise collide with this prefix match.
            extra = [c for c in pairs.columns if c.startswith(("rank_", "score_"))
                     or c in ("n_channels", "best_rank", "p")]
        match_queries = match_s1.filter(pl.col("entity_id").is_in(chunk.implode()))
        parts.append(features.build(pairs, match_queries, match_targets, name_idf, addr_idf))
    df = pl.concat(parts, how="diagonal_relaxed")
    return df, extra or []


def leaderboard() -> str:
    rows = []
    for p in sorted(EXPERIMENTS.glob("*/metrics.json")):
        m = json.loads(p.read_text())
        for k, b in m["budgets"].items():
            rows.append((m["run_id"] + " [rrf]", int(k), b))
        for k, b in m.get("ranked", {}).items():
            rows.append((m["run_id"] + " [ranker]", int(k), b))
        rows.append((m["run_id"], 10**9, m["union"]))
    lines = [
        "| run | K/src | oracle F0.5 | pair recall | full cov | mean cand |",
        "|---|---|---|---|---|---|",
    ]
    for run_id, k, b in sorted(rows, key=lambda r: (r[0], r[1])):
        lines.append(
            f"| {run_id} | {'union' if k == 10**9 else k} | {b['oracle_ceiling_f05']:.4f} | "
            f"{b['pair_recall']:.4f} | {b['full_coverage']:.4f} | {b['mean_candidates']:.1f} |"
        )
    text = ("# Blocking leaderboard\n\nGenerated by `er leaderboard`. dev = 10% of validation (44.5k S1); "
            "val = full 20% holdout.\n\n" + "\n".join(lines) + "\n")
    (io.REPORTS / "leaderboard.md").write_text(text, encoding="utf-8")
    return text

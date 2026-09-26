"""Step 4/5 v2: full-universe context (M1), fold-mean prediction (M2), per-pool IDF (M3), frozen-policy
country transfer check (M4), val-only training (M5). See docs/roadmap_steps4_6.md."""

from __future__ import annotations

import gc
import json
import pickle
import time

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from er import io
from er.blocking import ranker
from er.blocking.runner import release_caches
from er.decide import exact_f05, exclusivity, expected_f05, linkcount, pipeline, threshold
from er.match import features, train
from er.match.dataset import Universe
from er.metrics import macro_f05
from er.split import load_split, subset_ids

MODELS = io.ARTIFACTS / "models" / "match"


def _policy_predict(scored: pl.DataFrame, policy: dict, n_s1: int | None = None) -> pl.DataFrame:
    """Apply a frozen policy. `n_s1` (the scored universe, including S1 with no candidates) is needed
    only by the prior-matched rule, which sets its cutoff from the predicted link rate."""
    excl = (exclusivity.resolve_scarcity_weighted(scored) if policy["exclusivity_winner"] == "scarcity_weighted"
            else exclusivity.resolve_max_probability(scored))
    rule = policy["decision_winner"]
    if rule == "tuned_global_threshold":
        return threshold.predict(excl, policy["threshold"])
    if rule == "exact_expected_f05":
        return exact_f05.select(excl).select("s1_id", "target_id")
    if rule in ("prior_tilted_exact_f05", "capped_tuned_threshold", "expected_count_topk",
                "prior_matched_threshold"):
        prior = linkcount.load_prior()
        if rule == "prior_tilted_exact_f05":
            return linkcount.select_tilted(excl, prior).select("s1_id", "target_id")
        if rule == "capped_tuned_threshold":
            return threshold.predict(linkcount.apply_caps(excl, prior), policy["threshold"])
        if rule == "expected_count_topk":
            return linkcount.select_by_expected_count(excl, prior, blend=0.25).select("s1_id", "target_id")
        n = excl["s1_id"].n_unique() if n_s1 is None else n_s1
        return threshold.predict(excl, linkcount.prior_matched_threshold(excl, n, prior.mean_total))
    return expected_f05.select(excl).select("s1_id", "target_id")


def _fit_block(df: pl.DataFrame, y: np.ndarray, feats: list[str], truth: pl.DataFrame,
               backend: str = "xgb_gpu") -> dict:
    """CV + calibration + Step 5 policy on one set of rows: everything a frozen deployment needs."""
    oof, models, imp = train.cv_train(df, y, "cv_fold", feats, features.CATEGORICAL_FEATURES, backend=backend)
    calibrated, calib, iso = train.maybe_calibrate(y, oof)
    oof_df = df.select("s1_id", "target_id", "src", "q_country").with_columns(pl.Series("p", calibrated))
    decision = pipeline.run(oof_df, truth)
    policy = {"decision_winner": decision["decision_winner"], "exclusivity_winner": decision["exclusivity_winner"],
              "threshold": decision["global_threshold"]["best"]}
    return {"models": models, "iso": iso, "oof": oof_df, "importance": imp, "calibration": calib,
            "decision": decision, "policy": policy}


def _fit_block_stream(paths: list, feats: list[str], truth: pl.DataFrame, backend: str = "xgb_gpu",
                      macro_weight: bool = False) -> dict:
    """Same contract as `_fit_block` for training sets too big for RAM: per fold, read only that fold's
    rows from the chunk caches (features + label + cv_fold live in the caches), fit, predict, free.
    Peak RAM = one fold's matrix instead of the whole frame. REL features are recomputed per fold —
    safe because cv_fold is per-S1, so every S1's shortlist is complete within a fold.
    `macro_weight`: pairs weighted 1/max(true links in shortlist, 1), so each S1 counts ~equally (macro F)."""
    import xgboost as xgb

    def weights(df: pl.DataFrame) -> np.ndarray | None:
        if not macro_weight:
            return None
        return df.select(1 / pl.col("y").sum().over("s1_id").clip(lower_bound=1))[:, 0].to_numpy()

    folds = sorted({f for p in paths for f in pl.read_parquet(p, columns=["cv_fold"])["cv_fold"].unique().to_list()})
    oof_parts, models, importances = [], [], []
    for f in folds:
        tr_parts, va_parts = [], []
        for p in paths:
            chunk = pl.read_parquet(p)
            tr_parts.append(chunk.filter(pl.col("cv_fold") != f))
            va_parts.append(chunk.filter(pl.col("cv_fold") == f))
        tr = features.add_relative(pl.concat(tr_parts))
        Xtr = train.matrix(tr, feats)
        ytr, wtr = tr["y"].to_numpy(), weights(tr)
        del tr, tr_parts
        gc.collect()
        dtr = xgb.QuantileDMatrix(Xtr, label=ytr, weight=wtr, max_bin=train.XGB_PARAMS["max_bin"])
        del Xtr
        gc.collect()
        va = features.add_relative(pl.concat(va_parts))
        Xva = train.matrix(va, feats)
        yva = va["y"].to_numpy()
        dva = xgb.QuantileDMatrix(Xva, label=yva, weight=weights(va), ref=dtr)
        booster = xgb.train({**train.XGB_PARAMS}, dtr, num_boost_round=1000, evals=[(dva, "val")],
                            early_stopping_rounds=40, verbose_eval=False)
        m = train.XgbModel(booster)
        oof_parts.append(va.select("s1_id", "target_id", "src", "q_country").with_columns(
            pl.Series("p_raw", m.predict(Xva)), pl.Series("y", va["y"].to_numpy())))
        models.append(m)
        importances.append(m.feature_importance())
        del dtr, dva, va, Xva, va_parts
        gc.collect()
    oof_raw = pl.concat(oof_parts).sort(["s1_id", "target_id"])
    y = oof_raw["y"].to_numpy()
    calibrated, calib, iso = train.maybe_calibrate(y, oof_raw["p_raw"].to_numpy())
    oof_df = oof_raw.drop("p_raw", "y").with_columns(pl.Series("p", calibrated))
    decision = pipeline.run(oof_df, truth)
    policy = {"decision_winner": decision["decision_winner"], "exclusivity_winner": decision["exclusivity_winner"],
              "threshold": decision["global_threshold"]["best"]}
    return {"models": models, "iso": iso, "oof": oof_df, "importance": np.mean(importances, axis=0),
            "calibration": calib, "decision": decision, "policy": policy}


def _score(block: dict, X: np.ndarray) -> np.ndarray:
    raw = train.predict_mean(block["models"], X)
    return (block["iso"].transform(raw) if block["iso"] is not None else raw).astype(np.float32)


def country_transfer(df: pl.DataFrame, y: np.ndarray, feats: list[str], truth: pl.DataFrame,
                     holdout: str, backend: str = "xgb_gpu") -> dict:
    """France protocol: fit + calibrate + tune policy WITHOUT `holdout`, apply frozen to it (review M4)."""
    other = (df["q_country"] != holdout).to_numpy()
    block = _fit_block(df.filter(pl.Series(other)), y[other], feats, truth, backend)
    held = df.filter(pl.Series(~other))
    p = _score(block, train.matrix(held, feats))
    scored = held.select("s1_id", "target_id", "src").with_columns(pl.Series("p", p))
    t_sub = truth.filter(pl.col("s1_id").is_in(held["s1_id"].unique().implode()))
    return {"holdout": holdout, "auc": float(roc_auc_score(y[~other], p)),
            "pr_auc": float(average_precision_score(y[~other], p)),
            "macro_f05_frozen_policy": macro_f05(t_sub, _policy_predict(scored, block["policy"])),
            "policy": block["policy"]}


def full_scope_extras(cfg: dict, cap: int, seed: int = 0) -> pl.Series:
    """Train-fold S1 clean for training: not the ranker's subset (in-sample p) and not the dense fine-tune's S1
    (in-sample dense cosine; `train_dense.training_links` replays that draw). Seeded sample of `cap`."""
    from er.blocking.train_dense import training_links

    sp = load_split()
    rsub = cfg["ranker"].get("train_subset", "rtrain")
    seen = pl.concat([subset_ids(sp, rsub), training_links(600_000, rsub)["s1_id"]]).implode()
    pool = subset_ids(sp, "train")
    pool = pool.filter(~pool.is_in(seen))
    return pool.sample(n=min(cap, pool.len()), seed=seed).sort()


def fit(config_path: str, k: int = 10, name: str = "v2", transfer: bool = False, scope: str = "val",
        backend: str = "xgb_gpu", relative: bool = True, drop_groups: tuple[str, ...] = (),
        train_scope: str = "val", macro_weight: bool = False, fu_extras: int = 450_000) -> dict:
    """train_scope "val": the M5-clean val-only fit. "expanded": context over val + rtrain12 chunks, trained on val
    + clean rtrain12 S1 (ranker's own sample excluded). "full": context over all train S1, trained on val +
    `fu_extras` clean train-fold S1 (`full_scope_extras`). Non-val scopes use per-fold streaming CV."""
    t0 = time.perf_counter()
    truth = io.load_truth()
    if train_scope == "val":
        uni = Universe(config_path, "train", k, scope)
        uni.shortlists()
        df = uni.val_features()
        y = ranker.label(df, truth)
        df = df.join(load_split().select("s1_id", "cv_fold"), on="s1_id", how="left")
        feats = features.NUMERIC_FEATURES + uni.extra_features()
        if relative:
            df = features.add_relative(df)
            feats = feats + features.REL_FEATURES
        feats = features.apply_drop_groups(feats, drop_groups)
        print(f"val pairs {df.height:,}, pos rate {y.mean():.4f}, {len(feats)} features, "
              f"dropped groups {list(drop_groups) or 'none'}", flush=True)

        t_cv = time.perf_counter()
        block = _fit_block(df, y, feats, truth, backend)
        print(f"CV + policy in {time.perf_counter() - t_cv:.0f}s", flush=True)
    else:
        split_df = load_split()
        if train_scope == "expanded":
            uni = Universe(config_path, "train", k, "val")
            extra = subset_ids(split_df, "rtrain12")
            extra = extra.filter(~extra.is_in(subset_ids(split_df, "rtrain").implode()))
            uni.extend_chunks(extra)
            tag = "ex"
        else:  # "full": competitors are all train S1, as at test, so test scores at 1 partition (M1 at the source)
            uni = Universe(config_path, "train", k, "full")
            extra = full_scope_extras(uni.cfg, fu_extras)
            tag = "fu"
        ids = pl.concat([subset_ids(split_df, "val"), extra])
        uni.shortlists()
        feats = features.NUMERIC_FEATURES + uni.extra_features() + (features.REL_FEATURES if relative else [])
        feats = features.apply_drop_groups(feats, drop_groups)
        uni.subset_features(ids, tag, uni.global_context(), collect=False)  # caches only; CV streams them
        del ids, extra
        gc.collect()
        paths = uni.chunk_paths(tag)
        context_s1 = sum(c.len() for c in uni.chunks)  # what ctx_parts must density-match at test
        train_universe = sum(pl.read_parquet(p, columns=["s1_id"])["s1_id"].n_unique() for p in paths)
        print(f"{train_scope} training: {len(paths)} chunk caches, {train_universe:,} S1 (context {context_s1:,}), "
              f"{len(feats)} features, dropped groups {list(drop_groups) or 'none'}", flush=True)
        t_cv = time.perf_counter()
        block = _fit_block_stream(paths, feats, truth, backend, macro_weight)
        print(f"streaming CV + policy in {time.perf_counter() - t_cv:.0f}s", flush=True)
        df = None
        y = None
    by_country = {}
    for c in block["oof"]["q_country"].unique().sort().to_list():
        sub = block["oof"].filter(pl.col("q_country") == c)
        t_sub = truth.filter(pl.col("s1_id").is_in(sub["s1_id"].unique().implode()))
        by_country[c] = macro_f05(t_sub, _policy_predict(sub, block["policy"]))
    if train_scope == "val":
        n_pairs, pos_rate = df.height, float(y.mean())
    else:
        ys = pl.concat(pl.read_parquet(p, columns=["y"]) for p in paths)["y"]
        n_pairs, pos_rate = ys.len(), float(ys.mean())
        if transfer:  # transfer checks need the frame: reload from caches (caches carry label + fold)
            df = features.add_relative(pl.concat(pl.read_parquet(p) for p in paths))
            y = df["y"].to_numpy()
    result = {
        "name": name, "config": config_path, "k": k,
        "scope": scope if train_scope == "val" else train_scope, "backend": backend,
        "train_scope": train_scope, "val_pairs": n_pairs, "pos_rate": pos_rate,
        "train_universe_s1": context_s1 if train_scope != "val" else None,
        "features": feats, "drop_groups": list(drop_groups),
        "policy": block["policy"], "stages": block["decision"]["stages"],
        "final_macro_f05": block["decision"]["final"]["macro_f05"], "by_country_frozen_policy": by_country,
        "calibration": block["calibration"],
        "importance": sorted(zip(feats, block["importance"].tolist()), key=lambda kv: -kv[1])[:40],
    }
    if transfer:
        result["country_transfer"] = [country_transfer(df, y, feats, truth, c, backend)
                                      for c in df["q_country"].unique().sort().to_list()]
        del df, y
        gc.collect()
    MODELS.mkdir(parents=True, exist_ok=True)
    bundle = {"models": block["models"], "iso": block["iso"], "features": feats, "policy": block["policy"],
              "config": config_path, "k": k, "scope": result["scope"],
              "train_universe_s1": result["train_universe_s1"]}
    (MODELS / f"{name}.pkl").write_bytes(pickle.dumps(bundle))
    block["oof"].write_parquet(io.ARTIFACTS / "match" / f"oof_{name}.parquet", mkdir=True)
    result["total_s"] = round(time.perf_counter() - t0, 1)
    (io.REPORTS / "match" / f"{name}.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def ctx_parts(bundle: dict) -> int:
    """Test partitions giving the same S1-per-target density the bundle's context was trained on.
    Derived from the bundle's own training-universe size so any fit path stays density-matched."""
    n_tg = {s: sum(io.load_source(s, src).height for src in io.TARGET_SOURCES) for s in ("train", "test")}
    train_n = bundle.get("train_universe_s1") or subset_ids(load_split(), "val").len()
    train_density = train_n / n_tg["train"]
    test_density = io.load_source("test", "S1").height / n_tg["test"]
    return max(1, round(test_density / train_density))


def submit(name: str = "v2", parts: int | None = None) -> dict:
    """Score every test S1 with the frozen bundle (fold-mean + calibration + policy); write both TSVs."""
    bundle = pickle.loads((MODELS / f"{name}.pkl").read_bytes())
    parts = ctx_parts(bundle) if parts is None else parts
    uni = Universe(bundle["config"], "test", bundle["k"])
    uni.shortlists()
    uni._ctx = None  # pass 2 needs only cached shortlists + pools: drop the blocking context (~20 GB)
    release_caches()
    gctx = uni.global_context(parts)
    feats = bundle["features"]
    block = {"models": bundle["models"], "iso": bundle["iso"]}
    frames = []
    for i in range(len(uni.chunks)):
        df = uni.features(i, None, gctx)
        if features.REL_FEATURES[0] in feats:
            df = features.add_relative(df)
        frames.append(df.select("s1_id", "target_id", "src").with_columns(
            pl.Series("p", _score(block, train.matrix(df, feats)))))
        del df
        gc.collect()
    scored = pl.concat(frames)
    scored.write_parquet(io.ARTIFACTS / "match" / f"test_scored_{name}_p{parts}.parquet", mkdir=True)  # for diagnostics
    test_ids = pl.concat(uni.chunks)
    preds = _policy_predict(scored, bundle["policy"], n_s1=test_ids.len())

    out = io.ROOT / "output"
    out.mkdir(exist_ok=True)

    def write(frame: pl.DataFrame, col: str, fname: str) -> pl.DataFrame:
        lists = frame.sort("s1_id", "target_id").group_by("s1_id", maintain_order=True).agg(
            pl.col("target_id").str.join(",").alias(col))
        full = test_ids.to_frame("source1_entity_id").join(
            lists.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left"
        ).with_columns(pl.col(col).fill_null(""))
        full.write_csv(out / fname, separator="\t", quote_style="never")
        return full

    full = write(preds, "matched_entity_ids", "matching_results.tsv")
    write(scored, "candidate_entity_ids", "candidate_pairs.tsv")
    prior = linkcount.load_prior()
    stats = {"name": name, "test_s1": test_ids.len(), "scored_pairs": scored.height,
             "predicted_pairs": preds.height, "s1_with_matches": int((full["matched_entity_ids"] != "").sum()),
             "policy": bundle["policy"], "ctx_parts": parts,
             # Under-linking is the visible symptom of a threshold that did not transfer (area 4).
             "links_per_s1": preds.height / max(test_ids.len(), 1), "prior_links_per_s1": prior.mean_total}
    (io.REPORTS / f"submit_{name}.json").write_text(json.dumps(stats, indent=2))
    return stats


def redecide(name: str = "v2") -> dict:
    """Re-run the Step 5 policy on saved OOF (no retraining) and update the bundle's frozen policy."""
    oof = pl.read_parquet(io.ARTIFACTS / "match" / f"oof_{name}.parquet")
    if "cv_fold" not in oof.columns:
        oof = oof.join(load_split().select("s1_id", "cv_fold"), on="s1_id", how="left")
    truth = io.load_truth()
    decision = pipeline.run(oof, truth)
    policy = {"decision_winner": decision["decision_winner"], "exclusivity_winner": decision["exclusivity_winner"],
              "threshold": decision["global_threshold"]["best"]}
    by_country = {}
    for c in oof["q_country"].unique().sort().to_list():
        sub = oof.filter(pl.col("q_country") == c)
        t_sub = truth.filter(pl.col("s1_id").is_in(sub["s1_id"].unique().implode()))
        by_country[c] = macro_f05(t_sub, _policy_predict(sub, policy))
    bundle_path = MODELS / f"{name}.pkl"
    bundle = pickle.loads(bundle_path.read_bytes())
    bundle["policy"] = policy
    bundle_path.write_bytes(pickle.dumps(bundle))
    out = {"policy": policy, "by_country_frozen_policy": by_country,
           "stages": {k: v["macro_f05"] for k, v in decision["stages"].items()},
           "nested_threshold": decision["nested_threshold"]}
    (io.REPORTS / "match" / f"{name}_decision.json").write_text(json.dumps(out, indent=2, default=str))
    return out

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
from er.decide import exclusivity, expected_f05, pipeline, threshold
from er.match import features, train
from er.match.dataset import Universe
from er.metrics import macro_f05
from er.split import load_split

MODELS = io.ARTIFACTS / "models" / "match"


def _policy_predict(scored: pl.DataFrame, policy: dict) -> pl.DataFrame:
    excl = (exclusivity.resolve_scarcity_weighted(scored) if policy["exclusivity_winner"] == "scarcity_weighted"
            else exclusivity.resolve_max_probability(scored))
    if policy["decision_winner"] == "tuned_global_threshold":
        return threshold.predict(excl, policy["threshold"])
    return expected_f05.select(excl).select("s1_id", "target_id")


def _fit_block(df: pl.DataFrame, y: np.ndarray, feats: list[str], truth: pl.DataFrame) -> dict:
    """CV + calibration + Step 5 policy on one set of rows: everything a frozen deployment needs."""
    oof, models, imp = train.cv_train(df, y, "cv_fold", feats, features.CATEGORICAL_FEATURES)
    calibrated, calib, iso = train.maybe_calibrate(y, oof)
    oof_df = df.select("s1_id", "target_id", "src", "q_country").with_columns(pl.Series("p", calibrated))
    decision = pipeline.run(oof_df, truth)
    policy = {"decision_winner": decision["decision_winner"], "exclusivity_winner": decision["exclusivity_winner"],
              "threshold": decision["global_threshold"]["best"]}
    return {"models": models, "iso": iso, "oof": oof_df, "importance": imp, "calibration": calib,
            "decision": decision, "policy": policy}


def _score(block: dict, X: np.ndarray) -> np.ndarray:
    raw = train.predict_mean(block["models"], X)
    return (block["iso"].transform(raw) if block["iso"] is not None else raw).astype(np.float32)


def country_transfer(df: pl.DataFrame, y: np.ndarray, feats: list[str], truth: pl.DataFrame,
                     holdout: str) -> dict:
    """France protocol: fit + calibrate + tune policy WITHOUT `holdout`, apply frozen to it (review M4)."""
    other = (df["q_country"] != holdout).to_numpy()
    block = _fit_block(df.filter(pl.Series(other)), y[other], feats, truth)
    held = df.filter(pl.Series(~other))
    p = _score(block, train.matrix(held, feats))
    scored = held.select("s1_id", "target_id", "src").with_columns(pl.Series("p", p))
    t_sub = truth.filter(pl.col("s1_id").is_in(held["s1_id"].unique().implode()))
    return {"holdout": holdout, "auc": float(roc_auc_score(y[~other], p)),
            "pr_auc": float(average_precision_score(y[~other], p)),
            "macro_f05_frozen_policy": macro_f05(t_sub, _policy_predict(scored, block["policy"])),
            "policy": block["policy"]}


def fit(config_path: str, k: int = 10, name: str = "v2", transfer: bool = True) -> dict:
    t0 = time.perf_counter()
    uni = Universe(config_path, "train", k)
    uni.shortlists()
    df = uni.val_features()
    truth = io.load_truth()
    y = ranker.label(df, truth)
    df = df.join(load_split().select("s1_id", "cv_fold"), on="s1_id", how="left")
    feats = features.NUMERIC_FEATURES + uni.extra_features()
    print(f"val pairs {df.height:,}, pos rate {y.mean():.4f}, {len(feats)} features", flush=True)

    block = _fit_block(df, y, feats, truth)
    by_country = {}
    for c in df["q_country"].unique().sort().to_list():
        sub = block["oof"].filter(pl.col("q_country") == c)
        t_sub = truth.filter(pl.col("s1_id").is_in(sub["s1_id"].unique().implode()))
        by_country[c] = macro_f05(t_sub, _policy_predict(sub, block["policy"]))
    result = {
        "name": name, "config": config_path, "k": k, "val_pairs": df.height, "pos_rate": float(y.mean()),
        "features": feats, "policy": block["policy"], "stages": block["decision"]["stages"],
        "final_macro_f05": block["decision"]["final"]["macro_f05"], "by_country_frozen_policy": by_country,
        "calibration": block["calibration"],
        "importance": sorted(zip(feats, block["importance"].tolist()), key=lambda kv: -kv[1])[:40],
    }
    if transfer:
        result["country_transfer"] = [country_transfer(df, y, feats, truth, c)
                                      for c in df["q_country"].unique().sort().to_list()]
    MODELS.mkdir(parents=True, exist_ok=True)
    bundle = {"models": block["models"], "iso": block["iso"], "features": feats, "policy": block["policy"],
              "config": config_path, "k": k}
    (MODELS / f"{name}.pkl").write_bytes(pickle.dumps(bundle))
    block["oof"].write_parquet(io.ARTIFACTS / "match" / f"oof_{name}.parquet", mkdir=True)
    result["total_s"] = round(time.perf_counter() - t0, 1)
    (io.REPORTS / "match" / f"{name}.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def submit(name: str = "v2") -> dict:
    """Score every test S1 with the frozen bundle (fold-mean + calibration + policy); write both TSVs."""
    bundle = pickle.loads((MODELS / f"{name}.pkl").read_bytes())
    uni = Universe(bundle["config"], "test", bundle["k"])
    uni.shortlists()
    gctx = uni.global_context()
    feats = bundle["features"]
    block = {"models": bundle["models"], "iso": bundle["iso"]}
    parts = []
    for i in range(len(uni.chunks)):
        df = uni.features(i, None, gctx)
        parts.append(df.select("s1_id", "target_id", "src").with_columns(
            pl.Series("p", _score(block, train.matrix(df, feats)))))
        del df
        gc.collect()
    scored = pl.concat(parts)
    preds = _policy_predict(scored, bundle["policy"])

    test_ids = pl.concat(uni.chunks)
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
    stats = {"name": name, "test_s1": test_ids.len(), "scored_pairs": scored.height,
             "predicted_pairs": preds.height, "s1_with_matches": int((full["matched_entity_ids"] != "").sum()),
             "policy": bundle["policy"]}
    (io.REPORTS / f"submit_{name}.json").write_text(json.dumps(stats, indent=2))
    return stats

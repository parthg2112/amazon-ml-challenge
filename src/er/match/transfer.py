"""Unexplored area 2: which features actually transfer to an unseen country.

Two findings from the v2 run motivate this:

  * `p`, the Step-3 ranker probability, carries 28.1M of the total gain against 0.86M for the next
    feature. Step 4 is largely re-calibrating the ranker — and the ranker never saw France.
  * Leave-India-out loses 0.033 macro F0.5 (0.9769 -> 0.9441) while AUC barely moves (0.9985 ->
    0.9970). Re-tuning the threshold with oracle access recovers only ~0.004 of that, so the rest is
    the features themselves shifting, not the operating point.

So the experiment is an ablation crossed with the leave-country-out protocol: drop the ranker score,
add the within-record relative/rank views from `features.add_rank_relative`, or both. The number that
matters is not the in-distribution score (dropping `p` can only hurt that) but the *gap* between
in-distribution and held-out country, which is what France will pay.
"""

from __future__ import annotations

import json
import time

import polars as pl

from er import io
from er.blocking import ranker
from er.match import features, v2
from er.match.dataset import Universe
from er.metrics import macro_f05
from er.split import load_split

RANKER_COLS = ("p", "frank")  # Step-3 outputs: absolute, and fitted on US/India only
ARMS = ("base", "no_p", "rel", "rel_no_p")


def arm_features(arm: str, base_feats: list[str], rel_feats: list[str]) -> list[str]:
    """`rel_no_p` keeps `p_rel`/`p_rk` — the ranker's *ordering* within a record, not its absolute value."""
    keep = [f for f in base_feats if f not in RANKER_COLS] if arm.endswith("no_p") else list(base_feats)
    return keep + (rel_feats if arm.startswith("rel") else [])


def evaluate_arm(df: pl.DataFrame, y, feats: list[str], truth: pl.DataFrame, arm: str,
                backend: str = "xgb_gpu") -> dict:
    """In-distribution CV plus both leave-country-out directions, with the policy frozen each time."""
    t0 = time.perf_counter()
    block = v2._fit_block(df, y, feats, truth, backend)
    in_dist = {}
    for c in df["q_country"].unique().sort().to_list():
        sub = block["oof"].filter(pl.col("q_country") == c)
        t_sub = truth.filter(pl.col("s1_id").is_in(sub["s1_id"].unique().implode()))
        in_dist[c] = macro_f05(t_sub, v2._policy_predict(sub, block["policy"]))
    held = {t["holdout"]: t for t in
            (v2.country_transfer(df, y, feats, truth, c, backend)
             for c in df["q_country"].unique().sort().to_list())}
    return {
        "arm": arm, "n_features": len(feats),
        "overall": block["decision"]["final"]["macro_f05"],
        "policy": block["policy"],
        "in_distribution": in_dist,
        "held_out": {c: h["macro_f05_frozen_policy"] for c, h in held.items()},
        "held_out_auc": {c: h["auc"] for c, h in held.items()},
        "gap": {c: in_dist[c] - held[c]["macro_f05_frozen_policy"] for c in held},
        "seconds": round(time.perf_counter() - t0, 1),
    }


def run(config_path: str, k: int = 10, scope: str = "val", arms: tuple[str, ...] = ARMS,
        backend: str = "xgb_gpu", name: str = "transfer") -> dict:
    uni = Universe(config_path, "train", k, scope)
    uni.shortlists()
    df = uni.val_features()
    truth = io.load_truth()
    y = ranker.label(df, truth)
    df = df.join(load_split().select("s1_id", "cv_fold"), on="s1_id", how="left")
    df = features.add_rank_relative(df)

    base_feats = features.NUMERIC_FEATURES + uni.extra_features()
    rel_feats = [c for c in features.relative_names(features.RELATIVE_BASE) if c in df.columns]
    print(f"{df.height:,} pairs, {len(base_feats)} base + {len(rel_feats)} relative features", flush=True)

    results = [evaluate_arm(df, y, arm_features(a, base_feats, rel_feats), truth, a, backend)
               for a in arms]
    for r in results:
        print(f"  {r['arm']:10s} overall {r['overall']:.5f}  gap "
              + "  ".join(f"{c} {g:+.4f}" for c, g in sorted(r["gap"].items())), flush=True)

    best_gap = min(results, key=lambda r: max(r["gap"].values()))
    out = {"config": config_path, "scope": scope, "backend": backend, "arms": results,
           "baseline": next((r for r in results if r["arm"] == "base"), None),
           "smallest_worst_case_gap": best_gap["arm"],
           "relative_features": rel_feats}
    (io.REPORTS / "match" / f"{name}.json").write_text(json.dumps(out, indent=2, default=str))
    return out

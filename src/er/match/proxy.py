"""Unseen-country rehearsal with persisted scores: fit the bundle's recipe without one country, score it frozen.

`v2.country_transfer` reports only F0.5; decision-layer ideas for France (Σp matching, per-country tilt,
EXR, CE band) need the held-out scores themselves. Caveat: ranker p and the dense model saw India (M9).
Arms (group masking, added features) are compared against a `base` arm run by this same code, never
against older proxy reports: the fit path streams by fold, so numbers are only comparable within a pair.
"""

from __future__ import annotations

import gc
import json
import pickle

import polars as pl

from er import io
from er.match import features, train
from er.match.dataset import CACHE, FEATURE_VERSION
from er.match.v2 import MODELS, _fit_block_stream, _policy_predict, _score
from er.metrics import macro_f05

SCOPE_TAG = {"expanded": "ex", "full": "fu"}


def run(name: str = "v7", holdout: str = "India", backend: str = "xgb_gpu", arm: str = "base",
        mask_groups: tuple[str, ...] = (), mask_rate: float = 0.0, add_features: tuple[str, ...] = ()) -> dict:
    bundle = pickle.loads((MODELS / f"{name}.pkl").read_bytes())
    feats = bundle["features"] + [f for f in add_features if f not in bundle["features"]]
    tag = SCOPE_TAG[bundle["scope"]]
    paths = sorted(CACHE.glob(f"train_*/features_v{FEATURE_VERSION}_{tag}_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no features_v{FEATURE_VERSION}_{tag}_* caches: build them with fit2 first")
    truth = io.load_truth()
    mask = train.GroupMask(tuple(mask_groups), mask_rate) if mask_groups and mask_rate > 0 else None

    block = _fit_block_stream(paths, feats, truth, backend, mask=mask, where=pl.col("q_country") != holdout)
    gc.collect()
    parts = []
    for p in paths:
        held = pl.read_parquet(p).filter(pl.col("q_country") == holdout)
        if held.height:
            held = features.add_relative(held)
            parts.append(held.select("s1_id", "target_id", "src", "q_country", "cv_fold").with_columns(
                pl.Series("p", _score(block, train.matrix(held, feats)))))
        del held
        gc.collect()
    scored = pl.concat(parts)
    t_held = truth.filter(pl.col("s1_id").is_in(scored["s1_id"].unique().implode()))

    stem = f"proxy_{name}_{holdout.lower()}_{arm}"
    out = io.ARTIFACTS / "match" / f"{stem}.parquet"
    scored.write_parquet(out)
    block["oof"].write_parquet(io.ARTIFACTS / "match" / f"{stem}_trainoof.parquet")
    result = {"name": name, "holdout": holdout, "arm": arm, "group_mask": mask.to_json() if mask else None,
              "added_features": [f for f in add_features if f in feats], "n_features": len(feats),
              "policy": block["policy"], "held_pairs": scored.height,
              "train_final_macro_f05": block["decision"]["final"]["macro_f05"],
              "held_out_macro_f05": macro_f05(t_held, _policy_predict(scored, block["policy"])),
              "scores": str(out)}
    (io.REPORTS / "match" / f"{stem}.json").write_text(json.dumps(result, indent=2, default=str))
    return result

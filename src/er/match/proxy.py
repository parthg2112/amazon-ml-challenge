"""Unseen-country rehearsal with persisted scores: fit the v6 recipe without one country, score it frozen.

`v2.country_transfer` reports only F0.5; decision-layer ideas for France (Σp matching, per-country tilt,
EXR, CE band) need the held-out scores themselves. Caveat: ranker p and the dense model saw India (M9).
"""

from __future__ import annotations

import gc
import json
import pickle

import polars as pl

from er import io
from er.match import features, train
from er.match.dataset import CACHE, FEATURE_VERSION
from er.match.v2 import MODELS, _fit_block, _score


def run(name: str = "v6", holdout: str = "India", backend: str = "xgb_gpu") -> dict:
    bundle = pickle.loads((MODELS / f"{name}.pkl").read_bytes())
    feats = bundle["features"]
    paths = sorted(CACHE.glob(f"train_*/features_v{FEATURE_VERSION}_ex_*.parquet"))
    truth = io.load_truth()

    def load(keep_holdout: bool) -> pl.DataFrame:
        cond = pl.col("q_country") == holdout if keep_holdout else pl.col("q_country") != holdout
        return features.add_relative(pl.concat(pl.read_parquet(p).filter(cond) for p in paths))

    df = load(False)
    block = _fit_block(df, df["y"].to_numpy(), feats, truth, backend)
    del df
    gc.collect()
    held = load(True)
    scored = held.select("s1_id", "target_id", "src", "q_country", "cv_fold").with_columns(
        pl.Series("p", _score(block, train.matrix(held, feats))))
    out = io.ARTIFACTS / "match" / f"proxy_{name}_{holdout.lower()}.parquet"
    scored.write_parquet(out)
    block["oof"].write_parquet(io.ARTIFACTS / "match" / f"proxy_{name}_{holdout.lower()}_trainoof.parquet")
    result = {"name": name, "holdout": holdout, "policy": block["policy"], "held_pairs": scored.height,
              "train_final_macro_f05": block["decision"]["final"]["macro_f05"], "scores": str(out)}
    (io.REPORTS / "match" / f"proxy_{name}_{holdout.lower()}.json").write_text(json.dumps(result, indent=2, default=str))
    return result

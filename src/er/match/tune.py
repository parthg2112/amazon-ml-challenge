"""Unexplored area 3: hyper-parameters, monotone constraints, CatBoost.

The v2 model runs on XGBoost defaults, so this is the cheapest remaining headroom. Monotone
constraints matter for more than accuracy: forcing "more similar cannot mean less likely to match"
removes the model's freedom to fit non-monotone quirks that only exist in US/India data, which is
exactly the failure mode the France gap shows. They also make the model's behaviour on an unseen
country predictable rather than merely hoped-for.

Objective is the real metric — OOF macro F0.5 after exclusivity and a tuned threshold — not AUC,
because the v2 evidence is that AUC barely moves while F0.5 moves a lot (leave-India-out: AUC 0.9985
-> 0.9970, F0.5 0.9769 -> 0.9441).
"""

from __future__ import annotations

import json
import time

import numpy as np
import polars as pl

from er import io
from er.blocking import ranker
from er.decide import exclusivity, threshold
from er.match import features, train
from er.match.dataset import Universe
from er.split import load_split

# Direction of the effect a feature is allowed to have on the match probability. Anything genuinely
# ambiguous (empty-address flags, candidate counts, noise-catalogue flags, every categorical code)
# stays unconstrained at 0 — a wrong constraint is worse than none.
_INC_EXACT = {
    "p", "dense_cos", "score_B7_dense", "name_core_exact", "alias_match", "name_min_exact",
    "addr_num_overlap", "addr_num_jaccard", "margin_to_best_competitor", "support_opposite_src_best",
}
_DEC_EXACT = {
    "house_abs_diff", "house_rel_diff", "script_mismatch", "n_s1_claiming_record", "frank",
    "rank_within_s1_src", "rank_by_name", "rank_by_addr",
}
_INC_SUFFIX = ("_overlap_n", "_overlap_idf", "_rare_overlap_n", "_rel")
_DEC_SUFFIX = ("_extra_q_n", "_extra_q_idf", "_extra_t_n", "_extra_t_idf", "_rk")


def direction(feat: str) -> int:
    """+1 monotone increasing, -1 decreasing, 0 unconstrained."""
    if feat in features.CATEGORICAL_FEATURES:
        return 0
    if feat in _INC_EXACT:
        return 1
    if feat in _DEC_EXACT:
        return -1
    if feat.startswith(("sim_", "score_")):
        return 1
    if feat.endswith(_DEC_SUFFIX):
        return -1
    if feat.endswith(_INC_SUFFIX):
        return 1
    return 0


def monotone_vector(feats: list[str]) -> tuple[int, ...]:
    return tuple(direction(f) for f in feats)


def monotone_param(feats: list[str]) -> str:
    """XGBoost wants the constraint list as a bracketed string aligned to the feature matrix."""
    return "(" + ",".join(str(d) for d in monotone_vector(feats)) + ")"


def score_params(df: pl.DataFrame, y: np.ndarray, feats: list[str], truth: pl.DataFrame, params: dict,
                backend: str, grid: np.ndarray | None = None) -> float:
    """OOF macro F0.5 with max-probability exclusivity and a tuned cutoff — the deployed rule."""
    oof, _models, _imp = train.cv_train(df, y, "cv_fold", feats, features.CATEGORICAL_FEATURES,
                                        params=params, backend=backend)
    calibrated, _c, _iso = train.maybe_calibrate(y, oof)
    scored = df.select("s1_id", "target_id", "src").with_columns(pl.Series("p", calibrated))
    excl = exclusivity.resolve_max_probability(scored)
    t_sub = truth.filter(pl.col("s1_id").is_in(df["s1_id"].unique().implode()))
    _t, best, _curve = threshold.tune(excl, t_sub, grid=np.arange(0.30, 0.96, 0.01) if grid is None else grid)
    return best


def suggest(trial, backend: str) -> dict:
    if backend == "catboost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "depth": trial.suggest_int("depth", 5, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        }
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 11),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
    }


def subsample_s1(df: pl.DataFrame, y: np.ndarray, frac: float, seed: int = 0):
    """Trials run on a fraction of S1 (whole records, never split) to keep the search affordable."""
    if frac >= 1.0:
        return df, y
    keep = (pl.col("s1_id").hash(seed=seed) % 1000) < int(frac * 1000)
    mask = df.select(keep.alias("m"))["m"].to_numpy()
    return df.filter(pl.Series(mask)), y[mask]


def run(config_path: str, k: int = 10, scope: str = "val", backend: str = "xgb_gpu", trials: int = 30,
        frac: float = 0.3, monotone: bool = True, name: str = "tune", relative: bool = True,
        drop_groups: tuple[str, ...] = ()) -> dict:
    import optuna

    uni = Universe(config_path, "train", k, scope)
    uni.shortlists()
    df = uni.val_features()
    truth = io.load_truth()
    y = ranker.label(df, truth)
    df = df.join(load_split().select("s1_id", "cv_fold"), on="s1_id", how="left")
    feats = features.NUMERIC_FEATURES + uni.extra_features()
    if relative:
        df = features.add_relative(df)
        feats = feats + features.REL_FEATURES
    feats = features.apply_drop_groups(feats, drop_groups)
    sub_df, sub_y = subsample_s1(df, y, frac)
    print(f"search on {sub_df.height:,} of {df.height:,} pairs, {len(feats)} features", flush=True)

    mono = {"monotone_constraints": monotone_param(feats)} if monotone and backend != "catboost" else {}
    t0 = time.perf_counter()
    baseline = score_params(sub_df, sub_y, feats, truth, {}, backend)
    mono_only = score_params(sub_df, sub_y, feats, truth, mono, backend) if mono else baseline
    print(f"defaults {baseline:.5f} | monotone {mono_only:.5f}", flush=True)

    def objective(trial):
        return score_params(sub_df, sub_y, feats, truth, {**mono, **suggest(trial, backend)}, backend)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    best = {**mono, **study.best_params}
    full = score_params(df, y, feats, truth, best, backend)  # confirm on the full frame
    out = {
        "config": config_path, "backend": backend, "scope": scope, "trials": trials, "frac": frac,
        "monotone": monotone,
        "subsample_scores": {"defaults": baseline, "monotone_only": mono_only, "best": study.best_value},
        "best_params": {k_: v for k_, v in study.best_params.items()},
        "full_frame_macro_f05": full,
        "constrained_features": {f: direction(f) for f in feats if direction(f)},
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (io.REPORTS / "match" / f"{name}.json").write_text(json.dumps(out, indent=2, default=str))
    return out

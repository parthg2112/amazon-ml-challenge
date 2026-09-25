"""Step 4 model: LightGBM binary classifier, 5 grouped folds, OOF probabilities, calibration check.

Folds are the `cv_fold` column from `er.split` (SHA-256 bucketed by s1_id, train subset only), so every
candidate of a given S1 lands in one fold — no leakage across the train/validation boundary a model
could exploit.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from er.metrics import macro_f05

DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "is_unbalance": True,
    "verbosity": -1,
    "seed": 0,
}


XGB_PARAMS = {
    "device": "cuda", "tree_method": "hist", "objective": "binary:logistic", "eval_metric": "logloss",
    "learning_rate": 0.08, "max_depth": 8, "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.8,
    "max_bin": 256, "seed": 0,
}


class XgbModel:
    """Adapter so fold-mean prediction treats XGBoost like a LightGBM Booster (predict + best_iteration)."""

    def __init__(self, booster):
        self.booster = booster
        self.best_iteration = booster.best_iteration

    def predict(self, X: np.ndarray, num_iteration: int | None = None) -> np.ndarray:
        import xgboost as xgb

        it = self.best_iteration if num_iteration is None else num_iteration
        return self.booster.predict(xgb.DMatrix(X), iteration_range=(0, it + 1))

    def feature_importance(self, importance_type: str = "gain") -> np.ndarray:
        scores = self.booster.get_score(importance_type="total_gain")
        return np.array([scores.get(f"f{i}", 0.0) for i in range(self.booster.num_features())])


def cv_train_xgb(X: np.ndarray, y: np.ndarray, fold_arr: np.ndarray, params: dict | None = None,
                 num_boost_round: int = 1000, early_stopping_rounds: int = 40) -> tuple[np.ndarray, list, np.ndarray]:
    """Same contract as `cv_train`, on the GPU (~10x faster than CPU LightGBM at 9M rows)."""
    import xgboost as xgb

    params = {**XGB_PARAMS, **(params or {})}
    oof = np.zeros(len(y), dtype=np.float32)
    models, importances = [], []
    for f in sorted(set(fold_arr.tolist())):
        tr, va = fold_arr != f, fold_arr == f
        dtr = xgb.QuantileDMatrix(X[tr], label=y[tr], max_bin=params["max_bin"])
        dva = xgb.QuantileDMatrix(X[va], label=y[va], ref=dtr)
        booster = xgb.train(params, dtr, num_boost_round=num_boost_round, evals=[(dva, "val")],
                            early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
        m = XgbModel(booster)
        oof[va] = m.predict(X[va])
        models.append(m)
        importances.append(m.feature_importance())
        del dtr, dva
    return oof, models, np.mean(importances, axis=0)


class CatModel:
    """Same adapter contract as `XgbModel`, for CatBoost (Apache-2.0)."""

    def __init__(self, model):
        self.model = model
        self.best_iteration = model.get_best_iteration() or model.tree_count_ - 1

    def predict(self, X: np.ndarray, num_iteration: int | None = None) -> np.ndarray:
        it = self.best_iteration if num_iteration is None else num_iteration
        return self.model.predict_proba(X, ntree_end=it + 1)[:, 1]

    def feature_importance(self, importance_type: str = "gain") -> np.ndarray:
        return np.asarray(self.model.get_feature_importance())


CAT_PARAMS = {"loss_function": "Logloss", "eval_metric": "Logloss", "learning_rate": 0.08, "depth": 8,
              "l2_leaf_reg": 3.0, "random_seed": 0, "verbose": False, "allow_writing_files": False}


def cv_train_cat(X: np.ndarray, y: np.ndarray, fold_arr: np.ndarray, params: dict | None = None,
                num_boost_round: int = 1000, early_stopping_rounds: int = 40):
    from catboost import CatBoostClassifier

    params = {**CAT_PARAMS, **(params or {})}
    oof = np.zeros(len(y), dtype=np.float32)
    models, importances = [], []
    for f in sorted(set(fold_arr.tolist())):
        tr, va = fold_arr != f, fold_arr == f
        m = CatBoostClassifier(iterations=num_boost_round, early_stopping_rounds=early_stopping_rounds,
                               **params)
        m.fit(X[tr], y[tr], eval_set=(X[va], y[va]))
        wrapped = CatModel(m)
        oof[va] = wrapped.predict(X[va])
        models.append(wrapped)
        importances.append(wrapped.feature_importance())
    return oof, models, np.mean(importances, axis=0)


def cv_train(df: pl.DataFrame, y: np.ndarray, fold_col: str, feature_names: list[str],
            categorical: list[str], params: dict | None = None, num_boost_round: int = 500,
            early_stopping_rounds: int = 30, backend: str = "lgbm") -> tuple[np.ndarray, list, np.ndarray]:
    """Leave-one-fold-out training; returns out-of-fold probabilities, per-fold boosters, mean gain importance.

    backend: "lgbm" (CPU, categorical-aware) | "xgb_gpu" (CUDA; categorical codes are small ordered ints)
    | "xgb_cpu" (same trees, no CUDA) | "catboost"."""
    if backend in ("xgb_gpu", "xgb_cpu"):
        p = dict(params or {})
        if backend == "xgb_cpu":
            p.setdefault("device", "cpu")
        return cv_train_xgb(matrix(df, feature_names), y, df[fold_col].to_numpy(), p)
    if backend == "catboost":
        return cv_train_cat(matrix(df, feature_names), y, df[fold_col].to_numpy(), params)
    import lightgbm as lgb

    params = {**DEFAULT_PARAMS, **(params or {})}
    X = matrix(df, feature_names)
    cat_idx = [feature_names.index(c) for c in categorical if c in feature_names]
    fold_arr = df[fold_col].to_numpy()
    oof = np.zeros(len(y), dtype=np.float32)
    models, importances = [], []
    for f in sorted(set(fold_arr.tolist())):
        train_idx, val_idx = fold_arr != f, fold_arr == f
        dtrain = lgb.Dataset(X[train_idx], label=y[train_idx], feature_name=feature_names,
                             categorical_feature=cat_idx, free_raw_data=False)
        dval = lgb.Dataset(X[val_idx], label=y[val_idx], feature_name=feature_names,
                           categorical_feature=cat_idx, reference=dtrain, free_raw_data=False)
        booster = lgb.train(params, dtrain, num_boost_round=num_boost_round, valid_sets=[dval],
                            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False),
                                       lgb.log_evaluation(0)])
        oof[val_idx] = booster.predict(X[val_idx], num_iteration=booster.best_iteration)
        models.append(booster)
        importances.append(booster.feature_importance(importance_type="gain"))
    return oof, models, np.mean(importances, axis=0)


def matrix(df: pl.DataFrame, feature_names: list[str]) -> np.ndarray:
    """float32 (lossless for these features) halves memory vs the default float64 at 9M x 120."""
    return df.select(pl.col(feature_names).cast(pl.Float32)).to_numpy()


def predict_mean(models: list, X: np.ndarray) -> np.ndarray:
    """Average of the CV fold models (review M2): the same models the OOF calibration/threshold saw."""
    return np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)


def train_full(df: pl.DataFrame, y: np.ndarray, feature_names: list[str], categorical: list[str],
              params: dict | None = None, num_boost_round: int = 300):
    import lightgbm as lgb

    params = {**DEFAULT_PARAMS, **(params or {})}
    cat_idx = [feature_names.index(c) for c in categorical if c in feature_names]
    X = df.select(feature_names).to_numpy()
    dtrain = lgb.Dataset(X, label=y, feature_name=feature_names, categorical_feature=cat_idx)
    return lgb.train(params, dtrain, num_boost_round=num_boost_round)


def calibration_report(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    gap = float(np.max(np.abs(frac_pos - mean_pred))) if len(frac_pos) else 0.0
    return {
        "bin_mean_pred": mean_pred.tolist(), "bin_frac_pos": frac_pos.tolist(),
        "max_calibration_gap": gap, "brier_score": float(brier_score_loss(y, p)),
    }


def maybe_calibrate(y: np.ndarray, oof: np.ndarray, gap_threshold: float = 0.05):
    """Isotonic-calibrate only if the raw reliability curve is off by more than `gap_threshold`."""
    raw = calibration_report(y, oof)
    if raw["max_calibration_gap"] <= gap_threshold:
        return oof, {"raw": raw, "calibrated": None, "applied": False}, None
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof, y)
    calibrated = iso.transform(oof).astype(np.float32)
    return calibrated, {"raw": raw, "calibrated": calibration_report(y, calibrated), "applied": True}, iso


def _threshold_metrics(df: pl.DataFrame, y: np.ndarray, p: np.ndarray, s1_ids_all: pl.Series,
                       truth: pl.DataFrame, threshold: float = 0.5) -> dict:
    truth_subset = truth.filter(pl.col("s1_id").is_in(s1_ids_all.unique().implode()))
    preds = df.filter(pl.Series(p) >= threshold).select("s1_id", "target_id")
    return {
        "auc": float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else None,
        "pr_auc": float(average_precision_score(y, p)),
        "macro_f05_at_threshold": macro_f05(truth_subset, preds),
        "threshold": threshold,
        "n_pairs": len(y),
        "pos_rate": float(y.mean()),
    }


def evaluate(df: pl.DataFrame, y: np.ndarray, p: np.ndarray, truth: pl.DataFrame,
            threshold: float = 0.5) -> dict:
    overall = _threshold_metrics(df, y, p, df["s1_id"], truth, threshold)
    by_country = {}
    for country in df["q_country"].unique().sort().to_list():
        mask = (df["q_country"] == country).to_numpy()
        if mask.sum() == 0:
            continue
        sub = df.filter(pl.Series(mask))
        by_country[country] = _threshold_metrics(sub, y[mask], p[mask], sub["s1_id"], truth, threshold)
    overall["by_country"] = by_country
    return overall


def country_holdout_check(df: pl.DataFrame, y: np.ndarray, feature_names: list[str], categorical: list[str],
                          holdout_country: str, truth: pl.DataFrame, params: dict | None = None) -> dict:
    """Train with `holdout_country` fully excluded; evaluate on it — the France stand-in check."""
    country = df["q_country"].to_numpy()
    train_mask, test_mask = country != holdout_country, country == holdout_country
    if test_mask.sum() == 0 or train_mask.sum() == 0:
        return {"skipped": f"no rows for country in/out of {holdout_country!r}"}
    train_df, test_df = df.filter(pl.Series(train_mask)), df.filter(pl.Series(test_mask))
    booster = train_full(train_df, y[train_mask], feature_names, categorical, params)
    p = booster.predict(test_df.select(feature_names).to_numpy())
    return _threshold_metrics(test_df, y[test_mask], p, test_df["s1_id"], truth)


def fit_and_report(df: pl.DataFrame, extra_features: list[str], truth: pl.DataFrame, folds: pl.DataFrame,
                   holdout_country: str = "India") -> dict:
    """5-fold LightGBM → calibration → gate metrics, given an already-built match-feature frame.

    `df` is `er.match.features.build()` output (one row per candidate pair, `s1_id`/`target_id`/`q_country`
    present) for candidate collection and provenance columns, which stays in `er.blocking.runner` where the
    chunked blocking context already lives. `folds` is the train-subset slice of `er.split.load_split()`
    (needs `s1_id`, `cv_fold`).
    """
    from er.blocking import ranker
    from er.match import features

    y = ranker.label(df, truth)
    df = df.join(folds.select("s1_id", "cv_fold"), on="s1_id", how="left")

    feature_names = features.NUMERIC_FEATURES + extra_features
    dupes = {c for c in feature_names if feature_names.count(c) > 1}
    if dupes:
        raise ValueError(f"duplicate feature column(s) {dupes}: extra_features collides with NUMERIC_FEATURES")
    categorical = features.CATEGORICAL_FEATURES

    oof, models, importances = cv_train(df, y, "cv_fold", feature_names, categorical)
    calibrated, calibration, iso = maybe_calibrate(y, oof)
    metrics = evaluate(df, y, calibrated, truth)
    holdout = country_holdout_check(df, y, feature_names, categorical, holdout_country, truth)
    ranked_importance = sorted(zip(feature_names, importances.tolist()), key=lambda kv: -kv[1])

    return {
        "n_pairs": len(y), "pos_rate": float(y.mean()), "n_folds": len(models),
        "features": feature_names, "categorical_features": categorical,
        "metrics": metrics, "calibration": calibration,
        "country_holdout": {"holdout_country": holdout_country, "leave_country_out": holdout,
                            "in_distribution": metrics["by_country"].get(holdout_country)},
        "feature_importance": ranked_importance,
        "models": models, "isotonic": iso, "oof_raw": oof, "oof_calibrated": calibrated, "y": y, "df": df,
    }

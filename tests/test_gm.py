import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from er.match import features, train


def test_mask_columns_includes_relative_views_and_patterns():
    feats = ["p", "gap_p", "gapsrc_p", "z_p", "p_rel", "score_B7_dense", "rank_B2_name_word", "rank_within_s1_src",
             "dense_cos", "z_dense_cos", "sim_name_tset", "ng_p_gap"]
    ranker = {feats[i] for i in features.mask_columns(feats, "ranker")}
    assert ranker == {"p", "gap_p", "gapsrc_p", "z_p", "p_rel", "score_B7_dense", "rank_B2_name_word", "ng_p_gap"}
    assert {feats[i] for i in features.mask_columns(feats, "dense")} == {"dense_cos", "z_dense_cos"}
    with pytest.raises(ValueError):
        features.mask_columns(feats, "nope")


def test_group_mask_is_per_s1_and_group_only():
    n_s1, per = 2000, 3
    s1 = pl.Series("s1_id", [f"S1-{i}" for i in range(n_s1) for _ in range(per)])
    feats = ["p", "z_p", "sim_name_tset", "dense_cos"]
    X = np.ones((s1.len(), len(feats)), dtype=np.float32)
    train.GroupMask(("ranker",), 0.25).apply(X, feats, s1, fold=0)

    masked = np.isnan(X[:, 0])
    assert (masked == np.isnan(X[:, 1])).all()  # p and its view go together
    assert not np.isnan(X[:, 2:]).any()  # other groups untouched
    by_s1 = masked.reshape(n_s1, per)
    assert (by_s1.all(axis=1) | ~by_s1.any(axis=1)).all()  # all rows of one S1 share the decision
    assert 0.2 < by_s1[:, 0].mean() < 0.3

    X2 = np.ones_like(X)
    train.GroupMask(("ranker",), 0.25).apply(X2, feats, s1, fold=0)
    assert (np.isnan(X2) == np.isnan(X)).all()  # deterministic
    X3 = np.ones_like(X)
    train.GroupMask(("ranker",), 0.25).apply(X3, feats, s1, fold=1)
    assert (np.isnan(X3[:, 0]) != masked).any()  # folds draw different S1


def _signal_frame(n_s1: int = 300, per: int = 4, seed: int = 0) -> tuple[pl.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = n_s1 * per
    y = (rng.random(n) < 0.3).astype(int)
    df = pl.DataFrame({
        "s1_id": [f"S1-{i // per}" for i in range(n)],
        "cv_fold": [(i // per) % 5 for i in range(n)],
        "p": (y + rng.normal(0, 0.3, n)).astype(np.float32),  # all signal lives in the ranker group
        "sim_name_tset": rng.random(n).astype(np.float32),
    })
    return df, y


def test_cv_train_masks_training_rows_only():
    df, y = _signal_frame()
    feats = ["p", "sim_name_tset"]
    params = {"device": "cpu", "max_depth": 3}
    X = train.matrix(df, feats)
    oof, _, _ = train.cv_train_xgb(X, y, df["cv_fold"].to_numpy(), params, num_boost_round=30,
                                   early_stopping_rounds=10)
    assert roc_auc_score(y, oof) > 0.9

    blind = train.GroupMask(("ranker",), 1.0)  # training never sees p
    oof_m, _, _ = train.cv_train_xgb(X, y, df["cv_fold"].to_numpy(), params, num_boost_round=30,
                                     early_stopping_rounds=10, mask=blind, feats=feats, s1_ids=df["s1_id"])
    assert roc_auc_score(y, oof_m) < 0.7
    assert not np.isnan(X).any()  # the caller's matrix (and so the OOF rows) is never masked


def test_mask_requires_xgboost_backend():
    df, y = _signal_frame(50)
    with pytest.raises(ValueError):
        train.cv_train(df, y, "cv_fold", ["p"], [], backend="lgbm", mask=train.GroupMask(("ranker",), 0.1))

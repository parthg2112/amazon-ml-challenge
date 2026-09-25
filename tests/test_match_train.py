import numpy as np
import polars as pl

from er.match import features, train

RNG = np.random.default_rng(0)
N = 400
SMALL_PARAMS = {"min_data_in_leaf": 5, "num_leaves": 7, "verbosity": -1}


def _synthetic_df() -> tuple[pl.DataFrame, np.ndarray]:
    y = (RNG.random(N) < 0.3).astype(int)
    signal = y * RNG.uniform(60, 100, N) + (1 - y) * RNG.uniform(0, 60, N)
    data = {c: RNG.uniform(0, 1, N).astype(np.float32) for c in features.NUMERIC_FEATURES}
    data["sim_name_tset"] = signal.astype(np.float32)
    for c in features.CATEGORICAL_FEATURES:
        data[c] = RNG.integers(0, 4, N).astype(np.int8)
    s1_fold = RNG.integers(0, 5, 80).astype(np.int8)  # one fold per S1 (grouped, like the real split)
    s1_country = RNG.choice(["US", "India"], 80)
    data["s1_id"] = [f"S1-{i % 80}" for i in range(N)]
    data["target_id"] = [f"{'S2' if i % 2 == 0 else 'S3'}-{i}" for i in range(N)]
    data["src"] = ["S2" if i % 2 == 0 else "S3" for i in range(N)]
    data["q_country"] = [s1_country[i % 80] for i in range(N)]
    data["cv_fold"] = [s1_fold[i % 80] for i in range(N)]
    return pl.DataFrame(data), y


def test_cv_train_produces_oof_for_every_row():
    df, y = _synthetic_df()
    oof, models, importances = train.cv_train(
        df, y, "cv_fold", features.NUMERIC_FEATURES, features.CATEGORICAL_FEATURES,
        params=SMALL_PARAMS, num_boost_round=20, early_stopping_rounds=5,
    )
    assert oof.shape == (N,)
    assert ((oof >= 0) & (oof <= 1)).all()
    assert len(models) == 5
    assert importances.shape == (len(features.NUMERIC_FEATURES),)


def test_maybe_calibrate_roundtrip():
    y = (RNG.random(300) < 0.4).astype(int)
    oof = np.clip(y * 0.9 + RNG.normal(0, 0.3, 300), 0, 1).astype(np.float32)
    calibrated, report, iso = train.maybe_calibrate(y, oof, gap_threshold=1e-9)  # force calibration
    assert report["applied"] is True
    assert iso is not None
    assert calibrated.shape == oof.shape


def test_fit_and_report_end_to_end():
    df, y = _synthetic_df()
    truth = pl.DataFrame({
        "s1_id": sorted(set(df["s1_id"].to_list())),
    }).with_columns(pl.lit([]).cast(pl.List(pl.Utf8)).alias("truth"))
    # seed a handful of true links matching positive rows so macro_f05 has signal
    pos = df.filter(pl.Series(y == 1)).select("s1_id", "target_id")
    truth_map = {s1: [] for s1 in truth["s1_id"].to_list()}
    for s1, t in pos.iter_rows():
        truth_map[s1].append(t)
    truth = pl.DataFrame({"s1_id": list(truth_map), "truth": list(truth_map.values())})
    folds = df.select("s1_id", "cv_fold").unique()

    result = train.fit_and_report(df, [], truth, folds, holdout_country="India")
    assert 0 <= result["metrics"]["macro_f05_at_threshold"] <= 1
    assert "US" in result["metrics"]["by_country"] or "India" in result["metrics"]["by_country"]
    assert result["country_holdout"]["holdout_country"] == "India"
    assert len(result["feature_importance"]) == len(features.NUMERIC_FEATURES)

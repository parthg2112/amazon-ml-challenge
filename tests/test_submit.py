"""Exercises er.submit's actual wiring (classifier fit -> calibration -> Step 5 policy -> test scoring ->
matching_results.tsv) with the two expensive blocking/feature-collection calls swapped for synthetic data,
so this runs in seconds instead of needing real normalized parquet on disk."""

import numpy as np
import polars as pl

from er import submit
from er.match import features

RNG = np.random.default_rng(1)


def _fake_train_df(n=500, n_s1=80):
    y = (RNG.random(n) < 0.3).astype(int)
    signal = y * RNG.uniform(60, 100, n) + (1 - y) * RNG.uniform(0, 60, n)
    data = {c: RNG.uniform(0, 1, n).astype(np.float32) for c in features.NUMERIC_FEATURES}
    data["sim_name_tset"] = signal.astype(np.float32)
    for c in features.CATEGORICAL_FEATURES:
        data[c] = RNG.integers(0, 4, n).astype(np.int8)
    s1_country = RNG.choice(["US", "India"], n_s1)
    data["s1_id"] = [f"S1-{i % n_s1}" for i in range(n)]
    data["target_id"] = [f"{'S2' if i % 2 == 0 else 'S3'}-{i}" for i in range(n)]
    data["src"] = ["S2" if i % 2 == 0 else "S3" for i in range(n)]
    data["q_country"] = [s1_country[i % n_s1] for i in range(n)]
    return pl.DataFrame(data), y


def _fake_test_df(n=120, n_s1=40, n_empty=5, offset=1000):
    data = {c: RNG.uniform(0, 1, n).astype(np.float32) for c in features.NUMERIC_FEATURES}
    for c in features.CATEGORICAL_FEATURES:
        data[c] = RNG.integers(0, 4, n).astype(np.int8)
    s1_country = RNG.choice(["US", "India"], n_s1)
    data["s1_id"] = [f"T1-{i % n_s1}" for i in range(n)]
    data["target_id"] = [f"{'S2' if i % 2 == 0 else 'S3'}-{offset + i}" for i in range(n)]
    data["src"] = ["S2" if i % 2 == 0 else "S3" for i in range(n)]
    data["q_country"] = [s1_country[i % n_s1] for i in range(n)]
    all_ids = pl.Series("entity_id", [f"T1-{i}" for i in range(n_s1 + n_empty)])  # last n_empty: no candidates
    return pl.DataFrame(data), all_ids


def test_submit_writes_matching_results_for_every_test_s1(tmp_path, monkeypatch):
    train_df, y = _fake_train_df()
    test_df, test_ids = _fake_test_df()

    truth_map = {s1: [] for s1 in train_df["s1_id"].unique().to_list()}
    pos = train_df.filter(pl.Series(y == 1)).select("s1_id", "target_id")
    for s1, t in pos.iter_rows():
        truth_map[s1].append(t)
    truth = pl.DataFrame({"s1_id": list(truth_map), "truth": list(truth_map.values())})
    folds = pl.DataFrame({
        "s1_id": list(truth_map),
        "cv_fold": RNG.integers(0, 5, len(truth_map)).astype(np.int8),
    })

    monkeypatch.setattr(submit, "collect_match_features", lambda config_path, subset, k: (train_df, []))
    monkeypatch.setattr(submit, "collect_match_features_test", lambda config_path, k: (test_df, [], test_ids))
    monkeypatch.setattr(submit.io, "load_truth", lambda: truth)
    monkeypatch.setattr(submit, "load_split", lambda: folds)
    monkeypatch.setattr(submit.io, "ROOT", tmp_path)
    monkeypatch.setattr(submit.io, "REPORTS", tmp_path / "reports")
    (tmp_path / "reports").mkdir()

    stats = submit.run("fake_config.toml", k=10, match_subset="dev", holdout_country="India")

    out = tmp_path / "output" / "matching_results.tsv"
    assert out.exists()
    result = pl.read_csv(out, separator="\t", infer_schema=False)
    assert set(result.columns) == {"source1_entity_id", "matched_entity_ids"}
    assert result.height == test_ids.len()  # every test S1 present, including zero-candidate ones
    assert set(result["source1_entity_id"]) == set(test_ids.to_list())
    # the 5 S1s with no scored candidates at all must still get an (empty) row
    no_cand_ids = set(test_ids.to_list()) - set(test_df["s1_id"].unique().to_list())
    empties = result.filter(pl.col("source1_entity_id").is_in(no_cand_ids))
    assert (empties["matched_entity_ids"] == "").all()

    assert stats["test_s1"] == test_ids.len()
    assert stats["decision_winner"] in ("tuned_global_threshold", "per_entity_expected_f05")
    assert stats["exclusivity_winner"] in ("max_probability", "scarcity_weighted")
    assert 0 <= stats["oof_macro_f05"] <= 1

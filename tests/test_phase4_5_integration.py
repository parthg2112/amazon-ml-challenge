"""The actual seam between Step 4 and Step 5: er.match.report.write_oof's parquet is what er.decide reads."""

import polars as pl
from test_match_train import _synthetic_df

from er.decide import pipeline
from er.match import report, train


def test_write_oof_feeds_decide_pipeline(tmp_path):
    df, y = _synthetic_df()
    truth = pl.DataFrame({"s1_id": sorted(set(df["s1_id"].to_list()))}).with_columns(
        pl.lit([]).cast(pl.List(pl.Utf8)).alias("truth"))
    pos = df.filter(pl.Series(y == 1)).select("s1_id", "target_id")
    truth_map = {s1: [] for s1 in truth["s1_id"].to_list()}
    for s1, t in pos.iter_rows():
        truth_map[s1].append(t)
    truth = pl.DataFrame({"s1_id": list(truth_map), "truth": list(truth_map.values())})
    folds = df.select("s1_id", "cv_fold").unique()

    result = train.fit_and_report(df, [], truth, folds, holdout_country="India")
    oof_path = report.write_oof(result, tmp_path / "oof.parquet")
    assert oof_path.exists()

    oof = pl.read_parquet(oof_path)
    assert {"s1_id", "target_id", "src", "q_country", "y", "p_raw", "p"} <= set(oof.columns)
    assert oof.height == len(y)

    decided = pipeline.run(oof, truth)
    assert 0 <= decided["final"]["macro_f05"] <= 1
    assert decided["n_s1"] == oof["s1_id"].n_unique()

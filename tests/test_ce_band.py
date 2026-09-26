import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from er.match import ce_band


def test_apply_rescoring_touches_only_band_pairs():
    scored = pl.DataFrame({"s1_id": ["a", "a", "b"], "target_id": ["x", "y", "z"], "p": [0.95, 0.5, 0.1]},
                          schema_overrides={"p": pl.Float32})
    ce = pl.DataFrame({"s1_id": ["a"], "target_id": ["y"], "ce": [4.0]}, schema_overrides={"ce": pl.Float32})
    X = np.array([[0.0, -1.0], [0.0, 1.0], [2.0, 1.0], [-2.0, -1.0]])
    model = LogisticRegression().fit(X, [0, 1, 1, 0])
    out = ce_band.apply(scored, ce, model).sort("target_id")
    assert out["p"].to_list()[0] == scored["p"][0] and out["p"].to_list()[2] == scored["p"][2]
    assert out["p"][1] != scored["p"][1]


def test_apply_raises_when_band_pair_lacks_ce():
    scored = pl.DataFrame({"s1_id": ["a"], "target_id": ["y"], "p": [0.5]})
    ce = pl.DataFrame(schema={"s1_id": pl.String, "target_id": pl.String, "ce": pl.Float32})
    model = LogisticRegression().fit(np.array([[0.0, -1.0], [0.0, 1.0]]), [0, 1])
    try:
        ce_band.apply(scored, ce, model)
    except ValueError:
        return
    raise AssertionError("expected ValueError")

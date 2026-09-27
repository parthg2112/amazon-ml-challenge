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


def test_contest_features_prefer_unique_street_word_winner():
    # A tiny France-like contest: names and house number can collide, but only `louis barthou`
    # is the target's street.  The runner-up remains visible as a sibling instead of being forced
    # to own the target, preserving the downstream "none" option.
    claims = pl.DataFrame({
        "s1_id": ["S1-owner", "S1-sibling", "S1-other"],
        "target_id": ["S3-target"] * 3,
        "q_addr_street": ["louis barthou", "saint joseph", "leveux"],
        "t_addr_street": ["louis barthou"] * 3,
        "q_name_min_key": ["pole", "pole", "another"],
    })
    out = ce_band._contest_from_claims(claims).sort("s1_id")
    rows = {r["s1_id"]: r for r in out.to_dicts()}
    assert rows["S1-owner"]["contest_street_margin"] > 0.9
    assert rows["S1-owner"]["contest_best_street"] == 1.0
    assert rows["S1-sibling"]["contest_street_margin"] < 0
    assert rows["S1-sibling"]["contest_best_street"] == 0.0
    assert rows["S1-owner"]["contest_same_name_rivals"] > 0
    assert rows["S1-other"]["contest_same_name_rivals"] == 0

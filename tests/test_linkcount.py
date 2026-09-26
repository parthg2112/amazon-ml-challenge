import numpy as np
import polars as pl
import pytest

from er.decide import exact_f05, linkcount


def _truth(rows: list[list[str]]) -> pl.DataFrame:
    return pl.DataFrame({"s1_id": [f"S1-{i}" for i in range(len(rows))], "truth": rows},
                        schema={"s1_id": pl.Utf8, "truth": pl.List(pl.Utf8)})


def test_fit_prior_counts_and_caps():
    prior = linkcount.fit_prior(_truth([
        [], ["S2-1"], ["S2-1", "S3-1"], ["S2-1", "S2-2", "S3-1"], ["S3-1", "S3-2"],
    ]))
    assert prior.total[0] == pytest.approx(0.2)  # one singleton in five
    assert prior.mean_total == pytest.approx((0 + 1 + 2 + 3 + 2) / 5)
    assert prior.mean_src("S2") == pytest.approx((0 + 1 + 1 + 2 + 0) / 5)
    assert prior.caps == {"S2": 2, "S3": 2}


def test_prior_matched_threshold_hits_the_requested_rate():
    p = np.linspace(0.01, 0.99, 100)
    scored = pl.DataFrame({"s1_id": [f"S1-{i}" for i in range(100)],
                           "target_id": [f"S2-{i}" for i in range(100)], "p": p})
    t = linkcount.prior_matched_threshold(scored, n_s1=50, target_mean=1.0)  # want 50 predictions
    assert (scored["p"] >= t).sum() == 50


def test_prior_matched_threshold_uses_the_full_s1_denominator():
    """S1s with no candidates still count: the rate is per S1, not per scored S1."""
    scored = pl.DataFrame({"s1_id": ["S1-0"] * 10, "target_id": [f"S2-{i}" for i in range(10)],
                           "p": np.linspace(0.05, 0.95, 10)})
    dense = linkcount.prior_matched_threshold(scored, n_s1=10, target_mean=0.5)
    sparse = linkcount.prior_matched_threshold(scored, n_s1=2, target_mean=0.5)
    assert dense < sparse  # more S1 in the universe -> more predictions allowed -> lower cutoff


def test_apply_caps_enforces_per_source_ceiling():
    prior = linkcount.fit_prior(_truth([["S2-9"], ["S3-9", "S3-8"]]))  # caps S2=1, S3=2
    scored = pl.DataFrame({
        "s1_id": ["S1-0"] * 5, "src": ["S2", "S2", "S3", "S3", "S3"],
        "target_id": ["S2-1", "S2-2", "S3-1", "S3-2", "S3-3"], "p": [0.9, 0.8, 0.9, 0.8, 0.7]})
    out = linkcount.apply_caps(scored, prior)
    assert out.filter(pl.col("src") == "S2").height == 1
    assert out.filter(pl.col("src") == "S3").height == 2
    assert set(out["target_id"]) == {"S2-1", "S3-1", "S3-2"}


def test_select_by_expected_count_blends_toward_the_prior():
    prior = linkcount.fit_prior(_truth([["S2-1", "S2-2", "S3-1"]] * 5))  # mean 3.0
    scored = pl.DataFrame({"s1_id": ["S1-0"] * 5, "target_id": [f"S2-{i}" for i in range(5)],
                           "p": [0.9, 0.8, 0.2, 0.1, 0.05]})  # sum(p) = 2.05 -> k=2
    assert linkcount.select_by_expected_count(scored, prior, blend=0.0).height == 2
    assert linkcount.select_by_expected_count(scored, prior, blend=1.0).height == 3  # prior mean 3


def test_tilt_is_a_no_op_when_the_model_already_matches_the_prior():
    """The regression guard for the validated exact-F0.5 result: tilt=None must not change anything."""
    P = np.array([[0.9, 0.8, 0.3, 0.1], [0.6, 0.5, 0.4, 0.2]])
    base = exact_f05.expected_f05_matrix(P)
    ones = exact_f05.expected_f05_matrix(P, tilt=np.ones(P.shape[1] + 1))
    assert np.allclose(base, ones)
    implied = linkcount.implied_count_pmf(P)
    w = linkcount.tilt_weights(implied, implied)  # prior == implied -> weights all 1
    assert np.allclose(exact_f05.expected_f05_matrix(P, tilt=w), base)


def test_tilt_toward_more_links_selects_more():
    df = pl.DataFrame({"s1_id": ["S1-0"] * 4, "target_id": [f"S2-{i}" for i in range(4)],
                       "src": ["S2"] * 4, "p": [0.95, 0.6, 0.45, 0.4]})
    plain = exact_f05.select(df).height
    # a prior insisting on exactly 4 links must not select fewer than the independent rule
    many = linkcount.LinkPrior(total=np.array([0.0, 0.0, 0.0, 0.0, 1.0]), per_src={"S2": np.array([0.0, 1.0])})
    assert linkcount.select_tilted(df, many).height >= plain
    # a prior insisting on none must not select more
    none = linkcount.LinkPrior(total=np.array([1.0]), per_src={"S2": np.array([1.0])})
    assert linkcount.select_tilted(df, none).height <= plain


def test_real_truth_caps_match_the_documented_structure():
    """S2 0..5 and S3 0..6 are the ceilings the review quotes; guard them against silent drift."""
    prior = linkcount.fit_prior(_truth([["S2-1"] * 5 + ["S3-1"] * 6, []]))
    assert prior.caps == {"S2": 5, "S3": 6}


def test_match_expected_count_hits_target_and_is_noop_at_target():
    df = pl.DataFrame({"s1_id": ["a", "a", "b", "b"], "target_id": ["x", "y", "z", "w"],
                       "p": [0.9, 0.6, 0.8, 0.3]}, schema_overrides={"p": pl.Float32})
    cur = float(df["p"].sum()) / 2
    same = linkcount.match_expected_count(df, cur)
    assert np.allclose(same["p"].to_numpy(), df["p"].to_numpy(), atol=1e-4)
    up = linkcount.match_expected_count(df, cur + 0.3)
    assert float(up["p"].sum()) / 2 == pytest.approx(cur + 0.3, abs=1e-3)
    assert (np.argsort(up["p"].to_numpy()) == np.argsort(df["p"].to_numpy())).all()  # monotone


def test_select_tilted_single_group_equals_plain():
    prior = linkcount.LinkPrior(total=np.array([0.1, 0.3, 0.4, 0.2]), per_src={})
    df = pl.DataFrame({"s1_id": ["a", "a", "a", "b", "b"], "target_id": ["x", "y", "z", "u", "v"],
                       "p": [0.9, 0.5, 0.2, 0.7, 0.4], "country": ["C"] * 5})
    plain = linkcount.select_tilted(df.drop("country"), prior).select("s1_id", "target_id").sort("target_id")
    grouped = linkcount.select_tilted(df, prior, group_col="country").select("s1_id", "target_id").sort("target_id")
    assert plain.equals(grouped)


def test_match_expected_count_skips_groups_within_tolerance():
    df = pl.DataFrame({"s1_id": ["a", "a", "b", "b"], "target_id": ["x", "y", "z", "w"],
                       "p": [0.9, 0.6, 0.2, 0.1], "g": ["near", "near", "far", "far"]},
                      schema_overrides={"p": pl.Float32})
    out = linkcount.match_expected_count(df, 1.5, "g", tol=0.05)
    assert sorted(out.filter(pl.col("g") == "near")["p"].to_list()) == pytest.approx([0.6, 0.9], abs=1e-6)
    assert float(out.filter(pl.col("g") == "far")["p"].sum()) == pytest.approx(1.5, abs=1e-3)

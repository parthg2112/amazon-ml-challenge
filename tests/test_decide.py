import polars as pl
import pytest

from er.decide import evaluate, exclusivity, expected_f05, pipeline, threshold


def test_resolve_max_probability_keeps_highest():
    df = pl.DataFrame({
        "s1_id": ["S1-1", "S1-2"], "target_id": ["S2-1", "S2-1"], "src": ["S2", "S2"], "p": [0.9, 0.4],
    })
    out = exclusivity.resolve_max_probability(df)
    assert out.height == 1
    assert out["s1_id"][0] == "S1-1"


def test_resolve_scarcity_weighted_favors_sole_option():
    # S1-1 claims S2-1 with high raw probability but has 3 other strong candidates too.
    # S1-2 claims S2-1 with a lower raw probability but S2-1 is its ONLY candidate.
    df = pl.DataFrame({
        "s1_id": ["S1-1", "S1-1", "S1-1", "S1-2"],
        "target_id": ["S2-1", "S2-2", "S2-3", "S2-1"],
        "src": ["S2"] * 4,
        "p": [0.9, 0.85, 0.8, 0.5],
    })
    out = exclusivity.resolve_scarcity_weighted(df)
    winner = out.filter(pl.col("target_id") == "S2-1")
    assert winner["s1_id"][0] == "S1-2"  # 0.5/1 > 0.9/3
    # max-probability would have picked the other way
    assert exclusivity.resolve_max_probability(df).filter(
        pl.col("target_id") == "S2-1")["s1_id"][0] == "S1-1"


def test_conflict_stats():
    df = pl.DataFrame({
        "s1_id": ["S1-1", "S1-2", "S1-3"], "target_id": ["S2-1", "S2-1", "S2-2"],
        "src": ["S2"] * 3, "p": [0.9, 0.4, 0.7],
    })
    stats = exclusivity.conflict_stats(df)
    assert stats == {"targets_claimed": 2, "contested_targets": 1, "max_claimants": 2}


def test_choose_k_picks_best_prefix():
    df = pl.DataFrame({
        "s1_id": ["S1-1"] * 3, "target_id": ["S2-1", "S2-2", "S2-3"], "src": ["S2"] * 3,
        "p": [0.9, 0.8, 0.1],
    })
    chosen = expected_f05.choose_k(df)
    row = chosen.row(0, named=True)
    assert row["chosen_k"] == 2
    assert row["expected_f05"] == pytest.approx(1.25 * 1.7 / (2 + 0.25 * 1.8), abs=1e-6)


def test_choose_k_predicts_empty_when_all_weak():
    df = pl.DataFrame({
        "s1_id": ["S1-1"] * 2, "target_id": ["S2-1", "S2-2"], "src": ["S2"] * 2, "p": [0.02, 0.01],
    })
    row = expected_f05.choose_k(df).row(0, named=True)
    assert row["chosen_k"] == 0
    assert row["expected_f05"] == pytest.approx(0.98 * 0.99, abs=1e-6)


def test_select_returns_exactly_chosen_k_rows():
    df = pl.DataFrame({
        "s1_id": ["S1-1"] * 3 + ["S1-2"] * 2,
        "target_id": ["S2-1", "S2-2", "S2-3", "S3-1", "S3-2"],
        "src": ["S2", "S2", "S2", "S3", "S3"],
        "p": [0.9, 0.8, 0.1, 0.02, 0.01],
    })
    sel = expected_f05.select(df)
    assert set(sel.filter(pl.col("s1_id") == "S1-1")["target_id"]) == {"S2-1", "S2-2"}
    assert sel.filter(pl.col("s1_id") == "S1-2").height == 0


def test_threshold_tune_and_predict():
    df = pl.DataFrame({
        "s1_id": ["S1-1", "S1-1", "S1-2"], "target_id": ["S2-1", "S2-2", "S3-1"],
        "p": [0.9, 0.2, 0.9],
    })
    truth = pl.DataFrame({"s1_id": ["S1-1", "S1-2"], "truth": [["S2-1"], ["S3-1"]]})
    best_t, best_f05, curve = threshold.tune(df, truth, grid=[0.1, 0.5, 0.95])
    assert best_t in (0.1, 0.5)  # both catch the two true links; 0.95 misses everything
    assert best_f05 == pytest.approx(1.0, abs=1e-6)
    assert len(curve) == 3


def test_pipeline_runs_end_to_end():
    df = pl.DataFrame({
        "s1_id": ["S1-1"] * 3 + ["S1-2"] * 2 + ["S1-3"],
        "target_id": ["S2-1", "S2-2", "S2-3", "S3-1", "S3-2", "S2-4"],
        "src": ["S2", "S2", "S2", "S3", "S3", "S2"],
        "p": [0.9, 0.8, 0.1, 0.85, 0.3, 0.05],
        "q_country": ["US"] * 5 + ["India"],
    })
    truth = pl.DataFrame({
        "s1_id": ["S1-1", "S1-2", "S1-3"], "truth": [["S2-1", "S2-2"], ["S3-1"], []],
    })
    result = pipeline.run(df, truth)
    assert result["n_s1"] == 3
    assert 0 <= result["final"]["macro_f05"] <= 1
    assert result["decision_winner"] in ("tuned_global_threshold", "per_entity_expected_f05")
    assert result["exclusivity_winner"] in ("max_probability", "scarcity_weighted")
    assert "3_final" in result["stages"]
    assert len(result["pr_curve"]) > 0


def test_singleton_accuracy():
    truth = pl.DataFrame({"s1_id": ["S1-1", "S1-2"], "truth": [[], ["S2-1"]]})
    preds_correct_empty = pl.DataFrame({"s1_id": ["S1-2"], "target_id": ["S2-1"]})
    assert evaluate.singleton_accuracy(preds_correct_empty, truth) == 1.0
    preds_wrong = pl.DataFrame({"s1_id": ["S1-1"], "target_id": ["S2-9"]})
    assert evaluate.singleton_accuracy(preds_wrong, truth) == 0.0


def test_exact_expected_f05_matches_brute_force():
    import itertools

    import numpy as np

    from er.decide.exact_f05 import expected_f05_matrix

    p = np.array([0.9, 0.6, 0.3, 0.05])
    got = expected_f05_matrix(p[None, :])[0]
    for k in range(len(p) + 1):
        exp = 0.0
        for ys in itertools.product([0, 1], repeat=len(p)):
            prob = np.prod([pi if y else 1 - pi for pi, y in zip(p, ys)])
            t, tp = sum(ys), sum(ys[:k])
            f = 1.0 if (k == 0 and t == 0) else (1.25 * tp / (k + 0.25 * t) if (k + t) else 0.0)
            exp += prob * f
        assert got[k] == pytest.approx(exp), k


def test_exact_select_predicts_nothing_for_weak_candidates():
    from er.decide.exact_f05 import select

    df = pl.DataFrame({"s1_id": ["S1-1"] * 3 + ["S1-2"] * 2, "target_id": ["a", "b", "c", "d", "e"],
                       "src": ["S2"] * 5, "p": [0.95, 0.9, 0.2, 0.05, 0.02]})
    out = select(df)
    assert set(out.filter(pl.col("s1_id") == "S1-1")["target_id"]) == {"a", "b"}
    assert out.filter(pl.col("s1_id") == "S1-2").height == 0  # singleton: empty prediction wins


def test_renormalize_keeps_uncontested_and_shrinks_contested():
    df = pl.DataFrame({
        "s1_id": ["S1-1", "S1-2", "S1-3"], "target_id": ["S2-1", "S2-2", "S2-2"], "src": ["S2"] * 3,
        "p": [0.8, 0.9, 0.9],
    })
    out = exclusivity.renormalize(df)
    assert out["p"][0] == pytest.approx(0.8, abs=1e-6)
    assert out["p"][1] < 0.5 and out["p"][2] < 0.5  # 9 / (1 + 18) = 0.474
    assert out["p"][1] == pytest.approx(9 / 19, abs=1e-6)

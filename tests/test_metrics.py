import polars as pl
import pytest

from er.metrics import blocking_metrics, macro_f05


def _truth(rows):
    return pl.DataFrame({"s1_id": [r[0] for r in rows], "truth": [r[1] for r in rows]},
                        schema={"s1_id": pl.Utf8, "truth": pl.List(pl.Utf8)})


def _pairs(rows):
    return pl.DataFrame({"s1_id": [a for a, _ in rows], "target_id": [b for _, b in rows]},
                        schema={"s1_id": pl.Utf8, "target_id": pl.Utf8})


@pytest.mark.parametrize(
    "truth,pred,expected",
    [
        ([], [], 1.0),
        ([], ["S2-1"], 0.0),
        (["S2-1"], [], 0.0),
        (["S2-1"], ["S2-1"], 1.0),
        (["S2-1", "S3-1"], ["S2-1"], 1.25 / (1 + 0.5)),
        (["S2-1"], ["S2-1", "S2-9"], 1.25 / (2 + 0.25)),
        (["S2-47", "S3-812"], ["S2-47", "S2-193", "S3-812"], 2.5 / 3.5),  # PROBLEM_STATEMENT worked example
    ],
)
def test_row_cases(truth, pred, expected):
    t = _truth([("S1-1", truth)])
    p = _pairs([("S1-1", x) for x in pred])
    assert macro_f05(t, p) == pytest.approx(expected)


def test_macro_counts_missing_rows_as_empty():
    t = _truth([("S1-1", []), ("S1-2", ["S2-1"])])
    p = _pairs([("S1-2", "S2-1"), ("S1-2", "S2-1")])  # duplicate prediction collapses
    assert macro_f05(t, p) == pytest.approx(1.0)


def test_blocking_oracle():
    t = _truth([("S1-1", []), ("S1-2", ["S2-1", "S3-1"])])
    c = _pairs([("S1-1", "S2-5"), ("S1-2", "S2-1"), ("S1-2", "S2-7")])
    m = blocking_metrics(t, c, n_targets=100)
    # singleton oracle = 1 (classifier may return nothing); S1-2 oracle = 1.25/(1+0.5)
    assert m["oracle_ceiling_f05"] == pytest.approx((1 + 1.25 / 1.5) / 2)
    assert m["pair_recall"] == pytest.approx(0.5)
    assert m["full_coverage"] == pytest.approx(0.0)
    assert m["pair_recall_by_source"] == {"S2": 1.0, "S3": 0.0}

import polars as pl
import pytest

from er.match.contest import contest_from_claims, universe_claims
from er.normalize.address import normalize_address, street_line
from er.normalize.pipeline import NormConfig
from er.normalize.profiles import profile_for

FR, US, IN = profile_for("France"), profile_for("US"), profile_for("India")


def test_street_line_picks_the_house_segment_not_the_first():
    assert street_line("Bordeaux, Nouvelle-Aquitaine, 3 AV Louis Barthou", FR) == "louis barthou"
    assert street_line("Loire-Atlantique, St-herblain, 6 Allée Lakanal", FR) == "lakanal"
    assert street_line("1100 Gilbert Avenue, Belen, NM", US) == "gilbert"


def test_street_line_keeps_saint_and_falls_back_to_street_type_segment():
    assert street_line("3 Rue Saint-Joseph, Bordeaux, Nouvelle-Aquitaine", FR) == "saint joseph"
    assert street_line("Maharashtra, Unnati Shahaji Raje Marg, Villeparle (E), 17", IN) == "unnati shahaji raje"
    assert street_line("Mumbai, MH", IN) == ""
    assert street_line("", FR) == ""


def test_scorer_normalization_is_unchanged():
    # contest features must not move the scorer's fields or the cache variant (review item 4)
    assert normalize_address("3 Rue Saint-Joseph, Bordeaux, Nouvelle-Aquitaine", FR)["addr_content"] == "joseph bordeaux"
    assert "addr_street" not in normalize_address("3 Rue X, Lille", FR)
    assert NormConfig(learned="configs/learned/n4_v1.json").variant == "t1-auto-n4_v1"


def test_contest_features_pick_the_best_address_among_rivals():
    # Pole example: same name key and house; only the owner shares the target's street. Plus an uncontested target.
    claims = pl.DataFrame({
        "s1_id": ["S1-owner", "S1-sibling", "S1-other", "S1-alone"],
        "target_id": ["S3-target"] * 3 + ["S3-solo"],
        "q_addr_content": ["bordeaux louis barthou", "joseph bordeaux", "leveux calais", "lille nationale"],
        "t_addr_content": ["louis bartou bordeaux"] * 3 + ["lille nationale"],
        "q_addr_street": ["louis barthou", "saint joseph", "leveux", "nationale"],
        "t_addr_street": ["louis bartou"] * 3 + ["nationale"],
        "q_name_min_key": ["pole", "pole", "", "x"],
    })
    rows = {r["s1_id"]: r for r in contest_from_claims(claims).to_dicts()}
    assert rows["S1-owner"]["contest_addr_margin"] > 0.2 and rows["S1-owner"]["contest_best_addr"] == 1.0
    assert rows["S1-owner"]["contest_street_margin"] > 0.5
    assert rows["S1-sibling"]["contest_addr_margin"] < 0 and rows["S1-sibling"]["contest_best_addr"] == 0.0
    assert rows["S1-owner"]["contest_addr_margin"] == pytest.approx(-rows["S1-sibling"]["contest_addr_margin"])
    assert rows["S1-owner"]["contest_same_name_rivals"] > 0
    assert rows["S1-other"]["contest_same_name_rivals"] == 0  # empty name key never counts as a same-name rival
    assert rows["S1-alone"]["contest_addr_margin"] == 0.0 and rows["S1-alone"]["contest_street_margin"] == 0.0


def test_contest_ties_have_no_unique_best():
    claims = pl.DataFrame({"s1_id": ["a", "b"], "target_id": ["t", "t"], "q_addr_content": ["x y", "x y"],
                           "t_addr_content": ["x y", "x y"], "q_addr_street": ["", ""], "t_addr_street": ["", ""],
                           "q_name_min_key": ["k", "k"]})
    out = contest_from_claims(claims)
    assert out["contest_best_addr"].to_list() == [0.0, 0.0] and out["contest_addr_margin"].to_list() == [0.0, 0.0]


def test_contest_needs_full_universe_on_train():
    with pytest.raises(ValueError):
        universe_claims(pl.Series(["S2-1"]), "configs/blocking/b_v9.toml", "train", 10, "val")

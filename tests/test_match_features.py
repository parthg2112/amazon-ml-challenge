import pytest
import polars as pl

from er.match.features import CATEGORICAL_FEATURES, JOIN_COLS, NUMERIC_FEATURES, build
from er.match.idf import build_word_idf, token_overlap_features
from er.normalize.pipeline import NormConfig, normalize_record

CFG = NormConfig()


def norm_df(records: list[tuple[str, str, str, str]]) -> pl.DataFrame:
    """records: (entity_id, name, address, country)."""
    rows = []
    for entity_id, name, addr, country in records:
        rec = normalize_record(name, addr, country, CFG)
        rows.append({"entity_id": entity_id, "country": country, **{k: rec[k] for k in JOIN_COLS if k != "country"}})
    return pl.DataFrame(rows)


def test_build_features_core_signals():
    queries = norm_df([
        ("S1-1", "Hines Liberty Produce LLC", "123 Main St, NY 10001", "US"),
        ("S1-2", "Acme Traders aka Best Traders", "45 Park Rd", "US"),
    ])
    targets = norm_df([
        ("S2-1", "L.L.C. Hines Liberty Produce", "123 Main Street, NY 10001", "US"),   # exact-ish match
        ("S2-2", "Hines Liberty Produce PLC", "999 Other St, NY 10099", "US"),          # legal conflict
        ("S3-1", "Best Traders Inc", "45 Park Road", "US"),                            # DBA alias match
    ])
    pairs = pl.DataFrame({
        "s1_id": ["S1-1", "S1-1", "S1-2"],
        "target_id": ["S2-1", "S2-2", "S3-1"],
    })
    name_idf = build_word_idf(targets, "name_core")
    addr_idf = build_word_idf(targets, "addr_content")
    df = build(pairs, queries, targets, name_idf, addr_idf).sort("s1_id", "target_id")

    for col in NUMERIC_FEATURES:
        assert col in df.columns, col
    for col in CATEGORICAL_FEATURES:
        assert df[col].dtype == pl.Int8

    row1 = df.filter((pl.col("s1_id") == "S1-1") & (pl.col("target_id") == "S2-1")).row(0, named=True)
    assert row1["name_core_exact"] == 1
    assert row1["house_state"] == 1  # both "123"
    assert row1["legal_relation"] == 1  # both llc

    row2 = df.filter((pl.col("s1_id") == "S1-1") & (pl.col("target_id") == "S2-2")).row(0, named=True)
    assert row2["legal_relation"] == 3  # llc vs plc: conflicting
    assert row2["house_state"] == 2  # 123 vs 999: conflict

    row3 = df.filter((pl.col("s1_id") == "S1-2") & (pl.col("target_id") == "S3-1")).row(0, named=True)
    assert row3["alias_match"] == 1  # "best traders" alt matches target core


def test_context_features_reflect_competition():
    queries = norm_df([
        ("S1-1", "Acme Tools", "1 Main St", "US"),
        ("S1-2", "Acme Bolts", "1 Main St", "US"),
    ])
    targets = norm_df([("S2-1", "Acme Tools", "1 Main St", "US")])
    pairs = pl.DataFrame({"s1_id": ["S1-1", "S1-2"], "target_id": ["S2-1", "S2-1"]})
    name_idf = build_word_idf(targets, "name_core")
    addr_idf = build_word_idf(targets, "addr_content")
    df = build(pairs, queries, targets, name_idf, addr_idf)
    assert set(df["n_s1_claiming_record"].to_list()) == {2}
    # both claim the one target; margins should be roughly opposite in sign or one strictly better
    margins = dict(zip(df["s1_id"], df["margin_to_best_competitor"]))
    assert margins["S1-1"] >= margins["S1-2"]  # exact name match should win the competition


def test_token_overlap_features_counts():
    df = pl.DataFrame({"a": ["hines liberty produce"], "b": ["hines liberty imports"]})
    idf = build_word_idf(pl.DataFrame({"name_core": ["hines liberty imports", "other company", "hines corp"]}),
                         "name_core")
    out = token_overlap_features(df, "a", "b", idf, "tok")
    assert out["tok_overlap_n"][0] == 2  # hines, liberty
    assert out["tok_extra_q_n"][0] == 1  # produce
    assert out["tok_extra_t_n"][0] == 1  # imports


def test_chunked_local_plus_global_context_equals_single_universe():
    from er.match.features import CTX_SCORE, build_local, global_context

    queries = norm_df([
        ("S1-1", "Acme Tools", "1 Main St", "US"),
        ("S1-2", "Acme Bolts", "1 Main St", "US"),
        ("S1-3", "Zenith Foods", "9 Oak Ave", "US"),
    ])
    targets = norm_df([("S2-1", "Acme Tools", "1 Main St", "US"), ("S3-1", "Zenith Foods Inc", "9 Oak Ave", "US")])
    pairs = pl.DataFrame({"s1_id": ["S1-1", "S1-2", "S1-3", "S1-2"], "target_id": ["S2-1", "S2-1", "S3-1", "S3-1"]})
    name_idf = build_word_idf(targets, "name_core")
    addr_idf = build_word_idf(targets, "addr_content")
    whole = build(pairs, queries, targets, name_idf, addr_idf).sort("s1_id", "target_id")

    # competitors S1-1/S1-2 land in different chunks: per-chunk context would miss the competition
    chunks = [pairs.filter(pl.col("s1_id") == "S1-1"), pairs.filter(pl.col("s1_id") != "S1-1")]
    local = pl.concat([build_local(c, queries, targets, name_idf, addr_idf) for c in chunks])
    ctx = global_context(local.select("s1_id", "target_id", CTX_SCORE))
    chunked = local.join(ctx, on=["s1_id", "target_id"]).drop(CTX_SCORE).sort("s1_id", "target_id")

    for col in ("n_s1_claiming_record", "margin_to_best_competitor", "rank_within_s1_src", "sim_min_tset"):
        assert chunked[col].to_list() == whole[col].to_list(), col
    assert whole.filter(pl.col("target_id") == "S2-1")["n_s1_claiming_record"].to_list() == [2, 2]


def test_noise_flags():
    queries = norm_df([("S1-1", "Harbor Services LLC", "24221 Kittson Blvd", "US")])
    targets = norm_df([
        ("S2-1", "Harbor Foundation Services", "2422 KITTSON BLVD", "US"),  # digit dropped, noise-only name
        ("S2-2", "Harbor LLC", "24251 Kittson Blvd", "US"),                  # digit changed
    ])
    pairs = pl.DataFrame({"s1_id": ["S1-1", "S1-1"], "target_id": ["S2-1", "S2-2"]})
    df = build(pairs, queries, targets, build_word_idf(targets, "name_core"),
               build_word_idf(targets, "addr_content")).sort("target_id")
    r1, r2 = df.row(0, named=True), df.row(1, named=True)
    assert r1["house_digit_dropped"] == 1 and r1["house_digit_changed"] == 0
    assert r1["name_min_exact"] == 1 and r1["name_noise_only_diff"] == 1
    assert r2["house_digit_changed"] == 1 and r2["house_digit_dropped"] == 0


def test_add_relative_gaps_zero_at_best():
    from er.match.features import REL_BASE, add_relative

    df = pl.DataFrame({"s1_id": ["a", "a", "b"], "src": ["S2", "S3", "S2"]}).with_columns(
        [pl.Series(c, [0.9, 0.4, 0.5]) for c in REL_BASE])
    out = add_relative(df)
    assert out["gap_p"].to_list() == pytest.approx([0.0, 0.5, 0.0])
    assert out["gapsrc_p"].to_list() == [0.0, 0.0, 0.0]
    assert out["z_p"][2] == 0.0

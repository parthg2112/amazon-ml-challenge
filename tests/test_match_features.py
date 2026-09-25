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

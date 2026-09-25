import polars as pl

from er.blocking.channels.exact import exact_channel
from er.blocking.channels.sparse import FieldSpec, _blocks, _search, hash_texts, idf_weights, weighted
from er.blocking.fuse import add_ranks, fuse


def test_exact_channel_caps_blocks():
    q = pl.DataFrame({"entity_id": ["S1-1", "S1-2"], "k": ["rare", "common"]})
    t = pl.DataFrame({"entity_id": ["S2-1", "S2-2", "S2-3", "S2-4"], "k": ["rare", "common", "common", "common"]})
    pairs, stats = exact_channel(q, t, "k", "k", max_block=2)
    assert pairs.select("s1_id", "target_id").rows() == [("S1-1", "S2-1")]
    assert stats["keys_over_cap"] == 1


def test_partitioned_search_respects_region_and_unknown():
    f = FieldSpec(col="t")
    texts_t = ["acme tools ny", "acme tools ca", "acme tools"]
    tm = hash_texts(texts_t, f)
    idf = idf_weights({"S2": tm}, max_df=1.0)
    tmat = weighted(tm, idf, 1.0)
    qmat = weighted(hash_texts(["acme tools ny"], f), idf, 1.0)
    import numpy as np

    blocks = _blocks(pl.Series(["S2-ny", "S2-ca", "S2-unk"]), tmat, np.array(["us_ny", "us_ca", ""]))
    out = _search(pl.Series(["S1-1"]), qmat, blocks, np.array(["us_ny"]), k=5, threshold=0.0)
    assert set(out["target_id"]) == {"S2-ny", "S2-unk"}
    out = _search(pl.Series(["S1-1"]), qmat, blocks, np.array([""]), k=5, threshold=0.0)
    assert set(out["target_id"]) == {"S2-ny", "S2-ca", "S2-unk"}  # unkeyed query searches every block


def test_fuse_ranks_per_source():
    c = pl.DataFrame({
        "s1_id": ["S1-1"] * 3, "target_id": ["S2-1", "S2-2", "S3-1"], "score": [0.9, 0.5, 0.1],
    })
    c = add_ranks(c).with_columns(pl.lit("B").alias("channel"))
    f = fuse(c, {"B": 1.0}).sort("target_id")
    assert f["frank"].to_list() == [1, 2, 1]  # S3 ranked independently of S2

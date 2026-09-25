"""Step 4 pair features: similarity numbers only, never raw country labels or token identities.

Every feature here is a comparison (ratio, flag, rank, margin) so it transfers to an unseen profile
(France). The one country-touching feature, `same_country`, compares q vs t rather than naming either.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist
from sklearn.preprocessing import normalize

from er.blocking.channels.sparse import FieldSpec, hash_texts
from er.match.idf import WordIdf, token_overlap_features

JOIN_COLS = [
    "name_native", "name_full", "name_core", "name_legal", "name_concat", "name_fold", "name_alts",
    "name_domain", "addr_content", "addr_region", "addr_house", "addr_numbers", "addr_postal",
    "addr_city", "addr_unit", "addr_landmark", "addr_empty", "country", "name_min", "name_min_key",
]

SIMS = [
    ("sim_name_ratio", "name_core", fuzz.ratio),
    ("sim_name_tset", "name_core", fuzz.token_set_ratio),
    ("sim_name_tsort", "name_core", fuzz.token_sort_ratio),
    ("sim_name_partial", "name_core", fuzz.partial_ratio),
    ("sim_name_jw", "name_core", JaroWinkler.normalized_similarity),
    ("sim_full_tset", "name_full", fuzz.token_set_ratio),
    ("sim_addr_ratio", "addr_content", fuzz.ratio),
    ("sim_addr_tset", "addr_content", fuzz.token_set_ratio),
    ("sim_city_ratio", "addr_city", fuzz.ratio),
    ("sim_min_tset", "name_min", fuzz.token_set_ratio),
    ("sim_min_ratio", "name_min", fuzz.ratio),
]

# Coarse ownership-form buckets, used only to derive a same/compatible/conflicting relation code —
# the canonical tokens themselves never leave this module as a feature.
_PRIVATE_LEGAL = ["pvt", "llc", "llp", "gmbh", "sarl", "eurl", "sasu", "opc", "sci", "scop", "selarl",
                  "pllc", "lc", "pa"]
_PUBLIC_LEGAL = ["inc", "corp", "plc"]


def _char_ngram_cosine(a: list[str], b: list[str]) -> np.ndarray:
    f = FieldSpec(col="_", analyzer="char_wb", ngram=(3, 3))
    ma = normalize(hash_texts(a, f), norm="l2", copy=False)
    mb = normalize(hash_texts(b, f), norm="l2", copy=False)
    return np.asarray(ma.multiply(mb).sum(axis=1)).ravel().astype(np.float32)


# List columns must stay List(Utf8) through the join: an all-empty list column (e.g. no aliases anywhere
# in a small batch) otherwise infers as List(Null), which the Rust set-op kernels don't implement.
_LIST_COLS = ("name_alts", "addr_numbers")


def _side(df: pl.DataFrame, id_col: str, prefix: str) -> pl.DataFrame:
    cols = [pl.col(c).cast(pl.List(pl.Utf8)) if c in _LIST_COLS else pl.col(c) for c in JOIN_COLS]
    return df.select(pl.col(id_col).alias(f"{prefix}id_tmp"), *[c.alias(f"{prefix}_{n}") for c, n in zip(cols, JOIN_COLS)])


def _join(pairs: pl.DataFrame, queries: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    df = pairs if "src" in pairs.columns else pairs.with_columns(pl.col("target_id").str.slice(0, 2).alias("src"))
    q = _side(queries, "entity_id", "q").rename({"qid_tmp": "s1_id"})
    t = _side(targets, "entity_id", "t").rename({"tid_tmp": "target_id"})
    return df.join(q, on="s1_id", how="left").join(t, on="target_id", how="left")


def _tri(qcol: str, tcol: str) -> pl.Expr:
    """0=missing, 1=same, 2=conflict — non-negative so LightGBM's categorical handling doesn't silently
    fold "conflict" into "missing" (it converts negative categorical codes to NaN)."""
    a, b = pl.col(qcol), pl.col(tcol)
    known = (a != "") & (b != "")
    return pl.when(~known).then(0).when(a == b).then(1).otherwise(2).cast(pl.Int8)


def _legal_relation() -> pl.Expr:
    q = pl.col("q_name_legal").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    t = pl.col("t_name_legal").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    missing = (q.list.len() == 0) | (t.list.len() == 0)
    same = q.list.set_intersection(t).list.len() > 0
    q_priv, q_pub = q.list.eval(pl.element().is_in(_PRIVATE_LEGAL)).list.any(), \
        q.list.eval(pl.element().is_in(_PUBLIC_LEGAL)).list.any()
    t_priv, t_pub = t.list.eval(pl.element().is_in(_PRIVATE_LEGAL)).list.any(), \
        t.list.eval(pl.element().is_in(_PUBLIC_LEGAL)).list.any()
    conflicting = (q_priv & t_pub & ~t_priv) | (q_pub & t_priv & ~q_priv)
    return (
        pl.when(missing).then(0).when(same).then(1).when(conflicting).then(3).otherwise(2)
    ).cast(pl.Int8).alias("legal_relation")


CTX_SCORE = "_ctx_score"  # (name token-set + address token-set) / 2: the score competitors are compared on


REL_BASE = ["p", "dense_cos", "sim_full_tset", "sim_min_tset", "sim_name_tset", "sim_addr_tset", "sim_name_char3",
            "sim_addr_char3", "name_tok_overlap_idf", "addr_num_jaccard"]
REL_FEATURES = [f"{k}_{c}" for c in REL_BASE for k in ("gap", "gapsrc", "z")]


def add_relative(df: pl.DataFrame) -> pl.DataFrame:
    """Scores relative to the S1's own shortlist; transfer better to unseen countries
    (reports/match/v2_relative_features.json). Needs complete per-S1 shortlists in `df`."""
    exprs = []
    for c in REL_BASE:
        exprs += [(pl.col(c).max().over("s1_id") - pl.col(c)).alias(f"gap_{c}"),
                  (pl.col(c).max().over("s1_id", "src") - pl.col(c)).alias(f"gapsrc_{c}"),
                  ((pl.col(c) - pl.col(c).mean().over("s1_id")) / (pl.col(c).std().over("s1_id") + 1e-6))
                  .fill_null(0.0).alias(f"z_{c}")]
    return df.with_columns(exprs)


def add_local_context(df: pl.DataFrame, score_col: str = CTX_SCORE) -> pl.DataFrame:
    """Rank/count/support within the S1's own candidate set: needs only that S1's rows."""
    df = df.with_columns(
        pl.col(score_col).rank("ordinal", descending=True).over("s1_id", "src").alias("rank_within_s1_src"),
        pl.col("sim_name_tset").rank("ordinal", descending=True).over("s1_id", "src").alias("rank_by_name"),
        pl.col("sim_addr_tset").rank("ordinal", descending=True).over("s1_id", "src").alias("rank_by_addr"),
        pl.len().over("s1_id", "src").alias("n_cands_same_src"),
        pl.len().over("s1_id").alias("n_cands_total"),
    )
    per_s1_src = df.group_by("s1_id", "src").agg(pl.col(score_col).max().alias("_max"))
    wide = per_s1_src.pivot(on="src", index="s1_id", values="_max")
    for src in ("S2", "S3"):
        if src not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0).alias(src))
    df = df.join(wide.select("s1_id", "S2", "S3"), on="s1_id", how="left")
    support = pl.when(pl.col("src") == "S2").then(pl.col("S3")).otherwise(pl.col("S2")).fill_null(0.0)
    # fixed dtype: a chunk with no S3 candidates would otherwise yield Float64 and break concatenation
    return df.with_columns(support.cast(pl.Float32).alias("support_opposite_src_best")).drop("S2", "S3")


def global_context(slim: pl.DataFrame, score_col: str = CTX_SCORE) -> pl.DataFrame:
    """Cross-S1 competition per target, over whatever universe `slim` covers (s1_id, target_id, score).

    Must be computed over the full S1 universe the model will face (review M1), not per chunk.
    """
    slim = slim.select("s1_id", "target_id", score_col)
    top2 = slim.sort([score_col, "s1_id"], descending=[True, False]).group_by("target_id", maintain_order=True).head(2)
    agg = top2.group_by("target_id").agg(
        pl.col("s1_id").first().alias("_top_s1"),
        pl.col(score_col).first().alias("_top_score"),
        pl.col(score_col).get(1, null_on_oob=True).alias("_second_score"),
    )
    claims = slim.group_by("target_id").agg(pl.col("s1_id").n_unique().alias("n_s1_claiming_record"))
    out = slim.join(agg, on="target_id", how="left").join(claims, on="target_id", how="left")
    competitor_best = pl.when(pl.col("s1_id") == pl.col("_top_s1")).then(
        pl.col("_second_score").fill_null(0.0)).otherwise(pl.col("_top_score"))
    return out.select("s1_id", "target_id", "n_s1_claiming_record",
                      (pl.col(score_col) - competitor_best).alias("margin_to_best_competitor"))


def _add_context_features(df: pl.DataFrame, score_col: str) -> pl.DataFrame:
    """Local + global context over the rows present (single-universe path, used by `build`)."""
    df = add_local_context(df, score_col)
    return df.join(global_context(df, score_col), on=["s1_id", "target_id"], how="left")


def build(pairs: pl.DataFrame, queries: pl.DataFrame, targets: pl.DataFrame, name_idf: WordIdf,
         addr_idf: WordIdf) -> pl.DataFrame:
    """pairs: s1_id, target_id [, pass-through columns]. Context is computed over these rows only."""
    df = build_local(pairs, queries, targets, name_idf, addr_idf)
    df = df.join(global_context(df), on=["s1_id", "target_id"], how="left")
    return df.drop(CTX_SCORE)


def build_local(pairs: pl.DataFrame, queries: pl.DataFrame, targets: pl.DataFrame, name_idf: WordIdf,
                addr_idf: WordIdf) -> pl.DataFrame:
    """All features except the cross-S1 ones (see `global_context`); keeps `CTX_SCORE` for them."""
    df = _join(pairs, queries, targets)

    for name, col, scorer in SIMS:
        a = df[f"q_{col}"].fill_null("").to_list()
        b = df[f"t_{col}"].fill_null("").to_list()
        df = df.with_columns(pl.Series(name, cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)))
    df = df.with_columns(
        pl.Series("sim_name_char3", _char_ngram_cosine(
            df["q_name_core"].fill_null("").to_list(), df["t_name_core"].fill_null("").to_list())),
        pl.Series("sim_addr_char3", _char_ngram_cosine(
            df["q_addr_content"].fill_null("").to_list(), df["t_addr_content"].fill_null("").to_list())),
    )

    df = token_overlap_features(df, "q_name_core", "t_name_core", name_idf, "name_tok")
    df = token_overlap_features(df, "q_addr_content", "t_addr_content", addr_idf, "addr_tok")
    df = token_overlap_features(df, "q_addr_landmark", "t_addr_landmark", addr_idf, "landmark_tok")

    house_a, house_b = pl.col("q_addr_house"), pl.col("t_addr_house")
    house_known = (house_a != "") & (house_b != "")
    house_diff = pl.when(house_known).then((house_a.cast(pl.Int64) - house_b.cast(pl.Int64)).abs())
    house_max = pl.when(house_known).then(pl.max_horizontal(house_a.cast(pl.Int64).abs(),
                                                             house_b.cast(pl.Int64).abs()).clip(lower_bound=1))

    core_exact = ((pl.col("q_name_core") == pl.col("t_name_core")) & (pl.col("q_name_core") != ""))
    dba = (~core_exact) & (
        pl.col("q_name_alts").list.contains(pl.col("t_name_core"))
        | pl.col("t_name_alts").list.contains(pl.col("q_name_core"))
        | (pl.col("q_name_alts").list.set_intersection(pl.col("t_name_alts")).list.len() > 0)
    )
    domain_match = (pl.col("q_name_domain") | pl.col("t_name_domain")) & (pl.col("q_name_concat") != "") & \
        (pl.col("q_name_concat") == pl.col("t_name_concat"))
    translit_match = (~core_exact) & (pl.col("q_name_native") != pl.col("t_name_native")) & \
        (pl.col("q_name_fold") != "") & (pl.col("q_name_fold") == pl.col("t_name_fold"))

    both_addr_empty = pl.col("q_addr_empty") & pl.col("t_addr_empty")
    either_addr_empty = pl.col("q_addr_empty") | pl.col("t_addr_empty")
    num_inter = pl.col("q_addr_numbers").list.set_intersection(pl.col("t_addr_numbers"))
    num_union_len = pl.col("q_addr_numbers").list.set_union(pl.col("t_addr_numbers")).list.len()

    df = df.with_columns(
        core_exact.cast(pl.Int8).alias("name_core_exact"),
        (dba | domain_match | translit_match).cast(pl.Int8).alias("alias_match"),
        (pl.col("q_name_native") != pl.col("t_name_native")).cast(pl.Int8).alias("script_mismatch"),
        pl.col("q_name_native").cast(pl.Int8).alias("q_name_native_f"),
        pl.col("t_name_native").cast(pl.Int8).alias("t_name_native_f"),
        _legal_relation(),
        _tri("q_addr_house", "t_addr_house").alias("house_state"),
        house_diff.alias("house_abs_diff"),
        pl.when(house_known & (house_max > 0)).then(house_diff / house_max).alias("house_rel_diff"),
        _tri("q_addr_region", "t_addr_region").alias("region_state"),
        _tri("q_addr_postal", "t_addr_postal").alias("postal_state"),
        _tri("q_addr_city", "t_addr_city").alias("locality_state"),
        _tri("q_addr_unit", "t_addr_unit").alias("unit_state"),
        both_addr_empty.cast(pl.Int8).alias("both_addr_empty"),
        either_addr_empty.cast(pl.Int8).alias("either_addr_empty"),
        num_inter.list.len().alias("addr_num_overlap"),
        pl.when(num_union_len > 0).then(num_inter.list.len() / num_union_len).otherwise(0.0)
        .alias("addr_num_jaccard"),
        (pl.col("q_country") == pl.col("t_country")).cast(pl.Int8).alias("same_country"),
        (pl.col("src") == "S3").cast(pl.Int8).alias("is_s3"),
    )

    # Noise-catalogue flags (reports/noise/): digit dropped / changed vs a genuinely different number;
    # names that differ only by generator noise (inserted words, honorifics, legal variants, homoglyphs).
    ha, hb = df["q_addr_house"].fill_null("").to_list(), df["t_addr_house"].fill_null("").to_list()
    lev = pl.Series("_house_lev", cpdist(ha, hb, scorer=Levenshtein.distance, workers=-1, dtype=np.int32))
    df = df.with_columns(lev)
    len_a, len_b = pl.col("q_addr_house").str.len_chars(), pl.col("t_addr_house").str.len_chars()
    df = df.with_columns(
        (house_known & (pl.col("_house_lev") == 1) & ((len_a - len_b).abs() == 1)).cast(pl.Int8)
        .alias("house_digit_dropped"),
        (house_known & (pl.col("_house_lev") == 1) & (len_a == len_b)).cast(pl.Int8).alias("house_digit_changed"),
        ((pl.col("q_name_min_key") == pl.col("t_name_min_key")) & (pl.col("q_name_min_key") != ""))
        .cast(pl.Int8).alias("name_min_exact"),
        ((pl.col("q_name_min_key") == pl.col("t_name_min_key")) & (pl.col("q_name_min_key") != "") & ~core_exact)
        .cast(pl.Int8).alias("name_noise_only_diff"),
    ).drop("_house_lev")

    df = df.with_columns(((pl.col("sim_name_tset") + pl.col("sim_addr_tset")) / 2).alias(CTX_SCORE))
    return add_local_context(df)


NUMERIC_FEATURES = (
    [s[0] for s in SIMS] + ["sim_name_char3", "sim_addr_char3"]
    + [f"{p}_{s}" for p in ("name_tok", "addr_tok", "landmark_tok")
       for s in ("overlap_n", "overlap_idf", "rare_overlap_n", "extra_q_n", "extra_q_idf", "extra_t_n",
                 "extra_t_idf")]
    + ["name_core_exact", "alias_match", "script_mismatch", "q_name_native_f", "t_name_native_f",
       "legal_relation", "house_state", "house_abs_diff", "house_rel_diff", "region_state",
       "postal_state", "locality_state", "unit_state", "both_addr_empty", "either_addr_empty",
       "addr_num_overlap", "addr_num_jaccard", "same_country", "is_s3", "rank_within_s1_src",
       "rank_by_name", "rank_by_addr", "n_cands_same_src", "n_cands_total", "n_s1_claiming_record",
       "margin_to_best_competitor", "support_opposite_src_best", "house_digit_dropped", "house_digit_changed",
       "name_min_exact", "name_noise_only_diff"]
)
CATEGORICAL_FEATURES = ["legal_relation", "house_state", "region_state", "postal_state", "locality_state",
                        "unit_state"]

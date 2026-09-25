"""Stage-2 candidate ranker: channel provenance + cheap string sims → GBDT → per-source top-K."""

from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist
from sklearn.ensemble import HistGradientBoostingClassifier

from er import io

MISSING_RANK = 1000
PAIR_COLS = ["name_core", "name_concat", "name_fold", "addr_content", "addr_house", "addr_region",
             "addr_numbers", "name_legal", "name_domain", "name_native", "addr_empty", "country"]
SIMS = [
    ("sim_name_ratio", "name_core", fuzz.ratio),
    ("sim_name_tset", "name_core", fuzz.token_set_ratio),
    ("sim_name_tsort", "name_core", fuzz.token_sort_ratio),
    ("sim_name_partial", "name_core", fuzz.partial_ratio),
    ("sim_concat_ratio", "name_concat", fuzz.ratio),
    ("sim_fold_tset", "name_fold", fuzz.token_set_ratio),
    ("sim_addr_tset", "addr_content", fuzz.token_set_ratio),
    ("sim_addr_ratio", "addr_content", fuzz.ratio),
]


def provenance(cands: pl.DataFrame, channels: list[str]) -> pl.DataFrame:
    wide = cands.pivot(on="channel", index=["s1_id", "target_id", "src"], values=["rank", "score"],
                       aggregate_function="min")
    for ch in channels:
        for kind, fill in (("rank", MISSING_RANK), ("score", 0.0)):
            col = f"{kind}_{ch}"
            if col not in wide.columns:
                wide = wide.with_columns(pl.lit(fill).alias(col))
            wide = wide.with_columns(pl.col(col).fill_null(fill).cast(pl.Float32))
    rank_cols = [f"rank_{c}" for c in channels]
    return wide.with_columns(
        pl.sum_horizontal([(pl.col(c) < MISSING_RANK).cast(pl.Int8) for c in rank_cols]).alias("n_channels"),
        pl.min_horizontal(rank_cols).alias("best_rank"),
    )


def pair_features(pairs: pl.DataFrame, queries: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    q = queries.select(pl.col("entity_id").alias("s1_id"), *[pl.col(c).alias(f"q_{c}") for c in PAIR_COLS])
    t = targets.select(pl.col("entity_id").alias("target_id"), *[pl.col(c).alias(f"t_{c}") for c in PAIR_COLS])
    df = pairs.join(q, on="s1_id", how="left").join(t, on="target_id", how="left")
    sims, strs = {}, {}
    for name, col, scorer in SIMS:
        if col not in strs:  # Polars→Python string conversion is costly; do it once per column
            strs[col] = (df[f"q_{col}"].fill_null("").to_list(), df[f"t_{col}"].fill_null("").to_list())
        sims[name] = cpdist(*strs[col], scorer=scorer, workers=io.THREADS, dtype=np.float32)
    del strs
    df = df.with_columns([pl.Series(k, v) for k, v in sims.items()])

    def tri(col: str) -> pl.Expr:
        a, b = pl.col(f"q_{col}"), pl.col(f"t_{col}")
        known = (a != "") & (b != "")
        return pl.when(~known).then(0).when(a == b).then(1).otherwise(-1).cast(pl.Int8)

    raw = [f"{side}_{c}" for c in PAIR_COLS for side in ("q", "t")]
    return df.with_columns(
        tri("addr_house").alias("house_state"),
        tri("addr_region").alias("region_state"),
        tri("name_legal").alias("legal_state"),
        pl.col("q_addr_numbers").list.set_intersection(pl.col("t_addr_numbers")).list.len().alias("num_overlap"),
        pl.col("t_name_domain").cast(pl.Int8).alias("t_domain"),
        pl.col("t_name_native").cast(pl.Int8).alias("t_native"),
        pl.col("t_addr_empty").cast(pl.Int8).alias("t_addr_empty_f"),
        (pl.col("q_country") == pl.col("t_country")).cast(pl.Int8).alias("same_country"),
        (pl.col("src") == "S3").cast(pl.Int8).alias("is_s3"),
    ).drop(raw)  # raw string columns are ~11M rows × 24 per chunk; features no longer need them


def feature_names(channels: list[str]) -> list[str]:
    return ([f"rank_{c}" for c in channels] + [f"score_{c}" for c in channels] + ["n_channels", "best_rank"]
            + [s[0] for s in SIMS]
            + ["house_state", "region_state", "legal_state", "num_overlap", "t_domain", "t_native",
               "t_addr_empty_f", "same_country", "is_s3"])


def build(cands: pl.DataFrame, channels: list[str], queries: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    return pair_features(provenance(cands, channels), queries, targets)


def label(df: pl.DataFrame, truth: pl.DataFrame) -> np.ndarray:
    tl = truth.explode("truth", empty_as_null=True).drop_nulls("truth").select(
        "s1_id", pl.col("truth").alias("target_id"), pl.lit(1).alias("y")
    )
    return df.join(tl, on=["s1_id", "target_id"], how="left", maintain_order="left")["y"].fill_null(0).to_numpy()


def _matrix(df: pl.DataFrame, features: list[str]) -> np.ndarray:
    # float32 is exact here (inputs are float32 scores/sims and small ints) and halves memory vs float64
    return df.select(pl.col(features).cast(pl.Float32)).to_numpy()


def fit_matrix(X: np.ndarray, y: np.ndarray, kind: str = "hgb", groups: np.ndarray | None = None):
    """kind: hgb (binary, sklearn) | lgbm_rank (lambdarank over (s1, source) groups; X sorted by group)."""
    if kind == "hgb":
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                               early_stopping=True, validation_fraction=0.1, random_state=0)
        return model.fit(X, y)
    if kind == "lgbm_rank":
        import lightgbm as lgb

        _, sizes = np.unique(groups, return_counts=True)  # groups are contiguous run ids
        model = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05, num_leaves=63,
                               min_child_samples=50, lambdarank_truncation_level=20, eval_at=[10],
                               n_jobs=io.THREADS, random_state=0, verbose=-1)
        return model.fit(X, y, group=sizes)
    raise ValueError(kind)


def score(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def fit(df: pl.DataFrame, y: np.ndarray, features: list[str]) -> HistGradientBoostingClassifier:
    return fit_matrix(_matrix(df, features), y)


def rank_by_model(df: pl.DataFrame, model: HistGradientBoostingClassifier, features: list[str]) -> pl.DataFrame:
    p = score(model, _matrix(df, features))
    return df.select("s1_id", "target_id", "src").with_columns(pl.Series("p", p.astype(np.float32))).with_columns(
        pl.col("p").rank("ordinal", descending=True).over("s1_id", "src").cast(pl.Int32).alias("frank")
    )

"""Step 2 diagnostics: key agreement / clash / unknown rates on held-out true pairs."""

from __future__ import annotations

import polars as pl

from er import io
from er.normalize.pipeline import NormConfig, load_norm

KEYS = ["name_full", "name_core", "name_sorted", "name_concat", "name_fold", "name_min_key", "addr_content"]
TRISTATE = ["addr_house", "addr_region", "addr_postal"]


def _jaccard(a: str, b: str) -> pl.Expr:
    sa, sb = pl.col(a).str.split(" ").list.unique(), pl.col(b).str.split(" ").list.unique()
    inter = sa.list.set_intersection(sb).list.len()
    union = sa.list.set_union(sb).list.len()
    return pl.when(union > 0).then(inter / union).otherwise(0.0)


def pair_table(cfg: NormConfig, s1_ids: pl.Series) -> pl.DataFrame:
    truth = io.load_truth().filter(pl.col("s1_id").is_in(s1_ids.implode()))
    links = io.truth_long(truth)
    s1 = load_norm("train", "S1", cfg).rename(lambda c: c if c == "entity_id" else f"q_{c}")
    tg = pl.concat([load_norm("train", s, cfg) for s in io.TARGET_SOURCES]).rename(
        lambda c: c if c == "entity_id" else f"t_{c}"
    )
    return (
        links.join(s1, left_on="s1_id", right_on="entity_id")
        .join(tg, left_on="target_id", right_on="entity_id")
    )


def diagnostics(cfg: NormConfig, s1_ids: pl.Series) -> dict:
    df = pair_table(cfg, s1_ids)
    exprs = []
    for k in KEYS:
        exprs.append(((pl.col(f"q_{k}") == pl.col(f"t_{k}")) & (pl.col(f"q_{k}") != "")).mean().alias(f"eq_{k}"))
    exprs.append(_jaccard("q_name_core", "t_name_core").mean().alias("jac_name_core"))
    exprs.append(_jaccard("q_addr_content", "t_addr_content").mean().alias("jac_addr_content"))
    for k in TRISTATE:
        q, t = pl.col(f"q_{k}"), pl.col(f"t_{k}")
        known = (q != "") & (t != "")
        exprs += [
            (known & (q == t)).mean().alias(f"{k}_match"),
            (known & (q != t)).mean().alias(f"{k}_clash"),
            (~known).mean().alias(f"{k}_unknown"),
        ]
    exprs.append(
        pl.col("q_addr_numbers").list.set_intersection(pl.col("t_addr_numbers")).list.len().gt(0).mean()
        .alias("numbers_overlap")
    )
    exprs.append(pl.len().alias("pairs"))

    groups = {
        "all": df.select(exprs),
        "by_country_src": df.group_by("q_country", "src").agg(exprs).sort("q_country", "src"),
        "by_target_type": df.with_columns(
            pl.when(pl.col("t_name_domain")).then(pl.lit("domain"))
            .when(pl.col("t_name_native")).then(pl.lit("native_script"))
            .when(pl.col("t_addr_empty")).then(pl.lit("empty_addr"))
            .otherwise(pl.lit("latin")).alias("target_type")
        ).group_by("target_type").agg(exprs).sort("target_type"),
    }
    return {name: g.to_dicts() for name, g in groups.items()}

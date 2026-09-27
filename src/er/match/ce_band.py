"""Uncertain-band rescoring: for lo < p < hi, p becomes a logistic on [logit p, tanh(ce / 2), SC].

The cross-encoder (`cross.py`, trained on non-val rtrain10) is weak alone but decorrelated from the GBDT
(residual corr -0.02). tanh bounds its logit, whose scale drifts on an unseen country (India proxy: raw
logit -0.0004 vs tanh +0.0001 on the tilted rule). SC = sibling house consensus: a band target whose house
differs from S1's but equals a confident sibling's is 89% true. Optional EXR runs after the rescoring.
"""

from __future__ import annotations

import json
import pickle
import tomllib
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from er import io
from er.blocking import ranker
from er.decide import exclusivity, linkcount, pipeline
from er.match import cross
from er.match.v2 import MODELS, _policy_predict
from er.metrics import macro_f05
from er.split import load_split, subset_ids

BAND = (0.2, 0.9)
CE_MODEL = "ce_us"  # dir name under artifacts/models
CACHE = io.ARTIFACTS / "match" / "ce"
VAL_PROXY = io.ARTIFACTS / "match" / "ce_val_us.parquet"  # all 8.8M val pairs, scored by the CE proxy run


SC_COLS = ["sc_eq", "sc_nosib", "sc_lsib"]
RAW_COLS = ["raw_eq", "raw_nt", "raw_n1"]
ADDR_COLS = ["rawa_eq", "rawa_nt"]  # legal-form agreement was tried too: +0.00006 val, -0.0007 India rehearsal
SIB_P = 0.9
KPM_TOL = 0.03  # Σp drift that triggers matching; test US -1.0%, India -1.4%, France -6.0%


def _x(p: np.ndarray, ce: np.ndarray, sc: np.ndarray | None = None) -> np.ndarray:
    q = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    x = np.c_[np.log(q / (1 - q)), np.tanh(ce.astype(np.float64) / 2)]
    return x if sc is None else np.c_[x, sc]


def sibling_features(scored: pl.DataFrame, pairs: pl.DataFrame, split: str, config: str) -> pl.DataFrame:
    """SC columns for `pairs`, from the S1's confident siblings (p >= SIB_P) in `scored`; house numbers from the
    normalized records. Band pairs sit below SIB_P, so a pair is never its own sibling."""
    from er.normalize.pipeline import NormConfig, load_norm

    cfg = tomllib.loads(Path(config).read_text(encoding="utf-8"))
    nc = NormConfig(**cfg.get("norm", {}))
    h = pl.concat([load_norm(split, s, nc).select("entity_id", "addr_house") for s in io.SOURCES])
    q = h.rename({"entity_id": "s1_id", "addr_house": "qh"})
    t = h.rename({"entity_id": "target_id", "addr_house": "th"})
    sib = (scored.filter(pl.col("p") >= SIB_P).join(t, on="target_id").filter(pl.col("th") != "")
           .group_by("s1_id").agg(pl.col("th").alias("sib_h"), pl.len().alias("n_sib")))
    f = (pairs.select("s1_id", "target_id").join(q, on="s1_id", how="left").join(t, on="target_id", how="left")
         .join(sib, on="s1_id", how="left").with_columns(pl.col("n_sib").fill_null(0)))
    in_sib = pl.col("sib_h").list.contains(pl.col("th")).fill_null(False)
    differs = (pl.col("th") != "") & (pl.col("th") != pl.col("qh"))
    return f.select("s1_id", "target_id",
                    (differs & in_sib).cast(pl.Float32).alias("sc_eq"),
                    (differs & (pl.col("qh") != "") & (pl.col("n_sib") > 0) & ~in_sib).cast(pl.Float32)
                    .alias("sc_nosib"),
                    pl.col("n_sib").log1p().cast(pl.Float32).alias("sc_lsib"))


def _fold(col: str) -> pl.Expr:
    return (pl.col(col).fill_null("").str.to_lowercase().str.split(" ")
            .list.eval(pl.element().filter(pl.element() != "")).list.join(" "))


def raw_features(pairs: pl.DataFrame, split: str) -> pl.DataFrame:
    """Raw name / address agreement (case/whitespace-folded, legal form kept) and how many S1 of the country carry
    each raw value. Copies keep the owner's raw form more often than a same-key twin's: unique raw-equal claimant =
    owner 98% (val)."""
    s1 = io.load_source(split, "S1").select(pl.col("entity_id").alias("s1_id"), "country",
                                            _fold("business_name").alias("k1"), _fold("business_address").alias("a1"))
    tg = pl.concat([io.load_source(split, s) for s in io.SOURCES if s != "S1"]).select(
        pl.col("entity_id").alias("target_id"), _fold("business_name").alias("kt"), _fold("business_address").alias("at"))
    cnt = s1.group_by("country", "k1").len().rename({"len": "n"})
    acnt = s1.group_by("country", "a1").len().rename({"a1": "at", "len": "na_t"})
    return (pairs.select("s1_id", "target_id").join(s1, on="s1_id").join(tg, on="target_id")
            .join(cnt.rename({"k1": "kt", "n": "n_t"}), on=["country", "kt"], how="left")
            .join(cnt.rename({"n": "n_1"}), on=["country", "k1"], how="left")
            .join(acnt, on=["country", "at"], how="left")
            .select("s1_id", "target_id", (pl.col("k1") == pl.col("kt")).cast(pl.Float32).alias("raw_eq"),
                    pl.col("n_t").fill_null(0).log1p().cast(pl.Float32).alias("raw_nt"),
                    pl.col("n_1").fill_null(0).log1p().cast(pl.Float32).alias("raw_n1"),
                    ((pl.col("a1") == pl.col("at")) & (pl.col("a1") != "")).cast(pl.Float32).alias("rawa_eq"),
                    pl.col("na_t").fill_null(0).log1p().cast(pl.Float32).alias("rawa_nt")))


def _extras(scored: pl.DataFrame, pairs: pl.DataFrame, split: str, config: str, sc: bool, raw: bool,
            raw_addr: bool = False) -> tuple[pl.DataFrame | None, list[str]]:
    """Combiner inputs beyond [logit p, tanh(ce/2)]: SC and/or raw name (and address) columns for `pairs`."""
    out, cols = pairs.select("s1_id", "target_id"), []
    if sc:
        out, cols = out.join(sibling_features(scored, pairs, split, config), on=["s1_id", "target_id"], how="left"), SC_COLS
    if raw or raw_addr:
        out = out.join(raw_features(pairs, split), on=["s1_id", "target_id"], how="left")
        cols = cols + (RAW_COLS if raw else []) + (ADDR_COLS if raw_addr else [])
    return (out if cols else None), cols


def _band(df: pl.DataFrame, band: tuple[float, float]) -> pl.DataFrame:
    return df.filter((pl.col("p") > band[0]) & (pl.col("p") < band[1]))


def _tag(tag: str, ce_model: str) -> str:
    return tag if ce_model == CE_MODEL else f"{tag}_{ce_model}"  # keeps the original ce_us cache names


def ce_scores(pairs: pl.DataFrame, split: str, tag: str, ce_model: str = CE_MODEL) -> pl.DataFrame:
    """CE logits for (s1_id, target_id) pairs, cached per tag and model; only uncached pairs hit the GPU."""
    path = CACHE / f"ce_{_tag(tag, ce_model)}.parquet"
    keys = pairs.select("s1_id", "target_id").unique()
    have = pl.read_parquet(path) if path.exists() else pl.DataFrame(
        schema={"s1_id": pl.String, "target_id": pl.String, "ce": pl.Float32})
    if split == "train" and ce_model == CE_MODEL and VAL_PROXY.exists():  # proxy scores are ce_us scores
        have = pl.concat([have, pl.read_parquet(VAL_PROXY).join(keys, on=["s1_id", "target_id"], how="semi")
                          .select("s1_id", "target_id", pl.col("ce").cast(pl.Float32))]).unique(["s1_id", "target_id"])
    missing = keys.join(have, on=["s1_id", "target_id"], how="anti")
    if missing.height:
        model, tok = cross.load(io.ARTIFACTS / "models" / ce_model)
        logits = cross.score(cross.attach_text(missing, split), model, tok)
        have = pl.concat([have, missing.with_columns(pl.Series("ce", logits, dtype=pl.Float32))])
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".partial")
        have.write_parquet(tmp)
        tmp.replace(path)
        print(f"  ce[{_tag(tag, ce_model)}]: scored {missing.height:,} new pairs", flush=True)
    return have.join(keys, on=["s1_id", "target_id"], how="semi")


def apply(scored: pl.DataFrame, ce: pl.DataFrame, model: LogisticRegression,
          band: tuple[float, float] = BAND, sc: pl.DataFrame | None = None, cols: list[str] = SC_COLS) -> pl.DataFrame:
    """Replace in-band p with the combiner's probability; out-of-band rows keep p exactly."""
    b = _band(scored, band).join(ce, on=["s1_id", "target_id"], how="left")
    if b["ce"].null_count():
        raise ValueError(f"{b['ce'].null_count()} in-band pairs have no CE score")
    scm = None
    if sc is not None:
        b = b.join(sc, on=["s1_id", "target_id"], how="left")
        scm = b.select(cols).fill_null(0).to_numpy()
    new = b.select("s1_id", "target_id", pl.Series("_p2", model.predict_proba(
        _x(b["p"].to_numpy(), b["ce"].to_numpy(), scm))[:, 1]))
    return (scored.join(new, on=["s1_id", "target_id"], how="left")
            .with_columns(pl.coalesce("_p2", "p").cast(pl.Float32).alias("p")).drop("_p2"))


def _combiner(kind: str):
    """xgb on (0.02, 0.995) beat the band logistic by +0.0011 val and +0.0008 on the India proxy."""
    if kind == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08, tree_method="hist", n_jobs=io.THREADS)
    return LogisticRegression(C=1e3, max_iter=1000)


def fit_band(name: str, band: tuple[float, float] = BAND, exr: bool = True, use_sc: bool = True,
             ce_model: str = CE_MODEL, combiner: str = "logistic", use_raw: bool = False,
             raw_addr: bool = False) -> dict:
    """Cross-fit the combiner on the bundle's OOF (by cv_fold), re-select the policy on the rescored OOF,
    and store the final combiner (fit on all band pairs) + policy in the bundle under `ce_band`."""
    truth = io.load_truth()
    oof = pl.read_parquet(io.ARTIFACTS / "match" / f"oof_{name}.parquet")
    if "cv_fold" not in oof.columns:
        oof = oof.join(load_split().select("s1_id", "cv_fold"), on="s1_id", how="left")
    oof = oof.with_columns(pl.Series("y", ranker.label(oof, truth)))
    bundle_path = MODELS / f"{name}.pkl"
    bundle = pickle.loads(bundle_path.read_bytes())
    # the CE trained on the ranker's subset (`cross.training_pairs`); 141k of v6's expanded S1 sit in it
    cfg = tomllib.loads(Path(bundle["config"]).read_text(encoding="utf-8"))
    ce_seen = subset_ids(load_split(), cfg["ranker"].get("train_subset", "rtrain"))
    oof = oof.filter(~pl.col("s1_id").is_in(ce_seen.implode()))
    b = _band(oof, band).join(ce_scores(_band(oof, band), "train", f"{name}_oof", ce_model), on=["s1_id", "target_id"])
    ext, cols = _extras(oof, b, "train", bundle["config"], use_sc, use_raw, raw_addr)
    scm = None
    if ext is not None:
        b = b.join(ext, on=["s1_id", "target_id"], how="left")
        scm = b.select(cols).fill_null(0).to_numpy()
    X, y, folds = _x(b["p"].to_numpy(), b["ce"].to_numpy(), scm), b["y"].to_numpy(), b["cv_fold"].to_numpy()
    p_cf = np.empty(len(y))
    for f in np.unique(folds):
        m = folds == f
        p_cf[m] = _combiner(combiner).fit(X[~m], y[~m]).predict_proba(X[m])[:, 1]
    final = _combiner(combiner).fit(X, y)
    rescored = (oof.join(b.select("s1_id", "target_id", pl.Series("_p2", p_cf)), on=["s1_id", "target_id"], how="left")
                .with_columns(pl.coalesce("_p2", "p").cast(pl.Float32).alias("p")).drop("_p2", "y"))
    if exr:
        rescored = exclusivity.renormalize(rescored)
    decision = pipeline.run(rescored, truth)
    policy = {"decision_winner": decision["decision_winner"], "exclusivity_winner": decision["exclusivity_winner"],
              "threshold": decision["global_threshold"]["best"]}
    by_country = {}
    for c in rescored["q_country"].unique().sort().to_list():
        sub = rescored.filter(pl.col("q_country") == c)
        by_country[c] = macro_f05(truth.filter(pl.col("s1_id").is_in(sub["s1_id"].unique().implode())),
                                  _policy_predict(sub, policy))
    bundle["ce_band"] = {"model": final, "band": band, "exr": exr, "sc": use_sc, "raw": use_raw, "raw_addr": raw_addr,
                         "ce_model": ce_model, "combiner": combiner, "policy": policy}
    bundle_path.write_bytes(pickle.dumps(bundle))
    rescored.write_parquet(io.ARTIFACTS / "match" / f"oof_{name}_band.parquet")
    result = {"name": name, "band": band, "exr": exr, "sc": use_sc, "raw": use_raw, "raw_addr": raw_addr,
              "ce_model": ce_model, "combiner": combiner,
              "band_pairs": b.height,
              "coef": final.coef_.ravel().tolist() + final.intercept_.tolist() if combiner == "logistic" else None,
              "policy": policy,
              "final_macro_f05": decision["final"]["macro_f05"], "by_country": by_country,
              "stages": {k: v.get("macro_f05") for k, v in decision["stages"].items()}}
    (io.REPORTS / "match" / f"{name}_band.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def decide_test(name: str, parts: int, use_band: bool = True, exr: bool | None = None, kpm: bool = False,
                by_country: bool = False, out_dir: str | None = None) -> dict:
    """Re-decide cached test scores (no re-scoring): band rescore -> optional EXR -> optional KPM -> policy -> TSVs.

    `kpm` / `by_country` treat every country label alike (open set): Σp matched to the train prior per label,
    tilt computed per label. They only bite where a label's count calibration drifted (France Σp 3.33)."""
    bundle = pickle.loads((MODELS / f"{name}.pkl").read_bytes())
    scored = pl.read_parquet(io.ARTIFACTS / "match" / f"test_scored_{name}_p{parts}.parquet")
    cty = io.load_source("test", "S1").select(pl.col("entity_id").alias("s1_id"), pl.col("country"))
    policy = bundle["policy"]
    if use_band:
        cb = bundle["ce_band"]
        bp = _band(scored, cb["band"])
        sc, cols = _extras(scored, bp, "test", bundle["config"], bool(cb.get("sc")), bool(cb.get("raw")),
                           bool(cb.get("raw_addr")))
        ce = ce_scores(bp, "test", f"{name}_p{parts}_test", cb.get("ce_model", CE_MODEL))
        scored = apply(scored, ce, cb["model"], cb["band"], sc, cols)
        policy = cb["policy"]
        exr = cb["exr"] if exr is None else exr
    if exr:
        scored = exclusivity.renormalize(scored)
    prior = linkcount.load_prior()
    if kpm or by_country:
        scored = scored.join(cty, on="s1_id", how="left")
        if kpm:
            scored = linkcount.match_expected_count(scored, prior.mean_total, "country", tol=KPM_TOL)
    with_cty = scored if "country" in scored.columns else scored.join(cty, on="s1_id", how="left")
    sum_p = with_cty.group_by("country").agg((pl.col("p").sum() / pl.col("s1_id").n_unique()).alias("sp"))
    test_ids = io.load_source("test", "S1")["entity_id"]
    if by_country and policy["decision_winner"] == "prior_tilted_exact_f05":
        excl = exclusivity.resolve_max_probability(scored)
        preds = linkcount.select_tilted(excl, prior, group_col="country").select("s1_id", "target_id")
    else:
        preds = _policy_predict(scored.drop("country", strict=False), policy, n_s1=test_ids.len())
    scored = scored.drop("country", strict=False)
    out = Path(out_dir) if out_dir else io.ROOT / "output"
    out.mkdir(parents=True, exist_ok=True)

    def write(frame: pl.DataFrame, col: str, fname: str) -> pl.DataFrame:
        lists = frame.sort("s1_id", "target_id").group_by("s1_id", maintain_order=True).agg(
            pl.col("target_id").str.join(",").alias(col))
        full = test_ids.to_frame("source1_entity_id").join(
            lists.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left"
        ).with_columns(pl.col(col).fill_null(""))
        full.write_csv(out / fname, separator="\t", quote_style="never")
        return full

    full = write(preds, "matched_entity_ids", "matching_results.tsv")
    write(scored, "candidate_entity_ids", "candidate_pairs.tsv")
    per = (preds.join(cty, on="s1_id").group_by("country").len()
           .join(cty.group_by("country").len().rename({"len": "n"}), on="country")
           .with_columns((pl.col("len") / pl.col("n")).alias("links_per_s1")))
    stats = {"name": name, "parts": parts, "band": use_band, "sc": bool(use_band and bundle["ce_band"].get("sc")),
             "raw": bool(use_band and bundle["ce_band"].get("raw")), "exr": bool(exr), "kpm": kpm, "by_country": by_country, "policy": policy,
             "sum_p_per_s1_by_country": dict(zip(sum_p["country"].to_list(), sum_p["sp"].to_list())),
             "predicted_pairs": preds.height, "s1_with_matches": int((full["matched_entity_ids"] != "").sum()),
             "links_per_s1": preds.height / test_ids.len(),
             "links_per_s1_by_country": dict(zip(per["country"].to_list(), per["links_per_s1"].to_list()))}
    tag = (f"{name}_p{parts}" + ("_band" if use_band else "") + ("_exr" if exr else "") + ("_kpm" if kpm else "")
           + ("_bycty" if by_country else ""))
    (io.REPORTS / f"decide_{tag}.json").write_text(json.dumps(stats, indent=2, default=str))
    return stats

"""Command-line entry point: `uv run er <command>`."""

from __future__ import annotations

import argparse
import json
import sys
import time

import polars as pl

from er import io
from er.match import features
from er.metrics import macro_f05
from er.split import SEED


def cmd_prepare(args: argparse.Namespace) -> None:
    io.prepare(tuple(args.splits), force=args.force)


def split_frame() -> pl.DataFrame:
    from er.split import load_split

    return load_split()


def cmd_split(args: argparse.Namespace) -> None:
    truth = io.load_truth()
    split = split_frame()
    s1 = io.load_source("train", "S1").select(pl.col("entity_id").alias("s1_id"), "country")
    val = truth.join(split.filter(pl.col("is_val")), on="s1_id").join(s1, on="s1_id", how="left")
    counts = (
        split.join(s1, on="s1_id").group_by("country", "is_val").len().sort("country", "is_val")
    )
    card = val.group_by(pl.col("truth").list.len().alias("k")).len().sort("k")
    empty_preds = pl.DataFrame(schema={"s1_id": pl.Utf8, "target_id": pl.Utf8})
    report = {
        "seed": SEED,
        "validation_s1": val.height,
        "dev_s1": int(split.filter(pl.col("is_val") & pl.col("is_dev")).height),
        "entity_counts": {
            f"{r['country']}/{'validation' if r['is_val'] else 'train'}": r["len"]
            for r in counts.iter_rows(named=True)
        },
        "validation_cardinality": {str(k): n for k, n in card.iter_rows()},
        "all_empty_macro_f05": macro_f05(val, empty_preds),
    }
    perl = json.loads((io.REPORTS / "step1_metrics.json").read_text())
    parity = {
        "validation_s1": perl["validation"]["s1_entities"] == report["validation_s1"],
        "all_empty": abs(perl["validation"]["all_empty_prediction_macro_f05"] - report["all_empty_macro_f05"])
        < 1e-12,
        "cardinality": perl["validation"]["match_cardinality"] == report["validation_cardinality"],
        "country_counts": all(
            perl["split"]["entity_counts_by_country"][c][k] == report["entity_counts"][f"{c}/{k}"]
            for c in ("US", "India")
            for k in ("train", "validation")
        ),
    }
    report["parity_with_perl"] = parity
    (io.REPORTS / "step1_split.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not all(parity.values()):
        raise SystemExit("PARITY FAILED")


def _norm_cfg(args: argparse.Namespace):
    from er.normalize.pipeline import NormConfig

    return NormConfig(translit=not args.no_translit, profile_mode=args.profile_mode, learned=args.learned)


def _add_norm_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--no-translit", action="store_true", help="N2 ablation: keep native scripts")
    sp.add_argument("--profile-mode", default="auto", choices=["auto", "generic"])
    sp.add_argument("--learned", default=None, help="N4 learned tables json")


def cmd_normalize(args: argparse.Namespace) -> None:
    from er.normalize.pipeline import normalize_source

    cfg = _norm_cfg(args)
    for split in args.splits:
        for src in io.SOURCES:
            normalize_source(split, src, cfg, force=args.force)


def _transpose(rows: list[dict]) -> str:
    """Metric-per-line table: one column per group row."""
    label_keys = [k for k, v in rows[0].items() if isinstance(v, str)]
    heads = ["/".join(str(r[k]) for k in label_keys) or "all" for r in rows]
    lines = ["metric".ljust(22) + "".join(h[:14].rjust(15) for h in heads)]
    for k, v in rows[0].items():
        if k in label_keys:
            continue
        cells = [r[k] for r in rows]
        fmt = [f"{c:.4f}" if isinstance(c, float) else str(c) for c in cells]
        lines.append(k.ljust(22) + "".join(f.rjust(15) for f in fmt))
    return "\n".join(lines)


def cmd_diagnose(args: argparse.Namespace) -> None:
    from er.normalize.diagnostics import diagnostics
    from er.split import subset_ids

    cfg = _norm_cfg(args)
    res = diagnostics(cfg, subset_ids(split_frame(), args.subset))
    out = io.REPORTS / "normalization" / f"{cfg.variant}_{args.subset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    for name, rows in res.items():
        print(f"== {name}")
        print(_transpose(rows))


def cmd_block(args: argparse.Namespace) -> None:
    from er.blocking.runner import leaderboard, run

    m = run(args.config, args.subset, args.save_misses)
    rows = [{"K": k, **{x: b[x] for x in ("oracle_ceiling_f05", "pair_recall", "full_coverage",
                                            "mean_candidates")}} for k, b in m["budgets"].items()]
    rows.append({"K": "union", **{x: m["union"][x] for x in rows[0] if x != "K"}})
    print(_transpose([{**r, "K": str(r["K"])} for r in rows]))
    if "ranked" in m:
        pu = m["precut_union"]
        print(f"precut union: recall {pu['pair_recall']:.4f} oracle {pu['oracle_ceiling_f05']:.4f} "
              f"mean cand {pu['mean_candidates']:.1f}")
        print("ranker:")
        print(_transpose([{"K": k, **{x: b[x] for x in ("oracle_ceiling_f05", "pair_recall", "full_coverage",
                                                          "mean_candidates")}} for k, b in m["ranked"].items()]))
    print("per-channel:")
    keep = ("pair_recall", "full_coverage", "oracle_ceiling_f05", "mean_candidates", "unique_links", "unique_link_share")
    print(_transpose([{"ch": c, **{x: v[x] for x in keep}} for c, v in m["channels"].items()]))
    print(f"total {m['total_s']}s; channel timings {m['timings_s']}; peak RSS {m['peak_rss_gb']} GB")
    print("slices @max K:")
    for col, groups in m["budgets"][str(max(map(int, m["budgets"])))]["slices"].items():
        print(_transpose([{col: g, **v} for g, v in sorted(groups.items())]))
    leaderboard()


def cmd_infer(args: argparse.Namespace) -> None:
    from er.blocking.runner import infer

    stats = infer(args.config, args.k, io.ROOT / "output")
    print(json.dumps(stats, indent=2, default=str))


def cmd_finetune(args: argparse.Namespace) -> None:
    from er.blocking.channels.dense import embed_source
    from er.blocking.train_dense import finetune

    out = finetune(args.name, args.pairs, args.batch, args.epochs, args.lr)
    for split in args.embed:
        for src in io.SOURCES:
            embed_source(split, src, model=str(out))


def cmd_fit2(args: argparse.Namespace) -> None:
    from er.match import v2

    r = v2.fit(args.config, args.k, args.name, transfer=args.transfer, scope=args.scope,
               backend=args.backend, drop_groups=tuple(args.drop_groups), train_scope=args.train_scope,
               macro_weight=args.macro_weight, fu_extras=args.fu_extras, mask_groups=tuple(args.mask_groups),
               mask_rate=args.mask_rate)
    keep = ("final_macro_f05", "by_country_frozen_policy", "policy", "country_transfer", "total_s")
    print(json.dumps({k: r.get(k) for k in keep}, indent=2, default=str))


def cmd_proxy(args: argparse.Namespace) -> None:
    from er.match import proxy

    print(json.dumps(proxy.run(args.name, args.holdout, args.backend, args.arm, tuple(args.mask_groups),
                               args.mask_rate, tuple(args.add_features)), indent=2, default=str))


def cmd_decide2(args: argparse.Namespace) -> None:
    from er.match import v2

    print(json.dumps(v2.redecide(args.name), indent=2, default=str))


def cmd_submit2(args: argparse.Namespace) -> None:
    from er.match import v2

    print(json.dumps(v2.submit(args.name, args.ctx_parts), indent=2, default=str))


def cmd_ce_train(args: argparse.Namespace) -> None:
    from er.match import cross

    pairs = cross.attach_text(cross.training_pairs(args.config, args.chunks, args.countries or None), "train")
    print(json.dumps(cross.train(pairs, io.ARTIFACTS / "models" / args.name, init=args.init, lr=args.lr),
                     indent=2, default=str))


def cmd_ce_adapt(args: argparse.Namespace) -> None:
    from er.match import cross

    print(json.dumps(cross.adapt(args.name, args.parts, args.base, args.out, args.n_s1), indent=2, default=str))


def cmd_ce_train_mix(args: argparse.Namespace) -> None:
    from er.match import cross

    print(json.dumps(cross.train_mix(args.name, args.parts, args.out, args.n_labeled, args.init, args.lr),
                     indent=2, default=str))


def cmd_ce_band(args: argparse.Namespace) -> None:
    from er.match import ce_band

    print(json.dumps(ce_band.fit_band(args.name, (args.lo, args.hi), not args.no_exr, not args.no_sc, args.ce_model,
                                      args.combiner, args.raw, args.raw_addr), indent=2, default=str))


def cmd_decide_test(args: argparse.Namespace) -> None:
    from er.match import ce_band

    exr = None if args.exr is None else args.exr == "on"
    print(json.dumps(ce_band.decide_test(args.name, args.parts, not args.no_band, exr, args.kpm, args.by_country,
                                         args.out), indent=2, default=str))


def cmd_transfer(args: argparse.Namespace) -> None:
    from er.match import transfer

    r = transfer.run(args.config, args.k, args.scope, tuple(args.arms), args.backend, args.name)
    print(json.dumps({"arms": [{k: a[k] for k in ("arm", "overall", "in_distribution", "held_out", "gap")}
                               for a in r["arms"]],
                      "smallest_worst_case_gap": r["smallest_worst_case_gap"]}, indent=2, default=str))


def cmd_tune(args: argparse.Namespace) -> None:
    from er.match import tune

    r = tune.run(args.config, args.k, args.scope, args.backend, args.trials, args.frac,
                 not args.no_monotone, args.name, relative=not args.no_relative,
                 drop_groups=tuple(args.drop_groups))
    print(json.dumps({k: r[k] for k in ("subsample_scores", "best_params", "full_frame_macro_f05",
                                        "seconds")}, indent=2, default=str))


def cmd_crossenc(args: argparse.Namespace) -> None:
    from er.match import cross_encoder

    out = cross_encoder.finetune(args.name, args.config, args.k, args.n_s1, args.batch, args.epochs,
                                 args.lr, args.model)
    print(json.dumps({"model_dir": str(out)}, indent=2))


def cmd_linkcount(args: argparse.Namespace) -> None:
    """Seconds-long check on saved OOF: is the count prior worth wiring into the decision rule?"""
    from er.decide import linkcount

    oof = pl.read_parquet(io.ARTIFACTS / "match" / f"oof_{args.name}.parquet")
    prior = linkcount.load_prior()
    diag = linkcount.count_diagnostics(oof, prior, args.n_s1)
    truth = io.load_truth().filter(pl.col("s1_id").is_in(oof["s1_id"].unique().implode()))
    n_s1 = truth.height if args.n_s1 is None else args.n_s1
    diag["prior_matched_threshold"] = linkcount.prior_matched_threshold(oof, n_s1, prior.mean_total)
    diag["prior"] = {"mean_total": prior.mean_total, "caps": prior.caps,
                     "per_source_mean": {s: prior.mean_src(s) for s in prior.per_src}}
    print(json.dumps(diag, indent=2, default=str))


def cmd_leaderboard(args: argparse.Namespace) -> None:
    from er.blocking.runner import leaderboard

    print(leaderboard())


def cmd_match(args: argparse.Namespace) -> None:
    from er.blocking.runner import collect_match_features
    from er.match import report, train
    from er.split import load_split

    df, extra = collect_match_features(args.config, args.subset, args.k)
    truth = io.load_truth()
    folds = load_split()
    result = train.fit_and_report(df, extra, truth, folds, holdout_country=args.holdout_country)
    written = report.write(result)
    print(json.dumps(written["metrics"], indent=2, default=str))
    print(f"gate report -> {report.REPORT_DIR / 'gate_report.md'}")


def cmd_decide(args: argparse.Namespace) -> None:
    import polars as pl

    from er.decide import pipeline, report

    oof_path = io.ROOT / args.oof
    oof = pl.read_parquet(oof_path)
    truth = io.load_truth()
    result = pipeline.run(oof, truth)
    written = report.write(result)
    print(json.dumps({k: written[k] for k in ("stages", "decision_winner", "exclusivity_winner")},
                     indent=2, default=str))
    print(f"gate report -> {report.REPORT_DIR / 'gate_report.md'}")


def cmd_submit(args: argparse.Namespace) -> None:
    from er import submit

    stats = submit.run(args.config, args.k, args.match_subset, args.holdout_country)
    print(json.dumps(stats, indent=2, default=str))
    print(f"matching_results.tsv -> {io.ROOT / 'output' / 'matching_results.tsv'}")


def cmd_learn(args: argparse.Namespace) -> None:
    from er.normalize.learned import mine

    split_frame()
    out = io.ROOT / "configs" / "learned" / f"{args.name}.json"
    t = mine(_norm_cfg(args), out, args.min_support, args.min_precision)
    print(f"name_tokens={len(t['name_tokens'])} regions={len(t['regions'])} -> {out}")
    for k, v in list(t["name_tokens"].items())[:40]:
        print(f"  {k} -> {v}")
    for k, v in list(t["regions"].items())[:30]:
        print(f"  [region] {k} -> {v}")


def _add_mask_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--mask-groups", nargs="*", default=[], choices=sorted(features.MASK_GROUPS),
                    help="GM: feature groups masked per S1 on training rows (see features.MASK_GROUPS)")
    sp.add_argument("--mask-rate", type=float, default=0.0, help="GM: share of S1 masked per group")


def main() -> None:
    p = argparse.ArgumentParser(prog="er")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare", help="convert raw TSVs to parquet")
    sp.add_argument("--splits", nargs="+", default=["train", "test"])
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(fn=cmd_prepare)

    sp = sub.add_parser("split", help="build split and verify parity with the Perl scorer")
    sp.set_defaults(fn=cmd_split)

    sp = sub.add_parser("normalize", help="parallel normalization to artifacts/norm/<variant>/")
    sp.add_argument("--splits", nargs="+", default=["train"])
    sp.add_argument("--force", action="store_true")
    _add_norm_args(sp)
    sp.set_defaults(fn=cmd_normalize)

    sp = sub.add_parser("diagnose", help="Step 2 key agreement on held-out true pairs")
    sp.add_argument("--subset", default="dev", choices=["dev", "val"])
    _add_norm_args(sp)
    sp.set_defaults(fn=cmd_diagnose)

    sp = sub.add_parser("block", help="run a blocking config and evaluate it")
    sp.add_argument("config")
    sp.add_argument("--subset", default=None, choices=["dev", "val", "rtrain"])
    sp.add_argument("--save-misses", action="store_true", help="write lost true links to artifacts/misses/")
    sp.set_defaults(fn=cmd_block)

    sp = sub.add_parser("infer", help="test-set blocking → output/candidate_pairs.tsv")
    sp.add_argument("config")
    sp.add_argument("--k", type=int, default=10, help="ranker budget per target source")
    sp.set_defaults(fn=cmd_infer)

    sp = sub.add_parser("finetune", help="contrastive fine-tune of the dense retriever, then re-embed")
    sp.add_argument("--name", default="e5s_ft1")
    sp.add_argument("--pairs", type=int, default=600_000)
    sp.add_argument("--batch", type=int, default=512)
    sp.add_argument("--epochs", type=int, default=1)
    sp.add_argument("--lr", type=float, default=5e-5)
    sp.add_argument("--embed", nargs="*", default=["train"], help="splits to re-embed with the new model")
    sp.set_defaults(fn=cmd_finetune)

    sp = sub.add_parser("fit2", help="Step 4/5 v2: universe shortlists, val features, CV, policy, transfer")
    sp.add_argument("config")
    sp.add_argument("--k", type=int, default=10)
    sp.add_argument("--name", default="v2")
    sp.add_argument("--transfer", action="store_true", help="also run leave-country-out transfer checks")
    sp.add_argument("--drop-groups", nargs="*", default=[], choices=sorted(features.FEATURE_GROUPS),
                    help="feature groups to drop for the transfer ablation (see features.FEATURE_GROUPS)")
    sp.add_argument("--train-scope", default="val", choices=["val", "expanded", "full"],
                    help="val: M5-clean val-only fit; expanded: val + clean rtrain12 S1s; full: context over all train "
                         "S1 + clean train-fold extras (streaming CV)")
    sp.add_argument("--fu-extras", type=int, default=450_000, help="full scope: clean extra training S1 to sample")
    sp.add_argument("--scope", default="val", choices=["val", "full"], help="competitor universe for context")
    sp.add_argument("--backend", default="xgb_gpu", choices=["xgb_gpu", "lgbm"])
    sp.add_argument("--macro-weight", action="store_true", help="expanded scope: weight pairs 1/max(links, 1) per S1")
    _add_mask_args(sp)
    sp.set_defaults(fn=cmd_fit2)

    sp = sub.add_parser("proxy", help="unseen-country gate: fit without --holdout, score it frozen, held-out F0.5")
    sp.add_argument("--name", default="v7", help="bundle whose features and scope caches to use")
    sp.add_argument("--holdout", default="India")
    sp.add_argument("--arm", default="base", help="report/scores suffix; compare arms run by this command only")
    sp.add_argument("--backend", default="xgb_gpu", choices=["xgb_gpu", "xgb_cpu"])
    sp.add_argument("--add-features", nargs="*", default=[], help="columns to add to the bundle's features")
    _add_mask_args(sp)
    sp.set_defaults(fn=cmd_proxy)

    sp = sub.add_parser("decide2", help="re-run the Step 5 policy on saved v2 OOF (no retraining)")
    sp.add_argument("--name", default="v2")
    sp.set_defaults(fn=cmd_decide2)

    sp = sub.add_parser("submit2", help="score the test universe with a fit2 bundle; write both TSVs")
    sp.add_argument("--name", default="v2")
    sp.add_argument("--ctx-parts", type=int, default=None, help="competitor partitions (default: density-matched)")
    sp.set_defaults(fn=cmd_submit2)

    sp = sub.add_parser("ce-train", help="train the band cross-encoder on ranker-subset pairs (never val)")
    sp.add_argument("config")
    sp.add_argument("--name", default="ce_usin", help="output dir under artifacts/models")
    sp.add_argument("--chunks", type=int, default=5, help="25k-S1 chunks of the ranker subset to draw pairs from")
    sp.add_argument("--countries", nargs="*", default=[], help="restrict to these labels (default: all)")
    sp.add_argument("--init", default="artifacts/models/e5s_ft1")
    sp.add_argument("--lr", type=float, default=3e-5)
    sp.set_defaults(fn=cmd_ce_train)

    sp = sub.add_parser("ce-adapt", help="fine-tune a CE on pseudo-labels of test countries without training labels")
    sp.add_argument("--name", default="v7", help="scorer bundle whose test scores give the pseudo-labels")
    sp.add_argument("--parts", type=int, default=1)
    sp.add_argument("--base", default="ce_usin_base", help="CE dir under artifacts/models to start from")
    sp.add_argument("--out", default="ce_usin_base_fr", help="output dir under artifacts/models")
    sp.add_argument("--n-s1", type=int, default=80_000, help="pseudo-labeled S1 to sample")
    sp.set_defaults(fn=cmd_ce_adapt)

    sp = sub.add_parser("ce-train-mix", help="train a CE on labeled full-scope pairs + unlabeled-country pseudo-labels")
    sp.add_argument("--name", default="v7", help="scorer bundle: OOF S1 to exclude, test scores for pseudo-labels")
    sp.add_argument("--parts", type=int, default=1)
    sp.add_argument("--out", default="ce_usin_base4f", help="output dir under artifacts/models")
    sp.add_argument("--n-labeled", type=int, default=4_200_000)
    sp.add_argument("--init", default="intfloat/multilingual-e5-base")
    sp.add_argument("--lr", type=float, default=2e-5)
    sp.set_defaults(fn=cmd_ce_train_mix)

    sp = sub.add_parser("ce-band", help="fit the uncertain-band CE combiner on OOF; re-select and store the policy")
    sp.add_argument("--name", default="v6")
    sp.add_argument("--lo", type=float, default=0.2)
    sp.add_argument("--hi", type=float, default=0.9)
    sp.add_argument("--no-exr", action="store_true", help="skip exclusivity renormalization")
    sp.add_argument("--no-sc", action="store_true", help="skip sibling house-consensus features")
    sp.add_argument("--ce-model", default="ce_us", help="cross-encoder dir under artifacts/models")
    sp.add_argument("--combiner", default="logistic", choices=["logistic", "xgb"])
    sp.add_argument("--raw", action="store_true", help="add raw-name agreement / support features")
    sp.add_argument("--raw-addr", action="store_true", help="add raw-address agreement / support features")
    sp.set_defaults(fn=cmd_ce_band)

    sp = sub.add_parser("decide-test", help="re-decide cached test scores (band, EXR, policy); write both TSVs")
    sp.add_argument("--name", default="v6")
    sp.add_argument("--parts", type=int, required=True, help="which test_scored_<name>_p<parts> file")
    sp.add_argument("--no-band", action="store_true")
    sp.add_argument("--exr", choices=["on", "off"], default=None, help="override the bundle's EXR setting")
    sp.add_argument("--kpm", action="store_true", help="match Σp per country label to the train prior")
    sp.add_argument("--by-country", action="store_true", help="compute the count tilt per country label")
    sp.add_argument("--out", default=None, help="output dir for the two TSVs (default: output/)")
    sp.set_defaults(fn=cmd_decide_test)

    sp = sub.add_parser("transfer", help="area 2: ablate the ranker score / add within-record views, leave-country-out")
    sp.add_argument("config")
    sp.add_argument("--k", type=int, default=10)
    sp.add_argument("--name", default="transfer")
    sp.add_argument("--scope", default="val", choices=["val", "full"])
    sp.add_argument("--backend", default="xgb_gpu", choices=["xgb_gpu", "xgb_cpu", "lgbm", "catboost"])
    sp.add_argument("--arms", nargs="+", default=["base", "no_p", "rel", "rel_no_p"])
    sp.set_defaults(fn=cmd_transfer)

    sp = sub.add_parser("tune", help="area 3: Optuna search with monotone constraints (XGBoost or CatBoost)")
    sp.add_argument("config")
    sp.add_argument("--k", type=int, default=10)
    sp.add_argument("--name", default="tune")
    sp.add_argument("--scope", default="val", choices=["val", "full"])
    sp.add_argument("--backend", default="xgb_gpu", choices=["xgb_gpu", "xgb_cpu", "lgbm", "catboost"])
    sp.add_argument("--trials", type=int, default=30)
    sp.add_argument("--frac", type=float, default=0.3, help="fraction of S1 each trial fits on")
    sp.add_argument("--no-monotone", action="store_true", help="disable monotone constraints")
    sp.add_argument("--no-relative", action="store_true", help="exclude REL_FEATURES (tune the pre-v3 set)")
    sp.add_argument("--drop-groups", nargs="*", default=[], choices=sorted(features.FEATURE_GROUPS),
                    help="feature groups to drop (see features.FEATURE_GROUPS)")
    sp.set_defaults(fn=cmd_tune)

    sp = sub.add_parser("crossenc", help="area 1: fine-tune the Ditto-style cross-encoder on train-fold pairs")
    sp.add_argument("config")
    sp.add_argument("--k", type=int, default=10)
    sp.add_argument("--name", default="ce1")
    sp.add_argument("--n-s1", type=int, default=60_000)
    sp.add_argument("--batch", type=int, default=128)
    sp.add_argument("--epochs", type=int, default=1)
    sp.add_argument("--lr", type=float, default=2e-5)
    sp.add_argument("--model", default="intfloat/multilingual-e5-small")
    sp.set_defaults(fn=cmd_crossenc)

    sp = sub.add_parser("linkcount", help="area 4: is the link-count prior worth using? (seconds, on saved OOF)")
    sp.add_argument("--name", default="v2", help="bundle name: reads artifacts/match/oof_<name>.parquet")
    sp.add_argument("--n-s1", type=int, default=None, help="S1 universe size incl. records with no candidates")
    sp.set_defaults(fn=cmd_linkcount)

    sp = sub.add_parser("leaderboard", help="rebuild reports/leaderboard.md")
    sp.set_defaults(fn=cmd_leaderboard)

    sp = sub.add_parser("match", help="Step 4: candidate features + LightGBM classifier + gate report")
    sp.add_argument("config", help="blocking config (channels + ranker) to shortlist candidates with")
    sp.add_argument("--subset", default="dev", choices=["dev", "val", "train"])
    sp.add_argument("--k", type=int, default=10, help="ranker budget per target source")
    sp.add_argument("--holdout-country", default="India", help="country left out for the France stand-in check")
    sp.set_defaults(fn=cmd_match)

    sp = sub.add_parser("decide", help="Step 5: exclusivity + per-entity expected-F0.5 decision layer")
    sp.add_argument("--oof", default="reports/match/oof.parquet", help="OOF parquet written by `er match`")
    sp.set_defaults(fn=cmd_decide)

    sp = sub.add_parser("submit", help="real test-set matching_results.tsv: Step 4 classifier + Step 5 decision")
    sp.add_argument("config", help="blocking config (channels + ranker) to shortlist candidates with")
    sp.add_argument("--k", type=int, default=10, help="ranker budget per target source")
    sp.add_argument("--match-subset", default="dev", choices=["dev", "val", "train"],
                    help="train-subset used to fit the classifier and decide the Step 5 policy")
    sp.add_argument("--holdout-country", default="India", help="country left out for the France stand-in check")
    sp.set_defaults(fn=cmd_submit)

    sp = sub.add_parser("learn", help="mine N4 alias tables from train-fold pairs")
    sp.add_argument("--name", default="n4_v1")
    sp.add_argument("--min-support", type=int, default=20)
    sp.add_argument("--min-precision", type=float, default=0.9)
    _add_norm_args(sp)
    sp.set_defaults(fn=cmd_learn)

    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()
    args.fn(args)
    print(f"[{args.cmd}] {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()

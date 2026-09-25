"""Command-line entry point: `uv run er <command>`."""

from __future__ import annotations

import argparse
import json
import sys
import time

import polars as pl

from er import io
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

"""Gate report: metrics table, calibration plot, feature importance, holdout-country gap."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from er import io

REPORT_DIR = io.REPORTS / "match"

# json-safe top-level keys from er.match.train.run(); the rest (models/df/oof arrays/isotonic) are runtime state.
_JSON_KEYS = ("n_pairs", "pos_rate", "n_folds", "features", "categorical_features", "metrics",
             "calibration", "country_holdout", "feature_importance")


def _fmt(x, nd=4) -> str:
    return f"{x:.{nd}f}" if isinstance(x, float) else "n/a" if x is None else str(x)


def plot_calibration(calibration: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
    raw = calibration["raw"]
    ax.plot(raw["bin_mean_pred"], raw["bin_frac_pos"], "o-", label=f"raw (brier={raw['brier_score']:.4f})")
    if calibration.get("calibrated"):
        c = calibration["calibrated"]
        ax.plot(c["bin_mean_pred"], c["bin_frac_pos"], "s-", label=f"isotonic (brier={c['brier_score']:.4f})")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed match rate")
    ax.set_title("Step 4 classifier calibration")
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def gate_markdown(result: dict) -> str:
    m, cal, hc = result["metrics"], result["calibration"], result["country_holdout"]
    intro = (f"{result['n_pairs']:,} candidate pairs, positive rate {result['pos_rate']:.4f}, "
            f"{result['n_folds']} grouped folds.")
    overall_row = (f"| overall | {_fmt(m['auc'])} | {_fmt(m['pr_auc'])} | {_fmt(m['macro_f05_at_threshold'])} | "
                  f"{m['n_pairs']:,} | {_fmt(m['pos_rate'])} |")
    lines = [
        "# Step 4 gate report", "",
        intro, "",
        "## Metrics (out-of-fold, threshold 0.5)", "",
        "| slice | AUC | PR-AUC | macro F0.5 | n pairs | pos rate |",
        "|---|---|---|---|---|---|",
        overall_row,
    ]
    for country, cm in sorted(m["by_country"].items()):
        lines.append(f"| {country} | {_fmt(cm['auc'])} | {_fmt(cm['pr_auc'])} | "
                     f"{_fmt(cm['macro_f05_at_threshold'])} | {cm['n_pairs']:,} | {_fmt(cm['pos_rate'])} |")
    lines += ["", "## Calibration", "",
             f"raw max gap {_fmt(cal['raw']['max_calibration_gap'])}, "
             f"brier {_fmt(cal['raw']['brier_score'])}; "
             + ("isotonic calibration applied." if cal["applied"] else "no calibration needed.")]
    lines += ["", "## Country-holdout check (leave-country-out, stand-in for France)", "",
             f"Holdout country: **{hc['holdout_country']}**", "",
             "| | AUC | PR-AUC | macro F0.5 |", "|---|---|---|---|"]
    ind = hc["in_distribution"]
    lco = hc["leave_country_out"]
    lines.append(f"| in-distribution (in CV) | {_fmt(ind['auc']) if ind else 'n/a'} | "
                 f"{_fmt(ind['pr_auc']) if ind else 'n/a'} | "
                 f"{_fmt(ind['macro_f05_at_threshold']) if ind else 'n/a'} |")
    if "skipped" in lco:
        lines.append(f"| leave-country-out | {lco['skipped']} | | |")
    else:
        lines.append(f"| leave-country-out | {_fmt(lco['auc'])} | {_fmt(lco['pr_auc'])} | "
                     f"{_fmt(lco['macro_f05_at_threshold'])} |")
        if ind and ind.get("macro_f05_at_threshold") is not None:
            gap = ind["macro_f05_at_threshold"] - lco["macro_f05_at_threshold"]
            lines.append(f"\nholdout-country macro F0.5 gap: **{gap:.4f}** (in-distribution − leave-country-out)")
    lines += ["", "## Feature importance (mean gain across folds)", "",
             "| rank | feature | gain |", "|---|---|---|"]
    for i, (feat, gain) in enumerate(result["feature_importance"][:25], 1):
        lines.append(f"| {i} | {feat} | {gain:.1f} |")
    lines += ["", "![calibration](calibration.png)", ""]
    return "\n".join(lines)


def write_oof(result: dict, out_path: Path) -> Path:
    """Persist per-pair OOF labels/probabilities — Step 5's decision layer reads this directly."""
    df = result["df"].select("s1_id", "target_id", "src", "q_country").with_columns(
        pl.Series("y", result["y"]),
        pl.Series("p_raw", result["oof_raw"]),
        pl.Series("p", result["oof_calibrated"]),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    return out_path


def write(result: dict, out_dir: Path = REPORT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_safe = {k: result[k] for k in _JSON_KEYS}
    (out_dir / "metrics.json").write_text(json.dumps(json_safe, indent=2, default=str))
    plot_calibration(result["calibration"], out_dir / "calibration.png")
    (out_dir / "gate_report.md").write_text(gate_markdown(result), encoding="utf-8")
    write_oof(result, out_dir / "oof.parquet")
    return json_safe

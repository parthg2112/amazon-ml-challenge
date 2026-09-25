"""Step 5 gate report: OOF macro F0.5 before/after each decision-layer component, PR curve, singleton accuracy."""

from __future__ import annotations

import json
from pathlib import Path

from er import io

REPORT_DIR = io.REPORTS / "decide"

_STAGE_ORDER = [
    ("0_threshold_0.5_no_exclusivity", "baseline: threshold 0.5, no exclusivity"),
    ("1_threshold_0.5_plus_exclusivity", "+ exclusivity (max-probability)"),
    ("2a_tuned_global_threshold", "+ OOF-tuned global threshold"),
    ("2b_per_entity_expected_f05", "+ per-entity expected-F0.5 k-selection"),
    ("2b_per_entity_scarcity_exclusivity", "  (comparison: scarcity-weighted exclusivity)"),
    ("3_final", "FINAL (best decision rule x best exclusivity rule, on this OOF slice)"),
]


def _fmt(x, nd=4) -> str:
    return f"{x:.{nd}f}" if isinstance(x, float) else "n/a" if x is None else str(x)


def gate_markdown(result: dict) -> str:
    cstats = result["conflict_stats"]
    intro = (f"{result['n_s1']:,} S1 entities. Conflicts: {cstats['contested_targets']:,} of "
            f"{cstats['targets_claimed']:,} claimed records were claimed by more than one S1 "
            f"(max {cstats['max_claimants']} claimants).")
    decision_line = (f"Decision rule winner (OOF): **{result['decision_winner']}** "
                     f"(tuned global threshold = {result['global_threshold']['best']:.2f}).")
    exclusivity_line = f"Exclusivity rule winner (OOF): **{result['exclusivity_winner']}**."
    lines = [
        "# Step 5 gate report", "",
        intro, "",
        "## OOF macro F0.5, before/after each component", "",
        "| stage | macro F0.5 | singleton accuracy | predicted pairs |",
        "|---|---|---|---|",
    ]
    for key, label in _STAGE_ORDER:
        s = result["stages"].get(key)
        if s is None:
            continue
        lines.append(f"| {label} | {_fmt(s['macro_f05'])} | {_fmt(s['singleton_accuracy'])} | "
                     f"{s['n_predicted_pairs']:,} |")
    lines += ["", decision_line, exclusivity_line, ""]
    lines += ["## Precision/recall/F0.5 by threshold and country (max-probability exclusivity)", "",
             "| country | threshold | precision | recall | macro F0.5 |", "|---|---|---|---|---|"]
    for row in result["pr_curve"]:
        lines.append(f"| {row['country']} | {row['threshold']:.2f} | {_fmt(row['precision'])} | "
                     f"{_fmt(row['recall'])} | {_fmt(row['macro_f05'])} |")
    lines.append("")
    return "\n".join(lines)


def write(result: dict, out_dir: Path = REPORT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_safe = {k: v for k, v in result.items()}
    (out_dir / "metrics.json").write_text(json.dumps(json_safe, indent=2, default=str))
    (out_dir / "gate_report.md").write_text(gate_markdown(result), encoding="utf-8")
    return json_safe

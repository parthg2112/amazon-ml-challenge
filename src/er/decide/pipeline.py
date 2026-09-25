"""Step 5 end-to-end: exclusivity -> per-entity expected-F0.5 selection -> gate metrics, on OOF predictions.

Order matters: exclusivity is resolved first (a global pass dropping losing claims on contested
records), then each S1 chooses its own candidate-set size over its now-deduplicated pool. Doing it the
other way — each S1 picking a set, then arbitrating conflicts — can leave a losing S1 stuck short of its
chosen k with no chance to fall back to its next-best candidate, which is strictly worse.
"""

from __future__ import annotations

import polars as pl

import numpy as np

from er.decide import evaluate, exact_f05, exclusivity, expected_f05, threshold

FINE_GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)  # review M6: 0.05 steps were too coarse


def run(oof: pl.DataFrame, truth: pl.DataFrame) -> dict:
    truth_subset = truth.filter(pl.col("s1_id").is_in(oof["s1_id"].unique().implode()))
    stages: dict[str, dict] = {}

    stage0_preds = threshold.predict(oof, 0.5)
    stages["0_threshold_0.5_no_exclusivity"] = evaluate.stage_metrics(stage0_preds, truth_subset)

    excl_max = exclusivity.resolve_max_probability(oof)
    excl_scarcity = exclusivity.resolve_scarcity_weighted(oof)
    conflict_info = exclusivity.conflict_stats(oof)

    stage1_preds = threshold.predict(excl_max, 0.5)
    stages["1_threshold_0.5_plus_exclusivity"] = evaluate.stage_metrics(stage1_preds, truth_subset)

    best_t, best_t_f05, curve = threshold.tune(excl_max, truth_subset, grid=FINE_GRID)
    stage2a_preds = threshold.predict(excl_max, best_t)
    stages["2a_tuned_global_threshold"] = {
        **evaluate.stage_metrics(stage2a_preds, truth_subset), "threshold": best_t,
    }

    stage2b_preds = expected_f05.select(excl_max).select("s1_id", "target_id")
    stages["2b_per_entity_expected_f05"] = evaluate.stage_metrics(stage2b_preds, truth_subset)

    stage2c_preds = exact_f05.select(excl_max).select("s1_id", "target_id")
    stages["2c_exact_expected_f05"] = evaluate.stage_metrics(stage2c_preds, truth_subset)

    stage2b_scarcity_preds = expected_f05.select(excl_scarcity).select("s1_id", "target_id")
    stages["2b_per_entity_scarcity_exclusivity"] = evaluate.stage_metrics(stage2b_scarcity_preds, truth_subset)

    decision_winner = max(
        [("tuned_global_threshold", stages["2a_tuned_global_threshold"]["macro_f05"]),
         ("per_entity_expected_f05", stages["2b_per_entity_expected_f05"]["macro_f05"]),
         ("exact_expected_f05", stages["2c_exact_expected_f05"]["macro_f05"])],
        key=lambda kv: kv[1],
    )[0]
    exclusivity_winner = max(
        [("max_probability", stages["2b_per_entity_expected_f05"]["macro_f05"]),
         ("scarcity_weighted", stages["2b_per_entity_scarcity_exclusivity"]["macro_f05"])],
        key=lambda kv: kv[1],
    )[0]

    excl_by_name = {"max_probability": excl_max, "scarcity_weighted": excl_scarcity}
    chosen_excl = excl_by_name[exclusivity_winner]
    if decision_winner == "tuned_global_threshold":
        final_preds = threshold.predict(chosen_excl, best_t)
    elif decision_winner == "exact_expected_f05":
        final_preds = exact_f05.select(chosen_excl).select("s1_id", "target_id")
    else:
        final_preds = expected_f05.select(chosen_excl).select("s1_id", "target_id")
    final_metrics = evaluate.stage_metrics(final_preds, truth_subset)
    stages["3_final"] = final_metrics

    nested = nested_threshold(excl_max, truth_subset) if "cv_fold" in oof.columns else None
    return {
        "nested_threshold": nested,
        "n_s1": oof["s1_id"].n_unique(),
        "conflict_stats": conflict_info,
        "stages": stages,
        "decision_winner": decision_winner,
        "exclusivity_winner": exclusivity_winner,
        "global_threshold": {"best": best_t, "best_macro_f05": best_t_f05, "curve": curve},
        "pr_curve": evaluate.pr_curve(excl_max, truth_subset).to_dicts(),
        "final": final_metrics,
    }


def nested_threshold(excl: pl.DataFrame, truth: pl.DataFrame, grid: np.ndarray | None = None) -> dict:
    """Unbiased estimate of the tuned-threshold rule (review M6): tune on 4 folds, score the 5th."""
    grid = np.round(np.arange(0.04, 0.97, 0.02), 2) if grid is None else grid
    per_fold, weights, chosen = [], [], []
    for f in sorted(excl["cv_fold"].unique().to_list()):
        inner, outer = excl.filter(pl.col("cv_fold") != f), excl.filter(pl.col("cv_fold") == f)
        t, _, _ = threshold.tune(inner, truth, grid=grid)
        t_outer = truth.filter(pl.col("s1_id").is_in(outer["s1_id"].unique().implode()))
        per_fold.append(evaluate.stage_metrics(threshold.predict(outer, t), t_outer)["macro_f05"])
        weights.append(outer["s1_id"].n_unique())
        chosen.append(t)
    return {"macro_f05": float(np.average(per_fold, weights=weights)), "thresholds": chosen,
            "per_fold": per_fold}

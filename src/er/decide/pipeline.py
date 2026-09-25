"""Step 5 end-to-end: exclusivity -> per-entity expected-F0.5 selection -> gate metrics, on OOF predictions.

Order matters: exclusivity is resolved first (a global pass dropping losing claims on contested
records), then each S1 chooses its own candidate-set size over its now-deduplicated pool. Doing it the
other way — each S1 picking a set, then arbitrating conflicts — can leave a losing S1 stuck short of its
chosen k with no chance to fall back to its next-best candidate, which is strictly worse.
"""

from __future__ import annotations

import polars as pl

from er.decide import evaluate, exclusivity, expected_f05, threshold


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

    best_t, best_t_f05, curve = threshold.tune(excl_max, truth_subset)
    stage2a_preds = threshold.predict(excl_max, best_t)
    stages["2a_tuned_global_threshold"] = {
        **evaluate.stage_metrics(stage2a_preds, truth_subset), "threshold": best_t,
    }

    stage2b_preds = expected_f05.select(excl_max).select("s1_id", "target_id")
    stages["2b_per_entity_expected_f05"] = evaluate.stage_metrics(stage2b_preds, truth_subset)

    stage2b_scarcity_preds = expected_f05.select(excl_scarcity).select("s1_id", "target_id")
    stages["2b_per_entity_scarcity_exclusivity"] = evaluate.stage_metrics(stage2b_scarcity_preds, truth_subset)

    decision_winner = max(
        [("tuned_global_threshold", stages["2a_tuned_global_threshold"]["macro_f05"]),
         ("per_entity_expected_f05", stages["2b_per_entity_expected_f05"]["macro_f05"])],
        key=lambda kv: kv[1],
    )[0]
    exclusivity_winner = max(
        [("max_probability", stages["2b_per_entity_expected_f05"]["macro_f05"]),
         ("scarcity_weighted", stages["2b_per_entity_scarcity_exclusivity"]["macro_f05"])],
        key=lambda kv: kv[1],
    )[0]

    excl_by_name = {"max_probability": excl_max, "scarcity_weighted": excl_scarcity}
    chosen_excl = excl_by_name[exclusivity_winner]
    final_preds = (
        threshold.predict(chosen_excl, best_t) if decision_winner == "tuned_global_threshold"
        else expected_f05.select(chosen_excl).select("s1_id", "target_id")
    )
    final_metrics = evaluate.stage_metrics(final_preds, truth_subset)
    stages["3_final"] = final_metrics

    return {
        "n_s1": oof["s1_id"].n_unique(),
        "conflict_stats": conflict_info,
        "stages": stages,
        "decision_winner": decision_winner,
        "exclusivity_winner": exclusivity_winner,
        "global_threshold": {"best": best_t, "best_macro_f05": best_t_f05, "curve": curve},
        "pr_curve": evaluate.pr_curve(excl_max, truth_subset).to_dicts(),
        "final": final_metrics,
    }

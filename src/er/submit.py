"""Real test-set predictions: Step 4 classifier + Step 5 decision layer, actually applied — writes
`output/matching_results.tsv` with real (s1_id, target_id) matches instead of `infer()`'s all-empty
placeholder.

The Step 5 policy (which exclusivity rule, and a tuned threshold or the per-entity rule) is decided once
on labeled train-subset OOF predictions and then applied frozen to the test set, which has no ground
truth to tune against — tuning on test would be leakage into the only number that will ever grade this.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from er import io
from er.blocking import ranker
from er.blocking.runner import collect_match_features, collect_match_features_test
from er.decide import exclusivity, expected_f05, pipeline, threshold
from er.match import features
from er.match import train as match_train
from er.split import load_split


def run(config_path: str, k: int = 10, match_subset: str = "dev", holdout_country: str = "India") -> dict:
    # 1) Train-subset features + labels: fits the classifier and decides the Step 5 policy.
    df_train, extra = collect_match_features(config_path, match_subset, k)
    truth = io.load_truth()
    y = ranker.label(df_train, truth)
    folds = load_split()
    df_train = df_train.join(folds.select("s1_id", "cv_fold"), on="s1_id", how="left")

    feature_names = features.NUMERIC_FEATURES + extra
    categorical = features.CATEGORICAL_FEATURES

    oof, _models, _importances = match_train.cv_train(df_train, y, "cv_fold", feature_names, categorical)
    calibrated, calibration, iso = match_train.maybe_calibrate(y, oof)

    # 2) Decide the Step 5 policy on this OOF — never on the unlabeled test set.
    oof_df = df_train.select("s1_id", "target_id", "src", "q_country").with_columns(
        pl.Series("p", calibrated))
    decision = pipeline.run(oof_df, truth)

    # 3) One production model on ALL of the train-subset's rows (no folds held out — nothing left to
    # validate on since the policy was already fixed in step 2).
    final_model = match_train.train_full(df_train, y, feature_names, categorical)

    # 4) Score the real test set with that model, then the same calibration curve.
    df_test, extra_test, test_ids = collect_match_features_test(config_path, k)
    if extra_test != extra:
        raise ValueError("train/test blocking-provenance columns differ — configs must match")
    X_test = df_test.select(feature_names).to_numpy()
    raw_test = final_model.predict(X_test)
    p_test = iso.transform(raw_test).astype(np.float32) if iso is not None else raw_test.astype(np.float32)
    test_scored = df_test.select("s1_id", "target_id", "src").with_columns(pl.Series("p", p_test))

    # 5) Apply the frozen policy: exclusivity, then the decision rule that won on OOF.
    excl_fn = (exclusivity.resolve_scarcity_weighted if decision["exclusivity_winner"] == "scarcity_weighted"
              else exclusivity.resolve_max_probability)
    excl = excl_fn(test_scored)
    if decision["decision_winner"] == "tuned_global_threshold":
        final_preds = threshold.predict(excl, decision["global_threshold"]["best"])
    else:
        final_preds = expected_f05.select(excl).select("s1_id", "target_id")

    # 6) Write matching_results.tsv — one row per test S1, empty string for no match.
    lists = final_preds.sort("s1_id", "target_id").group_by("s1_id", maintain_order=True).agg(
        pl.col("target_id").str.join(",").alias("matched_entity_ids"))
    full = test_ids.to_frame("source1_entity_id").join(
        lists.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left"
    ).with_columns(pl.col("matched_entity_ids").fill_null(""))

    out_dir = io.ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    full.write_csv(out_dir / "matching_results.tsv", separator="\t", quote_style="never")

    stats = {
        "match_subset": match_subset, "holdout_country": holdout_country,
        "train_pairs": len(y), "test_s1": test_ids.len(),
        "test_pairs_scored": df_test.height, "predicted_pairs": final_preds.height,
        "s1_with_matches": int((full["matched_entity_ids"] != "").sum()),
        "decision_winner": decision["decision_winner"], "exclusivity_winner": decision["exclusivity_winner"],
        "threshold": decision["global_threshold"]["best"],
        "oof_macro_f05": decision["final"]["macro_f05"],
        "calibration_applied": calibration["applied"],
    }
    (io.REPORTS / "submit.json").write_text(json.dumps(stats, indent=2, default=str))
    return stats

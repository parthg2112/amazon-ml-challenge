# Step 4–5 review: what was explored, what is missing, what may be wrong

Scope: `src/er/match/` (Step 4), `src/er/decide/` (Step 5), `src/er/submit.py`, `reports/{match,decide}/gate_report.md`.
All reference numbers are dev OOF on the **old v2 blocking** (no dense channel): Step 4 0.9633 at threshold 0.5,
Step 5 final 0.9666 (tuned global threshold 0.70), leave-India-out 0.8323.

## 1. What was explored, and what was chosen or rejected

| Decision point | Options tried | Winner | Documented reason |
|---|---|---|---|
| Classifier | LightGBM binary only | LightGBM | none: no alternative compared |
| Hyper-parameters | fixed defaults (lr 0.05, 63 leaves, `is_unbalance`) | — | not tuned |
| Calibration | isotonic, applied only if reliability gap > 0.05 | applied (gap 0.085) | measured |
| Exclusivity | max-probability vs scarcity-weighted | max-probability | measured, equal on OOF |
| Output rule | tuned global threshold vs per-S1 plug-in expected F0.5 | global threshold (0.9666 vs 0.9664) | measured |
| Training data | dev subset only (44.5k S1, 890k pairs) | — | not varied |
| France proxy | leave-India-out | gap 0.12 | measured, not acted on |

The design is sound, and every accept/reject that was made is logged in the gate reports. However, the model
family, hyper-parameters, training size, feature families (cross-encoder, dense, target–target), and the
exact F-measure decision rule were never compared, so "best" is established only among the two Step 5 rules.

## 2. Likely mishaps (fix before tuning anything)

| # | Issue | Evidence | Effect | Fix |
|---|---|---|---|---|
| M1 | **Train/test mismatch in cross-S1 features and exclusivity.** `n_s1_claiming_record` and `margin_to_best_competitor` (the #2 feature) are computed per 25k-S1 chunk, and dev holds only 2% of all train S1. In training, most true competitors for a target are absent; at test time exclusivity runs over all 1.73M S1. The policy (threshold 0.70) is tuned where exclusivity barely fires (90k of 687k records contested) | `features._add_context_features`, `_collect_match_features_chunked`, `submit.run` step 5 | Miscalibrated `margin` on test; threshold tuned for the wrong conflict density | Block the **full train S1 universe** (cached channels make this ~1 h once), compute context features globally, and train/evaluate on val S1 only |
| M2 | **Final model ≠ OOF models.** CV uses early stopping (`best_iteration`); `train_full` uses a fixed 300 rounds. The isotonic map and threshold are fitted on OOF from different models | `train.py:54` vs `train.py:61-68` | Calibration and threshold drift on test | Predict test with the average of the 5 fold models (or retrain with mean `best_iteration`) |
| M3 | **Test IDF comes from the train pool.** Test token-overlap IDF is fitted on US/India train targets, so every French token looks maximally rare | `collect_match_features_test` → `build_word_idf(match_targets_train, …)` | `*_idf` features shift for France (15% of test) | Fit IDF on the pool being scored (test S2/S3; unlabelled inputs, allowed) |
| M4 | **The leave-country-out check measures threshold transfer, not just discrimination.** Leave-India-out AUC stays 0.993 (vs 0.9985) while F0.5 at a fixed 0.5 drops to 0.83; that check uses an uncalibrated `is_unbalance` model | `country_holdout_check` | The 0.12 "France gap" is mostly a calibration/threshold shift, which is fixable | Re-measure with calibrated or rank-relative decisions; report both holdout directions (US-out as well) |
| M5 | `--match-subset train` would leak: its S1 include the ranker's `rtrain10`, whose `p` is overfit | `submit.run` | Over-trusting `p` | Allow only `dev`/`val`, or exclude `rtrain10` |
| M6 | Threshold tuned and scored on the same OOF, on a 0.05 grid | `threshold.tune` | Small optimism (~0.001) | Nested: tune on 4 folds, score on the 5th; finer grid |

## 3. Improvement areas to study (ranked by expected payoff)

1. **Cross-encoder pair model (Ditto-style).** Fine-tune a small transformer on serialized pairs
   (`[CLS] name | addr [SEP] name | addr`), e.g. multilingual-e5-small/base or mDeBERTa-v3-base (MIT).
   Use its probability as a feature for LightGBM, or run it only on the borderline band (0.2 < p < 0.9).
   Study: Li et al. 2020 *Ditto*; Peeters & Bizer 2021 on cross-encoders vs bi-encoders for ER. GPU available;
   the fine-tuned bi-encoder already gained +0.0013 in blocking.
2. **Exact expected-F-measure decisions.** The plug-in estimator uses sum(p) for E[T] and ignores true links
   that blocking never retrieved. Study: Jansche 2007 (maximum expected F-measure), Dembczyński et al. 2011
   (General F-Measure Maximizer, GFM), Nan Ye et al. 2012 (*Optimizing F-measures: a tale of two approaches*).
   With ≤20 candidates per S1, the exact Poisson-binomial DP is cheap.
3. **S1-level singleton model.** 5.6% of S1 are singletons, each worth a full 1.0; singleton accuracy is 94.4%.
   A second classifier on S1-level aggregates (max p, top-2 gap, #candidates above 0.5, name rarity) predicts
   "no match at all" and gates the output.
4. **Collective / cluster features.** A true set is mutually consistent: its S2 and S3 records match each other.
   Add target–target similarity to the S1's current best (coherence), plus correlation clustering or
   connected components across S2/S3. `support_opposite_src_best` is a first step. Study: collective ER
   (Bhattacharya & Getoor 2007), correlation clustering.
5. **Features from the Steps 2–3 upgrades:** fine-tuned dense cosine, N5 `name_min` similarities, the
   noise-catalogue operations as explicit flags (digit dropped vs different house number, inserted-word-only
   difference, homoglyph/alias-only difference).
6. **More training data and tuning.** Train on val (442k S1, 10× dev) with M1's global context; tune with
   Optuna (leaves, min data, regularization, monotone constraints on similarities); drop `is_unbalance` once
   calibration handles priors. Compare CatBoost (Apache-2.0).
7. **Country robustness (France).** Rank- and margin-based features relative to the S1's own candidates;
   per-country calibration evaluated leave-country-out; pseudo-labelled France pairs (exact `name_min_key` +
   same house + same street) as a sanity check of the frozen threshold.
8. **Small LLM adjudicator (optional, ≤8B, Apache-2.0).** Only for the few percent of pairs in the uncertain
   band. Expensive per pair, and it must beat the cross-encoder on dev to justify its cost.

## 4. Suggested order

M1–M3 (correctness, ~2 h) → re-measure the Step 4/5 baseline on v9 blocking → items 5, 6 (cheap) →
item 1 (largest expected gain) → items 2, 3 (decision layer) → item 7 (France) → item 4.

## 5. Status after v2 (2026-09-25)

Current best: `fit2` v2 on val scope (442k S1, 8.84M pairs, 86 features), XGBoost GPU 5-fold, fold-mean,
isotonic, max-probability exclusivity, global threshold 0.72. OOF 0.97984 (nested 0.97981); leaderboard 0.972.

| Item | Outcome | Evidence |
|---|---|---|
| M1 global context | fixed (val-scope universe); test density mismatch found and fixed by S1 hash partitions (M7) | `reports/match/v2.json`; 5x-sparser context costs 0.0009 on val |
| M2 fold-mean scoring | fixed | `train.predict_mean` |
| M3 per-pool IDF | fixed | `Universe._feature_inputs` |
| M4 transfer, both directions | India-out 0.9441 (gap 0.033), US-out 0.9683; oracle threshold recovers only ~0.004 | `v2_transfer*.json` |
| M5 subset guard | fixed by design (val-only training) | |
| M6 nested threshold, 0.01 grid | optimism ≈ 0 (0.97981 vs 0.97984) | `v2_decision.json` |
| Item 5 new features | kept: +0.006 over the dev baseline | `v2.json` |
| Item 2 exact expected F0.5 | rejected: 0.97898 < 0.97984 (plug-in 0.97880) | `v2_decision.json` |
| Item 4 stage-2 coherence stacking | rejected: 0.97975 | `src/er/match/stack.py` |
| Item 7 self-training on pseudo-labels | rejected: India-out 0.9441 → 0.9449 at best | `v2_selftrain_proxy.json` |
| Item 7 near-certain pair acceptance | France 99.2% on test (US 99.5%, India 93.7%) | `france_proxy_v2.json` |
| Item 6 Optuna / CatBoost | not run | |
| Items 1, 3, 8 | not run | |

M7: a val-scope bundle sees ~4.3% of S1 per target pool; test sees 100%, so claims per record rose 23 → 80
and positive margins fell 41% → 27%. `submit2` now counts competitors within density-matched S1 partitions.

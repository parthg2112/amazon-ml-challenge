# Plan: Steps 4–5 fixes, retrain, decision layer, France robustness, submission v2

## Context

Steps 2–3 are done: v9 blocking has a validated ranker@10 oracle F0.5 of 0.99937 on the full 442k validation set. The first real submission passed the validator. Its Step 4 out-of-fold (OOF) score on dev is 0.9737, and France's prediction profile matches US/India.

The Step 4–5 review (`docs/step4_5_review.md`) found correctness gaps (M1–M6) and unexplored methods. The user's order is:
1. Squash the three commits from the atomic-cache fix onward (done: `e7bab63`).
2. Fix review items 1–3.
3. Retrain Step 4 with the new blocking features.
4. Improve the decision layer.
5. France robustness.
6. Test inference and the submission package.

This plan lists every improvement already identified so none is dropped.

Constraints:
- Commit messages are one line, max 20 words.
- Cache writes are atomic; never delete useful caches.
- Heavy jobs run detached (`artifacts/job.cmd`), because the memory safeguard flag isn't active.
- The Step 4 classifier trains **only on val S1** (442k). `rtrain10` S1 have an overfit ranker `p`, and most other train-fold S1 were seen by dense fine-tuning.
- Tools: 11 threads (`ER_THREADS`); ~26 GB free disk.

## Phase A: correctness fixes (review M1–M6)

| # | Fix | Where |
|---|---|---|
| A1 (M1) | **Full-universe context.** Two-pass pipeline. Pass 1 caches each chunk's ranked shortlist (`s1,target,src,p,frank` + provenance + `_ctx_score`) for **all 2.2M train S1**. Chunk order reuses the cached val and rtrain10 channel chunks, so ~64 chunks are new (~65 min once). Pass 2 computes `n_s1_claiming_record` and `margin_to_best_competitor` **globally**, then full features only for the S1 that need them. Test gets the same two passes over all 1.73M test S1 | new `src/er/match/dataset.py`; `features.py` split into `build_local()` / `add_local_context()` / `global_context()`, with `build()` kept for the tests |
| A2 (M2) | Test predictions = **mean of the 5 CV fold models**, the same models the isotonic map and threshold were fitted on. `train_full` is retired | `match/train.py`, submit v2 |
| A3 (M3) | IDF fitted on **the pool being scored** (test targets for test) | `dataset.py` |
| A4 (M4) | Leave-country-out re-measured in **both directions** (India-out and US-out), with calibrated and threshold-tuned decisions; reports AUC/PR-AUC separately from F0.5 | `train.country_holdout_check` |
| A5 (M5) | `--match-subset` restricted to `val`/`dev`; errors if S1 overlap `rtrain10` or dense-FT pairs | `submit.py`, CLI |
| A6 (M6) | Nested threshold tuning (tune on 4 folds, score on the 5th) and a finer grid (0.01) | `decide/threshold.py`, `pipeline.py` |
| A7 | Ranker model cache (pickle keyed by config hash), so commands stop re-training it (~5 min each) | `blocking/runner.py` |

## Phase B: retrain Step 4 on v9 (item 3)

New features, added to `features.build_local` and `extra`:
- **dense cosine** from the fine-tuned e5 embeddings (row maps built from `io.load_source` order);
- **N5**: `sim_min_tset`, `sim_min_ratio` on `name_min`, and a `name_min_key` exact match;
- **noise-catalogue flags**: house number with a digit dropped, a digit changed, or a different number; name differing only by generator noise (`name_min_key` equal while `name_core` differs);
- LambdaRank score `p` and channel provenance (already present).

Train with CV on val (442k S1, ~8.8M pairs) using global context. Tuning:
- LightGBM with Optuna, a small budget: leaves, min data, L1/L2, feature fraction;
- drop `is_unbalance`;
- monotone constraints on the similarity features;
- compare CatBoost (Apache-2.0).

Report OOF macro F0.5 per country against the 0.9737 dev baseline.

## Phase C: decision layer (item 4)

- Re-tune the global threshold on the new OOF, nested per A6.
- **Exact expected F0.5**: a Poisson-binomial dynamic program per S1 over ≤20 candidates, i.e. Jansche 2007 / General F-Measure Maximizer (Dembczyński 2011). It replaces the plug-in estimator; compare against the threshold.
- **S1-level singleton model**: S1 aggregates (max p, top-2 gap, count above t, name rarity) predict "no match". Gate the output and tune the gate on OOF.
- Re-check exclusivity (max-probability vs scarcity-weighted) now that it runs over the full universe.
- Keep whichever rule wins on nested OOF; log the rejected ones in the gate report.

## Phase D: France robustness (item 5)

- Rank- and margin-relative features: similarity minus the S1's best, and rank within the S1. These transfer across countries better than absolute values.
- Per-country calibration diagnostics from A4; if they diverge, use a country-agnostic calibration fitted on pooled OOF, plus a rank-relative decision rule.
- **France pseudo-labels**: near-certain test pairs (`name_min_key` equal, same house number, same street tokens). Measure the frozen model's recall and the fraction above the threshold on them; flag if France departs from US/India.
- France city → region table learned from unlabelled test S2/S3 records that carry both (targets currently find a region only 64–66% of the time). Additive normalization column; measure the region-state feature.

## Phase E: optional, larger-gain research (after A–D, each kept only if validated)

1. **Cross-encoder** (Ditto-style): fine-tune multilingual-e5-small or mDeBERTa-v3 on serialized pairs from val-excluded train S1. Use it as a feature, or only on the uncertain band (0.2 < p < 0.9).
2. **Collective / cluster features**: target–target similarity within an S1's predicted set (S2↔S3 coherence), and correlation clustering.
3. Dense fine-tune round 2 with **hard negatives** mined from blocker candidates (loss saturated at 0.003 with in-batch negatives).
4. N4 v2: fold-similarity token alignment for unequal token counts; character-level transliteration model.
5. Small LLM adjudicator (≤8B, Apache-2.0), only on the uncertain band, only if it beats item 1.

## Phase F: test inference and package (item 6)

- `er submit` v2 with the Phase A–D pipeline, reusing cached test channels and shortlist. Run `validate_submission.py` and compare per-country empty rate and matches per S1 against train truth.
- Package: `output/{matching_results,candidate_pairs}.tsv`, `code/business_entity_resolution/` (src, README, `requirements.txt` from `uv export`), and the filled `Documentation_template.md` (methodology from PLAN.md and the gate reports).
- Update PLAN.md (Steps 4–5 sections), commit (subject ≤20 words), push when the user asks.

## Verification

- `uv run pytest`: existing 40 tests plus new ones:
  - global context equals single-chunk context when all S1 are in one chunk;
  - fold-average prediction;
  - exact-F DP against brute force on small cases;
  - IDF pool selection;
  - the subset guard.
- Each phase: nested-OOF macro F0.5 on val (overall, US, India, singleton accuracy) and leave-country-out in both directions. Keep a change only if nested OOF improves.
- Final: validator PASS, and test prediction profile per country consistent with train truth (empty rate ≈5.6%, ~3.3–3.5 matches per S1).

# R&D cross-evaluation prompt: entity-resolution scorer, twin-separation problem

You are a senior ML R&D agent cross-evaluating an entity-resolution pipeline. Your job: challenge our
solution choices for three quantified failure modes, propose the strongest alternatives, rank them by
expected score gain per engineering cost, and design the cheapest experiment that would prove or kill
each one. Be adversarial with our reasoning; be concrete with numbers.

## Task and metric

- Match Source-1 business records to Source-2/Source-3 records (same real-world entity). Outputs are
  per-S1 lists of matched target IDs; empty list allowed (true singletons score 1.0).
- Metric: **macro F-beta with beta=0.5, averaged per S1 record** (singletons included). Precision
  weighted 2x over recall: F0.5 = 1.25PR / (0.25P + R). One false merge drags a record's F0.5 hard
  (a 3-link record with 1 FP drops 1.0 -> 0.79).
- Test set: 1.73M S1 records, 15% in a country never seen in training (France). Train: 2.2M S1
  (US ~60%, India ~40%), 5.8M targets, 34.6M blocked candidate pairs at test, 8.8M at val.

## Current architecture (all steps validated, ceiling measured)

1. Normalization (name_min_key, addr parsing, script folding, noise catalogue flags).
2. Blocking: multi-channel candidate generation + XGBoost ranker (k=10 shortlists). Measured recall
   of true links in shortlists: 99.78% (rank<=6: 99.60%). Blocking ceiling: F0.5 0.99937. NOT the
   bottleneck.
3. Scoring (Step 4/5): 122 features (rapidfuzz sims, char-3 cosine, IDF token overlap, house-number
   edit flags, region/postal/city/unit states, legal-form relation, dense-embedding cosine from a
   fine-tuned multilingual e5, competitor-context counts, per-S1 relative gap/z-scores, pool-level
   name-support counts, twin-group addr/p ranks) -> XGBoost GPU 5-fold CV + isotonic calibration +
   fold-mean prediction.
4. Decision: per-record max-probability exclusivity (targets are exclusive: truth max claims per
   target = 1; S1s may link up to 5 S2 / 6 S3) + a policy auto-selected on OOF (currently a
   link-count-prior-tilted exact-expected-F0.5 rule beating a tuned global threshold 0.98138 vs
   0.98122).
- Current: val OOF macro F0.5 = **0.98138** (loss 0.0186 vs blocking ceiling). Leaderboard 0.9742
  (US+India ~0.978, France ~0.955, gap France-vs-US+India ~0.022).

## The three quantified failure modes (val OOF, v5a)

| # | Finding | Numbers | Val F0.5 lost |
|---|---|---|---|
| 1 | Loss concentrates on low-count records (corrected: the original "3-link, 52%" bucket was clipped, i.e. T≤3) | Per-T loss: T=0 0.00142 · T=1 0.00365 (F0.5 0.932, worst per record) · T=2 0.00375 · T=3 0.00381 · T=4 0.00297 · T≥5 0.0032; recall-driven (58k FN vs 11.9k FP overall) | T≤2 ≈ 0.0088 (47%) |
| 2 | Twin problem (identical normalized names) | 1.41M identical-name pairs: 773k TP / 4,379 FP / **25,815 FN (median score 0.33!)** / 610k TN correctly rejected. Twins = 44% of all FN, 37% of all FP | ~0.0065 |
| 3 | Confident false positives | 8,833 of 11.9k FPs scored >= 0.69 (3,885 >= 0.9). Not a threshold artifact: raising cutoff costs more recall than it saves (F0.5-vs-t curve peaks at 0.69, nested-CV optimism ~0) | ~0.0037 (partially overlapping #2) |

Twin economics detail: trusting all identical-name pairs would add 25,815 TP but 609,792 FP
(4% precision). Promoting identical-name pairs with score >= 0.3 adds ~14.3k FP. So blanket rules
fail; the signal must be *within-group discrimination*: which twin is the true partner given address,
context and competition evidence.

## Already tried and rejected (with evidence — do not re-propose without a new angle)

- Cross-encoder on raw pair text (Ditto-style, US-only training): India-held-out AUC collapses
  0.9977 -> 0.978 vs 0.9991 for the feature model; rejected for unseen-country transfer.
- Self-training on confident pseudo-labels: labels 99.5% correct but +0.0008 only (reinforces existing
  beliefs). Untested variant: selecting UNCERTAIN pairs (0.3-0.69 band) for pseudo-labeling.
- Monotone constraints: -0.007 in-distribution. Optuna 30-trial hyperparameter search: no full-frame
  gain. LightGBM: same score as XGBoost, 5x slower. Stage-2 stacking with candidate agreement: no gain.
- Per-record exact expected-F0.5 decision rules: lose to tuned threshold in-distribution (calibration
  imperfection); a link-count-prior-tilted variant currently wins by +0.0002.
- France-only threshold, France link-count prior matching: rejected (flat curves; France's lower link
  rate is genuine — 5.53 targets/S1 vs 5.8).
- Greedy max-prob exclusivity loses only 172 true links to exclusivity — global assignment at decision
  time on current scores is near-worthless. Any assignment idea must change the SCORES, not the argmax.
- Label noise: none found (0 duplicate pairs, 0 over-cap records, max 1 claim per target).

## In progress (evaluate whether these are the right bets)

- v6: twin-group features (addr/p rank + gap within each S1 x name-key group, twin count), +38%
  training data (val 442k + 167k clean-OOF rtrain S1s, streaming per-fold CV), legal-form feature
  dropped (it hurt unseen-country transfer +0.0038 when removed, in-dist cost -0.0002).
- Queued: cost-sensitive retraining upweighting the confidently-wrong buckets (FN<0.3 and FP>0.9).

## Constraints

- 16 GB RAM, one RTX GPU (16 GB), Windows. XGBoost GPU ~2,100 pairs/s train, 9.4k pairs/s score.
- 4 leaderboard submissions/day. Each submission ~30 min to score.
- France/unseen-country robustness matters: features that learn country-specific surface forms do not
  transfer (proven by the cross-encoder failure and feature-group ablation). Country-agnostic
  similarity/relative features do transfer.

## Questions for you (rank answers by expected F0.5 per engineering hour)

1. Twin separation (#2, ~0.0065): beyond our within-group rank features — what is the strongest
   architecture for "which twin is the true partner"? Candidates we want challenged: a specialist
   twin-classifier trained only on identical-name pairs; a group-conditional second-stage rescorer;
   learning-to-rank losses within groups; target-claim competition features (how many other S1s claim
   this target and with what evidence); anything from record-linkage literature on within-block
   disambiguation. What would you build first and how would you validate it cheaply on OOF?
2. 3-link records (#1, ~0.0124): recall-side fixes. Is a per-record count-conditional decision rule
   (estimated link count -> record-specific cutoff) viable where global prior-matching failed? Cost-
   sensitive or focal-style weighting in XGBoost — practical recipe? Other recall levers that don't
   raise FP?
3. Confident FPs (#3, ~0.0037): what makes a model score a false twin >= 0.9, and what specifically
   fixes calibration-quality (not threshold-level) errors? Margin-to-competitor features? Per-band
   calibration? A verifier model only on high-p predictions?
4. Cross-check our priors: which of our "rejected" conclusions do you dispute, and what single number
   would falsify your dispute?
5. Anything structural we are missing entirely (loss functions, training-data construction, semi-
   supervised angles, candidate-set-level models) that could plausibly be worth >= 0.002?

Deliverable: a ranked table (solution, expected val F0.5 gain, cost, falsification experiment), then
a one-week execution order. Optimize for the macro-F0.5-per-record metric INCLUDING singletons, and
for transfer to an unseen country. Flag any proposal whose gain depends on labels we will not have
(France).

# Frontier R&D prompt: the "unrecoverable" slices and the unseen-country gap

You are a research scientist in probabilistic record linkage, domain adaptation and decision theory. Our entity-resolution pipeline is mature (val macro F0.5 0.983). The remaining loss sits in slices we have labelled "unrecoverable" or under-explored. Attack them with experimental, high-variance ideas. For each idea give:
- the mechanism on THIS data;
- the arithmetic of the maximum gain;
- the cheapest experiment that proves or kills it within 2–4 hours on our hardware.

Be adversarial toward our ceilings: for each one, name the measurement that would prove it wrong.

## 1. Task, metric, rules
- For each Source-1 (S1) business record, return all Source-2/Source-3 records of the same entity. An empty list is allowed.
- **Metric: macro F0.5 per S1**, F0.5 = 1.25PR/(0.25P+R), averaged over all S1 including singletons.
  - A true singleton scores 1.0 for an empty prediction and 0.0 for any prediction.
  - A linked S1 with an empty prediction scores 0.
  - Predicting 3 with 2 of 2 true scores 0.714; one extra FP on a 1-link record scores 0.556.
- Public leaderboard = a subset of test; the final ranking uses the private remainder. Nothing tuned to public-LB noise.
- **Hard rules:**
  - No external data, APIs, geocoders, gazetteers or internet augmentation (disqualification).
  - Country is an open set: nothing hard-coded to {US, India}. Any per-country behaviour must be derived generically from that country's unlabeled data, so it would work for a fourth unseen country.
  - Models MIT/Apache-2.0, ≤8B parameters. The code must reproduce end-to-end.
  - No signals from `entity_id` numbering or file row order.
- **Compute:** 1× RTX 16 GB GPU, Windows, consumer RAM (large jobs stream).
  - XGBoost 5-fold refit on ~12M pairs: ~25 min.
  - Full test scoring (34.6M pairs): ~30 min.
  - Cross-encoder: ~2.1k pairs/s train, ~9.4k pairs/s score.
  - 4 submissions/day; ~1 day left. Tag ideas **[≤4h]** or **[later]**.

## 2. Data facts
- Raw fields only: `entity_id, business_name, business_address, country`. No phone, website or category.
- **Train:** 2.21M S1 (US 1.32M, India 0.88M), 10.3M S2+S3 targets (4.68/S1).
- **Test:** 1.73M S1 (US 663k, India 810k, **France 259k = 15%, never seen in training**). Targets/S1: US 5.76, India 5.82, France 5.53.
- **Link prior (train truth):**
  - mean 3.46 links/S1; P(T=0)=5.6%, P(T=1)=5.4%, P(T=2)=17.0%, P(T=3)=24.1%; caps S2≤5, S3≤6;
  - identical in US and India (3.459 vs 3.465);
  - targets are exclusive (≤1 owner);
  - train owned-fraction of targets is 0.74. If the prior holds at test, test owned-fraction ≈ 0.60, i.e. ~1.9× more orphan targets per S1 than train.
- **The noise looks synthetic:** generator-style names ("vioonyxaria", "u j talxabe"), house numbers one digit apart, dropped components, typos, legal-suffix swaps, transliteration.
- **France surface forms:**
  - "10 Rue du Sud, Dunkerque, Hauts-de-France" · "N° 28 RUE DE BAPAUME, LILLE, Nord" · "2 R. André Huys, Roubaix, Nord" · "Gironde, 5 All Des Jardins De Caillavet, Mérignac";
  - names: "Dunkerque Ehpad SARL", "Bordeaux Itsittu [SAS]", "Comité dês Action Distribution".
- **France density:**
  - S1 sit in 3 regions (Hauts-de-France 39%, Nouvelle-Aquitaine 33%, Pays de la Loire 28%); Bordeaux+Nantes+Lille = 44% of records;
  - 34% of France S1 names repeat exactly (US 28%, India 43%);
  - postal codes in 0.4% of France addresses;
  - France S2 region present 64% (US/India 97%).

## 3. Pipeline and scores
1. **Normalization:** open-set country profiles (hand-written street types/regions for US, India, France; unknown → generic), script folding, noise-catalogue flags.
2. **Blocking:** 9 channels (exact/fuzzy name keys, name TF-IDF, address char n-grams, name+address, fine-tuned multilingual-e5-small dense) + LightGBM lambdarank → top-10 per S1. Pair recall 99.776% (99.94% before the cut); blocking ceiling F0.5 0.99937.
3. **Scorer:** 122 features → XGBoost GPU 5-fold on 618k S1, fold-mean prediction. Features:
   - rapidfuzz family, char-3 TF-IDF cosine, IDF token overlap/extras;
   - house-number edit flags; tri-state region/postal/locality/unit;
   - dense cosine; per-S1 relative gap/rank;
   - cross-S1 competition counts over the 618k-S1 training universe (test uses 3 density-matched S1 partitions), pool-level name support, twin-group ranks.
4. **Band rescoring:** pairs with 0.2<p<0.9 rescored by logistic([logit p, logit CE]). The CE is a cross-encoder fine-tuned from multilingual-e5-small on US pairs.
5. **Decision:** max-probability target exclusivity, then a per-record rule auto-selected on OOF (tuned threshold ≈0.70–0.73, prior-tilted exact expected F0.5, or a hybrid).

**Scores:**
- Val OOF (442k S1): 0.98122 (v5a) → 0.98190 (v6) → **0.98297 (v6 + CE band)**.
- Leaderboard (v5a): 0.9742; US+India ≈ 0.978, **France ≈ 0.955**.
- Unseen-country proxy (fit without India, apply frozen): India 0.9515 vs 0.9795 in-distribution (AUC 0.9976 vs ~0.999); US-out 0.9716.

## 4. Slices to attack (measured on val OOF)

### S-A: France / unseen country (~0.0035 LB, the largest LB pot)
**France test profile vs US/India** (all label-free):

| | France | US/India |
|---|---|---|
| predicted links/S1 | 3.12 | 3.35–3.37 |
| predicted empty | 6.4% | 5.8% |
| uncertain-band (0.3–0.69) pairs per S1 | **0.232** | 0.12–0.16 |
| pairs ≥0.9 per S1 | 3.00 | 3.21–3.28 |

Two independent upstream estimates of France's true link rate agree:
- record-count: 5.53 targets/S1 × 0.60 owned ≈ **3.31**;
- pre-decision Σp/S1 = **3.33**. Σp is unbiased on val: India 3.462 vs 3.469 true.

France predicts 3.12, i.e. 94% of expected vs 97% for US/India. So France is under-linked by ~0.2 links/S1 beyond the normal F0.5 conservatism.

**Known:**
- Leave-country-out AUC barely moves while F0.5 collapses. The oracle threshold recovers only +0.0037 of India's 0.028 gap.
- Dropping legal-form features: +0.0038 on India held-out. Per-S1 relative features: +0.003.
- A raw-text cross-encoder as a global GBDT feature **lost** (India CE AUC 0.978). As a band-only rescorer it **gains** +0.0009 on India with a US-only fit.

**Falsified:** France city is almost never parsed (a bug), but blanking city and 36% of regions on val costs only −0.0001. City is not the gap.

### S-B: Address-less popular-name twins (~0.0034 val; we call it unrecoverable)
- 72% of FN twins (identical normalized names) have an empty address on one side (TP twins 2.7%, TN twins 69%). Pair features see identical names and nothing else.
- Address-less identical-name pairs by number of S1s claiming the target:

| Claimants | Pairs | Base rate | FN at t=0.69 |
|---|---|---|---|
| 1 | 6.5k | 68% | 668 |
| 2 | 9.2k | 55% | 1,167 |
| 3–5 | 30k | 41% | 3,908 |
| 6–20 | 296k | 7.2% | **11,377** |
| >20 | 447k | 1.6% | **5,765** |

- 25% of all FNs have a *different* S1 scoring their target higher.
- 50% of FPs target orphans (no S1 owner); 46% target records whose owner S1 lies outside the val universe (present at test).

### S-C: Shortlist misses (~0.0006)
3,428 of 1,531,442 true links never reach the top-10. Recall@3 93.9%, @5 99.40%, @8 99.72%, @10 99.776%.

### S-D: Low-count records (decision context)
- Loss by true count: T=0 0.00142 · T=1 0.00365 (F0.5 0.932) · T=2 0.00375 · T=3 0.00381 · T=4 0.00297 · T≥5 0.0032.
- A 1-link miss costs the whole record. In those misses the true target is top-1 in its source 86% of the time, at median p 0.27.

## 5. Tried and rejected: don't re-propose without a new angle
- Cross-encoder as a global feature (above).
- Self-training on confident pseudo-labels (+0.0008).
- Monotone constraints (−0.007).
- Optuna, 30 trials (0).
- LightGBM (same score, 5× slower).
- Stage-2 S2↔S3 dense-coherence stacking (0.97975 vs 0.97984).
- Exact expected F0.5 under independence (loses to threshold).
- France-only / prior-matched thresholds (flat / −0.007).
- Group DRO (untestable: the transfer proxy trains on one country), beta calibration (a monotone map with a re-tuned threshold gives identical decisions), conformal thresholds (need France labels).
- Decision-time global assignment on current scores: only 172 links are lost to greedy exclusivity, so assignment must change the *scores*, not the argmax.
- **Falsified this week** (don't re-propose):
  - blending GBDTs: residual correlation 0.977; a v5a+v6 blend loses 0.0001. The CE's residual correlation is −0.02, which is why it helps;
  - beta/Platt/segment-specific calibration under the tilted rule: −0.00004 to +0.00007;
  - directional address-empty flags: S1 addresses are never empty, so it's a no-op;
  - "trust the address when the name is garbled": that slice is 6.1% true;
  - identical-text FNs as a bug: they're identical-text orphan duplicates (88% true at 1 claimant);
  - city parsing as the France gap: −0.0001.
- Where the p<0.3 FNs sit: 72% address-empty, 46% address-less twins with >5 claimants, 43% out-scored by another S1.

## 6. Seed hypotheses (challenge, extend or kill)

**Area A: France**
- **A1. Diagnose before fixing.** Per-feature domain-classifier AUC (France vs US/India candidate pairs), to find which features shift. Split France's band mass into twin density vs feature shift vs dense-embedding shift.
- **A2. Simulation-based adaptation.** Estimate the train noise generator per field (abbreviation swaps, legal-suffix changes, typos, drops, reorders) from train truth pairs. Apply it to France S1 to synthesize labeled French positives, plus hard negatives (same name/street, other entity). Fine-tune the CE / dense model / GBDT. **Falsify:** learn the noise from US only, synthesize India pairs, measure India-held-out closure (0.9515 → ?).
- **A3. Covariate-shift weighting** by density ratio P(France|x)/P(train|x) on country-agnostic features, or invariance-driven feature selection. Judge by the India-held-out gap.
- **A4. Test-time posterior adaptation.** EM prior re-estimation (Saerens 2002) or BBSE on France scores; a generic per-country policy that maximizes expected F0.5 under adapted posteriors. Falsify on India held-out.
- **A5. Structure as teacher.** Pseudo-labels from constraints, not confidence: exclusivity + per-S1 count prior + S2↔S3 mutual support (an S2 and an S3 that each match the S1 and each other). Co-train across the S2/S3 views, focused on the uncertain band.
- **A6. Unsupervised French normalization.** Mine rewrite rules from aligned near-duplicates ("R."↔"Rue", "All"↔"Allée", "N°", bracketed legal forms, diacritic noise "dês"). Measure token-alignment gain on France near-sure pairs.
- **A7. France blocking recall** (unmeasured). Use the leave-India-out ranker recall@10 as a proxy. Adaptive k for flat shortlists.

**Area B: address-less twins**
- **B1. Hidden-signal hunt.** Mutual information between y and residual raw-form features inside the slice: casing, punctuation, legal-suffix variant, token order, diacritics, S2 vs S3, name length. Does the generator leave entity-level signatures shared across an entity's copies?
- **B2. Generative copy model.** Estimate P(#copies per source | entity), P(address dropped | copy), and correlation of drops across copies. Derive ownership likelihoods for address-less copies from each claimant's already-confident copies (capacity), then compute joint marginals via Sinkhorn / belief propagation on the claimant × target graph with an orphan node.
- **B3. Joint expected-macro-F assignment.** Across competing S1s, not per-S1 greedy. Bound it with oracle-capacity experiments.
- **B4. Information ceiling.** The Bayes-optimal expected F0.5 per slice under perfect calibration, given only the available fields. Tells us exactly where to stop.
- **B5. Sibling-conditioned evidence.** Raw-form similarity of an address-less copy to this S1's confident copies vs to rival S1s' copies.

**Area C: shortlist misses**
- **C1.** Characterize the 3,428 misses (rank, channel coverage, twin/empty-address share).
- **C2.** Sibling-propagated retrieval: query with confident S2 copies' text to reach S3 (and vice versa). Adaptive k when a record-count model predicts more links than confident candidates. Report extra pairs per recovered link.

**Area D.** Anything structural we're missing worth ≥0.0005 LB: dependence-aware set decisions (GFM with learned joint marginals), second-stage direct macro-F surrogates, robustness to the public/private split.

## 7. Deliverable
1. **Per idea:**
   - mechanism on this data;
   - max-gain arithmetic (slice size × recoverable fraction × per-record F impact);
   - P(success) and hours;
   - cheapest falsification (slice, metric, kill threshold);
   - whether it needs labels France won't have;
   - rule risk.
2. **Ranked table** by expected LB gain × P(success) / hour, split into [≤4h] and [later].
3. **Per "unrecoverable" slice:** the Bayes ceiling given the available fields, and the evidence that would raise it.
4. **A 24-hour order** within 4 submissions/day, marking each step as decision-only (minutes, re-decide cached test scores), re-score (~30 min) or re-fit (~25 min GPU).
5. **Ideas to abandon**, each with the number that kills it.

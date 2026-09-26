# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [fill]  
**Team Members:** [fill]  
**Submission Date:** [fill]

---

## 1. Executive Summary

A four-stage pipeline: country-aware normalization, a 9-channel blocker with a learned ranker (99.78% of true links kept in
a top-10-per-source shortlist), a 122-feature XGBoost scorer whose uncertain band (0.2 < p < 0.9) is rescored by a
cross-encoder plus sibling-consensus evidence, and a decision layer that renormalizes contested targets and picks each
record's link set by exact expected F0.5 under a link-count prior. Out-of-fold macro F0.5 on a 477k-record validation set
is 0.9856. Everything is country-agnostic by construction, so it applies unchanged to the unseen test country (France).

---

## 2. Methodology

### 2.1 Problem Analysis

- **Four raw fields only** (id, name, address, country). Noise is generator-like: abbreviations, legal-suffix swaps,
  typos, word-order swaps, dropped address components, one-digit house-number edits, and name replacement by pseudo-word
  trade names.
- **The link-count distribution is sharp and country-invariant**: mean 3.46 links per Source-1 record, 5.6% singletons,
  at most 5 Source-2 and 6 Source-3 links, identical in US and India (3.459 vs 3.465). Every target record belongs to at
  most one Source-1 record.
- **Where the loss sits**: low-count records dominate. A missed link on a 1-link record costs the whole record, so 1-link
  records are the worst bucket (F0.5 0.932 before the band rescorer). Most remaining misses are identical-name "twins"
  whose target record has no address: with only name and country available, the owner among 6+ same-name businesses is
  not identifiable (base rate ≤7%).
- **Test differs from validation**: about 1.9× more unowned target records per Source-1 record, 3–20× more contested
  targets (strongest in France), and an unseen country whose scores are flatter (twice the uncertain-band mass).

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned ranker → gradient-boosted pair classifier → cross-encoder band rescoring →
set-level decision layer (hybrid).  
**Core Innovation:** Treat the problem as a per-record set decision under exclusivity. Pair scores are rescored only
where they are uncertain, contested targets are renormalized to an at-most-one-owner posterior, and each record's
link set maximizes expected F0.5 under the known link-count prior. The same machinery transfers to an unseen country.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys / channels (9):** exact joins on normalized name keys (concatenated, folded, minimal key) and on
  house-number tokens; word-level TF-IDF on names; character 3-gram TF-IDF on addresses and on name+address; a dense
  retriever (multilingual-e5-small, MIT, contrastively fine-tuned on training-fold pairs only).
- **Fusion and ranking:** candidates from all channels are fused and re-ranked by a LightGBM lambdarank model over
  channel ranks and scores. The top 10 per source per Source-1 record form the shortlist.
- **Candidate pairs generated:** 34,650,880 on test (1,732,544 Source-1 records, 20 per record).
- **How true matches were kept:**
  - Measured on a 442k-record validation holdout: the channel union keeps 99.94% of true links, the top-10 shortlist
    keeps 99.78%;
  - the resulting ceiling is macro F0.5 0.99937, so blocking costs about 0.0006;
  - ten blocking versions were compared on this recall-ceiling metric before freezing.

---

## 4. Matching Model

**Features used (122):**
- **Name:** rapidfuzz ratio / partial / token-sort / token-set, Jaro-Winkler, character-3-gram TF-IDF cosine,
  IDF-weighted token overlap and unmatched-token mass, exact normalized-key flags, noise-only-difference flags,
  alias matches, script mismatch.
- **Address:** token-set and ratio similarity, character-3-gram cosine, IDF token overlap, house-number state and edit
  flags (digit dropped / changed), number Jaccard, tri-state region / postal / city / unit agreement, empty-address flags.
- **Other:**
  - dense-embedding cosine and the blocker's channel ranks and scores;
  - per-record relative views (gap to the record's best candidate, z-score, rank);
  - competition context (how many Source-1 records claim the target, margin to the best rival);
  - pool-level name support (how common the name is);
  - twin-group ranks among same-name candidates.
  - The legal-form feature group is dropped because it hurt unseen-country transfer.

**Model type:**
- **Pair scorer:** XGBoost (GPU), 5-fold CV grouped by Source-1 record, trained on 618k records. Test predictions average
  the fold models. Competition features on test are computed within density-matched record partitions, so they match
  the training distribution.
- **Band rescorer:** for 0.2 < p < 0.9, a logistic model on [logit p, tanh(cross-encoder logit / 2), sibling-consensus
  features]. It is cross-fitted by fold and excludes every record the cross-encoder was trained on.
  - The cross-encoder is multilingual-e5-base (278M, MIT) fine-tuned on 1.42M US+India pairs from the ranker's training subset,
    never the validation records.
  - Sibling consensus: a band target whose house number differs from the Source-1 record's but equals that of a
    confident sibling link is 89% true.

**Threshold selection method:**
- **Contested targets:** renormalized to an at-most-one-owner posterior p / (1 + Σ rival odds).
- **Exclusivity:** each target goes to its highest-probability claimant.
- **Link set:** each record's set is chosen by exact expected F0.5 (Poisson-binomial dynamic programme), with the count
  posterior tilted onto the empirical link-count prior.
- **Rule selection:** the rule is auto-selected on out-of-fold predictions against a nested-CV tuned global threshold
  (optimism ≈ 0).

---

## 5. Results & Error Analysis

- **F_0.5 score (macro, out-of-fold, 477k validation records not seen by the cross-encoder):** 0.9856.
  - Progression: 0.9812 (base scorer) → 0.9819 (twin features + 40% more training data) → 0.9837 (band rescoring with a
    US-only cross-encoder, + sibling consensus, + contest renormalization) → 0.9853 (US+India cross-encoder,
    118M) → 0.9856 (278M cross-encoder).
- **Unseen-country rehearsal:** fit on US only and score India. Band rescoring lifts India from 0.9511 to 0.9588 and
  the full decision stack to 0.9595.
- **Common false positives:** near-twin decoy businesses at almost the same address (house 4445 vs 4448); unowned
  duplicate listings with byte-identical name and address; contested targets where a rival record is the true owner.
- **Common false negatives:** identical-name twins whose target has no address and whose name is shared by many
  records (not identifiable from the available fields); trade-name copies whose name was replaced by a pseudo-word
  while the address matches (the model is calibrated on these: 69% true at mean p 0.69).

---

## 6. Conclusion

High recall comes from a cheap multi-channel blocker with a learned ranker. Most of the remaining score came from
treating matching as a set decision: rescoring only the uncertain band, renormalizing contested targets, and choosing each
record's link count against a country-invariant prior. The main lessons were to measure every idea against a held-out
country, and to find where the loss actually sits before building anything: a third of the misses are not identifiable
from the given fields.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` ships `src/er/` (package), `configs/`, `README.md` and `requirements.txt` (pinned via
`uv export`). The README lists the end-to-end command chain:
- `prepare` → `split` → `normalize` → `finetune` → `fit2` → `ce-train` → `ce-band` → `submit2` → `decide-test`;
- it produces `output/matching_results.tsv` and `output/candidate_pairs.tsv`, which pass
  `student_resource/utils/validate_submission.py`.

**Key modules:**
- `src/er/normalize/` (normalization, open-set country profiles);
- `src/er/blocking/` (channels, fusion, ranker);
- `src/er/match/` (features, streaming CV scorer, cross-encoder, band rescorer, transfer rehearsal);
- `src/er/decide/` (exclusivity, expected-F0.5 rules, link-count prior).

### B. Additional Results

Per-count loss, twin analysis, transfer ablations and the full experiment log are in `docs/step4_5_deep_dive.md`,
`docs/rnd_frontier_plan.md` and `reports/match/*.json`.

**Compliance:**
- no external data or lookup services;
- all models are MIT / Apache-2.0 and ≤ 8B parameters (largest: 278M);
- country is treated as an open set, with no country-specific branches.

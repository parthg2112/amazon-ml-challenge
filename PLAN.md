# Entity Resolution Plan — Steps 1–3 (normalization and blocking)

Status: Steps 1–3 implemented in Python. Every number below is measured on the entity-level holdout unless labelled otherwise. dev = 10% of validation (44,535 S1); val = the full 20% holdout (442,233 S1). Queries always search the full training S2+S3 pool (10.32M records), as at test time. Run history lives in `reports/experiments/*/metrics.json` and `reports/leaderboard.md`.

## 1. Constraints that shape everything

- **Metric.** Macro F0.5 per S1 over all S1 rows, including singletons (empty/empty = 1). There are 7.64M true links, 3.46 per S1; 5.6% of S1 are singletons, and the all-empty baseline scores 0.0559.
- **Open-set country.** Train has US and India. Test adds France (15% of test S1) with no labels. Country selects a profile; it never filters records.
- **Scale.** S1 2.2M, S2 5.0M, S3 5.3M in train; test S1 1.73M, S2 4.9M, S3 5.1M. All-pairs comparison is impossible, so everything is streamed, sharded, or chunked.
- **Machine.** 12-core Ryzen 9 7900X, 31 GB RAM, RTX 4070 Ti SUPER 16 GB. Rule learned the hard way: run one 12-process job at a time. Two concurrent pools crashed on Windows.

## 2. Stack (replaces the Perl/SQLite prototype)

| Concern | Tool | Why |
|---|---|---|
| IO, joins, metrics | Polars (Rust, all cores) | 2.5 GB of TSV → Parquet in 3 s; exact-key blocking is hash joins |
| Normalization | Python + `ProcessPoolExecutor(12)` over 250k-row Parquet shards | 12.5M records in 66 s |
| Transliteration | `anyascii` (ISC) | Native-script names → Latin |
| Sparse retrieval | hashed TF-IDF (`HashingVectorizer` via joblib) + `sparse_dot_topn` (12 threads) | Exact cosine top-k without a vocabulary fit |
| Dense retrieval | `multilingual-e5-small` (MIT, 118M) fp16 + torch matmul top-k on GPU | Script- and language-agnostic |
| Stage-2 ranker | `rapidfuzz.process.cpdist` (multi-threaded) + sklearn HistGradientBoosting | Orders the union so a small budget keeps the true links |

## 3. Step 1: scorer and split (done, parity-verified)

`er.metrics` implements the leaderboard formula. The PROBLEM_STATEMENT worked example (0.714) and six boundary cases are in `tests/test_metrics.py`. `er.split` reproduces the retired Perl SHA-256 split byte for byte: validation counts, per-country counts, cardinality histogram, and all-empty F0.5 all match `reports/step1_metrics.json`; see `reports/step1_split.json`.

## 4. Step 2: normalization — approaches and measured value

Each approach adds a view; raw text is always kept. Numbers are **exact key agreement on 1.53M held-out true pairs** (`er diagnose --subset val`, reports in `reports/normalization/`).

| ID | Approach | Measured effect | Decision |
|---|---|---|---|
| N0 | NFKC, accent strip, casefold, punctuation → space | `name_full` agreement 27.6% | base |
| N1 | Legal forms anywhere (`llc hines…`), dotted initialisms (`L.L.C.`→`llc`), `aka/dba` split, `(ID: n)` removal, stopwords, dedupe → `core`/`sorted`/`concat` keys; domain names (`fullermeese.com`) → stem | core 52.2%, sorted 54.5%, concat 55.7%; on domain-name targets concat 70% vs 1.9% otherwise | keep |
| N2 | `anyascii` transliteration + vowel-free consonant fold (`vijay`≈`bijy`, `consultants`≈`kmsltemts`) | native-script targets: fold 8.7% vs 0.4% exact | keep (feeds N4) |
| N3 | Country profiles: region canon (US states, IN states incl. 16 native-script forms, FR regions↔départements), street-type canon, house/postal parsing | region match 94.9%, clash 0.7%; house match 67% / clash 10.5% / unknown 22%; postal usable in 0.04% of pairs | keep region + house; postal dropped |
| N4 | Tables mined from **train-fold** pairs only: 528 native→Latin tokens (`praivet→private`, `phaumdesn→foundation`, `imjiniyrimg→engineering`) | native-script targets: core agreement **0.4% → 78.3%**, name Jaccard 0.07 → 0.89 | keep (largest India gain) |

Country profiles (`src/er/normalize/profiles/`) contain only dictionaries; the parsing logic is shared. Unknown labels fall back to GENERIC. Hand-defined dictionaries only: no geocoding and no lookups.

### Leave-profile-out (France lower bound)

France has no labels, so the value of a profile is measured by re-running US and India with the GENERIC profile. Same v2 channels, dev, RRF fusion; see `configs/blocking/abl_v2_generic.toml`.

| | country profiles | GENERIC only |
|---|---|---|
| union pair recall / full coverage / oracle | 98.87% / 96.2% / 0.9967 | 98.66% / 95.5% / 0.9960 |
| B4 address-channel recall | 87.5% | 85.8% |
| B5 name+address recall | 95.2% | 94.8% |
| channel wall time | ≈72 s (region-partitioned) | 671 s (global search) |

Profiles are worth about 0.2 pp of union recall and 0.7 pp of full coverage, plus roughly a 9× speedup. Without a profile the pipeline degrades gracefully, which is the expected floor for an unseen country. The France profile covers the only three regions in the test data: Hauts-de-France, Nouvelle-Aquitaine, and Pays de la Loire, plus their départements (Nord, Pas-de-Calais, Gironde, Loire-Atlantique). It also handles French street abbreviations (`R.`, `Av`, `Bd`, `Imp`), `ST.-`→`saint`, and legal forms (SARL, SAS, S.A.S, SCI, Cie, Ets). This was checked against unlabelled test inputs only.

## 5. Step 3: blocking — channels, fusion, measured ceiling

Channels run independently per target source (S2, S3) and are unioned. Standalone numbers are dev, top-50 per source:

| ID | Channel | Pair recall | Oracle F0.5 | Unique links* | Decision |
|---|---|---|---|---|---|
| B1 | exact `name_concat` (catches domains) | 58.9% | 0.790 | 15 | keep (cheap) |
| B1 | exact `name_fold` | 56.5% | 0.772 | 39 | keep (cheap) |
| B1 | exact `name_sorted` | 57.8% | 0.786 | 1 | **dropped** |
| B1 | house number + 2 rarest address tokens | 63.8% | 0.770 | 53 | keep |
| B2 | name word TF-IDF | 67.6% | 0.822 | 75 | keep |
| B3 | name char-3 TF-IDF | 61.7% | 0.756 | 7 | **dropped** |
| B4 | address char-3 TF-IDF, region-partitioned | 87.5% | 0.947 | 186 | keep (name-replaced records) |
| B5 | name+address char-3 TF-IDF, region-partitioned | 95.2% | 0.981 | 319 | keep |
| B7 | dense e5-small on raw `name \| address`, GPU | **97.5%** | **0.992** | **969** | keep |

\*Unique links are true links only this channel found, measured in the v3 union.

Other findings:

- **Ranker pre-cut (v3_pc20) is kept.** Only pairs ranked ≤20 by at least one channel get ranker features. On full validation ranker@10 rises 0.99603 → **0.99658** (recall 98.69% → 98.87%) and ranker time halves: the long tail was noise for the ranker.
- **Partitioning B7 dense is rejected** (−0.43 pp B7 recall); partitioning B2 raised B2's own recall 67.6% → 79.4% but lowered ranker@10 on dev, so B2 stays global.
- **Speed:** targets are built once per process (pre-transposed region blocks, exact/house tables, GPU-resident embeddings); candidates are identical to before. Full validation 31 min → 16 min; all parallel work capped at 11 threads (`ER_THREADS`).
- **Region partitioning** is the country-aware speed lever. Each query searches only targets in its canonical region plus targets with unknown region. B5 went from 402 s to 27 s on dev (15×) with identical recall (95.18% vs 95.16%) and higher full coverage.
- **Global n-gram pruning is rejected.** `max_df=0.002` cut B5 recall from 95% to 76%.
- **RRF fusion is rejected.** At K=50 per source it scored below the best single channel (0.969 vs 0.981).
- **The learned ranker is kept.** It is trained on a separate 2% train-fold sample (`rtrain`, 35k S1). Features are per-channel rank and score plus 8 rapidfuzz similarities and house/region/legal agree-clash-unknown flags.

### Ceiling vs budget (full validation, 442k S1)

| K per source | cands/S1 | v3 oracle F0.5 | **v3_pc20** pair recall | **v3_pc20** full coverage | **v3_pc20** oracle F0.5 |
|---|---|---|---|---|---|
| 5 | 10 | 0.9943 | 98.30% | 94.2% | 0.9953 |
| 10 | 20 | 0.9960 | 98.87% | 96.1% | **0.9966** |
| 20 | 40 | 0.9967 | 99.06% | 96.8% | 0.9971 |
| 50 | 100 | 0.9971 | 99.20% | 97.2% | 0.9975 |
| union (no cap) | 388 | 0.9984 | 99.47% | 98.2% | 0.9984 |

Chosen shortlist: **v3_pc20, K=10 per source (20 candidates per S1)**: 98.87% pair recall, 96.1% full coverage, 0.9966 oracle F0.5 on full validation. It is above the 95% recall target, and Step 4 then scores 35M test pairs instead of 670M. The remaining India gap (95.9% vs US 99.0% recall at K=50) comes mostly from native-script and name-replaced targets.

### Deep dives, round 2 (dev unless noted)

| # | Item | Finding | Decision |
|---|---|---|---|
| 1 | Memory across chunks | Flat ~35 GB commit per chunk (one-time caches; B7 GPU memory is mirrored in Windows commit). No leak. | chunk 25k, float32 features |
| 2 | Miss taxonomy (`--save-misses`) | 1.10% of links lost at K=10: 53% empty-address targets with generic typo'd names, 19% name replaced, 10% native script; 54% were retrieved but ranked out | fix ranking first |
| 2b | B8 name-only search over empty-address targets | +9 of 496 unretrieved links; ranker@10 −0.0003 | **rejected** |
| 3 | N4 address tokens | only ~3% of India addresses have native script outside the state (already handled) | deprioritized |
| 4a | Ranker trained on 10% of train fold (5×), 20% negatives | ranker@10 0.99691 → 0.99760 | kept |
| 4b | + LightGBM lambdarank per (S1, source) | ranker@10 **0.99773**, recall 99.20%, full coverage 97.2% | kept (`b_v6b_lgbrank.toml`); scores are not probabilities, retrain Step 4 on them |
| 6 | Noise-operation catalogue (`reports/noise/`) | inserted words 48%/29% (India/US), legal changed 50%/28%, typos 12%/6%, name replaced 14%/8%, honorifics, homoglyphs (`c0m`, `5ervices`), alias phrases, `PO Box`/`null` | → N5 |
| 6b | **N5 `name_min` view** (additive): generator noise stripped; fuzzy legal variants guarded by the clean-S1 vocabulary (`prabhat` kept, `piraivet` removed) | exact name-key agreement on val true pairs **61.3% → 67.7%**; native-script 78.3% → 90.2% | kept; blocking test v7a/v7b pending |
| 5 | Dense fine-tune (`er finetune`) | MNRL on 600k train-fold pairs, excluding val and the ranker's `rtrain10` sample | running |

## 6. Test-set path

`uv run er infer configs/blocking/b_v3_pc20.toml --k 10` trains the ranker on train and blocks all 1.73M test S1 against the test S2/S3 pools, France included. It writes `output/candidate_pairs.tsv` and a placeholder all-empty `output/matching_results.tsv`, which `student_resource/utils/validate_submission.py` accepts. TEST_RESULTS

## 7. Step 4: features and model (validated end to end on real dev data)

`uv run er match configs/blocking/b_v3.toml --subset dev` shortlists candidates with the Step 3 ranker (`--k` per source), builds match features, fits LightGBM over 5 grouped (by `s1_id`) folds via `er.split.load_split()`'s `cv_fold` column, calibrates if needed, and writes `reports/match/{metrics.json,calibration.png,gate_report.md}`.

- `src/er/match/features.py`: similarity-only pair features (never raw country labels or token identities) — name (ratio/token-set/token-sort/partial/Jaro-Winkler/char-3gram cosine/IDF-weighted token overlap/core-exact/alias-DBA-domain-translit flag/script flag/legal same-compatible-conflicting-missing), address (street content ratio+token-set+char-3gram, region/postal/house tri-state, house abs/relative numeric diff, number-set overlap, emptiness flags), context (rank by name/address within the S1, margin to the best competing S1 for the same target, count of S1s claiming a record, opposite-source sibling support), and pair metadata (S2/S3 flag, candidate counts, blocking-channel provenance via `ranker.provenance`).
- `src/er/match/idf.py`: exact (unhashed) word-level IDF over the target pool, used for the token-overlap features.
- `src/er/match/train.py`: `cv_train` (LightGBM, numpy arrays + categorical-feature indices, no pandas dependency), `maybe_calibrate` (isotonic, applied only if the raw reliability-curve gap exceeds 0.05), `evaluate` (AUC/PR-AUC/macro F0.5 at threshold 0.5, overall and per country), `country_holdout_check` (leave-one-country-out — the France stand-in, same pattern as the Step 2 leave-profile-out check).
- `src/er/match/report.py`: renders the gate report (metrics table, calibration plot, feature importance, holdout-country gap).
- `normalize/address.py` now also extracts `addr_city`, `addr_unit`, and `addr_landmark` (city from the segment preceding the region, or the "POSTAL CITY" convention for France; unit from apt/ste/suite/unit/fl/floor markers; landmark from tokens following near/opp/behind/opposite). This is additive only — `addr_content`/`addr_house`/`addr_region`/`addr_numbers`/`addr_postal` are computed identically to before, so the already-measured Step 2/3 numbers in this document stay reproducible; only `addr_norm`'s downstream consumers gain three new columns. Match features use them for `sim_city_ratio`, `locality_state`, `unit_state`, and landmark token overlap.
- Run on the real dev subset (44,535 S1, `configs/blocking/b_v2.toml`, no dense channel — CUDA unavailable on this machine): OOF AUC 0.9989, PR-AUC 0.9953, macro F0.5 at threshold 0.5 = 0.9633 (US 0.9691, India 0.9545). `write_oof()` persists the OOF frame to `reports/match/oof.parquet` for the Step 5 decision layer to consume.

## 8. Step 5: decision layer (validated end to end on real dev data)

`uv run er decide [--oof reports/match/oof.parquet]` reads the OOF predictions `er match` writes and produces the final (s1_id, target_id) predictions plus a gate report. Order: exclusivity first (a global pass dropping losing claims on contested S2/S3 records), then each S1 picks its own candidate-set size over its now-deduplicated pool — doing it the other way (pick a set per S1, then arbitrate) can strand a losing S1 short of its choice with no chance to fall back to its next-best candidate.

- `src/er/decide/exclusivity.py`: primary rule keeps the highest-probability claimant per contested record. The "proper assignment method" comparison is scarcity-weighted (`probability / that S1's own candidate count`), so an S1 with few alternatives can outbid a stronger claimant with plenty of other options — genuinely different from raw-probability argmax (tested in `tests/test_decide.py`), not a re-branded copy of it.
- `src/er/decide/expected_f05.py`: for each S1, the optimal predicted set of a given size k is always its top-k probabilities (a classical F-measure result), so the only choice is k itself. Uses the standard plug-in estimator — E[TP|top-k] = sum of the k selected probabilities and E[T] = sum of all candidate probabilities, both exact by linearity of expectation, plugged into the F0.5 formula. This is not literally E[F0.5] (a ratio of correlated sums has no closed form without a full Poisson-binomial DP), but it's the standard practical estimator used throughout IR/ranking systems and is exact when probabilities are well separated near 0/1, which calibration makes more likely. k=0 (predict nothing) is handled exactly rather than by the same degenerate formula: P(no true link exists) = ∏(1-p_i).
- `src/er/decide/threshold.py`: the global-threshold baseline, grid-tuned on OOF only, per spec.
- `src/er/decide/evaluate.py`: precision/recall/F0.5 curve per country, singleton accuracy (correct empty predictions / true singletons).
- `src/er/decide/pipeline.py` + `report.py`: staged ablation (threshold-0.5-no-exclusivity → +exclusivity → +tuned global threshold vs. +per-entity expected-F0.5 → final), each compared on OOF and the winner picked automatically; writes `reports/decide/{metrics.json,gate_report.md}`.
- Run on the real dev subset (same OOF as Step 4 §7): baseline (threshold 0.5, no exclusivity) macro F0.5 0.9633 → +exclusivity 0.9633 (only 90,190 of 687,479 claimed records were actually contested, so it barely moves the aggregate) → +OOF-tuned global threshold (0.70) 0.9666 → +per-entity expected-F0.5 0.9664. The tuned global threshold won on OOF, so the pipeline used it as final rather than forcing the fancier per-entity rule — the guard the "compare and keep whichever wins" spec exists for. Final: macro F0.5 0.9666, singleton accuracy 94.4%.
- `tests/test_decide.py` and `tests/test_phase4_5_integration.py` (a real Step 4→5 integration test through `write_oof`) verify the logic; full test suite passes and ruff is clean.


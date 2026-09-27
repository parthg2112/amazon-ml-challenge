# Frontier R&D plan: remaining loss, unseen-country gap, test-only failure modes

Answer to `docs/rnd_frontier_prompt.md` (research agent, 2026-09-26). The verification section records what we re-measured
before building anything; the rest summarizes the research plan as received.

## Verification (re-measured on v6 OOF 442k/618k val and `test_scored_v6_p3`)

| Claim | Measurement | Verdict |
|---|---|---|
| `ctx_parts` returns 1 for "full" bundles | Already derives parts from `train_universe_s1`; v6 was scored with 3 parts | Stale, done |
| M1 contests denser on test | Contested winners per 1k S1: test France 48.0, India 18.7, US 10.1 vs val 2.1–3.4. Val contested winners 59% precise at mean p 0.86 | Confirmed |
| EXR odds renormalization | Raw v6: thr +0.00012, tilted +0.00010. On top of CE band (618k OOF): tilted 0.98349 → **0.98368** | Adopted (`exclusivity.renormalize`) |
| M5 sibling house consensus | Band 0.3–0.69: 83.2% true vs 47.6% base; 0.69–0.9: 96.0% vs 81.9% | Adopted as band-combiner features: +0.00018 val, +0.00012 India proxy (logistic). XGB combiner overfits (India −0.0003) |
| M7 count-model headroom ≤ 0.0003 | Count oracle (true T, top-T by p) = 0.98629 vs tilted 0.98211 | Holds for the T≤1 ambiguity zone only; the total headroom is larger |
| KPM premise (France Σp 3.33 under-confident vs invariant 3.46) | Σp confirmed. The record-count estimate (5.53 × 0.60 ≈ 3.31) allows the alternative that France's true rate is lower | Gate on the India proxy, then a paired LB check |
| Hungarian / optimal reassignment (follow-up) | Oracle reassignment ceiling at threshold on current scores: +0.0001 | Rejected; joint expected-F assignment (below-threshold) stays [later] |
| (country × address-availability) tilt (follow-up) | −0.0006 in-distribution | Rejected |
| CE feature transform (follow-up) | Logit: val 0.98349 / India 0.97979; tanh(ce/2): 0.98337 / 0.98024 | tanh adopted (transfer-robust) |

## Measured findings (research agent)

- **M1** Contested targets (winner p ≥ 0.69, runner-up ≥ 0.3) are 3–20× denser on test than val; the at-most-one-owner posterior
  odds_s/(1+Σodds) is almost exactly calibrated on val (0.51–0.62 vs 56–62% precision).
- **M2** Aggregate calibration holds on US/India test (Σp 3.47 vs prior 3.46; EM prior 0.1726 → 0.1735): no label shift.
- **M3** France Σp/S1 3.334; France band pairs sit on records short of confident links (2.83 vs 3.18 US).
- **M4** US test band excess is small (0.076 → 0.094 pairs/S1 in 0.69–0.9); decoy vs exclusivity biases roughly cancel.
- **M5** Sibling house consensus (above).
- **M6** Twin slice (identical name key, one side address-less; 1.11M pairs): positive rate by pool copies of the name
  98/97/87/31/6.9/1.3% (1, 2, 3–5, 6–20, 21–100, >100); raw-string/case equality flat at fixed p; legal-token equality
  at 0.3–0.69: 51% vs 35%.
- **M7** Records with max p in [0.1, 0.69): 1.0% of records, 25% linked, AUC 0.72 by max p.
- **M8** France city parsed in 0.2% (harmless: city tokens stay in address content); names carry address tokens
  (France 12.3%, India 8.6%, US 3.8%).
- **M9** The India-held-out proxy leaks India (ranker score and fine-tuned e5 saw India; dropping the ranker group costs
  −0.013), so it is optimistic for France.

## Ranked ideas (research agent's estimates; our status in the last column)

| Rank | Idea | Mode | E[LB gain] | P | Hours | Status |
|---|---|---|---|---|---|---|
| 1 | F0 CE band rescoring | decision | +0.0010 | 0.7 | 1.5 | Done: val +0.0013 (tanh), paired LB check pending |
| 2 | EXR renormalization | decision | +0.0005 | 0.5 | 1 | Done: val +0.0002 on top of CE band |
| 3 | KPM per-country Σp matching (+ per-country tilt) | decision + proxy | +0.0006 | 0.35 | 1.5 | Done: India proxy +0.0027 alone / +0.0004 on band+EXR; in v7 test (France 3.16 → 3.27 links/S1); **LB rejected for France** (E4 vs D4 −0.0001) |
| 4 | SC sibling-consensus band rescorer | decision | +0.0004 | 0.4 | 2 | Measured +0.00018 val / +0.00012 proxy |
| 5 | GM group masking + list-free legal equality | refit | +0.0003 | 0.35 | 2.5 | Masking implemented (`--mask-groups`), gate via `er proxy` arms, run pending. List-free legal **rejected** on real data: N1 lists already cover France (61.6% of FR targets vs ~60% legal-form share), and pool-frequent edge tokens discriminate far worse on same-name val pairs ('differ' 28%/43% true vs legal_relation 5%/0.01%, US/India). Macro-aware weights rejected (v6w 0.98193 vs v6 0.98224) |
| 6 | DA-CE domain-adversarial CE | CE train | +0.0005 | 0.3 | 3 | Later |
| 7 | F2 class-conditional feature alignment | refit | +0.0010 | 0.2 | 4 | Later |
| — | SIM test-faithful val simulator | tool | gatekeeper | — | 1.5 | Optional |
| 8 | FU full-universe competition context | re-block + refit | +0.0008 | 0.35 | 6 | Done: v7 (`fit2 --train-scope full`), OOF 0.98287 |
| 9 | AFJ S1–S1 reference-table negatives | feature | +0.0006 | 0.3 | 5 | Later |

## Bayes ceilings

- **S-B address-less twins:** posterior ≈ 1/m; the >20-copy buckets need a likelihood ratio ≥ 30 no field provides. Ceiling
  ≈ +0.0005 val. Would rise if, in name groups fully inside the universe, the true owner is the maximum-deficit S1 in ≥ 70% of
  groups.
- **S-A France:** decision-only fixes (KPM, EXR, tilt) capped at ≈ +0.006 France F (≈ +0.0009 LB); the rest is ranking error
  needing French-aware evidence (CE band, DA-CE, alignment, AFJ).
- **S-C shortlist misses:** ≤ 0.164% of links recoverable by raising k; ceiling ≤ +0.00025.
- **S-D T=0/T=1:** ambiguity zone 1% of records at AUC 0.72; ceiling ≈ +0.0003.
- **S-E test-only contests:** invisible on val; EXR targets it, FU fixes it at the source.

## Abandoned (with the number that kills each)

Raw-form/casing hunt (flat at fixed p) · decoy-clique FP feature (cliques 99.6% positive) · name-key sibling consensus
(flat) · EM/BBSE label shift for US/India (π 0.1726 → 0.1735) · naive EM for France (conditional shift) · confidence
self-training (+0.0008) · prior-matched threshold (−0.007) · raising k (≤ +0.00025) · singleton count model (AUC 0.72) ·
decoy-driven threshold shift (break-even).

## Sources

DADER (SIGMOD 2022) · ZeroER · Sudowoodo · AnyMatch · Splink TF adjustments · Auto-FuzzyJoin · Waegeman et al. (GFM, JMLR
2014) · Sadinle (bipartite record linkage, 2017) · Alexandari et al. (EM + BCTS) · Zhang et al. (target/conditional shift,
ICML 2013) · Park et al. (calibration under covariate shift) · CBST · GPL · Ristad–Yianilos · Bilenko & Mooney · GeCo.

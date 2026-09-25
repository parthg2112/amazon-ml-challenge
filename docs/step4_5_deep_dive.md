# Deep dive: the five unexplored areas

Follows `docs/step4_5_review.md` §3 and the post-v2 status table. Baseline is v2 on val scope:
OOF macro F0.5 **0.97984** (nested 0.97981), leaderboard 0.972, India 0.97693 / US 0.98175,
leave-India-out 0.9441 (gap 0.033), leave-US-out 0.9683.

Everything below is implemented and unit-tested. What is *measured* versus *estimated* is marked per
area — the machine this was written on has no CUDA and none of the cached match artifacts, so the
GPU-bound numbers are still open.

| # | Area | Review's estimate | After this dive | Status |
|---|---|---|---|---|
| 1 | Cross-encoder pair model | +0.002–0.005 | unchanged; largest remaining | implemented, needs GPU run |
| 2 | Features that transfer across countries | France +0.005–0.01 → +0.001–0.0015 | unchanged; best evidence behind it | implemented, needs GPU run |
| 3 | Optuna / monotone / CatBoost | +0.0005–0.0015 | unchanged | implemented, needs GPU run |
| 4 | Link-count-aware decisions | +0.001? | **split: ~0 in-distribution, ~+0.003 on an unseen country** | implemented + measured below |
| 5 | Full 2.2M universe | +0.0005–0.001 | unchanged, but it also *removes* an approximation | already supported; needs a run |

---

## Area 4 — link-count-aware decisions (measured here)

### The prior is real, sharp, and country-invariant

Fitted from `train_truth.parquet` and cached to `reports/decide/link_prior.json`:

| | value |
|---|---|
| mean links per S1 | **3.4613** |
| per source | S2 1.6737, S3 1.7875 |
| hard caps (max ever observed) | **S2 ≤ 5, S3 ≤ 6** |
| singleton rate P(T=0) | 5.585% |
| India vs US mean | 3.4645 vs 3.4591 |
| India vs US singleton rate | 5.588% vs 5.583% |

The last two rows are the important ones. The count distribution is *the same in both countries we can
measure*, which is what makes it usable as a prior for France — unlike the score distribution, which
demonstrably shifts.

### Why the exact expected-F0.5 rule lost (0.97898 vs 0.97984)

The review's explanation is that the Poisson-binomial assumes a record's candidates are independent
when they are not. That is measurable rather than rhetorical:

| | variance of the link count |
|---|---|
| truth | **2.908** |
| independent model, top-p 0.99 | 0.280 |
| independent model, top-p 0.95 | 0.380 |
| independent model, top-p 0.90 | 0.452 |

A confident, well-calibrated independent model implies a count distribution **6–10× too concentrated**.
It thinks it knows T to about ±0.5 when the truth varies by ±1.7, so it commits to a k far too
confidently. Marginal calibration (isotonic) cannot fix this: it fixes E[y|p] per candidate and says
nothing about the joint.

`linkcount.select_tilted` reweights the count posterior onto the empirical prior
(`exact_f05.expected_f05_matrix` gained an optional `tilt`; `tilt=None` reproduces the validated
numbers exactly, guarded by a test).

### But the payoff depends on one measurable quantity

Simulation at the real operating point (40k records, 20 candidates each, counts drawn from the real
prior, isotonic-calibrated scores; `sep` controls separability, `shared` the within-record correlation):

| within-record correlation | implied var | true var | tuned threshold | exact | **tilted** | tilted − threshold |
|---|---|---|---|---|---|---|
| none | 2.88 | 2.91 | 0.95557 | 0.95378 | 0.95479 | **−0.0008** |
| realistic | 5.05 | 2.91 | 0.95612 | 0.95405 | 0.96109 | **+0.0050** |
| severe | 13.28 | 2.91 | 0.72113 | 0.72633 | 0.82540 | +0.1043 |

The simulation reproduces the real ordering when candidates are independent (threshold 0.95557 beats
exact 0.95378, mirroring 0.97984 vs 0.97898). The tilt only wins once the model's implied count
distribution is actually wrong.

**So do not spend GPU time on this blind.** `er linkcount --name v2` reads the saved OOF and prints
`var_ratio` (implied variance ÷ prior variance) in seconds. Above ~1.25 the tilted rule is worth
running; at ~1.0 it is not, and the honest expected gain in-distribution is zero.

### What is worth taking regardless

* **`prior_matched_threshold`** — set the cutoff so predicted links per S1 equals 3.4613. Needs no
  labels, so France can use it. This is already measured in `reports/match/v2_transfer_oracle.json`:
  India-out 0.9441 → **0.9469** (the oracle threshold reaches only 0.9478, so prior-matching recovers
  ~75% of what an oracle would), US-out 0.96826 → 0.96757 (costs 0.0007 where transfer already worked).
  Net on a country that behaves like the bad case, positive; on one that behaves like the good case,
  nearly free. It is exposed as its own rule and reported separately rather than entered into the
  in-distribution winner comparison, which it cannot win by construction.
* **Hard caps** (`apply_caps`) — S2 > 5 or S3 > 6 links is something the truth has never once shown.
  This is a guard against a pathological record, not a source of gain: it only binds when the model
  over-predicts for one source, so expect it to score identically to the plain threshold on healthy
  data and to matter only in the tail. (It is not validated by the simulation above, which is
  single-source and therefore violates the per-source cap structure by construction.)
* **Under-linking is now visible**: `submit2` reports `links_per_s1` next to `prior_links_per_s1`. The
  review's France figure of 3.08 against 3.46 is a 11% shortfall; this makes it a standing check
  rather than a one-off observation.

**Rejected:** `select_by_expected_count` (top-k from round(Σp), the review's "top-k by estimated
count"). It scored 0.793 / 0.875 / 0.908 against 0.865 / 0.955 / 0.988 for a tuned threshold across
the three separability settings — clearly worse, because a hard k cannot express "predict nothing"
and cannot straddle the 3–4 peak. It stays in the comparison only so the rejection is on record.

---

## Area 2 — features that transfer (implemented; the strongest case of the five)

### The evidence that this is the real France problem

| observation | value | source |
|---|---|---|
| `p` importance vs the next feature | 28,142,670 vs 864,817 (**33×**) | `reports/match/v2.json` |
| leave-India-out AUC | 0.9985 → 0.9970 (−0.0015) | `v2_transfer.json` |
| leave-India-out macro F0.5 | 0.97693 → 0.9441 (**−0.033**) | `v2_transfer.json` |
| what an oracle threshold recovers | +0.0037 | `v2_transfer_oracle.json` |

Discrimination barely moves; the decision collapses. And only about an eighth of the collapse is the
operating point. The remaining ~0.029 is the feature distribution shifting — with `p`, a score fitted
on US/India only, carrying 33× the weight of anything else.

### What is implemented

`features.add_rank_relative` builds two views per base feature, strictly **within one S1's own shortlist**:

* `{f}_rel` — the feature minus that record's best candidate (0 for the best, negative below).
* `{f}_rk` — its rank among that record's candidates.

A country-wide shift in French tokenization, address shape or legal forms moves every candidate of a
record together, so it cancels in both views. There is also a structural argument: these are computed
inside one record, so unlike `margin_to_best_competitor` they cannot depend on how many other S1
exist — they are immune by construction to the train/test density mismatch that M1 and M7 had to fix.

`er transfer <config>` runs the ablation crossed with both hold-out directions:

| arm | ranker score | relative views |
|---|---|---|
| `base` | `p`, `frank` | — |
| `no_p` | dropped | — |
| `rel` | `p`, `frank` | added |
| `rel_no_p` | dropped (but `p_rel`, `p_rk` kept) | added |

`rel_no_p` is the interesting one: it keeps the ranker's *ordering within a record* while discarding
its absolute value, which is precisely the part that has no reason to transfer.

The number to read is the **gap** (in-distribution minus held-out), not the overall score — dropping
`p` can only cost in-distribution accuracy, and that is an acceptable trade if the gap closes by more.

---

## Area 1 — cross-encoder (implemented; largest remaining, GPU-bound)

Ditto-style (Li et al. 2020): serialize both records as `[COL] name [VAL] … [COL] addr [VAL] …`, let
one transformer attend across the pair, and feed its probability to the GBDT. Every existing string
feature compares through a fixed function; this is the only proposed feature that can read both
records jointly. The fine-tuned bi-encoder already returned +0.0013 in blocking, and a cross-encoder
is the strictly stronger form of the same idea (Peeters & Bizer 2021).

Two correctness rules are enforced in code, because either one gets silently rewarded if broken:

* **Training pairs come from the train fold only**, excluding the ranker's own `rtrain10`. Step 4 is
  cross-validated on val, so a cross-encoder that had seen val pairs would leak directly into the
  feature it produces — and the leak would look like a gain.
* **Negatives are the top-ranked non-matches from the same shortlist**, not random pairs. Random
  negatives are trivially separable and teach nothing about the pairs that decide the score.

`score_pairs(..., band=(0.2, 0.9))` restricts the transformer to the uncertain range and marks
everything else `ce_scored = 0` with a null probability, so the GBDT learns "not scored" as its own
case instead of reading a fabricated value.

Run: `er crossenc configs/blocking/b_v9.toml` then join `ce_prob` / `ce_scored` as features.

---

## Area 3 — tuning, monotone constraints, CatBoost (implemented)

`er tune <config>` searches with Optuna, objective = **OOF macro F0.5 after exclusivity and a tuned
threshold**, not AUC. That choice follows directly from the v2 transfer evidence above: AUC moved
0.0015 while F0.5 moved 0.033, so AUC is not a usable proxy for what this pipeline is scored on.
Trials fit on a configurable fraction of whole S1 records to stay affordable.

Monotone constraints are the part worth more than the hyper-parameters. `tune.direction` assigns +1 to
similarities, overlaps and exact-match flags; −1 to house-number differences, unmatched-token counts
and every rank; and **0 to everything genuinely ambiguous** — all categorical codes, candidate counts,
empty-address flags, noise-catalogue flags — because a wrong constraint is worse than none. Forcing
"more similar cannot mean less likely to match" removes the model's freedom to fit non-monotone quirks
that exist only in US/India data, which is the France failure mode. The report separates
`defaults` / `monotone_only` / `best` so the constraint effect is not hidden inside the search.

CatBoost (Apache-2.0) is wired as a backend (`--backend catboost`), and `xgb_cpu` exists so the whole
comparison can run without CUDA.

---

## Area 5 — full 2.2M universe (already supported)

`Universe(..., scope="full")` and `er fit2 --scope full` already exist; this has simply never been run.
The under-appreciated part is that it is a **correctness** change, not only a +0.0005 accuracy one:
with `scope="full"` the training context sees the same S1-per-target density as test, so `ctx_parts`
returns 1 and the M7 hash-partition correction is not needed at all. M7 is an approximation that
matches densities on average; running at full scope removes the need for it.

---

## Suggested order

1. `er linkcount --name v2` — seconds, decides whether area 4's tilt is worth anything.
2. `er transfer configs/blocking/b_v9.toml` — best evidence, no new model family.
3. `er tune configs/blocking/b_v9.toml --trials 30` — cheap, and monotone constraints also help area 2.
4. `er crossenc …` — largest expected gain, highest cost.
5. `er fit2 --scope full` — folds in area 5 and drops the M7 approximation.

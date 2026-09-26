# Business Entity Resolution Challenge

Python pipeline for the full challenge: exact macro-F0.5 scorer, country-aware normalization, multi-channel blocking with a measured recall ceiling (Steps 1–3), a 122-feature XGBoost scorer with an uncertain-band cross-encoder rescorer (Step 4), and an expected-F0.5 decision layer (Step 5). See `PLAN.md` and `docs/` for the approach matrix and results.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 and the CUDA 12.8 build of torch are pinned in `pyproject.toml`.

```powershell
uv sync --extra dense        # drop --extra dense to skip torch / sentence-transformers
uv export --no-hashes --extra dense > requirements.txt   # submission package
```

Raw data stays in `student_resource/dataset/{train,test}/` (git-ignored).

## Workflow

```powershell
uv run er prepare                     # TSV → artifacts/data/*.parquet (seconds)
uv run er split                       # SHA-256 entity split; checks parity with the retired Perl scorer
uv run er normalize                   # 12-process normalization → artifacts/norm/<variant>/
uv run er learn                       # N4 tables mined from train-fold pairs → configs/learned/
uv run er normalize --learned configs/learned/n4_v1.json
uv run er diagnose --subset val --learned configs/learned/n4_v1.json   # key agreement on held-out pairs
uv run er block configs/blocking/<config>.toml [--subset dev|val]      # channels → fusion → metrics
uv run er leaderboard                 # rebuild reports/leaderboard.md
uv run pytest -q
```

### Reproducing `output/` end to end (Steps 1–5)

```powershell
uv run er prepare; uv run er split
uv run er learn; uv run er normalize --learned configs/learned/n4_v1.json   # norm variant t1-auto-n4_v1
uv run er finetune --name e5s_ft1                     # dense retriever: multilingual-e5-small (MIT), train-fold pairs only
uv run er fit2 configs/blocking/b_v9.toml --name v7 --train-scope full --drop-groups legal   # see memory note below
uv run er ce-train configs/blocking/b_v9.toml --name ce_usin_base --init intfloat/multilingual-e5-base --lr 2e-5
uv run er ce-band --name v7 --ce-model ce_usin_base   # combiner on OOF (CE-seen S1 excluded) + policy, stored in the bundle
uv run er submit2 --name v7                           # test scoring; competitor context over all test S1 (1 partition)
uv run er decide-test --name v7 --parts 1             # band rescore -> EXR -> tilted expected-F0.5 -> output/*.tsv
python student_resource/utils/validate_submission.py -m output/matching_results.tsv -c output/candidate_pairs.tsv -t student_resource/dataset/test --check-ids
```

`fit2 --train-scope full` needs shortlists for all 2.2M train S1 (competition context) and features for val + 450k
clean extras. On 32 GB RAM build them in separate processes first (resumable caches), then run `fit2`:
`Universe(cfg, "train", 10, "full")` → `_context().collect()` per missing chunk (channels) → `.shortlists()` →
`.subset_features(ids, "fu", gctx, collect=False)`. One RTX-class GPU (16 GB); channels+shortlists ≈ 50 min,
features ≈ 25 min, `fit2` ≈ 16 min, `submit2` ≈ 26 min.

Subsets: `dev` is 10% of the validation holdout (44.5k S1), `val` is the full 20% holdout (442k S1), and `rtrain` is a 2% train-fold sample used to fit the stage-2 candidate ranker. Queries always search the full training S2/S3 pool, as at test time.

## Layout

| Path | Role |
|---|---|
| `src/er/metrics.py` | leaderboard F0.5 and blocking metrics (recall, full coverage, oracle ceiling, reduction ratio) |
| `src/er/normalize/` | N0–N4 normalization views and country profiles (`profiles/{us,india,france}.py`) |
| `src/er/blocking/channels/` | exact-key joins, sparse TF-IDF top-k (`sparse_dot_topn`), GPU dense retrieval |
| `src/er/blocking/{fuse,ranker,runner}.py` | RRF fusion, learned candidate ranker, experiment runner |
| `configs/blocking/*.toml` | experiment definitions |
| `reports/` | tracked JSON metrics and `leaderboard.md` |
| `artifacts/` | parquet, hashed-matrix, and embedding caches (ignored) |

`student_resource/utils/validate_submission.py` checks final `output/` files before submission.

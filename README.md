# Business Entity Resolution Challenge

Python pipeline for Steps 1–3: exact macro-F0.5 scorer, country-aware normalization, and multi-channel blocking with a measured recall ceiling. See `PLAN.md` for the approach matrix and results.

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

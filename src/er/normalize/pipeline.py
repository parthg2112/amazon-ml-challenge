"""Record normalization + sharded multi-process runner with parquet cache per variant."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import polars as pl

from er import io
from er.normalize.address import normalize_address
from er.normalize.names import normalize_name
from er.normalize.profiles import profile_for
from er.normalize.text import is_native_script

SHARD_ROWS = 250_000


@dataclass(frozen=True)
class NormConfig:
    translit: bool = True
    profile_mode: str = "auto"  # auto | generic (leave-profile-out ablation)
    learned: str | None = None  # path to N4 tables (json)

    @property
    def variant(self) -> str:
        v = f"t{int(self.translit)}-{self.profile_mode}"
        if self.learned:
            v += "-" + Path(self.learned).stem
        return v


@lru_cache(maxsize=4)
def _learned(path: str | None) -> tuple[dict[str, str], dict[str, str]]:
    if not path:
        return {}, {}
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d.get("name_tokens", {}), d.get("regions", {})


def normalize_record(name: str | None, address: str | None, country: str | None, cfg: NormConfig) -> dict:
    prof = profile_for(country, cfg.profile_mode)
    token_map, region_map = _learned(cfg.learned)
    rec = {"profile": prof.name, "name_native": is_native_script(name)}
    rec.update(normalize_name(name, prof, cfg.translit, token_map or None))
    rec.update(normalize_address(address, prof, cfg.translit, region_map or None))
    return rec


SCHEMA = {
    "profile": pl.Utf8, "name_native": pl.Boolean, "name_full": pl.Utf8, "name_core": pl.Utf8,
    "name_legal": pl.Utf8, "name_sorted": pl.Utf8, "name_concat": pl.Utf8, "name_fold": pl.Utf8,
    "name_alts": pl.List(pl.Utf8), "name_domain": pl.Boolean, "addr_norm": pl.Utf8,
    "addr_content": pl.Utf8, "addr_region": pl.Utf8, "addr_house": pl.Utf8,
    "addr_numbers": pl.List(pl.Utf8), "addr_postal": pl.Utf8, "addr_empty": pl.Boolean,
}


def _shard(args: tuple[str, int, int, dict]) -> pl.DataFrame:
    path, offset, length, cfg_d = args
    cfg = NormConfig(**cfg_d)
    df = pl.scan_parquet(path).slice(offset, length).collect()
    cols = {k: [] for k in SCHEMA}
    for name, addr, country in zip(df["business_name"], df["business_address"], df["country"]):
        rec = normalize_record(name, addr, country, cfg)
        for k in SCHEMA:
            cols[k].append(rec[k])
    out = pl.DataFrame(cols, schema=SCHEMA)
    return pl.concat([df.select("entity_id", "src", "country"), out], how="horizontal")


def norm_path(split: str, src: str, cfg: NormConfig) -> Path:
    return io.ARTIFACTS / "norm" / cfg.variant / f"{split}_{src}.parquet"


def normalize_source(split: str, src: str, cfg: NormConfig, workers: int | None = None,
                     force: bool = False) -> Path:
    out = norm_path(split, src, cfg)
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    path = str(io.parquet_path(split, src))
    n = pl.scan_parquet(path).select(pl.len()).collect().item()
    tasks = [(path, off, SHARD_ROWS, asdict(cfg)) for off in range(0, n, SHARD_ROWS)]
    workers = workers or os.cpu_count() or 12
    t0 = time.perf_counter()
    os.environ["POLARS_MAX_THREADS"] = "1"  # children only; parent pool already initialized
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_shard, tasks))
    pl.concat(parts).write_parquet(out)
    print(f"normalized {split}_{src} [{cfg.variant}] {n:,} rows in {time.perf_counter() - t0:.1f}s")
    return out


def load_norm(split: str, src: str, cfg: NormConfig) -> pl.DataFrame:
    return pl.read_parquet(normalize_source(split, src, cfg))

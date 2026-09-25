"""Raw TSV → Parquet conversion and cached loaders."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", str(os.cpu_count() or 12))

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "student_resource" / "dataset"
ARTIFACTS = ROOT / "artifacts"
DATA = ARTIFACTS / "data"
REPORTS = ROOT / "reports"

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
SOURCES = ("S1", "S2", "S3")
TARGET_SOURCES = ("S2", "S3")


def _read_tsv(path: Path) -> pl.DataFrame:
    # quote_char=None: fields contain bare quotes; tabs never appear inside values.
    return pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        infer_schema=False,
        missing_utf8_is_empty_string=True,
        encoding="utf8",
    )


def source_path(split: str, src: str) -> Path:
    return DATASET / split / f"{split}_source{src[-1]}.tsv"


def parquet_path(split: str, src: str) -> Path:
    return DATA / f"{split}_{src}.parquet"


def prepare(splits: tuple[str, ...] = ("train", "test"), force: bool = False) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for split in splits:
        for src in SOURCES:
            out = parquet_path(split, src)
            if out.exists() and not force:
                continue
            df = _read_tsv(source_path(split, src))
            missing = set(SOURCE_COLUMNS) - set(df.columns)
            if missing:
                raise ValueError(f"{split} {src}: missing columns {missing}")
            df = df.select(SOURCE_COLUMNS).with_columns(pl.lit(src).alias("src"))
            bad = df.filter(~pl.col("entity_id").str.starts_with(f"{src}-")).height
            if bad:
                raise ValueError(f"{split} {src}: {bad} ids without {src}- prefix")
            df.write_parquet(out)
            print(f"{out.name}: {df.height:,} rows")
        if split == "train":
            out = DATA / "train_truth.parquet"
            if force or not out.exists():
                gt = _read_tsv(DATASET / "train" / "train_ground_truth.tsv")
                gt = gt.select(
                    pl.col("source1_entity_id").alias("s1_id"),
                    pl.col("matched_entity_ids")
                    .str.split(",")
                    .list.eval(pl.element().str.strip_chars())
                    .list.eval(pl.element().filter(pl.element() != ""))
                    .alias("truth"),
                )
                gt.write_parquet(out)
                print(f"{out.name}: {gt.height:,} rows")


def load_source(split: str, src: str) -> pl.DataFrame:
    return pl.read_parquet(parquet_path(split, src))


def load_truth() -> pl.DataFrame:
    return pl.read_parquet(DATA / "train_truth.parquet")


def truth_long(truth: pl.DataFrame) -> pl.DataFrame:
    """One row per true (s1_id, target_id) link."""
    return (
        truth.explode("truth", empty_as_null=True)
        .drop_nulls("truth")
        .rename({"truth": "target_id"})
        .with_columns(pl.col("target_id").str.slice(0, 2).alias("src"))
    )

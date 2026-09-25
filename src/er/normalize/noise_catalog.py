"""Reverse-engineer the synthetic noise generator: frequency of each corruption on train-fold true pairs."""

from __future__ import annotations

import re
from collections import Counter

import polars as pl
from rapidfuzz.distance import Levenshtein

from er import io
from er.normalize.profiles import profile_for
from er.normalize.text import base_text, is_native_script, tokens
from er.split import load_split, subset_ids

_DOMAIN = re.compile(r"^\S+\.(com|net|org|in|co|biz|info|io|fr)\b", re.I)
_NUM_TAIL = re.compile(r"[-\s]\d{6,}\s*$")
_BRACKET = re.compile(r"[(\[<>{]")
_ACCENT = re.compile(r"[À-ɏ]")


def _toks(s: str | None, country: str) -> list[str]:
    return tokens(base_text(s or "", True, profile_for(country).apostrophe_splits))


def _typo_pairs(extra_t: list[str], extra_q: list[str]) -> int:
    """Unmatched target tokens within edit distance 2 (ratio-bounded) of an unmatched S1 token."""
    n = 0
    for t in extra_t:
        if any(Levenshtein.distance(t, q) <= min(2, max(1, len(q) // 4)) for q in extra_q):
            n += 1
    return n


def _house(toks: list[str]) -> str:
    return next((t for t in toks if t.isdigit()), "")


def pair_ops(qn, qa, tn, ta, country: str, inserted: Counter, legal: frozenset) -> dict:
    qt, tt = _toks(qn, country), _toks(tn, country)
    q_core = [t for t in qt if t not in legal]
    t_core = [t for t in tt if t not in legal]
    extra_t = [t for t in t_core if t not in q_core]
    extra_q = [t for t in q_core if t not in t_core]
    typos = _typo_pairs(extra_t, extra_q)
    for t in extra_t:
        if not any(Levenshtein.distance(t, q) <= 2 for q in extra_q):
            inserted[t] += 1
    qa_t, ta_t = _toks(qa, country), _toks(ta, country)
    qh, th = _house(qa_t), _house(ta_t)
    ta_raw = ta or ""
    return {
        "name_domain": bool(_DOMAIN.match((tn or "").strip())),
        "name_native": is_native_script(tn),
        "name_accent_injected": bool(_ACCENT.search(tn or "")) and not is_native_script(tn),
        "name_case_changed": (tn or "") != (qn or "") and (tn or "").lower() == (qn or "").lower(),
        "name_bracket_noise": bool(_BRACKET.search(tn or "")) and not _BRACKET.search(qn or ""),
        "name_numeric_tail": bool(_NUM_TAIL.search(tn or "")),
        "name_aka": " aka " in f" {(tn or '').lower()} ",
        "name_identical_norm": qt == tt,
        "name_reordered": sorted(qt) == sorted(tt) and qt != tt,
        "legal_changed": {t for t in qt if t in legal} != {t for t in tt if t in legal},
        "name_token_inserted": len(extra_t) > typos,
        "name_token_dropped": len(extra_q) > typos,
        "name_typo": typos > 0,
        "name_replaced": bool(q_core) and not (set(q_core) & set(t_core)) and typos == 0,
        "addr_empty": not ta_raw.strip(),
        "addr_null_token": bool(re.search(r"(^|,)\s*null\s*(,|$)", ta_raw, re.I)),
        "addr_po_box_pmb": bool(re.search(r"\b(po box|pmb)\b", ta_raw, re.I)),
        "addr_native": is_native_script(ta_raw),
        "addr_reordered": bool(ta_raw.strip()) and sorted(qa_t) == sorted(ta_t) and qa_t != ta_t,
        "house_same": bool(qh) and qh == th,
        "house_missing": bool(qh) and not th and bool(ta_raw.strip()),
        "house_digit_dropped": bool(qh and th) and len(th) == len(qh) - 1 and Levenshtein.distance(qh, th) == 1,
        "house_digit_changed": bool(qh and th) and len(th) == len(qh) and Levenshtein.distance(qh, th) == 1,
        "house_other_clash": bool(qh and th) and qh != th and Levenshtein.distance(qh, th) > 1,
    }


def catalog(n_pairs: int = 200_000, seed: int = 0) -> tuple[pl.DataFrame, dict[str, list[tuple[str, int]]]]:
    split = load_split()
    ids = subset_ids(split, "train")
    links = io.truth_long(io.load_truth().filter(pl.col("s1_id").is_in(ids.implode())))
    links = links.sample(n=min(n_pairs, links.height), seed=seed)
    s1 = io.load_source("train", "S1").select(pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("qn"),
                                             pl.col("business_address").alias("qa"), "country")
    tg = pl.concat([io.load_source("train", s) for s in io.TARGET_SOURCES]).select(
        pl.col("entity_id").alias("target_id"), pl.col("business_name").alias("tn"),
        pl.col("business_address").alias("ta"))
    d = links.join(s1, on="s1_id").join(tg, on="target_id")
    inserted = {c: Counter() for c in d["country"].unique()}
    legal = {c: frozenset(profile_for(c).legal) for c in d["country"].unique()}
    rows = [pair_ops(qn, qa, tn, ta, c, inserted[c], legal[c])
            for qn, qa, tn, ta, c in zip(d["qn"], d["qa"], d["tn"], d["ta"], d["country"])]
    ops = pl.DataFrame(rows).with_columns(d["country"], d["src"])
    rates = ops.group_by("country").agg(pl.exclude("src").mean()).sort("country")
    top = {c: cnt.most_common(40) for c, cnt in inserted.items()}
    return rates, top

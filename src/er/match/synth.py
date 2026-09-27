"""Labeled pairs for a country without labels: synthetic noisy copies (positives) and S1–S1 negatives.

Positives replay the train noise generator as catalogued by `normalize.noise_catalog` (reports/noise/op_rates.csv):
an S1 record and a corrupted copy of it. Negatives are pairs of distinct S1 records sharing a name key or a street +
house number: S1 is deduplicated in every split, so these are exact non-matches (the twin / decoy cases). Uses only
the split's own unlabeled records.
"""

from __future__ import annotations

import csv
import json
import random
import re

import polars as pl

from er import io
from er.normalize.profiles import profile_for

_OPS = io.REPORTS / "noise" / "op_rates.csv"
_INSERTED = io.REPORTS / "noise" / "inserted_tokens.json"
_VOWELS = {"a": "àâä", "e": "éèêë", "i": "îï", "o": "ôö", "u": "ùûü", "c": "ç"}
_SYL = [c + v for c in "bcdfghklmnprstvxz" for v in "aeiou"]


def op_rates(countries: list[str] | None = None) -> dict[str, float]:
    """Per-operator probability averaged over the labeled countries (or `countries`, e.g. US only for an India
    rehearsal); native-script ops off (unseen scripts)."""
    with _OPS.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = [c for c in rows[0] if c != "op" and (countries is None or c in countries)]
    return {r["op"]: (0.0 if r["op"].endswith("_native") else sum(float(r[c]) for c in cols) / len(cols)) for r in rows}


def _inserted() -> list[str]:
    d = json.loads(_INSERTED.read_text(encoding="utf-8"))
    return sorted({t for toks in d.values() for t, n in toks if n >= 20 and t.isalpha()})


class Corruptor:
    def __init__(self, country: str, seed: int = 0, rate_countries: list[str] | None = None):
        self.r = op_rates(rate_countries)
        self.rng = random.Random(seed)
        prof = profile_for(country)
        self.legal = sorted(prof.legal) or ["ltd", "inc"]
        variants: dict[str, list[str]] = {}
        for raw, canon in prof.street.items():
            variants.setdefault(canon, []).append(raw)
        self.street = {v: vs for vs in variants.values() for v in vs if len(vs) > 1}
        self.inserted = _inserted()

    def _p(self, op: str) -> bool:
        return self.rng.random() < self.r.get(op, 0.0)

    def _word(self) -> str:
        return "".join(self.rng.choice(_SYL) for _ in range(self.rng.randint(3, 4))).capitalize()

    def name(self, s: str) -> str:
        rng, toks = self.rng, s.split()
        if not toks:
            return s
        if self._p("name_replaced"):
            toks = [self._word()] + [t for t in toks if t.lower().strip(".,") in self.legal]
        if self._p("legal_changed"):
            toks = [t for t in toks if t.lower().strip(".,") not in self.legal]
            if rng.random() < 0.7:
                toks.append(rng.choice(self.legal).upper() if rng.random() < 0.5 else rng.choice(self.legal).title())
        if self._p("name_token_dropped") and len(toks) > 1:
            toks.pop(rng.randrange(len(toks)))
        if self._p("name_token_inserted"):
            toks.insert(rng.randrange(len(toks) + 1), rng.choice(self.inserted))
        if self._p("name_reordered") and len(toks) > 1:
            rng.shuffle(toks)
        if self._p("name_typo"):
            i = rng.randrange(len(toks))
            w = toks[i]
            if len(w) > 3:
                j = rng.randrange(1, len(w) - 1)
                op = rng.random()
                w = w[:j] + w[j + 1:] if op < 0.33 else w[:j] + rng.choice("aeiourstln") + w[j:] if op < 0.66 else \
                    w[:j - 1] + w[j] + w[j - 1] + w[j + 1:]
                toks[i] = w
        out = " ".join(toks)
        if self._p("name_accent_injected"):
            idx = [k for k, ch in enumerate(out) if ch.lower() in _VOWELS]
            if idx:
                k = rng.choice(idx)
                out = out[:k] + rng.choice(_VOWELS[out[k].lower()]) + out[k + 1:]
        if self._p("name_bracket_noise"):
            out = rng.choice([f"[{out}]", f"{out} ()", f"<{out}>", f"-- {out}"])
        if self._p("name_domain"):
            out = re.sub(r"[^a-z0-9]", "", out.lower())[:24] + rng.choice([".com", ".fr", ".net"])
        if self._p("name_case_changed"):
            out = rng.choice([out.upper(), out.lower(), out.title()])
        if self._p("name_numeric_tail"):
            out += f" - {rng.randint(10 ** 6, 10 ** 8)}"
        return out

    def address(self, s: str) -> str:
        rng = self.rng
        if not s or self._p("addr_empty"):
            return ""
        segs = [x.strip() for x in s.split(",") if x.strip()]
        words = segs[0].split() if segs else []
        num = next((k for k, w in enumerate(words) if w.isdigit()), None)
        if num is not None:
            h = words[num]
            if self._p("house_missing"):
                words.pop(num)
            elif self._p("house_digit_dropped") and len(h) > 1:
                j = rng.randrange(len(h))
                words[num] = h[:j] + h[j + 1:]
            elif self._p("house_digit_changed"):
                j = rng.randrange(len(h))
                words[num] = h[:j] + str((int(h[j]) + rng.randint(1, 9)) % 10) + h[j + 1:]
            elif self._p("house_other_clash"):
                words[num] = str(rng.randint(1, 9999))
        words = [rng.choice(self.street[w.lower()]).title() if w.lower() in self.street and rng.random() < 0.5 else w
                 for w in words]
        if segs:
            segs[0] = " ".join(words)
        if self._p("addr_reordered") and len(segs) > 1:
            rng.shuffle(segs)
        if self._p("addr_null_token"):
            segs.insert(rng.randrange(len(segs) + 1), "NULL")
        out = ", ".join(segs)
        return out.upper() if rng.random() < 0.3 else out


def _text(name: str, addr: str) -> str:
    return f"{name} | {addr}"


def positives(split: str, s1_ids: pl.Series, country: str, seed: int = 0,
              rate_countries: list[str] | None = None) -> pl.DataFrame:
    s1 = io.load_source(split, "S1").filter(pl.col("entity_id").is_in(s1_ids.implode()))
    c = Corruptor(country, seed, rate_countries)
    a, b = [], []
    for n, ad in zip(s1["business_name"].fill_null(""), s1["business_address"].fill_null("")):
        a.append(_text(n, ad))
        b.append(_text(c.name(n), c.address(ad)))
    return pl.DataFrame({"a": a, "b": b, "y": [1] * len(a)})


def s1_negatives(split: str, country: str, n: int, seed: int = 0, config: str = "configs/blocking/b_v9.toml",
                 rate_countries: list[str] | None = None) -> pl.DataFrame:
    """Distinct S1 records sharing a normalized name key (twins) or street + house (decoys): exact non-matches."""
    import tomllib
    from pathlib import Path

    from er.normalize.pipeline import NormConfig, load_norm

    cfg = tomllib.loads(Path(config).read_text(encoding="utf-8"))
    norm = load_norm(split, "S1", NormConfig(**cfg.get("norm", {}))).filter(pl.col("country") == country)
    raw = io.load_source(split, "S1").select("entity_id", "business_name", "business_address")
    d = norm.select("entity_id", "name_min_key", "addr_house", "addr_content").join(raw, on="entity_id")
    pairs = []
    for key in (["name_min_key"], ["addr_content", "addr_house"]):
        g = d.filter(pl.all_horizontal(pl.col(k) != "" for k in key))
        g = g.with_columns(pl.len().over(key).alias("_g")).filter((pl.col("_g") >= 2) & (pl.col("_g") <= 50))
        g = g.with_columns(pl.int_range(pl.len()).shuffle(seed=seed).over(key).alias("_r"))
        left = g.filter(pl.col("_r") == 0).select(*key, pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"))
        right = g.filter(pl.col("_r") == 1).select(*key, pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2"))
        pairs.append(left.join(right, on=key).select("n1", "a1", "n2", "a2"))
    p = pl.concat(pairs).sample(n=min(n, sum(x.height for x in pairs)), seed=seed)
    c = Corruptor(country, seed + 1, rate_countries)
    a = [_text(x or "", y or "") for x, y in zip(p["n1"], p["a1"])]
    b = [_text(c.name(x or ""), c.address(y or "")) for x, y in zip(p["n2"], p["a2"])]
    return pl.DataFrame({"a": a, "b": b, "y": [0] * len(a)})

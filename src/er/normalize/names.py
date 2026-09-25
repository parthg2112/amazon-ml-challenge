"""N1 name structure: legal forms anywhere, alias split, domain stems, core/sorted/concat/fold keys."""

from __future__ import annotations

import re

from er.normalize.profiles.base import ALIAS_MARKERS, NAME_STOPWORDS, Profile
from er.normalize.text import base_text, fold_token, tokens

_DOMAIN = re.compile(
    r"^\s*(?:https?\s*:\s*/+)?(?:www\.)?([a-z0-9][a-z0-9-]*)\."
    r"(?:com|net|org|co|in|biz|info|io|us|fr|online|shop|store|site|xyz|co\.in|org\.in|net\.in)\s*/?\s*$"
)
_ID_NOISE = re.compile(r"[(\[]\s*(?:id|ref|reg|no)\s*[:#.]?\s*\d+\s*[)\]]")
_ALIAS = frozenset(ALIAS_MARKERS)

# N5 (additive `name_min` view). Vocabularies measured on train-fold true pairs: reports/noise/.
_ALIAS_PHRASES = (("doing", "business", "as"), ("formerly", "known", "as"), ("also", "known", "as"),
                  ("known", "as"), ("trading", "as"), ("nee",), ("fka",), ("aka",), ("dba",), ("formerly",))
_INSERTED = frozenset({
    "center", "centre", "services", "service", "partners", "the", "enterprises", "labs", "sys", "trading",
    "trust", "foundation", "commission", "federation", "association", "council", "district", "board",
    "society", "ventures", "one", "authority", "www", "com", "id",
})
_HONORIFICS = frozenset({"sri", "shri", "shree", "smt", "dr", "mr", "mrs", "ms"})
_HOMOGLYPH = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "4": "a"})
_TD = str.maketrans({"d": "t"})


def _legal_fold(t: str) -> str:
    return fold_token(t.translate(_TD))


def _is_legal(t: str, profile: Profile, legal_folds: frozenset[str], real_words: frozenset[str]) -> bool:
    # Fuzzy match only for tokens never seen in clean S1 names: keeps real words like "prabhat", "limit".
    if t in profile.legal or (len(t) > 3 and t not in real_words and _legal_fold(t) in legal_folds):
        return True
    return any(len(t) == 2 * len(w) and t == w * 2 for w in profile.legal if len(w) > 1)  # llcllc


_LEGAL_FOLDS: dict[str, frozenset[str]] = {}


def name_min(toks: list[str], profile: Profile, real_words: frozenset[str] = frozenset()) -> list[str]:
    """Core name with generator noise removed: alias phrases, inserted words, honorifics, fold-matched legal
    variants (piraivet, limitet), homoglyphs (c0m, 5ervices). Falls back to the input when nothing is left."""
    if profile.name not in _LEGAL_FOLDS:
        _LEGAL_FOLDS[profile.name] = frozenset(_legal_fold(w) for w in profile.legal if len(w) > 3)
    folds = _LEGAL_FOLDS[profile.name]
    toks = [t.translate(_HOMOGLYPH) if not t.isdigit() and sum(c.isalpha() for c in t) >= 2 else t for t in toks]
    drop = [False] * len(toks)
    for phrase in _ALIAS_PHRASES:
        n = len(phrase)
        for i in range(len(toks) - n + 1):
            if tuple(toks[i:i + n]) == phrase:
                drop[i:i + n] = [True] * n
    out = [t for t, d in zip(toks, drop) if not d and t not in _INSERTED and t not in _HONORIFICS
           and t not in NAME_STOPWORDS and not _is_legal(t, profile, folds, real_words)]
    return list(dict.fromkeys(out)) or toks


def normalize_name(raw: str | None, profile: Profile, translit: bool = True,
                   token_map: dict[str, str] | None = None, real_words: frozenset[str] = frozenset()) -> dict:
    text = base_text(raw, translit, profile.apostrophe_splits)
    m = _DOMAIN.match(text)
    if m:
        stem = m.group(1).replace("-", "")
        return _pack([stem], [stem], [], [], is_domain=True, minimal=[stem])

    toks = tokens(_ID_NOISE.sub(" ", text))
    if token_map:
        toks = [token_map.get(t, t) for t in toks]
    stop = NAME_STOPWORDS | profile.extra_stopwords

    parts: list[list[str]] = [[]]
    for t in toks:
        if t in _ALIAS and parts[-1]:
            parts.append([])
        else:
            parts[-1].append(t)
    parts = [p for p in parts if p]

    legal: set[str] = set()
    core: list[str] = []
    part_cores: list[list[str]] = []
    for p in parts:
        pc = []
        for t in p:
            canon = profile.legal.get(t)
            if canon:
                legal.add(canon)
            elif t not in stop:
                pc.append(t)
        part_cores.append(pc)
        core.extend(pc)
    if not core:  # name made only of legal words / stopwords: keep it rather than emit nothing
        core = [t for t in toks if t not in _ALIAS]
    full = [t for t in toks if t not in _ALIAS]
    alts = [pc for pc in part_cores if pc] if len(part_cores) > 1 else []
    return _pack(full, core, sorted(legal), alts, is_domain=False, minimal=name_min(toks, profile, real_words))


def _pack(full: list[str], core: list[str], legal: list[str], alts: list[list[str]], is_domain: bool,
          minimal: list[str]) -> dict:
    return {
        "name_min": " ".join(minimal),
        "name_min_key": " ".join(sorted(set(minimal))),
        "name_full": " ".join(full),
        "name_core": " ".join(core),
        "name_legal": " ".join(legal),
        "name_sorted": " ".join(sorted(set(core))),
        "name_concat": "".join(core),
        "name_fold": " ".join(fold_token(t) for t in core),
        "name_alts": [" ".join(sorted(set(a))) for a in alts],
        "name_domain": is_domain,
    }

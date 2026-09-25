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


def normalize_name(raw: str | None, profile: Profile, translit: bool = True,
                   token_map: dict[str, str] | None = None) -> dict:
    text = base_text(raw, translit, profile.apostrophe_splits)
    m = _DOMAIN.match(text)
    if m:
        stem = m.group(1).replace("-", "")
        return _pack([stem], [stem], [], [], is_domain=True)

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
    return _pack(full, core, sorted(legal), alts, is_domain=False)


def _pack(full: list[str], core: list[str], legal: list[str], alts: list[list[str]], is_domain: bool) -> dict:
    return {
        "name_full": " ".join(full),
        "name_core": " ".join(core),
        "name_legal": " ".join(legal),
        "name_sorted": " ".join(sorted(set(core))),
        "name_concat": "".join(core),
        "name_fold": " ".join(fold_token(t) for t in core),
        "name_alts": [" ".join(sorted(set(a))) for a in alts],
        "name_domain": is_domain,
    }

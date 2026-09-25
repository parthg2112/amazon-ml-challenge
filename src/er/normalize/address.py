"""N3 profile-driven address parsing: region canon, street canon, house/postal/number extraction."""

from __future__ import annotations

import re

from er.normalize.profiles.base import Profile
from er.normalize.text import base_text, normalize_text, tokens

_REGION_CACHE: dict[tuple[str, bool], tuple[dict[str, str], int]] = {}
_CANON_CACHE: dict[str, frozenset[str]] = {}
_HOUSE_MARKER = re.compile(
    r"(?:\b(?:h|house|door|plot|flat|shop|survey|premises|bldg|building|s|d)\s*\.?\s*no\b|\bno\b|#|\bnumero\b)"
    r"\s*[.:#\-]*\s*(?:[a-z]\s*-?\s*)?(\d+)"
)
_SEGMENT_NUMBER = re.compile(r"(?:^|,)\s*(\d+)")
_DIGITS = re.compile(r"\d+")
_MARKER_TOKENS = frozenset({"no", "near", "opp", "behind", "flat", "plot", "shop", "floor", "fl", "ste", "apt"})


def _regions(profile: Profile, translit: bool) -> tuple[dict[str, str], int]:
    key = (profile.name, translit)
    if key not in _REGION_CACHE:
        table = {}
        for variant, rid in profile.regions.items():
            norm = normalize_text(variant, translit, profile.apostrophe_splits)
            if norm:
                table[norm] = rid
        longest = max((len(k.split()) for k in table), default=0)
        _REGION_CACHE[key] = (table, longest)
    return _REGION_CACHE[key]


def _street_canon(profile: Profile) -> frozenset[str]:
    if profile.name not in _CANON_CACHE:
        _CANON_CACHE[profile.name] = frozenset(profile.street.values())
    return _CANON_CACHE[profile.name]


def _postal(text: str, profile: Profile) -> str:
    digits = profile.postal_digits
    if not digits:
        return ""
    # Postal code closes a segment ("WB 700089", "NC 27260-1234") or, in France, precedes the city.
    pat = rf"(?<!\d)(\d{{{digits}}})(?:-\d{{4}})?\s*(?:,|$)"
    found = re.findall(pat, text)
    if not found and profile.postal_before_city:
        found = re.findall(r"(?:^|,)\s*(\d{5})\s+[a-z]", text)
    return found[-1] if found else ""


def normalize_address(raw: str | None, profile: Profile, translit: bool = True,
                      region_map: dict[str, str] | None = None) -> dict:
    text = base_text(raw, translit, profile.apostrophe_splits)
    if not text.strip(" ,"):
        return {"addr_norm": "", "addr_content": "", "addr_region": "", "addr_house": "",
                "addr_numbers": [], "addr_postal": "", "addr_empty": True}

    postal = _postal(text, profile)
    regions, longest = _regions(profile, translit)

    region = ""
    seg_tokens: list[list[str]] = []
    for seg in text.split(","):
        toks = tokens(seg)
        if not toks:
            continue
        words = [t for t in toks if not t.isdigit()]
        key = " ".join(words)
        rid = regions.get(key) or (region_map.get(key) if region_map else None)
        if rid:
            region = rid
            toks = [t if t.isdigit() else "" for t in toks]
            toks = [t for t in toks if t and t != postal] + [rid]
        seg_tokens.append(toks)

    # Region glued to the last segment without a comma ("Kolkata West Bengal 700089").
    if not region and seg_tokens:
        last = seg_tokens[-1]
        words = [t for t in last if not t.isdigit()]
        for n in range(min(longest, len(words) - 1), 0, -1):
            rid = regions.get(" ".join(words[-n:]))
            if rid:
                region = rid
                keep = words[:-n]
                seg_tokens[-1] = keep + [t for t in last if t.isdigit()] + [rid]
                break

    street = profile.street
    canon = _street_canon(profile)
    flat = [street.get(t, t) for toks in seg_tokens for t in toks]

    m = _HOUSE_MARKER.search(text) or _SEGMENT_NUMBER.search(text)
    house = str(int(m.group(1))) if m else ""
    if house == postal.lstrip("0"):
        house = ""
    numbers = sorted({str(int(d)) for d in _DIGITS.findall(text) if d != postal and len(d) <= 7})[:8]
    content = [t for t in flat if not t.isdigit() and t != region and t not in _MARKER_TOKENS
               and t not in canon and len(t) > 1]
    return {
        "addr_norm": " ".join(flat),
        "addr_content": " ".join(content),
        "addr_region": region,
        "addr_house": house,
        "addr_numbers": numbers,
        "addr_postal": postal,
        "addr_empty": False,
    }

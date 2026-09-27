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
# Unit/landmark extraction is purely additive (new fields alongside the ones above); it never changes
# addr_content, addr_house, addr_region, addr_numbers, or addr_postal, so Step 2/3 metrics stay reproducible.
_UNIT_MARKER = re.compile(r"\b(?:apt|ste|suite|unit|fl|floor)\.?\s*[.:#\-]*\s*(\d+[a-z]?)\b")
_LANDMARK_TRIGGERS = frozenset({"near", "opp", "behind", "opposite"})


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
        _CANON_CACHE[profile.name] = frozenset(profile.street.values()) - profile.street_content_keep
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


def _city_after_postal(text: str, profile: Profile) -> str:
    """"59000 LILLE" convention: city immediately follows the postal code (France)."""
    if not profile.postal_before_city or not profile.postal_digits:
        return ""
    pat = rf"(?:^|,)\s*\d{{{profile.postal_digits}}}\s+([a-z][a-z'\-\s]*?)(?:\s*,|\s*$)"
    m = re.search(pat, text)
    return m.group(1).strip() if m else ""


def _content_filter(toks: list[str], region: str, canon: frozenset[str]) -> list[str]:
    return [t for t in toks if not t.isdigit() and t != region and t not in _MARKER_TOKENS
            and t not in canon and len(t) > 1]


def normalize_address(raw: str | None, profile: Profile, translit: bool = True,
                      region_map: dict[str, str] | None = None) -> dict:
    text = base_text(raw, translit, profile.apostrophe_splits)
    if not text.strip(" ,"):
        return {"addr_norm": "", "addr_content": "", "addr_street": "", "addr_region": "", "addr_house": "",
                "addr_numbers": [], "addr_postal": "", "addr_city": "", "addr_unit": "",
                "addr_landmark": "", "addr_empty": True}

    postal = _postal(text, profile)
    regions, longest = _regions(profile, translit)

    region = ""
    region_seg_idx = -1  # additive: which comma-segment carried the region, for city extraction below
    city_glued: list[str] = []
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
            region_seg_idx = len(seg_tokens)
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
                city_glued = keep  # words right before the region name double as the city, additively
                seg_tokens[-1] = keep + [t for t in last if t.isdigit()] + [rid]
                break

    street = profile.street
    canon = _street_canon(profile)
    flat_segs = [[street.get(t, t) for t in toks] for toks in seg_tokens]
    flat = [t for seg in flat_segs for t in seg]

    m = _HOUSE_MARKER.search(text) or _SEGMENT_NUMBER.search(text)
    house = str(int(m.group(1))) if m else ""
    if house == postal.lstrip("0"):
        house = ""
    numbers = sorted({str(int(d)) for d in _DIGITS.findall(text) if d != postal and len(d) <= 7})[:8]

    # Additive: city, unit, and landmark are separate fields alongside addr_content, never subtracted
    # from it, so Step 2/3 measurements (which depend on addr_content) are unaffected.
    if profile.postal_before_city:
        city = " ".join(_content_filter(tokens(_city_after_postal(text, profile)), region, canon))
    elif region_seg_idx > 0:
        city = " ".join(_content_filter(flat_segs[region_seg_idx - 1], region, canon))
    elif city_glued:
        city = " ".join(_content_filter([street.get(t, t) for t in city_glued], region, canon))
    else:
        city = ""
    unit_m = _UNIT_MARKER.search(text)
    unit = unit_m.group(1) if unit_m else ""
    landmark_toks: list[str] = []
    for seg in flat_segs:
        i = 0
        while i < len(seg):
            if seg[i] in _LANDMARK_TRIGGERS:
                j = i + 1
                while j < len(seg) and not seg[j].isdigit() and seg[j] not in _MARKER_TOKENS \
                        and seg[j] not in _LANDMARK_TRIGGERS:
                    if seg[j] != region and seg[j] not in canon and len(seg[j]) > 1:
                        landmark_toks.append(seg[j])
                    j += 1
                i = j
            else:
                i += 1
    landmark = " ".join(landmark_toks)

    content = _content_filter(flat, region, canon)
    # The first address segment is the street line for the challenge's comma-separated addresses.
    # Keep it separately: city agreement is useful elsewhere, but must not drown the street-name
    # signal when several S1 records contest one target.
    street_content = _content_filter(flat_segs[0], region, canon) if flat_segs else []
    return {
        "addr_norm": " ".join(flat),
        "addr_content": " ".join(content),
        "addr_street": " ".join(street_content),
        "addr_region": region,
        "addr_house": house,
        "addr_numbers": numbers,
        "addr_postal": postal,
        "addr_city": city,
        "addr_unit": unit,
        "addr_landmark": landmark,
        "addr_empty": False,
    }

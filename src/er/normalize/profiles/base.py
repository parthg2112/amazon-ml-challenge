"""Country profile = dictionaries only; all parsing logic lives in er.normalize."""

from __future__ import annotations

from dataclasses import dataclass, field

# Legal forms safe to strip in any country (GENERIC and union base).
GENERIC_LEGAL = {
    "llc": "llc", "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt", "llp": "llp", "plc": "plc",
    "co": "co", "company": "co", "lp": "lp", "gmbh": "gmbh", "sarl": "sarl", "sas": "sas",
    "sasu": "sasu", "eurl": "eurl",
}
NAME_STOPWORDS = {"the", "of", "and", "de", "du", "des", "la", "le", "les", "l", "d", "et"}
ALIAS_MARKERS = ("aka", "dba", "fka", "formerly", "ta")  # "t/a" normalizes to "t a" → handled in names.py
GENERIC_STREET = {
    "rd": "road", "road": "road", "st": "street", "street": "street", "ave": "avenue", "avenue": "avenue",
    "blvd": "boulevard", "boulevard": "boulevard", "no": "no", "nr": "near", "near": "near",
}


@dataclass(frozen=True)
class Profile:
    name: str
    legal: dict[str, str]
    street: dict[str, str]  # address token → canonical token
    regions: dict[str, str]  # raw region variant → canonical region id (normalized at load)
    postal_digits: int | None = None
    postal_before_city: bool = False  # "59000 LILLE"
    apostrophe_splits: bool = False  # French elision: l'avenue → l avenue
    extra_stopwords: frozenset[str] = field(default_factory=frozenset)
    # Canonicalized address tokens which look like street markers but are meaningful street words.
    # They are retained in addr_content / addr_street rather than being stripped with "rue", "av", etc.
    street_content_keep: frozenset[str] = field(default_factory=frozenset)


def merged(base: dict[str, str], extra: dict[str, str]) -> dict[str, str]:
    out = dict(base)
    out.update(extra)
    return out

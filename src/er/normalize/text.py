"""N0 primitives (Unicode, case, punctuation, initialisms) and the N2 phonetic fold."""

from __future__ import annotations

import re
import sys
import unicodedata

from anyascii import anyascii


def _punct_table() -> dict[int, str]:
    # Every punctuation/symbol/separator/control codepoint → space; keeps letters, digits, and Indic marks.
    table = {}
    for cp in range(sys.maxunicode + 1):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        if unicodedata.category(chr(cp))[0] in "PSZC":
            table[cp] = " "
    return table


_PUNCT = _punct_table()
_LATIN_MARKS = re.compile("[̀-ͯ]")
_INITIALISM = re.compile(r"\b(?:[a-z]\.\s?){2,}")
_APOSTROPHES = re.compile(r"['’`´]")


def base_text(raw: str | None, translit: bool = True, apostrophe_splits: bool = False) -> str:
    """NFKC → (anyascii | strip Latin accents) → lowercase, before tokenization."""
    if not raw:
        return ""
    s = unicodedata.normalize("NFKC", raw)
    if not s.isascii():
        if translit:
            s = anyascii(s)
        else:
            s = unicodedata.normalize("NFC", _LATIN_MARKS.sub("", unicodedata.normalize("NFKD", s)))
    s = s.lower().replace("&", " and ")
    s = _INITIALISM.sub(lambda m: m.group(0).replace(".", "").replace(" ", "") + " ", s)
    return _APOSTROPHES.sub(" " if apostrophe_splits else "", s)


def tokens(s: str) -> list[str]:
    """Split on punctuation; merge runs of ≥2 single letters (`l l c` → `llc`); drop repeated tokens."""
    toks = s.translate(_PUNCT).split()
    out: list[str] = []
    run: list[str] = []
    for t in toks + [""]:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run) if len(run) > 1 else run[0])
            run = []
        if t and (not out or out[-1] != t):
            out.append(t)
    return out


def normalize_text(raw: str | None, translit: bool = True, apostrophe_splits: bool = False) -> str:
    return " ".join(tokens(base_text(raw, translit, apostrophe_splits)))


def is_native_script(raw: str | None) -> bool:
    """True when the string holds non-Latin letters (not just accented Latin)."""
    if not raw or raw.isascii():
        return False
    for ch in raw:
        if ord(ch) > 0x024F and ch.isalpha():
            return True
    return False


_FOLD_DIGRAPHS = (("ph", "f"), ("sh", "s"), ("ch", "c"), ("kh", "k"), ("gh", "g"), ("th", "t"),
                  ("dh", "d"), ("bh", "b"), ("jh", "j"), ("ck", "k"), ("x", "ks"))
_FOLD_MAP = str.maketrans({"q": "k", "c": "k", "w": "b", "v": "b", "z": "j", "y": "i"})
_VOWELS = set("aeiou")
_NASAL = re.compile(r"m(?=[^aeioubpm])")


def fold_token(t: str) -> str:
    """Vowel-free consonant skeleton; absorbs anyascii's dropped Indic vowels and b/v, ph/f swaps."""
    if not t.isalpha() or not t.isascii():
        return t
    for a, b in _FOLD_DIGRAPHS:
        t = t.replace(a, b)
    t = _NASAL.sub("n", t).translate(_FOLD_MAP)
    head, out = t[0], []
    for ch in t[1:]:
        if ch in _VOWELS or ch == "h":
            continue
        if not out or out[-1] != ch:
            out.append(ch)
    skel = head + "".join(out)
    return skel if len(skel) > 1 else t

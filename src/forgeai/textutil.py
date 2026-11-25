"""Normalisation and fuzzy-matching helpers shared by the eval generator and the judge."""
from __future__ import annotations

import re
import unicodedata

from .manifest import STOPWORDS

_NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?![\w])")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    s = re.sub(r"\[[^\]]*\]", " ", s)  # citation labels
    s = re.sub(r"[^\w%$€£₹.\- ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def numbers(s: str) -> set[str]:
    out = set()
    for m in _NUM.findall(s or ""):
        v = m.replace(",", "").lstrip("+")
        try:
            f = float(v)
            out.add(str(int(f)) if f.is_integer() else str(round(f, 4)))
        except ValueError:
            pass
    return out


def content_tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9][a-z0-9\-.]*", norm(s)) if t not in STOPWORDS and len(t) > 1}


def overlap(needle: str, haystack: str) -> float:
    """Fraction of the needle's content tokens present in the haystack."""
    a = content_tokens(needle)
    if not a:
        return 0.0
    b = content_tokens(haystack)
    return len(a & b) / len(a)


def quote_in(quote: str, source: str, min_len: int = 8) -> bool:
    """Is `quote` really (near-)verbatim in `source`? Defends the judge against invented quotes."""
    q, src = norm(quote), norm(source)
    if len(q) < min_len:
        return False
    if q in src:
        return True
    # tolerate small model slips: long quote whose tokens are 90% present in order-insensitive way
    return len(q) >= 25 and overlap(q, src) >= 0.9 and numbers(quote) <= numbers(source)


def fact_supported(fact: str, text: str) -> bool:
    """Deterministic: all numbers in the fact appear in the text, and most content words do."""
    nf = numbers(fact)
    if nf and not nf <= numbers(text):
        return False
    if nf and not {t for t in content_tokens(fact) if not re.fullmatch(r"[\d.,\-]+", t)}:
        return True  # the fact IS a number (e.g. "3", "45.50") and it is present
    return overlap(fact, text) >= (0.6 if nf else 0.7)

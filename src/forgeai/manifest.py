"""Pre-flight Evidence Manifest: a deterministic map of what the documents actually contain.

Built before any LLM call. Downstream stages see this instead of guessing, and the judge
uses it to verify that cited files / pages / rows really exist.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .parsing import LoadedDoc, detect_headings, load_folder
from .schemas import ColumnInfo, EvidenceManifest, FileEntry, PageInfo

STOPWORDS = set(
    """a an and are as at be been but by can could did do does for from had has have he her his how i if in into is it its
    may me might must my no not of on or our shall she should so such than that the their them then there these they this
    those to under up upon us was we were what when where which while who whom why will with would you your any all also
    each per other only more most same both either one two three within without about after before over between during
    page section policy document table row rows column columns shall""".split()
)

_CODE = re.compile(r"\b[A-Z]{1,5}[-_/]?\d{2,}[A-Z0-9\-]*\b")
_MONEY = re.compile(r"(?:USD|EUR|INR|GBP|SGD|\$|€|£|₹)\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|INR|GBP|SGD)\b")
_NUM_UNIT = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:%|percent|days?|weeks?|months?|years?|hours?|hrs?|minutes?|working days|calendar days|"
    r"kg|tonnes?|litres?|liters?|nm|nautical miles|knots|units?|pcs|pieces)\b",
    re.I,
)
_CAPPHRASE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+(?:of|and|for|the|&)\s+)?){1,4}[A-Z][a-z]+\b|\b[A-Z]{2,6}\b")


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z][a-z0-9\-]{2,}", text.lower()) if t not in STOPWORDS]


def _entities(text: str, limit: int = 25) -> list[str]:
    c: Counter[str] = Counter()
    for m in _CODE.findall(text):
        c[m] += 3
    for m in _CAPPHRASE.findall(text):
        m = m.strip()
        if m.lower() not in STOPWORDS and len(m) > 2:
            c[m] += 1
    return [e for e, _ in c.most_common(limit)]


def _column_info(name: str, values: list[str]) -> ColumnInfo:
    non_null = [v for v in values if v not in ("", None)]
    nums = []
    for v in non_null:
        try:
            nums.append(float(str(v).replace(",", "").replace("$", "")))
        except ValueError:
            pass
    if non_null and len(nums) == len(non_null):
        dtype = "integer" if all(n.is_integer() for n in nums) else "number"
    elif non_null and all(re.match(r"\d{4}-\d{2}-\d{2}", str(v)) for v in non_null):
        dtype = "date"
    else:
        dtype = "text"
    uniq = list(dict.fromkeys(non_null))
    return ColumnInfo(
        name=name,
        dtype=dtype,
        n_unique=len(uniq),
        n_null=len(values) - len(non_null),
        samples=[str(v)[:60] for v in uniq[:5]],
        min=min(nums) if nums and dtype != "text" else None,
        max=max(nums) if nums and dtype != "text" else None,
    )


def _file_entry(d: LoadedDoc) -> FileEntry:
    full = "\n".join(u.text for u in d.units)
    fe = FileEntry(
        filename=d.filename,
        kind=d.kind,  # type: ignore[arg-type]
        n_chars=len(full),
        warnings=list(d.warnings),
        numbers_with_units=list(dict.fromkeys(m.group(0) for m in list(_MONEY.finditer(full)) + list(_NUM_UNIT.finditer(full))))[:30],
    )
    if d.kind == "pdf":
        fe.n_pages = max((u.page for u in d.units), default=0)
        for u in d.units:
            heads = detect_headings(u.text)[:8]
            fe.pages.append(PageInfo(page=u.page, n_chars=len(u.text), headings=heads, preview=u.text[:160].replace("\n", " ")))
            fe.sections.extend(heads)
        fe.title = (d.units[0].text.splitlines()[0][:100] if d.units else d.filename)
        fe.key_entities = _entities(full)
    elif d.kind == "csv":
        fe.n_rows = len(d.table or [])
        cols = d.columns or []
        fe.columns = [_column_info(c, [r.get(c, "") for r in (d.table or [])]) for c in cols]
        fe.title = f"Table with {fe.n_rows} rows x {len(cols)} columns"
        # entities for CSV: values of identifier-like / low-cardinality text columns
        ents: list[str] = []
        for ci in fe.columns:
            if ci.dtype == "text":
                ents.extend(ci.samples[:4])
        fe.key_entities = ents[:25]
    else:
        fe.sections = [u.section for u in d.units]
        fe.title = d.units[0].section if d.units else d.filename
        fe.key_entities = _entities(full)
    fe.sections = list(dict.fromkeys(fe.sections))[:40]
    return fe


def build_manifest(folder: str | Path, docs: list[LoadedDoc] | None = None) -> tuple[EvidenceManifest, list[LoadedDoc]]:
    skipped: list[str] = []
    if docs is None:
        docs, skipped = load_folder(folder)
    files = [_file_entry(d) for d in docs]
    df: Counter[str] = Counter()
    tf: Counter[str] = Counter()
    for d in docs:
        toks = _tokens(" ".join(u.text for u in d.units))
        tf.update(toks)
        df.update(set(toks))
    n = max(1, len(docs))
    # terms that are frequent overall but not in literally every file score highest
    scored = {t: c * (1.0 + (n - df[t]) / n) for t, c in tf.items() if c >= 2}
    key_terms = [t for t, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:40]]
    warnings = [f"skipped unsupported file: {s}" for s in skipped]
    if not docs:
        warnings.append("no supported documents found (pdf, csv, txt, md)")
    for f in files:
        if f.n_chars == 0:
            warnings.append(f"{f.filename}: no extractable text - it cannot ground any answer")
    m = EvidenceManifest(
        folder=str(Path(folder)),
        files=files,
        corpus_key_terms=key_terms,
        total_chars=sum(f.n_chars for f in files),
        warnings=warnings,
    )
    return m, docs


def manifest_for_prompt(m: EvidenceManifest, max_chars: int = 3500) -> str:
    """Compact, model-friendly rendering. Small models choke on the full JSON."""
    lines = []
    for f in m.files:
        if f.kind == "pdf":
            lines.append(f"* {f.filename} (PDF, {f.n_pages} pages) - {f.title}")
            if f.sections:
                lines.append("  sections: " + "; ".join(f.sections[:12]))
        elif f.kind == "csv":
            lines.append(f"* {f.filename} (CSV, {f.n_rows} rows)")
            for c in f.columns[:15]:
                rng = f" range {c.min:g}-{c.max:g}" if c.min is not None and c.max is not None else ""
                lines.append(f"  - column '{c.name}' ({c.dtype}{rng}) e.g. {', '.join(c.samples[:3])}")
        else:
            lines.append(f"* {f.filename} (text) sections: " + "; ".join(f.sections[:12]))
        if f.key_entities:
            lines.append("  key entities: " + ", ".join(f.key_entities[:10]))
        if f.numbers_with_units:
            lines.append("  figures: " + ", ".join(f.numbers_with_units[:10]))
    if m.corpus_key_terms:
        lines.append("Corpus key terms: " + ", ".join(m.corpus_key_terms[:25]))
    out = "\n".join(lines)
    return out if len(out) <= max_chars else out[: max_chars - 20] + "\n...(truncated)"


def save_manifest(m: EvidenceManifest, path: Path) -> None:
    path.write_text(json.dumps(m.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")

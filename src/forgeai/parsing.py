"""Deterministic document loading and chunking. No LLM calls in this module."""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

SUPPORTED = {".pdf": "pdf", ".csv": "csv", ".txt": "text", ".md": "text", ".markdown": "text"}


@dataclass
class Unit:
    """A citable unit of source text: a PDF page, a CSV row, or a text section."""
    filename: str
    kind: str
    locator: str  # "p.3", "row 12", "§Leave Entitlement"
    text: str
    page: int = 0
    row: int = 0
    section: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    filename: str
    locator: str
    text: str
    kind: str


@dataclass
class LoadedDoc:
    path: Path
    filename: str
    kind: str
    units: list[Unit]
    warnings: list[str] = field(default_factory=list)
    table: list[dict] | None = None  # CSV only
    columns: list[str] | None = None


_HEADING = re.compile(r"^(\d+(\.\d+)*[.)]?\s+[A-Z][^\n]{2,80}|[A-Z][A-Z0-9 &/,\-()]{3,80}|#{1,6}\s+.+)$")


def _clean(text: str) -> str:
    text = text.replace("­", "").replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_headings(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if 3 < len(line) <= 90 and _HEADING.match(line) and not line.endswith((".", ",", ";")):
            out.append(line.lstrip("# ").strip())
    return out


def load_pdf(path: Path) -> LoadedDoc:
    from pypdf import PdfReader

    warnings: list[str] = []
    units: list[Unit] = []
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:  # noqa: BLE001
                warnings.append("encrypted PDF could not be decrypted")
        for i, page in enumerate(reader.pages, start=1):
            try:
                txt = _clean(page.extract_text() or "")
            except Exception as e:  # noqa: BLE001
                txt = ""
                warnings.append(f"page {i}: extraction error {e}")
            if not txt:
                warnings.append(f"page {i}: no extractable text (scanned image?)")
                continue
            units.append(Unit(path.name, "pdf", f"p.{i}", txt, page=i))
    except Exception as e:  # noqa: BLE001
        warnings.append(f"unreadable PDF: {e}")
    return LoadedDoc(path, path.name, "pdf", units, warnings)


def _sniff_read_csv(path: Path) -> tuple[list[str], list[dict]]:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    cols = [c.strip() for c in (reader.fieldnames or [])]
    rows = []
    for r in reader:
        rows.append({(k or "").strip(): (v or "").strip() if isinstance(v, str) else v for k, v in r.items() if k})
    return cols, rows


def load_csv(path: Path) -> LoadedDoc:
    warnings: list[str] = []
    try:
        cols, rows = _sniff_read_csv(path)
    except Exception as e:  # noqa: BLE001
        return LoadedDoc(path, path.name, "csv", [], [f"unreadable CSV: {e}"])
    units = []
    for i, r in enumerate(rows, start=1):
        text = "; ".join(f"{k}: {v}" for k, v in r.items() if v not in (None, ""))
        units.append(Unit(path.name, "csv", f"row {i}", text, row=i, meta=r))
    if not rows:
        warnings.append("CSV has no data rows")
    return LoadedDoc(path, path.name, "csv", units, warnings, table=rows, columns=cols)


def load_text(path: Path) -> LoadedDoc:
    try:
        text = _clean(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:  # noqa: BLE001
        return LoadedDoc(path, path.name, "text", [], [f"unreadable text file: {e}"])
    units: list[Unit] = []
    current, buf = "Introduction", []
    for line in text.splitlines():
        s = line.strip()
        is_md_heading = s.startswith("#")
        if is_md_heading or (s and _HEADING.match(s) and len(s) < 80 and not s.endswith(".")):
            if "\n".join(buf).strip():
                units.append(Unit(path.name, "text", f"§{current}", "\n".join(buf).strip(), section=current))
            current, buf = s.lstrip("# ").strip() or current, []
        else:
            buf.append(line)
    if "\n".join(buf).strip():
        units.append(Unit(path.name, "text", f"§{current}", "\n".join(buf).strip(), section=current))
    return LoadedDoc(path, path.name, "text", units)


def load_folder(folder: str | Path) -> tuple[list[LoadedDoc], list[str]]:
    folder = Path(folder)
    docs, skipped = [], []
    if not folder.is_dir():
        raise FileNotFoundError(f"documents folder not found: {folder}")
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or p.name.startswith((".", "~$")):
            continue
        if any(part.startswith((".", "__")) for part in p.relative_to(folder).parts[:-1]):
            continue  # hidden dirs, __pycache__, .git ...
        kind = SUPPORTED.get(p.suffix.lower())
        if kind == "pdf":
            docs.append(load_pdf(p))
        elif kind == "csv":
            docs.append(load_csv(p))
        elif kind == "text":
            docs.append(load_text(p))
        else:
            skipped.append(p.name)
    return docs, skipped


# --------------------------------------------------------------------------- chunking
def _windows(words: list[str], size: int, overlap: int) -> Iterable[list[str]]:
    if len(words) <= size:
        yield words
        return
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        win = words[start : start + size]
        if len(win) < max(15, overlap) and start > 0:
            break
        yield win
        if start + size >= len(words):
            break


def chunk_docs(docs: list[LoadedDoc], chunk_size: int = 180, overlap: int = 30, csv_rows_per_chunk: int = 1) -> list[Chunk]:
    """Chunks never cross a citable unit boundary, so every chunk has an exact locator."""
    chunks: list[Chunk] = []
    for d in docs:
        if d.kind == "csv":
            header = f"[{d.filename}] columns: {', '.join(d.columns or [])}\n"
            for i in range(0, len(d.units), csv_rows_per_chunk):
                group = d.units[i : i + csv_rows_per_chunk]
                loc = group[0].locator if len(group) == 1 else f"rows {group[0].row}-{group[-1].row}"
                text = header + "\n".join(f"{u.locator}: {u.text}" for u in group)
                chunks.append(Chunk(f"{d.filename}::{loc}", d.filename, group[0].locator, text, "csv"))
            continue
        for u in d.units:
            words = u.text.split()
            for j, win in enumerate(_windows(words, chunk_size, overlap)):
                cid = f"{d.filename}::{u.locator}::{j}"
                chunks.append(Chunk(cid, d.filename, u.locator, " ".join(win), d.kind))
    return chunks

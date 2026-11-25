"""Filter / search / aggregate rows of CSV tables from the knowledge base."""
from __future__ import annotations

import difflib
import re
from typing import Any

_OP = re.compile(r"^\s*(>=|<=|!=|>|<|=|~)\s*(.+)$")


def _num(v: Any) -> float | None:
    try:
        return float(str(v).replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def _resolve_col(name: str, cols: list[str]) -> str | None:
    if name in cols:
        return name
    low = {c.lower().replace(" ", "_"): c for c in cols}
    key = str(name).lower().replace(" ", "_")
    if key in low:
        return low[key]
    m = difflib.get_close_matches(key, list(low), n=1, cutoff=0.6)
    return low[m[0]] if m else None


def _match(cell: Any, cond: Any) -> bool:
    cs = str(cell or "").strip()
    m = _OP.match(str(cond))
    opx, val = (m.group(1), m.group(2).strip()) if m else ("=", str(cond).strip())
    if opx == "~":
        return val.lower() in cs.lower()
    a, b = _num(cs), _num(val)
    if opx in (">", "<", ">=", "<=") and a is not None and b is not None:
        return {">": a > b, "<": a < b, ">=": a >= b, "<=": a <= b}[opx]
    if opx == "!=":
        return cs.lower() != val.lower()
    if a is not None and b is not None:
        return a == b
    return cs.lower() == val.lower() or (len(val) >= 3 and val.lower() in cs.lower())


def run(args: dict, ctx) -> dict:
    tables: dict[str, list[dict]] = ctx.tables
    if not tables:
        raise ValueError("no CSV tables in the knowledge base")
    fname = args.get("file") or args.get("filename")
    if fname and fname not in tables:
        m = difflib.get_close_matches(str(fname), list(tables), n=1, cutoff=0.3)
        fname = m[0] if m else None
    if not fname:
        fname = next(iter(tables))
    rows = tables[fname]
    cols = list(rows[0].keys()) if rows else []

    filters = args.get("filters") or {}
    if not isinstance(filters, dict):
        filters = {}
    unknown = []
    hits = list(enumerate(rows, start=1))
    for col, cond in filters.items():
        rc = _resolve_col(col, cols)
        if rc is None:
            unknown.append(col)
            continue
        hits = [(i, r) for i, r in hits if _match(r.get(rc), cond)]

    query = str(args.get("query") or "").strip().lower()
    if query:
        qt = [t for t in re.findall(r"[a-z0-9\-\.]+", query) if len(t) > 1]
        scored = []
        for i, r in hits:
            blob = " ".join(str(v).lower() for v in r.values())
            s = sum(1 for t in qt if t in blob)
            if s:
                scored.append((s, i, r))
        scored.sort(key=lambda x: (-x[0], x[1]))
        hits = [(i, r) for _, i, r in scored]

    out: dict[str, Any] = {"file": fname, "n_matches": len(hits), "columns": cols}
    agg = args.get("aggregate")
    if isinstance(agg, dict) and agg.get("op"):
        opn = str(agg["op"]).lower()
        col = _resolve_col(agg.get("column", ""), cols)
        vals = [v for v in (_num(r.get(col)) for _, r in hits) if v is not None] if col else []
        if opn == "count":
            out["aggregate"] = {"op": "count", "value": len(hits)}
        elif vals:
            fn = {"sum": sum, "min": min, "max": max, "avg": lambda x: sum(x) / len(x), "mean": lambda x: sum(x) / len(x)}.get(opn)
            if fn:
                out["aggregate"] = {"op": opn, "column": col, "value": round(fn(vals), 4)}
    limit = max(1, min(int(args.get("limit") or 10), 25))
    out["rows"] = [{"row": i, **r} for i, r in hits[:limit]]
    if unknown:
        out["warning"] = f"unknown columns ignored: {unknown}; available: {cols}"
    return out

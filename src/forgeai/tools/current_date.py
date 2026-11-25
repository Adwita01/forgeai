"""Today's date. Overridable via FORGE_FIXED_DATE (YYYY-MM-DD) so eval runs are reproducible."""
from __future__ import annotations

import os
from datetime import date, datetime, timezone


def run(args: dict | None = None, ctx=None) -> dict:
    fixed = os.getenv("FORGE_FIXED_DATE")
    now = datetime.now(timezone.utc)
    d = date.fromisoformat(fixed) if fixed else now.date()
    return {"date": d.isoformat(), "weekday": d.strftime("%A"), "utc_time": now.strftime("%H:%M"), "fixed": bool(fixed)}

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parents[1]


def load_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def find_runs() -> list[Path]:
    """Every folder under the repo that contains a finished (or partial) factory run."""
    runs = {p.parent for p in REPO.rglob("evidence_manifest.json") if ".forge_cache" not in p.parts and "tests" not in p.parts}
    extra = st.session_state.get("extra_runs", [])
    return sorted(runs | {Path(x) for x in extra if Path(x, "evidence_manifest.json").exists()}, key=lambda p: str(p))


def find_agents() -> list[Path]:
    out = [p for p in REPO.rglob("agent_definition*.json") if ".forge_cache" not in p.parts and "iterations" not in p.parts]
    return sorted(out, key=lambda p: (str(p.parent), p.name))


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)

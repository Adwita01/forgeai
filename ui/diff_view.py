from __future__ import annotations

import difflib
import json

import streamlit as st


def _flat(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k in ("retrieval_config", "output_contract"):
            for k2, v2 in v.items():
                out[f"{k}.{k2}"] = v2
        else:
            out[k] = v
    return out


def render_definition_diff(a: dict, b: dict, label_a: str, label_b: str) -> None:
    fa, fb = _flat(a), _flat(b)
    changed = [k for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k) and k not in ("system_prompt", "prompt_rules", "version")]
    if changed:
        st.markdown("**Parameter changes**")
        st.table([{"field": k, label_a: json.dumps(fa.get(k), ensure_ascii=False)[:200], label_b: json.dumps(fb.get(k), ensure_ascii=False)[:200]} for k in changed])
    ra, rb = a.get("prompt_rules", []), b.get("prompt_rules", [])
    added, removed = [r for r in rb if r not in ra], [r for r in ra if r not in rb]
    if added or removed:
        st.markdown("**Prompt rules**")
        for r in added:
            st.markdown(f":green[+ {r}]")
        for r in removed:
            st.markdown(f":red[- {r}]")
    if a.get("system_prompt") != b.get("system_prompt"):
        diff = difflib.unified_diff(a.get("system_prompt", "").splitlines(), b.get("system_prompt", "").splitlines(), label_a, label_b, lineterm="")
        st.code("\n".join(diff), language="diff")
    if not changed and not added and not removed and a.get("system_prompt") == b.get("system_prompt"):
        st.caption("No differences.")

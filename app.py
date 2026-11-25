"""ForgeAI Streamlit UI:  streamlit run app.py"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT / "src", ROOT):  # works without `pip install -e .` (e.g. on a free hosting tier)
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402

from forgeai.config import get_settings  # noqa: E402
from forgeai.llm import LLM  # noqa: E402
from ui import build_view, chat_view, pipeline_view  # noqa: E402

st.set_page_config(page_title="ForgeAI", page_icon="⚒", layout="wide")


@st.cache_data(ttl=30, show_spinner=False)
def _health(base_url: str, model: str) -> dict:
    return LLM(get_settings()).health()


with st.sidebar:
    s = get_settings()
    st.title("⚒ ForgeAI")
    st.caption("The agent that builds agents")
    h = _health(s.base_url, s.model)
    if h.get("reachable") and h.get("model_available"):
        st.success(f"LLM ok: `{s.model}`")
    elif h.get("reachable"):
        st.warning(f"Endpoint up but `{s.model}` missing. Available: {h.get('models')}")
    else:
        st.error(f"LLM unreachable at {s.base_url}. Browsing past runs still works.")
    st.caption(f"judge: `{s.judge_model}` | embed: `{s.embed_model or 'TF-IDF'}`")
    st.markdown("---")
    st.markdown("**Pipeline**\n\n0. Evidence manifest\n1. Interviewer\n2. Groundability guard\n3. Agent generator\n"
                "4. Eval generator\n5. Hybrid judge\n6. Critic-optimizer (≤3)\n7. Quality report")

tab_runs, tab_chat, tab_build = st.tabs(["Runs", "Chat", "Build"])
with tab_runs:
    pipeline_view.render()
with tab_chat:
    chat_view.render()
with tab_build:
    build_view.render()

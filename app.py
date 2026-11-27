"""ForgeAI Streamlit UI:  streamlit run app.py"""
from __future__ import annotations

import html
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

# ---------------------------------------------------------------------------------------------- styling
# One stylesheet, no extra dependencies. Base colours come from .streamlit/config.toml; this layer adds
# typography, quiet borders, card-style metrics, toned-down callouts and crisp code / table / diff blocks.
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg: #0E1014; --panel: #13171D; --panel-2: #161A21; --border: #232832; --border-soft: #1C2028;
  --text: #E3E6EA; --muted: #8A919C; --faint: #5E6570;
  --accent: #8AB4F8; --ok: #6FCF97; --warn: #E2B867; --bad: #EF8A8A;
  --sans: 'Inter', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif;
  --mono: 'JetBrains Mono', ui-monospace, 'Cascadia Code', Consolas, monospace;
}

html, body, [class*="st-"], .stMarkdown, .stText, input, textarea, button { font-family: var(--sans); }
code, pre, kbd, .stCode, [data-testid="stCode"] * { font-family: var(--mono) !important; }

/* layout & chrome */
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"], footer { display: none !important; }
[data-testid="stMainBlockContainer"], .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }
hr { border-color: var(--border-soft) !important; margin: 1.1rem 0 !important; }
h1, h2, h3, h4 { letter-spacing: -0.01em; font-weight: 600; }
.stMarkdown p, .stMarkdown li { line-height: 1.6; }
.stMarkdown strong { color: var(--text); font-weight: 600; }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted) !important; }

/* sidebar */
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border-soft); }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 0.6rem; }
.fx-brand { font-size: 1.25rem; font-weight: 600; letter-spacing: -0.02em; color: var(--text); margin: 0.2rem 0 0; }
.fx-brand span { color: var(--accent); }
.fx-tag { color: var(--muted); font-size: 0.82rem; margin: 0.15rem 0 1.2rem; }
.fx-label { color: var(--faint); font-size: 0.7rem; font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase;
            margin: 1.3rem 0 0.55rem; }
.fx-status { display: flex; align-items: flex-start; gap: 0.55rem; padding: 0.6rem 0.7rem; border: 1px solid var(--border);
             border-radius: 8px; background: var(--panel-2); font-size: 0.86rem; color: var(--text); line-height: 1.45; }
.fx-dot { width: 8px; height: 8px; border-radius: 50%; margin-top: 0.38rem; flex: none; }
.fx-dot.ok { background: var(--ok); box-shadow: 0 0 0 3px rgba(111, 207, 151, 0.15); }
.fx-dot.warn { background: var(--warn); box-shadow: 0 0 0 3px rgba(226, 184, 103, 0.15); }
.fx-dot.bad { background: var(--bad); box-shadow: 0 0 0 3px rgba(239, 138, 138, 0.15); }
.fx-sub { color: var(--muted); font-size: 0.78rem; display: block; margin-top: 0.1rem; word-break: break-word; }
.fx-kv { display: grid; grid-template-columns: auto 1fr; gap: 0.35rem 0.8rem; font-size: 0.84rem; }
.fx-kv dt { color: var(--muted); }
.fx-kv dd { margin: 0; text-align: right; font-family: var(--mono); font-size: 0.78rem; color: var(--text);
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.fx-steps { list-style: none; padding: 0; margin: 0; counter-reset: step; }
.fx-steps li { counter-increment: step; display: flex; gap: 0.6rem; align-items: baseline; padding: 0.28rem 0;
               font-size: 0.85rem; color: #C9CDD3; border-bottom: 1px solid var(--border-soft); }
.fx-steps li:last-child { border-bottom: none; }
.fx-steps li::before { content: counter(step, decimal-leading-zero); font-family: var(--mono); font-size: 0.72rem; color: var(--faint);
                       min-width: 1.4rem; }

/* tabs */
[data-baseweb="tab-list"] { gap: 1.4rem; border-bottom: 1px solid var(--border-soft); }
[data-baseweb="tab"] { padding: 0.55rem 0 !important; background: transparent !important; color: var(--muted); font-weight: 500; }
[data-baseweb="tab"][aria-selected="true"] { color: var(--text); }
[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; height: 2px !important; }
[data-baseweb="tab-border"] { display: none; }

/* stage checklist (rendered by ui/pipeline_view.py) */
.fx-stages { display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.4rem 0 1.1rem; }
.fx-stage { display: inline-flex; align-items: center; gap: 0.45rem; padding: 0.3rem 0.7rem; border: 1px solid var(--border);
            border-radius: 999px; font-size: 0.8rem; color: var(--muted); background: var(--panel); white-space: nowrap; }
.fx-stage i { width: 6px; height: 6px; border-radius: 50%; background: var(--faint); display: inline-block; }
.fx-stage.done { color: var(--text); }
.fx-stage.done i { background: var(--ok); }
.fx-section { color: var(--faint); font-size: 0.7rem; font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase;
              margin: 0.6rem 0 0.4rem; }

/* metric cards */
[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--border-soft); border-radius: 10px; padding: 0.85rem 1rem; }
[data-testid="stMetricLabel"] p { color: var(--muted) !important; font-size: 0.74rem !important; font-weight: 500;
                                  letter-spacing: 0.05em; text-transform: uppercase; }
[data-testid="stMetricValue"] { font-size: 1.55rem !important; font-weight: 600; letter-spacing: -0.02em; }

/* callouts: quiet panel + thin coloured rule instead of saturated boxes */
[data-testid="stAlertContainer"], [data-testid="stAlert"] > div {
  background: var(--panel) !important; border: 1px solid var(--border-soft) !important; border-left: 2px solid var(--accent) !important;
  border-radius: 8px !important; color: var(--text) !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) { border-left-color: var(--bad) !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) { border-left-color: var(--warn) !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) { border-left-color: var(--ok) !important; }
[data-testid="stAlertContainer"] p { color: var(--text) !important; }

/* expanders, status, containers */
[data-testid="stExpander"] details { border: 1px solid var(--border-soft) !important; border-radius: 10px !important; background: var(--panel); }
[data-testid="stExpander"] summary { font-size: 0.9rem; color: var(--text); }
[data-testid="stExpander"] summary:hover { color: var(--accent); }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--border-soft) !important; border-radius: 10px !important; }

/* code, JSON, diffs */
[data-testid="stCode"] pre, .stCode pre, pre {
  background: #0B0D11 !important; border: 1px solid var(--border-soft) !important; border-radius: 8px !important;
  font-size: 0.8rem !important; line-height: 1.55 !important; }
[data-testid="stJson"] { background: #0B0D11; border: 1px solid var(--border-soft); border-radius: 8px; padding: 0.5rem 0.7rem;
                         font-family: var(--mono); font-size: 0.8rem; }
.stMarkdown code { background: #1A1F27; color: #C8D3E6; border: 1px solid var(--border-soft); border-radius: 4px;
                   padding: 0.08rem 0.32rem; font-size: 0.8em; }

/* tables & dataframes */
[data-testid="stTable"] table, .stMarkdown table { border-collapse: collapse; width: 100%; font-size: 0.84rem;
                                                  border: 1px solid var(--border-soft); border-radius: 8px; overflow: hidden; }
[data-testid="stTable"] th, .stMarkdown th { background: var(--panel-2); color: var(--muted); font-weight: 500; text-align: left;
                                             font-size: 0.74rem; letter-spacing: 0.04em; text-transform: uppercase;
                                             border-bottom: 1px solid var(--border) !important; padding: 0.5rem 0.7rem !important; }
[data-testid="stTable"] td, .stMarkdown td { border-bottom: 1px solid var(--border-soft) !important; padding: 0.45rem 0.7rem !important;
                                             vertical-align: top; color: #D3D7DD; }
[data-testid="stTable"] tr:hover td, .stMarkdown tr:hover td { background: rgba(138, 180, 248, 0.04); }
[data-testid="stDataFrame"] { border: 1px solid var(--border-soft); border-radius: 8px; overflow: hidden; }
.stMarkdown blockquote { border-left: 2px solid var(--border); color: #C9CDD3; padding-left: 0.9rem; margin-left: 0; }

/* inputs, buttons, chat */
.stButton > button, .stDownloadButton > button { border-radius: 8px; border: 1px solid var(--border); font-weight: 500; }
.stButton > button:hover { border-color: var(--accent); color: var(--accent); }
[data-testid="stChatMessage"] { background: var(--panel); border: 1px solid var(--border-soft); border-radius: 10px; padding: 0.8rem 1rem; }
[data-baseweb="select"] > div, [data-baseweb="input"], textarea { border-radius: 8px !important; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=30, show_spinner=False)
def _health(base_url: str, model: str) -> dict:
    return LLM(get_settings()).health()


def _e(x) -> str:
    return html.escape(str(x))


with st.sidebar:
    s = get_settings()
    st.markdown('<p class="fx-brand">⚒ Forge<span>AI</span></p><p class="fx-tag">The agent that builds agents</p>',
                unsafe_allow_html=True)

    h = _health(s.base_url, s.model)
    if h.get("reachable") and h.get("model_available"):
        dot, title, sub = "ok", "Model online", f"{s.model} · ready"
    elif h.get("reachable"):
        dot, title, sub = "warn", f"Endpoint up, {s.model} missing", f"Available: {', '.join(h.get('models') or []) or 'none'}"
    else:
        dot, title, sub = "bad", "LLM unreachable", f"{s.base_url} · browsing past runs still works"
    st.markdown('<div class="fx-label">Status</div>'
                f'<div class="fx-status"><span class="fx-dot {dot}"></span><div>{_e(title)}<span class="fx-sub">{_e(sub)}</span></div></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="fx-label">Models</div><dl class="fx-kv">'
                f'<dt>Agent</dt><dd title="{_e(s.model)}">{_e(s.model)}</dd>'
                f'<dt>Judge</dt><dd title="{_e(s.judge_model)}">{_e(s.judge_model)}</dd>'
                f'<dt>Embeddings</dt><dd title="{_e(s.embed_model or "TF-IDF")}">{_e(s.embed_model or "TF-IDF")}</dd>'
                '</dl>', unsafe_allow_html=True)

    steps = ["Evidence manifest", "Interviewer", "Groundability guard", "Agent generator", "Eval generator",
             "Hybrid judge", "Critic-optimizer (≤3)", "Quality report"]
    st.markdown('<div class="fx-label">Pipeline</div><ol class="fx-steps" start="0" style="counter-reset: step -1;">'
                + "".join(f"<li>{_e(x)}</li>" for x in steps) + "</ol>", unsafe_allow_html=True)

tab_runs, tab_chat, tab_build = st.tabs(["Runs", "Chat", "Build"])
with tab_runs:
    pipeline_view.render()
with tab_chat:
    chat_view.render()
with tab_build:
    build_view.render()

"""Chat with any generated agent definition."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from forgeai.agent_runtime import Agent
from forgeai.evaluation import load_definition, resolve_docs
from forgeai.llm import LLM
from forgeai.parsing import load_folder

from .common import find_agents, rel


@st.cache_resource(show_spinner="Loading agent and indexing documents...")
def _agent(path: str, mtime: float) -> Agent:
    defn, base = load_definition(path)
    docs, _ = load_folder(resolve_docs(defn, base))
    return Agent(defn, LLM(), docs=docs)


def render() -> None:
    agents = find_agents()
    custom = st.text_input("...or path to any agent_definition.json", "")
    options = ([Path(custom)] if custom and Path(custom).exists() else []) + agents
    if not options:
        st.info("No agent definitions found yet.")
        return
    path = st.selectbox("Agent definition", options, format_func=rel)
    agent = _agent(str(path), Path(path).stat().st_mtime)
    d = agent.d
    st.caption(f"**{d.name}** v{d.version} | tools: {d.selected_tools or 'none'} | {len(agent.retriever.chunks)} chunks | "
               f"dense: {agent.retriever.dense_backend} | top_k {d.retrieval_config.top_k}, alpha {d.retrieval_config.hybrid_alpha}")

    key = f"chat::{path}"
    history = st.session_state.setdefault(key, [])
    if st.button("Clear chat"):
        history.clear()
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                _meta(msg["meta"])
    q = st.chat_input("Ask the agent...")
    if q:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("retrieving, calling tools, answering..."):
                r = agent.answer(q, [{"role": m["role"], "content": m["content"]} for m in history])
            text = ("🚫 " if r.refused else "") + r.answer
            st.markdown(text)
            meta = {"citations": r.citations, "latency": r.latency_s,
                    "retrieved": [(c.filename, c.locator, c.score, c.text[:400]) for c in r.retrieved],
                    "tools": [(t.tool, t.args, str(t.output)[:600] if t.error is None else "ERROR " + t.error) for t in r.tool_results]}
            _meta(meta)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": text, "meta": meta}]


def _meta(meta: dict) -> None:
    if meta.get("citations"):
        st.caption("citations: " + "  ".join(meta["citations"]))
    with st.expander(f"sources & tools ({meta.get('latency')}s)"):
        for f, loc, score, text in meta.get("retrieved", []):
            st.markdown(f"`{score:.2f}` **[{f} {loc}]** {text}")
        for tool, args, out in meta.get("tools", []):
            st.markdown(f"🔧 `{tool}({args})` -> {out}")

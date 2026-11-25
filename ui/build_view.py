"""Launch a factory run from the browser. The run executes in a background thread; logs stream live."""
from __future__ import annotations

import tempfile
import threading
import traceback
from pathlib import Path

import streamlit as st

from forgeai.config import get_settings
from forgeai.llm import LLM
from forgeai.manifest import build_manifest
from forgeai.stages import interviewer

from .common import REPO


class _Job:
    def __init__(self):
        self.logs: list[str] = []
        self.done = False
        self.error: str | None = None
        self.out: Path | None = None


def _run(job: _Job, kw: dict) -> None:
    from forgeai.pipeline import BuildOptions, build

    try:
        build(BuildOptions(**kw, log=job.logs.append))
    except BaseException as e:  # noqa: BLE001 - surface everything in the UI
        job.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
    finally:
        job.done = True


def render() -> None:
    s = get_settings()
    st.markdown("Write a one-paragraph brief, point at a documents folder (or upload files), answer the interviewer, build.")
    brief = st.text_area("Brief", height=110, placeholder="I need an assistant that answers HR policy questions for shore staff...")
    c1, c2 = st.columns(2)
    docs_dir = c1.text_input("Documents folder (path on this machine)", "")
    uploads = c2.file_uploader("...or upload documents", type=["pdf", "csv", "txt", "md"], accept_multiple_files=True)
    out_dir = st.text_input("Output folder", str(REPO / "forge_out" / "ui_run"))
    c3, c4, c5 = st.columns(3)
    max_iters = c3.slider("Max optimizer iterations", 0, 3, 3)
    max_tokens = c4.number_input("Token budget (0 = none)", 0, 5_000_000, int(s.max_tokens), step=10_000)
    max_seconds = c5.number_input("Time budget, s (0 = none)", 0, 50_000, int(s.max_seconds), step=300)

    if uploads:
        tmp = Path(st.session_state.setdefault("upload_dir", tempfile.mkdtemp(prefix="forge_docs_")))
        for f in uploads:
            (tmp / f.name).write_bytes(f.getbuffer())
        docs_dir = str(tmp)
        st.caption(f"{len(uploads)} file(s) staged in {tmp}")

    # ---- step 1: interviewer (human-in-the-loop in the UI)
    if st.button("1 - Interview me", disabled=not (brief and docs_dir)):
        with st.spinner("Building evidence manifest and drafting clarifying questions..."):
            m, _ = build_manifest(Path(docs_dir))
            qs, src = interviewer.ask_questions(LLM(s), brief, m)
        st.session_state["questions"] = [q.model_dump() for q in qs]
        st.session_state["manifest_warn"] = m.warnings
    qs = st.session_state.get("questions")
    answers: list[str] = []
    if qs:
        if st.session_state.get("manifest_warn"):
            st.warning("; ".join(st.session_state["manifest_warn"]))
        st.markdown("**Clarifying questions** - edit the defaults if you like:")
        for i, q in enumerate(qs):
            answers.append(st.text_input(f"Q{i + 1}. {q['question']}", q["default_answer"], help=q.get("why"), key=f"ans{i}"))

    # ---- step 2: build
    job: _Job | None = st.session_state.get("job")
    running = job is not None and not job.done
    if st.button("2 - Build agent", type="primary", disabled=not (brief and docs_dir) or running):
        job = _Job()
        job.out = Path(out_dir)
        st.session_state["job"] = job
        st.session_state.setdefault("extra_runs", []).append(out_dir)
        kw = dict(brief=brief, docs=Path(docs_dir), out=Path(out_dir), mode="auto", answers=answers or None, max_iters=max_iters,
                  settings=s.with_overrides(max_tokens=int(max_tokens), max_seconds=float(max_seconds)))
        threading.Thread(target=_run, args=(job, kw), daemon=True).start()
        st.rerun()

    if job is not None:
        _progress(job)


@st.fragment(run_every=2)
def _progress(job: _Job) -> None:
    state = "running" if not job.done else ("error" if job.error else "complete")
    with st.status(f"Factory run: {state}", state=state, expanded=True):
        st.code("\n".join(job.logs[-40:]) or "starting...", language="text")
        if job.error:
            st.error(job.error)
        elif job.done:
            st.success(f"Done - inspect it in the **Runs** tab or chat with it in **Chat**. Output: {job.out}")

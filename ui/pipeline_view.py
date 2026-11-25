"""Inspect a finished factory run: every stage's evidence, iterations, diffs and failures."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from .common import find_runs, load_json, rel
from .diff_view import render_definition_diff

STAGES = [
    ("0 Manifest", "evidence_manifest.json"),
    ("1 Interviewer", "clarifications.json"),
    ("2 Groundability", "groundability.json"),
    ("3 Generator", "agent_definition.v0.json"),
    ("4 Eval set", "eval_set.json"),
    ("5-6 Judge + optimizer", "iteration_log.json"),
    ("7 Report", "quality_report.md"),
]


def render() -> None:
    runs = find_runs()
    if not runs:
        st.info("No runs yet. Build an agent in the **Build** tab or run `forge build`.")
        return
    run = st.selectbox("Factory run", runs, format_func=rel)
    run = Path(run)

    cols = st.columns(len(STAGES))
    for c, (name, f) in zip(cols, STAGES):
        ok = (run / f).exists()
        c.markdown(f"{'✅' if ok else '⏳'} **{name}**")

    sc = load_json(run / "scorecard.json", {})
    final = sc.get("final")
    if final:
        m = st.columns(6)
        m[0].metric("Pass rate", f"{final['pass_rate']:.0%}")
        m[1].metric("Overall", f"{final['overall']:.3f}")
        m[2].metric("Groundedness", f"{final['groundedness']:.2f}")
        m[3].metric("Factuality", f"{final['factuality']:.2f}")
        m[4].metric("Refusal acc.", f"{final['refusal_accuracy']:.2f}")
        m[5].metric("Cases", final["n_cases"])

    tabs = st.tabs(["Requirements", "Agent", "Eval set", "Iterations & diffs", "Failures", "Report", "Manifest", "Budget"])
    with tabs[0]:
        req = load_json(run / "requirements.json", {})
        cl = load_json(run / "clarifications.json", {})
        if (run / "brief.txt").exists():
            st.markdown("**Brief**")
            st.info((run / "brief.txt").read_text(encoding="utf-8"))
        if cl:
            st.markdown("**Interview**")
            for a in cl.get("answers", []):
                st.markdown(f"- **Q:** {a['question']}  \n  **A:** {a['answer']}  _({a['source']})_")
        if req:
            gaps = req.get("ungrounded_requirements", [])
            if gaps:
                st.error("Ungrounded requirements (agent will refuse): " + "; ".join(f"**{g['requirement']}** - {g['reason']}" for g in gaps))
            c1, c2 = st.columns(2)
            c1.markdown("**Must answer**\n" + "\n".join(f"- {t}" for t in req.get("must_answer_topics", [])))
            c2.markdown("**Must refuse**\n" + "\n".join(f"- {t}" for t in req.get("must_refuse_topics", [])))
            st.markdown("**Capabilities -> tools evidence**")
            st.table([{"capability": k, "enabled": req["capabilities"].get(k), "evidence": v} for k, v in req.get("capability_evidence", {}).items()])
        g = load_json(run / "groundability.json", {})
        if g:
            st.markdown(f"**Groundability verdicts** (dense backend: `{g.get('dense_backend')}`)")
            st.dataframe(pd.DataFrame(g.get("verdicts", [])), width="stretch", hide_index=True)
    with tabs[1]:
        agent = load_json(run / "agent_definition.json") or load_json(run / "agent_definition.v0.json", {})
        if agent:
            st.markdown(f"**{agent['name']}** v{agent['version']} - tools: `{agent['selected_tools'] or 'none'}`")
            st.table([{"tool": k, "decision": v} for k, v in agent.get("tool_rationale", {}).items()])
            st.json(agent["retrieval_config"], expanded=False)
            st.json(agent["output_contract"], expanded=False)
            st.markdown("**System prompt**")
            st.code(agent["system_prompt"] + ("\n\nADDITIONAL RULES:\n" + "\n".join("- " + r for r in agent["prompt_rules"]) if agent.get("prompt_rules") else ""), language="markdown")
    with tabs[2]:
        es = load_json(run / "eval_set.json", {})
        if es:
            df = pd.DataFrame(es["cases"])
            cats = st.multiselect("Category", sorted(df["category"].unique()), default=sorted(df["category"].unique()))
            st.dataframe(df[df["category"].isin(cats)][["id", "category", "question", "expected_behavior", "expected_sources", "key_facts", "rationale"]],
                         width="stretch", hide_index=True)
            if es.get("notes"):
                with st.expander(f"Generator notes ({len(es['notes'])})"):
                    for n in es["notes"]:
                        st.caption(n)
    with tabs[3]:
        hist = load_json(run / "iteration_log.json", [])
        per = sc.get("per_iteration", [])
        if per:
            chart = pd.DataFrame([{"iteration": p["iteration"], "overall": p["overall"], "pass_rate": p["pass_rate"],
                                   "groundedness": p["groundedness"], "factuality": p["factuality"]} for p in per]).set_index("iteration")
            st.line_chart(chart)
        for h in hist:
            with st.expander(f"Iteration {h['iteration']} - {h['decision']}  "
                             f"({h['score_before']:.3f} -> {h['score_after'] if h['score_after'] is None else round(h['score_after'], 3)})",
                             expanded=h["iteration"] > 0):
                st.write(f"Failures: {h['failures_by_type']}  |  tokens {h['tokens_used']}  |  {h['seconds_used']}s")
                if h.get("stop_reason"):
                    st.caption(h["stop_reason"])
                for p in h.get("patches", []):
                    st.markdown(f"- **{p['target']}** ({p['kind']}) `{p['field']}` - {p['rationale']}  \n  targets: {', '.join(p['addresses_cases'][:10])}")
                if h["iteration"] > 0:
                    a = load_json(run / "iterations" / f"agent_definition.iter{h['iteration'] - 1}.json") or {}
                    # compare against the definition that was actually patched (last kept one)
                    kept = [x for x in hist[: h["iteration"]] if x["decision"] in ("baseline", "kept")]
                    if kept:
                        a = load_json(run / "iterations" / f"agent_definition.iter{kept[-1]['iteration']}.json") or a
                    b = load_json(run / "iterations" / f"agent_definition.iter{h['iteration']}.json")
                    if a and b:
                        render_definition_diff(a, b, f"v{a['version']}", f"v{b['version']}")
    with tabs[4]:
        res = load_json(run / "final_results.json", [])
        if res:
            df = pd.DataFrame(res)
            only_fail = st.toggle("Only failures", value=True)
            if only_fail:
                df = df[~df["passed"]]
            st.dataframe(df[["case_id", "category", "passed", "failure_category", "failure_reason", "question", "answer", "groundedness",
                             "factuality", "citations", "invalid_citations", "retrieved_files"]], width="stretch", hide_index=True)
    with tabs[5]:
        if (run / "quality_report.md").exists():
            st.markdown((run / "quality_report.md").read_text(encoding="utf-8"))
    with tabs[6]:
        st.json(load_json(run / "evidence_manifest.json", {}), expanded=1)
    with tabs[7]:
        b = load_json(run / "budget.json", {})
        if b:
            st.write(f"Total tokens: **{b['tokens_used']}**, wall clock **{b.get('wall_clock_s', b['elapsed_s'])}s**")
            st.dataframe(pd.DataFrame(b["stages"]).T, width="stretch")

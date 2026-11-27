"""Fill the README results section from the committed run artifacts (deterministic, no LLM).

Usage:  python scripts/update_readme.py
Reads demos/<demo>/run/{scorecard,requirements,agent_definition,budget,eval_set}.json and
demos/demo1_hr/baseline_comparison/*_scorecard.json, then rewrites the block between
<!-- RESULTS:START --> and <!-- RESULTS:END --> in README.md. Missing artifacts are reported
as "not run yet" instead of being invented.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DEMOS = [("Demo 1 - Maritime HR & Shore-Leave", ROOT / "demos/demo1_hr/run"),
         ("Demo 2 - Vessel Spares & Procurement", ROOT / "demos/demo2_spares/run")]
CMP = ROOT / "demos/demo1_hr/baseline_comparison"


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def demo_block(title: str, run: Path) -> list[str]:
    sc = _load(run / "scorecard.json")
    if not sc:
        return [f"### {title}", "", f"_Not run yet: `{_rel(run)}` has no scorecard.json._", ""]
    req, agent = _load(run / "requirements.json") or {}, _load(run / "agent_definition.json") or {}
    budget, es = _load(run / "budget.json") or {}, _load(run / "eval_set.json") or {"cases": []}
    final, per = sc["final"], sc["per_iteration"]
    cats: dict[str, int] = {}
    for c in es["cases"]:
        cats[c["category"]] = cats.get(c["category"], 0) + 1
    rc = agent.get("retrieval_config", {})
    L = [f"### {title}", "",
         f"- **Agent:** {agent.get('name') or req.get('agent_name', '?')} (v{agent.get('version', '?')})",
         f"- **Tools attached:** {', '.join('`%s`' % t for t in agent.get('selected_tools', [])) or 'none'}",
         f"- **Retrieval:** chunk {rc.get('chunk_size')} / overlap {rc.get('chunk_overlap')}, top_k {rc.get('top_k')}, alpha {rc.get('hybrid_alpha')}",
         f"- **Ungrounded requirements caught:** {', '.join(g['requirement'] for g in req.get('ungrounded_requirements', [])) or 'none'}",
         f"- **Eval set:** {len(es['cases'])} cases ({', '.join(f'{k} {v}' for k, v in sorted(cats.items()))})",
         f"- **Cost:** {budget.get('tokens_used', '?')} tokens, {budget.get('wall_clock_s', '?')} s wall clock",
         "", "| Iter | Agent v | Pass rate | Groundedness | Factuality | Refusal acc. | Citation validity | Overall |",
         "|---|---|---|---|---|---|---|---|"]
    for s in per:
        L.append(f"| {s['iteration']} | {s['agent_version']} | {s['pass_rate']:.0%} | {s['groundedness']:.2f} | {s['factuality']:.2f} "
                 f"| {s['refusal_accuracy']:.2f} | {s['citation_validity']:.2f} | {s['overall']:.3f} |")
    L += ["", f"Final (best kept version v{final['agent_version']}): **{final['pass_rate']:.0%} pass, overall {final['overall']:.3f}**. "
          f"Full honest write-up: [`{_rel(run / 'quality_report.md')}`]({_rel(run / 'quality_report.md')}).", ""]
    return L


def comparison_block() -> list[str]:
    a, b = _load(CMP / "factory_scorecard.json"), _load(CMP / "baseline_scorecard.json")
    if not (a and b):
        return ["### Factory vs hand-crafted baseline (Demo 1)", "", f"_Not run yet: `{_rel(CMP)}` is missing._", ""]
    L = ["### Factory vs hand-crafted baseline (Demo 1)", "",
         "Same eval set, same model, same judge. Baseline: [`demos/demo1_hr/baseline/agent_definition.json`](demos/demo1_hr/baseline/agent_definition.json).", "",
         "| Metric | Factory | Hand-crafted | Δ |", "|---|---|---|---|"]
    for k in ("pass_rate", "overall", "groundedness", "factuality", "refusal_accuracy", "citation_validity"):
        L.append(f"| {k} | {a[k]:.3f} | {b[k]:.3f} | {a[k] - b[k]:+.3f} |")
    L += ["", f"Case-by-case breakdown: [`{_rel(CMP / 'comparison_report.md')}`]({_rel(CMP / 'comparison_report.md')}).", ""]
    return L


def main() -> int:
    text = README.read_text(encoding="utf-8")
    body = []
    for title, run in DEMOS:
        body += demo_block(title, run)
    body += comparison_block()
    new = "<!-- RESULTS:START -->\n" + "\n".join(body) + "<!-- RESULTS:END -->"
    out, n = re.subn(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", lambda _: new, text, flags=re.S)
    if not n:
        print("README.md has no RESULTS markers", file=sys.stderr)
        return 1
    README.write_text(out, encoding="utf-8")
    print("README.md results section updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

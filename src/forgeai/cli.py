"""`forge` command line: build | run | eval | compare | doctor | manifest."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

for _s in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

from .config import get_settings  # noqa: E402
from .llm import LLM, BackendUnavailable  # noqa: E402

app = typer.Typer(add_completion=False, no_args_is_help=True, help="ForgeAI - the agent that builds agents.")
con = Console()


def _settings(model: Optional[str], judge_model: Optional[str], max_tokens: Optional[int], max_seconds: Optional[float]):
    return get_settings().with_overrides(model=model, judge_model=judge_model or model, max_tokens=max_tokens, max_seconds=max_seconds)


def _read_brief(brief: str) -> str:
    p = Path(brief)
    if len(brief) < 260 and p.suffix.lower() in (".txt", ".md") and p.exists():
        return p.read_text(encoding="utf-8")
    return brief


def _check_backend(llm: LLM, strict: bool = False) -> None:
    h = llm.health()
    if not h.get("reachable"):
        con.print(f"[red]LLM endpoint {llm.s.base_url} unreachable:[/] {h.get('error')}")
        con.print("Start Ollama (open the app or run `ollama serve`) or set FORGE_BASE_URL.")
        if strict:
            raise typer.Exit(2)
    elif not h.get("model_available"):
        con.print(f"[yellow]Model {llm.s.model!r} not found on the server. Available: {h.get('models')}. Try: ollama pull {llm.s.model}[/]")
        if strict:
            raise typer.Exit(2)


def _scorecard_table(title: str, rows: list) -> Table:
    t = Table(title=title)
    for c in ("iter", "decision", "pass", "grounded", "factual", "refusal", "citations", "overall", "failures"):
        t.add_column(c)
    for decision, sc in rows:
        t.add_row(str(sc.iteration), decision, f"{sc.pass_rate:.0%}", f"{sc.groundedness:.2f}", f"{sc.factuality:.2f}",
                  f"{sc.refusal_accuracy:.2f}", f"{sc.citation_validity:.2f}", f"{sc.overall:.3f}",
                  ", ".join(f"{k}:{v}" for k, v in sc.failures_by_type.items()) or "-")
    return t


@app.command()
def build(
    brief: str = typer.Option(..., "--brief", "-b", help="Brief text, or path to a .txt/.md file containing it"),
    docs: Path = typer.Option(..., "--docs", "-d", exists=True, file_okay=False, help="Folder of PDFs / CSVs / TXT / MD"),
    out: Path = typer.Option(Path("forge_out"), "--out", "-o", help="Output folder for all artifacts"),
    answers: Optional[Path] = typer.Option(None, "--answers", help="JSON answers for the interviewer (list, or {keyword: answer})"),
    non_interactive: bool = typer.Option(False, "--non-interactive", "--auto-answer", "-y", help="Unattended: answer questions from --answers or defaults"),
    approve: bool = typer.Option(False, "--approve", help="Human checkpoint: pause to review/edit requirements.json (interactive only)"),
    max_iters: int = typer.Option(3, "--max-iters", min=0, max=3),
    max_cases: Optional[int] = typer.Option(None, "--max-cases", min=4, help="Quick verification: cap the generated eval set (default: >=22 cases)"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    judge_model: Optional[str] = typer.Option(None, "--judge-model"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Token budget for the whole run (0 = unlimited)"),
    max_seconds: Optional[float] = typer.Option(None, "--max-seconds", help="Wall-clock budget in seconds (0 = unlimited)"),
    reuse_eval_set: Optional[Path] = typer.Option(None, "--reuse-eval-set", help="Skip generation, use this eval_set.json (regression mode)"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable the LLM response cache"),
):
    """Run the full factory: interview -> guard -> generate -> eval-gen -> judge -> optimise -> report."""
    from .pipeline import BuildOptions, build as run_build
    from .stages.interviewer import load_answers

    s = _settings(model, judge_model, max_tokens, max_seconds)
    _check_backend(LLM(s), strict=True)  # a build without a model would only produce a junk agent
    interactive = not non_interactive and sys.stdin.isatty()
    text = _read_brief(brief)
    con.print(Panel(text.strip(), title="Brief", subtitle=f"docs: {docs}  mode: {'interactive' if interactive else 'unattended'}"))
    try:
        res = run_build(BuildOptions(
            brief=text, docs=docs, out=out, mode="interactive" if interactive else "auto", answers=load_answers(answers),
            approve=approve, max_iters=max_iters, max_cases=max_cases, reuse_eval_set=reuse_eval_set, use_cache=not no_cache, settings=s,
            log=lambda m: con.print(m, markup=False, highlight=False),
        ))
    except BackendUnavailable as e:
        con.print(f"[bold red]Build aborted:[/] {escape(str(e))}")
        con.print(f"Partial artifacts and run_status.json are in {out}. Fix the model server, then rerun the same command: "
                  "completed LLM calls replay from .forge_cache.")
        raise typer.Exit(2)
    status = json.loads((out / "run_status.json").read_text(encoding="utf-8")).get("status", "complete")
    if status != "complete":
        con.print("[yellow]Run cut short by its budget - quality_report.md says what was completed.[/]")
    rows = []
    for h in res.history:
        sc = next((x for x in res.scorecards if x.iteration == h.iteration), None)
        if sc:
            rows.append((h.decision, sc))
    con.print(_scorecard_table("Per-iteration scorecard", rows))
    con.print(f"Tools attached: {res.agent.selected_tools or 'none'}")
    if res.requirements.ungrounded_requirements:
        con.print("[yellow]Ungrounded requirements (agent will refuse):[/] " + ", ".join(g.requirement for g in res.requirements.ungrounded_requirements))
    con.print(f"\n[bold green]Done.[/] Agent: {out / 'agent_definition.json'}   Report: {out / 'quality_report.md'}")
    con.print(f"Chat with it:  forge run {out / 'agent_definition.json'}")


@app.command()
def run(
    agent_def: Path = typer.Argument(..., exists=True, help="agent_definition.json (or the build output folder)"),
    question: Optional[str] = typer.Option(None, "--question", "-q", help="Ask one question and exit"),
    show_context: bool = typer.Option(False, "--show-context", help="Print retrieved passages and tool calls"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
):
    """Generic chat runner: loads ANY produced agent definition."""
    from .agent_runtime import Agent
    from .evaluation import load_definition, resolve_docs

    s = _settings(model, None, 0, 0)
    llm = LLM(s)
    _check_backend(llm, strict=True)
    defn, base = load_definition(agent_def)
    if model:
        defn.model = model
    from .parsing import load_folder
    docs, _ = load_folder(resolve_docs(defn, base))
    agent = Agent(defn, llm, docs=docs)
    con.print(Panel(f"{defn.name} (v{defn.version})\n{defn.description}\ntools: {defn.selected_tools or 'none'} | "
                    f"{len(agent.retriever.chunks)} chunks | dense: {agent.retriever.dense_backend}",
                    title="ForgeAI agent", subtitle="type /quit to exit, /context to toggle sources"))
    if agent.model_note:
        con.print(f"[yellow]{escape(agent.model_note)}[/]")

    def ask(q: str, history: list[dict]) -> None:
        with con.status("thinking..."):
            r = agent.answer(q, history)
        style = "yellow" if r.refused else "green"
        con.print(Panel(escape(r.answer), title="refused" if r.refused else "answer", border_style=style))
        if r.citations:
            con.print("[dim]citations: " + escape("  ".join(r.citations)) + "[/]")
        if show_ctx[0]:
            for c in r.retrieved:
                con.print(f"[dim]  {c.score:.2f} " + escape(f"[{c.filename} {c.locator}] {c.text[:140]}...") + "[/]", highlight=False)
            for t in r.tool_results:
                con.print("[dim]  " + escape(f"tool {t.tool}({json.dumps(t.args)}) -> {str(t.output)[:200] if t.error is None else 'ERROR ' + t.error}") + "[/]")
        con.print(f"[dim]{r.latency_s}s[/]")
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": r.answer}]

    show_ctx = [show_context]
    history: list[dict] = []
    if question:
        ask(question, history)
        return
    while True:
        try:
            q = con.input("[bold cyan]you> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("/quit", "/exit", "quit", "exit"):
            break
        if q == "/context":
            show_ctx[0] = not show_ctx[0]
            con.print(f"show context: {show_ctx[0]}")
            continue
        ask(q, history)


@app.command("eval")
def eval_cmd(
    agent_def: Path = typer.Argument(..., exists=True),
    eval_set: Path = typer.Option(..., "--eval-set", "-e", exists=True),
    out: Path = typer.Option(Path("forge_eval"), "--out", "-o"),
    docs: Optional[Path] = typer.Option(None, "--docs", help="Override documents folder (regression: changed docs)"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    judge_model: Optional[str] = typer.Option(None, "--judge-model"),
):
    """Score an agent definition against an eval set (no optimisation)."""
    from .evaluation import evaluate_definition, save_results

    llm = LLM(_settings(model, judge_model, 0, 0))
    _check_backend(llm, strict=True)
    _, results, sc = evaluate_definition(agent_def, eval_set, llm, docs_override=docs,
                                         progress=lambda i, n, r: con.print(f"  {i:>2}/{n} {r.case_id:<5} {'PASS' if r.passed else 'FAIL ' + (r.failure_category.value if r.failure_category else '')}"))
    save_results(out, results, sc)
    con.print(_scorecard_table(f"Eval of {agent_def}", [("eval", sc)]))
    con.print(f"results -> {out}")


@app.command()
def compare(
    agent_a: Path = typer.Argument(..., exists=True),
    agent_b: Path = typer.Argument(..., exists=True),
    eval_set: Path = typer.Option(..., "--eval-set", "-e", exists=True),
    out: Path = typer.Option(Path("comparison"), "--out", "-o"),
    name_a: str = typer.Option("factory", "--name-a"),
    name_b: str = typer.Option("baseline", "--name-b"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
):
    """Side-by-side evaluation of two agents on the same eval set (e.g. factory vs hand-crafted)."""
    from .evaluation import compare_markdown, evaluate_definition, save_results

    llm = LLM(_settings(model, None, 0, 0))
    _check_backend(llm, strict=True)
    results = {}
    for name, path in ((name_a, agent_a), (name_b, agent_b)):
        con.print(f"[bold]evaluating {name}[/] ({path})")
        _, res, sc = evaluate_definition(path, eval_set, llm, progress=lambda i, n, r: con.print(f"  {i:>2}/{n} {r.case_id:<5} {'PASS' if r.passed else 'FAIL'}"))
        save_results(out, res, sc, stem=name)
        results[name] = (res, sc)
    md = compare_markdown(name_a, results[name_a][1], results[name_a][0], name_b, results[name_b][1], results[name_b][0], str(eval_set))
    (out / "comparison_report.md").write_text(md, encoding="utf-8")
    con.print(_scorecard_table("comparison", [(name_a, results[name_a][1]), (name_b, results[name_b][1])]))
    con.print(f"report -> {out / 'comparison_report.md'}")


@app.command()
def doctor():
    """Check the LLM backend, models and embeddings."""
    s = get_settings()
    llm = LLM(s)
    con.print(f"base_url={s.base_url} backend={llm.backend} model={s.model} judge={s.judge_model} embed={s.embed_model}")
    con.print(llm.health())
    v = llm.embed(["ping"])
    con.print(f"embeddings: {'ok, dim ' + str(len(v[0])) if v else 'unavailable -> TF-IDF fallback'}")


@app.command()
def manifest(docs: Path = typer.Argument(..., exists=True, file_okay=False), out: Optional[Path] = typer.Option(None, "--out", "-o")):
    """Build and print the deterministic Evidence Manifest for a folder (no LLM)."""
    from .manifest import build_manifest, manifest_for_prompt, save_manifest

    m, _ = build_manifest(docs)
    con.print(manifest_for_prompt(m, 10_000), markup=False, highlight=False)
    if m.warnings:
        con.print(f"[yellow]warnings: {m.warnings}[/]")
    if out:
        save_manifest(m, out)


if __name__ == "__main__":
    app()

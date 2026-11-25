"""The 4-tool registry. The generator may attach only tools whose capability the requirements need."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from . import calculator, csv_lookup, current_date, open_meteo_weather


@dataclass(frozen=True)
class ToolSpec:
    name: str
    capability: str  # field on schemas.Capabilities that justifies attaching it
    description: str
    args_doc: str
    run: Callable[[dict, "ToolContext"], Any]
    requires_csv: bool = False


@dataclass
class ToolContext:
    tables: dict[str, list[dict]]  # filename -> rows (CSV docs only)


REGISTRY: dict[str, ToolSpec] = {
    "csv_lookup": ToolSpec(
        "csv_lookup", "tabular_lookup",
        "Filter / search rows of a CSV table in the knowledge base and optionally aggregate a column. Returns matching rows with their row numbers for citation.",
        '{"file": "<csv filename, optional if only one>", "filters": {"<column>": "<value | >5 | <=10 | ~substring>"}, '
        '"query": "<free text matched against all cells, optional>", "aggregate": {"op": "sum|min|max|avg|count", "column": "<column>"}, "limit": 10}',
        csv_lookup.run, requires_csv=True,
    ),
    "calculator": ToolSpec(
        "calculator", "arithmetic",
        "Evaluate an arithmetic expression exactly (+ - * / ** %, parentheses, round/min/max/abs/sum). Use for totals, costs, conversions.",
        '{"expression": "<e.g. 12 * 45.5 + 30>"}',
        calculator.run,
    ),
    "open_meteo_weather": ToolSpec(
        "open_meteo_weather", "live_weather",
        "Current weather and short forecast for a place name or lat/lon, from the keyless open-meteo API.",
        '{"location": "<city or port name>"} or {"latitude": 0.0, "longitude": 0.0}',
        open_meteo_weather.run,
    ),
    "current_date": ToolSpec(
        "current_date", "date_awareness",
        "Today's date, weekday and UTC time. Use for questions about deadlines, notice periods, ages or 'how many days until'.",
        "{}",
        current_date.run,
    ),
}


def run_tool(name: str, args: dict, ctx: ToolContext) -> tuple[Any, str | None]:
    spec = REGISTRY.get(name)
    if spec is None:
        return None, f"unknown tool {name!r}"
    try:
        return spec.run(args or {}, ctx), None
    except Exception as e:  # noqa: BLE001 - tool errors are returned to the agent, never raised
        return None, f"{type(e).__name__}: {e}"


def describe(names: list[str]) -> str:
    return "\n".join(f"- {REGISTRY[n].name}: {REGISTRY[n].description}\n  args: {REGISTRY[n].args_doc}" for n in names if n in REGISTRY)

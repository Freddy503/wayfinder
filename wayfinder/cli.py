"""Command line entry point."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console

from wayfinder.agent import AgentConfig, plan_trip
from wayfinder.specs import load_itinerary_payload, load_spec
from wayfinder.verify import ConstraintReport, check_payload

app = typer.Typer(
    add_completion=False,
    help="Plan trips with a deep agent, and check the plans it produces.",
    no_args_is_help=True,
)
console = Console()


def _load_env() -> None:
    load_dotenv(Path.cwd() / ".env")


def _preflight(model: str, *, needs_langsmith: bool = False) -> None:
    """Fail loudly on a missing key, before burning a run on it.

    The provider's own error ("Could not resolve authentication method…")
    arrives several layers deep and doesn't say which file to edit.
    """
    import os

    from wayfinder.models import required_key_for

    required = []
    needed = required_key_for(model)
    if needed and not os.environ.get(needed):
        required.append(needed)
    if not os.environ.get("TAVILY_API_KEY"):
        required.append("TAVILY_API_KEY")
    if needs_langsmith and not os.environ.get("LANGSMITH_API_KEY"):
        required.append("LANGSMITH_API_KEY")

    if required:
        console.print(f"[red]Missing:[/] {', '.join(required)}")
        console.print("[dim]Copy .env.example to .env and fill them in.[/]")
        raise typer.Exit(2)

    if not os.environ.get("WAYFINDER_USER_AGENT"):
        console.print(
            "[yellow]WAYFINDER_USER_AGENT is unset[/] — Nominatim requires a "
            "descriptive User-Agent. Geocoding will fail without it."
        )


def _print_report(report: ConstraintReport) -> None:
    if report.passed and not report.violations:
        console.print("[bold green]PASS[/] — every constraint checks out.")
        return

    head = "[bold green]PASS[/]" if report.passed else "[bold red]FAIL[/]"
    console.print(
        f"{head} — {len(report.hard_violations)} hard, {len(report.soft_violations)} soft"
    )
    for v in report.violations:
        colour = "red" if v.severity == "hard" else "yellow"
        where = f" [dim]({v.where})[/]" if v.where else ""
        console.print(f"  [{colour}]{v.severity:<4}[/] [bold]{v.check}[/]{where} {v.message}")


def _print_metrics(report: ConstraintReport) -> None:
    interesting = [
        "hard_pass_rate",
        "soft_pass_rate",
        "budget_overrun_pct",
        "must_do_coverage",
        "grounded_pct",
        "total_cost",
    ]
    shown = {k: report.metrics[k] for k in interesting if k in report.metrics}
    console.print(
        "  ".join(f"[dim]{k}[/] {v:g}" for k, v in shown.items()),
        highlight=False,
    )


@app.command()
def plan(
    spec_path: Annotated[Path, typer.Argument(help="Trip spec YAML.")],
    model: Annotated[
        str, typer.Option(help="Main agent model, as provider:id.")
    ] = "openrouter:deepseek/deepseek-v4-flash",
    subagent_model: Annotated[
        str | None, typer.Option(help="Model for subagents. Defaults to --model.")
    ] = None,
    subagents: Annotated[bool, typer.Option(help="Use research subagents.")] = True,
    skills: Annotated[bool, typer.Option(help="Load the SKILL.md files.")] = True,
    repair: Annotated[
        bool, typer.Option(help="Give the agent the check_itinerary tool.")
    ] = True,
    single_researcher: Annotated[
        bool, typer.Option(help="One generalist researcher instead of three specialists.")
    ] = False,
) -> None:
    """Plan a trip. Writes a run directory with every artifact."""
    _load_env()
    _preflight(model)
    spec = load_spec(spec_path)
    config = AgentConfig(
        model=model,
        subagent_model=subagent_model,
        use_subagents=subagents,
        use_skills=skills,
        use_repair_loop=repair,
        single_researcher=single_researcher,
    )

    console.print(f"[bold]{spec.destination}[/] · {config.label()}")
    with console.status("planning…", spinner="dots"):
        result = plan_trip(spec, config)

    console.print(f"\n[dim]{result.run_dir}[/]")
    if result.error:
        console.print(f"[bold red]run failed[/] {result.error}")
    console.print(f"[dim]check_itinerary calls:[/] {result.check_calls}")
    console.print()

    if result.itinerary is not None and not result.itinerary.feasible:
        console.print("[bold yellow]Reported infeasible[/]")
        console.print(f"  {result.itinerary.infeasibility_reason}")
        console.print()

    _print_report(result.report)
    _print_metrics(result.report)

    raise typer.Exit(0 if result.report.passed else 1)


@app.command()
def check(
    target: Annotated[
        Path, typer.Argument(help="A run directory, or an itinerary.json (with --spec).")
    ],
    spec_path: Annotated[
        Path | None, typer.Option("--spec", help="Trip spec, if TARGET is a bare itinerary.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit the raw report.")] = False,
) -> None:
    """Re-run the constraint checker over an itinerary the agent already wrote.

    The same code path the agent calls mid-run and the evaluators call at
    scoring time — so what you see here is exactly what they saw.
    """
    _load_env()

    if target.is_dir():
        itinerary_path = target / "itinerary.json"
        resolved_spec = spec_path or target / "spec.yaml"
    else:
        itinerary_path = target
        resolved_spec = spec_path

    if resolved_spec is None:
        console.print("[red]Pass --spec when checking a bare itinerary file.[/]")
        raise typer.Exit(2)
    for path in (itinerary_path, resolved_spec):
        if not path.exists():
            console.print(f"[red]No such file:[/] {path}")
            raise typer.Exit(2)

    report = check_payload(load_spec(resolved_spec), load_itinerary_payload(itinerary_path))

    if as_json:
        console.print_json(json.dumps(report.to_dict()))
    else:
        _print_report(report)
        _print_metrics(report)

    raise typer.Exit(0 if report.passed else 1)


@app.command()
def serve(
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8100,
    host: Annotated[
        str, typer.Option(help="Bind address. Localhost only unless you know why.")
    ] = "127.0.0.1",
    model: Annotated[
        str, typer.Option(help="Default model in the UI.")
    ] = "openrouter:deepseek/deepseek-v4-flash",
) -> None:
    """Start the local web UI: live tool calls, approvals, and results.

    No authentication and runs held in memory — this binds to localhost on
    purpose. Don't put it on a public interface.
    """
    _load_env()
    _preflight(model)

    import uvicorn

    from wayfinder.server import create_app

    console.print(f"[bold green]Wayfinder[/] → [link]http://{host}:{port}[/]")
    console.print("[dim]Ctrl-C to stop.[/]")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


@app.command("sync-dataset")
def sync_dataset_command(
    dataset: Annotated[str, typer.Option(help="LangSmith dataset name.")] = "wayfinder-trips",
) -> None:
    """Push `evals/dataset.yaml` to LangSmith, replacing what's there."""
    _load_env()
    from wayfinder.evals.datasets import load_cases, sync_dataset

    cases = load_cases()
    sync_dataset(dataset)
    console.print(f"[green]Synced[/] {len(cases)} cases to [bold]{dataset}[/]")
    for category in sorted({c.category for c in cases}):
        names = [c.name for c in cases if c.category == category]
        console.print(f"  [dim]{category:<18}[/] {len(names)}")


@app.command("eval")
def eval_command(
    arm: Annotated[
        str, typer.Option(help="Experiment arm from the matrix, or 'baseline'.")
    ] = "baseline",
    dataset: Annotated[str, typer.Option(help="LangSmith dataset name.")] = "wayfinder-trips",
    repetitions: Annotated[
        int, typer.Option(help="Runs per case. Use 3+ to measure variance.")
    ] = 1,
    judges: Annotated[bool, typer.Option(help="Include the LLM judges.")] = True,
    concurrency: Annotated[int, typer.Option(help="Parallel cases.")] = 4,
    case: Annotated[
        list[str] | None,
        typer.Option("--case", help="Repeat to run only these cases. Default: all 20."),
    ] = None,
) -> None:
    """Run one experiment arm over the dataset and report it to LangSmith."""
    _load_env()
    from wayfinder.evals.run import EXPERIMENT_MATRIX, run_experiment

    if arm not in EXPERIMENT_MATRIX:
        console.print(f"[red]Unknown arm[/] {arm!r}. Known: {', '.join(EXPERIMENT_MATRIX)}")
        raise typer.Exit(2)

    config = EXPERIMENT_MATRIX[arm]
    _preflight(config.model, needs_langsmith=True)

    runs = (len(case) if case else 20) * repetitions
    scope = f"{len(case)} case(s)" if case else "all 20 cases"
    console.print(
        f"[bold]{arm}[/] · {config.label()} · {scope} × {repetitions} = "
        f"[bold]{runs} agentic run(s)[/]{'' if judges else ' · no judges'}"
    )

    results = run_experiment(
        config,
        experiment_prefix=arm,
        dataset_name=dataset,
        repetitions=repetitions,
        use_judges=judges,
        max_concurrency=concurrency,
        cases=case,
    )
    console.print(results)


@app.command()
def matrix(
    arms: Annotated[
        list[str] | None, typer.Option("--arm", help="Repeat to select arms. Default: all.")
    ] = None,
    dataset: Annotated[str, typer.Option(help="LangSmith dataset name.")] = "wayfinder-trips",
    repetitions: Annotated[int, typer.Option(help="Runs per case.")] = 1,
    judges: Annotated[bool, typer.Option(help="Include the LLM judges.")] = True,
) -> None:
    """Run several arms back to back, then compare them in the LangSmith UI."""
    _load_env()
    from wayfinder.evals.run import EXPERIMENT_MATRIX, run_matrix

    chosen = arms or list(EXPERIMENT_MATRIX)
    console.print(f"Running {len(chosen)} arms: {', '.join(chosen)}")
    run_matrix(chosen, dataset_name=dataset, repetitions=repetitions, use_judges=judges)
    console.print("[green]Done.[/] Compare them in the LangSmith experiment view.")


@app.command("eval-local")
def eval_local(
    arm: Annotated[str, typer.Option(help="Experiment arm.")] = "baseline",
    case: Annotated[
        list[str] | None, typer.Option("--case", help="Repeat to pick specific cases.")
    ] = None,
) -> None:
    """Score the dataset with the code evaluators only — no LangSmith needed.

    Useful while iterating on prompts, and the way to get numbers when you have
    a model key but no LangSmith key.
    """
    _load_env()
    from wayfinder.evals.run import EXPERIMENT_MATRIX, run_local

    if arm not in EXPERIMENT_MATRIX:
        console.print(f"[red]Unknown arm[/] {arm!r}. Known: {', '.join(EXPERIMENT_MATRIX)}")
        raise typer.Exit(2)

    _preflight(EXPERIMENT_MATRIX[arm].model)
    rows = run_local(EXPERIMENT_MATRIX[arm], case_names=case)
    if not rows:
        console.print("[yellow]No cases matched.[/]")
        raise typer.Exit(1)

    from rich.table import Table

    keys = [k for k in rows[0] if k not in ("name", "category")]
    table = Table(title=f"{arm} · code evaluators")
    table.add_column("case")
    table.add_column("category")
    for k in keys:
        table.add_column(k.replace("_", " "), justify="right")
    for row in rows:
        table.add_row(
            row["name"], row["category"], *[f"{row[k]:g}" for k in keys]
        )
    console.print(table)

    passed = sum(1 for r in rows if r["plan_passes"])
    console.print(f"\n{passed}/{len(rows)} plans passed every hard constraint.")


if __name__ == "__main__":
    app()

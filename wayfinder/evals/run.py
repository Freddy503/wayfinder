"""Experiment driver.

One experiment is one `AgentConfig` swept across the whole dataset. Comparing
two of them is the entire point of the project, so two things matter here more
than anywhere else:

- **Change one variable at a time.** The matrix below differs from the baseline
  in exactly one flag per arm.
- **Know your noise floor.** `repetitions` runs each case more than once. A 5%
  delta means nothing until you know whether re-running the *same* config moves
  it by 8%.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from langsmith import Client

from wayfinder.agent import AgentConfig, new_run_dir, plan_trip
from wayfinder.evals.datasets import DEFAULT_DATASET_NAME, load_cases
from wayfinder.evals.evaluators import build_evaluators
from wayfinder.render import render_markdown, render_sources
from wayfinder.schema import TripSpec

BASELINE = AgentConfig()

#: Each arm turns off exactly one thing. Read a column of results as "what did
#: this component buy us?" — which is only a fair question if nothing else moved.
EXPERIMENT_MATRIX: dict[str, AgentConfig] = {
    "baseline": BASELINE,
    "no-repair-loop": replace(BASELINE, use_repair_loop=False),
    "no-skills": replace(BASELINE, use_skills=False),
    "no-subagents": replace(BASELINE, use_subagents=False),
    "one-researcher": replace(BASELINE, single_researcher=True),
    #: The model-tiering arm, and the only one that isn't Flash. Same family,
    #: more capability, ~3x the price — it answers "does the harness need a
    #: better model, or does the checker close the gap?" without leaving
    #: OpenRouter. Opt-in via `--arm deepseek-pro`; nothing runs it by default.
    "deepseek-pro": replace(BASELINE, model="openrouter:deepseek/deepseek-v4-pro"),
}


def make_target(config: AgentConfig, runs_root: Path | None = None):
    """Build the callable LangSmith runs against each dataset example.

    A failed run returns a failing score rather than raising: an exception would
    drop the case from the experiment, and an experiment that silently scores
    fewer cases than another is not comparable to it.
    """

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        spec = TripSpec.model_validate(inputs["spec"])
        run_dir = new_run_dir(spec, root=runs_root)
        result = plan_trip(spec, config, run_dir=run_dir)

        markdown = ""
        sources = ""
        if result.itinerary is not None:
            markdown = render_markdown(spec, result.itinerary, result.report)
            sources = render_sources(result.itinerary)

        tool_calls = result.tool_calls()
        return {
            "passed": result.report.passed,
            "check_calls": result.check_calls,
            # Effort, so the matrix can compare arms on work done and not just
            # on quality — which saturates at 1.0 on anything easy.
            "tool_calls": sum(tool_calls.values()),
            "tool_breakdown": tool_calls,
            "verification_calls": sum(
                n for t, n in tool_calls.items()
                if t in ("geocode", "venue_rating", "estimate_travel")
            ),
            "error": result.error,
            "run_dir": str(result.run_dir),
            # Metrics and violations only — the full itinerary would balloon
            # every judge prompt, and the markdown below says the same thing in
            # a form a judge can actually read.
            "report": {
                "metrics": result.report.metrics,
                "violations": [v.to_dict() for v in result.report.violations],
            },
            "itinerary_markdown": markdown,
            "sources_markdown": sources,
        }

    return target


def select_examples(client: Client, dataset_name: str, names: list[str]) -> list[Any]:
    """Pick named examples out of the dataset.

    The cost lever. A full sweep is 20 agentic runs per repetition; most
    questions worth asking early — does the pipeline work, what is the noise
    floor — are answerable on a handful of well-chosen cases for a fraction of
    the spend.
    """
    wanted = set(names)
    examples = [
        e
        for e in client.list_examples(dataset_name=dataset_name)
        if (e.metadata or {}).get("name") in wanted
    ]
    found = {(e.metadata or {}).get("name") for e in examples}
    missing = wanted - found
    if missing:
        msg = f"no such case(s) in {dataset_name!r}: {sorted(missing)}"
        raise KeyError(msg)
    return examples


def run_experiment(
    config: AgentConfig,
    *,
    experiment_prefix: str,
    dataset_name: str = DEFAULT_DATASET_NAME,
    repetitions: int = 1,
    use_judges: bool = True,
    max_concurrency: int = 4,
    cases: list[str] | None = None,
    client: Client | None = None,
    runs_root: Path | None = None,
) -> Any:
    client = client or Client()
    data: Any = (
        select_examples(client, dataset_name, cases) if cases else dataset_name
    )
    return client.evaluate(
        make_target(config, runs_root=runs_root),
        data=data,
        evaluators=build_evaluators(use_judges=use_judges),
        experiment_prefix=experiment_prefix,
        num_repetitions=repetitions,
        max_concurrency=max_concurrency,
        metadata={
            "wayfinder_config": config.label(),
            "judges": use_judges,
            "cases": ",".join(cases) if cases else "all",
            **_config_metadata(config),
        },
    )


def run_matrix(
    arms: list[str] | None = None,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    repetitions: int = 1,
    use_judges: bool = True,
    max_concurrency: int = 4,
) -> dict[str, Any]:
    """Run several arms back to back. Compare them in the LangSmith UI."""
    client = Client()
    chosen = arms or list(EXPERIMENT_MATRIX)
    unknown = [a for a in chosen if a not in EXPERIMENT_MATRIX]
    if unknown:
        msg = f"unknown arms {unknown}; known: {sorted(EXPERIMENT_MATRIX)}"
        raise KeyError(msg)

    return {
        arm: run_experiment(
            EXPERIMENT_MATRIX[arm],
            experiment_prefix=arm,
            dataset_name=dataset_name,
            repetitions=repetitions,
            use_judges=use_judges,
            max_concurrency=max_concurrency,
            client=client,
        )
        for arm in chosen
    }


def run_local(
    config: AgentConfig,
    case_names: list[str] | None = None,
    runs_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Run the dataset without LangSmith, scoring with the code evaluators only.

    For when you want to iterate on prompts without creating experiments — or
    when you have a model key but no LangSmith key.
    """
    from wayfinder.evals.evaluators import CODE_EVALUATORS

    cases = [c for c in load_cases() if not case_names or c.name in (case_names or [])]
    target = make_target(config, runs_root=runs_root)

    rows = []
    for case in cases:
        outputs = target(case.inputs())
        scores = {}
        for evaluator in CODE_EVALUATORS:
            result = evaluator(case.inputs(), outputs, reference_outputs=case.reference())
            scores[result["key"]] = result["score"]
        rows.append({"name": case.name, "category": case.category, **scores})
    return rows


def _config_metadata(config: AgentConfig) -> dict[str, Any]:
    from dataclasses import asdict

    return {f"cfg_{k}": v for k, v in asdict(config).items()}

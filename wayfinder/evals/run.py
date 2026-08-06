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
from wayfinder.evals.evaluators import SUMMARY_EVALUATORS, build_evaluators
from wayfinder.render import render_markdown, render_sources
from wayfinder.schema import TripSpec

BASELINE = AgentConfig()

#: Tools that cost a network round trip per call, batch or not.
VERIFY_TOOLS = frozenset({
    "geocode", "venue_rating", "estimate_travel",
    "geocode_all", "venue_ratings", "estimate_travel_all",
})

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


def outputs_for(result: Any) -> dict[str, Any]:
    """The shape every evaluator reads, built from one finished run.

    Extracted so the dataset target and a one-off trip produce byte-identical
    inputs to the same evaluators. Two functions computing "the same" scores is
    how an experiment and a real run start quietly disagreeing.
    """
    markdown = sources = ""
    if result.itinerary is not None:
        markdown = render_markdown(result.spec, result.itinerary, result.report)
        sources = render_sources(result.itinerary)

    tool_calls = result.tool_calls()
    return {
        "passed": result.report.passed,
        "check_calls": result.check_calls,
        # Effort, so the matrix can compare arms on work done and not just
        # on quality — which saturates at 1.0 on anything easy.
        "tool_calls": sum(tool_calls.values()),
        "tool_breakdown": tool_calls,
        # Turns spent verifying, not items verified. The batch tools were
        # added precisely to collapse thirty of these into one, so counting
        # a batch as a single call is the point — the metric measures the
        # thing that costs wall-clock.
        "verification_calls": sum(n for t, n in tool_calls.items() if t in VERIFY_TOOLS),
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


def make_target(config: AgentConfig, runs_root: Path | None = None):
    """Build the callable LangSmith runs against each dataset example.

    A failed run returns a failing score rather than raising: an exception would
    drop the case from the experiment, and an experiment that silently scores
    fewer cases than another is not comparable to it.
    """

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        spec = TripSpec.model_validate(inputs["spec"])
        run_dir = new_run_dir(spec, root=runs_root)
        return outputs_for(plan_trip(spec, config, run_dir=run_dir))

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


def select_split(client: Client, dataset_name: str, split: str) -> list[Any]:
    """Every example in one split.

    Splits are the dataset's own categories — easy, tight-budget,
    constraint-dense, infeasible and so on. They behave differently enough
    that one average over all twenty hides more than it shows: an arm can gain
    on the easy cases and lose on the infeasible ones and look unchanged.
    """
    examples = [
        e
        for e in client.list_examples(dataset_name=dataset_name)
        if split in ((e.metadata or {}).get("dataset_split") or [])
    ]
    if not examples:
        known = sorted({
            s
            for e in client.list_examples(dataset_name=dataset_name)
            for s in ((e.metadata or {}).get("dataset_split") or [])
        })
        msg = f"no split {split!r} in {dataset_name!r}. Known: {known}"
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
    split: str | None = None,
    client: Client | None = None,
    runs_root: Path | None = None,
) -> Any:
    client = client or Client()
    if cases:
        data: Any = select_examples(client, dataset_name, cases)
    elif split:
        data = select_split(client, dataset_name, split)
    else:
        data = dataset_name
    return client.evaluate(
        make_target(config, runs_root=runs_root),
        data=data,
        evaluators=build_evaluators(use_judges=use_judges),
        # Run once over the whole experiment. An average is the wrong summary
        # for "did anything crash", and the right one for "how did we do on the
        # cases that should have been refused" — but only over that subset.
        summary_evaluators=SUMMARY_EVALUATORS,
        experiment_prefix=experiment_prefix,
        num_repetitions=repetitions,
        max_concurrency=max_concurrency,
        metadata={
            "wayfinder_config": config.label(),
            "judges": use_judges,
            "cases": ",".join(cases) if cases else "all",
            "split": split or "all",
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
    cases: list[str] | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    """Run several arms back to back. Compare them in the LangSmith UI.

    `cases` and `split` are the reason this is runnable at all. Every arm over
    all twenty cases with repetitions is ~22 hours and tens of dollars, so the
    only sweep anyone actually does is a narrow one: two arms, the cases where
    the answer can differ, three repetitions.

    One arm failing does not lose the others. Each is a separate LangSmith
    experiment that has already succeeded and cost real hours; a comprehension
    would throw all of that away because arm three raised.
    """
    client = Client()
    chosen = arms or list(EXPERIMENT_MATRIX)
    unknown = [a for a in chosen if a not in EXPERIMENT_MATRIX]
    if unknown:
        msg = f"unknown arms {unknown}; known: {sorted(EXPERIMENT_MATRIX)}"
        raise KeyError(msg)

    results: dict[str, Any] = {}
    for arm in chosen:
        try:
            results[arm] = run_experiment(
                EXPERIMENT_MATRIX[arm],
                experiment_prefix=f"{arm}-{split}" if split else arm,
                dataset_name=dataset_name,
                repetitions=repetitions,
                use_judges=use_judges,
                max_concurrency=max_concurrency,
                cases=cases,
                split=split,
                client=client,
            )
        except Exception as exc:  # noqa: BLE001 — one arm, not the sweep
            results[arm] = {"error": f"{type(exc).__name__}: {exc}"}
    return results


def experiment_names(results: dict[str, Any]) -> dict[str, str]:
    """The names LangSmith actually used, per arm.

    Not the prefixes we asked for. `experiment_prefix="baseline"` becomes
    `baseline-f1ffe9b0` — a suffix keeps repeated sweeps from colliding — so
    looking the scores up by prefix finds nothing and prints an empty table
    after two hours of runs. Ask the result object what it was called.
    """
    names: dict[str, str] = {}
    for arm, result in results.items():
        name = getattr(result, "experiment_name", None)
        if name:
            names[arm] = name
    return names


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


#: What the comparison table shows. Quality first, then effort — and effort
#: matters because quality saturates: on an easy case every arm scores 1.0 and
#: the only thing left that separates them is what they spent getting there.
COMPARE_KEYS = (
    "plan_passes",
    "hard_constraint_pass_rate",
    "soft_constraint_pass_rate",
    "route_efficiency",
    "correctly_refused",
    "verification_calls",
    "check_calls",
)


def collect_scores(
    experiment_names: list[str], client: Client | None = None
) -> dict[str, dict[str, list[float]]]:
    """Every feedback score for each experiment, keyed by arm then metric.

    Returns the raw lists rather than means, because the spread is the point:
    with repetitions, a difference smaller than the standard deviation is not a
    result, and only the samples can tell you that.
    """
    client = client or Client()
    out: dict[str, dict[str, list[float]]] = {}
    for name in experiment_names:
        scores: dict[str, list[float]] = {}
        try:
            runs = [r for r in client.list_runs(project_name=name, is_root=True, limit=100)]
            for run in runs:
                for feedback in client.list_feedback(run_ids=[run.id]):
                    if feedback.score is not None:
                        scores.setdefault(feedback.key, []).append(float(feedback.score))
        except Exception:  # noqa: BLE001 — a missing experiment is a blank row
            pass
        out[name] = scores
    return out


def summarise(values: list[float]) -> tuple[float, float]:
    """Mean and standard deviation. Zero spread when there is one sample —
    which is not stability, it is a sample size of one, and the caller has to
    say so rather than let a `0.000` imply confidence."""
    if not values:
        return (float("nan"), 0.0)
    mean = sum(values) / len(values)
    if len(values) < 2:
        return (mean, 0.0)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return (mean, variance**0.5)


def compare(
    arms: list[str],
    experiment_names: dict[str, str] | None = None,
    client: Client | None = None,
) -> dict[str, Any]:
    """Arms side by side, each metric against the baseline.

    A delta is only reported as meaningful when it exceeds the combined spread
    of the two arms. Anything inside the noise is reported as "can't tell",
    which is the honest answer and the one this project keeps needing.
    """
    names = experiment_names or {arm: arm for arm in arms}
    raw = collect_scores([names[a] for a in arms], client=client)

    table: dict[str, Any] = {}
    for arm in arms:
        scores = raw.get(names[arm], {})
        table[arm] = {k: summarise(scores.get(k, [])) for k in COMPARE_KEYS}
        table[arm]["_n"] = max((len(v) for v in scores.values()), default=0)

    baseline = table.get("baseline")
    if baseline:
        for arm in arms:
            if arm == "baseline":
                continue
            verdicts = {}
            for key in COMPARE_KEYS:
                mean, spread = table[arm][key]
                base_mean, base_spread = baseline[key]
                delta = mean - base_mean
                noise = base_spread + spread
                verdicts[key] = {
                    "delta": delta,
                    "meaningful": abs(delta) > noise > 0 or (noise == 0 and delta != 0),
                }
            table[arm]["_vs_baseline"] = verdicts
    return table

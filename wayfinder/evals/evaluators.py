"""Evaluators.

Two families, and the split is the point of the whole project.

**Code evaluators** read `verify.py`'s metrics straight off the run. They are
exact, free, deterministic, and they are the ones that will actually tell you
whether a change helped. Almost every question worth asking about a trip
planner — did it fit the budget, can you get between the stops, was the museum
open — is decidable in Python.

**LLM judges** cover only what code genuinely cannot reach: whether the plan
honours a preference like "neighbourhood restaurants over tourist-facing ones",
whether the claims trace to the cited sources, whether a human would enjoy
reading it. Three of them, deliberately — reach for a judge when you've
established that no code check will do, not before.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from openevals.llm import create_llm_as_judge

from wayfinder.models import DEFAULT_MODEL

#: The same model the planner uses, since it's the only one configured.
#:
#: Worth knowing what that costs: a judge grading its own family's output tends
#: to rate it generously, so `taste_match`, `groundedness` and `readability`
#: are best read as trends across arms rather than as absolute scores. The code
#: evaluators have no such problem — which is the argument for leaning on them,
#: and the reason this project has only three judges. Pass
#: `build_evaluators(judge_model=...)` for a second opinion from another family.
JUDGE_MODEL = DEFAULT_MODEL


def _metrics(outputs: dict[str, Any]) -> dict[str, float]:
    return (outputs or {}).get("report", {}).get("metrics", {}) or {}


def produced_output(outputs: dict[str, Any]) -> bool:
    """Did this run produce a validated itinerary at all?

    The distinction that matters for every "absence is good" metric below.
    A crashed run has no budget to overrun and no infeasible transit legs —
    which naively scores as perfect budget adherence and perfect transit
    feasibility. An arm that crashed on every case would top those columns.

    A declared-infeasible run *is* real output: it parsed, it reached a
    decision, and having no costs to check is the correct outcome rather than
    a missing one. So the gate is schema validity, not the presence of days.
    """
    return _metrics(outputs).get("schema_valid") == 1.0


def _from_metric(
    key: str,
    feedback_key: str,
    default: float = 0.0,
    transform: Callable[[float], float] | None = None,
    requires_output: bool = False,
) -> Callable[..., dict[str, Any]]:
    """Lift one of `verify.py`'s metrics into a LangSmith evaluator.

    Set `requires_output` on any metric where a *missing* value would read as
    a good score. Those return 0.0 for a run that produced nothing, so a crash
    is never mistaken for success.
    """

    def evaluator(inputs: dict, outputs: dict, **_: Any) -> dict[str, Any]:
        if requires_output and not produced_output(outputs):
            return {
                "key": feedback_key,
                "score": 0.0,
                "comment": "no itinerary produced — not scoreable",
            }
        value = _metrics(outputs).get(key, default)
        return {"key": feedback_key, "score": transform(value) if transform else value}

    evaluator.__name__ = feedback_key
    return evaluator


# --------------------------------------------------------------------------
# Code evaluators
# --------------------------------------------------------------------------

hard_constraint_pass_rate = _from_metric("hard_pass_rate", "hard_constraint_pass_rate")
soft_constraint_pass_rate = _from_metric("soft_pass_rate", "soft_constraint_pass_rate")
schema_valid = _from_metric("schema_valid", "schema_valid")
must_do_coverage = _from_metric("must_do_coverage", "must_do_coverage")
grounded_pct = _from_metric("grounded_pct", "grounded_pct")

#: Inverted so that, like every other score here, higher is better — mixing
#: directions in one experiment view is how you end up reading a regression as
#: an improvement.
budget_respected = _from_metric(
    "budget_overrun_pct",
    "budget_respected",
    transform=lambda v: max(0.0, 1.0 - v),
    requires_output=True,
)
transit_feasible = _from_metric(
    "transit_infeasibility_count",
    "transit_feasible",
    transform=lambda v: 1.0 if v == 0 else 0.0,
    requires_output=True,
)


def plan_passes(inputs: dict, outputs: dict, **_: Any) -> dict[str, Any]:
    """The headline number: did the plan clear every hard constraint?"""
    return {"key": "plan_passes", "score": float(bool((outputs or {}).get("passed")))}


def correctly_refused(
    inputs: dict, outputs: dict, reference_outputs: dict | None = None, **_: Any
) -> dict[str, Any]:
    """Did the agent refuse exactly when it should have?

    Scored on every case, not just the impossible ones — otherwise an agent
    that refuses everything would score perfectly on the infeasible subset and
    never be penalised for it.

    A run that produced no itinerary scores 0.0 regardless of the reference.
    Without that guard a crash reads as a deliberate decision: on a solvable
    spec, "didn't refuse" was trivially true of a run that never got far
    enough to refuse anything, and it collected full marks for it.
    """
    expected = bool((reference_outputs or {}).get("should_refuse", False))
    if not produced_output(outputs):
        return {
            "key": "correctly_refused",
            "score": 0.0,
            "comment": "no itinerary produced — the run failed rather than deciding",
        }
    actual = _metrics(outputs).get("refused", 0.0) == 1.0
    if expected == actual:
        comment = "refused as expected" if expected else "planned as expected"
    elif expected:
        comment = "fabricated a plan for a spec that cannot be satisfied"
    else:
        comment = "refused a spec that was satisfiable"
    return {"key": "correctly_refused", "score": float(expected == actual), "comment": comment}


def tool_calls(inputs: dict, outputs: dict, **_: Any) -> dict[str, Any]:
    """Total tool calls. Diagnostic — effort, not quality.

    The column that made the 4.3x wall-clock spread legible: two runs both
    scoring a perfect 1.0 did 55 and 145 tool calls respectively. Where
    quality saturates, this is the only thing left that can separate arms.
    """
    return {"key": "tool_calls", "score": float((outputs or {}).get("tool_calls", 0))}


def verification_calls(inputs: dict, outputs: dict, **_: Any) -> dict[str, Any]:
    """Calls to the expensive verifiers — geocode, ratings, travel estimates.

    Tracked apart from `tool_calls` because this is the number the
    shortlist-before-you-verify instruction is meant to move: searching wide is
    cheap and good, verifying candidates you then discard is neither.
    """
    return {
        "key": "verification_calls",
        "score": float((outputs or {}).get("verification_calls", 0)),
    }


def check_calls(inputs: dict, outputs: dict, **_: Any) -> dict[str, Any]:
    """How many times the repair loop ran. Diagnostic, not a quality score.

    Zero on a passing run means the agent got it right first try; zero on a
    failing run means it never looked.
    """
    return {"key": "check_calls", "score": float((outputs or {}).get("check_calls", 0))}


CODE_EVALUATORS = [
    plan_passes,
    hard_constraint_pass_rate,
    soft_constraint_pass_rate,
    schema_valid,
    budget_respected,
    transit_feasible,
    must_do_coverage,
    grounded_pct,
    correctly_refused,
    check_calls,
    tool_calls,
    verification_calls,
]


# --------------------------------------------------------------------------
# LLM judges — only for what code cannot decide
# --------------------------------------------------------------------------

TASTE_PROMPT = """\
You are judging whether a trip itinerary honours the traveller's stated \
preferences.

<preferences>
{inputs}
</preferences>

<itinerary>
{outputs}
</itinerary>

Score how well the plan reflects the soft preferences in the spec — the free-text \
wishes, dietary needs and stated tastes. Judge only preference-fit. Budget, \
opening hours and travel times are checked elsewhere; ignore them here.

1.0 — every preference is visibly reflected in specific choices.
0.5 — some are honoured, others ignored without reason.
0.0 — the plan reads as though the preferences were never given.

An itinerary that correctly reports the trip as infeasible should score 1.0: \
there was nothing to plan, so nothing to get wrong.
"""

GROUNDEDNESS_PROMPT = """\
You are judging whether an itinerary's factual claims are supported by the \
sources it cites.

<itinerary_and_sources>
{outputs}
</itinerary_and_sources>

Look at the concrete claims: opening hours, admission prices, addresses, \
whether a place exists at all. For each, is there a cited URL that plausibly \
supports it?

1.0 — every substantive claim has a plausible source.
0.5 — the main claims are sourced, details are not.
0.0 — specific-sounding facts with nothing behind them.

Judge sourcing, not correctness — you cannot open the links. A venue with no \
opening hours recorded is an honest gap, not an unsupported claim; penalise \
invented precision, not admitted uncertainty.
"""

READABILITY_PROMPT = """\
You are judging whether a rendered itinerary is one a traveller could actually \
follow.

<itinerary>
{outputs}
</itinerary>

Consider: is the shape of each day clear at a glance? Is it obvious what \
happens when, and how to get between things? Is the day's rhythm sensible, or \
does it read as a list of items that happen to be sorted by time? Is anything \
important missing — a booking that needs making, a warning worth having?

1.0 — you could hand this to someone and they would know what to do.
0.5 — the information is there but you would have to work to use it.
0.0 — hard to follow, or padded with commentary that gets in the way.
"""


def build_judges(model: str = JUDGE_MODEL) -> list[Callable[..., Any]]:
    """Build the three judges.

    openevals resolves a `model=` string through LangChain's own initialiser,
    which has no notion of an `openrouter:` prefix. Anything this project's
    resolver builds into a client is therefore passed as `judge=` instead.
    """
    from wayfinder.models import resolve_model

    resolved = resolve_model(model)
    kwargs: dict[str, Any] = (
        {"model": model} if isinstance(resolved, str) else {"judge": resolved}
    )
    return [
        create_llm_as_judge(
            prompt=prompt, feedback_key=key, continuous=True, **kwargs
        )
        for prompt, key in (
            (TASTE_PROMPT, "taste_match"),
            (GROUNDEDNESS_PROMPT, "groundedness"),
            (READABILITY_PROMPT, "readability"),
        )
    ]


def build_evaluators(use_judges: bool = True, judge_model: str = JUDGE_MODEL):
    return [*CODE_EVALUATORS, *(build_judges(judge_model) if use_judges else [])]


# --------------------------------------------------------------------------
# Summary evaluators — one score for the whole experiment
#
# Per-case scores are averaged in the UI, and an average is exactly the wrong
# summary for some questions. "Did anything crash?" is not a mean. "How did we
# do on the cases that were meant to be refused?" is a mean over a *subset*.
# These run once per experiment, over every run in it.
# --------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def crash_free_rate(outputs: list[dict], reference_outputs: list[dict], **_: Any):
    """Fraction of cases that produced a validated itinerary at all.

    Separated from `plan_passes` because the two failure modes want different
    responses: a plan that broke a constraint is a modelling problem, a run
    that crashed is an engineering one, and averaging them together tells you
    to fix the wrong thing.
    """
    return {
        "key": "crash_free_rate",
        "score": _mean([1.0 if produced_output(o) else 0.0 for o in outputs or []]),
    }


def refusal_accuracy(outputs: list[dict], reference_outputs: list[dict], **_: Any):
    """`correctly_refused`, but only over the cases that should be refused.

    The per-case version is scored on all twenty so an agent that refuses
    everything can't top the infeasible subset. That makes it a blended number.
    This one answers the narrower question directly: of the specs that cannot
    be satisfied, how many did it actually turn down?
    """
    scored = [
        1.0 if _metrics(o).get("refused", 0.0) == 1.0 and produced_output(o) else 0.0
        for o, r in zip(outputs or [], reference_outputs or [], strict=False)
        if (r or {}).get("should_refuse")
    ]
    if not scored:
        return {"key": "refusal_accuracy", "score": None,
                "comment": "no infeasible cases in this run"}
    return {"key": "refusal_accuracy", "score": _mean(scored)}


def median_verification_calls(outputs: list[dict], reference_outputs: list[dict], **_: Any):
    """Median rather than mean: one runaway case would otherwise set the number.

    This is the column the batch tools were built to move, so it wants to
    reflect the typical run and not the worst one.
    """
    values = sorted(float((o or {}).get("verification_calls", 0)) for o in outputs or [])
    if not values:
        return {"key": "median_verification_calls", "score": 0.0}
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
    return {"key": "median_verification_calls", "score": median}


SUMMARY_EVALUATORS = [crash_free_rate, refusal_accuracy, median_verification_calls]

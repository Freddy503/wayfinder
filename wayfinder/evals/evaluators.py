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

#: Judges run on a cheaper model than the planner. Grading is a much easier task
#: than planning, and a judge that costs as much as the run makes the eval
#: matrix too expensive to sweep — which is the thing you actually want to do.
JUDGE_MODEL = "anthropic:claude-sonnet-5"


def _metrics(outputs: dict[str, Any]) -> dict[str, float]:
    return (outputs or {}).get("report", {}).get("metrics", {}) or {}


def _from_metric(
    key: str,
    feedback_key: str,
    default: float = 0.0,
    transform: Callable[[float], float] | None = None,
) -> Callable[..., dict[str, Any]]:
    """Lift one of `verify.py`'s metrics into a LangSmith evaluator."""

    def evaluator(inputs: dict, outputs: dict, **_: Any) -> dict[str, Any]:
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
    "budget_overrun_pct", "budget_respected", transform=lambda v: max(0.0, 1.0 - v)
)
transit_feasible = _from_metric(
    "transit_infeasibility_count",
    "transit_feasible",
    transform=lambda v: 1.0 if v == 0 else 0.0,
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
    """
    expected = bool((reference_outputs or {}).get("should_refuse", False))
    actual = _metrics(outputs).get("refused", 0.0) == 1.0
    if expected == actual:
        comment = "refused as expected" if expected else "planned as expected"
    elif expected:
        comment = "fabricated a plan for a spec that cannot be satisfied"
    else:
        comment = "refused a spec that was satisfiable"
    return {"key": "correctly_refused", "score": float(expected == actual), "comment": comment}


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
    return [
        create_llm_as_judge(
            prompt=TASTE_PROMPT, feedback_key="taste_match", model=model, continuous=True
        ),
        create_llm_as_judge(
            prompt=GROUNDEDNESS_PROMPT,
            feedback_key="groundedness",
            model=model,
            continuous=True,
        ),
        create_llm_as_judge(
            prompt=READABILITY_PROMPT,
            feedback_key="readability",
            model=model,
            continuous=True,
        ),
    ]


def build_evaluators(use_judges: bool = True, judge_model: str = JUDGE_MODEL):
    return [*CODE_EVALUATORS, *(build_judges(judge_model) if use_judges else [])]

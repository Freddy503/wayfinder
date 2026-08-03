"""Tests for the dataset and the code evaluators.

Evaluators are pure functions over the output dict, so they're fully testable
offline — which matters, because a broken evaluator doesn't crash. It quietly
returns a plausible number, and every experiment scored with it is wrong in a
way nothing else will reveal.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from wayfinder.evals.datasets import load_cases
from wayfinder.evals.evaluators import (
    CODE_EVALUATORS,
    budget_respected,
    check_calls,
    correctly_refused,
    hard_constraint_pass_rate,
    plan_passes,
    schema_valid,
    transit_feasible,
)


def outputs(metrics: dict | None = None, **extra):
    """A run that produced a valid itinerary, unless the test says otherwise.

    `schema_valid` defaults to 1.0 because that is what a real run looks like;
    the evaluators now treat its absence as "produced nothing", which is the
    whole point of the crash guard. Tests that want a failed run use
    `crashed()` instead.
    """
    merged = {"schema_valid": 1.0, **(metrics or {})}
    return {"report": {"metrics": merged, "violations": []}, **extra}


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


def test_dataset_loads_and_validates():
    cases = load_cases()
    assert len(cases) >= 15
    assert all(case.spec.dates.end >= case.spec.dates.start for case in cases)


def test_dataset_spans_the_difficulty_range():
    """A dataset where everything passes measures nothing."""
    counts = Counter(case.category for case in load_cases())
    for category in ("easy", "tight-budget", "constraint-dense", "infeasible"):
        assert counts[category] >= 2, f"too few {category} cases: {counts[category]}"


def test_dataset_has_refusal_cases():
    refusals = [c for c in load_cases() if c.should_refuse]
    assert len(refusals) >= 3
    assert all(c.category == "infeasible" for c in refusals)


def test_answer_key_is_stripped_from_agent_inputs():
    """`should_refuse` is the reference output. Leaking it into the spec would
    turn the hardest cases in the set into a giveaway."""
    for case in load_cases():
        assert "should_refuse" not in case.inputs()["spec"]
        assert "should_refuse" in case.reference()


def test_case_names_are_unique():
    names = [c.name for c in load_cases()]
    assert len(names) == len(set(names))


# --------------------------------------------------------------------------
# Code evaluators
# --------------------------------------------------------------------------


def test_every_evaluator_survives_a_run_that_produced_nothing():
    """A crashed run must still score, or it drops out of the experiment and
    the arms stop being comparable."""
    for evaluator in CODE_EVALUATORS:
        result = evaluator({}, {}, reference_outputs={})
        assert isinstance(result["score"], float)


def test_plan_passes_is_binary():
    assert plan_passes({}, outputs(passed=True))["score"] == 1.0
    assert plan_passes({}, outputs(passed=False))["score"] == 0.0


def test_budget_respected_inverts_the_overrun():
    """Every score in the experiment view must point the same way."""
    assert budget_respected({}, outputs({"budget_overrun_pct": 0.0}))["score"] == 1.0
    assert budget_respected({}, outputs({"budget_overrun_pct": 0.25}))["score"] == 0.75
    assert budget_respected({}, outputs({"budget_overrun_pct": 3.0}))["score"] == 0.0


def test_transit_feasible_is_binary_on_the_count():
    assert transit_feasible({}, outputs({"transit_infeasibility_count": 0.0}))["score"] == 1.0
    assert transit_feasible({}, outputs({"transit_infeasibility_count": 1.0}))["score"] == 0.0


@pytest.mark.parametrize(
    ("should_refuse", "refused", "expected"),
    [
        (True, 1.0, 1.0),  # impossible spec, refused
        (True, 0.0, 0.0),  # impossible spec, plan fabricated
        (False, 0.0, 1.0),  # possible spec, planned
        (False, 1.0, 0.0),  # possible spec, refused anyway
    ],
)
def test_correctly_refused_scores_both_directions(should_refuse, refused, expected):
    """Scored on every case, so an agent that refuses everything is penalised
    on the twelve satisfiable specs rather than acing the three impossible ones."""
    result = correctly_refused(
        {}, outputs({"refused": refused}), reference_outputs={"should_refuse": should_refuse}
    )
    assert result["score"] == expected
    assert result["comment"]


# --------------------------------------------------------------------------
# A crashed run must never out-score a working one
#
# Regression for a real incident: an eval where every run died on a billing
# 400 still scored `correctly_refused` 0.67, `budget_respected` 1.00 and
# `transit_feasible` 1.00 — because an absent metric reads as "nothing went
# wrong". An arm that crashed on every case would have topped those columns.
# --------------------------------------------------------------------------


def crashed():
    """Exactly what `_missing_itinerary_report` yields for a failed run."""
    return {
        "passed": False,
        "check_calls": 0,
        "report": {
            "metrics": {
                "schema_valid": 0.0,
                "hard_pass_rate": 0.0,
                "soft_pass_rate": 0.0,
                "hard_violation_count": 1.0,
            },
            "violations": [],
        },
    }


@pytest.mark.parametrize(
    "evaluator", [budget_respected, transit_feasible, correctly_refused]
)
def test_absence_metrics_score_zero_on_a_crashed_run(evaluator):
    result = evaluator({}, crashed(), reference_outputs={"should_refuse": False})
    assert result["score"] == 0.0, f"{result['key']} rewarded a run that produced nothing"
    assert "no itinerary" in result["comment"]


def test_crash_on_a_solvable_spec_no_longer_earns_refusal_credit():
    """The precise 6-of-9 artifact: 'didn't refuse' was true of a dead run."""
    result = correctly_refused({}, crashed(), reference_outputs={"should_refuse": False})
    assert result["score"] == 0.0


def test_a_real_refusal_still_scores_full_marks():
    """The guard must not punish a legitimate 'this trip is impossible'."""
    refused = outputs({"schema_valid": 1.0, "refused": 1.0, "budget_overrun_pct": 0.0,
                       "transit_infeasibility_count": 0.0}, passed=True)
    assert correctly_refused({}, refused, reference_outputs={"should_refuse": True})["score"] == 1.0
    # A refusal has no costs or legs to check; that is correct output, not missing.
    assert budget_respected({}, refused)["score"] == 1.0
    assert transit_feasible({}, refused)["score"] == 1.0


def test_a_working_plan_is_unaffected():
    good = outputs({"schema_valid": 1.0, "refused": 0.0, "budget_overrun_pct": 0.2,
                    "transit_infeasibility_count": 0.0}, passed=True)
    assert budget_respected({}, good)["score"] == pytest.approx(0.8)
    assert transit_feasible({}, good)["score"] == 1.0
    assert correctly_refused({}, good, reference_outputs={"should_refuse": False})["score"] == 1.0


def test_a_crashed_arm_cannot_beat_a_working_one_on_any_evaluator():
    """The property that actually matters for comparing experiment arms."""
    working = outputs({"schema_valid": 1.0, "refused": 0.0, "hard_pass_rate": 1.0,
                       "soft_pass_rate": 1.0, "must_do_coverage": 1.0, "grounded_pct": 1.0,
                       "budget_overrun_pct": 0.0, "transit_infeasibility_count": 0.0},
                      passed=True, check_calls=1)
    reference = {"should_refuse": False}
    for evaluator in CODE_EVALUATORS:
        dead = evaluator({}, crashed(), reference_outputs=reference)["score"]
        alive = evaluator({}, working, reference_outputs=reference)["score"]
        assert dead <= alive, f"{evaluator({}, working, reference_outputs=reference)['key']}: crash scored higher"


def test_malformed_output_counts_as_no_output():
    """Schema failure is the same class of nothing as a crash."""
    malformed = outputs({"schema_valid": 0.0, "hard_pass_rate": 0.0})
    assert budget_respected({}, malformed)["score"] == 0.0


def test_metrics_pass_through_unchanged():
    assert hard_constraint_pass_rate({}, outputs({"hard_pass_rate": 0.8}))["score"] == 0.8
    assert schema_valid({}, outputs({"schema_valid": 0.0}))["score"] == 0.0


def test_effort_metrics_expose_work_done():
    """Diagnostics, not quality scores — but the only thing that separates two
    runs that both score a perfect 1.0 while doing 55 and 145 tool calls."""
    from wayfinder.evals.evaluators import tool_calls, verification_calls

    out = outputs(tool_calls=145, verification_calls=92)
    assert tool_calls({}, out)["score"] == 145.0
    assert verification_calls({}, out)["score"] == 92.0


def test_effort_metrics_default_to_zero_on_a_crashed_run():
    from wayfinder.evals.evaluators import tool_calls, verification_calls

    assert tool_calls({}, crashed())["score"] == 0.0
    assert verification_calls({}, crashed())["score"] == 0.0


def test_run_result_counts_tool_calls_from_message_history():
    """The counter must survive the message shapes LangGraph actually returns."""
    from langchain_core.messages import AIMessage

    from wayfinder.agent import AgentConfig, RunResult
    from wayfinder.verify import ConstraintReport

    def call(name, i):
        return {"name": name, "args": {}, "id": f"c{i}", "type": "tool_call"}

    result = RunResult(
        run_dir=Path("."), spec=None, config=AgentConfig(),
        report=ConstraintReport(True, [], [], {}), itinerary=None, payload=None,
        messages=[
            AIMessage(content="", tool_calls=[call("geocode", 1), call("web_search", 2)]),
            AIMessage(content="thinking out loud"),           # no tool calls
            AIMessage(content="", tool_calls=[call("geocode", 3)]),
        ],
    )
    counts = result.tool_calls()
    assert counts == {"geocode": 2, "web_search": 1}
    assert sum(counts.values()) == 3


def test_check_calls_is_diagnostic_not_a_score():
    assert check_calls({}, outputs(check_calls=4))["score"] == 4.0


def test_evaluator_keys_are_unique():
    keys = [e({}, outputs(), reference_outputs={})["key"] for e in CODE_EVALUATORS]
    assert len(keys) == len(set(keys)), "two evaluators writing the same feedback key"


# --------------------------------------------------------------------------
# Experiment matrix
# --------------------------------------------------------------------------


def test_each_arm_differs_from_the_baseline_in_exactly_one_way():
    """Otherwise a column can't be read as 'what did this component buy us?'."""
    from dataclasses import asdict

    from wayfinder.evals.run import BASELINE, EXPERIMENT_MATRIX

    base = asdict(BASELINE)
    for name, config in EXPERIMENT_MATRIX.items():
        if name == "baseline":
            continue
        differences = {k for k, v in asdict(config).items() if base[k] != v}
        assert len(differences) == 1, f"{name} varies {sorted(differences)}"

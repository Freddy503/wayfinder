"""Tests for the dataset and the code evaluators.

Evaluators are pure functions over the output dict, so they're fully testable
offline — which matters, because a broken evaluator doesn't crash. It quietly
returns a plausible number, and every experiment scored with it is wrong in a
way nothing else will reveal.
"""

from __future__ import annotations

from collections import Counter

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
    return {"report": {"metrics": metrics or {}, "violations": []}, **extra}


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


def test_metrics_pass_through_unchanged():
    assert hard_constraint_pass_rate({}, outputs({"hard_pass_rate": 0.8}))["score"] == 0.8
    assert schema_valid({}, outputs({"schema_valid": 0.0}))["score"] == 0.0


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

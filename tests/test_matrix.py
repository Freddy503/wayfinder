"""Phase 5 — making the experiment matrix something you can actually run.

Six arms have existed since the start and none has been swept against the
current system, for two reasons: every arm over all twenty cases with
repetitions is ~22 hours and tens of dollars, and there was no way to narrow it.

The other half is reading the result honestly. Twenty minutes before this was
written I compared two single runs of the same spec and could not tell whether
a change helped — output tokens went up 67% while wall clock fell 9%. A delta
smaller than the spread is not a result, and the table has to say so.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from wayfinder.cli import COST_PER_RUN, SECONDS_PER_RUN, app
from wayfinder.evals.run import COMPARE_KEYS, compare, run_matrix, summarise

runner = CliRunner()


# --------------------------------------------------------------------------
# Scoping — the reason it was never runnable
# --------------------------------------------------------------------------


@pytest.fixture
def spy(monkeypatch):
    calls = []

    def fake(config, **kw):
        calls.append(kw)
        return {"ok": True}

    import wayfinder.evals.run as run_module

    monkeypatch.setattr(run_module, "run_experiment", fake)
    monkeypatch.setattr(run_module, "Client", lambda *a, **kw: object())
    return calls


def test_cases_reach_every_arm(spy):
    run_matrix(["baseline", "no-repair-loop"], cases=["lisbon-tight"])
    assert [c["cases"] for c in spy] == [["lisbon-tight"], ["lisbon-tight"]]


def test_a_split_reaches_every_arm_and_names_the_experiment(spy):
    """So `baseline-infeasible` and `baseline-easy` don't collide in the UI."""
    run_matrix(["baseline", "no-skills"], split="infeasible")
    assert [c["split"] for c in spy] == ["infeasible", "infeasible"]
    assert [c["experiment_prefix"] for c in spy] == ["baseline-infeasible", "no-skills-infeasible"]


def test_without_a_scope_the_prefix_is_just_the_arm(spy):
    run_matrix(["baseline"])
    assert spy[0]["experiment_prefix"] == "baseline"


def test_an_unknown_arm_fails_before_anything_runs(spy):
    with pytest.raises(KeyError, match="nonsense"):
        run_matrix(["baseline", "nonsense"])
    assert spy == [], "nothing should have started"


# --------------------------------------------------------------------------
# One arm failing must not lose the others
# --------------------------------------------------------------------------


def test_a_failing_arm_does_not_abort_the_sweep(monkeypatch):
    """Each arm is hours of work and a separate LangSmith experiment that has
    already succeeded. A comprehension threw all of that away when arm three
    raised."""
    seen = []

    def flaky(config, *, experiment_prefix, **kw):
        seen.append(experiment_prefix)
        if experiment_prefix == "no-skills":
            raise RuntimeError("rate limited")
        return {"ok": experiment_prefix}

    import wayfinder.evals.run as run_module

    monkeypatch.setattr(run_module, "run_experiment", flaky)
    monkeypatch.setattr(run_module, "Client", lambda *a, **kw: object())

    results = run_matrix(["baseline", "no-skills", "no-subagents"])
    assert seen == ["baseline", "no-skills", "no-subagents"], "it kept going"
    assert results["baseline"] == {"ok": "baseline"}
    assert "rate limited" in results["no-skills"]["error"]
    assert results["no-subagents"] == {"ok": "no-subagents"}


# --------------------------------------------------------------------------
# Spread, and knowing when there isn't one
# --------------------------------------------------------------------------


def test_the_mean_and_spread_of_several_samples():
    mean, spread = summarise([1.0, 0.8, 0.9])
    assert mean == pytest.approx(0.9)
    assert spread == pytest.approx(0.1)


def test_one_sample_has_no_spread_which_is_not_the_same_as_stable():
    """`0.00` here means n=1. The table has to say which, or it implies a
    confidence that does not exist."""
    assert summarise([0.9]) == (0.9, 0.0)


def test_nothing_scored_is_not_zero():
    """A blank row must not read as "this arm scored 0.0 on everything"."""
    mean, _ = summarise([])
    assert mean != mean, "NaN, so the table can render it as —"


# --------------------------------------------------------------------------
# The comparison
# --------------------------------------------------------------------------


def fake_client(scores_by_experiment):
    class Fake:
        def list_runs(self, project_name=None, **kw):
            return [SimpleNamespace(id=f"{project_name}-{i}")
                    for i in range(len(next(iter(scores_by_experiment.get(project_name, {"x": []}).values()), [])))]

        def list_feedback(self, run_ids=None):
            experiment = run_ids[0].rsplit("-", 1)[0]
            index = int(run_ids[0].rsplit("-", 1)[1])
            return [
                SimpleNamespace(key=k, score=v[index])
                for k, v in scores_by_experiment.get(experiment, {}).items()
                if index < len(v)
            ]

    return Fake()


def test_a_clear_win_is_reported_as_one():
    client = fake_client({
        "baseline":       {"plan_passes": [1.0, 1.0, 1.0]},
        "no-repair-loop": {"plan_passes": [0.0, 0.0, 0.0]},
    })
    table = compare(["baseline", "no-repair-loop"], client=client)
    verdict = table["no-repair-loop"]["_vs_baseline"]["plan_passes"]
    assert verdict["delta"] == pytest.approx(-1.0)
    assert verdict["meaningful"] is True


def test_a_difference_inside_the_noise_is_not_a_result():
    """The exact trap this phase exists to avoid: two arms that look different
    and aren't."""
    client = fake_client({
        "baseline":       {"plan_passes": [1.0, 0.0, 1.0]},   # mean .67, sd .58
        "no-repair-loop": {"plan_passes": [1.0, 1.0, 0.0]},   # mean .67, sd .58
    })
    table = compare(["baseline", "no-repair-loop"], client=client)
    assert table["no-repair-loop"]["_vs_baseline"]["plan_passes"]["meaningful"] is False


def test_a_small_delta_against_a_wide_spread_is_not_a_result():
    client = fake_client({
        "baseline":       {"plan_passes": [1.0, 0.4, 0.7]},
        "no-repair-loop": {"plan_passes": [0.9, 0.3, 0.6]},
    })
    table = compare(["baseline", "no-repair-loop"], client=client)
    verdict = table["no-repair-loop"]["_vs_baseline"]["plan_passes"]
    assert verdict["delta"] < 0
    assert verdict["meaningful"] is False, "0.1 against a spread that wide is noise"


def test_the_baseline_gets_no_delta_against_itself():
    client = fake_client({"baseline": {"plan_passes": [1.0]}})
    assert "_vs_baseline" not in compare(["baseline"], client=client)["baseline"]


def test_a_missing_experiment_is_a_blank_row_not_a_crash():
    table = compare(["baseline"], client=fake_client({}))
    assert table["baseline"]["_n"] == 0


def test_the_comparison_covers_quality_and_effort():
    """Quality saturates at 1.0 on easy cases; effort is the only thing left
    that separates two arms there."""
    assert "plan_passes" in COMPARE_KEYS
    assert "route_efficiency" in COMPARE_KEYS
    assert "verification_calls" in COMPARE_KEYS


# --------------------------------------------------------------------------
# Saying what it costs before it costs it
# --------------------------------------------------------------------------


def test_the_estimate_is_built_from_measurement():
    """Two Amsterdam runs, 921s and 839s, $0.118 and $0.076."""
    assert 800 <= SECONDS_PER_RUN <= 950
    assert 0.05 <= COST_PER_RUN <= 0.20


def test_it_says_the_run_count_and_the_cost_before_starting(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setenv("WAYFINDER_USER_AGENT", "tests")

    result = runner.invoke(app, [
        "matrix", "--arm", "baseline", "--arm", "no-repair-loop",
        "--split", "tight-budget", "--repetitions", "3",
    ], input="n\n")
    assert "2 arms" in result.output
    assert "18 agentic runs" in result.output, "2 arms x 3 cases x 3 reps"
    assert "$" in result.output


def test_a_big_sweep_stops_and_asks(monkeypatch):
    """Hours of unattended work and real money. Declining must run nothing."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setenv("WAYFINDER_USER_AGENT", "tests")

    ran = []
    import wayfinder.evals.run as run_module

    monkeypatch.setattr(run_module, "run_experiment", lambda *a, **kw: ran.append(kw))
    result = runner.invoke(app, ["matrix", "--repetitions", "3"], input="n\n")
    assert result.exit_code == 1
    assert ran == []


def test_case_and_split_together_are_refused(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    result = runner.invoke(app, ["matrix", "--case", "lisbon-tight", "--split", "easy"])
    assert result.exit_code == 2
    assert "not both" in result.output


def test_an_unknown_split_lists_the_real_ones(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    result = runner.invoke(app, ["matrix", "--split", "nonsense"])
    assert result.exit_code == 2
    assert "infeasible" in result.output


# --------------------------------------------------------------------------
# The contract LangSmith actually enforces
# --------------------------------------------------------------------------


def test_summary_evaluators_are_accepted_by_langsmith():
    """Calling them directly proves nothing about whether LangSmith will take
    them, and my first version failed at exactly this point — after both arms
    had run. `**_: Any` shows up in the signature as a parameter named `_`,
    which is outside the supported set, so every experiment died at the
    summary stage having already spent the runs.
    """
    from langsmith.evaluation._runner import _normalize_summary_evaluator

    from wayfinder.evals.evaluators import SUMMARY_EVALUATORS

    for evaluator in SUMMARY_EVALUATORS:
        _normalize_summary_evaluator(evaluator)      # raises if unacceptable


def test_per_case_evaluators_are_accepted_too():
    """Same check for the thirteen that run on every case. These use `**_`
    and are fine — the per-case wrapper is more permissive than the summary
    one — but that is worth pinning rather than assuming."""
    import langsmith.evaluation.evaluator as evaluator_module

    from wayfinder.evals.evaluators import CODE_EVALUATORS

    for evaluator in CODE_EVALUATORS:
        evaluator_module.DynamicRunEvaluator(evaluator)


def test_no_summary_evaluator_takes_varargs():
    """The specific shape that broke it, pinned so it cannot come back."""
    import inspect

    from wayfinder.evals.evaluators import SUMMARY_EVALUATORS

    supported = {"runs", "examples", "inputs", "outputs", "reference_outputs"}
    for evaluator in SUMMARY_EVALUATORS:
        names = set(inspect.signature(evaluator).parameters)
        assert names <= supported, f"{evaluator.__name__} takes {names - supported}"


def test_the_real_experiment_names_are_read_off_the_results():
    """LangSmith appends a suffix: `experiment_prefix="baseline"` becomes
    `baseline-f1ffe9b0`, so looking scores up by prefix finds nothing and
    prints an empty table after hours of runs. That is exactly what the first
    dry run did — 70 minutes of agentic work, "runs scored: 0"."""
    from wayfinder.evals.run import experiment_names

    results = {
        "baseline": SimpleNamespace(experiment_name="baseline-f1ffe9b0"),
        "no-repair-loop": SimpleNamespace(experiment_name="no-repair-loop-1d9c78e5"),
    }
    assert experiment_names(results) == {
        "baseline": "baseline-f1ffe9b0",
        "no-repair-loop": "no-repair-loop-1d9c78e5",
    }


def test_an_arm_with_no_name_is_left_out_rather_than_guessed():
    """A failed arm has an error dict, not a result object. Guessing its name
    would send the lookup at a project that does not exist."""
    from wayfinder.evals.run import experiment_names

    results = {
        "baseline": SimpleNamespace(experiment_name="baseline-abc"),
        "no-skills": {"error": "RuntimeError: rate limited"},
    }
    assert experiment_names(results) == {"baseline": "baseline-abc"}

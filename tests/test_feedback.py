"""Code-evaluator scores attached to an ordinary run's trace.

These numbers existed only inside a dataset experiment. Plan a real trip and
LangSmith showed the trace, the tokens and the latency, but nothing about
whether the plan was any good — the one thing this project can decide in pure
Python was the one thing missing from the UI you actually look at.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.conftest import item, make_itinerary, make_spec
from wayfinder.evals import feedback
from wayfinder.verify import check_payload


def result_for(payload, spec=None, *, error=None, check_calls=2):
    """A `RunResult`-shaped object, without running an agent."""
    spec = spec or make_spec()
    report = check_payload(spec, payload) if payload else _empty_report(error)
    itinerary = None
    if payload:
        from wayfinder.schema import Itinerary

        try:
            itinerary = Itinerary.model_validate(payload)
        except Exception:
            itinerary = None
    return SimpleNamespace(
        spec=spec, report=report, itinerary=itinerary, payload=payload,
        error=error, check_calls=check_calls, run_dir="runs/x",
        messages=[], tool_calls=lambda: {"geocode_all": 1, "web_search": 4},
        trace_id=None,
    )


def _empty_report(error):
    from wayfinder.agent import _missing_itinerary_report

    return _missing_itinerary_report(error)


@pytest.fixture
def clean():
    return make_itinerary([
        item("10:00", "12:00", "Belfry", cost=12.0),
    ]).model_dump(mode="json")


# --------------------------------------------------------------------------
# The scores
# --------------------------------------------------------------------------


def test_a_good_run_scores_every_code_evaluator(clean):
    scores = feedback.scores_for(result_for(clean))
    for key in feedback.QUALITY_KEYS:
        assert key in scores, f"{key} missing"
    assert scores["plan_passes"] == 1.0
    assert scores["schema_valid"] == 1.0


def test_effort_is_reported_alongside_quality(clean):
    """Quality saturates at 1.0 on anything easy; effort is what separates
    two runs that both passed."""
    scores = feedback.scores_for(result_for(clean, check_calls=3))
    assert scores["check_calls"] == 3.0
    assert scores["tool_calls"] == 5.0
    assert scores["verification_calls"] == 1.0, "a batch is one turn"


def test_a_crashed_run_does_not_score_as_perfect():
    """The bug that made three whole experiments useless: with no itinerary
    there is no budget to overrun and no infeasible leg, so the absence read
    as success and a crash topped those columns."""
    scores = feedback.scores_for(result_for(None, error="BadRequestError: nope"))
    assert scores["plan_passes"] == 0.0
    assert scores["schema_valid"] == 0.0
    assert scores["budget_respected"] == 0.0
    assert scores["transit_feasible"] == 0.0
    assert scores["correctly_refused"] == 0.0


def test_a_real_trip_is_expected_to_be_plannable(clean):
    """A dataset case carries `should_refuse`; a trip you actually typed does
    not. Assuming it's satisfiable is the honest default — you asked for it."""
    scores = feedback.scores_for(result_for(clean))
    assert scores["correctly_refused"] == 1.0


def test_the_dataset_expectation_still_wins_when_given(clean):
    scores = feedback.scores_for(result_for(clean), should_refuse=True)
    assert scores["correctly_refused"] == 0.0, "it planned a trip it should have refused"


def test_one_broken_evaluator_does_not_lose_the_rest(clean, monkeypatch):
    def explode(*a, **kw):
        raise RuntimeError("boom")

    explode.__name__ = "explode"
    from wayfinder.evals import evaluators

    monkeypatch.setattr(
        evaluators, "CODE_EVALUATORS", [explode, evaluators.plan_passes], raising=True
    )
    monkeypatch.setattr(feedback, "scores_for", feedback.scores_for)
    scores = feedback.scores_for(result_for(clean))
    assert scores == {"plan_passes": 1.0}


# --------------------------------------------------------------------------
# Sending them
# --------------------------------------------------------------------------


def test_scores_are_posted_against_the_run(clean, monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    sent = []

    class FakeClient:
        def create_feedback(self, run_id, **kw):
            sent.append((str(run_id), kw["key"], kw["score"]))

    import langsmith

    monkeypatch.setattr(langsmith, "Client", lambda *a, **kw: FakeClient())
    run_id = feedback.new_run_id()
    scores = feedback.record(run_id, result_for(clean))

    assert {k for _, k, _ in sent} == set(scores)
    assert all(rid == str(run_id) for rid, _, _ in sent)


def test_feedback_is_marked_as_machine_generated(clean, monkeypatch):
    """So it sorts apart from anything you later thumbs-up by hand."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    captured = {}

    class FakeClient:
        def create_feedback(self, run_id, **kw):
            captured.update(kw)

    import langsmith

    monkeypatch.setattr(langsmith, "Client", lambda *a, **kw: FakeClient())
    feedback.record(feedback.new_run_id(), result_for(clean))
    assert captured["feedback_source_type"] == "model"


def test_a_langsmith_outage_does_not_lose_the_trip(clean, monkeypatch):
    """Tracing is observability. A trip that planned correctly but couldn't
    reach LangSmith is a successful trip."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")

    class FakeClient:
        def create_feedback(self, *a, **kw):
            raise ConnectionError("langsmith is down")

    import langsmith

    monkeypatch.setattr(langsmith, "Client", lambda *a, **kw: FakeClient())
    scores = feedback.record(feedback.new_run_id(), result_for(clean))
    assert scores["plan_passes"] == 1.0, "the numbers still come back"


def test_nothing_is_sent_without_a_key(clean, monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    def explode(*a, **kw):
        raise AssertionError("should not have built a client")

    import langsmith

    monkeypatch.setattr(langsmith, "Client", explode)
    assert feedback.record(feedback.new_run_id(), result_for(clean))["plan_passes"] == 1.0


def test_no_run_id_means_no_send(clean, monkeypatch):
    import langsmith

    monkeypatch.setattr(
        langsmith, "Client", lambda *a, **kw: (_ for _ in ()).throw(AssertionError())
    )
    assert feedback.record(None, result_for(clean))


# --------------------------------------------------------------------------
# One implementation, two callers
# --------------------------------------------------------------------------


def test_the_trace_scores_and_the_experiment_read_the_same_shape(clean):
    """`outputs_for` is shared by the dataset target and this path. Two
    functions computing "the same" scores is how an experiment and a real run
    start quietly disagreeing."""
    from wayfinder.evals.run import outputs_for

    outputs = outputs_for(result_for(clean))
    for key in ("passed", "report", "tool_calls", "verification_calls", "check_calls"):
        assert key in outputs


def test_both_run_paths_fix_a_trace_id_up_front():
    """Feedback is posted against a run id, and the tracer generates one too
    late to be useful — by then nothing has said which run it was."""
    import inspect

    from wayfinder import agent, server

    assert '"run_id": trace_id' in inspect.getsource(server.RunSession._run)
    assert '"run_id": trace_id' in inspect.getsource(agent.plan_trip)


def test_the_summary_line_is_stable_and_readable(clean):
    scores = feedback.scores_for(result_for(clean))
    line = feedback.summary(scores)
    assert line.startswith("plan_passes=1")
    assert "check_calls" not in line, "the console line is quality, not effort"

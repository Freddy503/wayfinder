"""Splits, summary evaluators and the review queue.

Everything here maps to a LangSmith concept the project had a use for and
wasn't using: dataset splits, experiment-level scores, and human review
feeding back into the dataset.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from wayfinder.evals import review
from wayfinder.evals.datasets import load_cases, splits, sync_dataset
from wayfinder.evals.evaluators import (
    SUMMARY_EVALUATORS,
    crash_free_rate,
    median_verification_calls,
    refusal_accuracy,
)


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


class FakeClient:
    def __init__(self):
        self.created = []

    def has_dataset(self, dataset_name):
        return False

    def create_dataset(self, dataset_name, description=""):
        return SimpleNamespace(id="ds-1")

    def create_examples(self, dataset_id=None, examples=None, **kw):
        self.created = examples


def test_every_case_is_pushed_into_its_category_split():
    client = FakeClient()
    sync_dataset(client=client)
    by_split = {}
    for example in client.created:
        for name in example["split"]:
            by_split.setdefault(name, 0)
            by_split[name] += 1
    assert by_split == {k: len(v) for k, v in splits().items()}


def test_split_is_a_field_not_a_metadata_key():
    """`metadata["dataset_split"]` is what the API reads *back*. Writing it on
    create is silently ignored, and every example stays in "base" — which is
    exactly what happened the first time."""
    client = FakeClient()
    sync_dataset(client=client)
    example = client.created[0]
    assert "split" in example
    assert "dataset_split" not in example["metadata"]


def test_the_splits_match_the_categories_in_the_file():
    grouped = splits()
    assert sum(len(v) for v in grouped.values()) == len(load_cases())
    for expected in ("easy", "infeasible", "tight-budget", "constraint-dense"):
        assert expected in grouped


def test_selecting_a_split_returns_only_its_examples():
    from wayfinder.evals.run import select_split

    class Listing:
        def list_examples(self, dataset_name=None):
            return [
                SimpleNamespace(metadata={"dataset_split": ["easy"], "name": "a"}),
                SimpleNamespace(metadata={"dataset_split": ["infeasible"], "name": "b"}),
            ]

    picked = select_split(Listing(), "wayfinder-trips", "infeasible")
    assert [e.metadata["name"] for e in picked] == ["b"]


def test_an_unknown_split_says_which_ones_exist():
    from wayfinder.evals.run import select_split

    class Listing:
        def list_examples(self, dataset_name=None):
            return [SimpleNamespace(metadata={"dataset_split": ["easy"]})]

    with pytest.raises(KeyError, match="easy"):
        select_split(Listing(), "wayfinder-trips", "nonsense")


# --------------------------------------------------------------------------
# Summary evaluators
# --------------------------------------------------------------------------


def out(*, schema_valid=1.0, refused=0.0, verification=10):
    return {
        "report": {"metrics": {"schema_valid": schema_valid, "refused": refused}},
        "verification_calls": verification,
    }


def test_crash_free_rate_counts_runs_that_produced_something():
    result = crash_free_rate([out(), out(schema_valid=0.0), out()], [{}, {}, {}])
    assert result["key"] == "crash_free_rate"
    assert result["score"] == pytest.approx(2 / 3)


def test_refusal_accuracy_looks_only_at_cases_that_should_refuse():
    """The per-case version is deliberately scored on all twenty so an agent
    that refuses everything can't top the infeasible subset — which makes it a
    blended number. This answers the narrow question."""
    outputs = [out(refused=1.0), out(refused=0.0), out(refused=0.0)]
    refs = [{"should_refuse": True}, {"should_refuse": True}, {"should_refuse": False}]
    assert refusal_accuracy(outputs, refs)["score"] == pytest.approx(0.5)


def test_refusal_accuracy_is_none_when_nothing_should_be_refused():
    """A subset score over an empty subset is not zero — reporting 0.0 would
    read as total failure on a split that simply doesn't test refusal."""
    result = refusal_accuracy([out()], [{"should_refuse": False}])
    assert result["score"] is None


def test_a_crashed_run_cannot_count_as_a_correct_refusal():
    outputs = [out(schema_valid=0.0, refused=1.0)]
    assert refusal_accuracy(outputs, [{"should_refuse": True}])["score"] == 0.0


def test_verification_calls_are_summarised_by_median():
    """One runaway case would otherwise set the number, and this is the column
    the batch tools exist to move."""
    outputs = [out(verification=5), out(verification=6), out(verification=200)]
    assert median_verification_calls(outputs, [{}] * 3)["score"] == 6


def test_summary_evaluators_survive_an_empty_experiment():
    for evaluator in SUMMARY_EVALUATORS:
        evaluator([], [])


def test_the_runner_passes_summary_evaluators_to_langsmith():
    import inspect

    from wayfinder.evals.run import run_experiment

    assert "summary_evaluators=SUMMARY_EVALUATORS" in inspect.getsource(run_experiment)


# --------------------------------------------------------------------------
# Review queue
# --------------------------------------------------------------------------


def result_for(*, passed=True, error=None, soft=0, feasible=True):
    violations = [SimpleNamespace(severity="soft", check="pace") for _ in range(soft)]
    return SimpleNamespace(
        report=SimpleNamespace(passed=passed, violations=violations),
        itinerary=SimpleNamespace(feasible=feasible),
        error=error,
    )


@pytest.mark.parametrize(
    ("result", "expected", "because"),
    [
        (result_for(), False, "clean pass — the code already knows"),
        (result_for(passed=False), True, "failed a hard constraint"),
        (result_for(error="BadRequest"), True, "errored"),
        (result_for(soft=2), True, "passed with warnings"),
        (result_for(feasible=False), True, "declared infeasible"),
    ],
)
def test_only_runs_worth_a_human_go_to_the_queue(result, expected, because):
    """Queueing everything makes the queue worthless: a hundred clean passes
    bury the one that needs an opinion."""
    queue_it, why = review.worth_reviewing(result)
    assert queue_it is expected, f"{because}: {why}"
    assert why


def test_a_run_with_no_report_at_all_is_queued():
    assert review.worth_reviewing(SimpleNamespace(report=None))[0] is True


def test_nothing_is_queued_without_a_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert review.send_for_review("run-1") is False
    assert review.ensure_queue() is None


def test_a_placeholder_key_is_not_configured(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "PASTE_HERE")
    assert review.is_configured() is False


def test_the_queue_is_found_by_name_before_being_created(monkeypatch):
    """So this works on a fresh checkout and after someone deletes it in the
    UI, without storing an id anywhere."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    created = []

    class Client:
        def list_annotation_queues(self, name=None, limit=None):
            return [SimpleNamespace(id="q-existing")]

        def create_annotation_queue(self, **kw):
            created.append(kw)
            return SimpleNamespace(id="q-new")

    assert review.ensure_queue(client=Client()) == "q-existing"
    assert created == []


def test_the_queue_is_created_with_a_rubric_about_the_trip(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    captured = {}

    class Client:
        def list_annotation_queues(self, name=None, limit=None):
            return []

        def create_annotation_queue(self, **kw):
            captured.update(kw)
            return SimpleNamespace(id="q-new")

    review.ensure_queue(client=Client())
    rubric = captured["rubric_instructions"].lower()
    assert "would you actually take this trip" in rubric
    # A reviewer looking at a plan should not need to know what a subagent is.
    assert "subagent" not in rubric and "evaluator" not in rubric


def test_a_langsmith_outage_does_not_lose_the_trip(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")

    class Client:
        def list_annotation_queues(self, **kw):
            raise ConnectionError("down")

    import langsmith

    monkeypatch.setattr(langsmith, "Client", lambda *a, **kw: Client())
    assert review.send_for_review("run-1") is False


def test_promoting_a_run_carries_the_spec_and_the_reason(monkeypatch):
    """Six months on, "why is this case here?" is the question you will have,
    and the spec alone never answers it."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")
    captured = {}

    class Client:
        def read_run(self, run_id):
            return SimpleNamespace(inputs={"spec": {"destination": "Ghent"}})

        def create_examples(self, dataset_name=None, examples=None):
            captured["example"] = examples[0]

    import langsmith

    monkeypatch.setattr(langsmith, "Client", lambda *a, **kw: Client())
    assert review.promote_to_dataset("run-1", "three restaurants were shut", name="ghent-shut")

    example = captured["example"]
    assert example["inputs"]["spec"]["destination"] == "Ghent"
    assert example["metadata"]["why"] == "three restaurants were shut"
    assert example["metadata"]["promoted_from_run"] == "run-1"


def test_promoting_a_run_with_no_spec_fails_quietly(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")

    class Client:
        def read_run(self, run_id):
            return SimpleNamespace(inputs={})

    import langsmith

    monkeypatch.setattr(langsmith, "Client", lambda *a, **kw: Client())
    assert review.promote_to_dataset("run-1", "note", name="x") is False

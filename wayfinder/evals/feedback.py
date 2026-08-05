"""Attach the code evaluators' scores to an ordinary run's trace.

Until now those numbers only existed inside a dataset experiment. Plan a real
trip — from the CLI or the browser — and LangSmith showed you the trace, the
tokens and the latency, but nothing about whether the plan was any *good*. The
one thing this project can decide in pure Python was the one thing missing from
the UI you actually look at.

So every run now posts its scores back as feedback on its own root trace. The
same evaluator functions, over the same output shape the experiment builds —
not a parallel reimplementation that drifts. In the LangSmith UI they show up
as feedback on the run, filterable and chartable, which is what turns "here are
my traces" into "here is how the system has been doing this week".

Nothing here is allowed to break a run. A trip that planned correctly but
couldn't reach LangSmith is a successful trip.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

#: Feedback that says how good the plan was, rather than how much work it took.
#: Both get posted; this list only drives what `summary()` prints.
QUALITY_KEYS = (
    "plan_passes",
    "hard_constraint_pass_rate",
    "soft_constraint_pass_rate",
    "schema_valid",
    "budget_respected",
    "transit_feasible",
    "must_do_coverage",
    "grounded_pct",
    "correctly_refused",
)


def new_run_id() -> uuid.UUID:
    """An id to hand LangChain so the root trace is findable afterwards.

    Feedback is posted against a run id, and the tracer generates one too late
    to be useful — by the time the graph returns, nothing has told us which run
    it was. Passing `run_id` in the config fixes the id up front.
    """
    return uuid.uuid4()


def scores_for(result: Any, *, should_refuse: bool | None = None) -> dict[str, float]:
    """Run the code evaluators over a finished run.

    `should_refuse` is the dataset's expectation, which a real trip doesn't
    have. Absent, it's taken from the spec — a trip nobody flagged as
    impossible is one that ought to be plannable, and scoring `correctly_
    refused` against that is the honest default.
    """
    from wayfinder.evals.evaluators import CODE_EVALUATORS
    from wayfinder.evals.run import outputs_for

    outputs = outputs_for(result)
    if should_refuse is None:
        should_refuse = bool(getattr(result.spec, "should_refuse", False))
    reference = {"should_refuse": should_refuse}

    scores: dict[str, float] = {}
    for evaluator in CODE_EVALUATORS:
        try:
            verdict = evaluator({}, outputs, reference_outputs=reference)
        except Exception:  # noqa: BLE001 — one broken evaluator, not no scores
            logger.debug("evaluator %s failed", getattr(evaluator, "__name__", "?"),
                         exc_info=True)
            continue
        if isinstance(verdict, dict) and verdict.get("score") is not None:
            scores[verdict["key"]] = float(verdict["score"])
    return scores


def record(run_id: Any, result: Any, *, should_refuse: bool | None = None) -> dict[str, float]:
    """Post the code evaluators' scores as feedback on `run_id`.

    Returns what it computed even when sending fails, so a caller can still
    print the numbers. Never raises: tracing is observability, and losing it
    must not lose the trip.
    """
    scores = scores_for(result, should_refuse=should_refuse)
    if not scores or run_id is None:
        return scores

    try:
        import os

        if not os.environ.get("LANGSMITH_API_KEY", "").strip():
            return scores

        from langsmith import Client

        client = Client()
        # `session_id` — the project the run lives in. Without it the SDK
        # warns that feedback creation "is deprecated and will stop working",
        # because it has to go looking for the run instead of being handed it.
        #
        # Not `project_id`: that names the same thing and the SDK raises
        # "project_id cannot be provided if run_id or trace_id is provided".
        # Worth stating because the two parameters sit next to each other in
        # the signature and only one of them works here.
        session_id = _session_id(client, os.environ.get("LANGSMITH_PROJECT", "").strip())
        for key, score in scores.items():
            client.create_feedback(
                run_id,
                key=key,
                score=score,
                **({"session_id": session_id} if session_id else {}),
                # Marks these as machine-generated, so they sort apart from
                # anything you later thumbs-up by hand in the UI.
                feedback_source_type="model",
                comment="wayfinder code evaluator",
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not send feedback to LangSmith: %s", exc)

    return scores


#: One lookup per process. The project id never changes within a run, and
#: resolving it per score would be twelve extra round trips per trip planned.
_project_cache: dict[str, Any] = {}


def _session_id(client: Any, name: str) -> Any:
    """The id of the project runs are traced into, or None if unknown.

    None is fine: the send still works, it just costs the deprecation warning
    the argument exists to avoid. A project that doesn't exist yet must not
    cost you the scores.
    """
    if not name:
        return None
    if name not in _project_cache:
        try:
            _project_cache[name] = client.read_project(project_name=name).id
        except Exception:  # noqa: BLE001 — the project may not exist yet
            _project_cache[name] = None
    return _project_cache[name]


def summary(scores: dict[str, float]) -> str:
    """One line for the console: the quality scores, in a fixed order."""
    parts = [f"{k}={scores[k]:g}" for k in QUALITY_KEYS if k in scores]
    return "  ".join(parts)


__all__ = ["QUALITY_KEYS", "new_run_id", "record", "scores_for", "summary"]

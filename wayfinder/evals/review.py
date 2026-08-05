"""Human review of real runs, and promoting the interesting ones into the set.

The loop the dataset can't close on its own. `verify.py` decides everything
decidable in Python, and the judges guess at the rest — but neither knows that
the "quiet canalside walk" was next to a building site, or that three of the
five restaurants were shut for refurbishment. Only the person who went knows
that, and until now there was nowhere to put it.

So: route finished runs into a LangSmith annotation queue, review them there
with a score and a note, and promote the ones that went wrong into
`wayfinder-trips` as new cases. That is what turns a fixed dataset into one
that grows from use — and a regression suite that only contains cases someone
imagined up front is a suite that keeps passing while the product gets worse.

Nothing here is required to plan a trip. Every function degrades to a no-op
without a LangSmith key.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_QUEUE = "wayfinder-review"

#: What a reviewer is being asked. Shown in the LangSmith queue UI, so it wants
#: to be about the trip rather than about the harness — a person reviewing a
#: plan should not need to know what a subagent is.
RUBRIC = """\
Would you actually take this trip?

Look past whether the checks passed — that part is already decided in code.
Judge what the code cannot: are these places worth going to, in an order that
makes sense, with enough room to enjoy them? Would a local wince at any of it?

Flag anything the checker had no way to know: a venue that has closed, hours
that are wrong, a "short walk" that is uphill in August, a neighbourhood
recommendation that misses the point of the preference it was answering.
"""


def is_configured() -> bool:
    key = os.environ.get("LANGSMITH_API_KEY", "").strip()
    return bool(key) and not key.lower().startswith("paste")


def ensure_queue(name: str = DEFAULT_QUEUE, client: Any = None) -> Any:
    """The queue's id, creating it the first time.

    Looked up by name rather than stored, so this works on a fresh checkout
    and after someone deletes the queue in the UI.
    """
    if not is_configured():
        return None
    from langsmith import Client

    client = client or Client()
    for queue in client.list_annotation_queues(name=name, limit=1):
        return queue.id
    return client.create_annotation_queue(
        name=name,
        description="Real Wayfinder trips awaiting a human opinion.",
        rubric_instructions=RUBRIC,
    ).id


def send_for_review(run_id: Any, name: str = DEFAULT_QUEUE, client: Any = None) -> bool:
    """Queue one finished run for a human to look at.

    Returns whether it was queued, and never raises: a trip that planned fine
    but could not be enqueued is still a planned trip.
    """
    if not is_configured() or run_id is None:
        return False
    try:
        from langsmith import Client

        client = client or Client()
        queue_id = ensure_queue(name, client)
        if queue_id is None:
            return False
        client.add_runs_to_annotation_queue(queue_id, run_ids=[str(run_id)])
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not queue run for review: %s", exc)
        return False
    return True


def worth_reviewing(result: Any) -> tuple[bool, str]:
    """Should this run go to the queue?

    Queueing everything makes the queue worthless — a hundred passing trips
    with nothing to say buries the one that needs a person. So: anything that
    failed, refused, or scraped through with soft warnings. A clean pass with
    no caveats is exactly the case where the code already knows the answer.
    """
    report = getattr(result, "report", None)
    if report is None:
        return True, "no report — something went badly wrong"
    if getattr(result, "error", None):
        return True, f"run errored: {result.error}"
    if not report.passed:
        return True, "failed a hard constraint"

    itinerary = getattr(result, "itinerary", None)
    if itinerary is not None and not itinerary.feasible:
        return True, "declared infeasible — worth confirming it really is"

    soft = [v for v in report.violations if v.severity == "soft"]
    if soft:
        return True, f"passed with {len(soft)} quality warning(s)"
    return False, "clean pass — the checker already knows"


def promote_to_dataset(
    run_id: Any,
    note: str,
    *,
    name: str,
    category: str = "from-review",
    dataset_name: str | None = None,
    client: Any = None,
) -> bool:
    """Add a reviewed run's spec to the dataset as a new case.

    The point of reviewing at all: a trip that disappointed becomes a case the
    suite will keep checking. The reviewer's note goes in the metadata, because
    six months on "why is this case here?" is the question you will have, and
    the spec alone never answers it.
    """
    if not is_configured():
        return False
    try:
        from langsmith import Client

        from wayfinder.evals.datasets import DEFAULT_DATASET_NAME

        client = client or Client()
        run = client.read_run(str(run_id))
        spec = (run.inputs or {}).get("spec")
        if not spec:
            logger.warning("run %s has no spec to promote", run_id)
            return False

        client.create_examples(
            dataset_name=dataset_name or DEFAULT_DATASET_NAME,
            examples=[{
                "inputs": {"name": name, "category": category, "spec": spec},
                "outputs": {"should_refuse": False, "category": category},
                "metadata": {
                    "name": name,
                    "category": category,
                    "dataset_split": [category],
                    "promoted_from_run": str(run_id),
                    "why": note,
                },
            }],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not promote run %s: %s", run_id, exc)
        return False
    return True


__all__ = [
    "DEFAULT_QUEUE",
    "RUBRIC",
    "ensure_queue",
    "is_configured",
    "promote_to_dataset",
    "send_for_review",
    "worth_reviewing",
]

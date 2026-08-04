"""Turn a rambling monologue into trip requirements.

Speech mode's job is requirements capture, not conversation: the traveller
talks (or types) freely, and every few seconds the whole transcript is
re-extracted into the same `TripSpec` fields the form edits. Stateless
re-extraction over the full transcript beats incremental patching — people
contradict themselves mid-ramble ("say four hundred… no, make it six"), and
only the full text lets the later statement win.

The model call is injectable so the merge logic is testable without a key.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from wayfinder.models import DEFAULT_MODEL

#: Parsing a monologue into a dozen fields is a much smaller job than planning
#: a trip, so it runs on the cheap end of the same provider the planner uses —
#: not on a second provider whose key you'd have to keep funded just for this.
EXTRACT_MODEL = DEFAULT_MODEL

_SYSTEM = """\
You extract trip-planning requirements from a rambling, unstructured \
monologue. The text is a live transcript: it may contain filler, corrections \
and contradictions. When statements conflict, the LATER one wins — "four \
hundred euros… actually let's say six hundred" means 600.

Extract only what was actually said or clearly implied. Leave everything else \
null. Do not invent dates, budgets or preferences the speaker never gave. \
Resolve relative dates against today's date, which is given in the prompt.

`follow_up_question`: the single most useful question to ask next, phrased \
conversationally, targeting the most important missing requirement. Null once \
destination, dates and budget are all known.
"""


class ExtractedRequirements(BaseModel):
    """What the model may fill in. Everything optional — null means unsaid."""

    destination: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    adults: int | None = Field(default=None, ge=1)
    children: int | None = Field(default=None, ge=0)
    budget_total: float | None = Field(default=None, gt=0)
    currency: str | None = Field(default=None, min_length=3, max_length=3)
    pace: Literal["relaxed", "standard", "packed"] | None = None
    earliest_start: str | None = Field(default=None, pattern=r"^\d{2}:\d{2}$")
    latest_end: str | None = Field(default=None, pattern=r"^\d{2}:\d{2}$")
    max_transit_minutes: int | None = Field(default=None, gt=0)
    min_free_block_minutes: int | None = Field(default=None, gt=0)
    max_walk_km_per_day: float | None = Field(default=None, gt=0)
    required_meals: list[Literal["breakfast", "lunch", "dinner"]] | None = None
    soft_preferences: list[str] | None = None
    must_do: list[str] | None = None
    dietary: list[str] | None = None
    follow_up_question: str | None = None


#: The three things a plan cannot start without.
REQUIRED: tuple[str, ...] = ("destination", "dates", "budget")


def _default_extractor(transcript: str) -> ExtractedRequirements:
    from wayfinder.models import resolve_model

    llm = resolve_model(EXTRACT_MODEL)
    structured = llm.with_structured_output(ExtractedRequirements)
    today = date.today()
    prompt = (
        f"{_SYSTEM}\nToday is {today.isoformat()} ({today:%A}).\n\n"
        f"Transcript so far:\n---\n{transcript}\n---"
    )
    return structured.invoke(prompt)


def extract_requirements(
    transcript: str,
    extractor: Callable[[str], ExtractedRequirements] | None = None,
) -> dict[str, Any]:
    """Extract requirements and report what's still missing.

    Returns `{"fields": {...}, "missing": [...], "follow_up": str|None}` where
    `fields` uses the flat names the browser form understands.
    """
    transcript = transcript.strip()
    if not transcript:
        return {
            "fields": {},
            "missing": list(REQUIRED),
            "follow_up": "Where do you want to go?",
        }

    extracted = (extractor or _default_extractor)(transcript)
    fields: dict[str, Any] = {}

    def put(key: str, value: Any) -> None:
        if value is not None:
            fields[key] = value

    put("destination", extracted.destination)
    put("start", extracted.start_date.isoformat() if extracted.start_date else None)
    put("end", _resolved_end(extracted))
    put("adults", extracted.adults)
    put("children", extracted.children)
    put("budget", extracted.budget_total)
    put("currency", extracted.currency.upper() if extracted.currency else None)
    put("pace", extracted.pace)
    put("early", extracted.earliest_start)
    put("late", extracted.latest_end)
    put("transit", extracted.max_transit_minutes)
    put("free", extracted.min_free_block_minutes)
    put("walk", extracted.max_walk_km_per_day)
    put("meals", extracted.required_meals)
    put("prefs", extracted.soft_preferences)
    put("must", extracted.must_do)
    put("diet", extracted.dietary)

    missing = []
    if "destination" not in fields:
        missing.append("destination")
    if "start" not in fields or "end" not in fields:
        missing.append("dates")
    if "budget" not in fields:
        missing.append("budget")

    return {
        "fields": fields,
        "missing": missing,
        "follow_up": extracted.follow_up_question if missing else None,
    }


def _resolved_end(extracted: ExtractedRequirements) -> str | None:
    """An end date, tolerating "three days in Rome" with only a start named."""
    if extracted.end_date:
        return extracted.end_date.isoformat()
    if extracted.start_date and extracted.end_date is None:
        # A one-day mention is a valid single-day trip rather than missing data.
        return None
    return None


def merge_transcript(previous: str, addition: str) -> str:
    """Append an utterance, keeping the transcript newline-separated."""
    previous, addition = previous.strip(), addition.strip()
    if not previous:
        return addition
    if not addition:
        return previous
    return f"{previous}\n{addition}"


__all__ = [
    "ExtractedRequirements",
    "REQUIRED",
    "extract_requirements",
    "merge_transcript",
]

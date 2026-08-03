"""Typed models for trip specs and itineraries.

The whole project hangs off one decision: the agent's deliverable is
`Itinerary` — a validated JSON document — not prose. Prose can only be judged;
JSON can be *checked*. `verify.py` does the checking, `render.py` turns the
same document into something a human wants to read.

Two kinds of constraint live in a `TripSpec`, and the split is deliberate:

- `constraints` / `budget` / `must_do` / `mobility` are **structured**, so
  `verify.py` can decide pass/fail in pure Python with no model in the loop.
- `soft_preferences` is **free text**, judged by an LLM evaluator.

If you find yourself wanting to express a new requirement, the question to ask
is which of those two piles it belongs in. Anything code can check belongs in
the first.
"""

from __future__ import annotations

from datetime import date, time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Weekday = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
Pace = Literal["relaxed", "standard", "packed"]
ItemKind = Literal["activity", "meal", "transit", "rest", "travel"]
MealSlot = Literal["breakfast", "lunch", "dinner"]
TransitMode = Literal["walk", "transit", "taxi", "bike", "other"]

WEEKDAYS: tuple[Weekday, ...] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

#: Activities per day each pace tolerates, as (min, max). Meals and rest don't
#: count — this is about how much *sightseeing* gets crammed in.
PACE_ACTIVITY_BOUNDS: dict[Pace, tuple[int, int]] = {
    "relaxed": (1, 3),
    "standard": (2, 5),
    "packed": (3, 8),
}


def weekday_of(d: date) -> Weekday:
    """Return the lowercase 3-letter weekday key used by `OpeningHours`."""
    return WEEKDAYS[d.weekday()]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# Trip spec — the input
# --------------------------------------------------------------------------


class DateRange(_Model):
    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> DateRange:
        if self.end < self.start:
            msg = f"end date {self.end} precedes start date {self.start}"
            raise ValueError(msg)
        return self

    def days(self) -> list[date]:
        span = (self.end - self.start).days
        return [date.fromordinal(self.start.toordinal() + offset) for offset in range(span + 1)]


class Party(_Model):
    adults: int = Field(default=2, ge=1)
    children: int = Field(default=0, ge=0)

    @property
    def size(self) -> int:
        return self.adults + self.children


class Budget(_Model):
    currency: str = Field(min_length=3, max_length=3, description="ISO 4217 code, e.g. EUR")
    total: float = Field(gt=0, description="Cap for the whole party across the whole trip")
    excludes: list[str] = Field(
        default_factory=list,
        description="Cost categories the cap does not cover, e.g. ['flights'].",
    )


class Mobility(_Model):
    max_walk_km_per_day: float | None = Field(default=None, gt=0)


class HardConstraints(_Model):
    """Requirements `verify.py` can decide without asking a model.

    Every field here maps to exactly one check. Leave a field unset and its
    check is skipped rather than failed — an unstated constraint is not a
    violated one.
    """

    earliest_start: time | None = Field(
        default=None, description="Nothing scheduled before this local time."
    )
    latest_end: time | None = Field(
        default=None, description="Nothing still running after this local time."
    )
    max_transit_minutes: int | None = Field(
        default=None, gt=0, description="Cap on any single hop between consecutive stops."
    )
    min_free_block_minutes: int | None = Field(
        default=None,
        gt=0,
        description="Each day needs one unscheduled gap at least this long.",
    )
    required_meals: list[MealSlot] = Field(
        default_factory=list, description="Meal slots that must appear on every day."
    )
    max_activities_per_day: int | None = Field(
        default=None,
        gt=0,
        description="Overrides the ceiling implied by `pace` when set.",
    )


class TripSpec(_Model):
    """What the traveller wants. The agent's input; the checker's yardstick."""

    destination: str
    dates: DateRange
    party: Party = Field(default_factory=Party)
    budget: Budget
    pace: Pace = "standard"
    constraints: HardConstraints = Field(default_factory=HardConstraints)
    soft_preferences: list[str] = Field(
        default_factory=list,
        description="Free text, judged by an LLM evaluator rather than by code.",
    )
    must_do: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    dietary: list[str] = Field(default_factory=list)
    mobility: Mobility = Field(default_factory=Mobility)

    #: Dataset flag. True for specs that cannot be satisfied — where the only
    #: correct answer is `feasible=False` with a reason, not a fabricated plan.
    should_refuse: bool = False

    @property
    def slug(self) -> str:
        city = self.destination.split(",")[0].strip().lower().replace(" ", "-")
        return f"{city}-{self.dates.start.isoformat()}"


# --------------------------------------------------------------------------
# Itinerary — the output
# --------------------------------------------------------------------------


class HourRange(_Model):
    open: time
    close: time

    @model_validator(mode="after")
    def _ordered(self) -> HourRange:
        if self.close <= self.open:
            msg = f"closing time {self.close} is not after opening time {self.open}"
            raise ValueError(msg)
        return self

    def covers(self, start: time, end: time) -> bool:
        return self.open <= start and end <= self.close


class OpeningHours(_Model):
    """Per-weekday opening times.

    A weekday **absent** from `week` means *unknown* — the agent didn't find
    out. A weekday present with an **empty list** means *closed*. The two are
    deliberately different: unknown is a soft violation nudging the agent to go
    look, closed is a hard failure if something is scheduled there.
    """

    week: dict[Weekday, list[HourRange]] = Field(default_factory=dict)

    def status(self, day: Weekday) -> Literal["unknown", "closed", "open"]:
        if day not in self.week:
            return "unknown"
        return "open" if self.week[day] else "closed"

    def covers(self, day: Weekday, start: time, end: time) -> bool:
        return any(r.covers(start, end) for r in self.week.get(day, []))


class Rating(_Model):
    """A public review score, recorded with its provenance.

    Ratings come from web-search snippets of Google-review data, not the
    billed Places API — so the source URL is mandatory: a score you can't
    trace is a score you can't trust.
    """

    score: float = Field(ge=0, le=5)
    count: int | None = Field(default=None, ge=0, description="Number of reviews behind it.")
    source: str = Field(min_length=1, description="URL the score was read from.")


class Venue(_Model):
    name: str
    address: str | None = None
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    opening_hours: OpeningHours | None = None
    rating: Rating | None = None

    @property
    def coords(self) -> tuple[float, float] | None:
        if self.lat is None or self.lon is None:
            return None
        return (self.lat, self.lon)


class TransitLeg(_Model):
    """How the traveller gets from the previous item to this one."""

    mode: TransitMode
    minutes: int = Field(ge=0)
    distance_km: float | None = Field(default=None, ge=0)
    note: str | None = None


class Item(_Model):
    start: time
    end: time
    kind: ItemKind
    title: str
    meal_slot: MealSlot | None = Field(
        default=None, description="Required when kind == 'meal'."
    )
    venue: Venue | None = None
    estimated_cost: float = Field(
        default=0.0, ge=0, description="For the whole party, in the itinerary's currency."
    )
    transit_from_previous: TransitLeg | None = None
    sources: list[str] = Field(
        default_factory=list, description="URLs backing the claims made about this item."
    )
    note: str | None = None

    @model_validator(mode="after")
    def _ordered(self) -> Item:
        if self.end <= self.start:
            msg = f"item {self.title!r}: end {self.end} is not after start {self.start}"
            raise ValueError(msg)
        if self.kind == "meal" and self.meal_slot is None:
            msg = f"item {self.title!r}: kind is 'meal' but meal_slot is unset"
            raise ValueError(msg)
        return self

    @property
    def duration_minutes(self) -> int:
        return _minutes(self.end) - _minutes(self.start)


class Day(_Model):
    date: date
    items: list[Item] = Field(default_factory=list)
    summary: str | None = None


class Itinerary(_Model):
    """The agent's deliverable.

    `feasible=False` is a first-class, *correct* outcome. A spec asking for five
    cities in six days on a shoulder-string budget should come back infeasible
    with a reason — not with a plan that quietly ignores half the constraints.
    """

    destination: str
    currency: str = Field(min_length=3, max_length=3)
    days: list[Day] = Field(default_factory=list)
    feasible: bool = True
    infeasibility_reason: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _refusal_has_reason(self) -> Itinerary:
        if not self.feasible and not (self.infeasibility_reason or "").strip():
            msg = "feasible=False requires a non-empty infeasibility_reason"
            raise ValueError(msg)
        return self

    def total_cost(self) -> float:
        return sum(item.estimated_cost for day in self.days for item in day.items)

    def all_items(self) -> list[tuple[Day, Item]]:
        return [(day, item) for day in self.days for item in day.items]


def _minutes(t: time) -> int:
    """Minutes since midnight. Local time only — no trip here crosses a TZ."""
    return t.hour * 60 + t.minute

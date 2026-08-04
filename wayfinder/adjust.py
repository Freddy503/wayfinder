"""Changing the constraints of a trip while it is being planned.

Until now an impossible spec was a dead end: the agent wrote `feasible: false`,
explained why, and the run was over. That is the correct machine answer and a
useless human one — a budget €140 short doesn't mean *don't go to Lisbon*, it
means *spend €140 more, or skip the boat trip*. The traveller knows which.

So a constraint is no longer fixed for the life of a run. `LiveSpec` holds the
current spec and swaps in a new one when the traveller changes their mind; the
checker reads through it, so the next check — and the final verdict — grade
against what was actually agreed rather than what was typed at the start.

Two ways a change starts:

- **The agent asks.** It's stuck, it calls `request_change` with the specific
  shortfall, and the traveller answers. The run resumes where it parked.
- **The traveller volunteers one.** Applied straight away; the agent is told at
  its next `check_itinerary`, which the repair loop calls constantly. No
  interrupt at all — this is the "without the whole run being interrupted"
  case.

Everything here is pure except `LiveSpec`'s single mutation point, so the
interesting logic is testable without an agent, a server, or a key.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import time
from typing import Any

from wayfinder.schema import TripSpec

#: What a traveller is allowed to change mid-run, as
#: `field -> (label, how to describe the new value)`.
#:
#: Deliberately not "any field". Changing the destination or the dates isn't an
#: adjustment, it's a different trip — the research already done would be
#: worthless, so those belong in a new run rather than a patch to this one.
ADJUSTABLE: dict[str, tuple[str, str]] = {
    "budget": ("budget", "total spend"),
    "max_transit_minutes": ("transit cap", "minutes between stops"),
    "earliest_start": ("earliest start", "time of day"),
    "latest_end": ("latest end", "time of day"),
    "min_free_block_minutes": ("downtime", "minutes free per day"),
    "max_walk_km_per_day": ("walking", "km per day"),
    "required_meals": ("required meals", "meals that must be scheduled"),
    "must_do": ("must-dos", "places the trip cannot skip"),
    "pace": ("pace", "how full the days are"),
}


class AdjustmentError(ValueError):
    """A change that cannot be applied, phrased for the traveller."""


def _parse_time(value: Any) -> time | None:
    if value in (None, ""):
        return None
    if isinstance(value, time):
        return value
    text = str(value).strip()
    try:
        hours, _, minutes = text.partition(":")
        return time(int(hours), int(minutes or 0))
    except (TypeError, ValueError) as exc:
        msg = f"{text!r} is not a time of day — use HH:MM, like 10:00."
        raise AdjustmentError(msg) from exc


def _positive(value: Any, label: str, *, allow_none: bool = True) -> Any:
    if value in (None, "") and allow_none:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AdjustmentError(f"{label} must be a number, not {value!r}.") from exc
    if number <= 0:
        raise AdjustmentError(f"{label} must be greater than zero.")
    return number


def apply_changes(spec: TripSpec, changes: dict[str, Any]) -> tuple[TripSpec, list[str]]:
    """Return a new spec with `changes` applied, plus a plain-English summary.

    Pure: the original is untouched, which matters because a rejected change
    must leave the run exactly as it was. Unknown keys raise rather than being
    dropped silently — a change the traveller thinks they made and didn't is
    worse than an error.
    """
    unknown = set(changes) - set(ADJUSTABLE)
    if unknown:
        allowed = ", ".join(sorted(ADJUSTABLE))
        msg = f"cannot change {', '.join(sorted(unknown))} mid-run. Adjustable: {allowed}."
        raise AdjustmentError(msg)

    data = spec.model_dump(mode="json")
    notes: list[str] = []

    if "budget" in changes:
        amount = _positive(changes["budget"], "budget", allow_none=False)
        was = data["budget"]["total"]
        data["budget"]["total"] = amount
        notes.append(f"budget {was:g} → {amount:g} {data['budget']['currency']}")

    if "pace" in changes:
        pace = str(changes["pace"]).strip().lower()
        if pace not in {"relaxed", "standard", "packed"}:
            raise AdjustmentError(f"{pace!r} is not a pace — relaxed, standard or packed.")
        notes.append(f"pace {data['pace']} → {pace}")
        data["pace"] = pace

    if "must_do" in changes:
        items = [str(i).strip() for i in (changes["must_do"] or []) if str(i).strip()]
        dropped = [m for m in data["must_do"] if m not in items]
        added = [m for m in items if m not in data["must_do"]]
        data["must_do"] = items
        if dropped:
            notes.append(f"dropped must-do: {', '.join(dropped)}")
        if added:
            notes.append(f"added must-do: {', '.join(added)}")

    constraints = data.setdefault("constraints", {})
    for key, label in (
        ("max_transit_minutes", "transit cap"),
        ("min_free_block_minutes", "daily downtime"),
    ):
        if key in changes:
            value = _positive(changes[key], label)
            constraints[key] = int(value) if value is not None else None
            notes.append(f"{label} → {'no limit' if value is None else f'{int(value)} min'}")

    for key, label in (("earliest_start", "earliest start"), ("latest_end", "latest end")):
        if key in changes:
            parsed = _parse_time(changes[key])
            constraints[key] = parsed.isoformat(timespec="minutes") if parsed else None
            notes.append(f"{label} → {'any time' if parsed is None else parsed.strftime('%H:%M')}")

    if "required_meals" in changes:
        meals = [str(m).strip().lower() for m in (changes["required_meals"] or [])]
        bad = [m for m in meals if m not in {"breakfast", "lunch", "dinner"}]
        if bad:
            raise AdjustmentError(f"not a meal: {', '.join(bad)}")
        constraints["required_meals"] = meals
        notes.append(f"required meals → {', '.join(meals) if meals else 'none'}")

    if "max_walk_km_per_day" in changes:
        value = _positive(changes["max_walk_km_per_day"], "walking limit")
        data.setdefault("mobility", {})["max_walk_km_per_day"] = value
        notes.append(f"walking → {'no limit' if value is None else f'{value:g} km/day'}")

    try:
        updated = TripSpec.model_validate(data)
    except Exception as exc:  # noqa: BLE001 — pydantic's message is the useful part
        raise AdjustmentError(str(exc)) from exc

    return updated, notes


@dataclass
class LiveSpec:
    """The spec as it stands right now, plus what changed since the agent looked.

    The agent holds a reference to this rather than to a `TripSpec`, so a change
    made at any moment is visible to the very next check without rebuilding the
    graph or restarting the run.

    Locked because the traveller's change arrives on the HTTP thread while the
    agent reads it on the worker thread — a half-applied spec would fail checks
    for reasons nobody could reproduce.
    """

    spec: TripSpec
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _unread: list[str] = field(default_factory=list, repr=False)
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def current(self) -> TripSpec:
        with self._lock:
            return self.spec

    def apply(self, changes: dict[str, Any], source: str = "traveller") -> list[str]:
        """Apply a change. Raises `AdjustmentError` and changes nothing if invalid."""
        with self._lock:
            updated, notes = apply_changes(self.spec, changes)
            if not notes:
                return []
            self.spec = updated
            self._unread.extend(notes)
            self.history.append({"source": source, "changes": changes, "notes": notes})
            return notes

    def take_news(self) -> list[str]:
        """Drain the changes the agent hasn't been told about yet.

        Read-once on purpose: repeating it on every check would have the agent
        re-planning around the same change indefinitely.
        """
        with self._lock:
            news, self._unread = self._unread, []
            return news


def summarise_constraints(spec: TripSpec) -> list[dict[str, Any]]:
    """The spec's live constraints, shaped for the traveller-facing panel.

    Only what is actually set: an unstated constraint isn't a constraint, and
    listing it as "none" invites people to fill it in for the sake of it.
    """
    out: list[dict[str, Any]] = []

    def put(key: str, label: str, value: Any, display: str) -> None:
        out.append({"key": key, "label": label, "value": value, "display": display})

    put("budget", "Budget", spec.budget.total,
        f"{spec.budget.total:g} {spec.budget.currency}"
        + (f" (excl. {', '.join(spec.budget.excludes)})" if spec.budget.excludes else ""))
    put("pace", "Pace", spec.pace, spec.pace)

    c = spec.constraints
    if c.earliest_start:
        put("earliest_start", "Earliest start", c.earliest_start.strftime("%H:%M"),
            f"nothing before {c.earliest_start.strftime('%H:%M')}")
    if c.latest_end:
        put("latest_end", "Latest end", c.latest_end.strftime("%H:%M"),
            f"nothing after {c.latest_end.strftime('%H:%M')}")
    if c.max_transit_minutes:
        put("max_transit_minutes", "Max transit", c.max_transit_minutes,
            f"under {c.max_transit_minutes} min between stops")
    if c.min_free_block_minutes:
        put("min_free_block_minutes", "Downtime", c.min_free_block_minutes,
            f"{c.min_free_block_minutes} min free each day")
    if c.required_meals:
        put("required_meals", "Meals", list(c.required_meals),
            ", ".join(c.required_meals) + " every day")
    if spec.mobility.max_walk_km_per_day:
        put("max_walk_km_per_day", "Walking", spec.mobility.max_walk_km_per_day,
            f"up to {spec.mobility.max_walk_km_per_day:g} km a day")
    if spec.must_do:
        put("must_do", "Must do", list(spec.must_do), ", ".join(spec.must_do))
    return out


__all__ = [
    "ADJUSTABLE",
    "AdjustmentError",
    "LiveSpec",
    "apply_changes",
    "summarise_constraints",
]

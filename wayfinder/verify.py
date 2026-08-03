"""The constraint checker. Pure Python, no model in the loop.

This file is the spec. It exists before the agent does, and everything
downstream trusts it:

- the agent calls it mid-run as the `check_itinerary` tool, and repairs what it
  reports;
- the LangSmith code evaluators read its `metrics` directly.

One implementation, two consumers — so the thing being optimised and the thing
doing the measuring can never drift apart.

Severity is not a mood. **Hard** means the traveller asked for it explicitly or
physics forbids it (you cannot be in two places at once, or inside a closed
museum). **Soft** means it's a heuristic about quality. Only hard violations
make `passed` False.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from langsmith import traceable
from pydantic import ValidationError

from wayfinder.schema import (
    PACE_ACTIVITY_BOUNDS,
    Day,
    Item,
    Itinerary,
    TripSpec,
    weekday_of,
)

Severity = Literal["hard", "soft"]

#: Wheels-down to standing outside the terminal: taxi in, deplane, immigration,
#: bags. Before the airport transfer even begins.
DEPLANE_MINUTES = 45

#: Recommended check-in for an international departure. Deliberately generous —
#: the failure this guards against (missing the flight) is unrecoverable, while
#: the cost of being wrong is an hour of airport café.
CHECKIN_MINUTES = 120


@dataclass(frozen=True)
class Violation:
    check: str
    severity: Severity
    message: str
    where: str | None = None

    def __str__(self) -> str:
        loc = f" [{self.where}]" if self.where else ""
        return f"{self.severity.upper()} {self.check}{loc}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "severity": self.severity,
            "message": self.message,
            "where": self.where,
        }


#: What each check actually verifies, in one line, for a human reading the
#: result. A verdict that only lists failures leaves the reader unable to tell
#: a thorough pass from a shallow one — "no violations" could mean anything
#: from fifteen satisfied constraints to fifteen skipped ones.
CHECK_DESCRIPTIONS: dict[str, str] = {
    "schema_valid": "The itinerary parses and every field has the right shape",
    "currency_match": "Costs are priced in the budget's currency",
    "dates_covered": "One day planned for every date of the trip, and no others",
    "chronology": "Items run in order and never overlap",
    "budget": "Total estimated cost stays within the cap",
    "transit_feasible": "Every move between venues has a leg that fits the gap",
    "opening_hours": "Nothing is scheduled at a closed venue or outside its hours",
    "must_do_coverage": "Everything on the must-do list is scheduled",
    "time_window": "Nothing starts before, or runs past, the stated times",
    "required_meals": "Each day includes the meals that were asked for",
    "mobility": "Daily walking stays within the limit",
    "free_block": "Each day has an unscheduled stretch, measured net of travel",
    "flights_present": "Both flight legs exist when an origin is given",
    "flight_alignment": "The outbound lands by day one; the return leaves no earlier than the last day",
    "arrival_realism": "Nothing starts before the traveller can get in from the airport",
    "departure_realism": "Nothing runs past the moment they must leave for the airport",
    "pace": "Activity count per day is sensible for the stated pace",
    "grounded": "Every venue has a source backing it",
    "no_duplicates": "No venue is scheduled twice",
    "hours_known": "Opening hours were actually established, not assumed",
    "flights_grounded": "Flight times and fares cite a source",
}


@dataclass
class CheckResult:
    name: str
    severity: Severity
    #: A check is skipped when the spec never asked for it. An unstated
    #: constraint is not a violated one, and skipped checks are excluded from
    #: the pass-rate denominator so they can't inflate a score.
    skipped: bool = False
    violations: list[Violation] = field(default_factory=list)
    #: What this check found, in the plan's own numbers — "216 of 600 EUR,
    #: 384 to spare" rather than a bare tick. This is the difference between
    #: a result you can audit and one you have to trust.
    detail: str | None = None

    @property
    def passed(self) -> bool:
        return not self.violations

    @property
    def description(self) -> str:
        return CHECK_DESCRIPTIONS.get(self.name, self.name.replace("_", " "))


@dataclass
class ConstraintReport:
    passed: bool
    violations: list[Violation]
    checks: list[CheckResult]
    metrics: dict[str, float]

    @property
    def hard_violations(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "hard"]

    @property
    def soft_violations(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "soft"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "metrics": self.metrics,
            "violations": [v.to_dict() for v in self.violations],
            "checks": [
                {
                    "name": c.name,
                    "severity": c.severity,
                    "skipped": c.skipped,
                    "passed": c.passed,
                    "violation_count": len(c.violations),
                    "description": c.description,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
        }

    def summary(self) -> str:
        if self.passed and not self.violations:
            return "PASS — all checks clean."
        head = "PASS" if self.passed else "FAIL"
        counts = f"{len(self.hard_violations)} hard, {len(self.soft_violations)} soft"
        lines = [f"{head} — {counts}"]
        lines += [f"  {v}" for v in self.violations]
        return "\n".join(lines)


def _minutes(t) -> int:
    return t.hour * 60 + t.minute


def _fmt(minutes: int) -> str:
    """Minutes-since-midnight back to HH:MM, tolerating spill past midnight."""
    minutes = max(0, minutes)
    return f"{minutes // 60 % 24:02d}:{minutes % 60:02d}"


def _where(day: Day, item: Item | None = None) -> str:
    if item is None:
        return day.date.isoformat()
    return f"{day.date.isoformat()} {item.start.strftime('%H:%M')} {item.title}"


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


@traceable(name="check_itinerary", run_type="tool")
def check_itinerary(spec: TripSpec, itinerary: Itinerary) -> ConstraintReport:
    """Check a parsed itinerary against its spec."""
    checks: list[CheckResult] = []
    metrics: dict[str, float] = {"schema_valid": 1.0}

    if not itinerary.feasible:
        # Declaring the trip impossible is a legitimate answer, and the schema
        # already guarantees a reason accompanies it. There is no plan to
        # check, so checking one would only manufacture noise.
        metrics.update(
            {
                "refused": 1.0,
                "hard_pass_rate": 1.0,
                "soft_pass_rate": 1.0,
                "hard_violation_count": 0.0,
                "total_cost": 0.0,
                "budget_overrun_pct": 0.0,
                "must_do_coverage": 1.0,
                "transit_infeasibility_count": 0.0,
                "grounded_pct": 1.0,
            }
        )
        return ConstraintReport(passed=True, violations=[], checks=[], metrics=metrics)

    metrics["refused"] = 0.0

    checks.append(_check_currency(spec, itinerary))
    checks.append(_check_dates(spec, itinerary))
    checks.append(_check_chronology(itinerary))
    checks.append(_check_budget(spec, itinerary, metrics))
    checks.append(_check_transit(spec, itinerary, metrics))
    checks.append(_check_opening_hours(itinerary, metrics))
    checks.append(_check_must_do(spec, itinerary, metrics))
    checks.append(_check_time_window(spec, itinerary))
    checks.append(_check_required_meals(spec, itinerary))
    checks.append(_check_mobility(spec, itinerary, metrics))
    checks.append(_check_free_block(spec, itinerary))
    checks.append(_check_pace(spec, itinerary))
    checks.append(_check_flights_present(spec, itinerary, metrics))
    checks.append(_check_flight_alignment(spec, itinerary))
    checks.append(_check_arrival_realism(spec, itinerary))
    checks.append(_check_departure_realism(spec, itinerary))
    checks.append(_check_grounded(itinerary, metrics))
    checks.append(_check_duplicates(itinerary))
    checks.append(_check_hours_known(itinerary))
    checks.append(_check_flights_grounded(itinerary))

    violations = [v for c in checks for v in c.violations]
    hard = [c for c in checks if c.severity == "hard" and not c.skipped]
    soft = [c for c in checks if c.severity == "soft" and not c.skipped]

    metrics["hard_pass_rate"] = _rate(hard)
    metrics["soft_pass_rate"] = _rate(soft)
    metrics["hard_violation_count"] = float(
        len([v for v in violations if v.severity == "hard"])
    )

    return ConstraintReport(
        passed=not any(v.severity == "hard" for v in violations),
        violations=violations,
        checks=checks,
        metrics=metrics,
    )


def check_payload(spec: TripSpec, payload: Any) -> ConstraintReport:
    """Check a raw (unparsed) itinerary payload — a dict straight off disk.

    A schema failure is itself the finding, so it comes back as a normal report
    rather than an exception. The agent's repair loop and the evaluators both
    want a `ConstraintReport` either way.
    """
    try:
        itinerary = Itinerary.model_validate(payload)
    except ValidationError as exc:
        violations = [
            Violation(
                check="schema_valid",
                severity="hard",
                message=f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}",
            )
            for err in exc.errors()
        ]
        return ConstraintReport(
            passed=False,
            violations=violations,
            checks=[CheckResult("schema_valid", "hard", violations=violations)],
            metrics={
                "schema_valid": 0.0,
                "hard_pass_rate": 0.0,
                "soft_pass_rate": 0.0,
                "hard_violation_count": float(len(violations)),
            },
        )
    return check_itinerary(spec, itinerary)


def _rate(checks: list[CheckResult]) -> float:
    if not checks:
        return 1.0
    return sum(1 for c in checks if c.passed) / len(checks)


# --------------------------------------------------------------------------
# Hard checks
# --------------------------------------------------------------------------


def _check_currency(spec: TripSpec, itin: Itinerary) -> CheckResult:
    result = CheckResult("currency_match", "hard")
    result.detail = f"priced in {itin.currency.upper()}"
    if itin.currency.upper() != spec.budget.currency.upper():
        result.violations.append(
            Violation(
                "currency_match",
                "hard",
                f"itinerary is priced in {itin.currency} but the budget is in "
                f"{spec.budget.currency}; convert before reporting costs",
            )
        )
    return result


def _check_dates(spec: TripSpec, itin: Itinerary) -> CheckResult:
    result = CheckResult("dates_covered", "hard")
    wanted = set(spec.dates.days())
    got = {d.date for d in itin.days}
    result.detail = (
        f"{len(wanted)} day{'s' if len(wanted) != 1 else ''}, "
        f"{spec.dates.start:%d %b}–{spec.dates.end:%d %b}"
    )
    for missing in sorted(wanted - got):
        result.violations.append(
            Violation("dates_covered", "hard", "no plan for this date", missing.isoformat())
        )
    for extra in sorted(got - wanted):
        result.violations.append(
            Violation(
                "dates_covered", "hard", "date falls outside the trip", extra.isoformat()
            )
        )
    if len(got) != len(itin.days):
        result.violations.append(
            Violation("dates_covered", "hard", "the same date appears more than once")
        )
    return result


def _check_chronology(itin: Itinerary) -> CheckResult:
    """Items within a day must run in order and not overlap.

    Overlap is the quiet way an over-full plan looks fine: schedule two things
    at once and the day fits. It doesn't.
    """
    result = CheckResult("chronology", "hard")
    total = sum(len(d.items) for d in itin.days)
    result.detail = f"{total} items, none overlapping"
    for day in itin.days:
        for prev, curr in zip(day.items, day.items[1:], strict=False):
            if _minutes(curr.start) < _minutes(prev.end):
                result.violations.append(
                    Violation(
                        "chronology",
                        "hard",
                        f"starts at {curr.start:%H:%M} but {prev.title!r} runs until "
                        f"{prev.end:%H:%M}",
                        _where(day, curr),
                    )
                )
    return result


def _check_budget(spec: TripSpec, itin: Itinerary, metrics: dict[str, float]) -> CheckResult:
    result = CheckResult("budget", "hard")
    # Flights count unless the traveller said they're handling them. A trip
    # that fits only because the airfare wasn't counted doesn't fit.
    counts_flights = not any(
        "flight" in exclusion.lower() for exclusion in spec.budget.excludes
    )
    ground = itin.total_cost()
    air = itin.flight_cost() if counts_flights else 0.0
    total = ground + air
    metrics["ground_cost"] = round(ground, 2)
    metrics["flight_cost"] = round(itin.flight_cost(), 2)
    cap = spec.budget.total
    metrics["total_cost"] = round(total, 2)
    metrics["budget_overrun_pct"] = round(max(0.0, (total - cap) / cap), 4)
    air_note = f" (incl. {air:,.0f} airfare)" if air else ""
    result.detail = (
        f"{total:,.0f} of {cap:,.0f} {itin.currency}{air_note} — "
        f"{cap - total:,.0f} to spare" if total <= cap
        else f"{total:,.0f} of {cap:,.0f} {itin.currency}{air_note}"
    )
    if total > cap:
        result.violations.append(
            Violation(
                "budget",
                "hard",
                f"estimated {total:,.0f} {itin.currency} against a cap of "
                f"{cap:,.0f} ({(total - cap) / cap:.0%} over)",
            )
        )
    return result


def _check_transit(spec: TripSpec, itin: Itinerary, metrics: dict[str, float]) -> CheckResult:
    """Consecutive stops must be reachable in the gap between them.

    A missing `transit_from_previous` between two different venues counts as a
    hard violation, not a soft one. Otherwise omitting the leg would be the
    cheapest way to make an impossible day look possible.
    """
    result = CheckResult("transit_feasible", "hard")
    cap = spec.constraints.max_transit_minutes
    infeasible = 0

    for day in itin.days:
        for prev, curr in zip(day.items, day.items[1:], strict=False):
            gap = _minutes(curr.start) - _minutes(prev.end)
            leg = curr.transit_from_previous
            moved = (
                prev.venue is not None
                and curr.venue is not None
                and prev.venue.name.strip().lower() != curr.venue.name.strip().lower()
            )

            if leg is None:
                if moved:
                    infeasible += 1
                    result.violations.append(
                        Violation(
                            "transit_feasible",
                            "hard",
                            f"no transit leg from {prev.venue.name!r}; "
                            "every move between venues needs one",
                            _where(day, curr),
                        )
                    )
                continue

            if leg.minutes > gap:
                infeasible += 1
                result.violations.append(
                    Violation(
                        "transit_feasible",
                        "hard",
                        f"{leg.minutes} min of travel into a {gap} min gap",
                        _where(day, curr),
                    )
                )
            if cap is not None and leg.minutes > cap:
                infeasible += 1
                result.violations.append(
                    Violation(
                        "transit_feasible",
                        "hard",
                        f"{leg.minutes} min hop exceeds the {cap} min cap",
                        _where(day, curr),
                    )
                )

    metrics["transit_infeasibility_count"] = float(infeasible)
    legs = [i.transit_from_previous for _, i in itin.all_items() if i.transit_from_previous]
    if legs:
        longest = max(l.minutes for l in legs)
        cap_note = f" (cap {cap})" if cap else ""
        result.detail = f"{len(legs)} legs, longest {longest} min{cap_note}"
    return result


def _check_opening_hours(itin: Itinerary, metrics: dict[str, float]) -> CheckResult:
    result = CheckResult("opening_hours", "hard")
    for day in itin.days:
        wd = weekday_of(day.date)
        for item in day.items:
            hours = item.venue.opening_hours if item.venue else None
            if hours is None:
                continue
            status = hours.status(wd)
            if status == "closed":
                result.violations.append(
                    Violation(
                        "opening_hours",
                        "hard",
                        f"{item.venue.name} is closed on {wd}",
                        _where(day, item),
                    )
                )
            elif status == "open" and not hours.covers(wd, item.start, item.end):
                windows = ", ".join(
                    f"{r.open:%H:%M}-{r.close:%H:%M}" for r in hours.week[wd]
                )
                result.violations.append(
                    Violation(
                        "opening_hours",
                        "hard",
                        f"scheduled {item.start:%H:%M}-{item.end:%H:%M} but "
                        f"{item.venue.name} is open {windows}",
                        _where(day, item),
                    )
                )
    metrics["opening_hours_violation_count"] = float(len(result.violations))
    checked = sum(
        1 for d, i in itin.all_items()
        if i.venue and i.venue.opening_hours
        and i.venue.opening_hours.status(weekday_of(d.date)) != "unknown"
    )
    result.detail = f"{checked} venues checked against their weekday hours"
    return result


def _check_must_do(spec: TripSpec, itin: Itinerary, metrics: dict[str, float]) -> CheckResult:
    result = CheckResult("must_do_coverage", "hard", skipped=not spec.must_do)
    if result.skipped:
        metrics["must_do_coverage"] = 1.0
        return result

    haystack = [
        f"{item.title} {item.venue.name if item.venue else ''}".lower()
        for _, item in itin.all_items()
    ]
    hits = 0
    for wanted in spec.must_do:
        needle = wanted.strip().lower()
        if any(needle in hay for hay in haystack):
            hits += 1
        else:
            result.violations.append(
                Violation("must_do_coverage", "hard", f"{wanted!r} was never scheduled")
            )
    metrics["must_do_coverage"] = round(hits / len(spec.must_do), 4)
    result.detail = f"{hits} of {len(spec.must_do)} scheduled"
    return result


def _check_time_window(spec: TripSpec, itin: Itinerary) -> CheckResult:
    earliest, latest = spec.constraints.earliest_start, spec.constraints.latest_end
    result = CheckResult("time_window", "hard", skipped=earliest is None and latest is None)
    if result.skipped:
        return result
    bounds = " to ".join(
        f"{t:%H:%M}" for t in (earliest, latest) if t is not None
    )
    result.detail = f"all items within {bounds}"
    for day in itin.days:
        for item in day.items:
            if earliest is not None and _minutes(item.start) < _minutes(earliest):
                result.violations.append(
                    Violation(
                        "time_window",
                        "hard",
                        f"starts {item.start:%H:%M}, before the {earliest:%H:%M} floor",
                        _where(day, item),
                    )
                )
            if latest is not None and _minutes(item.end) > _minutes(latest):
                result.violations.append(
                    Violation(
                        "time_window",
                        "hard",
                        f"runs to {item.end:%H:%M}, past the {latest:%H:%M} ceiling",
                        _where(day, item),
                    )
                )
    return result


def _check_required_meals(spec: TripSpec, itin: Itinerary) -> CheckResult:
    wanted = spec.constraints.required_meals
    result = CheckResult("required_meals", "hard", skipped=not wanted)
    if result.skipped:
        return result
    result.detail = f"{', '.join(wanted)} on all {len(itin.days)} days"
    for day in itin.days:
        present = {i.meal_slot for i in day.items if i.kind == "meal"}
        for slot in wanted:
            if slot not in present:
                result.violations.append(
                    Violation("required_meals", "hard", f"no {slot}", _where(day))
                )
    return result


def _check_mobility(spec: TripSpec, itin: Itinerary, metrics: dict[str, float]) -> CheckResult:
    cap = spec.mobility.max_walk_km_per_day
    result = CheckResult("mobility", "hard", skipped=cap is None)
    peak = 0.0
    for day in itin.days:
        walked = sum(
            item.transit_from_previous.distance_km or 0.0
            for item in day.items
            if item.transit_from_previous and item.transit_from_previous.mode == "walk"
        )
        peak = max(peak, walked)
        if cap is not None and walked > cap:
            result.violations.append(
                Violation(
                    "mobility",
                    "hard",
                    f"{walked:.1f} km on foot against a {cap:.1f} km limit",
                    _where(day),
                )
            )
    metrics["peak_walk_km"] = round(peak, 2)
    if cap is not None:
        result.detail = f"peak {peak:.1f} km of {cap:.1f} km allowed"
    return result


def _check_free_block(spec: TripSpec, itin: Itinerary) -> CheckResult:
    """Each day needs one genuinely unscheduled stretch.

    The gap is measured *between* scheduled items and net of travel — an hour
    that gets eaten by a 55-minute tram ride is not downtime.
    """
    want = spec.constraints.min_free_block_minutes
    result = CheckResult("free_block", "hard", skipped=want is None)
    if result.skipped:
        return result
    shortest = None
    for day in itin.days:
        if len(day.items) < 2:
            continue
        best = 0
        for prev, curr in zip(day.items, day.items[1:], strict=False):
            gap = _minutes(curr.start) - _minutes(prev.end)
            leg = curr.transit_from_previous
            best = max(best, gap - (leg.minutes if leg else 0))
        shortest = best if shortest is None else min(shortest, best)
        if best < want:
            result.violations.append(
                Violation(
                    "free_block",
                    "hard",
                    f"longest unscheduled stretch is {best} min, short of {want} min",
                    _where(day),
                )
            )
    if shortest is not None:
        result.detail = f"tightest day has {shortest} min free, needs {want}"
    return result


# --------------------------------------------------------------------------
# Soft checks
# --------------------------------------------------------------------------


def _check_pace(spec: TripSpec, itin: Itinerary) -> CheckResult:
    """Activity density.

    Hard when the traveller named a number, soft when it's only inferred from
    `pace` — a heuristic shouldn't be able to fail a run on its own.
    """
    explicit = spec.constraints.max_activities_per_day
    severity: Severity = "hard" if explicit is not None else "soft"
    ceiling = explicit if explicit is not None else PACE_ACTIVITY_BOUNDS[spec.pace][1]
    result = CheckResult("pace", severity)
    for day in itin.days:
        count = sum(1 for i in day.items if i.kind == "activity")
        if count > ceiling:
            result.violations.append(
                Violation(
                    "pace",
                    severity,
                    f"{count} activities against a ceiling of {ceiling} "
                    f"({'stated' if explicit else spec.pace + ' pace'})",
                    _where(day),
                )
            )
    return result


def _check_flights_present(
    spec: TripSpec, itin: Itinerary, metrics: dict[str, float]
) -> CheckResult:
    """Both legs must exist once an origin is given."""
    result = CheckResult("flights_present", "hard", skipped=spec.origin is None)
    if result.skipped:
        metrics["flights_planned"] = 1.0 if itin.flights else 0.0
        return result
    for direction in ("outbound", "return"):
        if itin.flight(direction) is None:
            result.violations.append(
                Violation(
                    "flights_present",
                    "hard",
                    f"no {direction} flight from {spec.origin}",
                )
            )
    metrics["flights_planned"] = 0.0 if result.violations else 1.0
    return result


def _check_flight_alignment(spec: TripSpec, itin: Itinerary) -> CheckResult:
    """The outbound has to land by day one; the return can't leave before the end."""
    result = CheckResult("flight_alignment", "hard", skipped=spec.origin is None)
    if result.skipped:
        return result
    first, last = spec.dates.start, spec.dates.end

    outbound = itin.flight("outbound")
    if outbound and outbound.arrival_date > first:
        result.violations.append(
            Violation(
                "flight_alignment",
                "hard",
                f"outbound lands {outbound.arrival_date} but the trip starts {first}",
            )
        )
    ret = itin.flight("return")
    if ret and ret.date < last:
        result.violations.append(
            Violation(
                "flight_alignment",
                "hard",
                f"return departs {ret.date} but the trip runs to {last}",
            )
        )
    return result


def _check_arrival_realism(spec: TripSpec, itin: Itinerary) -> CheckResult:
    """Nothing on the arrival day may start before the traveller can be there.

    The most common way a plausible-looking itinerary is wrong: sightseeing
    scheduled for 09:00 on a day the aircraft lands at 11:30.
    """
    outbound = itin.flight("outbound")
    result = CheckResult(
        "arrival_realism", "hard", skipped=spec.origin is None or outbound is None
    )
    if result.skipped or outbound is None:
        return result

    ready = (
        _minutes(outbound.arrive_time) + DEPLANE_MINUTES + spec.airport_transfer_minutes
    )
    for day in itin.days:
        if day.date != outbound.arrival_date:
            continue
        for item in day.items:
            # A leg of the journey itself may legitimately overlap the buffer.
            if item.kind in ("transit", "travel"):
                continue
            if _minutes(item.start) < ready:
                result.violations.append(
                    Violation(
                        "arrival_realism",
                        "hard",
                        f"starts {item.start:%H:%M}, but the flight lands "
                        f"{outbound.arrive_time:%H:%M} and the traveller cannot be "
                        f"in the city before {_fmt(ready)} "
                        f"({DEPLANE_MINUTES} min to clear the airport + "
                        f"{spec.airport_transfer_minutes} min transfer)",
                        _where(day, item),
                    )
                )
    return result


def _check_departure_realism(spec: TripSpec, itin: Itinerary) -> CheckResult:
    """Nothing may run past the moment the traveller has to leave for the airport."""
    ret = itin.flight("return")
    result = CheckResult(
        "departure_realism", "hard", skipped=spec.origin is None or ret is None
    )
    if result.skipped or ret is None:
        return result

    leave_by = _minutes(ret.depart_time) - CHECKIN_MINUTES - spec.airport_transfer_minutes
    for day in itin.days:
        if day.date != ret.date:
            continue
        for item in day.items:
            if item.kind in ("transit", "travel"):
                continue
            if _minutes(item.end) > leave_by:
                result.violations.append(
                    Violation(
                        "departure_realism",
                        "hard",
                        f"runs to {item.end:%H:%M}, but the flight leaves "
                        f"{ret.depart_time:%H:%M} and the traveller must set off by "
                        f"{_fmt(leave_by)} ({spec.airport_transfer_minutes} min transfer "
                        f"+ {CHECKIN_MINUTES} min check-in)",
                        _where(day, item),
                    )
                )
    return result


def _check_flights_grounded(itin: Itinerary) -> CheckResult:
    result = CheckResult("flights_grounded", "soft", skipped=not itin.flights)
    for flight in itin.flights:
        if not flight.sources:
            result.violations.append(
                Violation(
                    "flights_grounded",
                    "soft",
                    f"{flight.direction} flight has no source for its times or price",
                )
            )
    return result


def _check_grounded(itin: Itinerary, metrics: dict[str, float]) -> CheckResult:
    result = CheckResult("grounded", "soft")
    with_venue = [(d, i) for d, i in itin.all_items() if i.venue is not None]
    sourced = 0
    for day, item in with_venue:
        if item.sources:
            sourced += 1
        else:
            result.violations.append(
                Violation("grounded", "soft", "no source URL", _where(day, item))
            )
    metrics["grounded_pct"] = round(sourced / len(with_venue), 4) if with_venue else 1.0
    result.detail = f"{sourced} of {len(with_venue)} venues sourced"
    return result


def _check_duplicates(itin: Itinerary) -> CheckResult:
    result = CheckResult("no_duplicates", "soft")
    seen: dict[str, list[str]] = defaultdict(list)
    for day, item in itin.all_items():
        if item.kind not in ("activity", "meal") or item.venue is None:
            continue
        seen[item.venue.name.strip().lower()].append(_where(day, item))
    for name, spots in seen.items():
        if len(spots) > 1:
            result.violations.append(
                Violation(
                    "no_duplicates",
                    "soft",
                    f"{name!r} is scheduled {len(spots)} times: {'; '.join(spots)}",
                )
            )
    return result


def _check_hours_known(itin: Itinerary) -> CheckResult:
    """Absent opening hours aren't a failure, but they are a gap in research.

    Keeping this separate from `opening_hours` is what lets the hard check stay
    honest: unknown means the agent didn't look, not that the venue is shut.
    """
    result = CheckResult("hours_known", "soft")
    for day, item in itin.all_items():
        if item.kind not in ("activity", "meal") or item.venue is None:
            continue
        hours = item.venue.opening_hours
        if hours is None or hours.status(weekday_of(day.date)) == "unknown":
            result.violations.append(
                Violation(
                    "hours_known",
                    "soft",
                    f"opening hours for {item.venue.name} on "
                    f"{weekday_of(day.date)} were never established",
                    _where(day, item),
                )
            )
    return result


__all__ = [
    "CheckResult",
    "ConstraintReport",
    "Violation",
    "check_itinerary",
    "check_payload",
]

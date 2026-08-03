"""Turn a validated itinerary into things a person reads.

Deliberately mechanical: the model writes JSON, this writes the prose around
it. Keeping rendering out of the model's hands means the markdown can never
disagree with the checked artifact — a class of bug that is otherwise very hard
to notice, because both halves look fine on their own.
"""

from __future__ import annotations

from wayfinder.schema import Itinerary, TripSpec, weekday_of
from wayfinder.verify import ConstraintReport


def render_markdown(spec: TripSpec, itin: Itinerary, report: ConstraintReport) -> str:
    lines: list[str] = [f"# {itin.destination}", ""]

    nights = (spec.dates.end - spec.dates.start).days
    lines.append(
        f"{spec.dates.start:%a %d %b} – {spec.dates.end:%a %d %b %Y} · "
        f"{nights + 1} days · {spec.party.size} travellers · {spec.pace} pace"
    )
    lines.append("")

    if not itin.feasible:
        lines += [
            "## This trip doesn't fit",
            "",
            itin.infeasibility_reason or "",
            "",
            "Loosen one of the constraints above and run it again.",
            "",
        ]
        return "\n".join(lines)

    counts_flights = not any("flight" in e.lower() for e in spec.budget.excludes)
    total = itin.total_cost() + (itin.flight_cost() if counts_flights else 0.0)
    headroom = spec.budget.total - total
    lines.append(
        f"**Estimated cost** {total:,.0f} {itin.currency} of a "
        f"{spec.budget.total:,.0f} budget "
        f"({'`' + format(headroom, ',.0f') + '` to spare' if headroom >= 0 else 'over'})."
    )
    if spec.budget.excludes:
        lines.append(f"Excludes: {', '.join(spec.budget.excludes)}.")
    lines.append("")

    if itin.notes:
        lines += [itin.notes, ""]

    if itin.flights:
        lines += ["## Flights", ""]
        for f in sorted(itin.flights, key=lambda x: (x.direction != "outbound", x.date)):
            carrier = " ".join(x for x in (f.airline, f.flight_number) if x)
            stops = "direct" if f.stops == 0 else f"{f.stops} stop{'s' if f.stops > 1 else ''}"
            overnight = " **+1 day**" if f.arrives_next_day else ""
            cost = f" · {f.estimated_cost:,.0f} {itin.currency}" if f.estimated_cost else ""
            lines.append(
                f"- **{f.direction.title()}** {f.date:%a %d %b} · "
                f"{f.origin_airport} {f.depart_time:%H:%M} → "
                f"{f.destination_airport} {f.arrive_time:%H:%M}{overnight} · "
                f"{stops}{' · ' + carrier if carrier else ''}{cost}"
            )
            if f.note:
                lines.append(f"  - {f.note}")
        lines += [
            "",
            "_Flight times and fares are indicative, gathered from public sources. "
            "Nothing is booked — check live availability before you buy._",
            "",
        ]

    for day in itin.days:
        lines.append(f"## {day.date:%A %d %B}")
        if day.summary:
            lines += ["", f"*{day.summary}*"]
        lines.append("")
        if not day.items:
            lines += ["Nothing scheduled.", ""]
            continue

        for it in day.items:
            leg = it.transit_from_previous
            if leg:
                distance = f", {leg.distance_km:.1f} km" if leg.distance_km else ""
                lines.append(f"- *{leg.minutes} min by {leg.mode}{distance}*")
            cost = f" · {it.estimated_cost:,.0f} {itin.currency}" if it.estimated_cost else ""
            stars = ""
            if it.venue and it.venue.rating:
                r = it.venue.rating
                count = f" ({r.count:,})" if r.count else ""
                stars = f" · ★ {r.score:.1f}{count}"
            lines.append(f"- **{it.start:%H:%M}–{it.end:%H:%M}** {it.title}{cost}{stars}")
            if it.venue and it.venue.address:
                lines.append(f"  - {it.venue.address}")
            if it.note:
                lines.append(f"  - {it.note}")
            hours = it.venue.opening_hours if it.venue else None
            if hours and hours.status(weekday_of(day.date)) == "open":
                windows = ", ".join(
                    f"{r.open:%H:%M}–{r.close:%H:%M}" for r in hours.week[weekday_of(day.date)]
                )
                lines.append(f"  - Open {windows}")
        lines.append("")

    lines += ["---", "", "## Checks", ""]
    if report.passed and not report.violations:
        lines.append("Every constraint checks out.")
    else:
        lines.append(
            f"{len(report.hard_violations)} hard, {len(report.soft_violations)} soft."
        )
        lines.append("")
        for v in report.violations:
            where = f" ({v.where})" if v.where else ""
            lines.append(f"- **{v.severity}** `{v.check}`{where} — {v.message}")
    lines.append("")
    lines.append(
        "_Nothing here is booked. Times and prices are estimates from public "
        "sources; confirm before you commit money._"
    )
    lines.append("")
    return "\n".join(lines)


def render_sources(itin: Itinerary) -> str:
    """A flat bibliography, so claims can be traced back without reading JSON."""
    lines = ["# Sources", ""]
    seen: dict[str, list[str]] = {}
    for _, item in itin.all_items():
        for url in item.sources:
            seen.setdefault(url, []).append(item.title)
        if item.venue and item.venue.rating:
            seen.setdefault(item.venue.rating.source, []).append(
                f"{item.title} (rating)"
            )
    for flight in itin.flights:
        for url in flight.sources:
            seen.setdefault(url, []).append(f"{flight.direction} flight")
    if not seen:
        lines.append("_No sources were recorded._")
        lines.append("")
        return "\n".join(lines)
    for url, titles in sorted(seen.items()):
        unique = sorted(set(titles))
        lines.append(f"- <{url}> — {', '.join(unique)}")
    lines.append("")
    return "\n".join(lines)

"""Tests for flight planning and the first/last-day realism checks.

The value of putting flights in the schema is not that the agent lists them —
it is that the arrival and departure days can then be checked against them.
Every itinerary here is *internally plausible*: correct dates, sensible venues,
within budget. What makes them wrong is time the traveller doesn't have.
"""

from __future__ import annotations

import pytest
from conftest import item, make_itinerary, make_spec, violated

from wayfinder.schema import Itinerary, TripSpec
from wayfinder.tools.flights import (
    extract_airports,
    extract_duration_minutes,
    extract_price,
    flight_search,
)
from wayfinder.verify import check_itinerary


def flight(direction="outbound", **over):
    base = {
        "direction": direction,
        "date": "2026-10-12",
        "depart_time": "07:00",
        "arrive_time": "09:30",
        "origin_airport": "BER",
        "destination_airport": "LIS",
        "estimated_cost": 180,
        "sources": ["https://example.com/flight"],
    }
    base.update(over)
    return base


def spec_with_origin(**over):
    base = {"origin": "Berlin", "airport_transfer_minutes": 45}
    base.update(over)
    return make_spec(**base)


def itin_with(flights, items, **over):
    return Itinerary.model_validate(
        {
            "destination": "Lisbon",
            "currency": "EUR",
            "flights": flights,
            "days": [{"date": "2026-10-12", "items": items}],
            **over,
        }
    )


# --------------------------------------------------------------------------
# Presence and alignment
# --------------------------------------------------------------------------


def test_flight_checks_are_skipped_without_an_origin():
    """A trip planned for someone already in the city must not demand flights."""
    report = check_itinerary(make_spec(), make_itinerary([item("10:00", "11:00")]))
    assert "flights_present" not in violated(report)
    skipped = {c.name for c in report.checks if c.skipped}
    assert {"flights_present", "arrival_realism", "departure_realism"} <= skipped


def test_missing_legs_are_reported_once_an_origin_is_given():
    report = check_itinerary(spec_with_origin(), itin_with([], [item("10:00", "11:00")]))
    messages = " ".join(v.message for v in report.violations)
    assert "no outbound flight" in messages
    assert "no return flight" in messages
    assert not report.passed


def test_outbound_landing_after_the_trip_starts_is_rejected():
    spec = spec_with_origin(dates={"start": "2026-10-12", "end": "2026-10-12"})
    itin = itin_with(
        [flight(date="2026-10-13"), flight("return", date="2026-10-12")],
        [item("14:00", "15:00")],
    )
    assert "flight_alignment" in violated(check_itinerary(spec, itin))


def test_return_departing_before_the_trip_ends_is_rejected():
    spec = spec_with_origin(dates={"start": "2026-10-12", "end": "2026-10-14"})
    itin = Itinerary.model_validate(
        {
            "destination": "Lisbon", "currency": "EUR",
            "flights": [flight(), flight("return", date="2026-10-13")],
            "days": [{"date": d, "items": []} for d in
                     ("2026-10-12", "2026-10-13", "2026-10-14")],
        }
    )
    assert "flight_alignment" in violated(check_itinerary(spec, itin))


# --------------------------------------------------------------------------
# Arrival realism — the check this whole feature exists for
# --------------------------------------------------------------------------


def test_sightseeing_before_the_aircraft_lands_is_a_hard_failure():
    """The classic wrong itinerary: 09:00 Acropolis, 11:30 landing."""
    spec = spec_with_origin()
    itin = itin_with(
        [flight(arrive_time="11:30"), flight("return", date="2026-10-12")],
        [item("09:00", "10:30", "Acropolis")],
    )
    report = check_itinerary(spec, itin)
    assert "arrival_realism" in violated(report)
    assert not report.passed


def test_arrival_buffer_covers_airport_plus_transfer():
    """Lands 11:30 → 45 min to clear + 45 min transfer → nothing before 13:00."""
    spec = spec_with_origin(airport_transfer_minutes=45)
    ret = flight("return", date="2026-10-12", depart_time="23:00")

    too_early = itin_with([flight(arrive_time="11:30"), ret], [item("12:55", "14:00")])
    assert "arrival_realism" in violated(check_itinerary(spec, too_early))

    just_right = itin_with([flight(arrive_time="11:30"), ret], [item("13:00", "14:00")])
    assert "arrival_realism" not in violated(check_itinerary(spec, just_right))


def test_a_longer_transfer_pushes_the_first_activity_later():
    ret = flight("return", date="2026-10-12", depart_time="23:00")
    itin = itin_with([flight(arrive_time="11:30"), ret], [item("13:00", "14:00")])
    assert "arrival_realism" not in violated(
        check_itinerary(spec_with_origin(airport_transfer_minutes=45), itin)
    )
    # A 90-minute transfer from a distant airport moves the earliest start to 13:45.
    assert "arrival_realism" in violated(
        check_itinerary(spec_with_origin(airport_transfer_minutes=90), itin)
    )


def test_overnight_flights_are_measured_against_the_day_they_land():
    """A 06:00 landing after a red-eye is early morning, not eighteen hours early."""
    spec = spec_with_origin(dates={"start": "2026-10-12", "end": "2026-10-12"})
    itin = itin_with(
        [
            flight(date="2026-10-11", depart_time="22:00", arrive_time="06:00",
                   arrives_next_day=True),
            flight("return", date="2026-10-12", depart_time="23:30"),
        ],
        [item("07:00", "08:00")],
    )
    report = check_itinerary(spec, itin)
    # 06:00 + 45 + 45 = 07:30, so a 07:00 start is still too early...
    assert "arrival_realism" in violated(report)
    # ...but the alignment check must not complain: it landed on the right day.
    assert "flight_alignment" not in violated(report)


def test_the_transfer_itself_may_sit_inside_the_buffer():
    """The journey from the airport is not a violation of the airport buffer."""
    spec = spec_with_origin()
    itin = itin_with(
        [flight(arrive_time="11:30"), flight("return", date="2026-10-12", depart_time="23:00")],
        [
            item("11:30", "12:45", "Airport transfer", kind="transit", venue=None),
            item("13:00", "14:00", "First stop"),
        ],
    )
    assert "arrival_realism" not in violated(check_itinerary(spec, itin))


# --------------------------------------------------------------------------
# Departure realism
# --------------------------------------------------------------------------


def test_activities_running_past_the_airport_run_are_rejected():
    """Flight 18:00 → set off by 15:15 (45 transfer + 120 check-in)."""
    spec = spec_with_origin()
    itin = itin_with(
        [flight(arrive_time="08:00"), flight("return", date="2026-10-12", depart_time="18:00")],
        [item("14:00", "16:00", "A long lunch")],
    )
    report = check_itinerary(spec, itin)
    assert "departure_realism" in violated(report)
    assert "15:15" in " ".join(v.message for v in report.violations)


def test_finishing_before_the_airport_run_is_fine():
    spec = spec_with_origin()
    itin = itin_with(
        [flight(arrive_time="08:00"), flight("return", date="2026-10-12", depart_time="18:00")],
        [item("13:00", "15:15", "A long lunch")],
    )
    assert "departure_realism" not in violated(check_itinerary(spec, itin))


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------


def test_airfare_counts_toward_the_budget_by_default():
    """A trip that fits only because the airfare wasn't counted doesn't fit."""
    spec = spec_with_origin(budget={"currency": "EUR", "total": 300})
    itin = itin_with(
        [flight(estimated_cost=180), flight("return", date="2026-10-12", estimated_cost=180)],
        [item("13:00", "14:00", cost=50)],
    )
    report = check_itinerary(spec, itin)
    assert "budget" in violated(report)
    assert report.metrics["flight_cost"] == 360.0
    assert report.metrics["ground_cost"] == 50.0


def test_airfare_is_ignored_when_the_traveller_excludes_flights():
    spec = spec_with_origin(
        budget={"currency": "EUR", "total": 300, "excludes": ["flights"]}
    )
    itin = itin_with(
        [flight(estimated_cost=180), flight("return", date="2026-10-12", estimated_cost=180)],
        [item("13:00", "14:00", cost=50)],
    )
    report = check_itinerary(spec, itin)
    assert "budget" not in violated(report)
    # Still reported, so the traveller can see what they're on the hook for.
    assert report.metrics["flight_cost"] == 360.0


def test_unsourced_flights_are_a_soft_warning_not_a_failure():
    spec = spec_with_origin()
    itin = itin_with(
        [flight(sources=[]), flight("return", date="2026-10-12", depart_time="23:00")],
        [item("13:00", "14:00")],
    )
    report = check_itinerary(spec, itin)
    assert "flights_grounded" in violated(report)
    assert report.passed, "a missing source should not fail an otherwise valid plan"


# --------------------------------------------------------------------------
# Parsing search results
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Flights from BER to LIS from €128", ["BER", "LIS"]),
        ("Priced in EUR and USD only", []),
        ("LHR to CDG via AMS", ["AMS", "CDG", "LHR"]),
    ],
)
def test_airport_extraction_filters_false_positives(text, expected):
    assert extract_airports(text) == expected


@pytest.mark.parametrize(
    ("text", "amount", "currency"),
    [
        ("from €128 one way", 128.0, "EUR"),
        ("tickets at 210 USD", 210.0, "USD"),
        ("£89 return", 89.0, "GBP"),
    ],
)
def test_price_extraction(text, amount, currency):
    assert extract_price(text) == (amount, currency)


def test_price_extraction_needs_a_currency():
    assert extract_price("flight 447 departs at 7") is None


@pytest.mark.parametrize(
    ("text", "minutes"),
    [("2h 15m", 135), ("3 hours", 180), ("1h30", 90)],
)
def test_duration_extraction(text, minutes):
    assert extract_duration_minutes(text) == minutes


def test_implausible_durations_are_rejected():
    assert extract_duration_minutes("open 24 hours") is None


def test_flight_search_summarises_a_price_range(monkeypatch):
    monkeypatch.setattr(
        "wayfinder.tools.flights.web_search",
        lambda q, max_results=6: {
            "results": [
                {"title": "BER to LIS", "content": "from €128 · 3h 20m direct",
                 "url": "https://example.com/a"},
                {"title": "Cheap flights", "content": "tickets from €96",
                 "url": "https://example.com/b"},
            ]
        },
    )
    result = flight_search("Berlin", "Lisbon", "2026-10-12")
    assert result["found"]
    assert result["price_range"] == {"low": 96.0, "high": 128.0}
    assert "indicative" in result["caveat"].lower()


def test_flight_search_is_honest_when_nothing_is_found(monkeypatch):
    monkeypatch.setattr(
        "wayfinder.tools.flights.web_search", lambda q, max_results=6: {"results": []}
    )
    result = flight_search("Nowhere", "Elsewhere", "2026-10-12")
    assert result["found"] is False
    assert result["price_range"] is None


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def test_flights_render_with_their_caveat():
    from wayfinder.render import render_markdown, render_sources

    spec = spec_with_origin()
    itin = itin_with(
        [
            flight(airline="TAP", flight_number="TP123", arrive_time="09:30"),
            flight("return", date="2026-10-12", depart_time="20:00", arrive_time="23:00"),
        ],
        [item("13:00", "14:00")],
    )
    markdown = render_markdown(spec, itin, check_itinerary(spec, itin))
    assert "## Flights" in markdown
    assert "TAP TP123" in markdown
    assert "BER 07:00 → LIS 09:30" in markdown
    assert "Nothing is booked" in markdown
    assert "https://example.com/flight" in render_sources(itin)


def test_overnight_flights_are_flagged_in_the_rendering():
    from wayfinder.render import render_markdown

    spec = spec_with_origin()
    itin = itin_with(
        [
            flight(depart_time="22:00", arrive_time="06:00", arrives_next_day=True),
            flight("return", date="2026-10-12", depart_time="23:00"),
        ],
        [item("13:00", "14:00")],
    )
    assert "+1 day" in render_markdown(spec, itin, check_itinerary(spec, itin))

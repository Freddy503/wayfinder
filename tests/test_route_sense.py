"""Days that double back on themselves.

A real Amsterdam plan passed every hard check while scheduling
`Van Gogh Museum → bike rental in the Vondelpark → Rijksmuseum` — 2.6 km of
walking between two museums that share a square 310 m apart. Nothing was wrong
with any individual leg, which is exactly why nothing caught it: `transit_
feasible` asks whether the leg you recorded fits the gap you left, never
whether the order made sense.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import item, make_itinerary, make_spec, violated
from wayfinder.verify import DETOUR_FLOOR_KM, DETOUR_RATIO, check_itinerary, check_payload

# Real coordinates, so the distances in these tests are the real distances.
VAN_GOGH = (52.3584, 4.8811)
RIJKS = (52.3600, 4.8852)          # 310 m from the Van Gogh
VONDELPARK = (52.3580, 4.8686)     # ~1.3 km west of both
CENTRAAL = (52.3791, 4.9003)


def at(coords, title, start, end, **kw):
    payload = item(start, end, title, venue=title, **kw)
    payload["venue"]["lat"], payload["venue"]["lon"] = coords
    return payload


def report_for(items):
    return check_itinerary(make_spec(), make_itinerary(items))


def test_the_real_failure_is_caught():
    """The triple this check exists for."""
    report = report_for([
        at(VAN_GOGH, "Van Gogh Museum", "09:30", "12:00"),
        at(VONDELPARK, "Bike rental and Vondelpark ride", "12:15", "13:45"),
        at(RIJKS, "Rijksmuseum", "14:00", "16:00"),
    ])
    assert "route_sense" in violated(report)
    message = next(v.message for v in report.violations if v.check == "route_sense")
    assert "Van Gogh Museum" in message and "Rijksmuseum" in message
    assert "backtracking" in message


def test_it_never_fails_the_plan():
    """Soft on purpose: a bike ride between two museums may be the point.

    Asserted on the check's own severity rather than `report.passed`, because
    a bare fixture trips unrelated hard checks (no transit legs) and that would
    make this pass for the wrong reason.
    """
    report = report_for([
        at(VAN_GOGH, "Van Gogh Museum", "09:30", "12:00"),
        at(VONDELPARK, "Bike rental", "12:15", "13:45"),
        at(RIJKS, "Rijksmuseum", "14:00", "16:00"),
    ])
    route = [v for v in report.violations if v.check == "route_sense"]
    assert route, "the detour should still be reported"
    assert all(v.severity == "soft" for v in route)
    assert next(c for c in report.checks if c.name == "route_sense").severity == "soft"


def test_a_sensible_order_is_not_flagged():
    """The same three places, ordered by geography."""
    report = report_for([
        at(VAN_GOGH, "Van Gogh Museum", "09:30", "12:00"),
        at(RIJKS, "Rijksmuseum", "12:15", "14:15"),
        at(VONDELPARK, "Bike rental", "14:30", "16:00"),
    ])
    assert "route_sense" not in violated(report)


def test_a_genuinely_spread_out_day_is_not_a_detour():
    """Three places in a line, each far from the last. The ratio stays low
    because going via the middle is genuinely on the way."""
    report = report_for([
        at((52.3584, 4.8811), "West", "09:00", "10:00"),
        at((52.3680, 4.8900), "Middle", "11:00", "12:00"),
        at((52.3791, 4.9003), "East", "13:00", "14:00"),
    ])
    assert "route_sense" not in violated(report)


def test_a_small_wiggle_is_not_worth_reporting():
    """Two cafés on the same street via a shop round the corner clears the
    ratio easily but moves nobody anywhere. The distance floor drops it."""
    report = report_for([
        at((52.3584, 4.8811), "Café", "09:00", "10:00"),
        at((52.3590, 4.8790), "Shop", "10:15", "10:45"),
        at((52.3586, 4.8814), "Bakery", "11:00", "12:00"),
    ])
    assert "route_sense" not in violated(report)


def test_both_conditions_are_required():
    """Ratio alone flags the street-corner case; distance alone flags a
    legitimately spread-out day. Neither is a detour on its own."""
    assert DETOUR_RATIO > 1.0
    assert DETOUR_FLOOR_KM > 0.0


def test_venues_without_coordinates_are_skipped_not_failed():
    """An itinerary the agent never geocoded is a `grounded` problem. Failing
    it here would blame the wrong check."""
    report = report_for([
        item("09:00", "10:00", "A"), item("11:00", "12:00", "B"), item("13:00", "14:00", "C"),
    ])
    route = next(c for c in report.checks if c.name == "route_sense")
    assert route.skipped is True
    assert route.violations == []


def test_a_day_too_short_to_have_an_order_is_skipped():
    report = report_for([at(VAN_GOGH, "Van Gogh Museum", "09:30", "12:00")])
    assert next(c for c in report.checks if c.name == "route_sense").skipped is True


def test_the_detour_is_located_at_the_stop_that_causes_it():
    """So the UI can point at the item to move."""
    report = report_for([
        at(VAN_GOGH, "Van Gogh Museum", "09:30", "12:00"),
        at(VONDELPARK, "Bike rental", "12:15", "13:45"),
        at(RIJKS, "Rijksmuseum", "14:00", "16:00"),
    ])
    where = next(v.where for v in report.violations if v.check == "route_sense")
    assert "Bike rental" in where


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_metrics_are_counts_and_a_worst_case_not_a_sum():
    """Consecutive triples share a hop — A→B→C and B→C→D both contain B→C — so
    a sum of detour distances double-counts. The first version of this reported
    20.6 km of "avoidable" walking on a trip that walked about half that."""
    report = report_for([
        at(VAN_GOGH, "Van Gogh Museum", "09:30", "12:00"),
        at(VONDELPARK, "Bike rental", "12:15", "13:45"),
        at(RIJKS, "Rijksmuseum", "14:00", "16:00"),
    ])
    assert report.metrics["detour_count"] == 1.0
    # ~1.7 km: the detour out to the Vondelpark and back, over a 310 m hop.
    assert 1.0 < report.metrics["worst_detour_km"] < 2.5
    assert "detour_km" not in report.metrics, "the summed version was not defensible"


def test_route_efficiency_is_the_fraction_of_legs_that_go_somewhere():
    report = report_for([
        at(VAN_GOGH, "Van Gogh Museum", "09:30", "12:00"),
        at(VONDELPARK, "Bike rental", "12:15", "13:45"),
        at(RIJKS, "Rijksmuseum", "14:00", "16:00"),
        at(CENTRAAL, "Centraal", "16:30", "17:30"),
    ])
    # Two triples judged, one of them a detour.
    assert report.metrics["route_efficiency"] == 0.5


def test_a_clean_day_scores_one():
    report = report_for([
        at(VAN_GOGH, "Van Gogh Museum", "09:30", "12:00"),
        at(RIJKS, "Rijksmuseum", "12:15", "14:15"),
        at(CENTRAAL, "Centraal", "15:00", "16:00"),
    ])
    assert report.metrics["route_efficiency"] == 1.0
    assert report.metrics["detour_count"] == 0.0


# --------------------------------------------------------------------------
# Pinned to the run that prompted this
# --------------------------------------------------------------------------


AMSTERDAM = Path(__file__).resolve().parent.parent / "runs" / \
    "amsterdam-2026-11-05-20260806T152201"


@pytest.mark.skipif(not AMSTERDAM.exists(), reason="the reference run is not checked in")
def test_the_amsterdam_run_that_started_this():
    """Pins the thresholds against real output. If a tweak stops catching the
    Van Gogh triple, or starts flagging half the trip, this fails."""
    from wayfinder.specs import load_spec

    report = check_payload(
        load_spec(AMSTERDAM / "spec.yaml"),
        json.loads((AMSTERDAM / "itinerary.json").read_text(encoding="utf-8")),
    )
    detours = [v for v in report.violations if v.check == "route_sense"]
    assert len(detours) == 7
    assert any("Van Gogh" in v.message and "Rijksmuseum" in v.message for v in detours)
    assert report.metrics["route_efficiency"] == pytest.approx(0.417, abs=0.01)
    assert report.passed is True, "still soft — this plan is usable, just inefficient"


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def test_the_agent_is_told_to_order_days_by_geography():
    """The check is the safety net; the prompt is the fix."""
    from wayfinder.prompts import MAIN_PROMPT

    assert "Order each day by geography" in MAIN_PROMPT
    assert "Van Gogh Museum" in MAIN_PROMPT, "name the real failure"
    assert "route_sense" in MAIN_PROMPT, "and where the checker reports it"


def test_a_run_with_no_itinerary_does_not_score_a_perfect_route():
    """Absence of detours in absence of a plan is not a good route."""
    from wayfinder.evals.evaluators import route_efficiency

    assert route_efficiency({}, {"report": {"metrics": {"schema_valid": 0.0}}})["score"] == 0.0


def test_the_clean_fixture_stays_geographically_sane(spec, clean_payload):
    """It wasn't, when this check landed: Graça → the Azulejo museum (far
    east) → Time Out Market (south-west) is a real 3.5 km backtrack. The
    fixture predated the check and was reordered rather than excused — a
    reference itinerary called "clean" has to actually be clean."""
    report = check_payload(spec, clean_payload)
    assert [v for v in report.violations if v.check == "route_sense"] == []
    assert report.metrics["route_efficiency"] == 1.0

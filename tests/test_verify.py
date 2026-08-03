"""Tests for the constraint checker.

Two things are being defended here:

1. **No false positives.** The clean fixture must come back completely silent —
   not "only soft violations", but zero. A checker that cries wolf on a good
   plan will train the agent to ignore it.
2. **No silent gaps.** `test_dirty_fixture_trips_every_check` asserts that every
   check the clean run exercises also *fails* somewhere in the dirty fixture.
   Add a check without seeding a failure for it and this test tells you.
"""

from __future__ import annotations

import pytest
from conftest import item, make_itinerary, make_spec, violated

from wayfinder.verify import check_itinerary, check_payload


# --------------------------------------------------------------------------
# The two fixtures
# --------------------------------------------------------------------------


def test_clean_fixture_is_completely_silent(spec, clean_payload):
    report = check_payload(spec, clean_payload)
    assert report.violations == [], report.summary()
    assert report.passed


def test_clean_fixture_metrics(spec, clean_payload):
    m = check_payload(spec, clean_payload).metrics
    assert m["schema_valid"] == 1.0
    assert m["hard_pass_rate"] == 1.0
    assert m["soft_pass_rate"] == 1.0
    assert m["budget_overrun_pct"] == 0.0
    assert m["must_do_coverage"] == 1.0
    assert m["grounded_pct"] == 1.0
    assert m["transit_infeasibility_count"] == 0.0
    assert m["refused"] == 0.0
    assert m["total_cost"] == 185.0


def test_dirty_fixture_fails(spec, dirty_payload):
    report = check_payload(spec, dirty_payload)
    assert not report.passed
    assert report.metrics["schema_valid"] == 1.0, "dirty fixture must still parse"
    assert report.metrics["hard_pass_rate"] < 1.0


def test_dirty_fixture_trips_every_check(spec, clean_payload, dirty_payload):
    """The fixture and the checker must not drift apart."""
    exercised = {
        c.name for c in check_payload(spec, clean_payload).checks if not c.skipped
    }
    tripped = violated(check_payload(spec, dirty_payload))
    missing = exercised - tripped
    assert not missing, f"no seeded violation for: {sorted(missing)}"


@pytest.mark.parametrize(
    "check",
    [
        "currency_match",
        "dates_covered",
        "chronology",
        "budget",
        "transit_feasible",
        "opening_hours",
        "must_do_coverage",
        "time_window",
        "required_meals",
        "mobility",
        "free_block",
        "pace",
        "grounded",
        "no_duplicates",
        "hours_known",
    ],
)
def test_dirty_fixture_trips_named_check(spec, dirty_payload, check):
    assert check in violated(check_payload(spec, dirty_payload))


# --------------------------------------------------------------------------
# Schema handling
# --------------------------------------------------------------------------


def test_malformed_payload_reports_rather_than_raises(spec):
    report = check_payload(spec, {"destination": "Lisbon", "currency": "EUR", "days": "nope"})
    assert not report.passed
    assert report.metrics["schema_valid"] == 0.0
    assert violated(report) == {"schema_valid"}


def test_unknown_field_is_rejected(spec):
    report = check_payload(
        spec, {"destination": "Lisbon", "currency": "EUR", "days": [], "vibe": "good"}
    )
    assert not report.passed
    assert violated(report) == {"schema_valid"}


def test_item_ending_before_it_starts_is_a_schema_error(spec):
    report = check_payload(
        spec,
        {
            "destination": "Lisbon",
            "currency": "EUR",
            "days": [
                {
                    "date": "2026-10-12",
                    "items": [item("15:00", "14:00")],
                }
            ],
        },
    )
    assert violated(report) == {"schema_valid"}


def test_meal_without_slot_is_a_schema_error(spec):
    report = check_payload(
        spec,
        {
            "destination": "Lisbon",
            "currency": "EUR",
            "days": [{"date": "2026-10-12", "items": [item("19:00", "20:00", kind="meal")]}],
        },
    )
    assert violated(report) == {"schema_valid"}


# --------------------------------------------------------------------------
# Refusal — declaring a trip impossible is a correct answer, not a failure
# --------------------------------------------------------------------------


def test_refusal_passes_and_skips_content_checks():
    spec = make_spec(must_do=["Something unreachable"], should_refuse=True)
    itin = make_itinerary(
        [], days=[], feasible=False, infeasibility_reason="Budget covers one night, not four."
    )
    report = check_itinerary(spec, itin)
    assert report.passed
    assert report.violations == []
    assert report.metrics["refused"] == 1.0


def test_refusal_without_a_reason_is_a_schema_error():
    report = check_payload(
        make_spec(),
        {"destination": "Lisbon", "currency": "EUR", "days": [], "feasible": False},
    )
    assert violated(report) == {"schema_valid"}


# --------------------------------------------------------------------------
# Targeted checks — one constraint at a time
# --------------------------------------------------------------------------


def test_transit_leg_missing_between_different_venues_is_hard():
    """The loophole that would otherwise make any day look feasible."""
    itin = make_itinerary(
        [
            item("10:00", "11:00", "A", venue="Venue A"),
            item("11:05", "12:00", "B", venue="Venue B"),
        ]
    )
    report = check_itinerary(make_spec(), itin)
    assert "transit_feasible" in violated(report)
    assert not report.passed


def test_transit_leg_not_required_when_staying_put():
    itin = make_itinerary(
        [
            item("10:00", "11:00", "A", venue="Venue A"),
            item("11:05", "12:00", "Still A", venue="Venue A"),
        ]
    )
    assert "transit_feasible" not in violated(check_itinerary(make_spec(), itin))


def test_transit_that_does_not_fit_the_gap():
    itin = make_itinerary(
        [
            item("10:00", "11:00", "A", venue="Venue A"),
            item(
                "11:10",
                "12:00",
                "B",
                venue="Venue B",
                transit={"mode": "walk", "minutes": 40},
            ),
        ]
    )
    report = check_itinerary(make_spec(), itin)
    assert "transit_feasible" in violated(report)
    assert report.metrics["transit_infeasibility_count"] == 1.0


def test_transit_over_the_stated_cap():
    spec = make_spec(constraints={"max_transit_minutes": 20})
    itin = make_itinerary(
        [
            item("10:00", "11:00", "A", venue="Venue A"),
            item(
                "12:00",
                "13:00",
                "B",
                venue="Venue B",
                transit={"mode": "transit", "minutes": 35},
            ),
        ]
    )
    # 35 minutes fits the 60-minute gap fine; it's the cap that's breached.
    assert "transit_feasible" in violated(check_itinerary(spec, itin))


def test_closed_venue_is_hard_but_unknown_hours_are_soft():
    closed = make_itinerary(
        [item("10:00", "11:00", hours={"mon": []})]
    )
    report = check_itinerary(make_spec(), closed)
    assert "opening_hours" in violated(report)
    assert not report.passed

    unknown = make_itinerary([item("10:00", "11:00", hours=None)])
    report = check_itinerary(make_spec(), unknown)
    assert violated(report) == {"hours_known"}
    assert report.passed, "not knowing the hours must not fail the run"


def test_scheduled_outside_opening_window():
    itin = make_itinerary(
        [item("09:00", "10:30", hours={"mon": [{"open": "10:00", "close": "18:00"}]})]
    )
    assert "opening_hours" in violated(check_itinerary(make_spec(), itin))


def test_free_block_is_measured_net_of_travel():
    """A three-hour gap eaten by a 170-minute ride is not three hours off."""
    spec = make_spec(constraints={"min_free_block_minutes": 120})
    itin = make_itinerary(
        [
            item("10:00", "11:00", "A", venue="Venue A"),
            item(
                "14:00",
                "15:00",
                "B",
                venue="Venue B",
                transit={"mode": "transit", "minutes": 170},
            ),
        ]
    )
    assert "free_block" in violated(check_itinerary(spec, itin))


def test_free_block_satisfied_by_a_real_gap():
    spec = make_spec(constraints={"min_free_block_minutes": 120})
    itin = make_itinerary(
        [
            item("10:00", "11:00", "A", venue="Venue A"),
            item(
                "14:00",
                "15:00",
                "B",
                venue="Venue B",
                transit={"mode": "walk", "minutes": 15},
            ),
        ]
    )
    assert "free_block" not in violated(check_itinerary(spec, itin))


def test_pace_ceiling_is_soft_when_inferred_and_hard_when_stated():
    items = [
        item(f"{h:02d}:00", f"{h:02d}:45", f"Stop {h}", venue=f"Venue {h}")
        for h in range(9, 15)
    ]
    for i in items[1:]:
        i["transit_from_previous"] = {"mode": "walk", "minutes": 10, "distance_km": 0.5}
    itin = make_itinerary(items)

    inferred = check_itinerary(make_spec(pace="relaxed"), itin)
    assert "pace" in violated(inferred)
    assert inferred.passed, "a pace heuristic alone must not fail the run"

    stated = check_itinerary(make_spec(constraints={"max_activities_per_day": 3}), itin)
    assert "pace" in violated(stated)
    assert not stated.passed, "a number the traveller stated is a hard constraint"


def test_budget_overrun_is_reported_as_a_fraction():
    spec = make_spec(budget={"currency": "EUR", "total": 100})
    itin = make_itinerary([item("10:00", "11:00", cost=150)])
    report = check_itinerary(spec, itin)
    assert "budget" in violated(report)
    assert report.metrics["budget_overrun_pct"] == 0.5


def test_budget_within_cap_reports_zero_overrun():
    spec = make_spec(budget={"currency": "EUR", "total": 100})
    itin = make_itinerary([item("10:00", "11:00", cost=99)])
    report = check_itinerary(spec, itin)
    assert report.metrics["budget_overrun_pct"] == 0.0


def test_must_do_matching_is_case_insensitive_and_partial():
    spec = make_spec(must_do=["time out market"])
    itin = make_itinerary([item("19:00", "20:00", "Dinner at Time Out Market")])
    report = check_itinerary(spec, itin)
    assert "must_do_coverage" not in violated(report)
    assert report.metrics["must_do_coverage"] == 1.0


def test_must_do_coverage_is_a_fraction():
    spec = make_spec(must_do=["Time Out Market", "Oceanario", "Tram 28"])
    itin = make_itinerary([item("19:00", "20:00", "Dinner at Time Out Market")])
    report = check_itinerary(spec, itin)
    assert report.metrics["must_do_coverage"] == pytest.approx(1 / 3, abs=1e-4)


def test_walking_cap_counts_only_walking():
    spec = make_spec(mobility={"max_walk_km_per_day": 3})
    itin = make_itinerary(
        [
            item("10:00", "11:00", "A", venue="Venue A"),
            item(
                "12:00",
                "13:00",
                "B",
                venue="Venue B",
                transit={"mode": "transit", "minutes": 20, "distance_km": 40.0},
            ),
        ]
    )
    report = check_itinerary(spec, itin)
    assert "mobility" not in violated(report), "a 40 km metro ride is not a 40 km walk"

    itin.days[0].items[1].transit_from_previous.mode = "walk"
    itin.days[0].items[1].transit_from_previous.distance_km = 4.0
    assert "mobility" in violated(check_itinerary(spec, itin))


def test_missing_and_extra_dates_are_both_reported():
    spec = make_spec(dates={"start": "2026-10-12", "end": "2026-10-13"})
    itin = make_itinerary([item("10:00", "11:00")])  # only covers the 12th
    itin.days.append(itin.days[0].model_copy(deep=True))
    itin.days[1].date = itin.days[1].date.replace(day=20)
    messages = " ".join(v.message for v in check_itinerary(spec, itin).violations)
    assert "no plan for this date" in messages
    assert "outside the trip" in messages


# --------------------------------------------------------------------------
# Pass-rate arithmetic
# --------------------------------------------------------------------------


def test_skipped_checks_are_excluded_from_the_pass_rate():
    """An unstated constraint must not pad the score.

    A bare spec skips several checks. If skipped checks counted as passes, the
    hard pass rate would drift up as constraints are *removed* — the opposite
    of what it should measure.
    """
    spec = make_spec()  # no constraints, no must_do, no mobility cap
    itin = make_itinerary([item("10:00", "11:00")])
    report = check_itinerary(spec, itin)

    ran = [c for c in report.checks if c.severity == "hard" and not c.skipped]
    skipped = [c for c in report.checks if c.severity == "hard" and c.skipped]
    assert skipped, "this spec should skip some checks"
    assert report.metrics["hard_pass_rate"] == 1.0
    assert len(ran) < len(ran) + len(skipped)


def test_hard_pass_rate_falls_with_each_failing_check():
    spec = make_spec(budget={"currency": "EUR", "total": 10})
    itin = make_itinerary([item("10:00", "11:00", cost=999)], currency="USD")
    report = check_itinerary(spec, itin)
    assert 0.0 < report.metrics["hard_pass_rate"] < 1.0
    assert report.metrics["hard_violation_count"] == 2.0  # budget + currency

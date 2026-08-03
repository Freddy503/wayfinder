from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from wayfinder.schema import Itinerary, TripSpec
from wayfinder.specs import load_itinerary_payload, load_spec

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def spec() -> TripSpec:
    return load_spec(FIXTURES / "lisbon.spec.yaml")


@pytest.fixture
def clean_payload() -> Any:
    return load_itinerary_payload(FIXTURES / "clean.itinerary.json")


@pytest.fixture
def dirty_payload() -> Any:
    return load_itinerary_payload(FIXTURES / "dirty.itinerary.json")


def make_spec(**overrides: Any) -> TripSpec:
    """A one-day Lisbon spec with no constraints set, for targeted tests.

    Everything optional is left unset so each test opts in to exactly the one
    constraint it is exercising — otherwise an unrelated check fires and the
    assertion passes for the wrong reason.
    """
    base: dict[str, Any] = {
        "destination": "Lisbon, Portugal",
        "dates": {"start": "2026-10-12", "end": "2026-10-12"},
        "budget": {"currency": "EUR", "total": 1000},
    }
    base.update(overrides)
    return TripSpec.model_validate(base)


def make_itinerary(items: list[dict[str, Any]], **overrides: Any) -> Itinerary:
    """A one-day itinerary wrapping `items`, matching `make_spec`'s date."""
    base: dict[str, Any] = {
        "destination": "Lisbon, Portugal",
        "currency": "EUR",
        "days": [{"date": "2026-10-12", "items": items}],
    }
    base.update(overrides)
    return Itinerary.model_validate(base)


def item(
    start: str,
    end: str,
    title: str = "Something",
    *,
    kind: str = "activity",
    venue: str | None = "Venue A",
    hours: dict[str, list[dict[str, str]]] | None = None,
    cost: float = 0.0,
    transit: dict[str, Any] | None = None,
    sources: list[str] | None = None,
    meal_slot: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "start": start,
        "end": end,
        "kind": kind,
        "title": title,
        "estimated_cost": cost,
        "sources": sources if sources is not None else ["https://example.com"],
    }
    if meal_slot:
        payload["meal_slot"] = meal_slot
    if venue is not None:
        v: dict[str, Any] = {"name": venue}
        if hours is not None:
            v["opening_hours"] = {"week": hours}
        payload["venue"] = v
    if transit is not None:
        payload["transit_from_previous"] = transit
    return payload


def violated(report: Any) -> set[str]:
    return {v.check for v in report.violations}

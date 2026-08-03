"""Tests for rating extraction and speech-mode requirements capture.

Both parse messy real-world input, so the interesting cases are the ones where
the right answer is "I don't know": a snippet with no score, a transcript that
never mentions a budget. Guessing there is worse than admitting the gap —
an invented rating ends up printed on the traveller's itinerary.
"""

from __future__ import annotations

from datetime import date

import pytest

from wayfinder.extract import (
    ExtractedRequirements,
    extract_requirements,
    merge_transcript,
)
from wayfinder.schema import Itinerary, Rating
from wayfinder.tools.ratings import parse_rating, venue_rating

# --------------------------------------------------------------------------
# Rating parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("snippet", "score", "count"),
    [
        ("Rating: 4.6 ★ (21,391 reviews)", 4.6, 21391),
        ("4.7 out of 5 · 1,204 Google reviews", 4.7, 1204),
        ("Bewertung 4,3 von 5 — 892 Rezensionen", 4.3, None),
        ("Acropolis Museum 4.8/5 (48.221 reviews)", 4.8, 48221),
        ("Scored 3.9 stars", 3.9, None),
    ],
)
def test_parses_real_world_snippets(snippet, score, count):
    parsed = parse_rating(snippet)
    assert parsed is not None
    assert parsed[0] == score
    assert parsed[1] == count


@pytest.mark.parametrize(
    "snippet",
    [
        "Open 10:00-18:00 daily",
        "Admission EUR 20 for adults",
        "Established in 1834, rebuilt 1952",
        "",
        "Over 20,000 visitors a year",  # a count with no score is not a rating
    ],
)
def test_refuses_to_invent_a_rating(snippet):
    assert parse_rating(snippet) is None


def test_score_must_be_in_range():
    assert parse_rating("priced at 7.5 euro") is None


def test_tiny_counts_are_discarded_as_noise():
    """'(4)' next to a score is far more likely a footnote than a review count."""
    parsed = parse_rating("4.5 stars (4)")
    assert parsed is not None
    assert parsed[1] is None


def test_venue_rating_ignores_snippets_about_other_places(monkeypatch):
    """A high-scoring snippet for a different venue must not be attributed."""
    monkeypatch.setattr(
        "wayfinder.tools.ratings.web_search",
        lambda q, max_results=5: {
            "results": [
                {"title": "Best bars in Lisbon", "content": "4.9 (9,000 reviews)",
                 "url": "https://example.com/other"},
                {"title": "Cervejaria Ramiro", "content": "4.5 ★ (32,101 reviews)",
                 "url": "https://example.com/ramiro"},
            ]
        },
    )
    result = venue_rating("Cervejaria Ramiro", "Lisbon")
    assert result["found"] is True
    assert result["score"] == 4.5
    assert result["source"] == "https://example.com/ramiro"


def test_venue_rating_reports_not_found(monkeypatch):
    monkeypatch.setattr(
        "wayfinder.tools.ratings.web_search",
        lambda q, max_results=5: {"results": [{"title": "x", "content": "no numbers", "url": "u"}]},
    )
    assert venue_rating("Nowhere Cafe", "Lisbon")["found"] is False


# --------------------------------------------------------------------------
# Ratings in the itinerary schema
# --------------------------------------------------------------------------


def test_rating_requires_a_source():
    """A score you can't trace is a score you can't trust."""
    with pytest.raises(ValueError):
        Rating(score=4.5, count=100, source="")


def test_rating_rides_along_on_a_venue():
    itin = Itinerary.model_validate(
        {
            "destination": "Lisbon",
            "currency": "EUR",
            "days": [{"date": "2026-10-12", "items": [{
                "start": "19:00", "end": "20:30", "kind": "meal", "meal_slot": "dinner",
                "title": "Dinner", "venue": {"name": "Ramiro", "rating": {
                    "score": 4.5, "count": 32101, "source": "https://example.com"}},
            }]}],
        }
    )
    assert itin.days[0].items[0].venue.rating.score == 4.5


def test_ratings_appear_in_the_rendered_itinerary():
    from wayfinder.render import render_markdown, render_sources
    from wayfinder.verify import check_itinerary
    from conftest import make_spec

    itin = Itinerary.model_validate(
        {
            "destination": "Lisbon", "currency": "EUR",
            "days": [{"date": "2026-10-12", "items": [{
                "start": "10:00", "end": "11:00", "kind": "activity", "title": "Castle",
                "sources": ["https://example.com/castle"],
                "venue": {"name": "Castle", "rating": {
                    "score": 4.6, "count": 21391, "source": "https://example.com/rating"}},
            }]}],
        }
    )
    spec = make_spec()
    markdown = render_markdown(spec, itin, check_itinerary(spec, itin))
    assert "★ 4.6" in markdown
    assert "21,391" in markdown
    # The rating's source belongs in the bibliography too, not just the score.
    assert "https://example.com/rating" in render_sources(itin)


# --------------------------------------------------------------------------
# Speech-mode extraction
# --------------------------------------------------------------------------


def fake_extractor(**kwargs):
    return lambda transcript: ExtractedRequirements(**kwargs)


def test_empty_transcript_asks_the_first_question():
    result = extract_requirements("")
    assert result["missing"] == ["destination", "dates", "budget"]
    assert result["follow_up"]


def test_extraction_maps_onto_form_field_names():
    result = extract_requirements(
        "lisbon in october, two of us, six hundred euros",
        extractor=fake_extractor(
            destination="Lisbon, Portugal",
            start_date=date(2026, 10, 12),
            end_date=date(2026, 10, 15),
            adults=2,
            budget_total=600,
            currency="eur",
            soft_preferences=["viewpoints"],
        ),
    )
    fields = result["fields"]
    assert fields["destination"] == "Lisbon, Portugal"
    assert fields["start"] == "2026-10-12"
    assert fields["end"] == "2026-10-15"
    assert fields["currency"] == "EUR", "currency should be normalised for the schema"
    assert result["missing"] == []
    assert result["follow_up"] is None, "nothing left to ask once the essentials are in"


def test_missing_essentials_are_reported_individually():
    result = extract_requirements(
        "somewhere warm, maybe a long weekend",
        extractor=fake_extractor(follow_up_question="Which city did you have in mind?"),
    )
    assert set(result["missing"]) == {"destination", "dates", "budget"}
    assert result["follow_up"] == "Which city did you have in mind?"


def test_partial_dates_still_count_as_missing():
    """A start with no end is not a planned trip."""
    result = extract_requirements(
        "Rome from the 4th",
        extractor=fake_extractor(
            destination="Rome", start_date=date(2026, 5, 4), budget_total=800, currency="EUR"
        ),
    )
    assert "dates" in result["missing"]


def test_unset_fields_are_omitted_not_defaulted():
    """Nulls must not become zeros — an unmentioned budget is not a €0 budget."""
    result = extract_requirements(
        "Athens", extractor=fake_extractor(destination="Athens")
    )
    assert "budget" not in result["fields"]
    assert "adults" not in result["fields"]


@pytest.mark.parametrize(
    ("previous", "addition", "expected"),
    [
        ("", "hello", "hello"),
        ("hello", "", "hello"),
        ("hello", "world", "hello\nworld"),
        ("  hello  ", "  world  ", "hello\nworld"),
    ],
)
def test_transcript_merging(previous, addition, expected):
    assert merge_transcript(previous, addition) == expected


def test_extraction_failure_surfaces_as_a_ui_state_not_a_crash():
    """A flaky extraction must not take the whole speech session down."""
    from fastapi.testclient import TestClient

    from wayfinder.server import create_app

    client = TestClient(create_app())
    # No credentials in the test environment, so the real extractor will fail.
    response = client.post("/api/extract", json={"transcript": "lisbon in october"})
    assert response.status_code == 200
    body = response.json()
    assert "fields" in body and "missing" in body

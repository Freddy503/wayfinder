"""Pictures for the places in a plan.

A schedule of names in a table does not make anyone want to go anywhere. The
hard part is not fetching an image, it is not fetching the *wrong* one: near
the Van Gogh Museum, Wikipedia's geosearch also returns *Wheatfield with Crows*
and the *Stedelijk Museum*.
"""

from __future__ import annotations

import pytest

from wayfinder.tools import photos


@pytest.fixture(autouse=True)
def user_agent(monkeypatch):
    monkeypatch.setenv("WAYFINDER_USER_AGENT", "wayfinder-tests (test@example.com)")


@pytest.fixture
def wiki(monkeypatch):
    """A stubbed Wikipedia. Records what was asked for."""
    state = {"nearby": {}, "summaries": {}, "asked": []}

    def fake_nearby(lat, lon):
        state["asked"].append(("geosearch", lat, lon))
        return state["nearby"].get((round(lat, 4), round(lon, 4)), [])

    def fake_summary(title):
        state["asked"].append(("summary", title))
        return state["summaries"].get(title)

    monkeypatch.setattr(photos, "_nearby", fake_nearby)
    monkeypatch.setattr(photos, "_summary", fake_summary)
    return state


def article(title, extract="A museum."):
    return {"thumbnail": f"https://img/{title}.jpg", "extract": extract,
            "title": title, "source": f"https://en.wikipedia.org/wiki/{title}"}


# --------------------------------------------------------------------------
# Picking the right article
# --------------------------------------------------------------------------


def test_a_landmark_gets_its_photo(wiki):
    wiki["nearby"][(52.3584, 4.8811)] = ["Van Gogh Museum"]
    wiki["summaries"]["Van Gogh Museum"] = article("Van Gogh Museum")

    got = photos.venue_photo("Van Gogh Museum", 52.3584, 4.8811)
    assert got["kind"] == "photo"
    assert got["thumbnail"].endswith(".jpg")
    assert got["extract"]


def test_the_name_picks_between_what_is_nearby(wiki):
    """Geosearch alone would take the first result, which near the Van Gogh
    Museum is one of the paintings hanging inside it."""
    wiki["nearby"][(52.3584, 4.8811)] = [
        "Wheatfield with Crows", "Stedelijk Museum Amsterdam", "Van Gogh Museum",
    ]
    wiki["summaries"]["Van Gogh Museum"] = article("Van Gogh Museum")
    wiki["summaries"]["Wheatfield with Crows"] = article("Wheatfield with Crows")

    got = photos.venue_photo("Van Gogh Museum", 52.3584, 4.8811)
    assert got["title"] == "Van Gogh Museum"


def test_a_nearby_article_about_something_else_is_refused(wiki):
    """The café over the road from the museum must not borrow its photo."""
    wiki["nearby"][(52.3584, 4.8811)] = ["Stedelijk Museum Amsterdam"]
    wiki["summaries"]["Stedelijk Museum Amsterdam"] = article("Stedelijk Museum Amsterdam")

    got = photos.venue_photo("Café Loetje", 52.3584, 4.8811)
    assert got["kind"] == "map", "should fall back rather than show the wrong building"


def test_a_direct_lookup_covers_venues_with_no_coordinates(wiki):
    wiki["summaries"]["Rijksmuseum"] = article("Rijksmuseum")
    assert photos.venue_photo("Rijksmuseum")["kind"] == "photo"


def test_a_direct_lookup_still_has_to_match_the_name(wiki):
    """Wikipedia redirects freely; "Bocca" could land anywhere."""
    wiki["summaries"]["Bocca"] = article("Mouth", extract="An anatomical structure.")
    assert photos.venue_photo("Bocca")["found"] is False


def test_an_article_with_no_picture_is_not_a_photo(wiki):
    wiki["nearby"][(52.3584, 4.8811)] = ["Some Place"]
    wiki["summaries"]["Some Place"] = None      # `_summary` returns None without a thumbnail
    assert photos.venue_photo("Some Place", 52.3584, 4.8811)["kind"] == "map"


# --------------------------------------------------------------------------
# The fallback
# --------------------------------------------------------------------------


def test_a_restaurant_gets_a_map_of_where_it_is(wiki):
    """Verified against the real API: Cervejaria Ramiro 404s and a geosearch
    around it comes back empty. A map tile is honest; a stock photo is not."""
    got = photos.venue_photo("Cervejaria Ramiro", 38.7211, -9.1367)
    assert got["kind"] == "map"
    assert got["thumbnail"].startswith("https://")
    assert got["extract"] == "", "no borrowed description either"


def test_with_neither_an_article_nor_a_location_there_is_nothing_to_show(wiki):
    assert photos.venue_photo("Somewhere")["kind"] == "none"


def test_the_map_tile_points_at_the_right_place():
    """Slippy-map maths: Amsterdam and Lisbon must not resolve to one tile."""
    assert photos.map_tile(52.3584, 4.8811) != photos.map_tile(38.7211, -9.1367)
    assert "/15/" in photos.map_tile(52.3584, 4.8811)


@pytest.mark.parametrize("lat", [90.0, -90.0])
def test_the_poles_do_not_break_the_projection(lat):
    """Mercator diverges at the poles; the clamp keeps it finite."""
    assert photos.map_tile(lat, 0.0).startswith("https://")


# --------------------------------------------------------------------------
# Never costing anything else
# --------------------------------------------------------------------------


def test_a_wikipedia_outage_falls_back_rather_than_raising(monkeypatch):
    def boom(*a, **kw):
        raise ConnectionError("wikipedia is down")

    monkeypatch.setattr(photos, "_nearby", boom)
    monkeypatch.setattr(photos, "_summary", boom)
    got = photos.venue_photo("Van Gogh Museum", 52.3584, 4.8811)
    assert got["kind"] == "map", "a picture is a nicety; losing it must cost nothing else"


def test_a_venue_scheduled_twice_is_looked_up_once(wiki):
    wiki["nearby"][(52.3584, 4.8811)] = ["Van Gogh Museum"]
    wiki["summaries"]["Van Gogh Museum"] = article("Van Gogh Museum")

    photos.photos_for([
        {"name": "Van Gogh Museum", "lat": 52.3584, "lon": 4.8811},
        {"name": "Van Gogh Museum", "lat": 52.3584, "lon": 4.8811},
    ])
    assert len([a for a in wiki["asked"] if a[0] == "geosearch"]) == 1


def test_nameless_venues_are_skipped(wiki):
    assert photos.photos_for([{"name": "  ", "lat": 1, "lon": 2}]) == {}


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def test_the_endpoint_returns_a_photo_per_venue(tmp_path, monkeypatch):
    import json

    from fastapi.testclient import TestClient

    import wayfinder.server as server
    from tests.conftest import item, make_itinerary
    from wayfinder.server import create_app

    monkeypatch.setattr(server, "RUNS_ROOT", tmp_path.resolve())
    monkeypatch.setattr(
        photos, "venue_photo",
        lambda name, lat=None, lon=None: {"found": True, "kind": "photo",
                                          "thumbnail": f"https://img/{name}.jpg",
                                          "extract": "", "title": name},
    )
    run = "lisbon-2026-10-12-20260806T000000"
    (tmp_path / run / "research").mkdir(parents=True)
    plan = make_itinerary([item("10:00", "12:00", "Belfry", venue="Belfry")])
    (tmp_path / run / "itinerary.json").write_text(
        json.dumps(plan.model_dump(mode="json")), encoding="utf-8")

    body = TestClient(create_app()).get(f"/api/runs/{run}/photos").json()
    assert body["photos"]["Belfry"]["thumbnail"].endswith("Belfry.jpg")


def test_a_run_with_no_itinerary_returns_no_photos(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import wayfinder.server as server
    from wayfinder.server import create_app

    monkeypatch.setattr(server, "RUNS_ROOT", tmp_path.resolve())
    run = "lisbon-2026-10-12-20260806T000001"
    (tmp_path / run / "research").mkdir(parents=True)
    assert TestClient(create_app()).get(f"/api/runs/{run}/photos").json() == {"photos": {}}


def test_photos_are_not_part_of_the_itinerary_schema():
    """The itinerary is the agent's artifact. It must never be asked to invent
    an image URL, and `extra="forbid"` stays as it is."""
    from wayfinder.schema import Venue

    assert "photo" not in Venue.model_fields
    assert "thumbnail" not in Venue.model_fields

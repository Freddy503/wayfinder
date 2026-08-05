"""Serving the subagents' notes while the run is still going.

A five-day Amsterdam trip took 52 minutes. The researchers finished at minute
17 and the itinerary appeared at minute 53 — half an hour of watching a
spinner with three finished documents sitting on disk the whole time.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from wayfinder.server import RESEARCH_ICONS, RESEARCH_TITLES, create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    import wayfinder.server as server

    monkeypatch.setattr(server, "RUNS_ROOT", tmp_path.resolve())
    return TestClient(create_app())


def make_run(tmp_path, name="lisbon-2026-10-12-20260805T000000", **files):
    run = tmp_path / name
    (run / "research").mkdir(parents=True, exist_ok=True)
    for stem, text in files.items():
        (run / "research" / f"{stem}.md").write_text(text, encoding="utf-8")
    return name


def test_notes_are_served_as_soon_as_they_land(client, tmp_path):
    run = make_run(tmp_path, food="## Seafood\n- Ramiro — great", logistics="## Trams\n- Line 28")
    body = client.get(f"/api/runs/{run}/research").json()
    assert {n["name"] for n in body["notes"]} == {"food", "logistics"}
    assert "Ramiro" in next(n["markdown"] for n in body["notes"] if n["name"] == "food")


def test_each_note_is_named_for_a_traveller(client, tmp_path):
    run = make_run(tmp_path, food="x" * 20, neighborhoods="y" * 20, logistics="z" * 20)
    got = {n["name"]: (n["title"], n["icon"]) for n in client.get(f"/api/runs/{run}/research").json()["notes"]}
    assert got["food"] == (RESEARCH_TITLES["food"], RESEARCH_ICONS["food"])
    assert got["neighborhoods"][0] == "Neighbourhoods & sights"


def test_an_unrecognised_file_still_gets_a_readable_tab(client, tmp_path):
    """Subagents choose their own filenames; an unknown one must not render
    as a raw stem or vanish."""
    run = make_run(tmp_path, museum_hours="## Mondays\n- Most are shut")
    note = client.get(f"/api/runs/{run}/research").json()["notes"][0]
    assert note["title"] == "Museum Hours"
    assert note["icon"]


def test_an_empty_file_is_not_offered(client, tmp_path):
    """The directory is created up front, so a file can exist before its
    subagent has written anything — an empty tab is worse than no tab."""
    run = make_run(tmp_path, food="", logistics="## Trams\n- Line 28")
    assert [n["name"] for n in client.get(f"/api/runs/{run}/research").json()["notes"]] == ["logistics"]


def test_a_run_with_no_research_yet_returns_an_empty_list(client, tmp_path):
    run = make_run(tmp_path)
    assert client.get(f"/api/runs/{run}/research").json() == {"notes": []}


def test_size_lets_the_browser_skip_an_identical_re_render(client, tmp_path):
    """This is polled every 8 seconds; re-rendering unchanged markdown would
    reset your scroll position while you were reading."""
    run = make_run(tmp_path, food="## A\n- one")
    first = client.get(f"/api/runs/{run}/research").json()["notes"][0]
    assert first["size"] == len("## A\n- one")

    (tmp_path / run / "research" / "food.md").write_text("## A\n- one\n- two", encoding="utf-8")
    second = client.get(f"/api/runs/{run}/research").json()["notes"][0]
    assert second["size"] != first["size"]


def test_an_unknown_run_is_a_404(client):
    assert client.get("/api/runs/nope/research").status_code == 404


def test_the_run_directory_is_not_a_traversal_surface(client, tmp_path):
    (tmp_path / "secret.md").write_text("not yours", encoding="utf-8")
    make_run(tmp_path)
    for attempt in ("../..", "..%2f..", "/etc"):
        assert client.get(f"/api/runs/{attempt}/research").status_code in (400, 404)


# --------------------------------------------------------------------------
# The draft itinerary, mid-run
# --------------------------------------------------------------------------


def write_itinerary(tmp_path, run, text, where="."):
    target = tmp_path / run / where
    target.mkdir(parents=True, exist_ok=True)
    (target / "itinerary.json").write_text(text, encoding="utf-8")


def test_a_draft_is_served_before_the_run_finishes(client, tmp_path):
    """The agent writes the itinerary well before it finishes and then repairs
    it two or three times. None of that was visible: the map stayed empty for
    most of a run that had a plannable draft for the last third of it."""
    import json as _json

    from tests.conftest import item, make_itinerary

    run = make_run(tmp_path)
    payload = make_itinerary([item("10:00", "12:00", "Belfry")]).model_dump(mode="json")
    write_itinerary(tmp_path, run, _json.dumps(payload))

    body = client.get(f"/api/runs/{run}/draft").json()
    assert body["ready"] is True
    assert body["itinerary"]["days"][0]["items"][0]["title"] == "Belfry"


def test_a_file_caught_mid_write_is_not_an_error(client, tmp_path):
    """It is a file caught mid-write. Say so and let the caller try again —
    raising would turn a poll into a crash every few seconds."""
    run = make_run(tmp_path)
    write_itinerary(tmp_path, run, '{"destination": "Bruges", "days": [')
    body = client.get(f"/api/runs/{run}/draft").json()
    assert body == {"ready": False, "reason": "being written"}


def test_nothing_written_yet_is_reported_plainly(client, tmp_path):
    run = make_run(tmp_path)
    assert client.get(f"/api/runs/{run}/draft").json()["ready"] is False


def test_a_parsed_but_empty_itinerary_is_not_shown(client, tmp_path):
    """Valid JSON with no days would render as a trip with nothing in it."""
    run = make_run(tmp_path)
    write_itinerary(tmp_path, run, '{"destination": "Bruges", "days": []}')
    assert client.get(f"/api/runs/{run}/draft").json()["reason"] == "no days yet"


def test_a_draft_written_to_the_wrong_path_is_still_found(client, tmp_path):
    """Same `/root/itinerary.json` habit that made a finished plan read as no
    plan at all."""
    import json as _json

    from tests.conftest import item, make_itinerary

    run = make_run(tmp_path)
    payload = make_itinerary([item("10:00", "12:00", "Belfry")]).model_dump(mode="json")
    write_itinerary(tmp_path, run, _json.dumps(payload), where="root")
    assert client.get(f"/api/runs/{run}/draft").json()["ready"] is True


def test_size_lets_the_browser_skip_an_unchanged_draft(client, tmp_path):
    import json as _json

    from tests.conftest import item, make_itinerary

    run = make_run(tmp_path)
    one = make_itinerary([item("10:00", "12:00", "Belfry")]).model_dump(mode="json")
    write_itinerary(tmp_path, run, _json.dumps(one))
    first = client.get(f"/api/runs/{run}/draft").json()["size"]

    two = make_itinerary([
        item("10:00", "12:00", "Belfry"), item("13:00", "14:00", "Markt"),
    ]).model_dump(mode="json")
    write_itinerary(tmp_path, run, _json.dumps(two))
    assert client.get(f"/api/runs/{run}/draft").json()["size"] != first


def test_an_unknown_run_is_a_404_not_an_empty_draft(client):
    assert client.get("/api/runs/nope/draft").status_code == 404


# --------------------------------------------------------------------------
# Putting the city on the map straight away
# --------------------------------------------------------------------------


def test_the_destination_is_focused_before_planning_starts(monkeypatch):
    """Measured: the first real map point arrived 191 seconds in, because the
    researchers spend the opening minutes searching rather than geocoding. One
    cached geocode costs 0.3s and fills the pane immediately."""
    from wayfinder.server import PlanRequest, RunSession
    from wayfinder.tools import geo

    monkeypatch.setattr(
        geo, "geocode",
        lambda place: {"found": True, "name": "Ghent", "lat": 51.05, "lon": 3.72},
    )
    session = RunSession(PlanRequest(spec={
        "destination": "Ghent, Belgium",
        "dates": {"start": "2026-10-12", "end": "2026-10-13"},
        "budget": {"currency": "EUR", "total": 260},
    }))
    session._focus_destination()
    event = session.queue.get_nowait()
    assert event.type == "map.focus"
    assert (event.data["lat"], event.data["lon"]) == (51.05, 3.72)


def test_an_ungeocodable_destination_does_not_stop_the_run(monkeypatch):
    """The map fills in from the research anyway; a city we can't locate is
    not a reason to fail a trip."""
    from wayfinder.server import PlanRequest, RunSession
    from wayfinder.tools import geo

    def boom(place):
        raise ConnectionError("nominatim is down")

    monkeypatch.setattr(geo, "geocode", boom)
    session = RunSession(PlanRequest(spec={
        "destination": "Atlantis",
        "dates": {"start": "2026-10-12", "end": "2026-10-13"},
        "budget": {"currency": "EUR", "total": 260},
    }))
    session._focus_destination()          # must not raise
    assert "error" in session.queue.get_nowait().data


def test_a_miss_emits_nothing_to_plot(monkeypatch):
    from wayfinder.server import PlanRequest, RunSession
    from wayfinder.tools import geo

    monkeypatch.setattr(geo, "geocode", lambda place: {"found": False, "query": place})
    session = RunSession(PlanRequest(spec={
        "destination": "Nowhere At All",
        "dates": {"start": "2026-10-12", "end": "2026-10-13"},
        "budget": {"currency": "EUR", "total": 260},
    }))
    session._focus_destination()
    assert session.queue.empty()

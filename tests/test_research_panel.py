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

"""Changing a trip that already exists.

A real trip is planned iteratively: you read the plan, you want Tuesday's
dinner moved, you want a day trip added. Starting over throws away research
that is still perfectly good — and, at 56 output tokens a second, costs
fifteen minutes to rediscover what is already on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import item, make_itinerary
from wayfinder.agent import refine_message, version_itinerary
from wayfinder.prompts import MAIN_PROMPT, REFINE_PROMPT
from wayfinder.server import create_app

SPEC_YAML = """\
destination: Amsterdam, Netherlands
dates: {start: 2026-11-05, end: 2026-11-06}
party: {adults: 2}
budget: {currency: EUR, total: 400}
"""


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A finished run on disk: spec, itinerary and research."""
    import wayfinder.server as server

    monkeypatch.setattr(server, "RUNS_ROOT", tmp_path.resolve())
    name = "amsterdam-2026-11-05-20260806T000000"
    directory = tmp_path / name
    (directory / "research").mkdir(parents=True)
    (directory / "spec.yaml").write_text(SPEC_YAML, encoding="utf-8")
    (directory / "research" / "food.md").write_text(
        "## Dinner\n- Wilde Zwijnen — Oost, mid-range", encoding="utf-8")
    plan = make_itinerary([item("10:00", "12:00", "Van Gogh Museum")])
    (directory / "itinerary.json").write_text(
        json.dumps(plan.model_dump(mode="json"), indent=2), encoding="utf-8")
    return name, directory


# --------------------------------------------------------------------------
# Versioning
# --------------------------------------------------------------------------


def test_the_itinerary_is_copied_aside_before_it_changes(run):
    _, directory = run
    archived = version_itinerary(directory)
    assert archived == directory / "itinerary.v1.json"
    assert json.loads(archived.read_text())["destination"]


def test_versions_accumulate_rather_than_overwrite(run):
    """Undo has to reach further back than one step."""
    _, directory = run
    assert version_itinerary(directory).name == "itinerary.v1.json"
    assert version_itinerary(directory).name == "itinerary.v2.json"
    assert version_itinerary(directory).name == "itinerary.v3.json"


def test_nothing_to_version_is_not_an_error(tmp_path):
    assert version_itinerary(tmp_path) is None


# --------------------------------------------------------------------------
# The endpoint
# --------------------------------------------------------------------------


def test_refining_reuses_the_run_directory(run, monkeypatch):
    """The entire point: the research stays where it is."""
    name, directory = run
    started = {}

    import wayfinder.server as server

    class FakeSession:
        def __init__(self, request, *, run_dir=None, refine=None):
            started["run_dir"] = run_dir
            started["refine"] = refine
            self.id = "refined-1"

        def start(self):
            started["started"] = True

    monkeypatch.setattr(server, "RunSession", FakeSession)
    body = TestClient(create_app()).post(
        f"/api/refine/{name}", json={"request": "move dinner later"}).json()

    assert body["run_id"] == "refined-1"
    assert started["run_dir"] == directory
    assert started["refine"] == "move dinner later"
    assert started["started"]


def test_refining_versions_the_itinerary_first(run, monkeypatch):
    name, directory = run
    import wayfinder.server as server

    monkeypatch.setattr(
        server, "RunSession",
        lambda *a, **kw: type("S", (), {"id": "x", "start": lambda self: None})(),
    )
    TestClient(create_app()).post(f"/api/refine/{name}", json={"request": "later"})
    assert (directory / "itinerary.v1.json").exists()
    assert (directory / "research" / "food.md").exists(), "research must survive"


def test_a_refinement_gets_its_own_run_id(run, monkeypatch):
    """Its own stream and progress, against the same trip on disk."""
    name, _ = run
    import wayfinder.server as server

    monkeypatch.setattr(
        server, "RunSession",
        lambda *a, **kw: type("S", (), {"id": "new-id", "start": lambda self: None})(),
    )
    body = TestClient(create_app()).post(f"/api/refine/{name}", json={"request": "x"}).json()
    assert body["run_id"] != name


def test_refining_a_run_with_no_spec_is_a_404(tmp_path, monkeypatch):
    import wayfinder.server as server

    monkeypatch.setattr(server, "RUNS_ROOT", tmp_path.resolve())
    (tmp_path / "empty-run" / "research").mkdir(parents=True)
    res = TestClient(create_app()).post("/api/refine/empty-run", json={"request": "x"})
    assert res.status_code == 404


def test_an_empty_request_is_refused(run):
    name, _ = run
    assert TestClient(create_app()).post(
        f"/api/refine/{name}", json={"request": ""}).status_code == 422


# --------------------------------------------------------------------------
# What the agent is told
# --------------------------------------------------------------------------


def test_the_refine_prompt_is_about_changing_not_planning():
    """Handing the planning prompt to someone who wants a dinner moved is how
    you get a trip replanned from scratch."""
    assert "already exists" in REFINE_PROMPT
    assert "minimum that satisfies the request" in REFINE_PROMPT
    assert "Workflow" not in REFINE_PROMPT, "no dispatch-and-shortlist here"
    assert REFINE_PROMPT != MAIN_PROMPT


def test_it_is_told_to_read_what_is_already_on_disk():
    for expected in ("itinerary.json", "research/*.md", "Read them"):
        assert expected in REFINE_PROMPT


def test_it_is_told_when_to_search_again_and_when_not_to():
    """The difference between a change that takes seconds and one that takes
    minutes — and it has to say which it did."""
    assert "Only when the request needs a fact you do not already have" in REFINE_PROMPT
    assert "Haarlem" in REFINE_PROMPT, "name a case that genuinely needs research"
    assert "say so in your reply" in REFINE_PROMPT


def test_it_is_told_to_patch_rather_than_regenerate():
    assert "edit_file" in REFINE_PROMPT
    assert "tens of output tokens" in REFINE_PROMPT


def test_a_refinement_is_still_checked():
    """Moving a dinner later can push it past a closing time."""
    assert "check_itinerary" in REFINE_PROMPT
    assert "broken plan" in REFINE_PROMPT


def test_the_user_turn_carries_the_request_and_the_constraints(spec):
    """The constraints still bind. The itinerary deliberately does not go in —
    it is on disk, and pasting 7,000 tokens per turn would cost more than the
    change it is meant to make."""
    message = refine_message(spec, "move Tuesday's dinner an hour later")
    assert "move Tuesday's dinner an hour later" in message
    assert "budget" in message
    assert "Van Gogh" not in message


# --------------------------------------------------------------------------
# Patching over rewriting
# --------------------------------------------------------------------------


def test_the_planner_is_told_to_patch_its_repairs():
    """Measured: six full rewrites in one two-day run, the biggest 7,050
    output tokens and 146 seconds, at 56 tokens a second."""
    assert "edit_file`, not by \\\nwriting the file again" in MAIN_PROMPT or \
           "edit_file" in MAIN_PROMPT
    assert "7,000 tokens" in MAIN_PROMPT
    assert "two and a half minutes" in MAIN_PROMPT


def test_violations_carry_a_location_the_agent_can_patch_against():
    """`where` is what makes an exact `old_string` possible without re-reading
    the file."""
    from wayfinder.verify import Violation

    payload = Violation("budget", "hard", "over", where="2026-11-06 12:15 Bike rental").to_dict()
    assert payload["where"] == "2026-11-06 12:15 Bike rental"


# --------------------------------------------------------------------------
# The developer toggle
# --------------------------------------------------------------------------


PAGE = Path(__file__).resolve().parent.parent / "wayfinder" / "web" / "index.html"


def test_the_developer_toggle_is_not_inside_the_intake_form():
    """`body.planning #intake { display: none }` hides the form the moment a
    run starts, so a toggle in there could only be switched on before planning
    — exactly when you don't yet know you want it."""
    page = PAGE.read_text(encoding="utf-8")
    intake = page.split('<div id="intake">')[1].split("<!-- /#intake -->")[0]
    assert 'id="o-dev"' not in intake


def test_it_lives_in_the_progress_rail_instead():
    """Where the technical detail already is, and visible for the whole run."""
    page = PAGE.read_text(encoding="utf-8")
    # To `#rail-body`, not the first `</div>` — the header contains a spacer.
    head = page.split('<div id="rail-head">')[1].split('<div id="rail-body">')[0]
    assert 'id="o-dev"' in head

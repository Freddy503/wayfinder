"""Batch tools — the fix for a two-day trip taking thirteen minutes.

The single-item tools were always correct. What made them expensive was the
shape: one model turn per venue, per hop, per rating. A measured Bruges run
spent 184 tool calls and 13m22s on two days, of which 74 were geocode /
travel / rating lookups that are cached arithmetic once the coordinates are in.

Prompt wording failed to move this twice, so the tools changed instead.
"""

from __future__ import annotations

import pytest

from wayfinder.tools import geo, ratings


@pytest.fixture
def fake_places(monkeypatch):
    """Coordinates without the network. Records what was asked for."""
    known = {
        "belfort, bruges": (51.2087, 3.2247),
        "groeningemuseum, bruges": (51.2050, 3.2266),
        "markt, bruges": (51.2093, 3.2247),
    }
    asked: list[str] = []

    def fake(query: str):
        asked.append(query)
        hit = known.get(query.strip().lower())
        if hit is None:
            return None
        return {"name": query, "display_name": query, "lat": hit[0], "lon": hit[1]}

    monkeypatch.setattr(geo, "_geocode_raw", fake)
    return asked


def test_one_call_locates_the_whole_shortlist(fake_places):
    out = geo.geocode_all(["Belfort, Bruges", "Groeningemuseum, Bruges"])
    assert out["found"] == 2
    assert out["missing"] == []
    assert [r["name"] for r in out["results"]] == [
        "Belfort, Bruges", "Groeningemuseum, Bruges",
    ]


def test_results_stay_in_the_order_they_were_asked_for(fake_places):
    """The model matches results back to its own list by position."""
    places = ["Markt, Bruges", "Nowhere At All", "Belfort, Bruges"]
    out = geo.geocode_all(places)
    assert [r.get("name") or r["query"] for r in out["results"]] == places


def test_what_could_not_be_found_is_called_out(fake_places):
    out = geo.geocode_all(["Belfort, Bruges", "Nowhere At All"])
    assert out["missing"] == ["Nowhere At All"]
    assert out["found"] == 1


def test_blank_entries_are_dropped_not_looked_up(fake_places):
    geo.geocode_all(["Belfort, Bruges", "", "   "])
    assert fake_places == ["Belfort, Bruges"]


def test_a_bare_string_still_works(fake_places):
    """Models pass a scalar where a list is wanted often enough to matter."""
    assert geo.geocode_all("Belfort, Bruges")["found"] == 1


def test_an_oversized_batch_is_refused_with_advice(fake_places):
    out = geo.geocode_all([f"Place {i}, Bruges" for i in range(geo.MAX_BATCH + 1)])
    assert "shortlist" in out["error"]
    assert out["results"] == []
    assert fake_places == [], "nothing should have been looked up"


def test_one_call_times_the_whole_route(fake_places):
    legs = [
        {"origin": "Markt, Bruges", "destination": "Belfort, Bruges"},
        {"origin": "Belfort, Bruges", "destination": "Groeningemuseum, Bruges"},
    ]
    out = geo.estimate_travel_all(legs)
    assert len(out["results"]) == 2
    assert out["failed"] == []
    assert out["total_minutes"] == sum(r["minutes"] for r in out["results"])


def test_each_leg_reports_which_hop_it_was(fake_places):
    """Without this the model has to match by position and gets it wrong."""
    out = geo.estimate_travel_all(
        [{"origin": "Markt, Bruges", "destination": "Belfort, Bruges"}]
    )
    assert out["results"][0]["origin"] == "Markt, Bruges"
    assert out["results"][0]["destination"] == "Belfort, Bruges"


def test_a_leg_can_override_the_default_mode(fake_places):
    out = geo.estimate_travel_all(
        [
            {"origin": "Markt, Bruges", "destination": "Groeningemuseum, Bruges"},
            {"origin": "Markt, Bruges", "destination": "Groeningemuseum, Bruges",
             "mode": "taxi"},
        ],
        mode="walk",
    )
    assert out["results"][0]["mode"] == "walk"
    assert out["results"][1]["mode"] == "taxi"
    assert out["results"][1]["minutes"] < out["results"][0]["minutes"]


def test_one_unlocatable_leg_does_not_sink_the_batch(fake_places):
    """Thirty-nine good hops shouldn't be lost to one bad name."""
    out = geo.estimate_travel_all([
        {"origin": "Markt, Bruges", "destination": "Belfort, Bruges"},
        {"origin": "Markt, Bruges", "destination": "Nowhere At All"},
    ])
    assert len(out["results"]) == 2
    assert out["results"][0]["ok"] is True
    assert len(out["failed"]) == 1
    assert "Nowhere At All" in out["failed"][0]["reason"]


def test_malformed_legs_are_skipped_rather_than_raising(fake_places):
    out = geo.estimate_travel_all([
        "not a dict",
        {"origin": "Markt, Bruges"},                     # no destination
        {"origin": "Markt, Bruges", "destination": "Belfort, Bruges"},
    ])
    assert len(out["results"]) == 1


def test_batch_matches_single_call_exactly(fake_places):
    """One implementation, two entry points — if these ever diverge, the
    checker and the agent stop agreeing about travel times."""
    single = geo.estimate_travel("Markt, Bruges", "Groeningemuseum, Bruges", "transit")
    batch = geo.estimate_travel_all(
        [{"origin": "Markt, Bruges", "destination": "Groeningemuseum, Bruges"}],
        mode="transit",
    )["results"][0]
    assert single["minutes"] == batch["minutes"]
    assert single["distance_km"] == batch["distance_km"]


def test_ratings_batch_carries_the_venue_back(monkeypatch):
    monkeypatch.setattr(
        ratings, "venue_rating",
        lambda venue, city: {"found": True, "score": 4.5, "count": 10, "city": city},
    )
    out = ratings.venue_ratings(
        [{"venue": "Bocca", "city": "Bruges"}, "That's Toast"], city="Bruges"
    )
    assert [r["venue"] for r in out["results"]] == ["Bocca", "That's Toast"]
    assert out["results"][1]["city"] == "Bruges", "the default city has to apply"
    assert out["found"] == 2


def test_ratings_batch_has_a_limit_too(monkeypatch):
    called = []
    monkeypatch.setattr(
        ratings, "venue_rating",
        lambda venue, city: called.append(venue) or {"found": False},
    )
    out = ratings.venue_ratings([f"V{i}" for i in range(ratings.MAX_BATCH + 1)])
    assert "shortlist" in out["error"]
    assert called == []


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def test_the_agent_is_offered_the_batch_tools_first(tmp_path, monkeypatch):
    """A model choosing a tool reads the list top-down, and one turn per venue
    is exactly what this change exists to prevent."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    captured = {}
    import wayfinder.agent as agent_module

    real = agent_module.create_deep_agent
    monkeypatch.setattr(
        agent_module, "create_deep_agent",
        lambda **kw: captured.update(kw) or real(**kw),
    )
    from tests.conftest import make_spec

    agent_module.build_agent(
        make_spec(), tmp_path,
        agent_module.AgentConfig(use_subagents=False, use_skills=False),
        {"calls": 0},
    )
    names = [getattr(t, "__name__", "") for t in captured["tools"]]
    for batch, single in (
        ("geocode_all", "geocode"),
        ("estimate_travel_all", "estimate_travel"),
        ("venue_ratings", "venue_rating"),
    ):
        assert names.index(batch) < names.index(single), f"{batch} must precede {single}"


def test_the_prompt_tells_the_agent_to_batch():
    from wayfinder.prompts import MAIN_PROMPT

    for tool in ("geocode_all", "venue_ratings", "estimate_travel_all"):
        assert tool in MAIN_PROMPT
    assert "don't redo it" in MAIN_PROMPT.lower(), "re-searching was the other half"


def test_a_batch_counts_as_one_verification_call():
    """It costs one turn, and turns are what the metric is measuring."""
    from wayfinder.evals.run import VERIFY_TOOLS

    assert {"geocode_all", "estimate_travel_all", "venue_ratings"} <= VERIFY_TOOLS


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("geocode_all", {"places": ["a", "b", "c"]}, "Locating the shortlist (3 places)"),
        ("estimate_travel_all", {"legs": [{}, {}]}, "Timing the route (2 hops)"),
        ("venue_ratings", {"venues": ["x"]}, "Checking reviews (1 venues)"),
    ],
)
def test_batches_are_narrated_by_size(name, args, expected):
    """A first-item preview would misrepresent the other fifteen."""
    from wayfinder.stream import narrate

    assert narrate(name, args) == expected

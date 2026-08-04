"""Changing constraints while a run is in flight."""

from __future__ import annotations

import pytest

from wayfinder.adjust import (
    ADJUSTABLE,
    AdjustmentError,
    LiveSpec,
    apply_changes,
    summarise_constraints,
)
from wayfinder.schema import TripSpec


@pytest.fixture
def spec() -> TripSpec:
    return TripSpec.model_validate(
        {
            "destination": "Lisbon, Portugal",
            "dates": {"start": "2026-10-12", "end": "2026-10-16"},
            "budget": {"currency": "EUR", "total": 900},
            "party": {"adults": 2},
            "constraints": {"max_transit_minutes": 35, "earliest_start": "10:00"},
            "must_do": ["Time Out Market", "Belém Tower"],
        }
    )


# --------------------------------------------------------------------------
# Applying a change
# --------------------------------------------------------------------------


def test_raising_the_budget(spec):
    updated, notes = apply_changes(spec, {"budget": 1050})
    assert updated.budget.total == 1050
    assert updated.budget.currency == "EUR", "changing the amount is not changing currency"
    assert notes == ["budget 900 → 1050 EUR"]


def test_the_original_is_untouched(spec):
    """A rejected change has to leave the run exactly as it was, so the
    function that computes the new spec must not mutate the old one."""
    apply_changes(spec, {"budget": 1050})
    assert spec.budget.total == 900


def test_dropping_a_must_do(spec):
    updated, notes = apply_changes(spec, {"must_do": ["Time Out Market"]})
    assert updated.must_do == ["Time Out Market"]
    assert notes == ["dropped must-do: Belém Tower"]


def test_relaxing_a_constraint_to_nothing(spec):
    updated, notes = apply_changes(spec, {"max_transit_minutes": None})
    assert updated.constraints.max_transit_minutes is None
    assert "no limit" in notes[0]


def test_changing_a_time_of_day(spec):
    updated, notes = apply_changes(spec, {"earliest_start": "09:30"})
    assert updated.constraints.earliest_start.strftime("%H:%M") == "09:30"
    assert notes == ["earliest start → 09:30"]


def test_a_change_that_changes_nothing_reports_nothing(spec):
    _, notes = apply_changes(spec, {})
    assert notes == []


@pytest.mark.parametrize(
    "changes",
    [
        {"destination": "Porto"},
        {"dates": {"start": "2026-11-01"}},
        {"party": {"adults": 4}},
    ],
)
def test_what_makes_it_a_different_trip_is_refused(spec, changes):
    """Destination, dates and party aren't adjustments — they invalidate every
    piece of research already done, so they belong in a new run."""
    with pytest.raises(AdjustmentError, match="cannot change"):
        apply_changes(spec, changes)


def test_the_error_lists_what_is_allowed(spec):
    with pytest.raises(AdjustmentError) as exc:
        apply_changes(spec, {"weather": "sunny"})
    for key in ADJUSTABLE:
        assert key in str(exc.value)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"budget": -5}, "greater than zero"),
        ({"budget": "lots"}, "must be a number"),
        ({"pace": "frantic"}, "not a pace"),
        ({"earliest_start": "morning"}, "not a time of day"),
        ({"required_meals": ["brunch"]}, "not a meal"),
    ],
)
def test_bad_values_are_refused_in_the_traveller_s_language(spec, changes, message):
    with pytest.raises(AdjustmentError, match=message):
        apply_changes(spec, changes)


def test_a_rejected_change_leaves_the_live_spec_alone(spec):
    live = LiveSpec(spec=spec)
    with pytest.raises(AdjustmentError):
        live.apply({"budget": 0})
    assert live.current.budget.total == 900
    assert live.take_news() == [], "a failed change must not be announced"


# --------------------------------------------------------------------------
# LiveSpec — what the agent sees
# --------------------------------------------------------------------------


def test_the_agent_is_told_once_and_only_once(spec):
    """Read-once. Repeating it would have the agent re-planning around the same
    change on every single check for the rest of the run."""
    live = LiveSpec(spec=spec)
    live.apply({"budget": 1100})
    assert live.take_news() == ["budget 900 → 1100 EUR"]
    assert live.take_news() == []


def test_changes_accumulate_until_read(spec):
    live = LiveSpec(spec=spec)
    live.apply({"budget": 1100})
    live.apply({"pace": "relaxed"})
    assert len(live.take_news()) == 2


def test_history_survives_being_read(spec):
    """`take_news` drains what the agent hasn't seen; the audit trail stays."""
    live = LiveSpec(spec=spec)
    live.apply({"budget": 1100})
    live.take_news()
    assert len(live.history) == 1
    assert live.history[0]["source"] == "traveller"


def test_concurrent_changes_do_not_lose_one(spec):
    """The traveller's edit arrives on the HTTP thread while the agent reads
    the spec on the worker thread."""
    import threading

    live = LiveSpec(spec=spec)
    barrier = threading.Barrier(8)

    def bump(n):
        barrier.wait()
        live.apply({"budget": 1000 + n})

    threads = [threading.Thread(target=bump, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(live.history) == 8
    assert len(live.take_news()) == 8
    assert live.current.budget.total in {1000 + i for i in range(8)}


# --------------------------------------------------------------------------
# What the panel shows
# --------------------------------------------------------------------------


def test_only_constraints_that_were_actually_set_are_listed(spec):
    keys = {c["key"] for c in summarise_constraints(spec)}
    assert {"budget", "pace", "max_transit_minutes", "earliest_start"} <= keys
    # Never stated, so not a constraint — listing it as "none" invites people
    # to fill it in for the sake of it.
    assert "latest_end" not in keys
    assert "max_walk_km_per_day" not in keys


def test_every_listed_constraint_can_be_changed_back(spec):
    """The panel offers a Change button on each row, so anything it lists has
    to be something `apply_changes` accepts."""
    for entry in summarise_constraints(spec):
        assert entry["key"] in ADJUSTABLE


def test_the_display_reads_as_a_sentence(spec):
    shown = {c["key"]: c["display"] for c in summarise_constraints(spec)}
    assert shown["budget"] == "900 EUR"
    assert shown["max_transit_minutes"] == "under 35 min between stops"
    assert shown["earliest_start"] == "nothing before 10:00"


def test_the_panel_tracks_a_change(spec):
    live = LiveSpec(spec=spec)
    live.apply({"budget": 1200})
    shown = {c["key"]: c["display"] for c in summarise_constraints(live.current)}
    assert shown["budget"] == "1200 EUR"


# --------------------------------------------------------------------------
# The checker reads through the live spec
# --------------------------------------------------------------------------


def test_the_checker_grades_against_the_agreed_budget(tmp_path):
    """The whole point. If the traveller raises the budget mid-run and the
    checker keeps using the old one, the run fails for a reason both parties
    already agreed to drop."""
    import json

    from tests.conftest import item, make_itinerary, make_spec
    from wayfinder.agent import make_check_tool

    live = LiveSpec(spec=make_spec(budget={"currency": "EUR", "total": 900}))
    check = make_check_tool(live, tmp_path, {"calls": 0})

    itinerary = make_itinerary([item("10:00", "12:00", "Belém Tower", cost=1000.0)])
    (tmp_path / "itinerary.json").write_text(
        json.dumps(itinerary.model_dump(mode="json")), encoding="utf-8"
    )

    over = check()
    assert over["metrics"]["schema_valid"] == 1.0, "the fixture itself must parse"
    assert over["metrics"]["budget_overrun_pct"] > 0

    live.apply({"budget": 2000})
    within = check()
    assert within["metrics"]["budget_overrun_pct"] == 0
    assert "budget 900 → 2000 EUR" in within["traveller_changed"]


def test_the_agent_is_told_what_changed_in_the_check_result(spec, tmp_path):
    from wayfinder.agent import make_check_tool

    live = LiveSpec(spec=spec)
    check = make_check_tool(live, tmp_path, {"calls": 0})
    live.apply({"pace": "relaxed"})

    result = check()   # no itinerary written — the news still has to arrive
    assert "traveller_changed" in result
    assert "re-plan" in result["summary"].lower()
    assert "traveller_changed" not in check(), "read-once, or it re-plans forever"


def test_a_plain_spec_still_works(spec, tmp_path):
    """The CLI passes a `TripSpec`; only the server has a live one."""
    from wayfinder.agent import make_check_tool

    result = make_check_tool(spec, tmp_path, {"calls": 0})()
    assert result["passed"] is False
    assert "traveller_changed" not in result


# --------------------------------------------------------------------------
# Wiring: the agent only gets the tool where someone can answer it
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fake_key(monkeypatch):
    """Building the graph constructs a client, which wants a key. None of
    these tests call the model."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")


def build(spec_or_live, tmp_path, **config_kw):
    from langgraph.checkpoint.memory import InMemorySaver

    from wayfinder.agent import AgentConfig, build_agent

    checkpointer = config_kw.pop("checkpointer", InMemorySaver())
    config = AgentConfig(use_subagents=False, use_skills=False, **config_kw)
    agent = build_agent(spec_or_live, tmp_path, config, {"calls": 0}, checkpointer=checkpointer)
    return _tool_names(agent)


def _tool_names(agent):
    """Tool names on the built graph, however deepagents happens to hold them."""
    found = set()
    for node in getattr(agent, "nodes", {}).values():
        for attr in ("bound", "runnable", "func"):
            target = getattr(node, attr, None)
            for holder in ("tools_by_name", "_tools_by_name"):
                mapping = getattr(target, holder, None)
                if isinstance(mapping, dict):
                    found |= set(mapping)
    return found


def test_the_agent_gets_the_tool_when_someone_can_answer(spec, tmp_path):
    names = build(LiveSpec(spec=spec), tmp_path, allow_change_requests=True)
    assert "request_change" in names


def test_no_tool_without_a_live_spec(spec, tmp_path):
    """The CLI passes a fixed spec — a change it accepted could never take
    effect, so offering to ask would be a lie."""
    assert "request_change" not in build(spec, tmp_path, allow_change_requests=True)


def test_no_tool_without_a_checkpointer(spec, tmp_path):
    """An interrupt needs somewhere to park. Offering the agent a question
    nobody can answer would hang the run."""
    names = build(
        LiveSpec(spec=spec), tmp_path, allow_change_requests=True, checkpointer=None
    )
    assert "request_change" not in names


def test_off_by_default_so_evals_still_measure_refusal(spec, tmp_path):
    """`correctly_refused` is a real metric: in the eval matrix nobody is there
    to relax anything, and an agent that asks instead of refusing would park
    forever."""
    from wayfinder.agent import AgentConfig

    assert AgentConfig().allow_change_requests is False
    assert "request_change" not in build(LiveSpec(spec=spec), tmp_path)


def test_asking_always_interrupts(spec, tmp_path):
    """Not optional: without the interrupt the tool returns its own placeholder
    and the agent reads "no response recorded" as the traveller's answer."""
    from langgraph.checkpoint.memory import InMemorySaver

    from wayfinder.agent import AgentConfig, build_agent

    captured = {}
    import deepagents

    real = deepagents.create_deep_agent

    def spy(**kwargs):
        captured.update(kwargs)
        return real(**kwargs)

    import wayfinder.agent as agent_module

    original = agent_module.create_deep_agent
    agent_module.create_deep_agent = spy
    try:
        build_agent(
            LiveSpec(spec=spec),
            tmp_path,
            AgentConfig(
                use_subagents=False, use_skills=False,
                allow_change_requests=True, interrupt_on=(),
            ),
            {"calls": 0},
            checkpointer=InMemorySaver(),
        )
    finally:
        agent_module.create_deep_agent = original

    assert captured["interrupt_on"].get("request_change") is True


# --------------------------------------------------------------------------
# Finding the itinerary wherever the model put it
# --------------------------------------------------------------------------


def test_the_expected_path_wins(tmp_path):
    from wayfinder.agent import locate_itinerary

    (tmp_path / "itinerary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "itinerary.json").write_text("{}", encoding="utf-8")
    assert locate_itinerary(tmp_path) == tmp_path / "itinerary.json"


def test_an_itinerary_one_directory_down_is_still_found(tmp_path):
    """The backend anchors every path under the run directory, so a model that
    writes `/root/itinerary.json` out of habit lands one level down. The plan
    is fine; only the path is wrong, and reporting "never wrote itinerary.json"
    over a complete itinerary is the worst kind of wrong answer."""
    from wayfinder.agent import locate_itinerary

    (tmp_path / "root").mkdir()
    target = tmp_path / "root" / "itinerary.json"
    target.write_text("{}", encoding="utf-8")
    assert locate_itinerary(tmp_path) == target


def test_the_shallowest_copy_wins(tmp_path):
    from wayfinder.agent import locate_itinerary

    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "itinerary.json").write_text("{}", encoding="utf-8")
    shallow = tmp_path / "root"
    shallow.mkdir()
    (shallow / "itinerary.json").write_text("{}", encoding="utf-8")
    assert locate_itinerary(tmp_path) == shallow / "itinerary.json"


def test_skills_never_shadow_a_real_itinerary(tmp_path):
    """`skills/` is copied in by us; a fixture inside one is not the answer."""
    from wayfinder.agent import locate_itinerary

    skills = tmp_path / "skills" / "itinerary-format"
    skills.mkdir(parents=True)
    (skills / "itinerary.json").write_text("{}", encoding="utf-8")
    assert not locate_itinerary(tmp_path).exists()


def test_a_missing_itinerary_still_reports_the_path_we_wanted(tmp_path):
    """Callers keep their "doesn't exist yet" branch, and the message names
    the path the agent was told to write."""
    from wayfinder.agent import ITINERARY_FILE, locate_itinerary

    found = locate_itinerary(tmp_path)
    assert found == tmp_path / ITINERARY_FILE
    assert not found.exists()


def test_the_checker_reads_a_misplaced_itinerary(tmp_path):
    import json

    from tests.conftest import item, make_itinerary, make_spec
    from wayfinder.agent import make_check_tool

    (tmp_path / "root").mkdir()
    itinerary = make_itinerary([item("10:00", "12:00", "Belfry", cost=12.0)])
    (tmp_path / "root" / "itinerary.json").write_text(
        json.dumps(itinerary.model_dump(mode="json")), encoding="utf-8"
    )
    result = make_check_tool(make_spec(), tmp_path, {"calls": 0})()
    assert result["metrics"]["schema_valid"] == 1.0


def test_the_prompt_names_the_path_it_wants():
    """Tolerance is the safety net; the prompt is the fix."""
    from wayfinder.prompts import MAIN_PROMPT

    assert "/root/itinerary.json" in MAIN_PROMPT, "name the mistake to prevent it"

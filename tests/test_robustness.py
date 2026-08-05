"""Failures that took down a whole run, and the map that stopped filling in.

All three came from the same live run: a malformed itinerary raised instead of
being reported, the message then blamed a file that existed, and the map stayed
empty because batching had quietly bypassed the code that plots research.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from tests.conftest import make_spec
from wayfinder.agent import AgentConfig, finalise
from wayfinder.stream import StreamTranslator


def updates(payload, namespace=()):
    return (namespace, "updates", {"model": payload})


def call(name, args, id="c1"):
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


# --------------------------------------------------------------------------
# A malformed itinerary is a finding, not a crash
# --------------------------------------------------------------------------


@pytest.fixture
def truncated(tmp_path):
    """What a cut-off model response leaves behind: valid JSON, then nothing."""
    body = json.dumps({"destination": "Bruges", "currency": "EUR", "days": []}, indent=2)
    (tmp_path / "itinerary.json").write_text(body[: len(body) // 2], encoding="utf-8")
    return tmp_path


def test_a_truncated_itinerary_does_not_raise(truncated):
    """It escaped `finalise` and took the run with it."""
    result = finalise(make_spec(), AgentConfig(), truncated, 1, [], None)
    assert result.report.passed is False
    assert result.itinerary is None


def test_the_failure_is_reported_as_a_schema_violation(truncated):
    result = finalise(make_spec(), AgentConfig(), truncated, 1, [], None)
    assert result.report.metrics["schema_valid"] == 0.0
    assert [v.check for v in result.report.violations] == ["schema_valid"]


def test_the_message_does_not_claim_the_file_is_missing(truncated):
    """It said "the agent never wrote itinerary.json" about a file the agent
    had written — which sends you looking in entirely the wrong place."""
    message = finalise(make_spec(), AgentConfig(), truncated, 1, [], None).report.violations[0].message
    assert "never wrote" not in message
    assert "not valid JSON" in message
    assert "line" in message and "byte" in message, "say where it broke"


def test_a_genuinely_missing_file_still_says_so(tmp_path):
    message = finalise(make_spec(), AgentConfig(), tmp_path, 1, [], None).report.violations[0].message
    assert "never wrote" in message


def test_a_malformed_run_still_writes_its_verdict(truncated):
    """The run directory has to describe what happened even when nothing
    parsed — otherwise there is nothing to look at afterwards."""
    finalise(make_spec(), AgentConfig(), truncated, 1, [], None)
    report = json.loads((truncated / "constraints.json").read_text())
    assert report["passed"] is False


def test_it_is_scoreable_like_any_other_run(truncated):
    from wayfinder.evals.feedback import scores_for

    result = finalise(make_spec(), AgentConfig(), truncated, 1, [], None)
    result.tool_calls = lambda: {}
    scores = scores_for(result)
    assert scores["schema_valid"] == 0.0
    assert scores["plan_passes"] == 0.0


# --------------------------------------------------------------------------
# The map fills in from batch geocodes too
# --------------------------------------------------------------------------


BATCH = {
    "results": [
        {"found": True, "name": "Belfort", "lat": 51.2087, "lon": 3.2247},
        {"found": False, "query": "Nowhere At All"},
        {"found": True, "name": "Markt", "lat": 51.2093, "lon": 3.2247},
    ],
    "found": 2,
    "missing": ["Nowhere At All"],
}


def test_a_batch_geocode_plots_every_hit():
    """`geocode_all` replaced almost every `geocode` call, and only `geocode`
    produced map points — so the map sat blank for most of the run and filled
    in at the end when the finished route was drawn."""
    t = StreamTranslator()
    t.translate(updates({"messages": [AIMessage(content="", tool_calls=[
        call("geocode_all", {"places": ["Belfort, Bruges", "Nowhere At All", "Markt, Bruges"]})])]}))
    events = t.translate(updates({"messages": [
        ToolMessage(content=json.dumps(BATCH), tool_call_id="c1")]}))
    geo = [e for e in events if e.type == "geo"]
    assert [g.data["name"] for g in geo] == ["Belfort", "Markt"]
    assert geo[0].data["lat"] == 51.2087


def test_a_missed_place_is_not_plotted():
    t = StreamTranslator()
    t.translate(updates({"messages": [AIMessage(content="", tool_calls=[
        call("geocode_all", {"places": ["Nowhere At All"]})])]}))
    events = t.translate(updates({"messages": [ToolMessage(
        content=json.dumps({"results": [{"found": False, "query": "Nowhere At All"}]}),
        tool_call_id="c1")]}))
    assert [e for e in events if e.type == "geo"] == []


def test_the_single_tool_still_plots():
    t = StreamTranslator()
    t.translate(updates({"messages": [AIMessage(content="", tool_calls=[
        call("geocode", {"place": "Alfama, Lisbon"})])]}))
    events = t.translate(updates({"messages": [ToolMessage(
        content=json.dumps({"found": True, "name": "Alfama", "lat": 38.7, "lon": -9.1}),
        tool_call_id="c1")]}))
    assert [e.data["name"] for e in events if e.type == "geo"] == ["Alfama"]


def test_a_place_without_a_returned_name_falls_back_to_what_was_asked():
    t = StreamTranslator()
    t.translate(updates({"messages": [AIMessage(content="", tool_calls=[
        call("geocode_all", {"places": ["Belfort, Bruges"]})])]}))
    events = t.translate(updates({"messages": [ToolMessage(
        content=json.dumps({"results": [{"found": True, "lat": 51.2, "lon": 3.2}]}),
        tool_call_id="c1")]}))
    assert [e.data["name"] for e in events if e.type == "geo"] == ["Belfort, Bruges"]


def test_a_garbled_batch_result_is_skipped_not_raised():
    t = StreamTranslator()
    t.translate(updates({"messages": [AIMessage(content="", tool_calls=[
        call("geocode_all", {"places": ["a", "b"]})])]}))
    events = t.translate(updates({"messages": [ToolMessage(
        content=json.dumps({"results": ["not a dict", {"found": True, "lat": "x", "lon": 1}]}),
        tool_call_id="c1")]}))
    assert [e for e in events if e.type == "geo"] == []


# --------------------------------------------------------------------------
# The cache cannot poison a later run
# --------------------------------------------------------------------------


def test_a_corrupt_cache_entry_is_a_miss_not_a_crash(tmp_path, monkeypatch):
    from wayfinder.tools import cache

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    calls = []

    @cache.cached("demo")
    def lookup(q):
        calls.append(q)
        return {"answer": q}

    assert lookup("x") == {"answer": "x"}
    entry = next(tmp_path.rglob("*.json"))
    entry.write_text('{"answer": "x"', encoding="utf-8")     # truncated

    assert lookup("x") == {"answer": "x"}, "should refetch, not raise"
    assert calls == ["x", "x"]
    assert json.loads(entry.read_text()) == {"answer": "x"}, "and repair itself"


def test_the_cache_write_is_atomic(tmp_path, monkeypatch):
    """A plain write that is interrupted leaves a half-file that looks like a
    valid hit forever after — which is how a corrupt entry gets there at all."""
    from wayfinder.tools import cache

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    seen = {}
    real_replace = cache.os.replace

    def spy(src, dst):
        # Before the rename the destination must not exist even partially.
        seen["dst_existed"] = dst.exists()
        return real_replace(src, dst)

    monkeypatch.setattr(cache.os, "replace", spy)

    @cache.cached("demo")
    def lookup(q):
        return {"answer": q}

    lookup("y")
    assert seen["dst_existed"] is False
    assert not list(tmp_path.rglob("*.tmp")), "no temp files left behind"

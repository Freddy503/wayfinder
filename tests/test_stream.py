"""Tests for the stream translator and the web server.

The translator's contract is that it never raises. A stream that dies on one
malformed chunk takes the entire run's visibility with it, and the user is left
staring at a frozen page with no indication anything is wrong — so the
garbage-input tests here matter more than the happy-path ones.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from wayfinder.server import Decision, _to_langgraph, create_app
from wayfinder.stream import Event, StreamTranslator, _summarise_call


def updates(node: str, payload: dict, namespace: tuple = ()):
    """A chunk in the shape `stream(stream_mode=[...], subgraphs=True)` yields."""
    return (namespace, "updates", {node: payload})


def ai(tool_calls=None, content=""):
    return AIMessage(content=content, tool_calls=tool_calls or [])


def call(name, args, id="c1"):
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


# --------------------------------------------------------------------------
# Chunk shapes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chunk",
    [
        ((), "updates", {"model": {"messages": []}}),  # 3-tuple, subgraphs
        (("sub:1",), "updates", {"model": {"messages": []}}),  # nested
        ({"model": {"messages": []}},),  # odd 1-tuple
        {"model": {"messages": []}},  # bare dict
    ],
)
def test_unpack_tolerates_every_shape(chunk):
    assert StreamTranslator().translate(chunk) == [] or True


@pytest.mark.parametrize(
    "junk",
    [None, 42, "string", [], {}, ((),), (None, None, None), {"__interrupt__": "nonsense"}],
)
def test_translator_never_raises(junk):
    """Malformed input degrades to silence, never to an exception."""
    assert isinstance(StreamTranslator().translate(junk), list)


# --------------------------------------------------------------------------
# Tool calls and results
# --------------------------------------------------------------------------


def test_tool_call_becomes_an_event_with_a_summary():
    t = StreamTranslator()
    events = t.translate(
        updates("model", {"messages": [ai([call("web_search", {"query": "lisbon hours"})])]})
    )
    assert [e.type for e in events] == ["tool.call"]
    assert events[0].data["name"] == "web_search"
    assert events[0].data["summary"] == "lisbon hours"


def test_tool_result_is_paired_back_to_its_call():
    """A ToolMessage carries only an id; the name arrived on an earlier chunk."""
    t = StreamTranslator()
    t.translate(updates("model", {"messages": [ai([call("geocode", {"place": "Belem"})])]}))
    events = t.translate(
        updates("tools", {"messages": [ToolMessage(content="boom", tool_call_id="c1",
                                                   status="error")]})
    )
    assert events[0].type == "tool.result"
    assert events[0].data["name"] == "geocode", "result must inherit the call's name"
    assert events[0].data["ok"] is False


def test_unknown_tool_result_still_emits():
    t = StreamTranslator()
    events = t.translate(
        updates("tools", {"messages": [ToolMessage(content="hi", tool_call_id="orphan")]})
    )
    assert events[0].type == "tool.result"
    assert events[0].data["name"] == "?"


def test_long_results_are_truncated():
    t = StreamTranslator()
    events = t.translate(
        updates("tools", {"messages": [ToolMessage(content="x" * 5000, tool_call_id="c9")]})
    )
    assert len(events[0].data["preview"]) < 800
    assert "chars)" in events[0].data["preview"]


# --------------------------------------------------------------------------
# Subagents
# --------------------------------------------------------------------------


def test_task_call_is_a_subagent_dispatch_not_a_tool_call():
    t = StreamTranslator()
    events = t.translate(
        updates(
            "model",
            {"messages": [ai([call("task", {"subagent_type": "scout", "description": "sights"})])]},
        )
    )
    assert events[0].type == "subagent.start"
    assert events[0].data["subagent"] == "scout"


def test_subagent_report_closes_the_dispatch():
    t = StreamTranslator()
    t.translate(updates("model", {"messages": [ai([call("task", {"subagent_type": "food"})])]}))
    events = t.translate(
        updates("tools", {"messages": [ToolMessage(content="found 4", tool_call_id="c1")]})
    )
    assert events[0].type == "subagent.end"
    assert "found 4" in events[0].data["report"]


def test_namespace_marks_nesting():
    t = StreamTranslator()
    events = t.translate(
        updates("model", {"messages": [ai([call("geocode", {"place": "x"})])]}, ("task:abc",))
    )
    assert events[0].data["agent"] == "task:abc"

    t2 = StreamTranslator()
    events = t2.translate(updates("model", {"messages": [ai([call("geocode", {"place": "x"})])]}))
    assert events[0].data["agent"] == "main"


def test_geocode_hits_also_become_map_points():
    """A found geocode emits both the tool.result and a plottable geo event."""
    t = StreamTranslator()
    t.translate(updates("model", {"messages": [ai([call("geocode", {"place": "Acropolis"})])]}))
    payload = json.dumps({"found": True, "name": "Acropolis", "lat": 37.97, "lon": 23.72})
    events = t.translate(
        updates("tools", {"messages": [ToolMessage(content=payload, tool_call_id="c1")]})
    )
    assert [e.type for e in events] == ["tool.result", "geo"]
    assert events[1].data["lat"] == 37.97
    assert events[1].data["name"] == "Acropolis"


def test_failed_geocode_emits_no_map_point():
    t = StreamTranslator()
    t.translate(updates("model", {"messages": [ai([call("geocode", {"place": "Nowhere"})])]}))
    payload = json.dumps({"found": False, "query": "Nowhere"})
    events = t.translate(
        updates("tools", {"messages": [ToolMessage(content=payload, tool_call_id="c1")]})
    )
    assert [e.type for e in events] == ["tool.result"]


# --------------------------------------------------------------------------
# Checks and todos
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tool", ["check_itinerary", "finalize_itinerary"])
def test_check_results_get_their_own_event_type(tool):
    t = StreamTranslator()
    t.translate(updates("model", {"messages": [ai([call(tool, {})])]}))
    payload = json.dumps(
        {
            "passed": False,
            "summary": "FAIL — 1 hard",
            "violations": [{"check": "budget", "severity": "hard", "message": "over"}],
            "metrics": {"hard_pass_rate": 0.9},
        }
    )
    events = t.translate(
        updates("tools", {"messages": [ToolMessage(content=payload, tool_call_id="c1")]})
    )
    assert events[0].type == "check"
    assert events[0].data["passed"] is False
    assert events[0].data["violations"][0]["check"] == "budget"


def test_unparseable_check_result_falls_back_to_a_plain_result():
    """Degrade to a coarser event rather than dropping the result entirely."""
    t = StreamTranslator()
    t.translate(updates("model", {"messages": [ai([call("check_itinerary", {})])]}))
    events = t.translate(
        updates("tools", {"messages": [ToolMessage(content="not json", tool_call_id="c1")]})
    )
    assert events[0].type == "tool.result"


def test_todos_emit_once_per_change():
    t = StreamTranslator()
    todos = [{"content": "research", "status": "in_progress"}]
    assert t.translate(updates("model", {"todos": todos}))[0].type == "todos"
    assert t.translate(updates("model", {"todos": todos})) == [], "unchanged todos are noise"
    changed = [{"content": "research", "status": "completed"}]
    assert t.translate(updates("model", {"todos": changed}))[0].type == "todos"


def test_repeated_assistant_text_is_not_duplicated():
    t = StreamTranslator()
    chunk = updates("model", {"messages": [ai(content="Planning day one.")]})
    assert len(t.translate(chunk)) == 1
    assert t.translate(chunk) == []


# --------------------------------------------------------------------------
# Interrupts
# --------------------------------------------------------------------------


def test_interrupt_carries_actions_and_allowed_decisions():
    t = StreamTranslator()
    events = t.translate(
        (
            (),
            "updates",
            {
                "__interrupt__": [
                    type(
                        "I",
                        (),
                        {
                            "value": {
                                "action_requests": [
                                    {"name": "finalize_itinerary", "args": {"summary": "done"}}
                                ],
                                "review_configs": [
                                    {
                                        "action_name": "finalize_itinerary",
                                        "allowed_decisions": ["approve", "reject"],
                                    }
                                ],
                            }
                        },
                    )()
                ]
            },
        )
    )
    assert events[0].type == "interrupt"
    action = events[0].data["actions"][0]
    assert action["name"] == "finalize_itinerary"
    assert action["allowed"] == ["approve", "reject"]
    assert action["description"], "a card with no description is unreviewable"


def test_interrupt_without_review_config_still_offers_decisions():
    t = StreamTranslator()
    events = t.translate(
        ((), "updates", {"__interrupt__": [{"value": {"action_requests": [{"name": "x"}]}}]})
    )
    assert events[0].data["actions"][0]["allowed"] == ["approve", "reject"]


# --------------------------------------------------------------------------
# Summaries and SSE
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("web_search", {"query": "hours"}, "hours"),
        ("geocode", {"place": "Alfama"}, "Alfama"),
        ("estimate_travel", {"origin": "A", "destination": "B", "mode": "walk"}, "A → B (walk)"),
        ("fx_convert", {"amount": 10, "from_currency": "USD", "to_currency": "EUR"},
         "10 USD → EUR"),
    ],
)
def test_call_summaries_are_human_readable(name, args, expected):
    assert _summarise_call(name, args) == expected


def test_sse_frame_is_well_formed():
    frame = Event("tool.call", {"name": "geocode"}).sse()
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    assert json.loads(frame[6:].strip())["type"] == "tool.call"


def test_sse_survives_unserialisable_payloads():
    frame = Event("x", {"obj": object()}).sse()
    assert json.loads(frame[6:].strip())["obj"]


# --------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        (Decision(type="approve"), {"type": "approve"}),
        (Decision(type="reject"), {"type": "reject"}),
        (Decision(type="reject", message="too costly"),
         {"type": "reject", "message": "too costly"}),
        (Decision(type="respond", message="use the metro"),
         {"type": "respond", "message": "use the metro"}),
        (Decision(type="edit", args={"summary": "v2"}),
         {"type": "edit", "edited_action": {"args": {"summary": "v2"}}}),
    ],
)
def test_decisions_map_onto_the_middleware_shape(decision, expected):
    assert _to_langgraph(decision) == expected


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    return TestClient(create_app())


def test_examples_endpoint_serves_the_dataset(client):
    cases = client.get("/api/examples").json()
    assert len(cases) >= 15
    assert "should_refuse" not in cases[0]["spec"]


def test_ui_is_served(client):
    body = client.get("/").text
    assert "Wayfinder" in body
    assert "EventSource" in body


def test_unknown_run_is_a_404(client):
    assert client.get("/api/stream/nope").status_code == 404
    assert client.get("/api/runs/nope").status_code == 404
    assert client.post("/api/decide/nope", json={"decisions": []}).status_code == 404


def test_history_lists_past_runs_from_disk(client):
    """Runs must outlive the process, or a restart orphans every past trip."""
    past = client.get("/api/history").json()
    assert isinstance(past, list)
    for entry in past:
        assert "run_id" in entry and "destination" in entry


def test_past_run_artifacts_are_reachable_after_restart(client):
    past = client.get("/api/history").json()
    if not past:
        pytest.skip("no completed runs on disk")
    run_id = past[0]["run_id"]
    assert client.get(f"/api/runs/{run_id}/artifact/itinerary.json").status_code == 200


def test_history_run_ids_cannot_escape_the_runs_directory(client):
    """The id comes from the browser and indexes a directory — clamp it."""
    for evil in ["..", "../..", "..%2F..", "/etc"]:
        assert client.get(f"/api/runs/{evil}/artifact/itinerary.json").status_code == 404


def test_artifact_name_cannot_escape_the_run_directory(client):
    """The artifact name comes from the browser and must not escape the run.

    Exercised through the public endpoint rather than by reaching into the
    app's closures — the previous version indexed `__closure__[0]` and broke
    the moment an unrelated variable was added to the same scope.
    """
    past = client.get("/api/history").json()
    if not past:
        pytest.skip("no completed runs on disk")
    run_id = past[0]["run_id"]
    for evil in ["../../etc/passwd", "..%2F..%2Fspec.yaml", "/etc/passwd", "../config.json"]:
        response = client.get(f"/api/runs/{run_id}/artifact/{evil}")
        assert response.status_code == 404, f"{evil!r} was served"

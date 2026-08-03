"""Tests for the RouteStack MCP client and the live/indicative switch.

The contract that matters: a flight lookup must never take down a run. If
RouteStack is unconfigured, erroring, slow, or returns something unexpected,
`flight_search` still answers — with indicative data, honestly labelled.
"""

from __future__ import annotations

import json

import pytest

from wayfinder.tools import routestack
from wayfinder.tools.flights import flight_search


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in (
        "ROUTESTACK_API_KEY", "ROUTESTACK_API_SECRET",
        "ROUTESTACK_ACCOUNT_ID", "ROUTESTACK_MCP_URL",
    ):
        monkeypatch.delenv(var, raising=False)


def mcp_reply(payload):
    """A well-formed JSON-RPC tools/call reply carrying JSON in a text block."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"content": [{"type": "text", "text": json.dumps(payload)}]},
    }


class FakeResponse:
    def __init__(self, body, status=200):
        self._body = body if isinstance(body, str) else json.dumps(body)
        self.status_code = status

    @property
    def text(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError("boom", request=None, response=None)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def test_unconfigured_by_default():
    assert routestack.is_configured() is False
    assert routestack.live_flight_search("BER", "LIS", "2026-10-12")["available"] is False


def test_configured_when_a_key_is_present(monkeypatch):
    monkeypatch.setenv("ROUTESTACK_API_KEY", "rst_test")
    assert routestack.is_configured() is True


def test_credentials_travel_in_headers_not_the_body(monkeypatch):
    """A key in a request body ends up in logs and caches. Headers only."""
    monkeypatch.setenv("ROUTESTACK_API_KEY", "rst_secret")
    monkeypatch.setenv("ROUTESTACK_API_SECRET", "hexsecret")
    seen = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen["headers"] = headers
        seen["body"] = json
        return FakeResponse(mcp_reply({"offers": []}))

    monkeypatch.setattr("httpx.post", fake_post)
    routestack.call_tool("flight_session", {})

    assert seen["headers"]["authorization"] == "Bearer rst_secret"
    assert seen["headers"]["x-api-secret"] == "hexsecret"
    assert "rst_secret" not in json.dumps(seen["body"])


def test_endpoint_is_overridable(monkeypatch):
    monkeypatch.setenv("ROUTESTACK_API_KEY", "k")
    monkeypatch.setenv("ROUTESTACK_MCP_URL", "https://example.test/mcp")
    seen = {}
    monkeypatch.setattr(
        "httpx.post",
        lambda url, **kw: (seen.update(url=url), FakeResponse(mcp_reply({})))[1],
    )
    routestack.call_tool("flight_session", {})
    assert seen["url"] == "https://example.test/mcp"


# --------------------------------------------------------------------------
# Response decoding
# --------------------------------------------------------------------------


def test_sse_framed_bodies_are_decoded(monkeypatch):
    """The server may answer as SSE; a raw json.loads would choke on it."""
    monkeypatch.setenv("ROUTESTACK_API_KEY", "k")
    body = "event: message\ndata: " + json.dumps(mcp_reply({"offers": [{"id": "x"}]})) + "\n\n"
    monkeypatch.setattr("httpx.post", lambda url, **kw: FakeResponse(body))
    result = routestack.call_tool("search_flights", {})
    assert result["ok"] is True
    assert result["data"]["offers"] == [{"id": "x"}]


def test_structured_content_is_preferred(monkeypatch):
    monkeypatch.setenv("ROUTESTACK_API_KEY", "k")
    payload = {"jsonrpc": "2.0", "id": 1,
               "result": {"structuredContent": {"offers": [1, 2]}, "content": []}}
    monkeypatch.setattr("httpx.post", lambda url, **kw: FakeResponse(payload))
    assert routestack.call_tool("search_flights", {})["data"]["offers"] == [1, 2]


def test_jsonrpc_errors_become_data_not_exceptions(monkeypatch):
    monkeypatch.setenv("ROUTESTACK_API_KEY", "k")
    payload = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32001, "message": "unauthorized"}}
    monkeypatch.setattr("httpx.post", lambda url, **kw: FakeResponse(payload))
    result = routestack.call_tool("search_flights", {})
    assert result["ok"] is False
    assert "unauthorized" in result["error"]


def test_tool_level_errors_are_caught(monkeypatch):
    monkeypatch.setenv("ROUTESTACK_API_KEY", "k")
    payload = {"jsonrpc": "2.0", "id": 1,
               "result": {"isError": True, "content": [{"type": "text", "text": "no route"}]}}
    monkeypatch.setattr("httpx.post", lambda url, **kw: FakeResponse(payload))
    assert routestack.call_tool("search_flights", {})["ok"] is False


def test_network_failure_is_reported_not_raised(monkeypatch):
    import httpx

    monkeypatch.setenv("ROUTESTACK_API_KEY", "k")

    def boom(*a, **kw):
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr("httpx.post", boom)
    result = routestack.call_tool("search_flights", {})
    assert result["ok"] is False
    assert "ConnectTimeout" in result["error"]


def test_garbage_body_is_survivable(monkeypatch):
    monkeypatch.setenv("ROUTESTACK_API_KEY", "k")
    monkeypatch.setattr("httpx.post", lambda url, **kw: FakeResponse("<html>nope</html>"))
    assert routestack.call_tool("search_flights", {})["ok"] is False


# --------------------------------------------------------------------------
# The live / indicative switch — the property that protects every run
# --------------------------------------------------------------------------


def test_falls_back_to_indicative_when_unconfigured(monkeypatch):
    monkeypatch.setattr(
        "wayfinder.tools.flights.web_search",
        lambda q, max_results=6: {"results": [
            {"title": "BER-LIS", "content": "from €128 3h 20m", "url": "https://e.test/a"}]},
    )
    result = flight_search("Berlin", "Lisbon", "2026-10-12")
    assert result["source"] == "indicative"
    assert result["price_range"]["low"] == 128.0


def test_live_results_are_used_and_labelled(monkeypatch):
    monkeypatch.setenv("ROUTESTACK_API_KEY", "k")
    monkeypatch.setattr(
        routestack, "live_flight_search",
        lambda *a, **kw: {
            "available": True, "currency": "USD", "correlation_id": "corr-1",
            "flights": [{"direction": "outbound", "date": "2026-10-12",
                         "depart_time": "06:00", "arrive_time": "09:30",
                         "origin_airport": "BER", "destination_airport": "LIS",
                         "estimated_cost": 143, "_offer_id": "flt-1"}],
        },
    )
    result = flight_search("Berlin", "Lisbon", "2026-10-12")
    assert result["source"] == "live"
    assert result["correlation_id"] == "corr-1"
    assert result["flights"][0]["origin_airport"] == "BER"
    assert "Nothing is booked" in result["note"]
    assert "fx_convert" in result["note"], "a USD fare into a EUR budget needs converting"


def test_a_broken_live_lookup_degrades_instead_of_failing(monkeypatch):
    """Configured but erroring must not cost the traveller their itinerary."""
    monkeypatch.setenv("ROUTESTACK_API_KEY", "k")
    monkeypatch.setattr(
        routestack, "live_flight_search",
        lambda *a, **kw: {"available": False, "reason": "502 from upstream"},
    )
    monkeypatch.setattr(
        "wayfinder.tools.flights.web_search",
        lambda q, max_results=6: {"results": [
            {"title": "x", "content": "from €99", "url": "https://e.test/b"}]},
    )
    result = flight_search("Berlin", "Lisbon", "2026-10-12")
    assert result["source"] == "indicative"
    assert result["found"] is True


def test_live_but_empty_also_falls_back(monkeypatch):
    monkeypatch.setenv("ROUTESTACK_API_KEY", "k")
    monkeypatch.setattr(
        routestack, "live_flight_search",
        lambda *a, **kw: {"available": True, "offers": []},
    )
    monkeypatch.setattr(
        "wayfinder.tools.flights.web_search",
        lambda q, max_results=6: {"results": []},
    )
    assert flight_search("A", "B", "2026-10-12")["source"] == "indicative"


# --------------------------------------------------------------------------
# Offer → Flight mapping (shapes taken from a real RouteStack response)
# --------------------------------------------------------------------------


def offer(segments, **over):
    base = {
        "offerId": "flt-1", "fareSourceCode": "abc==", "ourprice": 347.96,
        "totalFare": 326.4, "stops": len(segments) - 1, "summary": "KLM 1770 BER→AMS",
        "flight": {"flights": segments},
    }
    base.update(over)
    return base


def seg(airline, number, dep, arr, dep_time, arr_time):
    return {"airline": airline, "flightNumber": number, "departure": dep, "arrival": arr,
            "departureTime": dep_time, "arrivalTime": arr_time, "cabin": "Economy"}


def test_multi_leg_offer_maps_to_the_flight_schema():
    from wayfinder.schema import Flight

    mapped = routestack.offer_to_flight(
        offer([
            seg("KLM", "1770", "BER", "AMS", "2026-10-12T06:00:00", "2026-10-12T07:25:00"),
            seg("KLM", "1587", "AMS", "LIS", "2026-10-12T20:50:00", "2026-10-12T22:50:00"),
        ]),
        "outbound", "USD",
    )
    assert mapped["origin_airport"] == "BER", "first segment's departure"
    assert mapped["destination_airport"] == "LIS", "last segment's arrival"
    assert mapped["depart_time"] == "06:00"
    assert mapped["arrive_time"] == "22:50"
    assert mapped["stops"] == 1
    Flight.model_validate({k: v for k, v in mapped.items() if not k.startswith("_")})


def test_overnight_offers_set_arrives_next_day():
    """The one derived field — and the one a model would get wrong by eye."""
    mapped = routestack.offer_to_flight(
        offer([
            seg("KLM", "1782", "BER", "AMS", "2026-10-12T19:15:00", "2026-10-12T20:40:00"),
            seg("TAP", "675", "AMS", "LIS", "2026-10-13T16:30:00", "2026-10-13T18:45:00"),
        ]),
        "outbound", "USD",
    )
    assert mapped["arrives_next_day"] is True
    assert mapped["date"] == "2026-10-12", "date is the departure date, not the arrival"


def test_codeshare_legs_list_every_carrier():
    mapped = routestack.offer_to_flight(
        offer([
            seg("KLM", "1770", "BER", "AMS", "2026-10-12T06:00:00", "2026-10-12T07:25:00"),
            seg("TAP", "675", "AMS", "LIS", "2026-10-12T10:00:00", "2026-10-12T12:20:00"),
        ]),
        "outbound", "USD",
    )
    assert mapped["airline"] == "KLM / TAP"


@pytest.mark.parametrize(
    "broken",
    [
        offer([]),                                            # no segments
        {"flight": {}},                                       # no flights key
        offer([{"airline": "KLM", "departure": "BER"}]),       # no times
    ],
)
def test_unusable_offers_are_dropped_not_guessed(broken):
    assert routestack.offer_to_flight(broken, "outbound", "USD") is None


def test_handles_are_underscore_prefixed_so_they_never_reach_the_schema():
    """Itinerary rejects unknown fields; booking handles must be distinguishable."""
    mapped = routestack.offer_to_flight(
        offer([seg("KLM", "1770", "BER", "LIS", "2026-10-12T06:00:00", "2026-10-12T09:30:00")]),
        "outbound", "EUR",
    )
    handles = {k for k in mapped if k.startswith("_")}
    assert {"_offer_id", "_fare_source_code", "_currency"} <= handles


# --------------------------------------------------------------------------
# Checkout stays a link, never a purchase
# --------------------------------------------------------------------------


def test_checkout_returns_a_link_for_the_traveller(monkeypatch):
    monkeypatch.setenv("ROUTESTACK_API_KEY", "k")
    monkeypatch.setattr(
        "httpx.post",
        lambda url, **kw: FakeResponse(mcp_reply({"url": "https://book.test/xyz"})),
    )
    result = routestack.checkout_url("corr-1", offer_id="o-1")
    assert result["ok"] is True
    assert result["url"] == "https://book.test/xyz"


def test_no_purchase_tool_is_exposed():
    """Wayfinder can hand over a link; it must not be able to complete a sale."""
    surface = dir(routestack)
    for forbidden in ("book_flight", "purchase", "pay", "confirm_booking"):
        assert forbidden not in surface

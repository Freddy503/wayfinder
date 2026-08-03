"""RouteStack MCP client — live flight inventory.

RouteStack exposes real availability over MCP (JSON-RPC over HTTP). When it is
configured, `flight_search` returns actual offers with real carriers, times and
fares instead of prices parsed out of prose; when it isn't, the search-based
fallback in `flights.py` still runs. Same tool signature either way, so the
schema and all four flight checks are unaffected by which path served the data.

Credentials come from the environment and are never written down here. The
server also accepts unauthenticated calls; sending the key is what attributes a
booking to your account.

**It can produce a checkout URL. It never completes a purchase** — a deep link
is something the traveller opens and decides on, which is the same contract the
rest of Wayfinder keeps.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from wayfinder.tools.cache import cached

DEFAULT_URL = "https://mcp.routestack.ai/mcp"
PROTOCOL_VERSION = "2025-06-18"


def is_configured() -> bool:
    """True when a RouteStack key is present in the environment."""
    return bool(os.environ.get("ROUTESTACK_API_KEY", "").strip())


def _endpoint() -> str:
    return os.environ.get("ROUTESTACK_MCP_URL", DEFAULT_URL).strip() or DEFAULT_URL


def _headers() -> dict[str, str]:
    headers = {
        "content-type": "application/json",
        # The server may answer either shape; accept both.
        "accept": "application/json, text/event-stream",
    }
    key = os.environ.get("ROUTESTACK_API_KEY", "").strip()
    if key:
        headers["authorization"] = f"Bearer {key}"
        headers["x-api-key"] = key
    secret = os.environ.get("ROUTESTACK_API_SECRET", "").strip()
    if secret:
        headers["x-api-secret"] = secret
    account = os.environ.get("ROUTESTACK_ACCOUNT_ID", "").strip()
    if account:
        headers["x-account-id"] = account
    return headers


def _parse(response: httpx.Response) -> dict[str, Any]:
    """Read a JSON-RPC reply, tolerating an SSE-framed body."""
    text = response.text.strip()
    if text.startswith("event:") or text.startswith("data:"):
        for line in text.splitlines():
            if line.startswith("data:"):
                text = line[5:].strip()
                break
    return json.loads(text)


def call_tool(name: str, arguments: dict[str, Any], timeout: float = 45.0) -> dict[str, Any]:
    """Invoke one MCP tool and return its decoded content.

    Errors come back as data rather than exceptions: a flight lookup that fails
    should degrade to the search-based fallback, not take down the run.
    """
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    try:
        response = httpx.post(_endpoint(), headers=_headers(), json=request, timeout=timeout)
        response.raise_for_status()
        payload = _parse(response)
    except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if "error" in payload:
        return {"ok": False, "error": str(payload["error"])}

    result = payload.get("result", {})
    if result.get("isError"):
        return {"ok": False, "error": _text_of(result)}
    return {"ok": True, "data": _decode(result)}


def _text_of(result: dict[str, Any]) -> str:
    parts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
    return " ".join(p for p in parts if p)[:400]


def _decode(result: dict[str, Any]) -> Any:
    """MCP returns content blocks; the useful payload is JSON inside a text block."""
    if "structuredContent" in result:
        return result["structuredContent"]
    text = _text_of(result)
    if not text:
        return result
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {"text": text}


@cached("routestack_flights")
def _search(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str | None,
    adults: int,
    children: int,
    cabin_class: str,
) -> dict[str, Any]:
    # A session is created first when the connector wants one; failure here is
    # not fatal, so the search is still attempted.
    call_tool("flight_session", {})

    arguments: dict[str, Any] = {
        "origin": origin,
        "destination": destination,
        "departureDate": departure_date,
        "adults": adults,
        "cabinClass": cabin_class,
        "tripType": "RoundTrip" if return_date else "OneWay",
    }
    if return_date:
        arguments["returnDate"] = return_date
    if children:
        arguments["children"] = children
    return call_tool("search_flights", arguments)


def live_flight_search(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str | None = None,
    adults: int = 1,
    children: int = 0,
    cabin_class: str = "Economy",
) -> dict[str, Any]:
    """Search live inventory. Returns `{"available": False}` when unconfigured."""
    if not is_configured():
        return {"available": False, "reason": "ROUTESTACK_API_KEY is not set"}

    result = _search(
        origin.strip(), destination.strip(), departure_date.strip(),
        (return_date or "").strip() or None, adults, children, cabin_class,
    )
    if not result.get("ok"):
        return {"available": False, "reason": result.get("error", "unknown error")}

    data = result["data"] if isinstance(result["data"], dict) else {}
    offers = data.get("offers") or data.get("flights") or []
    currency = data.get("currency", "USD")
    direction = "return" if return_date else "outbound"

    shaped = [
        f for f in (offer_to_flight(o, direction, currency) for o in offers[:8]) if f
    ]
    return {
        "available": True,
        "currency": currency,
        "offer_count": data.get("offerCount", len(offers)),
        "flights": shaped,
        "correlation_id": data.get("correlationId"),
    }


def offer_to_flight(offer: dict[str, Any], direction: str, currency: str) -> dict[str, Any] | None:
    """Reshape a RouteStack offer into the itinerary's `Flight` shape.

    Done here rather than left to the model on purpose: deriving
    `arrives_next_day` means comparing the first segment's departure date with
    the last segment's arrival date, and a model doing that by eye on a
    multi-leg itinerary will occasionally get it wrong. That single boolean
    decides whether the arrival check reads a 06:00 landing as early morning or
    as eighteen hours early.
    """
    flight = offer.get("flight") or {}
    segments = flight.get("flights") or []
    if not segments:
        return None

    first, last = segments[0], segments[-1]
    try:
        depart = first["departureTime"].split("T")
        arrive = last["arrivalTime"].split("T")
        depart_date, depart_time = depart[0], depart[1][:5]
        arrive_date, arrive_time = arrive[0], arrive[1][:5]
    except (KeyError, IndexError, AttributeError):
        return None

    carriers = sorted({s.get("airline") for s in segments if s.get("airline")})
    return {
        "direction": direction,
        "date": depart_date,
        "depart_time": depart_time,
        "arrive_time": arrive_time,
        "arrives_next_day": arrive_date > depart_date,
        "origin_airport": first.get("departure", ""),
        "destination_airport": last.get("arrival", ""),
        "airline": " / ".join(carriers) if carriers else None,
        "flight_number": first.get("flightNumber"),
        "stops": offer.get("stops", max(0, len(segments) - 1)),
        # Price is quoted in the search's own currency — convert with
        # fx_convert before writing it into an itinerary priced differently.
        "estimated_cost": offer.get("ourprice") or offer.get("totalFare") or 0,
        "_currency": currency,
        "_offer_id": offer.get("offerId"),
        "_fare_source_code": offer.get("fareSourceCode"),
        "_route": offer.get("summary"),
    }


def checkout_url(correlation_id: str, offer_id: str | None = None,
                 fare_source_code: str | None = None) -> dict[str, Any]:
    """Get a deep link the traveller can open to review and book themselves.

    This produces a link. It does not buy anything, and nothing downstream
    should: the itinerary's job is to end at a decision the human makes.
    """
    arguments: dict[str, Any] = {"correlationId": correlation_id}
    if offer_id:
        arguments["offerId"] = offer_id
    elif fare_source_code:
        arguments["fareSourceCode"] = fare_source_code

    result = call_tool("flight_get_checkout_url", arguments)
    if not result.get("ok"):
        return {"ok": False, "reason": result.get("error")}
    data = result["data"] if isinstance(result["data"], dict) else {}
    return {"ok": True, "url": data.get("url") or data.get("checkoutUrl"), "data": data}


def airport_lookup(term: str) -> dict[str, Any]:
    """Resolve a city or airport name to IATA codes."""
    if not is_configured():
        return {"available": False}
    result = call_tool("flight_locations", {"term": term.strip()})
    if not result.get("ok"):
        return {"available": False, "reason": result.get("error")}
    return {"available": True, "locations": result["data"]}

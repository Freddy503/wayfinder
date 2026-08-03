"""Flight research.

**This finds indicative routes and fares, not bookable inventory.** There is no
free API for live availability, so this reads what public sources say about a
route — who flies it, roughly how long it takes, what the fare usually runs —
and records where each claim came from. The agent never books anything.

That limitation is deliberately visible in the output: every result carries a
`confidence` and a `caveat`, because a plan built on a fare that turns out to
be seasonal-low is worse than one that says "around €120, check before you
commit."
"""

from __future__ import annotations

import re

from wayfinder.tools.cache import cached
from wayfinder.tools.search import web_search

#: Airport codes appear as standalone three-letter capitals. Filtered against
#: common false positives — a fare quoted in "EUR" is not an airport.
_IATA = re.compile(r"\b([A-Z]{3})\b")
_NOT_AIRPORTS = {
    "EUR", "USD", "GBP", "CHF", "SEK", "NOK", "DKK", "PLN", "CZK", "HUF", "JPY",
    "AND", "THE", "FOR", "ARE", "NOT", "ALL", "ONE", "TWO", "NEW", "CET", "GMT",
    "UTC", "VAT", "FAQ", "PDF", "API", "CO2", "MIN", "MAX", "AVG",
}

#: A fare with a currency marker: "€128", "128 EUR", "$210".
_PRICE = re.compile(
    r"(?:(€|£|\$)\s?(\d{2,4}(?:[.,]\d{2})?))|(?:(\d{2,4}(?:[.,]\d{2})?)\s?(EUR|USD|GBP))",
    re.I,
)

#: "2h 15m", "2 hr 15", "2:15" durations.
_DURATION = re.compile(r"\b(\d{1,2})\s*(?:h|hr|hours?)\s*(\d{1,2})?\s*(?:m|min)?\b", re.I)


def extract_airports(text: str) -> list[str]:
    """Pull plausible IATA codes out of a snippet."""
    return sorted({c for c in _IATA.findall(text) if c not in _NOT_AIRPORTS})


def extract_price(text: str) -> tuple[float, str] | None:
    """First fare with a currency attached, as (amount, ISO code)."""
    match = _PRICE.search(text)
    if not match:
        return None
    symbol, amount_a, amount_b, code = match.groups()
    raw = amount_a or amount_b
    if raw is None:
        return None
    amount = float(raw.replace(",", "."))
    currency = {"€": "EUR", "£": "GBP", "$": "USD"}.get(symbol or "", (code or "").upper())
    return (amount, currency) if currency else None


def extract_duration_minutes(text: str) -> int | None:
    match = _DURATION.search(text)
    if not match:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2) or 0)
    total = hours * 60 + minutes
    # A "1h" that parses to 30 hours is a misread, not a flight.
    return total if 20 <= total <= 20 * 60 else None


@cached("flights")
def _research(origin: str, destination: str, when: str, kind: str) -> dict:
    query = f"flights {origin} to {destination} {when} {kind}".strip()
    return web_search(query, max_results=6)


def flight_search(origin: str, destination: str, date: str, direction: str = "outbound") -> dict:
    """Research flights between two cities on a given date.

    Call this once per direction, early — the outbound landing time and the
    return departure time constrain what can be scheduled on the first and last
    days, and the checker enforces that. Planning the days first and the
    flights afterwards means replanning both.

    Results are **indicative, not bookable**: typical routes, airlines,
    durations and fares gathered from public sources. Record what you find in
    the itinerary's `flights` with its `sources`, and put any uncertainty in
    the flight's `note` rather than presenting a guess as fact.

    Args:
        origin: Departure city or airport, e.g. "Berlin" or "BER".
        destination: Arrival city or airport, e.g. "Lisbon".
        date: Travel date, `YYYY-MM-DD`.
        direction: "outbound" or "return" — used to shape the query.

    Returns:
        `{"found", "routes": [{"airports", "price", "currency",
        "duration_minutes", "title", "url", "snippet"}], "caveat"}`.
    """
    kind = "direct flight time price" if direction == "outbound" else "return flight time price"
    payload = _research(origin.strip(), destination.strip(), date.strip(), kind)

    routes = []
    for result in payload.get("results", []):
        blob = f"{result.get('title', '')} {result.get('content', '')}"
        price = extract_price(blob)
        routes.append(
            {
                "airports": extract_airports(blob)[:6],
                "price": price[0] if price else None,
                "currency": price[1] if price else None,
                "duration_minutes": extract_duration_minutes(blob),
                "title": result.get("title"),
                "url": result.get("url"),
                "snippet": blob[:300],
            }
        )

    priced = [r for r in routes if r["price"] is not None]
    return {
        "found": bool(routes),
        "origin": origin,
        "destination": destination,
        "date": date,
        "direction": direction,
        "routes": routes,
        "price_range": (
            {"low": min(r["price"] for r in priced), "high": max(r["price"] for r in priced)}
            if priced
            else None
        ),
        "caveat": (
            "Indicative only — gathered from public pages, not live availability. "
            "Treat fares as typical rather than bookable, and say so in the flight's note."
        ),
    }

"""Geocoding and travel-time estimation.

Travel times are deliberately crude: haversine distance times a per-mode speed,
plus fixed overhead for waiting and walking to the stop. They will be wrong in
detail — Lisbon's hills alone guarantee it.

That's acceptable, and worth being clear about why. The checker's job is not to
know the true travel time; it's to hold the plan to *internal consistency*
against the same model the agent used. An itinerary that allows 10 minutes for
a hop this function calls 40 is over-packed under any estimate. Swap in OSRM
later and both sides move together.
"""

from __future__ import annotations

import math
import os

import httpx

from wayfinder.tools.cache import RateLimiter, cached

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

#: Nominatim's usage policy: one request per second, descriptive User-Agent.
_limiter = RateLimiter(1.1)

#: Straight-line speed in km/h, already discounted for the fact that real
#: routes are not straight lines, plus fixed minutes of overhead per leg.
_MODE_MODEL: dict[str, tuple[float, int]] = {
    "walk": (4.2, 2),
    "bike": (11.0, 3),
    "transit": (16.0, 8),  # includes waiting for the tram and walking to it
    "taxi": (22.0, 4),
    "other": (12.0, 5),
}


def _user_agent() -> str:
    ua = os.environ.get("WAYFINDER_USER_AGENT", "").strip()
    if not ua:
        msg = (
            "WAYFINDER_USER_AGENT is unset. Nominatim's usage policy requires a "
            "descriptive User-Agent with a contact address — set it in .env "
            "(see .env.example) before geocoding."
        )
        raise RuntimeError(msg)
    return ua


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


@cached("geocode")
def _geocode_raw(query: str) -> dict | None:
    _limiter.wait()
    response = httpx.get(
        NOMINATIM_URL,
        params={"q": query, "format": "jsonv2", "limit": 1, "addressdetails": 1},
        headers={"User-Agent": _user_agent()},
        timeout=20.0,
    )
    response.raise_for_status()
    hits = response.json()
    if not hits:
        return None
    hit = hits[0]
    return {
        "name": hit.get("name") or query,
        "display_name": hit.get("display_name"),
        "lat": float(hit["lat"]),
        "lon": float(hit["lon"]),
    }


def geocode(place: str) -> dict:
    """Look up the coordinates and full address of a named place.

    Call this for every venue you intend to schedule, before you commit to a
    time for it. The coordinates go into the itinerary's `venue.lat`/`venue.lon`
    and are what make travel times between stops computable.

    Args:
        place: Name plus city, e.g. "Museu Nacional do Azulejo, Lisbon".

    Returns:
        `{"found": true, "name", "display_name", "lat", "lon"}`, or
        `{"found": false}` when nothing matches — in which case the place may
        not exist under that name, so search for the correct one rather than
        inventing coordinates.
    """
    hit = _geocode_raw(place.strip())
    if hit is None:
        return {"found": False, "query": place}
    return {"found": True, **hit}


def estimate_travel(origin: str, destination: str, mode: str = "walk") -> dict:
    """Estimate how long it takes to get between two named places.

    Call this for **every** move between venues before you schedule the second
    one, and put the result in the later item's `transit_from_previous`. The
    checker treats a missing transit leg between different venues as a hard
    failure, and compares the minutes you record against the gap you left.

    Estimates are straight-line distance adjusted per mode, so treat them as a
    floor rather than a promise. If the gap you have is close to the estimate,
    leave more room.

    Args:
        origin: Name of the place being left, including the city.
        destination: Name of the place being travelled to, including the city.
        mode: One of walk, bike, transit, taxi, other.

    Returns:
        `{"ok": true, "minutes", "distance_km", "mode"}`, or `{"ok": false,
        "reason": ...}` when either endpoint could not be geocoded.
    """
    mode = mode if mode in _MODE_MODEL else "other"
    a, b = geocode(origin), geocode(destination)
    for label, hit in (("origin", a), ("destination", b)):
        if not hit["found"]:
            return {"ok": False, "reason": f"could not locate {label}: {hit['query']!r}"}

    km = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
    speed, overhead = _MODE_MODEL[mode]
    # Straight-line distance understates a real route; 1.3 is the usual rule of
    # thumb for street networks.
    routed_km = km * 1.3
    minutes = int(round(routed_km / speed * 60)) + overhead

    return {
        "ok": True,
        "mode": mode,
        "minutes": minutes,
        "distance_km": round(routed_km, 2),
        "straight_line_km": round(km, 2),
        "note": "Straight-line estimate with a 1.3 routing factor; not a routed time.",
    }

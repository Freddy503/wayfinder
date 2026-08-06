"""Pictures of the places in a plan.

Travel planning is partly about wanting to go, and a schedule of names in a
table does not make anyone want to go anywhere. This fetches a photograph and a
one-line description for the venues in a finished itinerary.

**Wikipedia, because it is free, keyless and cached** — the same bargain as the
geocoder, and the same obligations: a descriptive User-Agent and restraint.

**Coordinates first, name second.** A title guess gets *Van Gogh Museum* right
and *Bocca* catastrophically wrong. Geosearch narrows to articles within a few
hundred metres, then the name picks between them — near the Van Gogh Museum,
geosearch also returns *Wheatfield with Crows* and the *Stedelijk Museum*, and
only the name says which one is the venue.

**Restaurants mostly have no article, and that is fine.** Verified: Cervejaria
Ramiro 404s and a geosearch around it comes back empty. Those get a map tile of
where they are — honest, and never a photograph of somewhere else.
"""

from __future__ import annotations

import math
import re
from typing import Any

import httpx

from wayfinder.tools.cache import RateLimiter, cached
from wayfinder.tools.geo import _user_agent

WIKI_API = "https://en.wikipedia.org/w/api.php"

#: Wikimedia asks for well under 200 requests/second and a real User-Agent. This
#: is far below that; the cache means a second run costs nothing at all.
_limiter = RateLimiter(0.15)

#: How far from the venue an article may sit and still be about it. A museum's
#: article coordinate is often the middle of the building, so a little slack —
#: but not enough to pick up the café over the road.
SEARCH_RADIUS_M = 300

#: Words that carry no identifying weight when matching a title to a venue.
_NOISE = {
    "the", "de", "het", "la", "le", "el", "cafe", "café", "bar", "restaurant",
    "museum", "at", "and", "van", "der", "den", "a", "of", "&",
}


def _tokens(text: str) -> set[str]:
    return {
        word
        for word in re.split(r"\W+", text.lower())
        if len(word) > 2 and word not in _NOISE
    }


def _match_score(venue: str, title: str) -> float:
    """How much of the venue's name the article's title accounts for.

    Scored against the *venue's* tokens rather than the intersection over the
    union: "Rijksmuseum" should match the article "Rijksmuseum" perfectly even
    though a longer title would dilute a symmetric measure.
    """
    wanted = _tokens(venue)
    if not wanted:
        return 0.0
    return len(wanted & _tokens(title)) / len(wanted)


@cached("wiki_geosearch")
def _nearby(lat: float, lon: float) -> list[str]:
    _limiter.wait()
    response = httpx.get(
        WIKI_API,
        params={
            "action": "query", "format": "json", "list": "geosearch",
            "gscoord": f"{lat}|{lon}", "gsradius": str(SEARCH_RADIUS_M), "gslimit": "10",
        },
        headers={"User-Agent": _user_agent()},
        timeout=15.0,
    )
    response.raise_for_status()
    return [hit["title"] for hit in response.json().get("query", {}).get("geosearch", [])]


@cached("wiki_summary")
def _summary(title: str) -> dict | None:
    _limiter.wait()
    response = httpx.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}",
        headers={"User-Agent": _user_agent()},
        timeout=15.0,
        follow_redirects=True,
    )
    if response.status_code != 200:
        return None
    body = response.json()
    thumbnail = (body.get("thumbnail") or {}).get("source")
    if not thumbnail:
        return None
    return {
        "thumbnail": thumbnail,
        "extract": body.get("extract", ""),
        "title": body.get("title", title),
        "source": (body.get("content_urls", {}).get("desktop", {}) or {}).get("page", ""),
    }


def map_tile(lat: float, lon: float, zoom: int = 15) -> str:
    """A map tile showing where a place is, for anywhere with no article.

    Same tile server the live map already uses. Honest by construction: it
    really is that spot, where a stock photo of "a restaurant" would not be.
    """
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(max(-85.05, min(85.05, lat)))
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return f"https://a.basemaps.cartocdn.com/dark_all/{zoom}/{x}/{y}@2x.png"


#: Below this, the nearest article is about something else — a neighbouring
#: museum, or a painting hanging inside the one you asked for.
MATCH_FLOOR = 0.5


def venue_photo(name: str, lat: float | None = None, lon: float | None = None) -> dict[str, Any]:
    """A photograph and a line of description for one venue.

    Returns `{"found", "kind", "thumbnail", "extract", "source"}`. `kind` is
    `"photo"` for a real picture, `"map"` for the coordinate fallback, and
    `"none"` when there is not even a location to fall back to.

    Never raises. A picture is a nicety; failing to find one must not cost you
    anything else.
    """
    best: dict | None = None
    try:
        if lat is not None and lon is not None:
            candidates = [(t, _match_score(name, t)) for t in _nearby(lat, lon)]
            title, score = max(candidates, key=lambda c: c[1], default=("", 0.0))
            if score >= MATCH_FLOOR:
                best = _summary(title)
        if best is None and _tokens(name):
            # No coordinates, or nothing nearby matched. Worth one direct
            # lookup: unambiguous landmarks resolve by title alone.
            direct = _summary(name)
            if direct and _match_score(name, direct["title"]) >= MATCH_FLOOR:
                best = direct
    except Exception:  # noqa: BLE001 — fall through to the map
        best = None

    if best:
        return {"found": True, "kind": "photo", **best}
    if lat is not None and lon is not None:
        return {
            "found": True, "kind": "map", "thumbnail": map_tile(lat, lon),
            "extract": "", "title": name, "source": "",
        }
    return {"found": False, "kind": "none", "thumbnail": None, "extract": "", "title": name}


def photos_for(venues: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Look up a whole itinerary's venues, keyed by name.

    `venues` is `[{"name", "lat", "lon"}, ...]`. Duplicates collapse, since the
    same place scheduled twice is the same photograph.
    """
    out: dict[str, dict[str, Any]] = {}
    for venue in venues:
        name = str(venue.get("name") or "").strip()
        if not name or name in out:
            continue
        out[name] = venue_photo(name, venue.get("lat"), venue.get("lon"))
    return out


__all__ = ["MATCH_FLOOR", "map_tile", "photos_for", "venue_photo"]

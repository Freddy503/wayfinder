"""Venue ratings, extracted from Google-review data in search snippets.

The official Places API needs a billed Google key; this reads the rating that
search results already quote ("4.6 ★ (21,391 reviews)") and records where it
was read from. Cruder than the API, but grounded: every score carries a source
URL, and a venue whose rating can't be found is reported as unknown rather
than invented — the same honesty rule the opening-hours pipeline follows.
"""

from __future__ import annotations

import re

from wayfinder.tools.search import web_search

#: "4.6", "4,6" — a plausible 0–5 score with one decimal.
_SCORE = re.compile(
    r"(?<![\d.])([0-5][.,]\d)(?!\d)\s*(?:/\s*5|★|stars?|out of 5)?", re.I
)
#: "(21,391)", "21,391 reviews", "21.391 Google reviews"
_COUNT = re.compile(r"\(?([\d][\d.,]{2,})\)?\s*(?:google\s+)?(?:reviews|ratings|avis)", re.I)
_PAREN_COUNT = re.compile(r"\(([\d][\d.,]{2,})\)")


def _to_int(raw: str) -> int | None:
    digits = re.sub(r"[^\d]", "", raw)
    return int(digits) if digits else None


def parse_rating(text: str) -> tuple[float, int | None] | None:
    """Pull a (score, review_count) pair out of one snippet, if present.

    Requires a score; takes a count when one sits in the same snippet. A count
    with no score is noise, not a rating.
    """
    score_match = _SCORE.search(text)
    if not score_match:
        return None
    score = float(score_match.group(1).replace(",", "."))
    if not 0.0 <= score <= 5.0:
        return None

    count = None
    count_match = _COUNT.search(text) or _PAREN_COUNT.search(text)
    if count_match:
        count = _to_int(count_match.group(1))
        # A "count" smaller than 3 digits is usually a price or a year fragment.
        if count is not None and count < 10:
            count = None
    return score, count


def venue_rating(venue: str, city: str) -> dict:
    """Look up a venue's Google-review rating via web search.

    Use this while choosing between candidate venues — a 4.7 with thousands of
    reviews and a 3.9 with hundreds is a real signal, and the traveller sees
    the score on the itinerary. Record the result in the venue's `rating`
    field: `{"score": ..., "count": ..., "source": ...}`.

    Treat it as one criterion, not the criterion: a beloved neighbourhood
    tasca with 400 reviews can beat a tourist trap with 20,000. If nothing is
    found, leave `rating` unset rather than guessing.

    Args:
        venue: Exact venue name, e.g. "Cervejaria Ramiro".
        city: City it's in, e.g. "Lisbon".

    Returns:
        `{"found": true, "score", "count", "source", "snippet"}`, or
        `{"found": false}` when no plausible rating surfaced.
    """
    query = f"{venue} {city} google reviews rating"
    results = web_search(query, max_results=5).get("results", [])

    venue_tokens = [t for t in re.split(r"\W+", venue.lower()) if len(t) > 2]
    for result in results:
        blob = f"{result.get('title', '')} {result.get('content', '')}"
        # Only trust a snippet that is actually about this venue.
        if venue_tokens and not any(t in blob.lower() for t in venue_tokens):
            continue
        parsed = parse_rating(blob)
        if parsed is None:
            continue
        score, count = parsed
        return {
            "found": True,
            "score": score,
            "count": count,
            "source": result.get("url", ""),
            "snippet": blob[:220],
        }
    return {"found": False, "query": query}


#: Same reasoning as the geo batches: one turn per venue is the expensive part,
#: not the lookup. See `wayfinder/tools/geo.py` for the measurements.
MAX_BATCH = 40


def venue_ratings(venues: list[dict | str], city: str = "") -> dict:
    """Look up review ratings for several venues at once.

    Prefer this over calling `venue_rating` repeatedly — do your whole
    shortlist in one call.

    Args:
        venues: Names as plain strings — `["Belfort, Bruges", ...]` — or
            `[{"venue": ..., "city": ...}, ...]` if they're in different
            cities. Up to 40.
        city: Default city for entries that don't name one.

    Returns:
        `{"results": [...], "found": n}` — one entry per venue, in order, each
        shaped like `venue_rating`'s return plus the `venue` it was for.
        Venues with `found: false` should be left without a `rating` rather
        than guessed at.
    """
    if not isinstance(venues, list):
        venues = [venues]
    if len(venues) > MAX_BATCH:
        return {
            "error": f"{len(venues)} venues is more than the {MAX_BATCH} limit — "
            "rate your shortlist, not every candidate.",
            "results": [],
        }

    results = []
    for entry in venues:
        name = entry.get("venue", "") if isinstance(entry, dict) else str(entry)
        where = (entry.get("city") if isinstance(entry, dict) else "") or city
        name = str(name).strip()
        if not name:
            continue
        results.append({"venue": name, **venue_rating(name, str(where).strip())})

    return {"results": results, "found": sum(1 for r in results if r.get("found"))}

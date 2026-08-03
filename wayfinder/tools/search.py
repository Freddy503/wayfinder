"""Web search, via Tavily.

Results are cached by query, so a repeated eval run costs nothing and — more
importantly — returns the same sources it did last time.
"""

from __future__ import annotations

import os

import httpx

from wayfinder.tools.cache import cached

TAVILY_URL = "https://api.tavily.com/search"


@cached("tavily")
def _search_raw(query: str, max_results: int, depth: str) -> dict:
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not key:
        msg = "TAVILY_API_KEY is unset — set it in .env (see .env.example) to enable search."
        raise RuntimeError(msg)
    response = httpx.post(
        TAVILY_URL,
        json={
            "api_key": key,
            "query": query,
            "max_results": max_results,
            "search_depth": depth,
            "include_answer": False,
        },
        timeout=45.0,
    )
    response.raise_for_status()
    return response.json()


def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web for current information about places, hours, and prices.

    Use this whenever a fact would change the plan and you don't already know
    it for certain: opening hours, seasonal closures, ticket prices, whether a
    restaurant takes walk-ins. Prefer official sites over aggregators — the URLs
    you cite end up in the itinerary's `sources` and are what the groundedness
    check reads.

    Args:
        query: A specific question, e.g. "Museu do Azulejo opening hours Monday".
        max_results: How many results to return (1-10).

    Returns:
        `{"results": [{"title", "url", "content"}], "query": ...}`.
    """
    payload = _search_raw(query.strip(), max(1, min(max_results, 10)), "basic")
    return {
        "query": query,
        "results": [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "content": (r.get("content") or "")[:1500],
            }
            for r in payload.get("results", [])
        ],
    }

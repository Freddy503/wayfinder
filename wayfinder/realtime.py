"""Speech-to-speech trip intake over OpenAI's Realtime API.

A voice interviewer: it asks about the trip, listens, asks the next thing it
still needs, and stops when it has enough to plan. The model drives the
conversation; this module gives it the schema, the rules, and one tool for
writing what it has learned back into the form.

**On the model.** GPT-Live-1 and GPT-Live-1 mini are not in the API yet — they
shipped on 8 July 2026 as the ChatGPT voice experience, with a developer
waitlist and no announced GA. Everything here is written against the Realtime
API that *is* available, and the model id is a single environment variable, so
switching is `REALTIME_MODEL=gpt-live-1` once access lands.

**On credentials.** The browser never sees the API key. The server mints a
short-lived client secret and hands that over instead; a leaked ephemeral token
expires on its own, a leaked API key does not.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

CLIENT_SECRETS_URL = "https://api.openai.com/v1/realtime/client_secrets"

#: Available today. `gpt-live-1` / `gpt-live-1-mini` become valid values here
#: the moment OpenAI opens API access — nothing else in this file changes.
DEFAULT_MODEL = "gpt-realtime-2.1"
DEFAULT_VOICE = "marin"

#: The fields a trip cannot be planned without. Mirrors `REQUIRED` in
#: `extract.py` so the voice path and the typed path agree on "done".
ESSENTIAL = ("destination", "dates", "budget")

INTERVIEWER_INSTRUCTIONS = """\
You are a travel planner's intake interviewer, talking to someone out loud. \
Your one job is to find out enough about their trip to plan it, then stop.

You need three things before planning can start:
  1. Where they're going.
  2. When — both dates, or a start plus a number of nights.
  3. The budget, and the currency.

After those, ask about whatever will most change the plan: who's travelling, \
how full they want the days, dietary needs, anything they must not miss, \
whether they're flying and from where, and any hard limits like "nothing \
before ten" or "no more than half an hour between places".

How to talk:
- One question at a time. Short. This is speech, not a form.
- Never re-ask something they've already told you, in any words.
- When they correct themselves, the later answer wins. "Four hundred… no, six \
hundred" means six hundred.
- Accept vagueness and move on: "a long weekend in October" is enough to \
start; you can firm it up later.
- Don't read their answers back unless you genuinely didn't catch them.
- Don't discuss what you'll plan, or make suggestions. You are gathering \
requirements, not planning the trip.

Call `record_requirements` every time you learn something concrete — after \
almost every answer — with only the fields you learned. It is silent and \
costs the traveller nothing; calling it often is correct.

Once you have the three essentials plus a sense of what they enjoy, say one \
short sentence confirming you have enough, call `record_requirements` a final \
time with `ready` set to true, and stop asking questions.
"""

#: The one tool the voice model gets. It writes into the same fields the form
#: and the question wizard write into — three intakes, one `TripSpec`.
RECORD_TOOL: dict[str, Any] = {
    "type": "function",
    "name": "record_requirements",
    "description": (
        "Record what you have learned about the trip. Send only fields you "
        "actually heard; omit anything not yet mentioned. Call this after "
        "almost every answer."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "destination": {"type": "string", "description": "City and country."},
            "origin": {
                "type": "string",
                "description": "Where they fly from, if they're flying.",
            },
            "start": {"type": "string", "description": "Start date, YYYY-MM-DD."},
            "end": {"type": "string", "description": "End date, YYYY-MM-DD."},
            "adults": {"type": "integer"},
            "children": {"type": "integer"},
            "budget": {"type": "number", "description": "Total for the whole party."},
            "currency": {"type": "string", "description": "ISO code, e.g. EUR."},
            "pace": {"type": "string", "enum": ["relaxed", "standard", "packed"]},
            "early": {"type": "string", "description": "Earliest start, HH:MM."},
            "late": {"type": "string", "description": "Latest end, HH:MM."},
            "transit": {"type": "integer", "description": "Max minutes between stops."},
            "free": {"type": "integer", "description": "Minutes of downtime wanted daily."},
            "walk": {"type": "number", "description": "Max km on foot per day."},
            "meals": {
                "type": "array",
                "items": {"type": "string", "enum": ["breakfast", "lunch", "dinner"]},
            },
            "prefs": {"type": "array", "items": {"type": "string"}},
            "must": {"type": "array", "items": {"type": "string"}},
            "diet": {"type": "array", "items": {"type": "string"}},
            "ready": {
                "type": "boolean",
                "description": "True only when you have enough to plan the trip.",
            },
        },
        "additionalProperties": False,
    },
}


def is_configured() -> bool:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return bool(key) and not key.lower().startswith("paste")


def session_config(today: str) -> dict[str, Any]:
    """The session the ephemeral token is minted for."""
    return {
        "type": "realtime",
        "model": os.environ.get("REALTIME_MODEL", DEFAULT_MODEL),
        "instructions": f"{INTERVIEWER_INSTRUCTIONS}\nToday's date is {today}.",
        "audio": {
            "input": {
                # Server-side turn detection: the model decides when the
                # traveller has finished a thought, which is what makes this
                # feel like a conversation rather than a walkie-talkie.
                "turn_detection": {"type": "semantic_vad"},
            },
            "output": {"voice": os.environ.get("REALTIME_VOICE", DEFAULT_VOICE)},
        },
        "tools": [RECORD_TOOL],
        "tool_choice": "auto",
    }


def mint_client_secret(today: str, timeout: float = 20.0) -> dict[str, Any]:
    """Create a short-lived credential for the browser to connect with.

    Returns `{"ok": False, "reason": ...}` rather than raising — an intake
    session that cannot start is a UI state, not a server error.
    """
    if not is_configured():
        return {"ok": False, "reason": "OPENAI_API_KEY is not set"}

    try:
        response = httpx.post(
            CLIENT_SECRETS_URL,
            headers={
                "authorization": f"Bearer {os.environ['OPENAI_API_KEY'].strip()}",
                "content-type": "application/json",
            },
            json={"session": session_config(today)},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:300] if exc.response is not None else ""
        return {"ok": False, "reason": f"OpenAI returned {exc.response.status_code}: {body}"}
    except (httpx.HTTPError, ValueError) as exc:
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}

    # The token lives under `value` on the current shape; older previews
    # returned `client_secret.value`. Accept both rather than break on a rename.
    secret = payload.get("value") or (payload.get("client_secret") or {}).get("value")
    if not secret:
        return {"ok": False, "reason": f"no client secret in response: {sorted(payload)}"}

    return {
        "ok": True,
        "client_secret": secret,
        "expires_at": payload.get("expires_at"),
        "model": session_config(today)["model"],
    }


def missing_essentials(fields: dict[str, Any]) -> list[str]:
    """Which of the three planning essentials are still absent."""
    missing = []
    if not fields.get("destination"):
        missing.append("destination")
    if not (fields.get("start") and fields.get("end")):
        missing.append("dates")
    if not fields.get("budget"):
        missing.append("budget")
    return missing

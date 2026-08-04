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

#: Everything the interviewer has to ask about before it may stop, as
#: `(topic id, what to show on the board, which spec fields it fills)`.
#:
#: Deliberately wider than `ESSENTIAL`. A trip is *plannable* without knowing
#: the traveller's dietary needs; it is not *good*. But those two gates have to
#: stay separate — making "dietary" block the Plan button would strand everyone
#: who eats anything, and the code checker has nothing to say about tastes.
#: So: `ESSENTIAL` gates planning, `TOPICS` gates the interviewer's own `ready`.
TOPICS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("departure", "Flying from", ("origin",)),
    ("destination", "Destination", ("destination",)),
    ("dates", "Dates", ("start", "end")),
    ("travellers", "Travellers", ("adults", "children")),
    ("budget", "Budget", ("budget", "currency")),
    ("tastes", "Tastes", ("prefs", "pace")),
    ("must_do", "Must do", ("must",)),
    ("dietary", "Dietary", ("diet",)),
)

TOPIC_IDS = tuple(topic for topic, _, _ in TOPICS)

INTERVIEWER_INSTRUCTIONS = """\
You are a travel planner's intake interviewer, talking to someone out loud. \
Your one job is to find out enough about their trip to plan it well, then stop.

Work through all eight of these. Ask about every one — do not skip a topic \
because you can guess the answer, and do not stop early because you already \
have enough to sketch something.

  1. Departure — where they're flying from, or that they aren't flying.
  2. Destination — city and country.
  3. Dates — both, or a start plus a number of nights.
  4. Travellers — how many adults, and any children.
  5. Budget — the number, the currency, and whether it includes flights.
  6. Tastes — what they enjoy, and how full they want the days: a couple of \
things done properly, or as much as fits.
  7. Must do — anything the trip would be a failure without. Also worth asking \
what they'd rather avoid.
  8. Dietary — restrictions, allergies, or preferences.

The first five are the ones without which nothing can be planned; get those \
first, in whatever order the conversation goes. The last three are what make \
the plan theirs rather than generic, so ask them properly rather than as an \
afterthought — but a plain "no restrictions" or "nothing in particular" is a \
complete answer, and once you have it, move on and never ask again.

Then, if the conversation allows it, ask about hard limits worth respecting: \
"nothing before ten", "no more than half an hour between places", how far \
they're willing to walk in a day. These are optional — don't interrogate.

How to talk:
- One question at a time. Short. This is speech, not a form.
- Never re-ask something they've already told you, in any words.
- When they correct themselves, the later answer wins. "Four hundred… no, six \
hundred" means six hundred.
- Accept vagueness and move on: "a long weekend in October" is enough to \
start; you can firm it up later.
- You may put two closely-related things in one breath — "just the two of you, \
or is anyone else coming?" — but never stack unrelated topics.
- Don't read their answers back unless you genuinely didn't catch them.
- Don't discuss what you'll plan, or make suggestions. You are gathering \
requirements, not planning the trip.

Call `record_requirements` after almost every answer, with only the fields you \
just learned, and with `covered` listing every topic you have now asked about \
and received an answer to — including topics where the answer was "none". That \
is how the board knows what is left; a topic you asked about but leave out of \
`covered` will be shown as still unasked. It is silent and costs the traveller \
nothing; calling it often is correct.

Only once all eight topics are covered: say one short sentence confirming you \
have what you need, call `record_requirements` a final time with `ready` set \
to true, and stop asking questions.
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
            "excludes_flights": {
                "type": "boolean",
                "description": "True if the budget is for the trip excluding airfare.",
            },
            "avoid": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Things they'd rather not do.",
            },
            "covered": {
                "type": "array",
                "items": {"type": "string", "enum": list(TOPIC_IDS)},
                "description": (
                    "Every topic you have asked about and had answered, "
                    "including ones answered with 'none' or 'no preference'. "
                    "Send the full list each time, not just the newest."
                ),
            },
            "ready": {
                "type": "boolean",
                "description": "True only once all eight topics are covered.",
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
    """Which of the three planning essentials are still absent.

    This is the Plan button's gate, and it stays narrow on purpose: these are
    the three the constraint checker cannot run without.
    """
    missing = []
    if not fields.get("destination"):
        missing.append("destination")
    if not (fields.get("start") and fields.get("end")):
        missing.append("dates")
    if not fields.get("budget"):
        missing.append("budget")
    return missing


def uncovered_topics(fields: dict[str, Any], covered: Any = ()) -> list[str]:
    """Which of the eight interview topics have not been asked about yet.

    A topic counts as covered two ways: the model said so in `covered`, or one
    of its fields came back filled. The second path is the safety net — a model
    that forgets to maintain `covered` still visibly makes progress, rather
    than driving the board to a permanent "nothing asked yet".

    The reverse doesn't hold, which is the whole reason `covered` exists: "no
    dietary restrictions" is a complete answer that sets no field, and without
    an explicit signal the interviewer would ask it forever.
    """
    claimed = {topic for topic in (covered or ()) if isinstance(topic, str)}
    return [
        topic
        for topic, _, keys in TOPICS
        if topic not in claimed
        and not any(_has_value(fields.get(key)) for key in keys)
    ]


def _has_value(value: Any) -> bool:
    """Zero is an answer: 0 children and a 0 budget are not the same as silence."""
    if value is None or value == "":
        return False
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return True

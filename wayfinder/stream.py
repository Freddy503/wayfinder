"""Turn LangGraph's stream into typed events the browser can render.

LangGraph emits state deltas — the shape it uses to run a graph, not a shape
anyone wants to render. This module is the translation layer, and it exists so
the UI never has to know what an `AIMessage` is.

Two rules it follows throughout:

- **Never raise.** A stream that dies because one chunk had an unexpected shape
  takes the whole run's visibility with it. Anything unrecognised is skipped,
  and anything malformed degrades to a coarser event rather than an exception.
- **Namespace is nesting.** `subgraphs=True` tags every chunk with the path of
  the graph that produced it, which is what lets the UI indent a subagent's
  tool calls underneath the dispatch that spawned them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

#: Tools whose results deserve their own event type rather than a text preview.
CHECK_TOOLS = {"check_itinerary", "finalize_itinerary"}

#: How much of a tool result to show before truncating. Research subagents
#: return whole documents; the feed wants a glance, not the transcript.
PREVIEW_CHARS = 600

#: Subagent reports get far more room. They are the only place in the run where
#: something says what it *found* — restaurants, opening hours, a warning about
#: Mondays — rather than what it did.
REPORT_CHARS = 6000


@dataclass
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def sse(self) -> str:
        """Render as a Server-Sent Event frame."""
        payload = json.dumps({"type": self.type, **self.data}, default=str)
        return f"data: {payload}\n\n"


def _text_of(content: Any) -> str:
    """Flatten message content, which may be a string or a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def _preview(value: Any, limit: int = PREVIEW_CHARS) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + f"… (+{len(text) - limit} chars)"


def _namespace_label(namespace: tuple[str, ...]) -> str:
    """A stable id for the graph that produced a chunk.

    Empty tuple is the main agent. Anything deeper is a subagent, and the
    string is used by the UI purely as a grouping key.
    """
    return "main" if not namespace else "/".join(str(n) for n in namespace)


class StreamTranslator:
    """Stateful translator — it pairs tool results back to their calls.

    Stateful because a `ToolMessage` carries only a `tool_call_id`; the name and
    arguments arrived earlier on a different chunk. Without the pairing the UI
    would show results it cannot label.
    """

    def __init__(self) -> None:
        self._calls: dict[str, dict[str, Any]] = {}
        self._seen_text: set[str] = set()
        self._todos: list[dict[str, Any]] = []

    def translate(self, chunk: Any) -> list[Event]:
        """Convert one raw stream chunk into zero or more events."""
        try:
            namespace, mode, payload = self._unpack(chunk)
        except Exception:  # noqa: BLE001 — a weird chunk must not kill the stream
            return []

        if mode != "updates" or not isinstance(payload, dict):
            return []

        events: list[Event] = []
        for node_name, update in payload.items():
            if node_name == "__interrupt__":
                events.extend(self._interrupts(update))
                continue
            if not isinstance(update, dict):
                continue
            events.extend(self._todo_events(update))
            for message in update.get("messages", []) or []:
                events.extend(self._message_events(message, namespace))
        return events

    # -- chunk shapes ------------------------------------------------------

    @staticmethod
    def _unpack(chunk: Any) -> tuple[tuple[str, ...], str, Any]:
        """Normalise the several shapes `stream()` can yield.

        With `subgraphs=True` and multiple modes it's a 3-tuple; with one mode
        a 2-tuple; without subgraphs, neither carries a namespace. Handling all
        of them here keeps the call sites free of shape checks.
        """
        if isinstance(chunk, tuple):
            if len(chunk) == 3:
                namespace, mode, payload = chunk
                return tuple(namespace or ()), mode, payload
            if len(chunk) == 2:
                first, second = chunk
                if isinstance(first, tuple):
                    return tuple(first or ()), "updates", second
                return (), str(first), second
        return (), "updates", chunk

    # -- events ------------------------------------------------------------

    def _message_events(self, message: Any, namespace: tuple[str, ...]) -> list[Event]:
        events: list[Event] = []
        agent = _namespace_label(namespace)
        tool_calls = getattr(message, "tool_calls", None) or []
        tool_call_id = getattr(message, "tool_call_id", None)

        if tool_call_id:
            events.append(self._tool_result(message, tool_call_id, agent))
            # Geocode hits additionally become map points, so the UI can plot
            # the research as it happens rather than only the finished route.
            geo = self._geo_event(message, tool_call_id, agent)
            if geo is not None:
                events.append(geo)
            return events

        text = _text_of(getattr(message, "content", "")).strip()
        if text:
            # The same assistant turn can arrive on more than one chunk; a hash
            # is cheaper than reconstructing message identity.
            key = f"{agent}:{hash(text)}"
            if key not in self._seen_text:
                self._seen_text.add(key)
                events.append(Event("agent.text", {"agent": agent, "text": text}))

        for call in tool_calls:
            name = call.get("name", "?")
            args = call.get("args", {}) or {}
            call_id = call.get("id") or f"{name}-{len(self._calls)}"
            self._calls[call_id] = {"name": name, "args": args, "agent": agent}

            if name == "task":
                role = args.get("subagent_type") or args.get("name") or "?"
                label, icon = describe_role(role)
                events.append(
                    Event(
                        "subagent.start",
                        {
                            "id": call_id,
                            "agent": agent,
                            "subagent": role,
                            "role": label,
                            "icon": icon,
                            "task": _preview(
                                args.get("description") or args.get("task") or "", 300
                            ),
                        },
                    )
                )
            else:
                events.append(
                    Event(
                        "tool.call",
                        {
                            "id": call_id,
                            "agent": agent,
                            "name": name,
                            "args": args,
                            "summary": _summarise_call(name, args),
                            # What the traveller reads. `summary` stays as-is
                            # for the developer feed and the approval card.
                            "narration": narrate(name, args),
                        },
                    )
                )
        return events

    def _tool_result(self, message: Any, call_id: str, agent: str) -> Event:
        origin = self._calls.get(call_id, {})
        name = origin.get("name") or getattr(message, "name", None) or "?"
        content = getattr(message, "content", "")
        status = getattr(message, "status", None)
        ok = status != "error"

        if name == "task":
            role = (
                origin.get("args", {}).get("subagent_type")
                or origin.get("args", {}).get("name")
                or "?"
            )
            label, icon = describe_role(role)
            return Event(
                "subagent.end",
                {
                    "id": call_id,
                    "agent": agent,
                    "subagent": role,
                    "role": label,
                    "icon": icon,
                    # Longer than a glance: this is the one place a research
                    # subagent says what it actually found, and truncating it
                    # to a preview was throwing away the only interesting text
                    # in the run.
                    "report": _preview(content, REPORT_CHARS),
                    "findings": _findings(content),
                    "ok": ok,
                },
            )

        if name in CHECK_TOOLS:
            parsed = _as_dict(content)
            if parsed is not None:
                return Event(
                    "check",
                    {
                        "id": call_id,
                        "agent": agent,
                        "tool": name,
                        "passed": bool(parsed.get("passed")),
                        "summary": parsed.get("summary", ""),
                        "agent_summary": parsed.get("agent_summary"),
                        "violations": parsed.get("violations", []),
                        "metrics": parsed.get("metrics", {}),
                    },
                )

        return Event(
            "tool.result",
            {
                "id": call_id,
                "agent": agent,
                "name": name,
                "ok": ok,
                "preview": _preview(content),
            },
        )

    def _geo_event(self, message: Any, call_id: str, agent: str) -> Event | None:
        """A successful geocode, as a plottable point."""
        origin = self._calls.get(call_id, {})
        if origin.get("name") != "geocode":
            return None
        parsed = _as_dict(getattr(message, "content", ""))
        if not parsed or not parsed.get("found"):
            return None
        try:
            lat, lon = float(parsed["lat"]), float(parsed["lon"])
        except (KeyError, TypeError, ValueError):
            return None
        return Event(
            "geo",
            {
                "agent": agent,
                "name": parsed.get("name") or origin.get("args", {}).get("place", ""),
                "lat": lat,
                "lon": lon,
            },
        )

    def _todo_events(self, update: dict[str, Any]) -> list[Event]:
        todos = update.get("todos")
        if not isinstance(todos, list) or todos == self._todos:
            return []
        self._todos = todos
        items = [
            {
                "content": t.get("content", "") if isinstance(t, dict) else str(t),
                "status": t.get("status", "pending") if isinstance(t, dict) else "pending",
            }
            for t in todos
        ]
        return [Event("todos", {"items": items})]

    @staticmethod
    def _interrupts(update: Any) -> list[Event]:
        """Unpack a `__interrupt__` update into a reviewable request.

        The payload is the `HITLRequest` the middleware raised: the actions
        awaiting a decision, plus which decisions each one allows.
        """
        entries = update if isinstance(update, (list, tuple)) else [update]
        events: list[Event] = []
        for entry in entries:
            # An `Interrupt` exposes `.value`; a serialised one is a dict with a
            # "value" key. Reading only the attribute silently drops the dict
            # form — and a dropped interrupt is a UI that hangs with no prompt.
            if isinstance(entry, dict):
                value = entry.get("value", entry)
            else:
                value = getattr(entry, "value", entry)
            if not isinstance(value, dict):
                continue
            requests = value.get("action_requests", []) or []
            configs = {
                c.get("action_name"): c for c in value.get("review_configs", []) or [] if c
            }

            # A change request is a question about the trip, not a tool awaiting
            # approval. It gets its own event so the UI can ask it in the
            # traveller's language instead of rendering an args blob.
            asks = [r for r in requests if r.get("name") == "request_change"]
            for ask in asks:
                args = ask.get("args", {}) or {}
                events.append(
                    Event(
                        "change.requested",
                        {
                            "constraint": args.get("constraint", ""),
                            "problem": args.get("problem", ""),
                            "suggestion": args.get("suggestion", ""),
                            "shortfall": args.get("shortfall", ""),
                        },
                    )
                )
            requests = [r for r in requests if r.get("name") != "request_change"]
            if not requests:
                continue

            events.append(
                Event(
                    "interrupt",
                    {
                        "actions": [
                            {
                                "name": r.get("name", "?"),
                                "args": r.get("args", {}),
                                "description": r.get("description")
                                or _summarise_call(r.get("name", ""), r.get("args", {}) or {}),
                                "allowed": (configs.get(r.get("name")) or {}).get(
                                    "allowed_decisions", ["approve", "reject"]
                                ),
                                "args_schema": (configs.get(r.get("name")) or {}).get(
                                    "args_schema"
                                ),
                            }
                            for r in requests
                        ]
                    },
                )
            )
        return events


#: A bullet worth showing: long enough to say something, short enough to read
#: at a glance while the run is going.
_MIN_FINDING, _MAX_FINDING = 12, 240
_MAX_FINDINGS = 12


def _findings(content: Any) -> list[dict[str, str]]:
    """Pull the named things out of a subagent's report.

    Research subagents write markdown — headings, bullets, the occasional
    table. The traveller wants *"Cervejaria Ramiro — seafood, no reservations,
    expect a queue"*, not four hundred words of prose with that sentence
    somewhere inside it. So: take the bullets, split each into a name and the
    rest, and let the UI lay them out as cards.

    Best-effort by design. A report that doesn't use bullets yields nothing
    here and the full text is still shown underneath, so a subagent that writes
    in paragraphs degrades to what we had before rather than to a blank panel.
    """
    text = content if isinstance(content, str) else _preview(content, REPORT_CHARS)
    findings: list[dict[str, str]] = []
    section = ""

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            section = line.lstrip("#").strip()
            continue
        if not (line.startswith(("- ", "* ", "+ ")) or _numbered(line)):
            continue

        body = line.lstrip("-*+ ").strip()
        body = body.split(". ", 1)[1] if _numbered(line) and ". " in body else body
        if not _MIN_FINDING <= len(body) <= _MAX_FINDING:
            continue

        # "**Name** — detail", "Name: detail", "Name — detail" all show up
        # depending on the model's mood. Split on whichever comes first.
        name, detail = body, ""
        stripped = body.replace("**", "")
        for sep in (" — ", " – ", ": ", " - "):
            if sep in stripped:
                name, detail = stripped.split(sep, 1)
                break

        findings.append(
            {"name": name.strip(" *_"), "detail": detail.strip(), "section": section}
        )
        if len(findings) >= _MAX_FINDINGS:
            break
    return findings


def _numbered(line: str) -> bool:
    head = line.split(".", 1)[0]
    return head.isdigit() and len(head) <= 2 and "." in line


def _as_dict(content: Any) -> dict | None:
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _summarise_call(name: str, args: dict[str, Any]) -> str:
    """A one-line human description of a tool call, for the feed and the
    approval card. Falls back to compact JSON for anything unrecognised."""
    if name == "web_search":
        return str(args.get("query", ""))
    if name == "geocode":
        return str(args.get("place", ""))
    if name == "venue_rating":
        return f"{args.get('venue', '?')} ({args.get('city', '?')})"
    if name == "estimate_travel":
        return (
            f"{args.get('origin', '?')} → {args.get('destination', '?')}"
            f" ({args.get('mode', 'walk')})"
        )
    if name == "fx_convert":
        return (
            f"{args.get('amount', '?')} {args.get('from_currency', '?')}"
            f" → {args.get('to_currency', '?')}"
        )
    if name in {"write_file", "read_file", "edit_file"}:
        return str(args.get("file_path") or args.get("path", ""))
    if name == "finalize_itinerary":
        return _preview(args.get("summary", ""), 300)
    if name in CHECK_TOOLS:
        return str(args.get("path", "itinerary.json"))
    return _preview(args, 200)


#: What each tool is *doing*, in the traveller's terms. The feed used to read
#: `geocode → Praça do Comércio`, which tells you the harness is working and
#: nothing about your trip. This reads as someone narrating their research.
_NARRATION: dict[str, str] = {
    "web_search": "Looking up",
    "geocode": "Finding",
    "venue_rating": "Checking reviews for",
    "estimate_travel": "Timing the trip from",
    "fx_convert": "Converting",
    "flight_search": "Searching flights",
    "check_itinerary": "Checking the plan",
    "finalize_itinerary": "Finishing up",
    "write_file": "Writing",
    "read_file": "Re-reading",
    "edit_file": "Revising",
    "write_todos": "Planning the work",
}

#: Subagent names as a traveller would describe the job.
_ROLES: dict[str, tuple[str, str]] = {
    "scout": ("Neighbourhoods & sights", "🗺"),
    "food": ("Places to eat", "🍽"),
    "logistics": ("Getting around", "🚊"),
    "researcher": ("Research", "🔍"),
    "critic": ("Second opinion", "🧐"),
}


def narrate(name: str, args: dict[str, Any]) -> str:
    """One human sentence for a tool call.

    Falls back to the tool's own name rather than inventing a phrase for
    something unrecognised — a wrong description is worse than a bare one.
    """
    detail = _summarise_call(name, args)
    verb = _NARRATION.get(name)
    if verb is None:
        return f"{name}{f' — {detail}' if detail else ''}"
    if name in {"check_itinerary", "finalize_itinerary", "write_todos", "flight_search"}:
        return verb
    return f"{verb} {detail}".strip() if detail else verb


def describe_role(subagent: str) -> tuple[str, str]:
    """Label and icon for a subagent, defaulting to its own name."""
    return _ROLES.get(subagent, (subagent.replace("_", " ").title(), "•"))

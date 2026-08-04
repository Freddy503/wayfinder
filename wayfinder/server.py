"""Local web server: stream a planning run, and answer its interrupts.

The shape that makes human-in-the-loop work here:

    browser ──POST /api/plan──▶ RunSession ──▶ worker thread ──▶ agent.stream()
       ▲                            │                                 │
       └──GET /api/stream (SSE)─────┘◀──── events queue ◀─────────────┘
       │
       └──POST /api/decide──▶ threading.Event ──▶ Command(resume=…)

The graph is synchronous and an interrupt ends the stream, so the worker runs
in its own thread and loops: stream until it stops, and if the state shows a
pending interrupt, block on a `threading.Event` until the browser posts a
decision, then resume. The SSE connection never closes across that pause — it
is draining a queue, not the graph — so from the browser it looks like one
continuous run that happens to ask questions.

Local-only by design: it binds to 127.0.0.1, holds runs in memory, and has no
authentication. Don't expose it.
"""

from __future__ import annotations

import json
import queue
import threading
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from wayfinder.agent import (
    AgentConfig,
    build_agent,
    finalise,
    new_run_dir,
    user_message,
)
from wayfinder.evals.datasets import load_cases
from wayfinder.schema import TripSpec
from wayfinder.stream import Event, StreamTranslator

WEB_DIR = Path(__file__).resolve().parent / "web"
RUNS_ROOT = (Path(__file__).resolve().parent.parent / "runs").resolve()

#: Terminates a finished SSE stream. The browser closes on seeing it rather
#: than reconnecting, which EventSource would otherwise do automatically.
DONE = Event("run.done", {})


class PlanRequest(BaseModel):
    spec: dict[str, Any]
    #: Deferred to `AgentConfig` rather than repeated. Hardcoding it here meant
    #: the CLI and the browser could disagree about which model runs — and they
    #: silently did, so a provider switch left the web path billing the old one.
    model: str = Field(default_factory=lambda: AgentConfig().model)
    subagents: bool = True
    skills: bool = True
    repair: bool = True
    single_researcher: bool = False
    interrupt_on: list[str] = Field(default_factory=lambda: ["finalize_itinerary"])


class ExtractRequest(BaseModel):
    transcript: str = Field(max_length=20_000)


class Decision(BaseModel):
    """One human answer to one pending action."""

    type: str  # approve | edit | reject | respond
    message: str | None = None
    args: dict[str, Any] | None = None


class DecisionRequest(BaseModel):
    decisions: list[Decision]


class RunSession:
    """One planning run: a worker thread, an event queue, and a resume latch."""

    def __init__(self, request: PlanRequest) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.spec = TripSpec.model_validate(request.spec)
        self.config = AgentConfig(
            model=request.model,
            use_subagents=request.subagents,
            use_skills=request.skills,
            use_repair_loop=request.repair,
            single_researcher=request.single_researcher,
            use_finalize=bool(request.interrupt_on),
            interrupt_on=tuple(request.interrupt_on),
        )
        self.run_dir = new_run_dir(self.spec)
        self.queue: queue.Queue[Event | None] = queue.Queue()
        self.status = "starting"
        self.result: dict[str, Any] | None = None
        self.pending: list[dict[str, Any]] = []

        self._decision: Any = None
        self._decision_ready = threading.Event()
        self._counter = {"calls": 0}
        self._thread = threading.Thread(target=self._run, daemon=True)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def decide(self, decisions: list[Decision]) -> None:
        """Hand the worker a human decision and let it resume."""
        if self.status != "waiting":
            msg = f"run is {self.status}, not waiting for a decision"
            raise HTTPException(status_code=409, detail=msg)
        self._decision = {"decisions": [_to_langgraph(d) for d in decisions]}
        self.pending = []
        self._decision_ready.set()

    def events(self):
        """Drain the queue as SSE frames until the run ends."""
        while True:
            event = self.queue.get()
            if event is None:
                yield DONE.sse()
                return
            yield event.sse()

    # -- worker ------------------------------------------------------------

    def _emit(self, event: Event) -> None:
        self.queue.put(event)

    def _run(self) -> None:
        translator = StreamTranslator()
        error: str | None = None
        try:
            # Same snapshot the CLI path writes: what ran, with which config —
            # a run directory should be self-describing regardless of entry point.
            (self.run_dir / "spec.yaml").write_text(
                yaml.safe_dump(
                    self.spec.model_dump(mode="json"), sort_keys=False, allow_unicode=True
                ),
                encoding="utf-8",
            )
            (self.run_dir / "config.json").write_text(
                json.dumps(asdict(self.config), indent=2, default=str), encoding="utf-8"
            )
            agent = build_agent(
                self.spec,
                self.run_dir,
                self.config,
                self._counter,
                checkpointer=InMemorySaver(),
            )
            graph_config = {
                "configurable": {"thread_id": self.id},
                "recursion_limit": self.config.recursion_limit,
                "run_name": f"wayfinder-web:{self.spec.slug}",
                "metadata": {"wayfinder_config": self.config.label(), "run_id": self.id},
            }

            self.status = "running"
            self._emit(
                Event(
                    "run.started",
                    {
                        "run_id": self.id,
                        "run_dir": str(self.run_dir),
                        "destination": self.spec.destination,
                        "config": self.config.label(),
                        "interrupt_on": list(self.config.interrupt_on),
                    },
                )
            )

            payload: Any = {"messages": [{"role": "user", "content": user_message(self.spec)}]}

            while True:
                for chunk in agent.stream(
                    payload,
                    config=graph_config,
                    stream_mode=["updates"],
                    subgraphs=True,
                ):
                    for event in translator.translate(chunk):
                        if event.type == "interrupt":
                            self.pending = event.data.get("actions", [])
                        self._emit(event)

                # The stream ending means either the run finished or it is
                # parked on an interrupt. Only the checkpoint knows which.
                state = agent.get_state(graph_config)
                if not getattr(state, "interrupts", None):
                    break

                self.status = "waiting"
                self._emit(Event("run.waiting", {"actions": self.pending}))
                self._decision_ready.wait()
                self._decision_ready.clear()
                self.status = "running"
                self._emit(Event("run.resumed", {}))
                payload = Command(resume=self._decision)

        except Exception as exc:  # noqa: BLE001 — surface it, don't kill the server
            error = f"{type(exc).__name__}: {exc}"
            self._emit(Event("run.error", {"message": error, "trace": traceback.format_exc()}))

        # A failed run still gets scored, exactly as on the CLI path, so the
        # browser always ends with a verdict rather than an empty panel.
        try:
            result = finalise(
                self.spec,
                self.config,
                self.run_dir,
                check_calls=self._counter["calls"],
                messages=[],
                error=error,
            )
            self.result = {
                "passed": result.report.passed,
                "metrics": result.report.metrics,
                "violations": [v.to_dict() for v in result.report.violations],
                # Every check, not just the failures — a verdict that only
                # lists what broke can't tell a thorough pass from a shallow one.
                "checks": result.report.to_dict()["checks"],
                "check_calls": result.check_calls,
                "run_dir": str(result.run_dir),
                "error": result.error,
                "has_itinerary": result.itinerary is not None,
                "feasible": result.itinerary.feasible if result.itinerary else None,
                "infeasibility_reason": (
                    result.itinerary.infeasibility_reason if result.itinerary else None
                ),
            }
            self._emit(Event("run.finished", self.result))
        except Exception as exc:  # noqa: BLE001
            self._emit(Event("run.error", {"message": f"scoring failed: {exc}"}))

        self.status = "finished"
        self.queue.put(None)


def _to_langgraph(decision: Decision) -> dict[str, Any]:
    """Map the browser's decision onto the middleware's payload shape."""
    kind = decision.type
    if kind == "approve":
        return {"type": "approve"}
    if kind == "edit":
        return {"type": "edit", "edited_action": {"args": decision.args or {}}}
    if kind == "respond":
        return {"type": "respond", "message": decision.message or ""}
    payload: dict[str, Any] = {"type": "reject"}
    if decision.message:
        payload["message"] = decision.message
    return payload


def create_app() -> FastAPI:
    app = FastAPI(title="Wayfinder", docs_url=None, redoc_url=None)
    runs: dict[str, RunSession] = {}

    @app.post("/api/plan")
    def start_plan(request: PlanRequest) -> dict[str, str]:
        session = RunSession(request)
        runs[session.id] = session
        session.start()
        return {"run_id": session.id}

    @app.get("/api/stream/{run_id}")
    def stream(run_id: str) -> StreamingResponse:
        session = runs.get(run_id)
        if session is None:
            raise HTTPException(status_code=404, detail="no such run")
        return StreamingResponse(
            session.events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/decide/{run_id}")
    def decide(run_id: str, request: DecisionRequest) -> dict[str, str]:
        session = runs.get(run_id)
        if session is None:
            raise HTTPException(status_code=404, detail="no such run")
        session.decide(request.decisions)
        return {"status": "resumed"}

    @app.get("/api/runs/{run_id}")
    def run_status(run_id: str) -> dict[str, Any]:
        session = runs.get(run_id)
        if session is None:
            raise HTTPException(status_code=404, detail="no such run")
        return {
            "run_id": session.id,
            "status": session.status,
            "pending": session.pending,
            "result": session.result,
        }

    def _run_dir_for(run_id: str) -> Path:
        """Resolve a run id to a directory, in memory or on disk.

        The registry only holds runs from this process, so a restart would
        otherwise orphan every past trip — no revisiting, no re-export. Past
        runs are addressed by directory name, which is already unique.
        """
        session = runs.get(run_id)
        if session is not None:
            return session.run_dir
        candidate = RUNS_ROOT / Path(run_id).name
        if candidate.is_dir() and candidate.parent == RUNS_ROOT:
            return candidate
        raise HTTPException(status_code=404, detail="no such run")

    @app.get("/api/history")
    def history() -> list[dict[str, Any]]:
        """Past runs on disk, newest first — so a trip outlives the process."""
        if not RUNS_ROOT.is_dir():
            return []
        entries = []
        for path in sorted(RUNS_ROOT.iterdir(), reverse=True):
            itinerary = path / "itinerary.json"
            if not (path.is_dir() and itinerary.is_file()):
                continue
            entry: dict[str, Any] = {"run_id": path.name, "destination": path.name}
            try:
                data = json.loads(itinerary.read_text(encoding="utf-8"))
                entry["destination"] = data.get("destination", path.name)
                entry["days"] = len(data.get("days", []))
                entry["feasible"] = data.get("feasible", True)
            except (OSError, json.JSONDecodeError):
                continue
            report = path / "constraints.json"
            if report.is_file():
                try:
                    entry["passed"] = json.loads(report.read_text(encoding="utf-8"))["passed"]
                except (OSError, json.JSONDecodeError, KeyError):
                    pass
            entries.append(entry)
        return entries[:40]

    @app.get("/api/runs/{run_id}/artifact/{name}")
    def artifact(run_id: str, name: str) -> FileResponse:
        run_dir = _run_dir_for(run_id)
        # Basename only — the name comes from the browser, and a run directory
        # is not somewhere to allow path traversal from.
        path = run_dir / Path(name).name
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"no artifact {name!r}")
        return FileResponse(path)

    @app.post("/api/extract")
    def extract(request: ExtractRequest) -> dict[str, Any]:
        """Turn a speech-mode transcript into form fields.

        Called on a debounce as the transcript grows; each call re-extracts the
        whole text so later corrections override earlier statements.
        """
        from wayfinder.extract import extract_requirements

        try:
            return extract_requirements(request.transcript)
        except Exception as exc:  # noqa: BLE001 — a failed extraction is a UI state
            return {
                "fields": {},
                "missing": ["destination", "dates", "budget"],
                "follow_up": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    @app.post("/api/realtime/token")
    def realtime_token() -> dict[str, Any]:
        """Mint an ephemeral credential for the browser's voice session.

        The API key stays here. The browser gets a short-lived secret that
        expires on its own, which is the difference between a leaked token and
        a leaked account.
        """
        from datetime import date

        from wayfinder.realtime import mint_client_secret

        return mint_client_secret(date.today().isoformat())

    @app.get("/api/realtime/topics")
    def realtime_topics() -> dict[str, Any]:
        """The interview checklist, so the board can show it before you start.

        Split from the token route because that one mints a real credential:
        rendering a checklist on page load shouldn't cost an unused ephemeral
        token every time someone opens the page.
        """
        from wayfinder.realtime import TOPICS, is_configured

        return {
            "topics": [{"id": t, "label": label} for t, label, _ in TOPICS],
            "configured": is_configured(),
        }

    @app.get("/api/examples")
    def examples() -> list[dict[str, Any]]:
        """The eval dataset, reused as one-click starting points."""
        return [
            {"name": c.name, "category": c.category, "spec": c.inputs()["spec"]}
            for c in load_cases()
        ]

    @app.get("/api/defaults")
    def defaults() -> dict[str, Any]:
        return {
            "config": asdict(AgentConfig()),
            "tools": [
                "web_search",
                "geocode",
                "estimate_travel",
                "fx_convert",
                "check_itinerary",
                "finalize_itinerary",
                "task",
            ],
        }

    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    return app

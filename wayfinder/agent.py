"""Agent assembly and the run loop.

Everything that varies between experiments lives in `AgentConfig`. That's not
premature generality — the point of the project is to answer "does this part of
the harness earn its tokens?", and you can only answer that if each part can be
switched off without touching the code around it.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from wayfinder.models import DEFAULT_MODEL, resolve_model
from wayfinder.prompts import MAIN_PROMPT
from wayfinder.render import render_markdown, render_sources
from wayfinder.schema import Itinerary, TripSpec
from wayfinder.specs import load_itinerary_payload
from wayfinder.tools.flights import flight_search
from wayfinder.tools.geo import estimate_travel, geocode
from wayfinder.tools.money import fx_convert
from wayfinder.tools.ratings import venue_rating
from wayfinder.tools.search import web_search
from wayfinder.verify import ConstraintReport, check_payload

ITINERARY_FILE = "itinerary.json"

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_SOURCE = REPO_ROOT / "skills"

#: Skills resolve through the *backend*, not the host filesystem, and the
#: backend is rooted at the run directory. So they get copied in at setup and
#: referenced by their virtual path. The copy is a happy side effect: each run
#: keeps a snapshot of the skills it actually ran with, which is what makes an
#: old experiment reproducible after you've edited them.
SKILLS_VIRTUAL_PATH = "/skills"


@dataclass
class AgentConfig:
    """One point in the experiment space.

    Every flag here is an axis the eval matrix sweeps. The defaults are the
    full harness on Sonnet 5 — the baseline everything else is measured
    against. The `opus-throughout` arm asks whether the stronger model is worth
    roughly double the spend, which is a question the numbers should answer
    rather than a guess made up front.
    """

    #: A model spec (`provider:id`) or a chat model instance.
    model: Any = DEFAULT_MODEL
    #: Subagents inherit `model` unless this is set. Sweeping it answers "does
    #: the cheap model do the research just as well?"
    subagent_model: str | None = None
    use_subagents: bool = True
    use_skills: bool = True
    #: With this off the agent never sees `check_itinerary`. The single most
    #: interesting ablation: it isolates the value of an in-loop verifier.
    use_repair_loop: bool = True
    #: Collapses the three specialists into one generalist researcher.
    single_researcher: bool = False
    #: Gives the agent an explicit "I'm done" tool. Off in the eval matrix —
    #: it's a review gate for interactive use, not something to measure.
    use_finalize: bool = False
    #: Tool names that pause for human approval. Needs a checkpointer.
    interrupt_on: tuple[str, ...] = ()
    recursion_limit: int = 250

    def label(self) -> str:
        from wayfinder.models import label_of

        bits = [label_of(self.model)]
        if not self.use_repair_loop:
            bits.append("no-repair")
        if not self.use_skills:
            bits.append("no-skills")
        if not self.use_subagents:
            bits.append("no-subagents")
        elif self.single_researcher:
            bits.append("one-researcher")
        if isinstance(self.subagent_model, str):
            from wayfinder.models import label_of

            bits.append(f"sub={label_of(self.subagent_model)}")
        return "+".join(bits)


@dataclass
class RunResult:
    run_dir: Path
    spec: TripSpec
    config: AgentConfig
    report: ConstraintReport
    itinerary: Itinerary | None
    payload: Any | None
    error: str | None = None
    check_calls: int = 0
    messages: list[Any] = field(default_factory=list)

    def tool_calls(self) -> dict[str, int]:
        """How many times each tool was called, from the message history.

        Effort is otherwise invisible to the experiment view: two runs can
        both score a perfect 1.0 while one did 55 tool calls and the other
        145. On easy cases — where every quality metric saturates — this is
        the only column that can still tell two arms apart.
        """
        counts: dict[str, int] = {}
        for message in self.messages:
            for call in getattr(message, "tool_calls", None) or []:
                name = call.get("name", "?")
                counts[name] = counts.get(name, 0) + 1
        return counts


def make_check_tool(spec: TripSpec, run_dir: Path, counter: dict[str, int]):
    """Build the `check_itinerary` tool, bound to this run's spec and directory.

    The same `check_payload` the evaluators call — so the agent is optimising
    against the exact function that will later grade it, not an approximation
    of it.
    """

    def check_itinerary(path: str = ITINERARY_FILE) -> dict:
        """Check the itinerary you have written against the trip's constraints.

        Call this after every write to `itinerary.json`, and again after every
        repair. It reports schema errors, hard violations (which fail the plan)
        and soft violations (quality warnings). Keep going until `passed` is
        true or you are confident the spec is impossible.

        Args:
            path: Itinerary file to check. Defaults to `itinerary.json`.

        Returns:
            `{"passed", "summary", "violations", "metrics"}`.
        """
        counter["calls"] += 1
        target = run_dir / path.lstrip("/")
        if not target.exists():
            return {
                "passed": False,
                "summary": f"{path} does not exist yet — write it first.",
                "violations": [],
                "metrics": {},
            }
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return {
                "passed": False,
                "summary": f"{path} is not valid JSON: {exc}",
                "violations": [],
                "metrics": {"schema_valid": 0.0},
            }
        report = check_payload(spec, payload)
        return {
            "passed": report.passed,
            "summary": report.summary(),
            "violations": [v.to_dict() for v in report.violations],
            "metrics": report.metrics,
        }

    return check_itinerary


def make_finalize_tool(spec: TripSpec, run_dir: Path):
    """Build `finalize_itinerary` — the agent's explicit "I'm done" signal.

    Without a tool like this there is no single moment to hang human review on:
    the agent just stops writing files and the turn ends. Making the agent
    announce completion gives the interrupt somewhere to fire, and gives the
    person a last look before the plan is treated as final.

    It re-checks rather than trusting the agent's claim — the summary the human
    sees is generated here, not by the model.
    """

    def finalize_itinerary(summary: str) -> dict:
        """Declare the itinerary finished and submit it for review.

        Call this once, as your last action, after `check_itinerary` passes (or
        after you've concluded the trip is infeasible and recorded that in
        `itinerary.json`). Do not call it on a plan you know still has hard
        violations.

        A human may review your submission and send it back with changes to
        make. If they do, address their notes and call this again.

        Args:
            summary: Two or three sentences on the trip you've planned, or on
                why it can't be done. This is what the reviewer reads first.

        Returns:
            `{"accepted", "passed", "summary", "violations"}`.
        """
        target = run_dir / ITINERARY_FILE
        if not target.exists():
            return {
                "accepted": False,
                "passed": False,
                "summary": f"{ITINERARY_FILE} does not exist — write it before finalizing.",
                "violations": [],
            }
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return {
                "accepted": False,
                "passed": False,
                "summary": f"{ITINERARY_FILE} is not valid JSON: {exc}",
                "violations": [],
            }

        report = check_payload(spec, payload)
        return {
            "accepted": True,
            "passed": report.passed,
            "agent_summary": summary,
            "summary": report.summary(),
            "violations": [v.to_dict() for v in report.violations],
            "metrics": report.metrics,
        }

    return finalize_itinerary


def stage_skills(run_dir: Path) -> None:
    """Copy the skill directories into the run so the backend can see them."""
    destination = run_dir / "skills"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(SKILLS_SOURCE, destination)


def build_agent(
    spec: TripSpec,
    run_dir: Path,
    config: AgentConfig,
    counter: dict[str, int],
    checkpointer: Any | None = None,
):
    """Assemble the agent.

    `checkpointer` is what makes human-in-the-loop possible: an interrupt has
    to persist the graph mid-run so it can be resumed after a human answers.
    The CLI leaves it unset and runs straight through; the server passes one in.
    """
    research_tools = [
        web_search,
        geocode,
        estimate_travel,
        fx_convert,
        venue_rating,
        flight_search,
    ]
    tools = list(research_tools)
    if config.use_repair_loop:
        tools.append(make_check_tool(spec, run_dir, counter))
    if config.use_finalize:
        tools.append(make_finalize_tool(spec, run_dir))

    if config.use_skills:
        stage_skills(run_dir)

    subagents = []
    if config.use_subagents:
        from wayfinder.subagents import build_subagents

        subagents = build_subagents(
            tools=research_tools,
            model=resolve_model(config.subagent_model or config.model),
            single_researcher=config.single_researcher,
        )

    # Only gate on tools that actually exist, or the middleware waits forever
    # for an approval on a tool the agent can never call.
    available = {getattr(t, "__name__", getattr(t, "name", "")) for t in tools}
    interrupt_on = {
        name: True for name in config.interrupt_on if name in available or name == "task"
    }

    return create_deep_agent(
        model=resolve_model(config.model),
        system_prompt=MAIN_PROMPT,
        tools=tools,
        subagents=subagents or None,
        skills=[SKILLS_VIRTUAL_PATH] if config.use_skills else None,
        # Virtual mode anchors every path to the run directory and blocks `..`
        # and `~`, so a stray write lands inside the run rather than in the repo.
        backend=FilesystemBackend(root_dir=str(run_dir), virtual_mode=True),
        interrupt_on=interrupt_on or None,
        checkpointer=checkpointer,
    )


def user_message(spec: TripSpec) -> str:
    """The per-run half of the prompt.

    Kept separate from `MAIN_PROMPT` so the system prefix stays byte-identical
    across runs and keeps its cache.
    """
    body = yaml.safe_dump(
        spec.model_dump(mode="json", exclude={"should_refuse"}),
        sort_keys=False,
        allow_unicode=True,
    )
    return f"Plan this trip.\n\n```yaml\n{body}```"


def new_run_dir(spec: TripSpec, root: Path | None = None) -> Path:
    """Allocate a unique directory for one run.

    The timestamp alone is not enough. Evals run cases concurrently and repeat
    each one, so two repetitions of the same spec routinely start within the
    same second — which used to hand them the same directory. They then raced
    to stage skills into it, and worse, would have overwritten each other's
    `itinerary.json`, silently scoring one plan twice.

    `mkdir(exist_ok=False)` is what makes this safe rather than merely
    unlikely: the filesystem, not the clock, guarantees the winner.
    """
    base = root or REPO_ROOT / "runs"
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    for attempt in range(1000):
        suffix = "" if attempt == 0 else f"-{attempt}"
        run_dir = base / f"{spec.slug}-{stamp}{suffix}"
        try:
            (run_dir / "research").mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return run_dir
    msg = f"could not allocate a run directory under {base}"
    raise RuntimeError(msg)


def plan_trip(
    spec: TripSpec,
    config: AgentConfig | None = None,
    run_dir: Path | None = None,
) -> RunResult:
    """Run the agent end to end and write every artifact to disk."""
    config = config or AgentConfig()
    run_dir = run_dir or new_run_dir(spec)
    run_dir.mkdir(parents=True, exist_ok=True)

    counter = {"calls": 0}
    agent = build_agent(spec, run_dir, config, counter)

    (run_dir / "spec.yaml").write_text(
        yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    error: str | None = None
    messages: list[Any] = []
    try:
        state = agent.invoke(
            {"messages": [{"role": "user", "content": user_message(spec)}]},
            config={
                "recursion_limit": config.recursion_limit,
                "run_name": f"wayfinder:{spec.slug}",
                "metadata": {
                    "wayfinder_config": config.label(),
                    "destination": spec.destination,
                    **{f"cfg_{k}": v for k, v in asdict(config).items()},
                },
            },
        )
        messages = state.get("messages", [])
    except Exception as exc:  # noqa: BLE001 - the run's failure is a result, not a crash
        error = f"{type(exc).__name__}: {exc}"

    return finalise(spec, config, run_dir, counter["calls"], messages, error)


def finalise(
    spec: TripSpec,
    config: AgentConfig,
    run_dir: Path,
    check_calls: int,
    messages: list[Any],
    error: str | None,
) -> RunResult:
    """Check whatever the agent left behind and render the rest.

    A missing or malformed `itinerary.json` is reported the same way as a bad
    one — as a failing `ConstraintReport`. The evaluators then score every run
    on the same scale, instead of some runs producing a score and others
    producing an exception.
    """
    path = run_dir / ITINERARY_FILE
    payload: Any | None = None
    itinerary: Itinerary | None = None

    if not path.exists():
        report = _missing_itinerary_report(error)
    else:
        payload = load_itinerary_payload(path)
        report = check_payload(spec, payload)
        if report.metrics.get("schema_valid") == 1.0:
            itinerary = Itinerary.model_validate(payload)

    (run_dir / "constraints.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if itinerary is not None:
        (run_dir / "itinerary.md").write_text(
            render_markdown(spec, itinerary, report), encoding="utf-8"
        )
        (run_dir / "sources.md").write_text(render_sources(itinerary), encoding="utf-8")

    return RunResult(
        run_dir=run_dir,
        spec=spec,
        config=config,
        report=report,
        itinerary=itinerary,
        payload=payload,
        error=error,
        check_calls=check_calls,
        messages=messages,
    )


def _missing_itinerary_report(error: str | None) -> ConstraintReport:
    from wayfinder.verify import CheckResult, Violation

    message = f"the agent never wrote {ITINERARY_FILE}"
    if error:
        message += f" (run failed: {error})"
    violation = Violation("schema_valid", "hard", message)
    return ConstraintReport(
        passed=False,
        violations=[violation],
        checks=[CheckResult("schema_valid", "hard", violations=[violation])],
        metrics={
            "schema_valid": 0.0,
            "hard_pass_rate": 0.0,
            "soft_pass_rate": 0.0,
            "hard_violation_count": 1.0,
        },
    )

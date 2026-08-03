"""Tests for the plumbing between the agent and its artifacts.

The model↔tool loop itself needs credentials, so it isn't covered here. What is
covered is everything around it — the parts that silently produce a wrong
*score* rather than an obvious error, which is the failure mode that would
quietly corrupt an experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import FIXTURES
from deepagents.backends import FilesystemBackend
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from wayfinder.agent import (
    AgentConfig,
    build_agent,
    finalise,
    make_check_tool,
    stage_skills,
    user_message,
)
from wayfinder.render import render_markdown, render_sources
from wayfinder.schema import Itinerary
from wayfinder.specs import load_itinerary_payload
from wayfinder.verify import check_payload


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "run"
    (d / "research").mkdir(parents=True)
    return d


def fake_model() -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter(["done"]))


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def test_graph_compiles_with_the_full_harness(spec, run_dir):
    config = AgentConfig(model=fake_model(), use_subagents=True, use_skills=True)
    agent = build_agent(spec, run_dir, config, {"calls": 0})
    nodes = set(agent.get_graph().nodes)
    assert "model" in nodes
    assert any("Skills" in n for n in nodes), "skills middleware should be wired in"


def test_graph_compiles_with_everything_off(spec, run_dir):
    config = AgentConfig(
        model=fake_model(), use_subagents=False, use_skills=False, use_repair_loop=False
    )
    agent = build_agent(spec, run_dir, config, {"calls": 0})
    nodes = set(agent.get_graph().nodes)
    assert not any("Skills" in n for n in nodes)


def test_skills_are_staged_into_the_run(spec, run_dir):
    """Skills resolve through the backend, which is rooted at the run dir.

    Referencing them at their repo path would silently load nothing — the agent
    would run without its reference material and no one would be told.
    """
    build_agent(spec, run_dir, AgentConfig(model=fake_model(), use_skills=True), {"calls": 0})
    staged = sorted(p.name for p in (run_dir / "skills").iterdir())
    assert staged == ["budget-discipline", "itinerary-format", "transit-realism"]
    for skill in staged:
        text = (run_dir / "skills" / skill / "SKILL.md").read_text(encoding="utf-8")
        assert text.startswith("---"), f"{skill} needs YAML frontmatter to be discoverable"
        assert f"name: {skill}" in text


def test_config_labels_are_distinct():
    labels = {
        AgentConfig().label(),
        AgentConfig(use_repair_loop=False).label(),
        AgentConfig(use_skills=False).label(),
        AgentConfig(use_subagents=False).label(),
        AgentConfig(single_researcher=True).label(),
        AgentConfig(subagent_model="anthropic:claude-haiku-4-5").label(),
    }
    assert len(labels) == 6, "each experiment arm needs its own label"


def test_spec_goes_in_the_user_turn(spec):
    """Keeping per-run content out of the system prompt is what makes the
    cached prefix stable across runs."""
    message = user_message(spec)
    assert "Lisbon" in message
    assert "should_refuse" not in message, "the dataset flag must not leak to the agent"


# --------------------------------------------------------------------------
# The filesystem guardrail
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["../escape.json", "/../escape.json", "../../escape.json"])
def test_traversal_is_rejected(run_dir, path):
    backend = FilesystemBackend(root_dir=str(run_dir), virtual_mode=True)
    with pytest.raises(ValueError, match="traversal"):
        backend.write(path, "nope")


@pytest.mark.parametrize("path", ["/etc/passwd", "~/.ssh/config", "/tmp/wayfinder-probe"])
def test_absolute_paths_are_contained_not_followed(run_dir, path):
    """An absolute-looking path is treated as virtual and lands inside the run.

    Testing the mechanism directly beats trying to bait the agent into
    escaping: this asserts the guarantee holds for every path, not just the one
    a particular run happened to try.
    """
    backend = FilesystemBackend(root_dir=str(run_dir), virtual_mode=True)
    backend.write(path, "contained")
    assert not (run_dir.parent / Path(path).name).exists()
    written = [p for p in run_dir.rglob("*") if p.is_file()]
    assert written, "the write should have landed somewhere under the run dir"
    assert all(run_dir in p.parents for p in written)


# --------------------------------------------------------------------------
# The check tool
# --------------------------------------------------------------------------


def test_check_tool_reports_a_missing_file_without_raising(spec, run_dir):
    counter = {"calls": 0}
    result = make_check_tool(spec, run_dir, counter)()
    assert result["passed"] is False
    assert "does not exist" in result["summary"]
    assert counter["calls"] == 1


def test_check_tool_reports_malformed_json(spec, run_dir):
    (run_dir / "itinerary.json").write_text("{not json", encoding="utf-8")
    result = make_check_tool(spec, run_dir, {"calls": 0})()
    assert result["passed"] is False
    assert result["metrics"]["schema_valid"] == 0.0


def test_check_tool_agrees_with_the_evaluators(spec, run_dir):
    """The agent and the grader must be reading the same function.

    If these ever diverge, the agent optimises against one bar and gets scored
    against another — and every experiment after that is measuring noise.
    """
    payload = load_itinerary_payload(FIXTURES / "clean.itinerary.json")
    (run_dir / "itinerary.json").write_text(json.dumps(payload), encoding="utf-8")

    from_tool = make_check_tool(spec, run_dir, {"calls": 0})()
    from_evaluator = check_payload(spec, payload)

    assert from_tool["passed"] == from_evaluator.passed
    assert from_tool["metrics"] == from_evaluator.metrics


# --------------------------------------------------------------------------
# finalise() — every run must produce a score, including the broken ones
# --------------------------------------------------------------------------


def test_finalise_on_a_good_run_writes_every_artifact(spec, run_dir):
    payload = load_itinerary_payload(FIXTURES / "clean.itinerary.json")
    (run_dir / "itinerary.json").write_text(json.dumps(payload), encoding="utf-8")

    result = finalise(spec, AgentConfig(), run_dir, check_calls=2, messages=[], error=None)

    assert result.report.passed
    assert result.itinerary is not None
    for name in ("constraints.json", "itinerary.md", "sources.md"):
        assert (run_dir / name).exists(), f"{name} was not written"
    assert "Time Out Market" in (run_dir / "itinerary.md").read_text(encoding="utf-8")


def test_finalise_when_the_agent_wrote_nothing(spec, run_dir):
    result = finalise(
        spec, AgentConfig(), run_dir, check_calls=0, messages=[], error="Timeout: boom"
    )
    assert not result.report.passed
    assert result.report.metrics["schema_valid"] == 0.0
    assert "never wrote" in result.report.violations[0].message
    assert "Timeout" in result.report.violations[0].message
    assert (run_dir / "constraints.json").exists(), "a failed run still needs a score"


def test_finalise_on_malformed_output_still_scores(spec, run_dir):
    (run_dir / "itinerary.json").write_text('{"destination": "Lisbon"}', encoding="utf-8")
    result = finalise(spec, AgentConfig(), run_dir, check_calls=1, messages=[], error=None)
    assert result.itinerary is None
    assert result.report.metrics["schema_valid"] == 0.0
    assert not (run_dir / "itinerary.md").exists()


def test_finalise_records_a_refusal_as_a_pass(spec, run_dir):
    (run_dir / "itinerary.json").write_text(
        json.dumps(
            {
                "destination": "Lisbon, Portugal",
                "currency": "EUR",
                "days": [],
                "feasible": False,
                "infeasibility_reason": "The budget covers one night of the four requested.",
            }
        ),
        encoding="utf-8",
    )
    result = finalise(spec, AgentConfig(), run_dir, check_calls=1, messages=[], error=None)
    assert result.report.passed
    assert result.report.metrics["refused"] == 1.0
    assert "doesn't fit" in (run_dir / "itinerary.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def test_markdown_carries_the_check_result(spec):
    payload = load_itinerary_payload(FIXTURES / "dirty.itinerary.json")
    report = check_payload(spec, payload)
    markdown = render_markdown(spec, Itinerary.model_validate(payload), report)
    assert "## Checks" in markdown
    assert "hard" in markdown
    assert "Nothing here is booked" in markdown


def test_sources_are_deduplicated():
    itin = Itinerary.model_validate(
        {
            "destination": "Lisbon",
            "currency": "EUR",
            "days": [
                {
                    "date": "2026-10-12",
                    "items": [
                        {
                            "start": "10:00",
                            "end": "11:00",
                            "kind": "activity",
                            "title": "A",
                            "sources": ["https://example.com/x"],
                        },
                        {
                            "start": "12:00",
                            "end": "13:00",
                            "kind": "activity",
                            "title": "B",
                            "sources": ["https://example.com/x"],
                        },
                    ],
                }
            ],
        }
    )
    out = render_sources(itin)
    assert out.count("https://example.com/x") == 1
    assert "A, B" in out


def test_stage_skills_is_idempotent(run_dir):
    stage_skills(run_dir)
    (run_dir / "skills" / "stale-leftover").mkdir()
    stage_skills(run_dir)
    assert not (run_dir / "skills" / "stale-leftover").exists()

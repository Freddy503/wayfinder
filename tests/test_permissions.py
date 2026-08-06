"""Filesystem permissions — the piece of the original plan never built.

`virtual_mode` already confines every path to the run directory, so this is not
about escaping to the repo. It is about `skills/`, which is copied into each run
precisely so the run keeps a snapshot of the rules it executed under. An agent
that rewrites its own `SKILL.md` mid-run makes that snapshot a lie — and makes
the experiment comparing `skills` against `no-skills` meaningless, because the
two arms no longer differ in one thing.
"""

from __future__ import annotations

import pytest
from deepagents.middleware.filesystem import _check_fs_permission as decide

from wayfinder.agent import WORKSPACE_PERMISSIONS


@pytest.mark.parametrize(
    ("operation", "path", "expected"),
    [
        # The whole point: readable, never writable.
        ("read", "/skills/budget-discipline/SKILL.md", "allow"),
        ("read", "/skills/transit-realism/SKILL.md", "allow"),
        ("write", "/skills/budget-discipline/SKILL.md", "deny"),
        ("write", "/skills/itinerary-format/SKILL.md", "deny"),
        ("write", "/skills/anything/at/all.md", "deny"),
        # Everything the agent actually produces stays writable.
        ("write", "/itinerary.json", "allow"),
        ("write", "/research/food.md", "allow"),
        ("write", "/todos.md", "allow"),
        ("read", "/research/logistics.md", "allow"),
    ],
)
def test_the_rules_decide_what_they_claim(operation, path, expected):
    assert decide(WORKSPACE_PERMISSIONS, operation, path) == expected


def test_reading_skills_is_never_blocked():
    """Loading them is the entire reason they are there. A deny that caught
    reads would silently turn every run into a `no-skills` run."""
    for path in ("/skills/a/SKILL.md", "/skills/b/reference.md", "/skills/"):
        assert decide(WORKSPACE_PERMISSIONS, "read", path) == "allow"


def test_the_deny_comes_before_the_catch_all():
    """First match wins, so a broad allow placed above the deny would make the
    deny unreachable — a guardrail that permits everything."""
    modes = [r.mode for r in WORKSPACE_PERMISSIONS]
    paths = [r.paths for r in WORKSPACE_PERMISSIONS]
    assert modes[-1] == "allow" and paths[-1] == ["/**"], "catch-all goes last"
    assert "deny" in modes[:-1], "the deny must sit above it"


def test_the_agent_is_actually_given_them(tmp_path, monkeypatch):
    """A rule list nobody passes to `create_deep_agent` is decoration."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    captured = {}

    import wayfinder.agent as agent_module

    real = agent_module.create_deep_agent
    monkeypatch.setattr(
        agent_module, "create_deep_agent",
        lambda **kw: captured.update(kw) or real(**kw),
    )
    from tests.conftest import make_spec

    agent_module.build_agent(
        make_spec(), tmp_path,
        agent_module.AgentConfig(use_subagents=False, use_skills=False),
        {"calls": 0},
    )
    assert captured["permissions"] is WORKSPACE_PERMISSIONS

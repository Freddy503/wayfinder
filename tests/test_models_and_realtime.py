"""Tests for provider resolution and the speech-to-speech intake.

Neither needs credentials: model resolution is string handling plus client
construction, and the realtime module's job is to shape a request and refuse
to leak a key.
"""

from __future__ import annotations

import json

import pytest

from wayfinder import realtime
from wayfinder.models import (
    OPENROUTER_BASE_URL,
    label_of,
    provider_of,
    required_key_for,
    resolve_model,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "REALTIME_MODEL", "REALTIME_VOICE"):
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------
# Model resolution
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "provider", "key"),
    [
        ("openrouter:deepseek/deepseek-v4-flash", "openrouter", "OPENROUTER_API_KEY"),
        ("anthropic:claude-sonnet-5", "anthropic", "ANTHROPIC_API_KEY"),
        ("openai:gpt-5.5", "openai", "OPENAI_API_KEY"),
        ("bare-model-name", "", None),
    ],
)
def test_provider_and_required_key(spec, provider, key):
    assert provider_of(spec) == provider
    assert required_key_for(spec) == key


def test_openrouter_spec_builds_a_client_pointed_at_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    model = resolve_model("openrouter:deepseek/deepseek-v4-flash")
    assert model.model_name == "deepseek/deepseek-v4-flash"
    assert model.openai_api_base == OPENROUTER_BASE_URL
    assert model.temperature == 0, "tool-call validity beats sampling variety here"


def test_openrouter_without_a_key_says_which_variable(monkeypatch):
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        resolve_model("openrouter:deepseek/deepseek-v4-flash")


def test_other_providers_pass_through_untouched():
    """LangChain already understands these; building a client here would
    duplicate its logic and diverge from it."""
    assert resolve_model("anthropic:claude-sonnet-5") == "anthropic:claude-sonnet-5"


def test_model_instances_pass_through():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

    fake = GenericFakeChatModel(messages=iter(["hi"]))
    assert resolve_model(fake) is fake


@pytest.mark.parametrize(
    ("spec", "label"),
    [
        ("openrouter:deepseek/deepseek-v4-flash", "deepseek-v4-flash"),
        ("anthropic:claude-sonnet-5", "claude-sonnet-5"),
    ],
)
def test_labels_stay_short_for_experiment_names(spec, label):
    assert label_of(spec) == label


def test_agent_config_defaults_to_deepseek_via_openrouter():
    from wayfinder.agent import AgentConfig

    config = AgentConfig()
    assert config.model == "openrouter:deepseek/deepseek-v4-flash"
    assert config.label().startswith("deepseek-v4-flash")


def test_every_matrix_arm_still_differs_in_exactly_one_way():
    from dataclasses import asdict

    from wayfinder.evals.run import BASELINE, EXPERIMENT_MATRIX

    base = asdict(BASELINE)
    for name, config in EXPERIMENT_MATRIX.items():
        if name == "baseline":
            continue
        diffs = {k for k, v in asdict(config).items() if base[k] != v}
        assert len(diffs) == 1, f"{name} varies {sorted(diffs)}"


# --------------------------------------------------------------------------
# Realtime intake
# --------------------------------------------------------------------------


def test_unconfigured_without_a_key():
    assert realtime.is_configured() is False
    result = realtime.mint_client_secret("2026-08-03")
    assert result["ok"] is False
    assert "OPENAI_API_KEY" in result["reason"]


def test_placeholder_key_is_not_configured(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "PASTE_HERE")
    assert realtime.is_configured() is False


def test_session_names_the_model_and_carries_todays_date(monkeypatch):
    config = realtime.session_config("2026-08-03")
    assert config["model"] == realtime.DEFAULT_MODEL
    assert "2026-08-03" in config["instructions"]

    # The whole point of the indirection: one env var switches models when
    # GPT-Live opens up in the API.
    monkeypatch.setenv("REALTIME_MODEL", "gpt-live-1")
    assert realtime.session_config("2026-08-03")["model"] == "gpt-live-1"


def test_session_exposes_exactly_one_tool_matching_the_form():
    config = realtime.session_config("2026-08-03")
    assert [t["name"] for t in config["tools"]] == ["record_requirements"]
    props = config["tools"][0]["parameters"]["properties"]
    for field in ("destination", "start", "end", "budget", "currency", "ready"):
        assert field in props


def test_instructions_state_the_three_essentials():
    text = realtime.INTERVIEWER_INSTRUCTIONS.lower()
    assert "budget" in text and "where" in text
    assert "one question at a time" in text
    assert "later answer wins" in text, "self-correction has to survive speech"


def test_the_api_key_never_appears_in_what_the_browser_receives(monkeypatch):
    """A leaked ephemeral token expires; a leaked API key does not."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-do-not-leak")
    seen = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self): ...

        def json(self):
            return {"value": "ek_ephemeral_abc", "expires_at": 123}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen["headers"] = headers
        seen["body"] = json
        return FakeResponse()

    monkeypatch.setattr("httpx.post", fake_post)
    result = realtime.mint_client_secret("2026-08-03")

    assert result["ok"] and result["client_secret"] == "ek_ephemeral_abc"
    assert "sk-secret-do-not-leak" not in json.dumps(result)
    assert seen["headers"]["authorization"].endswith("sk-secret-do-not-leak")
    assert "sk-secret-do-not-leak" not in json.dumps(seen["body"])


def test_legacy_client_secret_shape_is_accepted(monkeypatch):
    """An earlier preview nested the token; a rename shouldn't break intake."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

    class FakeResponse:
        status_code = 200

        def raise_for_status(self): ...

        def json(self):
            return {"client_secret": {"value": "ek_nested"}}

    monkeypatch.setattr("httpx.post", lambda *a, **kw: FakeResponse())
    assert realtime.mint_client_secret("2026-08-03")["client_secret"] == "ek_nested"


def test_upstream_failure_is_a_ui_state_not_an_exception(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

    def boom(*a, **kw):
        raise httpx_error()

    def httpx_error():
        import httpx

        return httpx.ConnectTimeout("timed out")

    monkeypatch.setattr("httpx.post", boom)
    result = realtime.mint_client_secret("2026-08-03")
    assert result["ok"] is False
    assert "ConnectTimeout" in result["reason"]


def test_missing_secret_in_response_is_reported(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

    class FakeResponse:
        status_code = 200

        def raise_for_status(self): ...

        def json(self):
            return {"unexpected": True}

    monkeypatch.setattr("httpx.post", lambda *a, **kw: FakeResponse())
    assert realtime.mint_client_secret("2026-08-03")["ok"] is False


@pytest.mark.parametrize(
    ("fields", "missing"),
    [
        ({}, ["destination", "dates", "budget"]),
        ({"destination": "Lisbon"}, ["dates", "budget"]),
        ({"destination": "L", "start": "2026-10-12", "budget": 600}, ["dates"]),
        ({"destination": "L", "start": "a", "end": "b", "budget": 600}, []),
    ],
)
def test_essentials_gate_matches_the_typed_path(fields, missing):
    assert realtime.missing_essentials(fields) == missing


def test_token_endpoint_is_wired(monkeypatch):
    from fastapi.testclient import TestClient

    from wayfinder.server import create_app

    response = TestClient(create_app()).post("/api/realtime/token")
    assert response.status_code == 200
    assert response.json()["ok"] is False, "no key configured in tests"

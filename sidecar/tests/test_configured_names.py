"""The texts a model reads name the instance's configured owner and assistant, never a fixed person."""
from datetime import datetime, timezone
import json
import re
from types import SimpleNamespace

import pytest

from colonymind.contacts.continuity import recent_elsewhere
from test_grounded_ideas import env  # noqa: F401  (fixture)
from test_grounded_ideas_commitments import add_open, broken_first_pass, commitments, debounced_refresh  # noqa: F401

NOW = datetime(2026, 10, 17, 14, 14, tzinfo=timezone.utc)
ROWS = [
    {"ts": "2026-10-17T14:12:00+00:00", "channel": "whatsapp:+10000000000", "direction": "in",
     "summary": "Is the report ready?"},
    {"ts": "2026-10-17T14:13:00+00:00", "channel": "whatsapp:+10000000000", "direction": "out",
     "summary": "It is ready."},
]


def _configure(monkeypatch, **names):
    for name, value in names.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


@pytest.mark.parametrize("owner,assistant,labels", [
    ("Robin", "Nova", ("Robin", "Nova")),
    (None, None, ("the owner", "the assistant")),
])
def test_the_speaker_labels_are_the_configured_owner_and_assistant(monkeypatch, owner, assistant, labels):
    _configure(monkeypatch, COLONYMIND_OWNER_NAME=owner, COLONYMIND_PERSONA_NAME=assistant)
    assert recent_elsewhere(ROWS, "telegram:1", "UTC", now=NOW).splitlines() == [
        f"WhatsApp, Sat Oct 17 2:12 PM: {labels[0]}: Is the report ready?",
        f"WhatsApp, Sat Oct 17 2:13 PM: {labels[1]}: It is ready.",
    ]


def test_the_speaker_labels_follow_a_name_configured_after_import(monkeypatch):
    _configure(monkeypatch, COLONYMIND_OWNER_NAME="Robin", COLONYMIND_PERSONA_NAME="Nova")
    first = recent_elsewhere(ROWS, "telegram:1", "UTC", now=NOW)
    _configure(monkeypatch, COLONYMIND_OWNER_NAME="Sasha", COLONYMIND_PERSONA_NAME="Iris")
    second = recent_elsewhere(ROWS, "telegram:1", "UTC", now=NOW)
    assert ": Robin: " in first and ": Nova: " in first
    assert second == first.replace(": Robin: ", ": Sasha: ").replace(": Nova: ", ": Iris: ")


@pytest.mark.asyncio
@pytest.mark.parametrize("configured,name", [("Nova", "Nova"), ("Iris", "Iris"), (None, "the assistant")])
async def test_each_prompt_with_the_copy_rules_reaches_the_model_with_the_configured_name(
        env, commitments, monkeypatch, configured, name):
    """The offers prompt and the copy-only retry prompt, as the model receives them."""
    _configure(monkeypatch, COLONYMIND_PERSONA_NAME=configured)
    add_open(env, commitments)
    broken_first_pass(env)                      # evidence that verifies, copy that is refused: the retry follows
    first_pass, prompts = env.model.complete, []

    async def complete(messages, **kwargs):
        prompts.append(messages[0]["content"])
        if len(prompts) == 1:
            return await first_pass(messages, **kwargs)
        return SimpleNamespace(content=json.dumps({"copies": []}), usage={}, cost_usd=0.0, model_id="test")

    env.model.complete = complete
    await debounced_refresh(env)
    offers, *retries = prompts
    assert offers.startswith("Select up to six useful unresolved personal work offers")
    assert retries and all(retry.startswith("Rewrite the display copy for these verified offers") for retry in retries)
    for prompt in prompts:
        assert f"Write clean, specific cards in {name}'s natural first-person voice:\n" in prompt
        assert f"one or two short sentences in {name}'s voice, about 160 characters.\n" in prompt
        assert re.search(r"\{\w*\}", prompt) is None      # no placeholder is left for the model to read
        assert prompt.count(f"{name}'s") == 2

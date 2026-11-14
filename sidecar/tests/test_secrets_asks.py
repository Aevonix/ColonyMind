"""Using a sealed value in a call: bound to the complete call, on the owner's own words, shown to them in full.

A fill without an ask needs the reference sealed from the owner's own words just now (the plugin names them, and
the vault must have seen the value sealed recently). Any other fill is a one-tap ask whose permit is bound to the
digest of the complete action (its tool and every argument, never a prefix), and whose words show where the value
goes in full. A value that cannot be read now is answered as unavailable (the caller waits), and the gateway's
seal answer names the spans a released value keeps.

secret-scan: synthetic fixtures (every value below is made up and assembled at run time)
"""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from colonymind.secrets.vault import Vault, VaultUnavailable, reset_vaults
from test_mind_loop import OWNER, Fixture, pinned_clock  # noqa: F401  (pinned_clock: autouse pytest fixture)

KEY = "sk-" + "proj" + "Qz7Wx3Er9Ty1Ui5Op2As8Df4"


@pytest.fixture
def app(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from colonymind.api.routers import secrets as secrets_router
    reset_vaults()
    fx = Fixture(tmp_path)
    vault = Vault(tmp_path)
    monkeypatch.setattr(secrets_router, "_vault", vault)
    monkeypatch.setattr(secrets_router, "_owner_id", lambda: OWNER)
    monkeypatch.setattr(secrets_router, "_mind", lambda: fx.mind)
    application = FastAPI()
    application.include_router(secrets_router.router)
    yield application, vault, fx
    fx.store.close()
    reset_vaults()


async def _post(app, path, body):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return (await client.post(path, json=body)).json()


def _fill(ref, action, **extra):
    return {"refs": [ref], "purpose": "fill", "lane": "owner", "action": action,
            "destination": "run the command: " + action["args"]["command"], **extra}


@pytest.mark.asyncio
async def test_a_permit_is_bound_to_the_complete_action_and_the_ask_shows_it_in_full(app):
    application, vault, fx = app
    ref = vault.seal(KEY, contact_id=OWNER, kind="prefix", label="API key")
    prefix = "echo " + "x" * 2100 + " | "
    to_a = {"tool": "terminal", "args": {"command": prefix + f"curl -d @- https://a.vendor.io [secret {ref}: API key]"}}
    to_b = {"tool": "terminal", "args": {"command": prefix + f"curl -d @- https://b.vendor.io [secret {ref}: API key]"}}
    ask_a = (await _post(application, "/v1/secrets/resolve", _fill(ref, to_a)))["asks"][ref]
    ask_b = (await _post(application, "/v1/secrets/resolve", _fill(ref, to_b)))["asks"][ref]
    assert ask_a["code"] != ask_b["code"]                       # two calls, two asks: no shared prefix digest
    assert "https://a.vendor.io" in ask_a["text"] and "https://b.vendor.io" in ask_b["text"]
    row = next(r for r in fx.store.intentions(status=["asked"], kind=["note"], limit=10) if r.ask_code == ask_a["code"])
    assert row.context["destination"].endswith(f"https://a.vendor.io [secret {ref}: API key]")
    assert json.loads(row.context["action"]) == to_a            # the complete action, kept for review
    fx.store.transition(row.id, "approved", action="approved", at=fx.now)
    other = await _post(application, "/v1/secrets/resolve", _fill(ref, to_b))
    assert other["values"] == {} and ref in other["asks"]       # the yes for one call never lets another through
    once = await _post(application, "/v1/secrets/resolve", _fill(ref, to_a))
    assert once["values"] == {ref: KEY}
    again = await _post(application, "/v1/secrets/resolve", _fill(ref, to_a))
    assert again["values"] == {} and ref in again["asks"]       # a permit is used once


@pytest.mark.asyncio
async def test_only_the_owners_own_fresh_words_fill_without_an_ask(app, monkeypatch):
    application, vault, fx = app
    ref = vault.seal(KEY, contact_id=OWNER, kind="prefix", label="API key")
    action = {"tool": "terminal", "args": {"command": f"deploy --key [secret {ref}: API key]"}}
    named = await _post(application, "/v1/secrets/resolve", _fill(ref, action, own_refs=[ref]))
    assert named["values"] == {ref: KEY}
    # Text that merely mentions the reference (a quote, a pasted line) is not their words: the plugin does not name it.
    mentioned = await _post(application, "/v1/secrets/resolve", _fill(ref, action))
    assert mentioned["values"] == {} and ref in mentioned["asks"]
    # Named, but the vault has not seen it sealed from a message for over half an hour: asked.
    from datetime import timedelta
    import time as clock
    real = clock.time
    monkeypatch.setattr(clock, "time", lambda: real() + timedelta(hours=1).total_seconds())
    stale = await _post(application, "/v1/secrets/resolve", _fill(ref, action, own_refs=[ref]))
    assert stale["values"] == {} and ref in stale["asks"]
    missing = await _post(application, "/v1/secrets/resolve", {"refs": [ref], "purpose": "fill", "lane": "owner",
                                                               "own_refs": [ref]})
    assert missing["values"] == {} and ref in missing["refused"]   # a fill names its complete action


@pytest.mark.asyncio
async def test_a_value_that_cannot_be_read_now_is_unavailable_not_refused(app, monkeypatch):
    application, vault, fx = app
    ref = vault.seal(KEY, contact_id=OWNER, kind="prefix", label="API key")

    def locked(*_args, **_kwargs):
        raise VaultUnavailable("the key is locked")
    monkeypatch.setattr(vault, "resolve", locked)
    answer = await _post(application, "/v1/secrets/resolve", {"refs": [ref], "purpose": "use", "lane": "owner"})
    assert answer["values"] == {} and ref in answer["unavailable"] and ref not in answer["refused"]


@pytest.mark.asyncio
async def test_the_seal_answer_names_where_a_released_value_stands(app):
    application, vault, fx = app
    ref = vault.seal(KEY, contact_id=OWNER, kind="prefix", label="API key")
    vault.release(ref)
    text = f"the build id is {KEY} ok"
    answer = await _post(application, "/v1/secrets/seal", {"text": text, "owner": True})
    assert answer["text"] == text and answer["sealed"] == 0
    start = text.index(KEY)
    assert answer["released"] == [[start, start + len(KEY)]]
    fresh = await _post(application, "/v1/secrets/seal", {"text": "the wifi password is Zq8Zq8Zq8!x", "owner": True})
    assert fresh["released"] == [] and fresh["sealed"] == 1

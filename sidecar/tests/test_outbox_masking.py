"""What leaves the outbox: a message the mind composed hides every secret for anyone but the owner; the owner's own
dictated words go out as they gave them; a message to the owner is never hidden (docs/SECRETS.md, section 7).

secret-scan: synthetic fixtures (every value below is made up and assembled at run time)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from colonymind.mind import audit
from colonymind.mind.outbox import message_payload
from colonymind.secrets.detect import HIDDEN
from colonymind.secrets.refs import reference_text
from colonymind.secrets.vault import Vault
from test_mind_loop import CONTACT, OWNER, fx, pinned_clock  # noqa: F401  (pytest fixtures)
from owner_answer import owner_answers

TOKEN = "sk-" + "abcdefghijklmnopqrstuvwxyz0123456789"
WIFI = "Hunter" + "22!" + "wifi"


def row(text, recipient, *, type="check_in", status="approved", **context):
    return SimpleNamespace(id="i-1", kind="message", type=type, status=status, entity_id=recipient,
                           context={"text": text, **context}, description="", created_at=None, expires_at=None,
                           ask_code=None)


def test_a_message_sent_without_an_ask_hides_its_secret():
    """The probe restated: an approved check-in to a contact on auto leaves with the key hidden."""
    sent = message_payload(row(f"the key is {TOKEN}", CONTACT), owner_id=OWNER)["text"]
    assert TOKEN not in sent and HIDDEN in sent


@pytest.mark.parametrize("words", [f"the key is {TOKEN}", f"the wifi password is {WIFI}",
                                   "the code is " + reference_text("S-ABCDEFGH", "API key")])
def test_what_the_mind_composed_for_a_contact_hides_values_and_references(words):
    sent = message_payload(row(words, CONTACT), owner_id=OWNER)["text"]
    assert TOKEN not in sent and WIFI not in sent and "S-ABCDEFGH" not in sent and HIDDEN in sent


def test_a_message_to_the_owner_is_never_hidden():
    words = f"your key is {TOKEN} and " + reference_text("S-ABCDEFGH", "API key")
    assert message_payload(row(words, OWNER), owner_id=OWNER)["text"] == words


@pytest.mark.parametrize("words", [f"Tell them the door code is {WIFI}",
                                   "Their key: " + reference_text("S-ABCDEFGH", "API key")])
@pytest.mark.parametrize("context", [{"grant": "owner"}, {"ask_owner": True}], ids=["granted", "recipient_check"])
def test_the_owners_dictated_words_go_out_as_he_gave_them(words, context):
    notice = row(words, CONTACT, type="commitment_notice", **context)
    assert message_payload(notice, owner_id=OWNER)["text"] == words
    assert audit.frozen_for_ask(notice, owner_id=OWNER) is None     # what they approve is what they said


def test_only_a_notice_of_his_confirmed_request_is_his_dictation():
    for type, context in (("commitment_notice", {}), ("check_in", {"grant": "owner"}),
                          ("commitment_check_in", {"grant": "owner"})):
        sent = message_payload(row(f"key {TOKEN}", CONTACT, type=type, **context), owner_id=OWNER)["text"]
        assert TOKEN not in sent, (type, context)


def test_a_caller_that_forgets_the_owner_can_only_hide():
    assert TOKEN not in audit.outgoing_text(row(f"key {TOKEN}", OWNER))
    assert TOKEN not in audit.outgoing_text(row(f"key {TOKEN}", CONTACT))


def test_the_ask_fixes_the_words_that_leave():
    """Asked words equal sent words, secrets hidden in both, for a composed message to a contact."""
    words = f"Here is the key {TOKEN}"
    asked = row(words, CONTACT, status="asked")
    frozen = audit.frozen_for_ask(asked, owner_id=OWNER)
    assert frozen["hidden"] is True and TOKEN not in frozen["text"]
    approved = row(frozen["text"], CONTACT, hidden=True)
    assert message_payload(approved, owner_id=OWNER)["text"] == frozen["text"]


# -- the body's resolve at the send: the sidecar decides which reference may be filled ---------------------------

@pytest.fixture
def send_app(fx, monkeypatch):
    from fastapi import FastAPI
    from colonymind.api.routers import secrets as secrets_router
    vault = Vault(fx.state)
    monkeypatch.setattr(secrets_router, "_vault", vault)
    monkeypatch.setattr(secrets_router, "_owner_id", lambda: OWNER)
    monkeypatch.setattr(secrets_router, "_mind", lambda: fx.mind)
    app = FastAPI()
    app.include_router(secrets_router.router)
    return app, vault


def _message(fx, text, recipient, *, type, **context):
    created, _ = fx.store.create_intention(
        kind="message", type=type, title="a message", drive="duty", cls="contact", decision="act",
        decision_reason="test", status="approved", dedup_key=None, recipient=recipient,
        context={"text": text, **context}, created_at=fx.now)
    return created


async def _resolve(app, ref, intention_id):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/secrets/resolve", json={"refs": [ref], "purpose": "send",
                                                                    "intention_id": intention_id})
    assert response.status_code == 200, response.text
    return response.json()


async def test_the_send_fills_only_the_owners_reference_in_his_own_words_or_to_him(fx, send_app):
    app, vault = send_app
    mine = vault.seal(TOKEN, contact_id=OWNER, kind="prefix", label="API key")
    theirs = vault.seal(WIFI, contact_id="p-03", kind="labeled", label="wifi password")
    mine_text, theirs_text = reference_text(mine, "API key"), reference_text(theirs, "wifi password")
    dictated = _message(fx, f"Their key: {mine_text}", CONTACT, type="commitment_notice", grant="owner")
    composed = _message(fx, f"The key is {mine_text}", CONTACT, type="check_in")
    to_owner = _message(fx, f"Your key: {mine_text}", OWNER, type="task_outcome")
    relayed = _message(fx, f"Their wifi: {theirs_text}", CONTACT, type="commitment_notice", grant="owner")
    assert (await _resolve(app, mine, dictated.id))["values"] == {mine: TOKEN}
    assert (await _resolve(app, mine, to_owner.id))["values"] == {mine: TOKEN}
    assert (await _resolve(app, mine, composed.id))["values"] == {}
    assert (await _resolve(app, theirs, relayed.id))["values"] == {}          # a contact's value, never
    assert (await _resolve(app, mine, relayed.id))["values"] == {}            # not in this message's words
    assert (await _resolve(app, mine, "no-such-message"))["values"] == {}


# -- a fill the owner did not name this turn: one tap, then exactly that call once -------------------------------

async def _fill(app, ref, *, words="the key: {ref}", target="telegram:p-02-handle"):
    """A fill of one complete call (its tool and every argument), as the plugin asks for it."""
    action = {"tool": "send_message", "args": {"target": target, "message": words.format(ref=f"[secret {ref}: API key]")}}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/secrets/resolve", json={
            "refs": [ref], "purpose": "fill", "lane": "owner", "contact_id": OWNER, "action": action,
            "destination": target, "tool": "send_message"})
    assert response.status_code == 200, response.text
    return response.json()


async def test_an_older_reference_asks_once_and_the_yes_fills_that_call_once(fx, send_app):
    app, vault = send_app
    ref = vault.seal(TOKEN, contact_id=OWNER, kind="prefix", label="API key")
    first = await _fill(app, ref)
    code = first["asks"][ref]["code"]
    assert first["values"] == {} and f"yes {code}" in first["asks"][ref]["text"]
    assert "API key" in first["asks"][ref]["text"] and TOKEN not in str(first)
    assert "telegram:p-02-handle" in first["asks"][ref]["text"]            # where it goes, in the owner's words
    again = await _fill(app, ref)
    assert again["asks"][ref]["code"] == code                              # one open ask, reused
    elsewhere = await _fill(app, ref, target="telegram:p-03-handle")
    assert elsewhere["asks"][ref]["code"] != code                          # another target is another ask
    reworded = await _fill(app, ref, words="the key is {ref}, and more")
    assert reworded["asks"][ref]["code"] != code                           # another call is another ask
    await owner_answers(fx.mind, code)
    assert (await _fill(app, ref, words="the key is {ref}, and more"))["values"] == {}   # not the call they allowed
    assert (await _fill(app, ref))["values"] == {ref: TOKEN}
    assert (await _fill(app, ref))["values"] == {}           # once
    asked, = [row for row in fx.store.intentions(kind=["note"], limit=20) if row.ask_code == code]
    assert asked.type == "secret_use" and asked.status == "done"

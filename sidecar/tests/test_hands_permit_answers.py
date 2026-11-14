"""Hands permit asks read and answered by their id from an owner channel (docs/hands-permit-answer-contract.md).

Real code in isolated stores: a hand's hold forms the ask (``Approvals.hold``), the RCS copy is answered the way the
plugin hands an owner message to the sidecar (``/v1/mind/answers`` or ``Mind.read_answer``), and a companion app
answers through ``POST /v1/mind/permit-asks/{id}/answer`` with its relay's own ``answer`` credential. Both settle
one row through one guarded settle path: the first commit wins and the second gets the recorded winner.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import threading
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from onekey import AUTH, KEY
from colonymind.api.auth import credential_authority, key_authority
from colonymind.api.credentials import CredentialTable, secret_digest
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import mind as mind_routes
from colonymind.hands import approvals, sessions, wire
from colonymind.initiatives.store import InitiativeStore, SettleGuard, TransitionRefused
from colonymind.mind import permit_asks
from colonymind.mind.outcomes import owner_answer
from colonymind.mind.tick import Mind
from test_hands_permits import EPOCH, fx, query, seed  # noqa: F401
from test_mind_loop import OWNER, pinned_clock  # noqa: F401

RCS_ONLY_SECRET, GUEST_SECRET = "rcs-only-secret", "guest-secret"
HUB_SECRET, LANE_SECRET = "companion-hub-secret", "companion-lane-secret"
ORIGIN_SECRET, ORIGIN2_SECRET = "tool-origin-secret", "tool-origin-second-secret"
HUB = {"Authorization": f"Bearer {HUB_SECRET}"}
TABLE = CredentialTable.from_config({
    "google-tools": {"sha256": secret_digest(ORIGIN_SECRET), "access": ["tool-origin"]},
    "other-tools": {"sha256": secret_digest(ORIGIN2_SECRET), "access": ["tool-origin"]},
    # The instance key granted the owner on companion too (the Hermes adapter's own chat lane there).
    "api-key": {"owner": ["rcs", "companion"], "contacts": "all"},
    # The owner channel relay's own credential: it answers asks and does nothing else.
    "companion-hub": {"sha256": secret_digest(HUB_SECRET), "owner": ["companion"], "access": ["answer"]},
    # A credential granted the owner on companion with the owner's side of the API, but no answer.
    "companion-lane": {"sha256": secret_digest(LANE_SECRET), "owner": ["companion"], "contacts": "all",
                       "access": ["read", "write", "manage"]},
    "rcs-lane": {"sha256": secret_digest(RCS_ONLY_SECRET), "owner": ["rcs"], "contacts": "all",
                 "access": ["read", "write", "manage"]},
    "guest": {"sha256": secret_digest(GUEST_SECRET), "owner": [], "contacts": "all",
              "access": ["read", "write", "manage"]}})
HUB_AUTHORITY = credential_authority(TABLE.get("companion-hub"))
CHAT = "owner-rcs-chat"
_numbers = itertools.count(1)


@pytest.fixture
async def rig(fx, monkeypatch):
    """The mind router behind the real middleware (``TABLE``): ``client`` presents the instance key, ``hub`` the
    companion relay's ``answer`` credential."""
    mind, store, service, body, _ = fx

    async def resolve_messaging_handle(gateway, address):
        return SimpleNamespace(contact_id=OWNER) if (gateway, address) == ("companion", "owner") else None
    mind.contacts.resolve_messaging_handle = resolve_messaging_handle

    @asynccontextmanager
    async def opened():
        yield store
    monkeypatch.setattr(sessions, "open_store", opened)
    app = FastAPI()
    app.state.hands_store = store
    app.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=TABLE)
    app.include_router(mind_routes.router)
    mind_routes.set_mind(mind.mind)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", headers=AUTH) as client, \
            AsyncClient(transport=transport, base_url="http://test", headers=HUB) as hub:
        yield SimpleNamespace(fx=mind, mind=mind.mind, store=store, service=service, body=body, client=client,
                              hub=hub)
    mind_routes.set_mind(None)


def answer_body(choice="yes", **extra):
    return {"choice": choice, "platform": "companion", "contact_id": OWNER, **extra}


async def app_answer(rig, ask_id, choice="yes", *, key=None, headers=None, **extra):
    return await rig.hub.post(f"/v1/mind/permit-asks/{ask_id}/answer", json=answer_body(choice, **extra),
                              headers={"Idempotency-Key": key or f"key-{next(_numbers)}", **(headers or {})})


def rcs_evidence(mind, ask_id, text="yes", *, reaction=None):
    """The owner's RCS message as the plugin hands it to the sidecar: a quoted reply (or a reaction) to the ask's
    own delivered message in their direct chat."""
    number = next(_numbers)
    shown = f"rcs-ask-{number}"
    mind.record_shown([ask_id], platform="rcs", message_id=shown, chat=CHAT, at=mind.clock() - timedelta(seconds=30))
    return {"platform": "rcs", "contact_id": OWNER, "chat_id": CHAT, "chat_type": "dm",
            "message_id": f"rcs-in-{number}", "turn_id": f"rcs-turn-{number}", "arrived_at": mind.clock().isoformat(),
            "text": None if reaction else text, "reaction": reaction, "reply_to_message_id": shown}


def settling_rows(store, ask_id):
    return [item for item in store.get_history(ask_id, limit=500)
            if item.action == "queued" or item.action.startswith("outcome_")]


async def hold_other(rig, tmp, *, hand, task, name):
    """A second hand's hold on a different call (its own ask)."""
    await seed(rig.store, hand=hand, task=task)
    target = tmp / name
    target.write_text(name)
    body = wire.HoldRequest(epoch=EPOCH, call={"tool": "terminal", "arguments": {"command": f"rm {target}"}},
                            workdir=str(tmp), targets=[{"kind": "file", "target": str(target),
                                                        "value": hashlib.sha256(target.read_bytes()).hexdigest()}])
    return await rig.service.hold(hand, body, f"hold-{hand}"), body


# -- read --------------------------------------------------------------------------------------------------------

async def test_listing_shows_each_open_permit_ask_with_its_full_terms_and_choices(rig):
    long_body = rig.body.model_copy(update={"call": wire.ExactCall(tool="terminal", arguments={
        "command": rig.body.call.arguments["command"] + " && echo " + "x" * 400})})
    held = await rig.service.hold("h-000000000001", long_body, "hold")
    ordinary, _ = rig.fx.store.create_intention(
        kind="task", type="research", title="Tidy the album", drive="curiosity", cls="self", decision="ask",
        decision_reason="r", status="asked", dedup_key="test:ordinary", ask_code="K7F", context={"notice": True},
        hermes_kind="none", created_at=rig.fx.now, expires_at=rig.fx.now + timedelta(days=3))
    listed = await rig.hub.get("/v1/mind/permit-asks")
    assert listed.status_code == 200, listed.text
    asks = listed.json()["asks"]
    assert [item["id"] for item in asks] == [held.ask_id]       # never the ordinary ask
    view = asks[0]
    assert view["kind"] == "hands_permit" and view["status"] == "asked" and view["open"] is True
    assert view["hand_id"] == "h-000000000001" and view["winner"] is None and view["settlement"] is None
    assert view["choices"] == [{"value": "yes", "label": "Yes", "means": "the hand may make exactly this call, once"},
                               {"value": "no", "label": "No", "means": "the hand does not make this call"}]
    terms = view["terms"]
    # The whole call, never clipped: the audit title is cut at 160 characters, the terms are not.
    assert long_body.call.arguments["command"] in terms["text"] and len(terms["text"]) > 400
    assert long_body.targets[0].value in terms["text"]
    assert terms["call"] == {"tool": "terminal", "arguments": long_body.call.arguments, "workdir": long_body.workdir,
                             "targets": [item.model_dump() for item in long_body.targets]}
    assert terms["reply"].endswith("Reply yes or no.")
    assert terms["effect_class"] == rig.fx.store.get(held.ask_id).context["effect"]["cls"]
    # The audit view clips the reason to 200 characters; the terms keep it whole.
    audit = (await rig.client.get("/v1/mind/asks")).json()["asks"]
    clipped = next(item for item in audit if item["id"] == held.ask_id)["decision_reason"]
    assert len(clipped) <= 200 < len(terms["reason"]) and long_body.call.arguments["command"] in terms["reason"]
    one = await rig.hub.get(f"/v1/mind/permit-asks/{held.ask_id}")
    assert one.status_code == 200 and one.json() == view
    for ident in (ordinary.id, "no-such-ask"):
        missing = await rig.hub.get(f"/v1/mind/permit-asks/{ident}")
        assert missing.status_code == 404 and missing.json()["detail"]["code"] == "unknown_ask"
    assert (await rig.hub.get("/v1/mind/permit-asks?state=settled")).json()["asks"] == []
    assert (await rig.hub.get("/v1/mind/permit-asks?state=bogus")).status_code == 422


async def test_reads_are_the_answer_credentials_only(rig):
    """The instance key (whatever it is granted), a credential granted the owner on companion with the owner's side
    of the API, one granted the owner elsewhere and one granted no owner all get 403 ``answer_credential_required``."""
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    for path in ("/v1/mind/permit-asks", f"/v1/mind/permit-asks/{held.ask_id}"):
        assert (await rig.hub.get(path)).status_code == 200
        for secret in (KEY, LANE_SECRET, RCS_ONLY_SECRET, GUEST_SECRET):
            refused = await rig.client.get(path, headers={"Authorization": f"Bearer {secret}"})
            assert refused.status_code == 403, (path, secret, refused.text)
            assert refused.json()["detail"]["code"] == "answer_credential_required"


# -- answer by id ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("channel", ["rcs", "companion"])
async def test_a_yes_by_id_grants_the_hand_exactly_as_an_rcs_yes(rig, channel):
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    assert (await rig.service.check(query(rig.body), "before")).decision == "hold"
    if channel == "rcs":
        answered = await rig.client.post("/v1/mind/answers", json=rcs_evidence(rig.mind, held.ask_id))
        assert answered.status_code == 200 and answered.json()["approved"] == [held.ask_id]
    else:
        answered = await app_answer(rig, held.ask_id)
        assert answered.status_code == 200, answered.text
        result = answered.json()
        assert result["result"] == "settled" and result["status"] == "approved" and result["replayed"] is False
        assert result["winner"] == {"channel": "companion", "choice": "yes", "at": rig.fx.now.isoformat(),
                                    "path": "explicit"}
    ask = rig.fx.store.get(held.ask_id)
    assert ask.status == "approved"
    settled, = settling_rows(rig.fx.store, held.ask_id)
    assert settled.action == "queued" and owner_answer(settled.details["by"], settled.details) is True
    assert settled.details["answer"]["channel"] == channel
    # The durable grant, refreshed before the response, as after an RCS answer; one use, then held again.
    permit = dict((await rig.store.db.execute_fetchall("SELECT * FROM permits"))[0])
    assert permit["state"] == "granted" and permit["owner_answer_ref"] == settled.details["answer"]["message"]
    held_now = await approvals.held_permits("owner-session")
    assert [(item.ask_id, item.state) for item in held_now.permits] == [(held.ask_id, "granted")]
    assert (await rig.service.check(query(rig.body), "run")).decision == "allow"
    assert (await rig.service.check(query(rig.body), "again")).decision == "hold"
    view = (await rig.hub.get(f"/v1/mind/permit-asks/{held.ask_id}")).json()
    assert view["open"] is False and view["status"] == "approved"
    assert view["winner"]["channel"] == channel and view["winner"]["choice"] == "yes"
    assert view["settlement"]["by"] == "owner"


async def test_a_no_by_id_refuses_the_hand(rig):
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    answered = await app_answer(rig, held.ask_id, "no")
    assert answered.status_code == 200, answered.text
    result = answered.json()
    assert (result["status"], result["outcome"], result["winner"]["choice"]) == ("cancelled", "denied", "no")
    ask = rig.fx.store.get(held.ask_id)
    assert (ask.status, ask.outcome) == ("cancelled", "denied")
    assert (await rig.store.db.execute_fetchall("SELECT state FROM permits"))[0][0] == "refused"
    assert (await rig.service.check(query(rig.body), "run")).decision == "hold"


# -- idempotency ---------------------------------------------------------------------------------------------------

async def test_the_same_key_and_body_replays_the_original_result_even_after_a_restart(rig):
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    first = await app_answer(rig, held.ask_id, key="tap-1")
    rig.fx.shift(minutes=5)
    again = await app_answer(rig, held.ask_id, key="tap-1")
    assert first.status_code == again.status_code == 200
    assert again.json() == {**first.json(), "replayed": True}
    mind_routes.set_mind(rig.fx.restart())        # a new process: the record is durable, not in memory
    later = await app_answer(rig, held.ask_id, key="tap-1")
    assert later.status_code == 200 and later.json() == {**first.json(), "replayed": True}
    reopened = InitiativeStore(state_dir=rig.fx.state)
    try:
        assert reopened.answer_request("companion-hub", "tap-1")["response"] == first.json()
    finally:
        reopened.close()
    assert len(settling_rows(rig.fx.store, held.ask_id)) == 1


async def test_the_same_key_with_another_body_is_a_conflict(rig, tmp_path):
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    other, _ = await hold_other(rig, tmp_path, hand="h-000000000002", task="t-000000000002", name="other.txt")
    assert (await app_answer(rig, held.ask_id, "yes", key="tap-1")).status_code == 200
    for ask_id, choice in ((held.ask_id, "no"), (other.ask_id, "yes")):
        conflict = await app_answer(rig, ask_id, choice, key="tap-1")
        assert conflict.status_code == 409 and conflict.json()["detail"]["code"] == "idempotency_conflict"
    assert rig.fx.store.get(held.ask_id).status == "approved"
    assert rig.fx.store.get(other.ask_id).status == "asked"
    assert len(settling_rows(rig.fx.store, held.ask_id)) == 1


async def test_a_blank_or_missing_key_is_refused(rig):
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    missing = await rig.hub.post(f"/v1/mind/permit-asks/{held.ask_id}/answer", json=answer_body())
    assert missing.status_code == 422
    blank = await app_answer(rig, held.ask_id, key="   ")
    assert blank.status_code == 422 and blank.json()["detail"]["code"] == "invalid_idempotency_key"
    assert rig.fx.store.get(held.ask_id).status == "asked"


# -- the second channel gets the winner, in both orders -----------------------------------------------------------

@pytest.mark.parametrize("how", ["reply", "reaction"])
async def test_rcs_first_then_the_app_gets_the_rcs_winner(rig, how):
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    evidence = rcs_evidence(rig.mind, held.ask_id, "yes", reaction="👍" if how == "reaction" else None)
    read = await rig.mind.read_answer(evidence)
    assert read["approved"] == [held.ask_id] and read["path"] == how
    rig.fx.shift(minutes=1)
    late = await app_answer(rig, held.ask_id, "no", key="late")
    assert late.status_code == 409, late.text
    detail = late.json()["detail"]
    assert detail["code"] == "already_answered" and detail["status"] == "approved"
    answered_at = (rig.fx.now - timedelta(minutes=1)).isoformat()
    assert detail["winner"] == {"channel": "rcs", "choice": "yes", "at": answered_at, "path": how}
    replay = await app_answer(rig, held.ask_id, "no", key="late")
    assert replay.status_code == 409 and replay.json()["detail"] == {**detail, "replayed": True}
    assert rig.fx.store.get(held.ask_id).status == "approved"
    assert len(settling_rows(rig.fx.store, held.ask_id)) == 1


async def test_app_first_then_rcs_settles_nothing_and_the_app_stays_the_winner(rig):
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    evidence = rcs_evidence(rig.mind, held.ask_id, "no")      # shown in RCS while still open
    won = await app_answer(rig, held.ask_id, "yes")
    assert won.status_code == 200
    read = await rig.mind.read_answer(evidence)
    assert read["read"] is True and read["settled"] == [] and read["approved"] == []
    assert rig.fx.store.get(held.ask_id).status == "approved"
    assert len(settling_rows(rig.fx.store, held.ask_id)) == 1
    view = (await rig.hub.get(f"/v1/mind/permit-asks/{held.ask_id}")).json()
    assert view["winner"]["channel"] == "companion" and view["winner"]["choice"] == "yes"
    second = await app_answer(rig, held.ask_id, "no")
    assert second.status_code == 409 and second.json()["detail"]["winner"]["channel"] == "companion"


# -- closed asks --------------------------------------------------------------------------------------------------

async def test_an_expired_ask_lapses_and_says_so(rig):
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    rig.fx.shift(days=4)
    view = (await rig.hub.get(f"/v1/mind/permit-asks/{held.ask_id}")).json()
    assert view["status"] == "asked" and view["open"] is False      # past expiry, before the sweep
    late = await app_answer(rig, held.ask_id, "yes")
    assert late.status_code == 409, late.text
    detail = late.json()["detail"]
    assert detail["code"] == "ask_closed" and detail["status"] == "expired" and detail["winner"] is None
    assert detail["settlement"]["by"] == "lapse"
    assert rig.fx.store.get(held.ask_id).status == "expired"
    assert (await rig.service.check(query(rig.body), "run")).decision == "hold"
    swept = await app_answer(rig, held.ask_id, "yes")
    assert swept.status_code == 409 and swept.json()["detail"]["settlement"]["by"] == "lapse"


async def test_a_withdrawn_ask_says_how_it_closed(rig):
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    await seed(rig.store, hand="h-000000000002", task="t-000000000002")
    newer = await rig.service.hold("h-000000000002", rig.body, "hold-2")     # replaces the identical held call
    assert newer.ask_id != held.ask_id and rig.fx.store.get(held.ask_id).status == "cancelled"
    late = await app_answer(rig, held.ask_id, "yes")
    assert late.status_code == 409
    detail = late.json()["detail"]
    assert detail["code"] == "ask_closed" and detail["settlement"]["by"] == "mind" and detail["winner"] is None
    assert rig.fx.store.get(newer.ask_id).status == "asked"


async def test_an_unknown_ask_or_a_choice_it_does_not_offer_is_refused(rig):
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    ordinary, _ = rig.fx.store.create_intention(
        kind="task", type="research", title="Tidy the album", drive="curiosity", cls="self", decision="ask",
        decision_reason="r", status="asked", dedup_key="test:ordinary", ask_code="K7F", context={"notice": True},
        hermes_kind="none", created_at=rig.fx.now, expires_at=rig.fx.now + timedelta(days=3))
    for ident in ("no-such-ask", ordinary.id, rig.fx.store.get(held.ask_id).ask_code):
        unknown = await app_answer(rig, ident, "yes")
        assert unknown.status_code == 404 and unknown.json()["detail"]["code"] == "unknown_ask"
    for choice in ("always", "maybe", "YES", " yes", "yes ", "no\n"):
        invalid = await app_answer(rig, held.ask_id, choice)
        assert invalid.status_code == 422 and invalid.json()["detail"]["code"] == "invalid_choice"
        assert invalid.json()["detail"]["choices"] == ["yes", "no"]
    extra = await app_answer(rig, held.ask_id, "yes", approved=True)
    assert extra.status_code == 422
    assert rig.fx.store.get(held.ask_id).status == "asked" and rig.fx.store.get(ordinary.id).status == "asked"


@pytest.mark.parametrize("choice", ["yes", "no"])
async def test_a_yes_to_an_ask_too_long_for_one_message_waits_for_its_full_text(rig, choice):
    """The same guard as an RCS yes: an ask whose terms do not fit its own message counts a yes only once its full
    text was delivered to the owner in its own message. A no is taken at once."""
    long_body = rig.body.model_copy(update={"call": wire.ExactCall(tool="terminal", arguments={
        "command": rig.body.call.arguments["command"] + " && echo " + "x" * 1600})})
    held = await rig.service.hold("h-000000000001", long_body, "hold")
    await rig.mind._deliver_asks([rig.fx.store.get(held.ask_id)], rig.mind.clock())
    terms = (await rig.hub.get(f"/v1/mind/permit-asks/{held.ask_id}")).json()["terms"]
    assert terms["full_text_required"] is True and terms["full_text_delivered"] is False
    assert long_body.call.arguments["command"] in terms["text"]
    answered = await app_answer(rig, held.ask_id, choice, key="long")
    if choice == "no":
        assert answered.status_code == 200 and answered.json()["status"] == "cancelled"
        return
    assert answered.status_code == 409 and answered.json()["detail"]["code"] == "full_text_not_delivered"
    assert rig.fx.store.get(held.ask_id).status == "asked"
    full = rig.fx.store.get(held.ask_id).context["full_view"]
    notice = rig.fx.store.get(full["notice"])
    rig.mind.outbox.sending(notice.id, target=f"rcs:{CHAT}")
    rig.mind.outbox.sent(notice.id, result="sent", hermes_ref="full-text-delivery")
    rig.fx.shift(minutes=1)
    again = await app_answer(rig, held.ask_id, "yes", key="long")      # the refusal was not recorded
    assert again.status_code == 200 and again.json()["status"] == "approved"


# -- owner attribution -----------------------------------------------------------------------------------------------

async def test_only_the_owner_through_the_answer_credential_answers(rig):
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    cases = [
        # The instance key, granted the owner on companion, answers nothing this way; nor does any other credential.
        ({}, {"Authorization": f"Bearer {KEY}"}, 403, "answer_credential_required"),
        ({}, {"Authorization": f"Bearer {LANE_SECRET}"}, 403, "answer_credential_required"),
        ({}, {"Authorization": f"Bearer {RCS_ONLY_SECRET}"}, 403, "answer_credential_required"),
        ({}, {"Authorization": f"Bearer {GUEST_SECRET}"}, 403, "answer_credential_required"),
        ({}, {"Authorization": f"Bearer {KEY}", "X-ColonyMind-Session": "child-session"}, 403,
         "answer_credential_required"),
        ({"platform": "rcs"}, None, 403, "platform_not_allowed"),
        ({"platform": "whatsapp"}, None, 403, "platform_not_allowed"),
        # Taken exactly as spelled: never lowercased or trimmed.
        ({"platform": "Companion"}, None, 422, "invalid_platform"),
        ({"platform": " companion"}, None, 422, "invalid_platform"),
        ({"platform": "companion "}, None, 422, "invalid_platform"),
        ({"contact_id": "p-02"}, None, 403, "not_owner"),
        ({"contact_id": None, "sender_id": "someone-else"}, None, 403, "not_owner"),
        ({"contact_id": "p-02", "sender_id": "someone-else"}, None, 403, "not_owner"),
        ({}, {"X-ColonyMind-Session": "child-session"}, 403, "hand_session_refused"),
    ]
    await rig.store.db.execute("INSERT INTO hand_sessions(session_id,hand_id,parent_session,epoch,registered_at) "
                               "VALUES (?,?,?,?,?)", ("child-session", "h-000000000001", "owner-session",
                                                      json.dumps(EPOCH, sort_keys=True), rig.fx.now.isoformat()))
    await rig.store.db.commit()
    for extra, headers, status, code in cases:
        refused = await app_answer(rig, held.ask_id, "yes", headers=headers, **extra)
        assert refused.status_code == status, (extra, refused.text)
        assert refused.json()["detail"]["code"] == code, (extra, refused.text)
    no_one = await rig.hub.post(f"/v1/mind/permit-asks/{held.ask_id}/answer", headers={"Idempotency-Key": "k"},
                                json={"choice": "yes", "platform": "companion"})
    assert no_one.status_code == 422
    assert rig.fx.store.get(held.ask_id).status == "asked"
    assert settling_rows(rig.fx.store, held.ask_id) == []
    # The owner's companion handle, resolved by the contact store, answers.
    owner = await app_answer(rig, held.ask_id, "yes", contact_id=None, sender_id="owner")
    assert owner.status_code == 200, owner.text


async def test_the_mind_refuses_an_unattributed_answer_whatever_its_caller(rig):
    """The route's checks are not the only ones: the mind's own entry point refuses any authority but an answer
    credential's (the instance key's, an in-process call's, one granted every platform, development mode), anyone
    but the owner, any other platform and one spelled otherwise."""
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    base = dict(choice="yes", channel="companion", contact_id=OWNER, authority=HUB_AUTHORITY, key="k",
                request_hash="h")
    everywhere = CredentialTable.from_config({"anywhere": {"sha256": "c" * 64, "owner": "any", "contacts": "all",
                                                           "access": ["read", "write", "manage"]}})
    from colonymind.api.auth import anonymous_authority
    for change, code in (({"authority": key_authority(TABLE.instance)}, "answer_credential_required"),
                         ({"authority": key_authority()}, "answer_credential_required"),
                         ({"authority": credential_authority(everywhere.get("anywhere"))},
                          "answer_credential_required"),
                         ({"authority": credential_authority(TABLE.get("companion-lane"))},
                          "answer_credential_required"),
                         ({"authority": anonymous_authority()}, "answer_credential_required"),
                         ({"authority": None}, "answer_credential_required"),
                         ({"contact_id": "p-02"}, "not_owner"), ({"contact_id": None}, "not_owner"),
                         ({"channel": "rcs"}, "platform_not_allowed"), ({"channel": "COMPANION"}, "invalid_platform"),
                         ({"channel": "companion "}, "invalid_platform")):
        with pytest.raises(permit_asks.PermitAnswerError) as refused:
            await rig.mind.answer_explicit(held.ask_id, **{**base, **change})
        assert refused.value.code == code
    assert rig.fx.store.get(held.ask_id).status == "asked"


# -- the race: one row, one settlement ---------------------------------------------------------------------------

async def test_an_rcs_bind_and_an_app_answer_interleaved_settle_once(rig, monkeypatch):
    """The RCS reader holds the answer lock while it binds and reads (here paused inside it); the app's answer
    waits for it, then reads the RCS winner. Nothing is settled twice."""
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    evidence = rcs_evidence(rig.mind, held.ask_id, "yes")
    inside, release = asyncio.Event(), asyncio.Event()
    names = rig.mind._names

    async def paused(ids):
        inside.set()
        await release.wait()
        return await names(ids)
    monkeypatch.setattr(rig.mind, "_names", paused)
    app_inside = asyncio.Event()
    lookup = rig.fx.store.answer_request

    def entered(*args):
        app_inside.set()        # the explicit answer's first read inside the lock
        return lookup(*args)
    monkeypatch.setattr(rig.fx.store, "answer_request", entered)
    rcs = asyncio.create_task(rig.mind.read_answer(evidence))
    await asyncio.wait_for(inside.wait(), 5)
    app = asyncio.create_task(app_answer(rig, held.ask_id, "no", key="racer"))
    for _ in range(20):
        await asyncio.sleep(0.01)
    assert not app_inside.is_set() and not app.done() and rig.fx.store.get(held.ask_id).status == "asked"
    release.set()
    read, late = await asyncio.gather(rcs, app)
    assert read["approved"] == [held.ask_id]
    assert late.status_code == 409 and late.json()["detail"]["winner"]["channel"] == "rcs"
    assert rig.fx.store.get(held.ask_id).status == "approved"
    assert len(settling_rows(rig.fx.store, held.ask_id)) == 1


async def test_an_app_answer_and_an_rcs_bind_interleaved_settle_once(rig, monkeypatch):
    """The other order: the app's answer holds the lock (paused before its settling write); the RCS reader waits,
    then finds the ask settled and binds nothing."""
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    evidence = rcs_evidence(rig.mind, held.ask_id, "no")
    inside, release = asyncio.Event(), asyncio.Event()
    settle = rig.mind._settle_ask

    async def paused(row, **kwargs):
        inside.set()
        await release.wait()
        return await settle(row, **kwargs)
    monkeypatch.setattr(rig.mind, "_settle_ask", paused)
    rcs_inside = asyncio.Event()
    find = rig.mind.answer_turns.find

    def entered(*args, **kwargs):
        rcs_inside.set()        # the reader's first read inside the lock
        return find(*args, **kwargs)
    monkeypatch.setattr(rig.mind.answer_turns, "find", entered)
    app = asyncio.create_task(app_answer(rig, held.ask_id, "yes", key="first"))
    await asyncio.wait_for(inside.wait(), 5)
    rcs = asyncio.create_task(rig.mind.read_answer(evidence))
    for _ in range(20):
        await asyncio.sleep(0.01)
    assert not rcs_inside.is_set() and not rcs.done()
    release.set()
    won, read = await asyncio.gather(app, rcs)
    assert won.status_code == 200 and won.json()["winner"]["channel"] == "companion"
    assert read["settled"] == [] and read["approved"] == []
    assert rig.fx.store.get(held.ask_id).status == "approved"
    assert len(settling_rows(rig.fx.store, held.ask_id)) == 1


@pytest.mark.parametrize("choice", ["yes", "no"])
async def test_two_processes_racing_on_one_row_settle_it_once(rig, monkeypatch, choice):
    """No shared lock across processes: a second sidecar (its own Mind and its own SQLite connection to the same
    store) reads the ask open, then the first commits an RCS yes before the second writes. The second's guarded
    write finds the row moved on (a yes: the compare-and-set UPDATE itself; a no: the outcome path's re-read),
    writes nothing (no settlement, no idempotency record of a settlement) and answers the recorded winner."""
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    other_store = InitiativeStore(state_dir=rig.fx.state)
    other = Mind(config=rig.fx.config, store=other_store, state_dir=rig.fx.state, owner_id=OWNER,
                 contacts=rig.fx.contacts, ledger=rig.fx.ledger, clock=lambda: rig.fx.now, backups=False)
    inside, release = asyncio.Event(), asyncio.Event()
    settle = other._settle_ask

    async def paused(row, **kwargs):
        inside.set()
        await release.wait()
        return await settle(row, **kwargs)
    monkeypatch.setattr(other, "_settle_ask", paused)
    try:
        second = asyncio.create_task(other.answer_explicit(
            held.ask_id, choice=choice, channel="companion", contact_id=OWNER, authority=HUB_AUTHORITY,
            key="second", request_hash="h-second"))
        await asyncio.wait_for(inside.wait(), 5)
        read = await rig.mind.read_answer(rcs_evidence(rig.mind, held.ask_id, "yes"))
        assert read["approved"] == [held.ask_id]
        release.set()
        code, result = await second
        assert code == 409 and result["result"] == "already_answered"
        assert result["winner"]["channel"] == "rcs" and result["winner"]["choice"] == "yes"
        assert rig.fx.store.get(held.ask_id).status == "approved"
        assert len(settling_rows(rig.fx.store, held.ask_id)) == 1
        recorded = rig.fx.store.answer_request("companion-hub", "second")
        assert recorded["status_code"] == 409 and recorded["response"]["result"] == "already_answered"
    finally:
        other_store.close()


async def test_the_rcs_reader_never_overwrites_a_settlement_committed_while_it_read(rig, monkeypatch):
    """The messaging path's settling write is guarded too: an ask another process settled while the reader was
    binding (after it selected the open asks) is not settled again, and the reader says why."""
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    evidence = rcs_evidence(rig.mind, held.ask_id, "yes")
    other_store = InitiativeStore(state_dir=rig.fx.state)
    other = Mind(config=rig.fx.config, store=other_store, state_dir=rig.fx.state, owner_id=OWNER,
                 contacts=rig.fx.contacts, ledger=rig.fx.ledger, clock=lambda: rig.fx.now, backups=False)
    names = rig.mind._names

    async def meanwhile(ids):
        code, _ = await other.answer_explicit(held.ask_id, choice="no", channel="companion", contact_id=OWNER,
                                              authority=HUB_AUTHORITY, key="meanwhile", request_hash="h")
        assert code == 200
        return await names(ids)
    monkeypatch.setattr(rig.mind, "_names", meanwhile)
    try:
        read = await rig.mind.read_answer(evidence)
    finally:
        other_store.close()
    assert read["approved"] == [] and read["settled"] == []
    assert [item["id"] for item in read["refused"]] == [held.ask_id]
    ask = rig.fx.store.get(held.ask_id)
    assert (ask.status, ask.outcome) == ("cancelled", "denied")
    assert len(settling_rows(rig.fx.store, held.ask_id)) == 1
    assert (await rig.hub.get(f"/v1/mind/permit-asks/{held.ask_id}")).json()["winner"]["channel"] == "companion"


async def test_the_idempotency_record_commits_with_the_settlement_or_not_at_all(rig, monkeypatch):
    """A key another process recorded in the moment before this settlement commits: the settlement rolls back with
    the failed record (the ask stays as it was), and the answer replays or conflicts with that record."""
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    other_store = InitiativeStore(state_dir=rig.fx.state)
    settle = rig.mind._settle_ask
    digest = permit_asks.request_hash(held.ask_id, answer_body("yes"))

    async def raced(row, **kwargs):
        other_store.record_answer_request(scope="companion-hub", key="same", ask_id=row.id,
                                          request_hash="another-request",
                                          status_code=200, response={"result": "settled"}, at=rig.fx.now)
        return await settle(row, **kwargs)
    monkeypatch.setattr(rig.mind, "_settle_ask", raced)
    try:
        with pytest.raises(permit_asks.PermitAnswerError) as refused:
            await rig.mind.answer_explicit(held.ask_id, choice="yes", channel="companion", contact_id=OWNER,
                                           authority=HUB_AUTHORITY, key="same", request_hash=digest)
    finally:
        other_store.close()
    assert refused.value.code == "idempotency_conflict"
    assert rig.fx.store.get(held.ask_id).status == "asked"
    assert settling_rows(rig.fx.store, held.ask_id) == []


def test_guarded_transitions_from_many_connections_settle_a_row_once(tmp_path):
    """The store's compare-and-set under real concurrency: threads, each with its own connection (a process each, as
    far as SQLite knows), settle one asked row at once. Exactly one commits; every other is refused, writes nothing,
    and sees the winner's status."""
    first = InitiativeStore(state_dir=tmp_path)
    row, _ = first.create_intention(kind="note", type="floor_effect", title="t", drive="duty", cls="floor",
                                    decision="ask", decision_reason="r", status="asked", dedup_key=None,
                                    ask_code="ABC")
    stores = [first, *(InitiativeStore(state_dir=tmp_path) for _ in range(7))]
    start, results = threading.Barrier(len(stores)), []

    def settle(store, index):
        start.wait()
        try:
            store.transition(row.id, "approved" if index % 2 else "cancelled", action="queued" if index % 2
                             else "outcome_denied", details={"by": f"racer-{index}"},
                             guard=SettleGuard(writes=lambda status, at: [
                                 store.answer_request_write(scope="s", key=f"k-{index}", ask_id=row.id,
                                                            request_hash="h", status_code=200,
                                                            response={"status": status}, at=at)]))
            results.append(("won", index))
        except TransitionRefused as refused:
            results.append(("refused", refused.current.status))
    threads = [threading.Thread(target=settle, args=(store, index)) for index, store in enumerate(stores)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    try:
        won = [item for item in results if item[0] == "won"]
        assert len(results) == len(stores) and len(won) == 1
        final = first.get(row.id).status
        assert all(item[1] == final for item in results if item[0] == "refused")
        history = [item for item in first.get_history(row.id) if item.action in {"queued", "outcome_denied"}]
        assert len(history) == 1 and history[0].details["by"] == f"racer-{won[0][1]}"
        recorded = first._db.execute("SELECT key FROM ask_answer_requests").fetchall()
        assert [item[0] for item in recorded] == [f"k-{won[0][1]}"]
    finally:
        for store in stores:
            store.close()


# -- the readers never take an explicit-answer channel from another credential ------------------------------------

def ordinary_ask(fx, code="K7F"):
    row, _ = fx.store.create_intention(
        kind="task", type="research", title="Tidy the album", drive="curiosity", cls="self", decision="ask",
        decision_reason="r", status="asked", dedup_key=f"test:{code}", ask_code=code, context={"notice": True},
        hermes_kind="none", created_at=fx.now, expires_at=fx.now + timedelta(days=3))
    return row


def companion_evidence(fx, shown, number, platform="companion"):
    return {"platform": platform, "contact_id": OWNER, "chat_id": "hub-chat", "chat_type": "dm",
            "message_id": f"hub-in-{number}", "turn_id": f"hub-turn-{number}", "arrived_at": fx.now.isoformat(),
            "text": "yes", "reply_to_message_id": shown}


@pytest.mark.parametrize("secret,code", [(LANE_SECRET, "platform_not_allowed"), (HUB_SECRET, "route_not_granted")])
async def test_a_companion_credential_cannot_register_or_quote_its_own_shown_copy(rig, secret, code):
    """A credential granted the owner on companion (the relay's answer credential, or one with the owner's side of the
    API) could register its own copy of any ask as shown and quote it, approving a task ask, a standing grant or a
    people change without the explicit route's checks. Both readers refuse companion from any credential but the
    instance key, however it is spelled."""
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    task = ordinary_ask(rig.fx)
    headers = {"Authorization": f"Bearer {secret}"}
    for number, ident in enumerate((task.id, held.ask_id)):
        for platform in ("companion", "Companion", " companion"):
            shown = await rig.client.post(f"/v1/mind/asks/{ident}/shown", headers=headers, json={
                "platform": platform, "chat_id": "hub-chat", "message_id": f"hub-copy-{number}"})
            assert shown.status_code == 403 and shown.json()["detail"]["code"] == code, shown.text
            answered = await rig.client.post("/v1/mind/answers", headers=headers, json=companion_evidence(
                rig.fx, f"hub-copy-{number}", number, platform))
            assert answered.status_code == 403 and answered.json()["detail"]["code"] == code, answered.text
    for ident in (task.id, held.ask_id):
        assert rig.fx.store.get(ident).status == "asked" and settling_rows(rig.fx.store, ident) == []


async def test_the_instance_keys_companion_chat_lane_still_answers_through_the_readers(rig):
    """The Hermes adapter's own chat lane on companion (the instance key, granted the owner there) is unchanged: a
    quoted reply to an ask shown in that chat answers it, as on any messaging channel."""
    task = ordinary_ask(rig.fx)
    shown = await rig.client.post(f"/v1/mind/asks/{task.id}/shown", json={
        "platform": "companion", "chat_id": "hub-chat", "message_id": "lane-copy", "at": rig.fx.now.isoformat()})
    assert shown.status_code == 200, shown.text
    rig.fx.shift(seconds=30)
    answered = await rig.client.post("/v1/mind/answers", json=companion_evidence(rig.fx, "lane-copy", 99))
    assert answered.status_code == 200 and answered.json()["approved"] == [task.id], answered.text


# -- the owner is told where they read --------------------------------------------------------------------------

def told(fx):
    return [row for row in fx.store.intentions(kind=["message"], types=["ask_answered"], limit=50)]


async def rcs_started_hold(rig, hand="h-000000000009", task="t-000000000009"):
    """A hand whose task came from the owner's RCS message: its ask is anchored in that RCS chat."""
    await rig.store.db.execute("INSERT INTO tasks(id,origin,origin_ref,kind) VALUES (?,?,?,?)", (task, "owner_turn",
                               json.dumps({"platform": "rcs", "chat": CHAT, "message_id": "rcs-origin"}), "coding"))
    await seed(rig.store, hand=hand, task=task)
    return await rig.service.hold(hand, rig.body, f"hold-{hand}")


@pytest.mark.parametrize("choice", ["yes", "no"])
async def test_an_app_answer_is_told_to_the_owner_at_the_asks_rcs_anchor(rig, choice):
    held = await rcs_started_hold(rig)
    answered = await app_answer(rig, held.ask_id, choice, key="told")
    assert answered.status_code == 200, answered.text
    message, = told(rig.fx)
    text = message.context["text"]
    assert text.startswith("Answered in the app: ") and text.endswith(
        f"-> yes. An answer given in the app is not undone from this chat." if choice == "yes" else "-> no.")
    assert message.entity_id == OWNER and message.status == "approved"
    assert "asks" not in message.context and message.context["answered_ask"] == held.ask_id
    offered = next(item for item in rig.mind.outbox.ready() if item["id"] == message.id)
    assert offered["target"] == f"rcs:{CHAT}"
    assert offered["delivery_ref"] == {"platform": "rcs", "chat": CHAT, "message_id": "rcs-origin"}
    assert rig.mind.outbox.sending(message.id, target="companion:hub-chat") is None   # never another chat
    assert rig.mind.outbox.sending(message.id, target=f"rcs:{CHAT}").status == "sending"
    # Told once: a replay, a late answer and a read change nothing.
    await app_answer(rig, held.ask_id, choice, key="told")
    await app_answer(rig, held.ask_id, "no" if choice == "yes" else "yes", key="late")
    assert len(told(rig.fx)) == 1


async def test_without_an_rcs_anchor_the_line_goes_to_the_primary_handle_not_the_last_chat(rig):
    """A hand ask with no RCS origin: the line names no chat, so the body sends it to the owner's primary handle,
    even when the owner's latest message the answer route read came in on companion."""
    task = ordinary_ask(rig.fx)
    await rig.client.post(f"/v1/mind/asks/{task.id}/shown", json={
        "platform": "companion", "chat_id": "hub-chat", "message_id": "lane-copy", "at": rig.fx.now.isoformat()})
    assert (await rig.client.post("/v1/mind/answers", json=companion_evidence(rig.fx, "lane-copy", 98))).status_code \
        == 200
    assert rig.mind._owner_chat(task) == "companion:hub-chat"       # where an ask's own message would go now
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    assert (await app_answer(rig, held.ask_id)).status_code == 200
    message, = told(rig.fx)
    offered = next(item for item in rig.mind.outbox.ready() if item["id"] == message.id)
    assert "target" not in offered and "delivery_ref" not in offered and offered["recipient_is_owner"] is True


def test_the_line_is_never_held_or_dropped_and_counts_as_a_notice():
    from colonymind.mind import audit, gate
    assert "ask_answered" in gate.PROMPTED_TYPES and "ask_answered" not in gate.PAUSED_TYPES
    assert "ask_answered" in audit.NOTICE_TYPES


# -- undo -------------------------------------------------------------------------------------------------------

def undo_words(fx, number):
    return {"platform": "rcs", "contact_id": OWNER, "chat_id": CHAT, "chat_type": "dm",
            "message_id": f"rcs-undo-{number}", "turn_id": f"rcs-undo-turn-{number}",
            "arrived_at": fx.now.isoformat(), "text": "undo"}


async def test_rcs_undo_words_take_back_nothing_after_a_newer_app_yes(rig, tmp_path):
    """Undo words name no ask: after an RCS yes to one ask and a newer app yes to another, they would take back the
    RCS one while the owner means the app one. Nothing is undone, and the owner is told the app answer stands."""
    here = await rig.service.hold("h-000000000001", rig.body, "hold")
    there, _ = await hold_other(rig, tmp_path, hand="h-000000000002", task="t-000000000002", name="other.txt")
    assert (await rig.mind.read_answer(rcs_evidence(rig.mind, here.ask_id, "yes")))["approved"] == [here.ask_id]
    rig.fx.shift(minutes=1)
    assert (await app_answer(rig, there.ask_id)).status_code == 200
    rig.fx.shift(minutes=1)
    undo = await rig.mind.read_answer(undo_words(rig.fx, 1))
    assert [(item["id"], item["undone"]) for item in undo["undone"]] == [(there.ask_id, "elsewhere")]
    assert "answered in the app" in undo["context"] and "took back nothing" in undo["context"]
    for ident in (here.ask_id, there.ask_id):
        assert rig.fx.store.get(ident).status == "approved"
        assert not any(item.action == "ask_undone" for item in rig.fx.store.get_history(ident))


async def test_rcs_undo_words_still_take_back_an_rcs_yes_newer_than_an_app_yes(rig, tmp_path):
    there = await rig.service.hold("h-000000000001", rig.body, "hold")
    here, _ = await hold_other(rig, tmp_path, hand="h-000000000002", task="t-000000000002", name="other.txt")
    assert (await app_answer(rig, there.ask_id)).status_code == 200
    rig.fx.shift(minutes=1)
    assert (await rig.mind.read_answer(rcs_evidence(rig.mind, here.ask_id, "yes")))["approved"] == [here.ask_id]
    rig.fx.shift(minutes=1)
    undo = await rig.mind.read_answer(undo_words(rig.fx, 2))
    assert [(item["id"], item["undone"]) for item in undo["undone"]] == [(here.ask_id, "cancelled")]
    assert rig.fx.store.get(here.ask_id).status == "cancelled" and rig.fx.store.get(there.ask_id).status == "approved"


# -- the full text, read as a yes reads it ------------------------------------------------------------------------

async def test_full_text_delivered_reads_the_durable_record_without_marking_it(rig):
    """A full-text message the body reported sent counts though the hook that marks it seen never ran, exactly as an
    answer's own check reads it; the view only reads, and the explicit yes that follows is taken."""
    long_body = rig.body.model_copy(update={"call": wire.ExactCall(tool="terminal", arguments={
        "command": rig.body.call.arguments["command"] + " && echo " + "x" * 1600})})
    held = await rig.service.hold("h-000000000001", long_body, "hold")
    await rig.mind._deliver_asks([rig.fx.store.get(held.ask_id)], rig.mind.clock())
    notice = rig.fx.store.get(rig.fx.store.get(held.ask_id).context["full_view"]["notice"])
    rig.fx.store.transition(notice.id, "sent", action="sent", completed_at=rig.fx.now)    # no hook ran
    terms = (await rig.hub.get(f"/v1/mind/permit-asks/{held.ask_id}")).json()["terms"]
    assert terms["full_text_required"] is True and terms["full_text_delivered"] is True
    assert not rig.fx.store.get(held.ask_id).context["full_view"].get("seen")        # the read marked nothing
    rig.fx.shift(minutes=1)
    assert (await app_answer(rig, held.ask_id)).json()["status"] == "approved"


# -- the lapse sweep is guarded too --------------------------------------------------------------------------------

async def test_the_lapse_sweep_leaves_an_ask_answered_after_it_was_listed(rig, monkeypatch):
    held = await rig.service.hold("h-000000000001", rig.body, "hold")
    listed = rig.fx.store.get(held.ask_id)
    assert (await app_answer(rig, held.ask_id)).status_code == 200
    rig.fx.shift(days=4)
    intentions = rig.fx.store.intentions

    def stale(**kwargs):
        return [listed] if kwargs.get("status") == ["asked"] else intentions(**kwargs)
    monkeypatch.setattr(rig.fx.store, "intentions", stale)
    assert rig.mind._expire_asks(rig.fx.now) == 0
    assert rig.fx.store.get(held.ask_id).status == "approved"
    assert len(settling_rows(rig.fx.store, held.ask_id)) == 1

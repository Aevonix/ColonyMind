"""The instance key alone approves nothing (docs/MIND.md, Asks: console and API answers).

An agent with a shell can read the instance key, so a request that cites no owner turn (the CLI, a script with the
key, any credential) may not widen anything: a yes to an ask and a new standing grant are refused
``approve_required`` for every credential. It may always narrow: a no is taken and recorded ``by: operator``, never
as the owner's own answer. The console can put an ask in front of the owner again (``show``), in their chat, where
their own answer approves it.
"""

from __future__ import annotations

import json
from datetime import timedelta

import httpx
import pytest
import yaml
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from colonymind.api.credentials import CredentialTable, secret_digest
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import mind as mind_router
from colonymind.mind import asks_report
from colonymind.mind import cli as mind_cli
from colonymind.mind.authority import parse_grant
from owner_answer import CHAT, PLATFORM, owner_answers
from test_mind_cli import _Transport, _parse, home  # noqa: F401  (home: the CLI's instance fixture)
from test_mind_grants import PAY
from test_mind_loop import AUTH, CONTACT, KEY, OWNER, Fixture, pinned_clock  # noqa: F401  (autouse fixture)

APPROVE_REQUIRED = "approve_required"
# A second credential granted the owner on every platform: the strongest grant a credential can have.
CONSOLE_SECRET = "console-credential-secret-0123456789"


@pytest.fixture
def fx(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fixture = Fixture(tmp_path)
    yield fixture
    mind_router.set_mind(None)
    fixture.store.close()


def _app(fx, *, keyed: bool = True) -> FastAPI:
    """The mind routes behind the middleware with the instance key and a console credential granted the owner
    everywhere (``keyed``), or mounted bare (no credential stands behind a request)."""
    app = FastAPI()
    if keyed:
        app.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=CredentialTable.from_config({
            "api-key": {"owner": ["whatsapp", "sms", "rcs", "voice"], "contacts": "all"},
            "console": {"sha256": secret_digest(CONSOLE_SECRET), "owner": "any", "contacts": "all",
                        "access": ["read", "write", "manage"]}}))
    app.include_router(mind_router.router)
    mind_router.set_mind(fx.mind)
    return app


async def _open_ask(fx, client) -> dict:
    """One open ask: a message to a contact on ``ask`` that a due commitment formed."""
    fx.commitments.create(person_id=CONTACT, description="send the contact the slides",
                          due_at=(fx.now + timedelta(minutes=1)).isoformat())
    fx.shift(minutes=10)
    await client.post("/v1/mind/tick", headers=AUTH)
    ask, = (await client.get("/v1/mind/asks", headers=AUTH)).json()["asks"]
    return ask


def _refused(response) -> str:
    assert response.status_code == 403, response.text
    detail = response.json()["detail"]
    assert detail["code"] == APPROVE_REQUIRED, detail
    return detail["message"]


def _settling(fx, ident: str):
    return next(item for item in fx.store.get_history(ident, limit=50) if item.action.startswith("outcome_"))


async def test_key_only_yes_refused(fx):
    """The one-call route: the instance key's yes approves nothing, whatever its body claims, and says where a yes
    comes from and how to put the ask in front of the owner again."""
    async with AsyncClient(transport=ASGITransport(app=_app(fx)), base_url="http://test") as client:
        ask = await _open_ask(fx, client)
        code = ask["ask_code"]
        for body in (None, {"by": "cli"}, {"by": "owner"}, {"contact_id": OWNER, "message": f"yes {code}"}):
            message = _refused(await client.post(f"/v1/mind/asks/{code}/yes", headers=AUTH, json=body))
            assert "own" in message and "chat" in message and f"colonymind mind show {code}" in message, message
        _refused(await client.post("/v1/mind/asks/QQQ/yes", headers=AUTH))       # refused before any lookup
        assert fx.store.get(ask["id"]).status == "asked"
        assert (await client.get("/v1/mind/dispatch", headers=AUTH)).json() == []


async def test_every_credential_yes_is_refused(fx):
    """Not only the instance key: a credential granted the owner on every platform, and a request no credential
    stands behind, are refused the same way."""
    async with AsyncClient(transport=ASGITransport(app=_app(fx)), base_url="http://test") as client:
        ask = await _open_ask(fx, client)
        console = {"Authorization": "Bearer " + CONSOLE_SECRET}
        assert (await client.get("/v1/mind/asks", headers=console)).json()["asks"]    # it reaches the owner's side
        _refused(await client.post(f"/v1/mind/asks/{ask['ask_code']}/yes", headers=console, json={"by": "owner"}))
    async with AsyncClient(transport=ASGITransport(app=_app(fx, keyed=False)), base_url="http://test") as client:
        _refused(await client.post(f"/v1/mind/asks/{ask['ask_code']}/yes", json={"by": "cli"}))
    assert fx.store.get(ask["id"]).status == "asked"


async def test_a_console_no_is_the_operators_never_the_owners(fx):
    """A no narrows, so the console may say it; it is recorded ``by: operator`` with the console path, whatever the
    body calls itself, and the asks report counts it as the operator's."""
    async with AsyncClient(transport=ASGITransport(app=_app(fx)), base_url="http://test") as client:
        ask = await _open_ask(fx, client)
        declined = await client.post(f"/v1/mind/asks/{ask['ask_code']}/no", headers=AUTH, json={"by": "owner"})
        assert declined.status_code == 200 and declined.json()["ok"] is True
    row = fx.store.get(ask["id"])
    assert row.status == "cancelled"
    settling = _settling(fx, ask["id"])
    assert settling.details["by"] == "operator"
    assert settling.details["answer"] == {"path": "console", "by": "operator"}
    assert asks_report.settler(settling.details["by"], "denied") == "operator"


def test_only_a_path_the_answer_reader_records_is_the_owners_own_answer():
    """The owner's own answer is one ``Mind.read_answer`` recorded after its checks: a reaction, a quoted reply, a
    follow-up or a list (``answers.PATHS``), or one ``Mind.answer_explicit`` recorded after the credential's owner
    grant on the answering channel and the owner's identity were checked (``answers.EXPLICIT``). No path of an earlier
    release stands beside them: a settling row naming the retired code-in-words path (``code``) is nobody's own
    answer, whatever ``by`` it names."""
    from colonymind.mind.answers import EXPLICIT, PATHS
    from colonymind.mind.outcomes import OWNER_ANSWER_PATHS, owner_answer
    assert OWNER_ANSWER_PATHS == frozenset((*PATHS, EXPLICIT))
    for path in (*PATHS, EXPLICIT):
        assert owner_answer("owner-turn:t1", {"answer": {"path": path, "by": "owner-turn:t1"}}) is True
    assert owner_answer("owner", {"answer": {"path": "code", "reader": "tables", "by": "owner"}}) is False
    assert asks_report.settler("owner", "denied", {"answer": {"path": "code", "by": "owner"}}) == "operator"


async def test_a_console_no_teaches_nothing(fx):
    """The operator's no settles the ask and is learned as nothing: no verdict, no feedback to the ranker and no
    dismissal the agent feels, as for any answer that is not the owner's own (docs/MIND.md, Asks). The route's part
    is the ``by: operator`` and the console path it records; the outcomes read them (``outcomes.owner_answer``)."""
    row, _ = fx.store.create_intention(
        kind="task", type="research", title="Re-run the archive export", drive="curiosity", cls="self",
        decision="ask", decision_reason="suggest: self -> ask", status="asked", dedup_key="test:console-no",
        ask_code="K7F", context={"notice": True}, hermes_kind="none", created_at=fx.now,
        expires_at=fx.now + timedelta(days=3))
    async with AsyncClient(transport=ASGITransport(app=_app(fx)), base_url="http://test") as client:
        declined = await client.post("/v1/mind/asks/K7F/no", headers=AUTH, json={"by": "owner"})
        assert declined.status_code == 200, declined.text
    settled = fx.store.get(row.id)
    assert (settled.status, settled.outcome) == ("cancelled", "denied")
    assert _settling(fx, row.id).details["by"] == "operator"
    assert settled.verdict is None
    assert fx.feedback.multiplier("research:curiosity") == 1.0
    fx.mind.feelings.update(fx.now)
    assert fx.mind.feelings.view().dismissals == 0
    assert float((fx.mind.mind_state.get("affect.dismissed") or {}).get("level") or 0.0) == 0.0


async def test_the_record_says_who_said_no(fx, monkeypatch):
    """What the mind keeps of a no (the settling row's result, which its autobiography narrates in the owner's own
    memory) names who said it: the operator's no at the console is never written down as the owner's, for an
    ordinary ask, a people change or a standing grant; the owner's own no, from their reply, is theirs."""
    written = []
    autobiography = fx.mind.outcomes.autobiography
    monkeypatch.setattr(autobiography, "record", lambda ident, event, text, **extra: written.append((ident, text)))

    def ask(code, **fields):
        row, _ = fx.store.create_intention(
            **{"kind": "task", "type": "research", "title": f"Re-run export {code}", "drive": "curiosity",
               "cls": "self", "context": {"notice": True}, **fields},
            decision="ask", decision_reason="an ask", status="asked", dedup_key=f"test:{code}", ask_code=code,
            hermes_kind="none", created_at=fx.now, expires_at=fx.now + timedelta(days=3))
        return row
    change = {"operation": "set_permission", "contact_id": CONTACT, "value": "auto", "from": {}}
    rows = [ask("K7F"), ask("P3Q", kind="note", type="people_change", cls="owner", context={"change": change}),
            ask("G4R", kind="note", type="grant_confirm", cls="owner", context={"grant": PAY})]
    for row in rows:
        await fx.mind.answer(row.ask_code, yes=False, by="cli")
    for row in rows:
        settled = fx.store.get(row.id)
        assert settled.status == "cancelled" and "owner" not in settled.result and "operator" in settled.result, \
            settled.result
        narrated = [text for ident, text in written if ident == row.id]
        assert narrated and not any("the owner" in text for text in narrated), narrated
    own = ask("M3P")
    await owner_answers(fx.mind, own.ask_code, "no")
    assert fx.store.get(own.id).result.startswith("the owner said no")


async def test_the_record_of_an_undo_keeps_no_message_key(fx, monkeypatch):
    """An undo is the owner's own (it is read only from their message), and what the mind keeps of it (the settling
    row's result, which its autobiography narrates in the owner's own memory) says so in words: never the key of
    the message it came in, which is the answer turns' bookkeeping and stays in the settling history."""
    written = []
    monkeypatch.setattr(fx.mind.outcomes.autobiography, "record",
                        lambda ident, event, text, **extra: written.append((ident, text)))
    row, _ = fx.store.create_intention(
        kind="task", type="research", title="Re-run export U7K", drive="curiosity", cls="self",
        context={"notice": True}, decision="ask", decision_reason="an ask", status="asked", dedup_key="test:U7K",
        ask_code="U7K", hermes_kind="none", created_at=fx.now, expires_at=fx.now + timedelta(days=3))
    assert (await owner_answers(fx.mind, row.ask_code, "yes")).status == "approved"
    fx.shift(minutes=1)
    result = await fx.mind.read_answer({"platform": PLATFORM, "contact_id": OWNER, "chat_id": CHAT, "chat_type": "dm",
                                        "message_id": "undo-u7k", "turn_id": "t-undo-u7k",
                                        "arrived_at": fx.now.isoformat(), "text": "undo"})
    assert [item["undone"] for item in result["undone"]] == ["cancelled"]
    settled = fx.store.get(row.id)
    assert settled.status == "cancelled" and settled.result.startswith("the owner undid"), settled.result
    narrated = [text for ident, text in written if ident == row.id]
    for text in (settled.result, *narrated):
        assert "owner-turn" not in text and "undo-u7k" not in text and CHAT not in text, text


async def test_the_retired_decide_route_is_gone(fx):
    """``POST /v1/mind/decide`` (the 1.11 chat path) took a caller-supplied ``message`` as the owner's turn, so any
    holder of a credential granted the owner could approve an ask by quoting a code. No plugin calls it since the
    owner's own message goes to ``/v1/mind/answers``; the route is removed, not narrowed. Whatever the body claims and
    whoever sends it, it is not found, and the ask stays open."""
    assert not [route for route in mind_router.router.routes if getattr(route, "path", "").endswith("/decide")]
    async with AsyncClient(transport=ASGITransport(app=_app(fx)), base_url="http://test") as client:
        ask = await _open_ask(fx, client)
        code = ask["ask_code"]
        owner_turn = {"code": code, "answer": "yes", "session_id": "s-owner", "contact_id": OWNER,
                      "message": f"yes {code}", "by": "owner"}
        for headers in (AUTH, {"Authorization": "Bearer " + CONSOLE_SECRET}):
            for body in (owner_turn, {**owner_turn, "answer": "no"}):
                response = await client.post("/v1/mind/decide", headers=headers, json=body)
                assert response.status_code in {404, 410}, response.text
    async with AsyncClient(transport=ASGITransport(app=_app(fx, keyed=False)), base_url="http://test") as client:
        assert (await client.post("/v1/mind/decide", json=owner_turn)).status_code in {404, 410}
    assert fx.store.get(ask["id"]).status == "asked"
    assert not [item for item in fx.store.get_history(ask["id"], limit=50) if item.action.startswith("outcome_")]


async def test_a_grant_without_an_owner_turn_is_refused_and_one_asked_in_conversation_waits(fx):
    """A standing grant is never made active through the API: naming no conversation is refused for every
    credential; from the owner's conversation it becomes their ask, answered in their chat."""
    async with AsyncClient(transport=ASGITransport(app=_app(fx)), base_url="http://test") as client:
        for headers in (AUTH, {"Authorization": "Bearer " + CONSOLE_SECRET}):
            message = _refused(await client.post("/v1/mind/grants", headers=headers, json={
                "grant": PAY, "reason": "pay the small ones", "by": "cli"}))
            assert "mind.grants" in message and "chat" in message, message
        asked = await client.post("/v1/mind/grants", headers=AUTH, json={
            "grant": PAY, "reason": "pay the small ones", "by": "owner", "contact_id": OWNER,
            "message": "you may pay the small invoices"})
        assert asked.status_code == 200 and asked.json()["status"] == "asked"
    async with AsyncClient(transport=ASGITransport(app=_app(fx, keyed=False)), base_url="http://test") as client:
        _refused(await client.post("/v1/mind/grants", json={"grant": {**PAY, "id": "pay-bare"}, "by": "cli"}))
    active = [item for item in fx.mind.grants_state()["grants"] if item.get("valid") and not item.get("revoked")]
    assert active == []


async def test_show_puts_an_open_ask_in_front_of_the_owner_again(fx):
    """The console's way to an approval: the ask goes to the owner's chat again as its own message, in plain words
    and without its code; one waiting to go is not doubled, and only an open ask can be shown."""
    async with AsyncClient(transport=ASGITransport(app=_app(fx)), base_url="http://test") as client:
        ask = await _open_ask(fx, client)
        code = ask["ask_code"]
        for ident in (code.lower(), ask["id"]):
            shown = await client.post(f"/v1/mind/asks/{ident}/show", headers=AUTH)
            assert shown.status_code == 200 and shown.json()["ok"] is True, shown.text
        messages = [row for row in fx.store.intentions(kind=["message"], types=["ask_show"])]
        assert len(messages) == 1 and messages[0].entity_id == OWNER
        text = messages[0].context["text"]
        assert "slides" in text and code not in text and "yes or no" in text.lower()
        assert shown.json()["message"] == messages[0].id
        assert (await client.post("/v1/mind/asks/QQQ/show", headers=AUTH)).status_code == 404
    assert fx.store.get(ask["id"]).status == "asked"


# -- the CLI ---------------------------------------------------------------------------------------------------------

def test_the_cli_yes_says_where_a_yes_comes_from_and_calls_nothing(home, monkeypatch, capsys):
    transport = _Transport({})
    monkeypatch.setattr(httpx, "Client", transport.client)
    assert mind_cli.run(_parse(["mind", "yes", "k7f"])) == 1
    out = capsys.readouterr()
    said = out.out + out.err
    assert "chat" in said and "colonymind mind show K7F" in said and "colonymind mind no K7F" in said
    assert transport.calls == []


def test_the_cli_show_sends_the_ask_again_and_no_still_declines(home, monkeypatch, capsys):
    transport = _Transport({
        ("POST", "/v1/mind/asks/K7F/show"): {"ok": True, "id": "i-1", "message": "m-1",
                                            "text": "sent to the owner's chat again: Send the slides"},
        ("POST", "/v1/mind/asks/K7F/no"): {"ok": True, "text": "no: x -> cancelled"},
    })
    monkeypatch.setattr(httpx, "Client", transport.client)
    assert mind_cli.run(_parse(["mind", "show", "K7F"])) == 0        # a code or an id, passed as typed
    assert "sent to the owner's chat again" in capsys.readouterr().out
    assert mind_cli.run(_parse(["mind", "no", "k7f"])) == 0
    assert [(method, path) for method, path, _ in transport.calls] == [
        ("POST", "/v1/mind/asks/K7F/show"), ("POST", "/v1/mind/asks/K7F/no")]


def test_the_cli_grant_add_writes_nothing_and_prints_the_configuration_entry(home, monkeypatch, capsys):
    """A standing grant is the owner's to give in their chat, or the operator's to configure: ``grants add`` calls no
    route and prints the ``mind.grants`` entry to paste, which parses as the same grant."""
    transport = _Transport({})
    monkeypatch.setattr(httpx, "Client", transport.client)
    assert mind_cli.run(_parse(["mind", "grants", "add", "renew-certs", "--class", "credential_change", "--scope",
                                "renew TLS certificates", "--tools", "renew_certificates", "--match", "example.org",
                                "--per-day", "2"])) == 1
    out = capsys.readouterr().out
    assert transport.calls == []
    assert "chat" in out and "mind.grants" in out and "colonymind.yaml" in out
    entry = yaml.safe_load(out[out.index("mind:"):])["mind"]["grants"][0]
    assert entry == {"id": "renew-certs", "class": "credential_change", "scope": "renew TLS certificates",
                     "tools": ["renew_certificates"], "match": ["example.org"], "per_day": 2}
    assert parse_grant(entry).valid
    assert mind_cli.run(_parse(["mind", "grants", "add", "bad", "--class", "nope", "--scope", "x"])) == 2
    assert "class" in capsys.readouterr().err


def test_every_subcommand_still_parses_with_show():
    assert "show" in mind_cli.COMMANDS
    assert _parse(["mind", "show", "K7F"]).code == "K7F"
    assert json.dumps(sorted(mind_cli.COMMANDS))

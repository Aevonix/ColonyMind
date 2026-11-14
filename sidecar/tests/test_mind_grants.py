"""Standing grants: the owner's routine floor work, within limits the owner sets and sees.

A grant names one floor class, a scope and the structured tools it covers (a declared action, a declared tool, a
messaging tool, a cron, skill or memory removal, a vault fill): never a shell, code, browser or delegation call,
whose every effect outside the workspace is the owner's one-tap ask. Its limits: exact targets, an amount per
effect (an unknown amount is never covered), uses per day, an amount per month and an expiry. Only the owner sets
one: the configuration, or the owner's own answer to a ``grant_confirm`` ask (never the console: the instance key
alone approves nothing). A malformed grant is inert and shown as invalid; a revoke is immediate; a change to the
configured set is reported.
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from colonymind.initiatives.store import InitiativeStore
from colonymind.mind import cli as mind_cli
from colonymind.mind.authority import Authority, Effect, Policy, effect_of, parse_grant
from test_mind_loop import AUTH, CONTACT, OWNER, Fixture, pinned_clock  # noqa: F401  (pinned_clock: autouse fixture)
from owner_answer import owner_answers

# The worked example (docs/MIND.md, "Declared actions"): renewing one site's certificate is a dedicated tool with
# a fixed command and one checked argument, and the grant names that tool.
ACTIONS = {"renew_certificates": {
    "description": "Renew the TLS certificate of one site", "class": "credential_change",
    "command": ["certbot", "renew", "--cert-name", "{site}", "--non-interactive"],
    "args": {"site": {"type": "string", "enum": ["example.org", "www.example.org"]}}, "target": "site",
    "timeout": 300}}
CERTS = {"id": "renew-certs", "class": "credential_change", "scope": "renew TLS certificates",
         "tools": ["renew_certificates"], "per_day": 2}
PAY = {"id": "small-payments", "class": "money_movement", "scope": "pay invoices up to 50", "tools": ["pay_invoice"],
       "max_amount": 50, "currency": "USD", "per_month_amount": 80}
RENEW = ("colonymind_action", {"action": "renew_certificates", "args": {"site": "example.org"}})
# A payment tool the owner declared (``mind.floor.tools``): its amount and currency come from its arguments.
DECLARED = {"pay_invoice": {"class": "money_movement", "amount": "amount", "currency": "currency", "target": "payee"}}
FLOOR = {"tools": DECLARED}


def renew(site: str = "example.org"):
    return effect_of("colonymind_action", {"action": "renew_certificates", "args": {"site": site}}, actions=ACTIONS)


def paid(amount, currency: str = "USD"):
    """The effect of paying ``amount`` through the declared payment tool."""
    return effect_of("pay_invoice", {"amount": amount, "currency": currency, "payee": "acme"}, declared=DECLARED)


# ---------------------------------------------------------------------------
# What a grant is
# ---------------------------------------------------------------------------

def test_a_well_formed_grant_and_every_malformed_one_is_inert():
    grant = parse_grant(CERTS)
    assert grant.valid and grant.cls == "credential_change" and grant.tools == ("renew_certificates",)
    assert grant.per_day == 2 and grant.as_dict()["active"] is True and grant.as_dict()["source"] == "config"
    bad = [
        "not a mapping",
        {"class": "credential_change", "scope": "x", "tools": ["t"]},                    # no id
        {"id": "a", "class": "anything", "scope": "x", "tools": ["t"]},                  # no such class
        {"id": "a", "class": "credential_change", "tools": ["t"]},                       # no scope
        {"id": "a b", "class": "credential_change", "scope": "x", "tools": ["t"]},       # a malformed id
        {"id": "a", "class": "credential_change", "scope": "x"},                         # names no tool
        {"id": "a", "class": "credential_change", "scope": "x", "tools": ["t"], "per_day": 0},
        {"id": "a", "class": "credential_change", "scope": "x", "tools": ["t"], "per_day": "twice"},
        {"id": "a", "class": "money_movement", "scope": "x", "tools": ["t"], "max_amount": -5},
        {"id": "a", "class": "money_movement", "scope": "x", "tools": ["t"], "currency": "dollars"},
        {"id": "a", "class": "credential_change", "scope": "x", "tools": ["t"], "max_amount": 5},   # money's limit
        {"id": "a", "class": "credential_change", "scope": "x", "tools": ["t"], "expires": "next year"},
        {"id": "a", "class": "credential_change", "scope": "x", "tools": ["t"], "limit": 3},         # unknown field
        {"id": "a", "class": "credential_change", "scope": "x", "tools": [1, 2]},
        # a shell, code, browser or delegation call is never a grant's, and neither is executor use
        {"id": "a", "class": "executor", "scope": "run the tests", "tools": ["terminal"]},
        {"id": "a", "class": "credential_change", "scope": "x", "tools": ["terminal"]},
        {"id": "a", "class": "credential_change", "scope": "x", "tools": ["execute_code"]},
        {"id": "a", "class": "money_movement", "scope": "x", "tools": ["browser_click"]},
        {"id": "a", "class": "money_movement", "scope": "x", "tools": ["browser"]},
        {"id": "a", "class": "irreversible_deletion", "scope": "x", "tools": ["delegate_task"]},
        {"id": "a", "class": "credential_change", "scope": "x", "tools": ["colonymind_action"]},   # name the action
        # targets are exact values, never a pattern
        {"id": "a", "class": "bulk_third_party_messaging", "scope": "x", "tools": ["send_message"], "match": ["p-*"]},
    ]
    for raw in bad:
        parsed = parse_grant(raw)
        assert not parsed.valid and parsed.problem, raw
        assert parsed.as_dict()["active"] is False
    assert "never a grant's" in parse_grant(bad[-7]).problem
    policy = Policy.from_config({"grants": [CERTS, {**CERTS, "scope": "again"}, {"id": "x", "class": "nope"}]})
    assert [grant.valid for grant in policy.grants] == [True, False, False]     # a second id is inert too
    assert "second grant" in policy.grants[1].problem
    assert Policy.from_config({"grants": "all of them"}).grants[0].valid is False


def test_what_a_grant_covers(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    try:
        guard = Authority(Policy.from_config({"autonomy": "standard", "grants": [CERTS, PAY], "floor": FLOOR,
                                              "actions": ACTIONS}), store, owner_id=OWNER)
        assert guard.covering_grant(renew()).id == "renew-certs"
        assert guard.covering_grant(renew("www.example.org")).id == "renew-certs"
        # the same effect through a shell or code is never a grant's, whatever the command says
        for tool, args in (("terminal", {"command": "certbot renew --cert-name example.org"}),
                           ("terminal", {"command": "certbot renew --deploy-hook 'printf SYNTHETIC_MARKER'"}),
                           ("execute_code", {"code": "import os; os.system('certbot renew')"}),
                           ("terminal", {"command": "rm -rf /srv"})):
            effect = effect_of(tool, args)
            assert effect.floor and guard.covering_grant(effect) is None, (tool, args)
        # money: the amount per effect, its currency, an unknown amount never
        assert guard.covering_grant(paid(25)).id == "small-payments"
        assert guard.covering_grant(paid(75)) is None
        assert guard.covering_grant(paid(25, "EUR")) is None
        assert guard.covering_grant(effect_of("terminal", {"command": "curl -X POST https://api.stripe.com/v1/x"})) \
            is None
        # uses per day and the amount per month come from the recorded uses
        for _ in range(2):
            guard.record_grant_use("renew-certs", renew())
        assert guard.covering_grant(renew()) is None and guard.grant_usage("renew-certs")["uses_24h"] == 2
        guard.record_grant_use("small-payments", paid(50))
        assert guard.covering_grant(paid(35)) is None                          # 50 + 35 > 80 this month
        assert guard.covering_grant(paid(30)).id == "small-payments"
        assert guard.grant_usage("small-payments")["amount_30d"] == 50.0
    finally:
        store.close()


def test_expiry_and_exact_targets(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    try:
        expired = {**CERTS, "id": "old", "expires": "2001-01-01"}
        guard = Authority(Policy.from_config({"autonomy": "trusted", "grants": [expired], "actions": ACTIONS}), store,
                          owner_id=OWNER)
        assert guard.covering_grant(renew()) is None
        assert parse_grant(expired).as_dict(guard.clock())["expired"] is True
        # ``match`` names the exact targets a grant covers: the declared target argument, the recipients
        one = {**CERTS, "id": "one-site", "match": ["example.org"]}
        party = {"id": "party", "class": "bulk_third_party_messaging", "scope": "invite the neighbours",
                 "tools": ["send_message", "outbox"], "match": ["p-04", "p-05", "p-06", "p-07"]}
        guard = Authority(Policy.from_config({"autonomy": "trusted", "grants": [one, party], "actions": ACTIONS}),
                          store, owner_id=OWNER)
        assert guard.covering_grant(renew()).id == "one-site"
        assert guard.covering_grant(renew("www.example.org")) is None
        bulk = Effect("bulk_third_party_messaging", "send_message", target="telegram:7", recipients=("p-07",))
        assert guard.covering_grant(bulk).id == "party"
        assert guard.covering_grant(Effect("bulk_third_party_messaging", "send_message", recipients=("p-08",))) is None
        assert guard.covering_grant(Effect("bulk_third_party_messaging", "outbox", recipients=("p-05",))).id == "party"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# A grant at execution
# ---------------------------------------------------------------------------

def _task(fx, title="renew the certificates"):
    row, _ = fx.store.create_intention(kind="task", type="upkeep", title=title, drive="upkeep", cls="internal",
                                       decision="act", decision_reason="standard: internal -> act",
                                       status="approved", dedup_key=f"task:{title}", context={"body": title},
                                       created_at=fx.now)
    fx.store.transition(row.id, "approved", action="queued", at=fx.now)
    return fx.mind.bound(row.id, f"kanban-{row.id[:6]}")


async def test_a_granted_effect_acts_and_its_use_is_counted_until_the_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fx = Fixture(tmp_path, autonomy="standard", config={"grants": [CERTS], "actions": ACTIONS})
    try:
        task = _task(fx)
        for _ in range(2):
            verdict = await fx.mind.guard(tool=RENEW[0], args=RENEW[1], run="mind", task_id=task.hermes_ref)
            assert verdict["allow"] is True and verdict["grant"] == "renew-certs", verdict
        uses = [row for row in fx.store.intentions(kind=["note"]) if row.type == "grant_use"]
        assert len(uses) == 2 and uses[0].context["task"] == task.id and "renew_certificates" in uses[0].description
        third = await fx.mind.guard(tool=RENEW[0], args=RENEW[1], run="mind", task_id=task.hermes_ref)
        assert third["action"] == "block" and third["held"]                    # the day's uses are spent
        state = {item["id"]: item for item in fx.mind.state()["grants"]}
        assert state["renew-certs"]["uses_24h"] == 2 and state["renew-certs"]["per_day"] == 2
        fx.shift(hours=25)
        again = await fx.mind.guard(tool=RENEW[0], args=RENEW[1], run="mind", task_id=task.hermes_ref)
        assert again["allow"] is True                                         # a rolling day
    finally:
        fx.store.close()


# ---------------------------------------------------------------------------
# Who sets a grant
# ---------------------------------------------------------------------------

async def test_a_grant_asked_in_conversation_is_active_only_after_the_owners_own_yes(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fx = Fixture(tmp_path, autonomy="standard")
    try:
        with pytest.raises(PermissionError):
            fx.mind.request_grant(CERTS, reason="let it renew", contact_id=CONTACT)
        with pytest.raises(ValueError):
            fx.mind.request_grant(CERTS, reason="", contact_id=OWNER)            # the owner's words are required
        with pytest.raises(ValueError):
            fx.mind.request_grant({"id": "x", "class": "nope", "scope": "y"}, reason="go", contact_id=OWNER)
        asked = fx.mind.request_grant(CERTS, reason="you can renew the certificates yourself", contact_id=OWNER)
        code = asked["code"]
        assert asked["status"] == "asked" and code and code not in asked["text"] and "own yes" in asked["text"]
        row = fx.store.get_by_ask_code(code)
        assert row.type == "grant_confirm" and row.status == "asked" and row.context["grant"]["id"] == "renew-certs"
        assert "renew_certificates" in row.context["text"] and row.context["notice"] is True   # shown in full
        again = fx.mind.request_grant(CERTS, reason="same again", contact_id=OWNER)
        assert again["code"] == code                                              # one ask per grant
        assert fx.mind.authority.covering_grant(renew()) is None                  # not active yet
        assert (await owner_answers(fx.mind, code, contact_id=CONTACT)).status == "asked"   # not the owner's
        with pytest.raises(PermissionError):
            await fx.mind.answer(code, yes=True)                                   # nor the console's
        assert fx.mind.authority.covering_grant(renew()) is None
        await owner_answers(fx.mind, code, "sure, go ahead")
        grant = fx.mind.authority.covering_grant(renew())
        assert grant is not None and grant.source == f"ask {code}"
        shown = {item["id"]: item for item in fx.mind.state()["grants"]}["renew-certs"]
        assert shown["source"] == f"ask {code}" and shown["valid"] and not shown["revoked"]
        with pytest.raises(ValueError):
            fx.mind.request_grant(CERTS, reason="once more", contact_id=OWNER)    # an active id: revoke it first
        # a no drops the grant it asked about
        denied = fx.mind.request_grant({**PAY}, reason="pay small invoices", contact_id=OWNER)
        await owner_answers(fx.mind, denied["code"], "no")
        assert "small-payments" not in {item["id"] for item in fx.mind.state()["grants"]}
    finally:
        fx.store.close()


async def test_a_revoke_is_immediate_and_a_revoked_configured_grant_stays_revoked(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fx = Fixture(tmp_path, autonomy="standard", config={"grants": [CERTS], "actions": ACTIONS})
    try:
        task = _task(fx)
        assert (await fx.mind.guard(tool=RENEW[0], args=RENEW[1], run="mind", task_id=task.hermes_ref))["allow"]
        with pytest.raises(PermissionError):
            fx.mind.revoke_grant("renew-certs", contact_id=CONTACT)
        assert fx.mind.revoke_grant("renew-certs", contact_id=OWNER)["status"] == "revoked"
        held = await fx.mind.guard(tool=RENEW[0], args=RENEW[1], run="mind", task_id=task.hermes_ref)
        assert held["held"]                                                        # at once
        with pytest.raises(KeyError):
            fx.mind.revoke_grant("renew-certs")
        fx.restart()
        assert {item["id"]: item for item in fx.mind.state()["grants"]}["renew-certs"]["revoked"] is True
        assert fx.mind.authority.covering_grant(renew()) is None
        # the owner's own yes makes a grant they asked for active, and the console's revoke deletes it
        added = fx.mind.request_grant(PAY, reason="pay the small invoices", contact_id=OWNER)
        await owner_answers(fx.mind, added["code"])
        assert fx.mind.authority.covering_grant(paid(10))
        fx.mind.revoke_grant("small-payments", by="cli")
        assert "small-payments" not in {item["id"] for item in fx.mind.state()["grants"]}
    finally:
        fx.store.close()


async def test_a_change_to_the_configured_grants_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fx = Fixture(tmp_path, autonomy="standard", config={"grants": [CERTS]})
    try:
        first = fx.mind.state()["grants_changed"]
        assert first == "standing grants changed: added renew-certs"
        fx.restart()
        assert fx.mind.state()["grants_changed"] == first                          # unchanged: nothing new
        fx.config = {**fx.config, "grants": [{**CERTS, "per_day": 5}, PAY]}
        fx.restart()
        assert fx.mind.state()["grants_changed"] == "standing grants changed: added small-payments; changed renew-certs"
        fx.config = {**fx.config, "grants": [PAY, {"id": "broken", "class": "nope", "scope": "x"}]}
        fx.restart()
        changed = fx.mind.state()["grants_changed"]
        assert "removed renew-certs" in changed and "added broken" in changed
        broken = {item["id"]: item for item in fx.mind.state()["grants"]}["broken"]
        assert broken["valid"] is False and broken["active"] is False and broken["problem"]
        notices = [row for row in fx.store.intentions(kind=["message"]) if row.type == "grants_changed"]
        assert len(notices) == 3 and all(row.entity_id == OWNER for row in notices)
    finally:
        fx.store.close()


# ---------------------------------------------------------------------------
# The routes and the CLI
# ---------------------------------------------------------------------------

async def test_the_grant_routes(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fx = Fixture(tmp_path, autonomy="standard", config={"grants": [CERTS]})
    try:
        app = fx.app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            listed = (await client.get("/v1/mind/grants", headers=AUTH)).json()
            assert [item["id"] for item in listed["grants"]] == ["renew-certs"] and "renew-certs" in listed["text"]
            guest = (await client.get("/v1/mind/grants", headers=AUTH, params={"viewer": CONTACT})).json()
            assert guest["grants"] == []
            refused = await client.post("/v1/mind/grants", headers=AUTH, json={
                "grant": PAY, "reason": "pay", "by": "owner", "contact_id": CONTACT})
            assert refused.status_code == 403
            invalid = await client.post("/v1/mind/grants", headers=AUTH, json={
                "grant": {"id": "x", "class": "nope", "scope": "y"}, "reason": "y", "by": "owner",
                "contact_id": OWNER})
            assert invalid.status_code == 422 and "class is one of" in invalid.json()["detail"]["message"]
            asked = (await client.post("/v1/mind/grants", headers=AUTH, json={
                "grant": PAY, "reason": "pay the small ones", "by": "owner", "contact_id": OWNER})).json()
            assert asked["status"] == "asked" and asked["code"]
            operator = await client.post("/v1/mind/grants", headers=AUTH, json={
                "grant": {**PAY, "id": "cli-pay"}, "by": "cli"})
            assert operator.status_code == 403 and operator.json()["detail"]["code"] == "approve_required"
            await owner_answers(fx.mind, asked["code"])              # the owner's yes, in process
            assert (await client.post("/v1/mind/grants/nope/revoke", headers=AUTH, json={})).status_code == 404
            revoked = await client.post(f"/v1/mind/grants/{PAY['id']}/revoke", headers=AUTH, json={"by": "cli"})
            assert revoked.status_code == 200 and revoked.json()["status"] == "revoked"
            origin = (await client.get("/v1/mind/cron-origin", headers=AUTH, params={"job_id": "j-owner"})).json()
            assert origin["mind"] is False and origin["origin"] == "owner"
            recorded = await client.post("/v1/mind/cron-origin", headers=AUTH, json={
                "job_id": "j-mind", "task_id": "kanban-9", "tool_call_id": "c-1", "action": "create", "ok": True})
            assert recorded.status_code == 200
            origin = (await client.get("/v1/mind/cron-origin", headers=AUTH, params={"job_id": "j-mind"})).json()
            assert origin["mind"] is True and origin["task_id"] == "kanban-9"
            lineage = await client.post("/v1/mind/lineage", headers=AUTH, json={"child": "k-2", "parent": "k-1"})
            assert lineage.status_code == 200 and lineage.json()["ok"] is True
            state = (await client.get("/v1/mind/state", headers=AUTH)).json()
            assert {item["id"] for item in state["grants"]} == {"renew-certs"}
    finally:
        fx.store.close()


async def test_no_request_makes_a_grant_active_at_once(tmp_path, monkeypatch):
    """A standing grant is never active at once through the API: from the owner's conversation it is their ask
    (``by: cli`` beside it changes nothing); naming no conversation cites no owner turn and is refused
    ``approve_required``, the instance key's own call included; a credential without the owner grant reaches no
    grant route; with no credential behind it (a router mounted without the middleware) nothing is added, and the
    grant, wish and learning reads show no owner view."""
    from fastapi import FastAPI
    from colonymind.api.credentials import CredentialTable, secret_digest
    from colonymind.api.middleware import ApiKeyMiddleware
    from colonymind.api.routers import mind as mind_router
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    manager = "manager-credential-secret-0123456789"
    fx = Fixture(tmp_path, autonomy="standard", config={"grants": [CERTS]})
    try:
        app = FastAPI()
        app.add_middleware(ApiKeyMiddleware, api_key=AUTH["Authorization"].split()[1],
                           credentials=CredentialTable.from_config({
                               "api-key": {"owner": ["whatsapp", "sms", "rcs", "voice"], "contacts": "all"},
                               "manager": {"sha256": secret_digest(manager), "owner": [], "contacts": "all",
                                           "access": ["read", "write", "manage"]}}))
        app.include_router(mind_router.router)
        mind_router.set_mind(fx.mind)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            for index, conversation in enumerate(({"contact_id": OWNER}, {"session_id": "chat-1"},
                                                  {"message": "you may pay the small ones"})):
                claimed = (await client.post("/v1/mind/grants", headers=AUTH, json={
                    "grant": {**PAY, "id": f"pay-{index}"}, "reason": "pay the small ones", "by": "cli",
                    **conversation})).json()
                assert claimed["status"] == "asked" and claimed["code"], conversation
            refused = await client.post("/v1/mind/grants", headers={"Authorization": "Bearer " + manager},
                                        json={"grant": {**PAY, "id": "pay-manager"}, "by": "cli"})
            assert refused.status_code == 403 and refused.json()["detail"]["code"] == "not_owner"
            operator = await client.post("/v1/mind/grants", headers=AUTH, json={"grant": {**PAY, "id": "pay-operator"}})
            assert operator.status_code == 403 and operator.json()["detail"]["code"] == "approve_required"
        unkeyed = FastAPI()
        unkeyed.include_router(mind_router.router)
        async with AsyncClient(transport=ASGITransport(app=unkeyed), base_url="http://test") as client:
            bare = await client.post("/v1/mind/grants", json={
                "grant": {**PAY, "id": "pay-bare"}, "reason": "pay", "by": "cli"})
            assert bare.status_code == 403 and bare.json()["detail"]["code"] == "approve_required"
            assert (await client.get("/v1/mind/grants")).json()["grants"] == []
            assert (await client.get("/v1/mind/wishes")).json()["wishes"] == []
            assert (await client.get("/v1/mind/learning")).json()["log"] == []
        active = {item["id"] for item in fx.mind.grants_state()["grants"] if item.get("valid") and not item.get("revoked")}
        assert not active & {"pay-0", "pay-1", "pay-2", "pay-manager", "pay-operator", "pay-bare"}
    finally:
        mind_router.set_mind(None)
        fx.store.close()


def _parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default=None)
    sub = parser.add_subparsers(dest="command")
    mind_cli.add_parser(sub)
    return parser.parse_args(argv)


def test_the_cli_lists_and_revokes(tmp_path, monkeypatch, capsys):
    from colonymind.config import DEFAULTS, save_config
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("COLONYMIND_API_KEY", raising=False)
    save_config({**DEFAULTS, "owner": {"contact_id": OWNER}}, tmp_path)
    (tmp_path / "api.key").write_text("k\n")
    calls = []
    real = httpx.Client

    def client(**kwargs):
        def handler(request):
            body = json.loads(request.content.decode()) if request.content else None
            calls.append((request.method, request.url.path, body))
            return httpx.Response(200, json={"grants": [], "text": "done", "status": "ok"})
        return real(transport=httpx.MockTransport(handler), timeout=kwargs.get("timeout", 5))
    monkeypatch.setattr(httpx, "Client", client)
    assert mind_cli.run(_parse(["mind", "grants"])) == 0
    assert mind_cli.run(_parse(["mind", "grants", "add", "renew-certs", "--class", "credential_change", "--scope",
                                "renew TLS certificates", "--tools", "renew_certificates", "--match", "example.org",
                                "--per-day", "2"])) == 1           # adds nothing: test_approve_required.py
    assert mind_cli.run(_parse(["mind", "grants", "revoke", "renew-certs"])) == 0
    assert mind_cli.run(_parse(["mind", "grants", "revoke"])) == 2              # the id is required
    assert [(method, path) for method, path, _ in calls] == [
        ("GET", "/v1/mind/grants"), ("POST", "/v1/mind/grants/renew-certs/revoke")]
    assert calls[1][2] == {"by": "cli"}
    assert "done" in capsys.readouterr().out


def test_every_grant_use_is_a_note_with_its_amount(tmp_path):
    store = InitiativeStore(state_dir=tmp_path)
    try:
        guard = Authority(Policy.from_config({"autonomy": "standard", "grants": [PAY], "floor": FLOOR}), store,
                          owner_id=OWNER)
        row = guard.record_grant_use("small-payments", paid(12.5), task="i-1")
        assert row.kind == "note" and row.type == "grant_use" and row.status == "done"
        assert row.context == {**row.context, "grant": "small-payments", "amount": 12.5, "currency": "USD",
                               "task": "i-1"}
        assert "small-payments" in row.description
        later = guard.clock() + timedelta(days=31)
        assert guard.grant_usage("small-payments", later) == {"uses_24h": 0, "amount_30d": 0.0}
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Round 1 of the adversarial review
# ---------------------------------------------------------------------------

def test_a_limit_is_a_finite_number():
    """NaN and infinity compare false with every amount, so a limit written as either would limit nothing: such a
    grant is malformed. An effect whose amount is not a finite number is an unknown amount."""
    for value in (float("nan"), float("inf"), float("-inf"), "nan", "inf", "1e999", "NaN"):
        for key in ("max_amount", "per_month_amount"):
            grant = parse_grant({**PAY, key: value})
            assert not grant.valid and key in grant.problem, (key, value)
        assert not parse_grant({**CERTS, "per_day": value}).valid, value
    guard = Authority(Policy.from_config({"autonomy": "standard", "grants": [PAY]}), None, owner_id=OWNER)
    for amount in (float("nan"), float("inf"), -1.0):
        assert guard.covering_grant(Effect("money_movement", "pay_invoice", amount=amount, currency="USD")) is None


async def test_a_grant_asked_for_is_not_activated_by_a_yes_after_the_ask_expired(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fx = Fixture(tmp_path, autonomy="standard")
    try:
        asked = fx.mind.request_grant(CERTS, reason="you can renew the certificates yourself", contact_id=OWNER)
        fx.shift(hours=73)
        late = await owner_answers(fx.mind, asked["code"])
        assert late.status == "expired"
        assert fx.mind.authority.covering_grant(renew()) is None
        assert "renew-certs" not in {item["id"] for item in fx.mind.state()["grants"]}
    finally:
        fx.store.close()


async def test_a_configured_grants_id_stays_its_own(tmp_path, monkeypatch):
    """A revoked configured grant keeps its id: a grant added under that id is refused, not reported active and
    hidden. A grant confirmed before the configuration took its id is shown, inactive, never silently dropped."""
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fx = Fixture(tmp_path, autonomy="standard", config={"grants": [CERTS]})
    try:
        fx.mind.revoke_grant("renew-certs", contact_id=OWNER)
        with pytest.raises(ValueError, match="configured"):
            fx.mind.request_grant({**CERTS, "scope": "renew again"}, reason="renew them", contact_id=OWNER)
        assert fx.mind.grants() == []
        await owner_answers(fx.mind, fx.mind.request_grant(PAY, reason="pay them", contact_id=OWNER)["code"])
        fx.config = {**fx.config, "grants": [CERTS, {**PAY, "max_amount": 20}]}
        fx.restart()
        listed = [item for item in fx.mind.state()["grants"] if item["id"] == "small-payments"]
        assert sorted(item["source"].split()[0] for item in listed) == ["ask", "config"]
        shadowed, = [item for item in listed if item["source"].startswith("ask")]
        assert shadowed["active"] is False and "configured" in shadowed["problem"]
        active, = [grant for grant in fx.mind.grants() if grant.id == "small-payments"]
        assert active.source == "config" and active.max_amount == 20
    finally:
        fx.store.close()

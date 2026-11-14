"""The owner's view of the mind is decided by the request's credential, never by what the request asserts.

``viewer``, ``contact_id`` and ``by`` may narrow a view to a guest's; they never widen one to the owner's.
Only a credential granted the owner sees the owner's state, log, asks, lessons and opinions; development
mode (no key) and a credential without the owner grant get the non-owner view, however they name the owner
or claim to be the CLI (docs/CREDENTIALS.md).
"""

from __future__ import annotations

from secrets import token_hex

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from onekey import KEY
from colonymind.api.credentials import CredentialTable, secret_digest
from colonymind.api.errors import install_exception_handlers
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import host
from colonymind.api.routers import mind as mind_router
from colonymind.api.routers import opinions as opinions_router
from colonymind.initiatives.store import InitiativeStore
from colonymind.turns.idempotency import get_turn_idempotency_ledger

OWNER = "cid-owner"
OPS_SECRET = token_hex(24)
NOW = datetime(2027, 3, 1, 12, 0, tzinfo=timezone.utc)
PRIVATE = "the owner's private plan"


class Lessons:
    enabled = True

    def tally(self):
        return {}

    def all(self, include_closed=False):
        return [SimpleNamespace(id="l-1", status="active", as_dict=lambda: {
            "id": "l-1", "status": "active", "kind": "rule", "verified": "verified", "title": PRIVATE,
            "when_to_use": "always"})]

    def uses(self):
        return []


class Opinions:
    enabled = True

    def __init__(self, ledger):
        self.store = self
        self.ledger = ledger
        self.owner_id = OWNER
        self.rows = [{"id": 1, "audience": "owner", "text": PRIVATE, "topic": None, "premises": []},
                     {"id": 2, "audience": "all", "text": "tides are fascinating", "topic": None, "premises": []}]

    def revisions(self, history=False, audience=None, limit=10):
        return [row for row in self.rows if audience is None or row["audience"] == audience]

    def relevant(self, q, audience=None, limit=10):
        return self.revisions(audience=audience)

    def get(self, opinion_id):
        return next((row for row in self.rows if row["id"] == opinion_id), None)


@pytest.fixture
def mind(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    store = InitiativeStore(state_dir=tmp_path)
    row, _ = store.create_intention(kind="task", type="research", title=PRIVATE, drive="curiosity", cls="internal",
                                    decision="act", decision_reason="test", status="dispatched", dedup_key=None,
                                    recipient=None, created_at=NOW)
    stand_in = SimpleNamespace(
        store=store, clock=lambda: NOW, enabled=True, owner_id=OWNER, lessons=Lessons(),
        opinions=Opinions(get_turn_idempotency_ledger(tmp_path)),
        state=lambda: {"enabled": True, "autonomy": "standard", "working_on": PRIVATE},
        asks=lambda: [{"ask_code": "AB12", "title": PRIVATE, "decision_reason": "r", "expires_at": "later"}])
    mind_router.set_mind(stand_in)
    monkeypatch.setattr(host, "_goals_store", None, raising=False)
    try:
        yield SimpleNamespace(row=row.id)
    finally:
        mind_router.set_mind(None)
        store.close()


def get(path: str, *, secret: str | None, params: dict | None = None):
    """A GET behind the middleware: with ``secret``, or keyless from loopback (development mode)."""
    table = CredentialTable.from_config({"ops": {"sha256": secret_digest(OPS_SECRET), "owner": [], "contacts": [],
                                                 "access": ["manage"]}})
    app = FastAPI()
    install_exception_handlers(app)
    for router in (host.router, opinions_router.router, mind_router.router):
        app.include_router(router)
    app.add_middleware(ApiKeyMiddleware, api_key=KEY if secret else None, credentials=table)

    async def run():
        transport = ASGITransport(app=app, client=("127.0.0.1", 50000))
        async with AsyncClient(transport=transport, base_url="http://127.0.0.1:7777") as client:
            headers = {"Authorization": "Bearer " + secret} if secret else {}
            return await client.get(path, params=params, headers=headers)
    return asyncio.run(run())


def post(path: str, body: dict, *, secret: str):
    table = CredentialTable.from_config({"ops": {"sha256": secret_digest(OPS_SECRET), "owner": [], "contacts": [],
                                                 "access": ["manage"]}})
    app = FastAPI()
    install_exception_handlers(app)
    for router in (host.router, opinions_router.router, mind_router.router):
        app.include_router(router)
    app.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=table)

    async def run():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.post(path, json=body, headers={"Authorization": "Bearer " + secret})
    return asyncio.run(run())


ASSERTIONS = [{}, {"viewer": OWNER}, {"viewer": ""}, {"by": "cli"}, {"contact_id": ""}]


@pytest.mark.parametrize("secret", [None, OPS_SECRET], ids=["development", "no-owner-grant"])
def test_without_the_owner_grant_every_read_is_the_non_owner_view(mind, secret):
    for params in ASSERTIONS:
        for path in ("/v1/mind/state", "/v1/mind/status"):
            response = get(path, secret=secret, params=params)
            value = response.json()
            assert response.status_code == 200 and value == {"enabled": True, "autonomy": "standard"}
        for path in ("/v1/mind/log", "/v1/mind/asks", "/v1/mind/lessons", "/v1/mind/opinions"):
            response = get(path, secret=secret, params=params)
            assert response.status_code == 200, (path, response.text)
            assert PRIVATE not in response.text, (path, params)
        for path in (f"/v1/mind/why/{mind.row}", f"/v1/mind/log/{mind.row}", "/v1/mind/opinions/1"):
            assert get(path, secret=secret, params=params).status_code == 404, (path, params)
        assert "tides are fascinating" in get("/v1/mind/opinions", secret=secret, params=params).text
    # Naming the owner as the contact is refused outright (a query selector beyond its grant).
    for path in ("/v1/mind/state", "/v1/mind/opinions", "/v1/mind/opinions/1"):
        response = get(path, secret=secret, params={"contact_id": OWNER})
        assert response.status_code == 403 and PRIVATE not in response.text, path


def test_without_the_owner_grant_nothing_else_of_the_mind_is_reached(mind):
    """Development mode reads nothing else of the owner's side; a credential that reaches the rest of the API
    but is not granted the owner gets nothing else of the mind, and answers no ask."""
    for path in ("/v1/mind/concerns", "/v1/mind/stats", "/v1/host/goals"):
        response = get(path, secret=None)
        assert response.status_code == 403 and response.json()["detail"]["code"] == "route_not_granted", path
    for path in ("/v1/mind/concerns", "/v1/mind/stats", "/v1/mind/goals", "/v1/mind/narrative", "/v1/mind/outbox"):
        response = get(path, secret=OPS_SECRET)
        assert response.status_code == 403 and response.json()["detail"]["code"] == "not_owner", path
    assert get("/v1/host/goals", secret=OPS_SECRET).status_code != 403       # the rest of the API it has
    for path, body in (("/v1/mind/asks/AB12/no", {"by": "cli"}),
                       ("/v1/mind/asks/AB12/yes", {"by": "cli"}), ("/v1/mind/off", {"by": "cli"}),
                       ("/v1/mind/opinions/1/withdraw", {"reason": "no", "by": "cli"})):
        assert post(path, body, secret=OPS_SECRET).status_code == 403, path


def test_the_owner_grant_sees_the_owners_view_and_a_viewer_narrows_it(mind):
    """Not vacuous: the instance key, granted the owner, sees it all; a guest viewer narrows it."""
    assert PRIVATE in get("/v1/mind/state", secret=KEY).text
    for path in ("/v1/mind/log", "/v1/mind/asks", "/v1/mind/lessons", "/v1/mind/opinions",
                 f"/v1/mind/why/{mind.row}", "/v1/mind/opinions/1"):
        response = get(path, secret=KEY)
        assert response.status_code == 200 and PRIVATE in response.text, path
    for path in ("/v1/mind/state", "/v1/mind/log", "/v1/mind/asks", "/v1/mind/lessons"):
        assert PRIVATE not in get(path, secret=KEY, params={"viewer": "cid-guest"}).text, path
    for params in ({"contact_id": "cid-guest"}, {"contact_id": ""}):     # a guest's or a worker's session
        assert PRIVATE not in get("/v1/mind/opinions", secret=KEY, params=params).text, params

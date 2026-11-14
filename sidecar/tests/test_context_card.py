"""The voice card gate, with synthetic owner data and in-process ASGI only."""
from dataclasses import replace
from types import SimpleNamespace
import copy
import stat
import threading

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.api.credentials import CredentialTable, route_operation, secret_digest
from colonymind.api.legacy_scopes import required_scope
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers.context_card import router
from colonymind.context_card.builder import CardInputs, build, conservative_tokens, fingerprint
from colonymind.context_card.inputs import InputCollector
from colonymind.context_card.refresh import CardService, RevisionProbe
from colonymind.context_card.store import CardStore
from colonymind.router.tokens import estimate_tokens
from colonymind.turns import get_turn_idempotency_ledger

OWNER = "cid-synthetic-owner"
DAY = "2026-10-14"
ROUTE = "/v1/host/context/card"


def sparse():
    return CardInputs(OWNER, "Robin", DAY)


def busy():
    return replace(sparse(), preferences="Use concise replies and plain words. " * 150,
        narrative="I help connect ongoing work across channels. " * 150,
        work=tuple(f"Task {index}: review the synthetic greenhouse plan before Friday." for index in range(120)),
        highlights=tuple(f"voice: USER: I chose basil for bed {index}. ASSISTANT: I noted the choice." for index in range(120)),
        digest="Daily digest. Waiting for your OK: approve the synthetic seed order. " * 150)


@pytest.mark.parametrize("fixture", [sparse, busy])
@pytest.mark.parametrize("ratio", [None, "3", "4", "5"])
def test_token_gate_sparse_and_busy(fixture, ratio, monkeypatch, record_property):
    if ratio is None:
        monkeypatch.delenv("COLONYMIND_CONTEXT_CHARS_PER_TOKEN", raising=False)
    else:
        monkeypatch.setenv("COLONYMIND_CONTEXT_CHARS_PER_TOKEN", ratio)
    inputs = fixture()
    card = build(inputs)
    # Review fix 1: the 400 floor is a target reached only with supported
    # evidence. A sparse card has none to add, so it stays below the floor
    # instead of being padded; every section shows its missing-evidence line.
    assert card.token_count == conservative_tokens(card.text) == estimate_tokens(card.text)
    if fixture is busy:
        assert 400 <= card.token_count <= 800
    else:
        assert 0 < card.token_count < 400
        assert card.text.count("This private snapshot") == 1
        assert all(line in card.text for line in ("No supported preferences in this snapshot.",
            "No available self-narrative.", "No available work observations.",
            "No available recent channel highlights.", "No available owner-visible daily digest."))
    assert card == build(inputs)
    assert "Robin" in card.text and DAY in card.text
    assert all(label in card.text for label in ("Owner today", "In flight", "Cross-channel highlights"))
    record_property("card_tokens", card.token_count)


def test_dense_and_non_english_inputs_are_bounded(monkeypatch):
    monkeypatch.delenv("COLONYMIND_CONTEXT_CHARS_PER_TOKEN", raising=False)
    for payload in ("0123456789{}[] " * 1000, "你好植物🌱" * 1000):
        card = build(replace(busy(), owner_name=payload, preferences=payload, narrative=payload,
                             work=(payload,) * 8, highlights=(payload,) * 8, digest=payload))
        assert 400 <= card.token_count <= 800


class MutableCollector:
    def __init__(self, inputs):
        self.inputs, self.calls, self.fail, self.on_collect = inputs, 0, False, None
        self.thread = None

    def collect(self):
        self.calls += 1
        self.thread = threading.get_ident()
        if self.on_collect:
            self.on_collect()
        if self.fail:
            raise RuntimeError("synthetic failure")
        return self.inputs


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    monkeypatch.setenv("COLONYMIND_OWNER_PERSON_ID", OWNER)
    collector = MutableCollector(sparse())
    moment = [0]
    probe = RevisionProbe(tmp_path, day_provider=lambda: collector.inputs.local_day)
    store = CardStore(tmp_path / "context-card.db")
    result = CardService(store, collector, probe, clock=lambda: moment[0])
    result.test_time = moment
    yield result
    store.close()


def mark_change(service, leaf="identity.yaml"):
    path = service.probe.state_dir / leaf
    path.write_text(str(service.collector.calls) + "\n" + path.read_text() if path.exists() else "synthetic")


@pytest.mark.asyncio
async def test_cheap_unchanged_probe_and_real_change_versions(service, monkeypatch):
    from colonymind.context_card import refresh
    calls = []
    original = refresh.build
    monkeypatch.setattr(refresh, "build", lambda inputs: calls.append(inputs) or original(inputs))
    assert await service.refresh()
    first = service.fetch()
    assert first.version == 1
    assert service.collector.thread != threading.get_ident()
    for _ in range(10):
        assert not await service.refresh()
        assert service.fetch().text == first.text
    assert service.collector.calls == 1 and len(calls) == 1
    # Metadata changes trigger collection but not rendering or a new version.
    # Review fix 2: an ordinary input change serves the last card marked stale
    # instead of withholding it; reconciliation clears the stale flag.
    mark_change(service)
    pending = service.fetch()
    assert pending.stale and pending.version == 1 and pending.text == first.text
    assert not await service.refresh()
    assert service.collector.calls == 2 and len(calls) == 1
    assert service.fetch().version == 1 and not service.fetch().stale
    service.collector.inputs = replace(sparse(), preferences="Keep replies concise.")
    mark_change(service)
    assert await service.refresh()
    assert service.fetch().version == 2 and len(calls) == 2
    assert not await service.refresh()
    assert service.fetch().version == 2


@pytest.mark.asyncio
async def test_real_input_change_that_does_not_change_bytes_keeps_version(service):
    service.collector.inputs = replace(sparse(), digest="a" * 2400)
    await service.refresh()
    before = service.fetch()
    service.collector.inputs = replace(service.collector.inputs, digest="a" * 2399 + "b")
    assert fingerprint(service.collector.inputs) != before.input_fingerprint
    mark_change(service)
    assert not await service.refresh()
    assert service.fetch().text == before.text and service.fetch().version == before.version


@pytest.mark.asyncio
async def test_reconcile_in_memory_changes_and_local_day_rollover(service):
    await service.refresh()
    service.collector.inputs = replace(sparse(), preferences="Prefer flowing prose.")
    service.test_time[0] = 301
    assert await service.refresh()
    assert service.fetch().version == 2
    service.collector.inputs = replace(service.collector.inputs, local_day="2026-10-15")
    # Review fix 2: a day rollover is not an owner rebind or erasure, so the
    # previous card is served stale until the new day's card publishes.
    pending = service.fetch()
    assert pending.stale and pending.version == 2
    assert await service.refresh()
    assert not service.fetch().stale
    assert service.fetch().version == 3
    assert not await service.refresh()


@pytest.mark.asyncio
async def test_failed_refresh_does_not_serve_changed_sources(service):
    await service.refresh()
    service.collector.fail = True
    service.test_time[0] = 301
    assert not await service.refresh()
    assert service.fetch().stale and service.fetch().version == 1
    # Review fix 2: with an ordinary change and a failing collector the last
    # card stays served, always marked stale, never presented as fresh.
    mark_change(service)
    assert service.fetch().stale and service.fetch().version == 1
    assert not await service.refresh()
    assert service.fetch().stale and service.fetch().version == 1
    service.collector.fail = False
    assert not await service.refresh()
    assert service.fetch().version == 1 and not service.fetch().stale


@pytest.mark.asyncio
async def test_racing_source_write_is_coalesced_without_publication(service):
    service.collector.on_collect = lambda: mark_change(service)
    assert not await service.refresh()
    assert service.fetch() is None
    assert not service.store.conn.execute("SELECT * FROM context_card_versions").fetchall()
    service.collector.on_collect = None
    assert await service.refresh()
    assert service.fetch().version == 1


@pytest.mark.asyncio
async def test_restart_reconciles_then_reuses_immutable_version(service):
    await service.refresh()
    before = service.fetch()
    stored = service.store.conn.execute("SELECT payload FROM context_card_versions").fetchone()[0]
    other = CardStore(service.probe.state_dir / "context-card.db")
    try:
        restarted = CardService(other, service.collector, service.probe)
        assert restarted.fetch() is None
        assert not await restarted.refresh()
        assert restarted.fetch().version == before.version
        assert restarted.fetch().text == before.text
        assert other.conn.execute("SELECT payload FROM context_card_versions").fetchone()[0] == stored
    finally:
        other.close()


@pytest.mark.asyncio
async def test_owner_rebinding_withholds_and_purges_former_owner(service, monkeypatch):
    await service.refresh()
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", "cid-other-owner")
    monkeypatch.setenv("COLONYMIND_OWNER_PERSON_ID", "cid-other-owner")
    assert service.fetch() is None
    service.collector.inputs = replace(sparse(), owner_id="cid-other-owner", owner_name="Taylor")
    assert await service.refresh()
    assert service.fetch().owner_id == "cid-other-owner"
    assert service.fetch().version == 2
    assert service.store.conn.execute("SELECT DISTINCT owner_id FROM context_card_versions").fetchall()[0][0] == "cid-other-owner"


def test_wal_commits_and_database_replacement_change_the_cheap_probe(tmp_path, monkeypatch):
    import sqlite3
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    path = tmp_path / "mind.db"
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE fixture(value TEXT)")
        conn.commit()
        probe = RevisionProbe(tmp_path, owner_provider=lambda: OWNER, day_provider=lambda: DAY)
        before = probe()
        conn.execute("INSERT INTO fixture VALUES ('new')")
        conn.commit()
        assert probe() != before
        assert probe() == probe()
    finally:
        conn.close()
    before = probe()
    path.unlink()
    assert probe() != before


def card_app(service, *, api_key="owner-key"):
    app = FastAPI()
    app.include_router(router)
    app.state.context_card = service
    app.add_middleware(ApiKeyMiddleware, api_key=api_key, credentials=CredentialTable.from_config({
        "api-key": {"owner": ["voice"], "access": ["read"]},
        "voice-reader": {"sha256": secret_digest("owner-reader"), "owner": ["voice"], "contacts": "all", "access": ["read"]},
        "guest": {"sha256": secret_digest("guest-key"), "owner": [], "contacts": "own", "access": ["read"]},
        "owner-writer": {"sha256": secret_digest("writer-key"), "owner": ["voice"], "contacts": "all", "access": ["write"]},
    }))
    return app


@pytest.mark.asyncio
async def test_fetch_current_etag_conditional_and_no_hot_path_build(service, monkeypatch):
    await service.refresh()
    app = card_app(service)
    def forbidden(*args, **kwargs):
        raise AssertionError("fetch called a producer or model")
    from colonymind.context_card import refresh
    from colonymind.router.router import LLMRouter
    monkeypatch.setattr(service.collector, "collect", forbidden)
    monkeypatch.setattr(refresh, "build", forbidden)
    monkeypatch.setattr(LLMRouter, "route", forbidden)
    monkeypatch.setattr(LLMRouter, "complete", forbidden)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        response = await client.get(ROUTE, headers={"Authorization": "Bearer owner-reader"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["version"] == 1 and body["text"] == service.fetch().text
        # Review fix 1: the sparse card is not padded to the 400 floor.
        assert body["token_count"] == conservative_tokens(body["text"]) and 0 < body["token_count"] < 400
        assert body["readiness"] == "ready_for_prewarm"
        etag = response.headers["etag"]
        assert response.headers["x-context-card-version"] == "1"
        assert response.headers["cache-control"] == "private, no-cache"
        assert response.headers["vary"] == "Authorization"
        for conditional in (etag, etag.removeprefix("W/"), '"other", ' + etag, "*"):
            response = await client.get(ROUTE, headers={"Authorization": "Bearer owner-reader", "If-None-Match": conditional})
            assert response.status_code == 304 and response.content == b""
        assert service.collector.calls == 1
        # Review fix 2: an ordinary change keeps the same bytes served, flagged
        # stale, with no hot-path collection or build.
        mark_change(service)
        response = await client.get(ROUTE, headers={"Authorization": "Bearer owner-reader", "If-None-Match": etag})
        assert response.status_code == 304 and response.content == b""
        assert response.headers["etag"] == etag and response.headers["x-context-card-stale"] == "true"
        response = await client.get(ROUTE, headers={"Authorization": "Bearer owner-reader"})
        assert response.status_code == 200 and response.json()["stale"] is True
        assert response.json()["text"] == body["text"] and response.json()["version"] == 1
        assert service.collector.calls == 1
        # Withholding remains for an owner rebind: no bytes and no ETag.
        monkeypatch.setenv("COLONYMIND_OWNER_PERSON_ID", "cid-other-owner")
        monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", "cid-other-owner")
        response = await client.get(ROUTE, headers={"Authorization": "Bearer owner-reader", "If-None-Match": etag})
        assert response.status_code in (403, 503) and "etag" not in response.headers
        assert "Robin" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("token,expected", [(None, 401), ("guest-key", 403), ("writer-key", 403), ("invalid", 401)])
async def test_no_non_owner_or_wrong_operation_can_read_bytes_or_etag(service, token, expected):
    await service.refresh()
    app = card_app(service)
    headers = {"If-None-Match": service.fetch().etag}
    if token:
        headers["Authorization"] = "Bearer " + token
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://127.0.0.1") as client:
        response = await client.get(ROUTE + "?owner_id=" + OWNER, headers=headers)
        assert response.status_code == expected, response.text
        assert "Robin" not in response.text and "text" not in response.json()
        assert "etag" not in response.headers and "x-context-card-version" not in response.headers


@pytest.mark.asyncio
async def test_dev_mode_never_gets_owner_card(service):
    await service.refresh()
    async with AsyncClient(transport=ASGITransport(app=card_app(service, api_key=None)), base_url="http://127.0.0.1") as client:
        response = await client.get(ROUTE)
        assert response.status_code == 403
        assert "Robin" not in response.text


@pytest.mark.asyncio
async def test_unbuilt_card_returns_retryable_no_store_response(service):
    async with AsyncClient(transport=ASGITransport(app=card_app(service)), base_url="http://127.0.0.1") as client:
        response = await client.get(ROUTE, headers={"Authorization": "Bearer owner-reader"})
        assert response.status_code == 503
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["retry-after"] == "5"
        assert service.collector.calls == 0


def test_route_uses_existing_read_grants_and_scopes():
    assert route_operation("GET", ROUTE) == "read"
    assert required_scope("GET", ROUTE) == "context:read"


@pytest.mark.parametrize("settings", [{"poll_seconds": 0}, {"reconcile_seconds": True}, {"poll_seconds": float("nan")}])
def test_new_config_rejects_unbounded_or_invalid_worker_intervals(settings):
    from colonymind.config import DEFAULTS, validate, ConfigError
    data = copy.deepcopy(DEFAULTS)
    data["context_card"].update(settings)
    with pytest.raises(ConfigError):
        validate(data)


def test_staged_config_default_and_real_app_registration(tmp_path, monkeypatch):
    from colonymind.config import DEFAULTS
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    assert DEFAULTS["context_card"] == {"enabled": False, "poll_seconds": 5, "reconcile_seconds": 300}
    from colonymind.server import create_app
    assert ROUTE in create_app().openapi()["paths"]


@pytest.fixture
def real_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    monkeypatch.setenv("COLONYMIND_OWNER_PERSON_ID", OWNER)
    ledger = get_turn_idempotency_ledger(tmp_path)
    from colonymind.memory import recent
    # Fix the existing reader's as-of clock so synthetic occurrence times do
    # not depend on the hour at which this offline suite runs.
    monkeypatch.setattr(recent, "time", SimpleNamespace(time=lambda: 1792000800.0))
    host = SimpleNamespace(_preference_learner=None, _initiative_store=None, _comms_log=None)
    collector = InputCollector(tmp_path, host=host, mind_provider=lambda: None,
        owner_provider=lambda: OWNER, name_provider=lambda: "Robin", day_provider=lambda: DAY)
    return ledger, collector, host


def record(ledger, identifier, text, *, person=OWNER, channel="voice:synthetic", scope="person", messages=None):
    ledger.record_source(identifier, contact_id=person, session_id="synthetic-chat", scope=scope,
        messages=messages or [{"role": "user", "content": text}], occurred_at="2026-10-14T12:00:00+00:00",
        channel_id=channel, derive_claims=False)


def test_real_producers_cross_channel_and_private_scope(real_inputs):
    ledger, collector, host = real_inputs
    record(ledger, "voice-1", "Plant basil in the small bed.")
    record(ledger, "text-1", "The greenhouse inspection is on Friday.", channel="telegram:synthetic")
    record(ledger, "guest-1", "GUEST_PRIVATE", person="cid-guest")
    record(ledger, "session-1", "SESSION_PRIVATE", scope="session")
    host._preference_learner = SimpleNamespace(perspective=None, build_brief=lambda **_: "Keep replies concise.")
    inputs = collector.collect()
    text = build(inputs).text
    assert "Plant basil" in text and "greenhouse inspection" in text
    assert "voice" in text and "telegram" in text and "Keep replies concise" in text
    assert "GUEST_PRIVATE" not in text and "SESSION_PRIVATE" not in text
    assert set(inputs.source_refs) == {"voice-1", "text-1"}


def test_raw_pendant_record_cannot_enter_even_with_direct_channel_metadata(real_inputs):
    ledger, collector, _ = real_inputs
    block = {"type": "observed_conversation", "attributed_speaker": "owner", "epistemic_state": "derived_unverified",
        "source": "omi", "conversation_id": "synthetic-pendant", "speakers": [],
        "pieces": [{"speaker": "owner", "text": "OWNER_HEARD"}, {"speaker": "other", "text": "OTHER_VERBATIM_SECRET"}],
        "context_before": [], "part": 1, "parts": 1}
    # A forged/legacy locator must not flatten the observation into an excerpt.
    record(ledger, "pendant-1", "", messages=[{"role": "user", "_observed_conversation": "observed-conversation-v1", "content": [block]}])
    inputs = collector.collect()
    assert not inputs.highlights
    assert "OTHER_VERBATIM_SECRET" not in build(inputs).text


@pytest.mark.asyncio
async def test_lifecycle_is_opt_in_and_stops_worker_before_closing_store(tmp_path, monkeypatch):
    from colonymind.context_card import lifecycle
    from colonymind.config import Config
    settings = {"enabled": False, "poll_seconds": 5, "reconcile_seconds": 300}
    monkeypatch.setattr(lifecycle, "load_config", lambda: Config(home=tmp_path, data={"context_card": settings}))
    app = FastAPI()
    async with lifecycle.context_card_lifespan(app, state_dir=tmp_path):
        assert not (tmp_path / "context-card.db").exists()
    settings["enabled"] = True
    async with lifecycle.context_card_lifespan(app, state_dir=tmp_path):
        assert isinstance(app.state.context_card, CardService)
        assert stat.S_IMODE((tmp_path / "context-card.db").stat().st_mode) == 0o600
        store = app.state.context_card.store
    assert app.state.context_card is None
    import sqlite3
    with pytest.raises(sqlite3.ProgrammingError):
        store.conn.execute("SELECT 1")


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("owner_name", "Taylor"), ("preferences", "Prefer prose."),
    ("narrative", "I connect useful details across channels."), ("work", ("Review the seed order.",)),
    ("highlights", ("voice: USER: I chose basil.",)), ("digest", "Daily digest: seed order waiting.")])
async def test_every_card_input_changes_the_published_version(service, field, value):
    await service.refresh()
    service.collector.inputs = replace(sparse(), **{field: value})
    mark_change(service)
    assert await service.refresh()
    assert service.fetch().version == 2
    assert not await service.refresh()
    assert service.fetch().version == 2


@pytest.mark.asyncio
async def test_unrelated_erasure_revalidates_without_incrementing_unchanged_bytes(service):
    await service.refresh()
    first = service.fetch()
    service.collector.inputs = replace(sparse(), erasure_watermark=1)
    mark_change(service)
    assert not await service.refresh()
    assert service.fetch().text == first.text
    assert service.fetch().version == first.version
    assert service.fetch().erasure_watermark == 1
    assert len(service.store.conn.execute("SELECT * FROM context_card_versions").fetchall()) == 1


@pytest.mark.asyncio
async def test_owner_rebinding_across_restart_purges_prior_owner(service, monkeypatch):
    await service.refresh()
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", "cid-next-owner")
    monkeypatch.setenv("COLONYMIND_OWNER_PERSON_ID", "cid-next-owner")
    service.collector.inputs = replace(sparse(), owner_id="cid-next-owner", owner_name="Taylor")
    other = CardStore(service.probe.state_dir / "context-card.db")
    try:
        restarted = CardService(other, service.collector, service.probe)
        assert restarted.fetch() is None
        assert await restarted.refresh()
        assert restarted.fetch().version == 2
        assert other.conn.execute("SELECT DISTINCT owner_id FROM context_card_versions").fetchall()[0][0] == "cid-next-owner"
        assert len(other.conn.execute("SELECT * FROM context_card_versions").fetchall()) == 1
    finally:
        other.close()


@pytest.mark.asyncio
async def test_summaryless_canonical_capture_rebuilds_without_journal_event(real_inputs, tmp_path):
    ledger, collector, _ = real_inputs
    store = CardStore(tmp_path / "context-card.db")
    service = CardService(store, collector, RevisionProbe(tmp_path, day_provider=lambda: DAY))
    try:
        await service.refresh()
        if service.fetch() is None:
            await service.refresh()
        first = service.fetch()
        record(ledger, "summaryless-voice", "The synthetic seed order is ready for review.")
        assert not (tmp_path / "events" / ".cursor").exists()
        # Review fix 2: a ledger write (no erasure) serves the old card stale.
        pending = service.fetch()
        assert pending.stale and pending.version == first.version and pending.text == first.text
        assert await service.refresh()
        assert service.fetch().version == first.version + 1
        assert "seed order is ready" in service.fetch().text
        assert not await service.refresh()
    finally:
        store.close()


@pytest.mark.asyncio
async def test_version_history_retention_is_bounded_and_sequence_survives_purge(service):
    for index in range(12):
        service.collector.inputs = replace(sparse(), preferences=f"Synthetic preference {index}.")
        mark_change(service)
        assert await service.refresh()
    assert service.fetch().version == 12
    # Review fix 5: only the current version is retained.
    rows = service.store.conn.execute("SELECT version FROM context_card_versions").fetchall()
    assert [tuple(row) for row in rows] == [(12,)]
    service.store.purge()
    assert service.fetch() is None
    service.collector.inputs = replace(sparse(), preferences="A new synthetic preference.")
    mark_change(service)
    assert await service.refresh()
    assert service.fetch().version == 13


@pytest.mark.asyncio
async def test_same_bytes_keep_version_and_updated_estimate_across_restart(service, monkeypatch):
    monkeypatch.setenv("COLONYMIND_CONTEXT_CHARS_PER_TOKEN", "4")
    service.collector.inputs = replace(sparse(), preferences="A" * 160, narrative="B" * 160,
        work=("C" * 160,), highlights=("D" * 160,), digest="E" * 160)
    await service.refresh()
    first = service.fetch()
    monkeypatch.setenv("COLONYMIND_CONTEXT_CHARS_PER_TOKEN", "3")
    assert not await service.refresh()
    assert service.fetch().version == first.version
    assert service.fetch().token_count != first.token_count
    other = CardStore(service.probe.state_dir / "context-card.db")
    try:
        restarted = CardService(other, service.collector, service.probe)
        await restarted.refresh()
        assert restarted.fetch().token_count == estimate_tokens(restarted.fetch().text)
        assert restarted.fetch().version == first.version
    finally:
        other.close()


from colonymind.memory import curated
@pytest.mark.asyncio
async def test_real_erasure_withholds_before_get_then_removes_derived_versions(real_inputs, tmp_path):
    ledger, collector, _ = real_inputs
    record(ledger, 'erase-me', 'SYNTHETIC_ERASABLE greenhouse plan.')
    store = CardStore(tmp_path / 'context-card.db')
    service = CardService(store, collector, RevisionProbe(tmp_path, day_provider=lambda: DAY))
    try:
        await service.refresh()
        if service.fetch() is None:
            await service.refresh()
        before = service.fetch()
        assert before is not None and 'SYNTHETIC_ERASABLE' in before.text
        from test_curated_memory import historical_bytes
        _native_before_withdrawal = historical_bytes(ledger)
        curated.forget_sources(ledger, OWNER, source_ids=['erase-me'])
        assert historical_bytes(ledger) == _native_before_withdrawal
        assert service.fetch() is None
        async with AsyncClient(transport=ASGITransport(app=card_app(service)), base_url='http://127.0.0.1') as client:
            response = await client.get(ROUTE, headers={'Authorization': 'Bearer owner-reader', 'If-None-Match': before.etag})
            assert response.status_code == 503 and 'SYNTHETIC_ERASABLE' not in response.text
        await service.refresh()
        if service.fetch() is None:
            await service.refresh()
        assert 'SYNTHETIC_ERASABLE' not in service.fetch().text
        assert service.fetch().version == before.version + 1
        assert all(('SYNTHETIC_ERASABLE' not in row[0] for row in store.conn.execute('SELECT payload FROM context_card_versions')))
    finally:
        store.close()

def test_collector_reuses_owner_mind_and_delivered_digest(real_inputs, tmp_path):
    from datetime import datetime, timezone
    from colonymind.initiatives.store import InitiativeStore
    ledger, collector, host = real_inputs
    host._initiative_store = InitiativeStore(state_dir=tmp_path)
    try:
        sent, _ = host._initiative_store.create_intention(kind='message', type='digest', title='Synthetic digest', drive='upkeep', cls='owner', decision='act', decision_reason='synthetic', status='sent', dedup_key='synthetic-digest', recipient=OWNER, context={'text': 'Daily digest\nSeed order waiting.'}, created_at=datetime(2026, 10, 13, tzinfo=timezone.utc))
        host._initiative_store.create_intention(kind='message', type='digest', title='Synthetic pending digest', drive='upkeep', cls='owner', decision='act', decision_reason='synthetic', status='approved', dedup_key='synthetic-pending', recipient=OWNER, context={'text': 'Daily digest\nUNDELIVERED_PRIVATE'})
        collector.mind_provider = lambda: SimpleNamespace(owner_id=OWNER, narrative=lambda: {'enabled': True, 'text': 'I connect useful details.', 'cites': []}, section=lambda: 'On my mind: review the seed order.')
        inputs = collector.collect()
        assert inputs.narrative == 'I connect useful details.'
        assert 'review the seed order' in inputs.work[0]
        assert 'Seed order waiting' in inputs.digest and 'UNDELIVERED_PRIVATE' not in inputs.digest
        record(ledger, 'digest-erase-input', 'seed order')
        from test_curated_memory import historical_bytes
        _native_before_withdrawal = historical_bytes(ledger)
        curated.forget_sources(ledger, OWNER, source_ids=['digest-erase-input'])
        assert historical_bytes(ledger) == _native_before_withdrawal
        assert collector.collect().digest == ''
    finally:
        host._initiative_store.close()

def test_open_obligations_use_existing_store_and_drop_erased_lineage(real_inputs, tmp_path):
    from colonymind.commitments.store import CommitmentStore
    ledger, collector, host = real_inputs
    host._commitment_store = CommitmentStore(tmp_path / 'colonymind-commitments.db')
    record(ledger, 'obligation-input', 'Review the synthetic seed order.')
    host._commitment_store.create(OWNER, 'Review the synthetic seed order.', source_type='turn', metadata={'source_turn': 'obligation-input'})
    host._commitment_store.create('cid-guest', 'GUEST_PRIVATE_OBLIGATION')
    before = collector.collect()
    assert any(('Review the synthetic seed order' in line for line in before.work))
    assert 'obligation-input' in before.source_refs
    assert 'GUEST_PRIVATE_OBLIGATION' not in build(before).text
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger)
    curated.forget_sources(ledger, OWNER, source_ids=['obligation-input'])
    assert historical_bytes(ledger) == _native_before_withdrawal
    after = collector.collect()
    assert not any(('Review the synthetic seed order' in line for line in after.work))
    assert 'obligation-input' not in after.source_refs


async def test_refresh_failure_reports_type_code_and_counter(service, caplog):
    import sqlite3
    from colonymind.health_counters import snapshot

    def failed():
        error = sqlite3.OperationalError("private query text")
        error.sqlite_errorname = "SQLITE_BUSY"
        raise error

    service.collector.collect = failed
    before = snapshot().get("context_card_refresh_failed", 0)
    assert not await service.refresh()
    assert snapshot()["context_card_refresh_failed"] == before + 1
    assert "OperationalError" in caplog.text and "SQLITE_BUSY" in caplog.text
    assert "private query text" not in caplog.text

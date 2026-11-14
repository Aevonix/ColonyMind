"""Owner cross-channel continuity uses the erased-source-aware comms ledger."""
from datetime import datetime, timedelta, timezone
import importlib.util
import time
from uuid import uuid4

from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.api.routers import host
from colonymind.contacts.comms import CommsLog
from colonymind.contacts.continuity import recent_elsewhere
from colonymind.turns import TurnIdempotencyLedger
from colonymind.util import temporal
from test_context_packet import Row, app_with_key, assemble_body, wired
from onekey import KEY

needs_hermes = pytest.mark.skipif(importlib.util.find_spec("hermes_time") is None,
                                  reason="needs stock Hermes in the test interpreter")


def marker():
    return "marker-" + str(uuid4())


@pytest.fixture
def continuity(wired, tmp_path, monkeypatch):
    owner, guest = marker(), marker()
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", owner)
    monkeypatch.setenv("COLONYMIND_PERSONA_NAME", "Nova")
    monkeypatch.setenv("COLONYMIND_AGENT_TIMEZONE", "America/New_York")
    wired.rows = {owner: Row(owner, timezone="America/New_York", trust_tier="owner"),
                  guest: Row(guest)}
    monkeypatch.setattr(host, "_preference_learner", None)
    ledger = TurnIdempotencyLedger(tmp_path / "turn-idempotency.db")
    log = CommsLog(str(tmp_path / "communications.db"), source_ledger=ledger)
    monkeypatch.setattr(host, "_comms_log", log)
    now = datetime(2026, 10, 17, 14, 14, tzinfo=timezone.utc)
    # Shift the shared wall clock, so imports during this fixture retain the
    # production function rather than capturing a temporary frozen-clock lambda.
    monkeypatch.setattr(time, "time", lambda: now.timestamp())
    yield owner, guest, log, ledger, now
    log._conn.close()


async def brief(owner, channel="companion:owner"):
    async with AsyncClient(transport=ASGITransport(app=app_with_key()), base_url="http://test") as client:
        result = await client.get("/v1/host/context/temporal", params={
            "contact_id": owner, "channel": channel, "owner_private_turn": True},
            headers={"Authorization": "Bearer " + KEY})
    assert result.status_code == 200, result.text
    return result.json()


@pytest.mark.asyncio
async def test_content_order_window_cap_platform_and_no_handles(continuity, monkeypatch):
    owner, _, log, _, now = continuity
    monkeypatch.setenv("COLONYMIND_OWNER_NAME", "Robin")
    hidden, older_platform, too_old, system, automatic, receipt = [marker() for _ in range(6)]
    chat, other_chat = marker(), marker()
    log.log(owner, channel="whatsapp:" + chat, summary=too_old, ts=(now - timedelta(hours=72, seconds=1)).isoformat())
    log.log(owner, channel="telegram:" + marker(), summary=older_platform, ts=(now - timedelta(hours=5)).isoformat())
    messages = [marker() for _ in range(8)]
    for index, text in enumerate(messages):
        log.log(owner, channel=(" WhatsApp :" + chat if index % 2 else "whatsapp:" + other_chat),
                direction="out" if index % 2 else "in", summary=text + "x" * 400,
                ts=(now - timedelta(minutes=8-index)).isoformat())
    log.log(owner, channel="companion:" + marker(), summary=hidden, ts=now.isoformat())
    log.log(owner, channel="system:" + marker(), summary=system, ts=now.isoformat())
    log.log(owner, channel="whatsapp:" + chat, summary=automatic, external_ref="automated:" + marker(), ts=now.isoformat())
    log.log(owner, channel="whatsapp:" + chat, summary=receipt, receipt_ref=marker(), ts=now.isoformat())
    data = await brief(owner)
    text = data["continuity"]["body"]
    lines = text.splitlines()
    assert len(lines) == 6
    for index, line in enumerate(lines, start=2):
        assert line.startswith("WhatsApp, Sat Oct 17 10:")
        assert (": Robin: " if index % 2 == 0 else ": Nova: ") in line
        assert messages[index] in line and len(line.split(": ", 2)[-1]) == 280
    assert lines[-1].startswith("WhatsApp, Sat Oct 17 10:13 AM: Nova:")
    assert all(value not in text for value in [chat, other_chat, hidden, older_platform,
                                               too_old, system, automatic, receipt, *messages[:2]])
    assert data["owner_timezone"] == "America/New_York"
    assert data["body"].splitlines() == ["Nova's time: Saturday 10:14 AM EDT (America/New_York).",
        f"{owner}'s location: no recent report; assume EDT (America/New_York)."]


@pytest.mark.asyncio
async def test_normal_assembly_has_continuity_and_collapsed_landscape(continuity):
    owner, _, log, _, now = continuity
    handles = [marker(), marker(), marker()]
    text = marker()
    for index, handle in enumerate(handles):
        log.log(owner, channel="whatsapp:" + handle, direction="out", summary=text,
                ts=(now - timedelta(days=2-index)).isoformat())
    async with AsyncClient(transport=ASGITransport(app=app_with_key()), base_url="http://test") as client:
        body = assemble_body(owner, query=marker())
        body["context"]["channel_id"] = "companion:owner"
        body["context"]["metadata"] = {"owner_private_turn": True}
        result = await client.post("/v1/host/context/assemble", json=body,
                                  headers={"Authorization": "Bearer " + KEY})
    assert result.status_code == 200, result.text
    sections = {section["id"]: section for section in result.json()["sections"]}
    assert text in sections["colonymind-owner-continuity"]["body"]
    landscape = sections["colonymind-comms-landscape"]["body"]
    assert "Recorded channels: WhatsApp 2026-10-17." in landscape
    assert "Last recorded outgoing message: WhatsApp on 2026-10-17." in landscape
    assert all(handle not in str(sections) for handle in handles)


@pytest.mark.asyncio
async def test_non_owner_and_empty_or_same_platform_have_no_section(continuity):
    owner, guest, log, _, now = continuity
    for contact in (owner, guest):
        log.log(contact, channel="companion:" + marker(), summary=marker(), ts=now.isoformat())
    assert "continuity" not in await brief(owner)
    assert "continuity" not in await brief(guest, "whatsapp")
    section, zone = await host._owner_continuity(guest, "whatsapp")
    assert section is None and zone is None
    async with AsyncClient(transport=ASGITransport(app=app_with_key()), base_url="http://test") as client:
        body = assemble_body(guest, query=marker())
        body["context"]["channel_id"] = "whatsapp"
        result = await client.post("/v1/host/context/assemble", json=body,
                                  headers={"Authorization": "Bearer " + KEY})
    assert result.status_code == 200, result.text
    assert "colonymind-owner-continuity" not in str(result.json())


@pytest.mark.asyncio
async def test_greeting_read_avoids_global_heads_up_and_uses_read_only_worker(continuity, monkeypatch):
    import threading
    owner, _, log, _, now = continuity
    content = marker()
    log.log(owner, channel="whatsapp:" + marker(), summary=content, ts=now.isoformat())
    loop_thread = threading.get_ident()
    original = log.elsewhere_history
    def history(*args, **kwargs):
        assert threading.get_ident() != loop_thread
        assert log._conn.execute("PRAGMA query_only").fetchone()[0] == 1
        return original(*args, **kwargs)
    monkeypatch.setattr(log, "elsewhere_history", history)
    def forbidden(*args, **kwargs):
        pytest.fail("the cheap greeting read must not scan global heads-up producers")
    from types import SimpleNamespace
    monkeypatch.setattr(host, "_commitment_store", SimpleNamespace(get_overdue=forbidden))
    assert content in (await brief(owner))["continuity"]["body"]


@pytest.mark.asyncio
async def test_continuity_keeps_existing_turn_budget_and_deferred_writes(continuity):
    from colonymind.memory import turn_budget
    owner, _, log, _, now = continuity
    log.log(owner, channel="whatsapp:" + marker(), summary=marker(), ts=now.isoformat())
    with turn_budget.turn(turn_budget.TurnBudget(wait_s=1.5)) as budget:
        turn_budget.defer_write(lambda: None)
        before = list(budget.writes)
        section, _ = await host._owner_continuity(owner, "companion:owner", owner_private_turn=True)
        assert section is not None and turn_budget.current() is budget
        assert before and budget.writes == before


def test_window_boundary_other_platform_and_handle_redaction(monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_NAME", "Robin")
    now = datetime(2026, 10, 17, 14, 14, tzinfo=timezone.utc)
    handle, content = marker(), marker()
    rows = [{"channel": "whatsapp:" + handle, "direction": "in", "summary": content + " " + handle,
             "ts": (now - timedelta(hours=72)).isoformat()}]
    result = recent_elsewhere(rows, "COMPANION:owner", "America/New_York", now=now)
    assert content in result and handle not in result and "[handle]" in result
    assert "Wed Oct 14 10:14 AM: Robin:" in result
    assert not recent_elsewhere(rows, " WHATSAPP :" + marker(), "America/New_York", now=now)
    assert not recent_elsewhere(rows, "", "America/New_York", now=now)
    rows[0]["ts"] = (now + timedelta(seconds=1)).isoformat()
    assert not recent_elsewhere(rows, "companion", "America/New_York", now=now)


@pytest.mark.asyncio
@pytest.mark.parametrize("report_age", [None, 14 * 60, 24 * 3600 + 1])
@needs_hermes
async def test_greeting_hook_reads_real_sidecar_brief_without_recall(continuity, monkeypatch, report_age, tmp_path):
    import asyncio
    from types import SimpleNamespace
    from agent.subagent_lifecycle import bind_subagent_parent
    import colonymind_memory
    from colonymind_memory import provider as provider_module

    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path / "instance"))
    monkeypatch.setenv("COLONYMIND_OWNER_NAME", "Robin")
    owner, _, log, _, now = continuity
    contact = await host._contacts_store.get(owner)
    contact.display_name = marker()
    contact.timezone = "Asia/Tokyo"
    contact.location_city, contact.location_region, contact.location_country_code = "Testville", "Test Region", "JP"
    contact.location_reported_at = (now - timedelta(seconds=report_age)).isoformat() if report_age is not None else None
    monkeypatch.setenv("COLONYMIND_AGENT_TIMEZONE", "Pacific/Honolulu")
    text = marker()
    log.log(owner, channel="whatsapp:" + marker(), direction="in", summary=text,
            ts=(now - timedelta(hours=12, minutes=16)).isoformat())
    provider = provider_module.ColonyMindMemoryProvider(config={
        "url": "http://test", "api_key": KEY, "contact_id": owner})
    provider.initialize(marker(), platform="cli")
    monkeypatch.setattr(provider_module._ttime, "time", lambda: now.timestamp())
    monkeypatch.setattr(colonymind_memory, "ColonyMindMemoryProvider", type(provider))
    def no_recall(*args, **kwargs):
        pytest.fail("a greeting must not perform semantic recall")
    monkeypatch.setattr(provider, "_assemble", no_recall)
    app = app_with_key()
    async def get(url, **kwargs):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.get(url, **kwargs)
    def read(timeout, method, url, **kwargs):
        assert method == "get" and 0 < timeout <= 1.5
        return asyncio.run(get(url, **kwargs))
    monkeypatch.setattr(provider_module, "_request_within", read)
    hooks = []
    colonymind_memory.register(SimpleNamespace(register_memory_provider=lambda p: None,
                                            register_hook=lambda name, callback: hooks.append(callback)))
    assert provider_module.is_trivial_prompt("hi")  # Hermes' own skip predicate
    agent = SimpleNamespace(session_id=provider._session_id,
                            _memory_manager=SimpleNamespace(providers=[provider]))
    with bind_subagent_parent(agent):
        result = await asyncio.to_thread(hooks[0], user_message="hi", session_id=provider._session_id,
                                         platform="cli", sender_id="", turn_id=marker())
    assert result["context"].startswith("Nova's time: Saturday 4:14 AM HST (Pacific/Honolulu).\n")
    expected = (f"{contact.display_name}: Testville, Test Region (JP), local time 11:14 PM (Asia/Tokyo), reported 14 min ago."
                if report_age == 14 * 60 else
                f"{contact.display_name}'s location: no recent report; assume HST (Pacific/Honolulu).")
    assert expected in result["context"]
    # The normal packet renders exactly the same owner clock lines.
    body = assemble_body(owner, query=marker())
    body['context']['metadata'] = {'owner_private_turn': True}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assembled = await client.post("/v1/host/context/assemble", json=body,
                                      headers={"Authorization": "Bearer " + KEY})
    assert assembled.status_code == 200, assembled.text
    clock = next(s["body"] for s in assembled.json()["sections"] if s["id"] == "temporal-context")
    assert clock.splitlines()[:2] == result["context"].splitlines()[:2]
    assert "## Recent conversation elsewhere" in result["context"]
    # The existing explicit communication override still dates continuity in Honolulu.
    assert "WhatsApp, Fri Oct 16 3:58 PM: Robin: " + text in result["context"]


from colonymind.memory import curated
@pytest.mark.asyncio
async def test_erasure_removes_continuity_and_exposes_next_platform(continuity):
    owner, _, log, ledger, now = continuity
    source, text, retained = (marker(), marker(), marker())
    ledger.record_source(source, contact_id=owner, session_id=marker(), messages=[{'role': 'user', 'content': text}], derive_claims=False)
    lineage, _ = log.source_input(source, owner)
    log.log(owner, channel='whatsapp:' + marker(), summary=text, ts=now.isoformat(), source_lineage=lineage)
    log.log(owner, channel='telegram:' + marker(), summary=retained, ts=(now - timedelta(minutes=1)).isoformat())
    assert text in (await brief(owner))['continuity']['body']
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(ledger)
    curated.forget_sources(ledger, owner, source_ids=[source])
    assert historical_bytes(ledger) == _native_before_withdrawal
    after = await brief(owner)
    assert text not in str(after) and retained in after['continuity']['body']
    assert after['continuity']['body'].startswith('Telegram,')

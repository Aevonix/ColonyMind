"""Timeline endpoint over the event journal (v0.21.0)."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest


@pytest.fixture(autouse=True)
def _iso(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("COLONYMIND_EVENT_JOURNAL_DIR", raising=False)
    monkeypatch.setenv("COLONYMIND_AGENT_TIMEZONE", "UTC")
    yield


@pytest.mark.asyncio
async def test_timeline_endpoint_filters_and_digest(tmp_path):
    from colonymind.events.journal import append_event
    from colonymind.api.routers import host

    append_event("conversation.turn", {"contact_id": "cid-a", "summary": "talked about the roadmap"})
    append_event("outreach.sent", {"contact_id": "cid-b", "reason": "checked in"})
    append_event("conversation.turn", {"contact_id": "cid-a", "summary": "second chat"})

    resp = await host.get_timeline(since="24h", types=None, contact_id=None, limit=100)
    assert resp.count == 3
    assert resp.digest and ("💬" in resp.digest or "talked" in resp.digest)
    # newest first
    assert resp.events[0].seq > resp.events[-1].seq
    # humanized + bucketed
    assert resp.events[0].when == "just now" or resp.events[0].when.endswith("ago")
    assert resp.events[0].bucket == "today"

    # contact filter
    r2 = await host.get_timeline(since="24h", types=None, contact_id="cid-a", limit=100)
    assert r2.count == 2 and all(e.contact_id == "cid-a" for e in r2.events)

    # type filter
    r3 = await host.get_timeline(since="24h", types="outreach.sent", contact_id=None, limit=100)
    assert r3.count == 1 and r3.events[0].type == "outreach.sent"

    # narrow window excludes everything just-written? (1s window — written "now")
    r4 = await host.get_timeline(since="2026-01-01T00:00:00+00:00", types=None, contact_id=None, limit=100)
    assert r4.count == 3  # all after Jan 1


@pytest.fixture
def timeline_generation(tmp_path, monkeypatch):
    from colonymind.events import journal
    from colonymind.util import temporal

    now = datetime(2026, 10, 17, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(temporal, "now_utc", lambda: now)
    directory = tmp_path / "events"
    directory.mkdir()
    for seq in range(1, 621):
        data = {"summary": "A long café conversation\n" + "x" * 150}
        if seq % 3 == 0:
            data.update(contact_id="a", person_id="fallback")
        elif seq % 3 == 1:
            data.update(contact_id="", person_id="fallback")
        else:
            data.update(contact_id="b")
        raw = {"seq": seq, "ulid": f"event-{seq}",
               "type": "outreach.sent" if seq % 7 == 0 else "conversation.turn",
               "recordedAt": (now - timedelta(minutes=621 - seq)).isoformat(),
               "data": data}
        (directory / f"{seq:06d}.event-{seq}.json").write_text(json.dumps(raw))
    # Preserve identity/checksum validation, even when a duplicate straddles a page.
    (directory / "000620.duplicate.json").write_text("{}")
    (directory / "000619.event-619.json").write_text('{"checksum":"bad"}')
    (directory / "000618.event-618.json").write_text("not json")
    journal._write_cursor(directory, 620, 0)
    return directory


@pytest.mark.parametrize("since,types,contact_id,limit", [
    ("24h", None, None, 4),
    ("6h", "conversation.turn", "a", 4),
    ("24h", None, "fallback", 4),
    ("24h", "conversation.turn, outreach.sent, ", "b", 500),
    ("24h", None, "missing", 4),
    ("24h", None, "a", 500),
    ("24h", "outreach.sent", "a", 4),
    ("24h", "unknown", None, 4),
    ("1s", None, None, 4),
    ("10m", None, "fallback", 4),
    ("10m", None, None, 4),
    ("2026-10-17T11:56:00Z", None, None, 4),
])
async def test_timeline_response_bytes_equal_legacy_replay(
    timeline_generation, monkeypatch, since, types, contact_id, limit,
):
    from colonymind.api.routers import host
    from colonymind.events import journal

    contacts = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(
        display_name="Café contact", given_name=None)))
    monkeypatch.setattr(host, "_contacts_store", contacts)
    app = FastAPI()
    app.include_router(host.router)
    params = {"since": since, "limit": limit}
    if types is not None:
        params["types"] = types
    if contact_id is not None:
        params["contact_id"] = contact_id
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        actual = await client.get("/v1/host/timeline", params=params)

        def legacy_read(since, limit, types, contact_id):
            return journal.replay_events(since, limit=max(500, limit), types=types,
                                         newest_first=True)

        monkeypatch.setattr(journal, "timeline_events", legacy_read)
        expected = await client.get("/v1/host/timeline", params=params)
    assert actual.status_code == expected.status_code == 200
    assert actual.content == expected.content


def test_small_timeline_page_reads_only_one_extra_matching_payload(tmp_path, monkeypatch):
    from colonymind.events import journal

    directory = tmp_path / "events"
    directory.mkdir()
    for seq in range(1, 501):
        raw = {"seq": seq, "type": "conversation.turn", "data": {"contact_id": "a"}}
        (directory / f"{seq:06d}.event.json").write_text(json.dumps(raw))
    reads = []
    original = Path.read_text

    def read(path, *args, **kwargs):
        reads.append(path.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    result = journal.timeline_events("", limit=4, types=["conversation.turn"], contact_id="a")
    assert [e["seq"] for e in result["events"]] == [500, 499, 498, 497]
    assert result["hasMore"] is True
    assert len(reads) == 5


def test_contact_outside_legacy_window_keeps_global_overflow(tmp_path):
    from colonymind.events import journal

    directory = tmp_path / "events"
    directory.mkdir()
    for seq in range(1, 503):
        raw = {"seq": seq, "type": "conversation.turn",
               "data": {"contact_id": "a" if seq == 1 else "b"}}
        (directory / f"{seq:06d}.event.json").write_text(json.dumps(raw))
    assert journal.timeline_events("", limit=4, contact_id="a") == {
        "events": [], "hasMore": True,
    }

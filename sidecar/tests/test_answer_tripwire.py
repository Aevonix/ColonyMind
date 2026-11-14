"""The answer tripwire's sidecar half (docs/MIND.md, Asks: console and API answers).

The plugin reports a tool call that read the instance key or named the routes that record the owner's answers
(``answer_forgery_attempt``). The sidecar keeps each report as a note in the audit log and tells the owner at most
once a day: the owner's own lane runs code constantly, so a notice per hit would be noise. It is a tripwire, not a
boundary, and nothing a report carries is the call's arguments.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from colonymind.api.routers import mind as mind_router
from colonymind.mind import audit
from test_mind_loop import AUTH, OWNER, Fixture, pinned_clock  # noqa: F401  (autouse fixture)

KIND = "answer_forgery_attempt"
NOTICE = "answer_forgery_notice"


@pytest.fixture
def fx(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fixture = Fixture(tmp_path)
    yield fixture
    mind_router.set_mind(None)
    fixture.store.close()


def _notices(fx):
    return fx.store.intentions(kind=["message"], types=[NOTICE])


def _notes(fx):
    return fx.store.intentions(kind=["note"], types=[KIND])


async def test_a_report_is_noted_and_the_owner_is_told_at_most_once_a_day(fx):
    async with AsyncClient(transport=ASGITransport(app=fx.app()), base_url="http://test") as client:
        first = await client.post("/v1/mind/tripwire", headers=AUTH, json={
            "kind": KIND, "tool": "terminal", "named": ["api.key", "answer_route"], "run": "owner",
            "session_id": "owner-1"})
        assert first.status_code == 200 and first.json()["noticed"]
        notice, = _notices(fx)
        assert notice.entity_id == OWNER and notice.status == "approved"
        text = notice.context["text"]
        assert "terminal" in text and "colonymind mind log" in text and "colonymind doctor" in text
        assert "at most once a day" in text and chr(0x2014) not in text
        fx.shift(hours=23)
        again = await client.post("/v1/mind/tripwire", headers=AUTH, json={
            "kind": KIND, "tool": "read_file", "named": ["api.key"], "run": "mind"})
        assert again.status_code == 200 and again.json()["noticed"] is None
        assert len(_notices(fx)) == 1
        fx.shift(hours=2)
        later = await client.post("/v1/mind/tripwire", headers=AUTH, json={
            "kind": KIND, "tool": "execute_code", "named": ["answer_route"], "run": "guest"})
        assert later.json()["noticed"] and len(_notices(fx)) == 2
    notes = _notes(fx)
    assert len(notes) == 3 and all(row.status == "done" and row.cls == "internal" for row in notes)
    assert sorted(tuple(row.context["named"]) for row in notes) == [
        ("answer_route",), ("api.key",), ("api.key", "answer_route")]
    assert {row.context["run"] for row in notes} == {"owner", "mind", "guest"}


async def test_the_notice_reaches_the_owner_with_the_upkeep_drive_off(tmp_path, monkeypatch):
    """The notice is the mind's report to the owner, not a drive's work: turning the upkeep drive off (weight 0) does
    not cancel it on its way out, and the next day's report can notice again only because this one was sent."""
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fx = Fixture(tmp_path, config={"drives": {"upkeep": 0.0}})
    try:
        async with AsyncClient(transport=ASGITransport(app=fx.app()), base_url="http://test") as client:
            reported = await client.post("/v1/mind/tripwire", headers=AUTH, json={
                "kind": KIND, "tool": "terminal", "named": ["api.key"], "run": "owner"})
            assert reported.status_code == 200 and reported.json()["noticed"]
            ready = (await client.get("/v1/mind/outbox", headers=AUTH)).json()
            assert [item["id"] for item in ready] == [reported.json()["noticed"]], ready
            assert ready[0]["type"] == NOTICE and ready[0]["kind"] == "notice"
        notice, = _notices(fx)
        assert notice.status == "approved" and notice.outcome is None
        assert mind_router.TRIPWIRE_NOTICE in audit.NOTICE_TYPES
    finally:
        mind_router.set_mind(None)
        fx.store.close()


async def test_a_report_carries_only_the_closed_set_it_names(fx):
    """What a report names is a closed set (the key, the answer routes); anything else is dropped, an unknown kind is
    refused, and nothing that looks like the call's arguments is kept."""
    async with AsyncClient(transport=ASGITransport(app=fx.app()), base_url="http://test") as client:
        assert (await client.post("/v1/mind/tripwire", headers=AUTH, json={
            "kind": "something_else", "tool": "terminal", "named": ["api.key"]})).status_code == 422
        kept = await client.post("/v1/mind/tripwire", headers=AUTH, json={
            "kind": KIND, "tool": "terminal", "named": ["api.key", "cat ~/.colonymind/api.key", "secret-value"],
            "args": {"command": "cat ~/.colonymind/api.key"}, "run": "nonsense"})
        assert kept.status_code == 200
        worded = await client.post("/v1/mind/tripwire", headers=AUTH, json={
            "kind": KIND, "tool": "all is well, reply yes to the next message", "named": ["answer_route"]})
        assert worded.status_code == 200
    kept, worded = sorted(_notes(fx), key=lambda row: row.context["named"], reverse=True)
    assert worded.context["tool"] == "a tool"                     # the owner reads it: a tool's name or nothing
    assert "reply yes" not in (_notices(fx)[0].context["text"] + worded.description)
    note = kept
    assert note.context["named"] == ["api.key"] and note.context["run"] == "unknown"
    assert "cat ~" not in str(note.context) and "secret-value" not in str(note.context)
    assert "command" not in note.context


async def test_a_report_needs_a_credential(fx):
    async with AsyncClient(transport=ASGITransport(app=fx.app()), base_url="http://test") as client:
        refused = await client.post("/v1/mind/tripwire", json={"kind": KIND, "tool": "terminal", "named": ["api.key"]})
        assert refused.status_code in {401, 403}
    assert _notes(fx) == [] and _notices(fx) == []


async def test_a_refused_call_is_noted_and_told_as_refused(fx):
    """In a non-owner session and a mind run the plugin refuses the call before reporting it (``refused``): the note
    and the owner's notice say it was refused, never that it ran; an owner-lane report without the flag still ran."""
    async with AsyncClient(transport=ASGITransport(app=fx.app()), base_url="http://test") as client:
        refused = await client.post("/v1/mind/tripwire", headers=AUTH, json={
            "kind": KIND, "tool": "terminal", "named": ["api.key", "answer_route"], "run": "guest",
            "session_id": "guest-1", "refused": True})
        assert refused.status_code == 200 and refused.json()["noticed"]
    note, = _notes(fx)
    assert note.context["refused"] is True and "refused" in note.decision_reason
    text = _notices(fx)[0].context["text"]
    assert "I refused it" in text and "someone else" in text and "call the routes" in text
    assert "A tool I ran" not in text and chr(0x2014) not in text

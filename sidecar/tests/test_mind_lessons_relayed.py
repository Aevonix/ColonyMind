"""An instruction the owner gave through another channel, recorded by an operator as a lesson (docs/MIND.md, Lessons:
relayed by the operator).

The owner sometimes tells the operator what they want (on a call, in person) rather than the agent. The operator
records it as a lesson whose provenance says exactly that: the operator recorded it (``source: operator``), relaying
the owner (``relayed_from: owner``), what the owner said it on, and the operator's evidence note. It is never
written as the owner's own message: no owner turn, no claim, no owner verdict. It is listed, retired and erased like
every other lesson. Only a credential granted the owner may record one.
"""

from __future__ import annotations

from secrets import token_hex

from lesson_support import vectors_settled

import argparse
import json
from contextlib import closing
from datetime import timedelta

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from colonymind.api.credentials import CredentialTable, secret_digest
from colonymind.api.middleware import ApiKeyMiddleware
from colonymind.api.routers import mind as mind_router
from colonymind.mind import cli as mind_cli
from colonymind.mind.lessons import Lesson, lesson_id
from test_mind_cli import _Transport, home  # noqa: F401  (home: the CLI's instance fixture)
from test_mind_loop import AUTH, KEY, OWNER, Fixture, pinned_clock  # noqa: F401  (pinned_clock: autouse fixture)
from test_mind_lessons import embedder  # noqa: F401,E402  (pytest fixture: lesson selection is by meaning)

ROOMY = {"breaker": {"failures": 50}, "budgets": {"tasks_per_hour": 50, "concurrent_tasks": 50}}
INSTRUCTION = {"kind": "pitfall", "title": "No long dashes in writing",
               "when_to_use": "writing any message or reply",
               "content": "Never use the long dash; use a comma, a colon or a new sentence instead."}
NOTE = "The owner said this on a phone call with the operator."
CONSOLE_SECRET = token_hex(24)
READER_SECRET = token_hex(24)


@pytest.fixture
def fx(tmp_path, monkeypatch):
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    fixture = Fixture(tmp_path, config=ROOMY)
    yield fixture
    mind_router.set_mind(None)
    fixture.store.close()


def relay(fx, **changes):
    fields = {**INSTRUCTION, **changes}
    said = fields.pop("said_at", (fx.now - timedelta(hours=3)).isoformat())
    note = fields.pop("note", NOTE)
    return fx.mind.lessons.relay(fields, said_at=said, note=note, recorded_by="credential:console")


def entries(fx):
    with closing(fx.ledger._connect()) as conn:
        return {row["turn_id"]: dict(row) for row in conn.execute("SELECT * FROM turn_sources WHERE session_id='mind'")}


def test_a_relayed_instruction_is_a_lesson_with_its_own_provenance(fx):
    lesson, created = relay(fx)
    assert created and isinstance(lesson, Lesson) and lesson.status == "active" and lesson.kind == "pitfall"
    assert lesson.origin == "operator" and lesson.verified == "operator" and lesson.signature.startswith("relayed:")
    assert lesson.relayed == {"source": "operator", "relayed_from": "owner",
                              "said_at": (fx.now - timedelta(hours=3)).isoformat(), "note": NOTE,
                              "recorded_by": "credential:console", "recorded_at": fx.now.isoformat()}
    assert fx.mind.lessons.get(lesson.id) == lesson
    # Two entries of the mind's own session: the operator's note and the lesson resting on it. Neither is a
    # message of the owner's, and no claim is derived from either.
    found = entries(fx)
    note_id, admitted_id = f"mind:relayed:{lesson.id}:note", f"mind:lesson:{lesson.id}:admitted"
    assert set(found) == {note_id, admitted_id}
    for row in found.values():
        assert (row["contact_id"], row["scope"]) == (OWNER, "session")
        [message] = json.loads(row["messages_json"])
        assert message["role"] == "assistant"
    [note] = json.loads(found[note_id]["messages_json"])
    assert note["metadata"]["origin"] == "operator" and note["metadata"]["relayed"] == lesson.relayed
    assert NOTE in note["content"] and "operator" in note["content"]
    [admitted] = json.loads(found[admitted_id]["messages_json"])
    assert [ref["source_id"] for ref in admitted["_supplied_sources"]] == [note_id]
    with closing(fx.ledger._connect()) as conn:
        assert conn.execute("SELECT count(*) FROM source_claim_jobs WHERE turn_id LIKE 'mind:%'").fetchone()[0] == 0
    for session in ("day-02", "later"):
        found = fx.ledger.search_sources("long dash comma colon", contact_id=OWNER, session_id=session)
        assert all(not hit["turn_id"].startswith("mind:") for hit in found)
    # In a turn it says where it came from: the operator relayed it, never the owner's verdicts.
    section = lesson.section()
    assert "relayed by the operator" in section and "owner's verdicts" not in section
    assert section.startswith(f"[lesson {lesson.id}, the owner's instruction, relayed by the operator")
    assert fx.mind.lessons.stats()["admitted_by_source"] == {"operator": 1}


def test_relaying_the_same_instruction_again_is_the_same_lesson_and_a_changed_one_replaces_it(fx):
    first, created = relay(fx)
    again, created_again = relay(fx)
    assert created and not created_again and again.id == first.id
    changed, created = relay(fx, content="Never use the long dash or the en dash; a comma or a colon will do.")
    assert created and changed.supersedes == first.id and fx.mind.lessons.get(first.id).status == "superseded"
    # An owner's own lesson on the same subject is its own class: a relay never supersedes it.
    owner = fx.mind.lessons.admit({"signature": "topic:no-long-dashes-in-writing", **INSTRUCTION}, verified="owner",
                                  origin="night", status="active", evidence=[], lineage=[], now=fx.now)
    relay(fx, content="Keep sentences short, and never a long dash.")
    assert fx.mind.lessons.get(owner.id).status == "active"


def test_a_secret_in_a_relayed_instruction_is_sealed_before_anything_is_derived_from_it(fx):
    # The operator's words arrive raw, not through turn capture: they are sealed before the lesson's class and id
    # are taken from them, so no value survives in the signature, the id, a stored entry or the route's answer.
    secrets = ("correcthorsebattery", "hunter2hunter2", "s3cretPa55word!")
    lesson, created = relay(fx, title="WiFi password: correcthorsebattery",
                            content="Give visitors the guest network password: s3cretPa55word! and nothing else.")
    topical, _ = relay(fx, kind="strategy", topic="the wifi password is hunter2hunter2")
    assert created and lesson.signature.startswith("relayed:") and topical.signature.startswith("relayed:")
    stored = json.dumps(entries(fx)) + json.dumps([item.as_dict() for item in (lesson, topical)])
    assert not [value for value in secrets if value.casefold() in stored.casefold()]
    assert "[secret S-" in lesson.title and "[secret S-" in lesson.content
    assert lesson.id == lesson_id(lesson.signature, lesson.kind, lesson.content)      # the id: of the sealed words
    # The values the stored words refer to are in the sealed store, as a captured message's are.
    vault = fx.ledger.vault
    assert vault.find("correcthorsebattery", contact_id=OWNER) and vault.find("s3cretPa55word!", contact_id=OWNER)
    # The same words again are the same lesson; a changed instruction on that topic replaces it.
    again, created_again = relay(fx, title="WiFi password: correcthorsebattery",
                                 content="Give visitors the guest network password: s3cretPa55word! and nothing else.")
    assert not created_again and again.id == lesson.id
    changed, _ = relay(fx, kind="strategy", topic="the wifi password is hunter2hunter2",
                       content="Read the guest network name aloud before anything else.")
    assert changed.signature == topical.signature and changed.supersedes == topical.id


async def test_a_relay_never_supersedes_what_the_owners_verdicts_made_of_an_earlier_relay(tmp_path, monkeypatch):
    # The night replaces a relayed lesson on the owner's own verdict: the replacement is the owner's lesson, of the
    # owner's class, so a later relay on that topic replaces relays only and never the owner-derived lesson.
    from test_mind_lessons_night import RULE_QUOTE, add, label, make, night, ruled, training_day
    state = {}

    def answer(prompt):
        verdict = label(prompt, "Verdict on train-01.json")
        return {"verdicts": ruled(prompt), "ops": [
            {**add([verdict], quote=RULE_QUOTE, content="Channel letter, then the order's last two digits."),
             "op": "supersede", "lesson_id": state["relayed"].id}]}
    night_fx = make(tmp_path, monkeypatch, answer)
    lessons = night_fx.mind.lessons
    fields = {"kind": "strategy", "title": "Order codes by channel", "when_to_use": "an order code is asked for",
              "content": "Start with the channel letter.", "topic": "order codes"}
    state["relayed"], _ = lessons.relay(fields, said_at=night_fx.now - timedelta(hours=1), note=NOTE,
                                        recorded_by="credential:console")
    training_day(night_fx)
    result = await night(night_fx)
    assert result["counts"]["lessons_superseded"] == 1
    [owners] = lessons.all()
    assert owners.supersedes == state["relayed"].id and owners.verified == "owner" and owners.relayed is None
    assert not owners.signature.startswith("relayed:")
    later, created = lessons.relay({**fields, "content": "Start with the order number."},
                                   said_at=night_fx.now - timedelta(hours=1), note=NOTE,
                                   recorded_by="credential:console")
    assert created and later.supersedes is None and lessons.get(owners.id).status == "active"
    night_fx.store.close()


async def test_a_task_body_and_every_other_rendering_say_a_relayed_lesson_is_the_operators_relay(tmp_path, monkeypatch,
                                                                                           embedder):
    # A relayed lesson rests on the operator's word, not on a verified result: the task body that carries it, the
    # deliberation's lines, the night's packet and a skill made of it say so, never "from verified results".
    from colonymind.mind.skills import Skills
    from test_mind_lessons_night import LESSONS_ONLY, make
    fx = make(tmp_path, monkeypatch, config={**LESSONS_ONLY, **ROOMY, "act_threshold": 0.3})
    lesson, _ = fx.mind.lessons.relay({"kind": "strategy", "title": "Tide tables research",
                                       "when_to_use": "tide tables are researched",
                                       "content": "Use the harbour office's published table, not the forecast site."},
                                      said_at=fx.now - timedelta(hours=1), note=NOTE, recorded_by="credential:console")
    source = f"the owner's instruction, relayed by the operator ({(fx.now - timedelta(hours=1)).date().isoformat()})"
    assert lesson.line().startswith(f"[lesson {lesson.id}, strategy, {source}] When tide tables are researched:")
    await vectors_settled(fx.mind.lessons)
    fx.mind.add_interest("tide tables")
    formed = await fx.mind.tick(force=True)
    [row] = [fx.store.get(item["id"]) for item in formed["formed"] if item["type"] == "research"]
    assert json.loads(row.lesson_ids) == [lesson.id]
    [sent] = [item for item in fx.mind.dispatch() if item["id"] == row.id]
    assert f"[lesson {lesson.id}, strategy, {source}]" in sent["body"]
    assert "from verified results" not in sent["body"]
    fx.turn("day-01-ask", OWNER, "day-01", "Please do the tide tables research again.", "On it.")
    fx.shift(minutes=2)
    fx.turn("day-01-more", OWNER, "day-01", "The tide tables research is for the weekend.", "Noted.")
    fx.shift(minutes=2)
    packet = fx.mind.lessons._packet(fx.now)["text"]
    assert f"{lesson.id} | active | strategy | {source} | {lesson.signature} |" in packet
    skill = Skills.render("colonymind-tide-tables", lesson, {"wins": 3, "uses": 3})
    assert "relayed by the operator" in skill and "learned from verified results" not in skill
    fx.store.close()


@pytest.mark.parametrize("changes, match", [
    ({"kind": "rule"}, "kind"),
    ({"title": ""}, "title"),
    ({"content": "x" * 401}, "content"),
    ({"note": " "}, "note"),
    ({"said_at": "yesterday"}, "said_at"),
    ({"said_at": "2999-01-01"}, "said_at"),
])
def test_a_relayed_instruction_needs_every_field_of_its_provenance(fx, changes, match):
    with pytest.raises(ValueError, match=match):
        relay(fx, **changes)
    assert fx.mind.lessons.all(include_closed=True) == [] and entries(fx) == {}


def _app(fx, *, keyed: bool = True) -> FastAPI:
    app = FastAPI()
    if keyed:
        app.add_middleware(ApiKeyMiddleware, api_key=KEY, credentials=CredentialTable.from_config({
            "api-key": {"owner": ["whatsapp", "sms"], "contacts": "all"},
            "console": {"sha256": secret_digest(CONSOLE_SECRET), "owner": "any", "contacts": "all",
                        "access": ["read", "write", "manage"]},
            "reader": {"sha256": secret_digest(READER_SECRET), "contacts": "all",
                       "access": ["read", "write", "manage"]}}))
    app.include_router(mind_router.router)
    mind_router.set_mind(fx.mind)
    return app


def _body(fx, **changes):
    return {**INSTRUCTION, "said_at": (fx.now - timedelta(hours=3)).isoformat(), "note": NOTE, **changes}


async def test_the_route_records_it_for_a_credential_granted_the_owner_and_lists_it(fx):
    console = {"Authorization": "Bearer " + CONSOLE_SECRET}
    async with AsyncClient(transport=ASGITransport(app=_app(fx)), base_url="http://test") as client:
        recorded = await client.post("/v1/mind/lessons/relayed", headers=console, json=_body(fx))
        assert recorded.status_code == 200, recorded.text
        value = recorded.json()
        assert value["created"] is True and value["relayed"]["recorded_by"] == "credential:console"
        assert value["relayed"]["relayed_from"] == "owner" and value["verified"] == "operator"
        again = await client.post("/v1/mind/lessons/relayed", headers=AUTH, json=_body(fx))
        assert again.status_code == 200 and again.json()["created"] is False and again.json()["id"] == value["id"]
        listed = (await client.get("/v1/mind/lessons", headers=AUTH)).json()
        [row] = listed["lessons"]
        assert row["id"] == value["id"] and row["relayed"]["note"] == NOTE
        assert "relayed by the operator" in listed["text"]
        bad = await client.post("/v1/mind/lessons/relayed", headers=AUTH, json=_body(fx, said_at="2999-01-01"))
        assert bad.status_code == 422 and bad.json()["detail"]["code"] == "invalid_lesson"
        extra = await client.post("/v1/mind/lessons/relayed", headers=AUTH, json=_body(fx, by="owner"))
        assert extra.status_code == 422        # nothing in the body can say who gave it: the credential does
        retired = await client.post(f"/v1/mind/lessons/{value['id']}/retire", headers=AUTH,
                                    json={"reason": "no longer wanted"})
        assert retired.status_code == 200
        closed = await client.post("/v1/mind/lessons/relayed", headers=AUTH, json=_body(fx))
        assert closed.status_code == 409 and closed.json()["detail"]["code"] == "lesson_closed"


async def test_only_a_credential_granted_the_owner_may_relay_the_owners_instruction(fx):
    reader = {"Authorization": "Bearer " + READER_SECRET}
    async with AsyncClient(transport=ASGITransport(app=_app(fx)), base_url="http://test") as client:
        refused = await client.post("/v1/mind/lessons/relayed", headers=reader, json=_body(fx))
        assert refused.status_code == 403 and refused.json()["detail"]["code"] == "not_owner"
    async with AsyncClient(transport=ASGITransport(app=_app(fx, keyed=False)), base_url="http://test") as client:
        bare = await client.post("/v1/mind/lessons/relayed", json=_body(fx))
        assert bare.status_code == 403
    assert fx.mind.lessons.all(include_closed=True) == []
    fx.mind.lessons.enabled = False
    async with AsyncClient(transport=ASGITransport(app=_app(fx)), base_url="http://test") as client:
        off = await client.post("/v1/mind/lessons/relayed", headers=AUTH, json=_body(fx))
        assert off.status_code == 409 and off.json()["detail"]["code"] == "lessons_off"


def _parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", default=None)
    sub = parser.add_subparsers(dest="command")
    mind_cli.add_parser(sub)
    return parser.parse_args(argv)


def test_the_cli_relays_an_instruction_and_shows_its_provenance(home, monkeypatch, capsys):  # noqa: F811
    lesson = {"id": "L-1a2b3c4d5e", "status": "active", "kind": "pitfall", "verified": "operator",
              "origin": "operator", "signature": "relayed:no-long-dashes-in-writing", "evidence": [],
              "title": INSTRUCTION["title"], "when_to_use": INSTRUCTION["when_to_use"],
              "content": INSTRUCTION["content"], "tally": {"uses": 0, "wins": 0},
              "relayed": {"source": "operator", "relayed_from": "owner", "said_at": "2026-09-29T08:00:00+00:00",
                          "note": NOTE, "recorded_by": "credential:api-key",
                          "recorded_at": "2026-09-29T11:00:00+00:00"}}
    transport = _Transport({("POST", "/v1/mind/lessons/relayed"): {**lesson, "created": True},
                            ("GET", "/v1/mind/lessons"): {"enabled": True, "lessons": [lesson], "uses": []}})
    monkeypatch.setattr(httpx, "Client", transport.client)
    argv = ["mind", "lessons", "relay", "--kind", "pitfall", "--title", INSTRUCTION["title"], "--when",
            INSTRUCTION["when_to_use"], "--content", INSTRUCTION["content"], "--note", NOTE]
    assert mind_cli.run(_parse(argv)) == 2                                   # --said-at is required
    assert "--said-at" in capsys.readouterr().err and transport.calls == []
    assert mind_cli.run(_parse([*argv, "--said-at", "2026-09-29T08:00:00+00:00"])) == 0
    method, path, body = transport.calls[-1]
    assert (method, path) == ("POST", "/v1/mind/lessons/relayed")
    assert json.loads(body) == {**INSTRUCTION, "said_at": "2026-09-29T08:00:00+00:00", "note": NOTE}
    out = capsys.readouterr().out
    assert "L-1a2b3c4d5e" in out and "relayed by the operator" in out
    assert mind_cli.run(_parse(["mind", "lessons", "show", "L-1a2b3c4d5e"])) == 0
    shown = capsys.readouterr().out
    assert "relayed by the operator (credential:api-key, 2026-09-29T11:00:00+00:00)" in shown and NOTE in shown
    assert "the owner said it 2026-09-29T08:00:00+00:00" in shown


from colonymind.memory import curated
def test_a_relayed_lesson_is_retired_and_erased_like_every_lesson(fx):
    lesson, _ = relay(fx)
    retired = fx.mind.lessons.retire(lesson.id, reason='the owner changed their mind', by='cli')
    assert retired.status == 'retired' and retired.relayed == lesson.relayed
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(fx.ledger)
    curated.forget_sources(fx.ledger, OWNER, source_ids=[f'mind:relayed:{lesson.id}:note'])
    assert historical_bytes(fx.ledger) == _native_before_withdrawal
    assert fx.mind.lessons.all(include_closed=True) == [] and len(entries(fx)) == 3
    assert relay(fx, content='A comma, a colon or a new sentence: never a long dash.') == (None, False)
    assert fx.mind.lessons.all(include_closed=True) == []
    again, _ = relay(fx, content='Use commas and short sentences in the next report.', note='The owner repeated it in person.')
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(fx.ledger)
    curated.forget_sources(fx.ledger, OWNER, source_ids=[f'mind:lesson:{again.id}:admitted'])
    assert historical_bytes(fx.ledger) == _native_before_withdrawal
    assert fx.mind.lessons.get(again.id) is None
    assert f'mind:relayed:{again.id}:note' in entries(fx)
    assert len(entries(fx)) >= 5

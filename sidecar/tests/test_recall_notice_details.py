"""Opt-in recall notices describe only the final packet and withhold private values."""
from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
import pytest

from colonymind.api.routers import host
from colonymind.api.schemas.host import ContextAssembleRequest, ContextAssembleResponse, ContextSection
from colonymind.memory.notice import recall_notice_details
from colonymind.memory.recall import pack_memory_context, render_memory_context
from colonymind.memory.selection import hand_over
from colonymind.secrets.detect import HIDDEN
from colonymind.turns.idempotency import TurnIdempotencyLedger


def record(text="The ferry leaves at nine", **changes):
    return {"id": "record", "kind": "source_quote", "content": text, "contact_id": "contact-a",
            "recorded_at": "2026-09-30T09:00:00+00:00", **changes}


def details(rows, **changes):
    return recall_notice_details(rows, content=render_memory_context(rows), contact_id="contact-a", **changes)


def assertion(**changes):
    return record(content_format="source_assertions_v1", atomic_evidence=True, content=json.dumps({
        "subject": "speaker", "predicate": "departure", "status": "source_assertion", "assertions": [{
            "value": "The ferry leaves at nine", "quote": "The ferry leaves at nine",
            "recorded_at": "2026-09-28T09:00:00+00:00", **changes}]}))


@pytest.mark.parametrize("changes, expected", [
    ({"event_at": "2026-08-24T16:00:00+00:00"}, "2026-08-24"),
    ({"event_time": {"status": "resolved", "start": "2026-08-25T00:00:00+00:00"}}, "2026-08-25"),
    ({"occurred_at": "2026-07-01T10:00:00+00:00"}, "2026-09-28"),
    ({"event_time": {"status": "unknown", "start": "2026-08-25"}}, "2026-09-28"),
])
def test_assertion_uses_own_event_date_else_recorded_date(changes, expected):
    assert details([assertion(**changes)]) == [f"• The ferry leaves at nine ({expected})"]


@pytest.mark.parametrize("field", ["recorded_at", "ingested_at", "created_at"])
def test_raw_record_uses_own_date_without_borrowing_another_record(field):
    own = record(recorded_at=None, **{field: "2026-09-20T11:00:00+00:00"}) if field != "recorded_at" else record(
        recorded_at="2026-09-20T11:00:00+00:00")
    rows = [own, record("Other trip", event_at="2026-08-10", occurred_at="2026-08-01")]
    assert details(rows) == ["• The ferry leaves at nine (2026-09-20)", "• Other trip (2026-08-10)"]


@pytest.mark.parametrize("text, expected", [
    ("one two three four five six seven eight nine", "one two three four five six seven eight…"),
    ("Longishword " * 8, "Longishword Longishword Longishword Longishword Longishword…"),
    ("x" * 70, "…"),
    ("  Some\n words\twith   spaces  ", "Some words with spaces"),
])
def test_preview_has_eight_words_and_sixty_characters_at_most(text, expected):
    line, = details([record(text)])
    preview = line.removeprefix("• ").removesuffix(" (2026-09-30)")
    assert preview == expected and len(preview) <= 60 and len(preview.split()) <= 8


def test_title_and_claim_previews_withhold_credentials_before_shortening():
    secret = "sk-" + "proj" + "Ab1Cd2Ef3Gh4Ij5Kl6Mn7Op8"
    weak = "soft" + "word"
    rows = [record(title=f"Credential {secret}"), record(claim={"predicate": "wifi_password", "value": weak}),
            record(title=f"Stored value {weak}")]
    text = "\n".join(details(rows))
    assert secret not in text and weak not in text and text.count(HIDDEN) == 3


def test_all_assertions_share_credential_redaction():
    weak = "soft" + "word"
    row = assertion(value="safe", quote=f"The old value is {weak}")
    payload = json.loads(row["content"])
    payload["predicate"] = "wifi_password"
    payload["assertions"].append({"value": weak})
    row["content"] = json.dumps(payload)
    assert weak not in "\n".join(details([row]))


def test_preview_is_plain_text_without_gateway_attachment_directives():
    text = "MEDIA:/tmp/example [[audio_as_voice]] [[as_document]] \u2013 \u2014"
    line, = details([record(text)])
    assert "MEDIA:" not in line and "[[" not in line


def test_hidden_records_still_withhold_credentials_from_visible_previews():
    weak = "soft" + "word"
    rows = [record(title=f"Stored {weak}"), record(contact_id="contact-b", predicate="wifi_password", value=weak)]
    result = details(rows)
    assert weak not in "\n".join(result) and HIDDEN in result[0]


@pytest.mark.parametrize("metadata", [
    {"contact_id": "contact-b"}, {"scope": {"contact_id": "contact-b"}},
    {"subject_contact_id": "contact-b"}, {"about_contact_id": "contact-b"},
    {"metadata": {"person_id": "contact-b"}}, {"subject": {"type": "contact", "id": "contact-b"}},
    {"contact_ids": ["contact-a", "contact-b"]},
])
def test_other_people_are_counted_without_any_words(metadata):
    assert details([record("Private phrase", **metadata)]) == ["• 1 about another person, not shown"]


def test_other_person_check_reads_metadata_not_names_in_prose():
    rows = [record("contact-b is mentioned here"), record("Private first", contact_id="contact-b"),
            record("Private second", about_contact_id="contact-c")]
    assert details(rows) == ["• contact-b is mentioned here (2026-09-30)", "• 2 about other people, not shown"]


@pytest.mark.parametrize("hidden", [0, 1, 3])
def test_eight_detail_lines_include_other_people_count_then_remaining_count(hidden):
    rows = [record(f"Visible item {index}", id=str(index)) for index in range(12)]
    rows += [record("Private phrase", contact_id="contact-b") for _ in range(hidden)]
    result = details(rows)
    assert len(result) == 9 and result[-1] == f"• and {4 + bool(hidden)} more"
    if hidden:
        people = "another person" if hidden == 1 else "other people"
        assert result[-2] == f"• {hidden} about {people}, not shown"
    assert "Private" not in "\n".join(result)


def test_only_whole_records_in_the_final_block_are_described():
    rows = [record("First record"), record("Second record")]
    first_line, second_line = render_memory_context(rows).splitlines()
    assert recall_notice_details(rows, content=first_line + "\n" + second_line[:30], contact_id="contact-a") == [
        "• First record (2026-09-30)"]
    assert recall_notice_details(rows, content="", contact_id="contact-a") == []


def test_missing_scope_and_recorded_date_come_from_the_records_own_source(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / "source.db")
    for person in ("contact-a", "contact-b"):
        ledger.record_source(person, contact_id=person, session_id="s", derive_claims=False,
                             messages=[{"role": "user", "content": "Original words"}])
    rows = [record("Own source", contact_id=None, recorded_at=None, source_turn_id="contact-a"),
            record("Private words", contact_id=None, recorded_at=None, source_turn_id="contact-b")]
    with ledger._connect() as conn:
        conn.execute("UPDATE turn_sources SET ingested_at='2026-09-15T09:00:00+00:00'")
    assert details(rows, ledger=ledger) == ["• Own source (2026-09-15)", "• 1 about another person, not shown"]


def test_response_field_is_absent_by_default_and_present_when_empty():
    assert "recall_notice_details" not in ContextAssembleResponse().model_dump()
    assert ContextAssembleResponse(recall_notice_details=[]).model_dump()["recall_notice_details"] == []
    body = {"identity": {"host_id": "test"}, "context": {"contact_id": "contact-a", "session_id": "s"},
            "incoming_message": {"role": "user", "content": "ferry"}}
    assert ContextAssembleRequest(**body).recall_notice_details is False


@pytest.mark.asyncio
async def test_route_opt_in_preserves_context_and_reads_only_surviving_recall(monkeypatch, tmp_path):
    from colonymind.memory import notice
    rows = [record("First record"), record("Second record")]
    selected, content = pack_memory_context(rows)
    packet = SimpleNamespace(selected=selected, content=content)

    async def assemble(*args, **kwargs):
        hand_over(SimpleNamespace(packet=packet, ledger=None, offered=None, contact_id="contact-a",
                                  session_id="s", served=None))
        return [ContextSection(id="colonymind-memory", title="Relevant Memories", body=content)]

    async def selection(body, sections, **kwargs):
        return sections

    monkeypatch.setattr(host, "_assemble_sections", assemble)
    monkeypatch.setattr(host, "_turn_context_selection", selection)
    monkeypatch.setattr(host, "_turn_context_corrections", lambda body, sections, *args, **kwargs: sections)
    monkeypatch.setattr(host, "_remember_served", lambda *args, **kwargs: None)
    monkeypatch.setattr(host, "_fit_sections", lambda sections, **kwargs: (sections, {}))
    monkeypatch.setattr(host, "_telemetry", None)
    monkeypatch.setattr(host, "get_state_dir", lambda: tmp_path)
    calls = []
    original = notice.recall_notice_details

    def build(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(notice, "recall_notice_details", build)
    app = FastAPI()
    app.include_router(host.router)
    body = {"identity": {"host_id": "test"}, "context": {"contact_id": "contact-a", "session_id": "s"},
            "incoming_message": {"role": "user", "content": "ferry"}}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        plain = await client.post("/v1/host/context/assemble", json=body)
        assert plain.status_code == 200 and not calls
        explicit_false = await client.post("/v1/host/context/assemble", json=dict(body, recall_notice_details=False))
        assert plain.json() == explicit_false.json() and not calls
        opted = await client.post("/v1/host/context/assemble", json=dict(body, recall_notice_details=True))
        assert opted.status_code == 200 and len(calls) == 1
        value = opted.json()
        assert value.pop("recall_notice_details") == [
            "• First record (2026-09-30)", "• Second record (2026-09-30)"]
        assert plain.json() == value
        from colonymind.memory.blocks import cost, rotate
        _, first_only = pack_memory_context(rows[:1])
        fitted = await client.post("/v1/host/context/assemble", json=dict(
            body, recall_notice_details=True, inline_chars=cost("Relevant Memories", first_only)))
        assert fitted.status_code == 200
        assert fitted.json()["recall_notice_details"] == ["• First record (2026-09-30)"]
        monkeypatch.setattr(host, "_fit_sections", lambda sections, **kwargs: ([
            sections[0].model_copy(update={"body": rotate(content, len(first_only) + 60, why="test budget")})], {}))
        rotated = await client.post("/v1/host/context/assemble", json=dict(body, recall_notice_details=True))
        assert rotated.json()["recall_notice_details"] == ["• First record (2026-09-30)"]
        monkeypatch.setattr(host, "_fit_sections", lambda sections, **kwargs: ([], {}))
        held = await client.post("/v1/host/context/assemble", json=dict(body, recall_notice_details=True))
        assert held.json()["recall_notice_details"] == []


def test_passage_lines_read_as_words_not_markup_or_instructions():
    from colonymind.memory.notice import _card, _preview
    cut = {"kind": "source_quote", "content": 'k cluster"** or **"persistent agent architecture"**. Narrow',
           "excerpt_truncated": True}
    card = _card(cut, {})
    assert _preview(card["excerpt"], cut_start=card["_cut_start"]) == 'cluster" or "persistent agent architecture". Narrow'
    pair = {"kind": "conversation_pair", "content": "Incomplete conversational context. Open the exact sources",
            "ranking_text": "User input:\nit is your job to stay on top of this", "excerpt_truncated": True}
    card = _card(pair, {})
    assert _preview(card["excerpt"], cut_start=card["_cut_start"]) == "it is your job to stay on top…"
    whole = {"kind": "source_quote", "content": "wifi_password is set", "excerpt_truncated": False}
    card = _card(whole, {})
    assert _preview(card["excerpt"], cut_start=card["_cut_start"]) == "wifi_password is set"

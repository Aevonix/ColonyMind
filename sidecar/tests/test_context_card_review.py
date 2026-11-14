"""Regression coverage for the context-card review; all data is local and synthetic."""
from dataclasses import replace
from contextlib import closing, contextmanager
from pathlib import Path
from types import SimpleNamespace
import asyncio
import json
import sqlite3
import threading

import pytest

from colonymind.context_card.builder import CardInputs, build, conservative_tokens, MIN_TOKENS, MAX_TOKENS
from colonymind.context_card.inputs import DIRECT_PLATFORMS
from colonymind.context_card.refresh import CardService, RevisionProbe, configured_owner, _resolved_owner
from colonymind.context_card.store import CardStore
from colonymind.router.tokens import estimate_tokens
from test_context_card import OWNER, DAY, busy, sparse, record, mark_change, real_inputs, service


@pytest.mark.parametrize("text,old_estimate,measured", [
    ("A" * 1828 + "界" * 1200, 757, 1398),
    ("A" * 585 + "🌱" * 1800, 795, 1958),
])
def test_conservative_count_covers_review_ratios(text, old_estimate, measured, monkeypatch):
    # The review did not include its source cards. These stress fixtures match
    # its old estimates and must exceed the supplied GLM measurement thresholds.
    monkeypatch.setenv("COLONYMIND_CONTEXT_CHARS_PER_TOKEN", "4")
    assert estimate_tokens(text) == old_estimate
    assert conservative_tokens(text) >= measured
    assert conservative_tokens(text) / old_estimate >= measured / old_estimate


@pytest.mark.parametrize("payload", ["界", "🌱", "👩🏽\u200d💻", "Ж", "Recorded ASCII evidence. "])
def test_each_script_is_packed_using_the_conservative_cap(payload, monkeypatch):
    monkeypatch.setenv("COLONYMIND_CONTEXT_CHARS_PER_TOKEN", "4")
    text = payload * 1200
    card = build(replace(busy(), owner_name="Robin", preferences=text, narrative=text,
                         work=(text,), highlights=(text,), digest=text))
    assert card.token_count == conservative_tokens(card.text) <= MAX_TOKENS
    assert card == build(replace(busy(), owner_name="Robin", preferences=text, narrative=text,
                                work=(text,), highlights=(text,), digest=text))
    assert "[excerpt]" in card.text


@pytest.mark.parametrize("text", ["你好世界", "Привет мир", "Ελληνικά", "العربية", "हिन्दी"])
def test_non_latin_codepoints_cost_at_least_one(text):
    assert conservative_tokens(text) >= sum(ord(char) >= 128 for char in text)


@pytest.mark.parametrize("emoji", ["🌱", "👩\u200d💻", "👨\u200d👩\u200d👧\u200d👦", "🇺🇸", "1\ufe0f\u20e3", "1\u20e3", "©\ufe0f", "🏳️\u200d🌈"])
def test_emoji_and_zwj_sequences_receive_at_least_three_tokens(emoji):
    assert conservative_tokens(emoji) >= 3
    assert conservative_tokens(emoji * 100) >= 300


@pytest.mark.parametrize("ratio", ["3", "4", "5"])
def test_ascii_uses_existing_estimator_without_floor_filler(ratio, monkeypatch):
    monkeypatch.setenv("COLONYMIND_CONTEXT_CHARS_PER_TOKEN", ratio)
    card = build(sparse())
    assert conservative_tokens(card.text) == estimate_tokens(card.text) == card.token_count
    assert card.token_count < MIN_TOKENS
    assert "Unknown context:" not in card.text
    assert "Coverage limits:" not in card.text
    assert "Continuity:" not in card.text
    assert card.text.count("This private snapshot") == 1
    assert "Do not invent" in card.text


def test_floor_uses_additional_supported_ascii_evidence(monkeypatch):
    monkeypatch.setenv("COLONYMIND_CONTEXT_CHARS_PER_TOKEN", "4")
    card = build(replace(sparse(), preferences="Supported owner preference. " * 100))
    assert MIN_TOKENS <= card.token_count <= MAX_TOKENS
    assert card.text.count("Supported owner preference.") > 14
    assert "Coverage limits:" not in card.text


def test_no_em_or_en_dash_can_enter_any_card_field():
    value = "First – second — third"
    inputs = CardInputs(OWNER, value, value, preferences=value, narrative=value,
                       work=(value,), highlights=(value,), digest=value, unavailable=(value,))
    text = build(inputs).text
    assert text.startswith("Daily context card: ")
    assert "–" not in text and "—" not in text
    assert "First - second - third" in text


@pytest.mark.asyncio
async def test_continuous_unrelated_writes_serve_stale_and_publish_within_two_passes(service):
    assert await service.refresh()
    first = service.fetch()
    service.collector.inputs = replace(sparse(), preferences="New supported preference.")
    mark_change(service, "mind.db")
    service.collector.on_collect = lambda: mark_change(service, "mind.db")
    assert service.fetch().stale
    assert service.fetch().version == first.version
    assert not await service.refresh()
    assert await service.refresh()
    current = service.fetch()
    assert current.stale and current.version == first.version + 1
    assert "New supported preference" in current.text
    for _ in range(6):
        assert service.fetch() is not None
        await service.refresh()
    assert service.fetch().version == current.version
    service.collector.on_collect = None
    assert not await service.refresh()
    assert not service.fetch().stale


@pytest.mark.asyncio
async def test_first_card_eventually_publishes_during_continuous_writes(service):
    service.collector.on_collect = lambda: mark_change(service, "mind.db")
    assert not await service.refresh()
    assert service.fetch() is None
    assert await service.refresh()
    assert service.fetch().stale


@pytest.mark.asyncio
async def test_failed_refresh_with_unrelated_changes_keeps_last_card_stale(service):
    await service.refresh()
    first = service.fetch()
    service.collector.fail = True
    mark_change(service, "mind.db")
    assert not await service.refresh()
    assert service.fetch().stale and service.fetch().text == first.text
    service.collector.fail = False
    assert not await service.refresh()
    assert not service.fetch().stale


@pytest.mark.asyncio
async def test_continuous_unrelated_ledger_commits_cannot_starve_publication(real_inputs, tmp_path, monkeypatch):
    ledger, collector, _ = real_inputs
    record(ledger, "owner-source", "Owner evidence.")
    with closing_store(tmp_path) as store:
        service = CardService(store, collector, RevisionProbe(tmp_path, day_provider=lambda: DAY))
        for _ in range(2):
            await service.refresh()
        first = service.fetch()
        assert first is not None
        record(ledger, "zz-new-owner-source", "New owner evidence.")
        original, commits = collector.collect_ledger, []
        def racing(prepared):
            result = original(prepared)
            commits.append(len(commits))
            record(ledger, f"racing-guest-{len(commits)}", "Guest write", person="cid-guest")
            return result
        monkeypatch.setattr(collector, "collect_ledger", racing)
        assert not await service.refresh()
        assert service.fetch().stale and service.fetch().version == first.version
        assert await service.refresh()
        assert "New owner evidence" in service.fetch().text
        assert service.fetch().stale
        for _ in range(4):
            await service.refresh()
            assert service.fetch() is not None


@pytest.mark.asyncio
async def test_rebinding_race_never_forces_publication(service, monkeypatch):
    await service.refresh()
    service.max_postponed_passes = 1
    def rebind():
        monkeypatch.setenv("COLONYMIND_OWNER_PERSON_ID", "cid-next-owner")
    service.collector.on_collect = rebind
    mark_change(service, "mind.db")
    assert not await service.refresh()
    assert service.fetch() is None


@pytest.mark.asyncio
async def test_shared_producers_stay_on_loop_and_ledger_runs_in_thread(real_inputs, tmp_path, monkeypatch):
    ledger, collector, host = real_inputs
    from colonymind.initiatives.store import InitiativeStore
    from colonymind.commitments.store import CommitmentStore
    loop_thread = threading.get_ident()
    seen, worker_seen = [], []
    host._initiative_store = InitiativeStore(state_dir=tmp_path)
    host._commitment_store = CommitmentStore(tmp_path / "colonymind-commitments.db")
    record(ledger, "thread-source", "Keep replies concise.")
    host._commitment_store.create(OWNER, "Review plans.", source_type="turn", metadata={"source_turn": "thread-source"})
    def checked(name, result):
        assert threading.get_ident() == loop_thread
        seen.append(name)
        return result
    host._preference_learner = SimpleNamespace(
        perspective=SimpleNamespace(brief=lambda **_: checked("perspective", "Owner preference.")),
        build_brief=lambda **_: checked("preferences", "Concise replies."))
    collector.mind_provider = lambda: checked("mind_provider", SimpleNamespace(owner_id=OWNER,
        narrative=lambda: checked("narrative", {"enabled": True, "text": "I connect evidence.", "cites": []}),
        section=lambda: checked("section", "On my mind: review plans.")))
    for name, producer, method in [("commitments", host._commitment_store, "list"),
                                  ("initiatives", host._initiative_store, "intentions")]:
        original = getattr(producer, method)
        monkeypatch.setattr(producer, method, lambda *a, _name=name, _original=original, **kw:
                            checked(_name, _original(*a, **kw)))
    original = collector.collect_ledger
    def ledger_read(prepared):
        worker_seen.append(threading.get_ident())
        assert threading.get_ident() != loop_thread
        return original(prepared)
    monkeypatch.setattr(collector, "collect_ledger", ledger_read)
    try:
        with closing_store(tmp_path) as store:
            service = CardService(store, collector, RevisionProbe(tmp_path, day_provider=lambda: DAY))
            for _ in range(3):
                await service.refresh()
            assert service.fetch() is not None
        assert set(seen) == {"mind_provider", "preferences", "perspective", "narrative", "section", "commitments", "initiatives"}
        assert worker_seen
    finally:
        host._initiative_store.close()


@pytest.mark.asyncio
async def test_legacy_comms_locators_are_detached_before_worker_read(real_inputs, tmp_path, monkeypatch):
    from colonymind.contacts.comms import CommsLog
    ledger, collector, host = real_inputs
    record(ledger, "legacy-direct-source", "Legacy source-backed owner evidence.")
    with closing(ledger._connect()) as conn, conn:
        conn.execute("DELETE FROM source_channels WHERE turn_id=?", ("legacy-direct-source",))
    log = CommsLog(str(tmp_path / "colonymind-comms.db"), source_ledger=ledger)
    host._comms_log = log
    try:
        lineage, _ = log.source_input("legacy-direct-source", OWNER)
        log.log(OWNER, channel="voice:synthetic", summary="SUMMARY_MUST_NOT_ENTER", source_lineage=lineage)
        loop_thread, original, captures = threading.get_ident(), log.read_connection, []
        def capture():
            assert threading.get_ident() == loop_thread
            captures.append(True)
            return original()
        monkeypatch.setattr(log, "read_connection", capture)
        prepared = collector.prepare()
        assert prepared.comms is not None
        # Closing the live store after capture proves no connection escapes.
        log._conn.close()
        inputs = await asyncio.to_thread(collector.collect_ledger, prepared)
        assert captures == [True]
        assert any("Legacy source-backed owner evidence" in line for line in inputs.highlights)
        assert "SUMMARY_MUST_NOT_ENTER" not in build(inputs).text
    finally:
        log._conn.close()


@pytest.mark.asyncio
async def test_shutdown_waits_for_ledger_worker(real_inputs, tmp_path, monkeypatch):
    _, collector, _ = real_inputs
    started, finish = threading.Event(), threading.Event()
    original = collector.collect_ledger
    def waiting(prepared):
        started.set()
        assert finish.wait(timeout=5)
        return original(prepared)
    monkeypatch.setattr(collector, "collect_ledger", waiting)
    with closing_store(tmp_path) as store:
        service = CardService(store, collector, RevisionProbe(tmp_path, day_provider=lambda: DAY))
        task = asyncio.create_task(service.refresh())
        while not started.is_set():
            await asyncio.sleep(0.001)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task


def test_api_server_automation_never_becomes_a_highlight(real_inputs):
    ledger, collector, _ = real_inputs
    record(ledger, "automation", "AUTOMATION_PRIVATE_DETAIL", channel="api_server:synthetic")
    record(ledger, "voice", "Owner direct conversation.", channel="voice:synthetic")
    inputs = collector.collect()
    assert "api_server" not in DIRECT_PLATFORMS
    assert "automation" not in inputs.source_refs
    assert "AUTOMATION_PRIVATE_DETAIL" not in build(inputs).text
    assert any("Owner direct conversation" in entry for entry in inputs.highlights)


@pytest.mark.asyncio
async def test_only_current_version_is_retained_and_sequence_survives_restart(service):
    for index in range(12):
        service.collector.inputs = replace(sparse(), preferences=f"Supported preference {index}.")
        mark_change(service, "mind.db")
        assert await service.refresh()
        rows = service.store.conn.execute("SELECT owner_id,version,payload FROM context_card_versions").fetchall()
        assert len(rows) == 1 and rows[0][1] == index + 1
        assert json.loads(rows[0][2])["text"] == service.fetch().text
    other = CardStore(service.probe.state_dir / "context-card.db")
    try:
        restarted = CardService(other, service.collector, service.probe)
        assert restarted.fetch() is None
        await restarted.refresh()
        assert restarted.fetch().version == 12
        other.purge()
        service.collector.inputs = replace(sparse(), preferences="New supported evidence.")
        mark_change(service, "mind.db")
        assert await restarted.refresh()
        assert restarted.fetch().version == 13
        assert len(other.conn.execute("SELECT * FROM context_card_versions").fetchall()) == 1
    finally:
        other.close()


def test_open_prunes_legacy_versions(tmp_path):
    with closing_store(tmp_path) as store:
        for index in range(4):
            inputs = replace(sparse(), preferences=f"Supported preference {index}.")
            store.publish(build(inputs), inputs)
        payload = store.conn.execute("SELECT payload FROM context_card_versions").fetchone()[0]
        for version in (1, 2, 3):
            store.conn.execute("INSERT INTO context_card_versions VALUES (?,?,?)", (OWNER, version, payload))
        store.conn.commit()
    with closing_store(tmp_path) as store:
        assert [row[0] for row in store.conn.execute("SELECT version FROM context_card_versions")] == [4]
        assert store.load(OWNER).version == 4


@pytest.mark.asyncio
async def test_disabled_lifecycle_deletes_existing_database_and_side_files(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from colonymind.config import Config
    from colonymind.context_card import lifecycle
    with closing_store(tmp_path) as store:
        store.publish(build(sparse()), sparse())
    path = tmp_path / "context-card.db"
    for suffix in ("-wal", "-shm", "-journal"):
        Path(str(path) + suffix).write_bytes(b"stale derived bytes")
    monkeypatch.setattr(lifecycle, "load_config", lambda: Config(home=tmp_path,
        data={"context_card": {"enabled": False, "poll_seconds": 5, "reconcile_seconds": 300}}))
    app = FastAPI()
    async with lifecycle.context_card_lifespan(app, state_dir=tmp_path):
        assert app.state.context_card is None
        assert not any(Path(str(path) + suffix).exists() for suffix in ("", "-wal", "-shm", "-journal"))
    assert not path.exists()


def test_owner_resolution_is_cached_and_rebinding_invalidates(monkeypatch):
    from colonymind import identity
    monkeypatch.delenv("COLONYMIND_OWNER_PERSON_ID", raising=False)
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", OWNER)
    calls = []
    original = identity.get_owner_contact_id
    monkeypatch.setattr(identity, "get_owner_contact_id", lambda: calls.append(True) or original())
    _resolved_owner.cache_clear()
    for _ in range(50):
        assert configured_owner() == OWNER
    assert len(calls) == 1
    monkeypatch.setenv("COLONYMIND_OWNER_CONTACT_ID", "cid-next-owner")
    assert configured_owner() == "cid-next-owner"
    assert len(calls) == 2
    _resolved_owner.cache_clear()


def test_legacy_deprecation_warning_occurs_once_per_binding(monkeypatch, caplog):
    monkeypatch.delenv("COLONYMIND_OWNER_PERSON_ID", raising=False)
    monkeypatch.delenv("COLONYMIND_OWNER_CONTACT_ID", raising=False)
    monkeypatch.setenv("COLONYMIND_HOST_CONTACT_ID", "owner")
    _resolved_owner.cache_clear()
    for _ in range(50):
        assert configured_owner() == "owner"
    assert sum("COLONYMIND_HOST_CONTACT_ID is deprecated" in item.message for item in caplog.records) == 1
    _resolved_owner.cache_clear()


@pytest.mark.asyncio
async def test_unreadable_privacy_metadata_fails_closed(service, monkeypatch):
    await service.refresh()
    def unreadable():
        raise sqlite3.DatabaseError("synthetic unreadable head")
    monkeypatch.setattr(service.probe, "binding", unreadable)
    assert service.fetch() is None
    assert not await service.refresh()
    assert service.fetch() is None


@contextmanager
def closing_store(tmp_path):
    store = CardStore(tmp_path / "context-card.db")
    try:
        yield store
    finally:
        store.close()

from colonymind.memory import curated

def test_privacy_probe_is_read_only_and_caches_head_until_a_ledger_commit(real_inputs, tmp_path, monkeypatch):
    from colonymind.context_card import refresh
    ledger, _, _ = real_inputs
    record(ledger, "erasure-source", "Private source.")
    original, statements = refresh.sqlite3.connect, []
    def read_only(*args, **kwargs):
        assert "mode=ro" in str(args[0])
        conn = original(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn
    probe = RevisionProbe(tmp_path, day_provider=lambda: DAY)
    with monkeypatch.context() as patch:
        patch.setattr(refresh.sqlite3, "connect", read_only)
        for _ in range(30):
            assert probe.binding() == (OWNER, 0)
    assert sum(statement.startswith("SELECT") for statement in statements) == 1
    assert all("messages_json" not in statement for statement in statements)
    from test_curated_memory import historical_bytes
    _native_before_forget = historical_bytes(ledger)
    _source_forget_receipt = curated.forget_sources(ledger, OWNER, source_ids=['erasure-source'])
    assert historical_bytes(ledger) == _native_before_forget
    assert all(curated.is_forgotten(ledger, sid) for sid in _source_forget_receipt['source_ids'])
    assert probe.binding() == (OWNER, 1)


@pytest.mark.asyncio
async def test_real_ledger_unrelated_writes_do_not_hide_card_but_erasure_does(real_inputs, tmp_path):
    ledger, collector, _ = real_inputs
    record(ledger, 'private-source', 'ERASABLE_PRIVATE_DETAIL')
    with closing_store(tmp_path) as store:
        service = CardService(store, collector, RevisionProbe(tmp_path, day_provider=lambda: DAY))
        for _ in range(3):
            await service.refresh()
        assert 'ERASABLE_PRIVATE_DETAIL' in service.fetch().text
        for index in range(6):
            record(ledger, f'guest-{index}', 'Unrelated guest', person='cid-guest')
            assert service.fetch().stale
            assert service.fetch() is not None
        from test_curated_memory import historical_bytes
        _native_before_withdrawal = historical_bytes(ledger)
        curated.forget_sources(ledger, OWNER, source_ids=['private-source'])
        assert historical_bytes(ledger) == _native_before_withdrawal
        assert service.fetch() is None
        for _ in range(3):
            await service.refresh()
        assert 'ERASABLE_PRIVATE_DETAIL' not in service.fetch().text

@pytest.mark.asyncio
async def test_erasure_race_never_forces_publication(real_inputs, tmp_path, monkeypatch):
    ledger, collector, _ = real_inputs
    record(ledger, 'private-source', 'ERASABLE_PRIVATE_DETAIL')
    with closing_store(tmp_path) as store:
        service = CardService(store, collector, RevisionProbe(tmp_path, day_provider=lambda: DAY), max_postponed_passes=1)
        for _ in range(3):
            await service.refresh()
        original = collector.collect_ledger

        def racing(prepared):
            result = original(prepared)
            from test_curated_memory import historical_bytes
            _native_before_withdrawal = historical_bytes(ledger)
            curated.forget_sources(ledger, OWNER, source_ids=['private-source'])
            assert historical_bytes(ledger) == _native_before_withdrawal
            return result
        monkeypatch.setattr(collector, 'collect_ledger', racing)
        mark_change(service, 'mind.db')
        assert not await service.refresh()
        assert service.fetch() is None
        assert not store.conn.execute('SELECT * FROM context_card_versions').fetchall()
        monkeypatch.setattr(collector, 'collect_ledger', original)
        for _ in range(3):
            await service.refresh()
        assert 'ERASABLE_PRIVATE_DETAIL' not in service.fetch().text

"""A composite answer retries after a peer withdrawal between its readers."""
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from colonymind.contacts.comms import CommsLog
from colonymind.memory import curated
from colonymind.turns import TurnIdempotencyLedger
from test_voice_affect import affect_app, add_appraisal, call, stamp
from test_work_projection import stores as work_stores, admit, root as work_root
from test_context_card import real_inputs


OWNER = 'fixture-owner'


def withdraw(ledger, source, person=OWNER):
    subprocess.run([sys.executable, '-B', '-c', '''from pathlib import Path
import sys
from colonymind.turns import TurnIdempotencyLedger
from colonymind.memory import curated
curated.forget_sources(TurnIdempotencyLedger(Path(sys.argv[1])), sys.argv[3], source_ids=[sys.argv[2]])
''', str(ledger.db_path), source, person], check=True, capture_output=True)


@pytest.fixture
def traffic(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    log = CommsLog(str(tmp_path / 'comms.db'), source_ledger=ledger)
    for source in ('first', 'second', 'third', 'seed'):
        ledger.record_source(source, contact_id=OWNER, session_id='fixture-session',
            messages=[{'role': 'user', 'content': source}], derive_claims=False)
        if source != 'seed':
            lineage, _ = log.source_input(source, OWNER)
            log.log(OWNER, channel='fixture', direction='in', summary=source, source_lineage=lineage)
    curated.forget_sources(ledger, OWNER, source_ids=['seed'])
    yield ledger, log
    log._conn.close()


def between(monkeypatch, target, attribute, ledger, enabled=True, *, person=OWNER):
    original = getattr(target, attribute)
    fired = []
    def read(*args, **kwargs):
        if enabled and not fired:
            fired.append(True)
            withdraw(ledger, 'first', person)
        return original(*args, **kwargs)
    monkeypatch.setattr(target, attribute, read)
    return fired


@pytest.mark.parametrize('stage', ['rollup', '_current_summary_rows', None])
def test_recent_snapshot_complete(traffic, monkeypatch, stage):
    ledger, log = traffic
    before = log.recent_snapshot()
    assert len(before['entries']) == before['by_channel']['fixture']['in'] == 3
    fired = between(monkeypatch, CommsLog, stage, ledger) if stage else []
    offered = log.recent_snapshot()
    assert offered == log.recent_snapshot()
    assert bool(fired) is (stage is not None)
    assert len(offered['entries']) == offered['by_channel']['fixture']['in'] == (2 if stage else 3)
    if not stage:
        assert offered == before


@pytest.mark.parametrize('withdrawn', [True, False])
async def test_voice_affect_complete(affect_app, monkeypatch, withdrawn):
    from colonymind.api.routers import host, social_state
    state = affect_app
    monkeypatch.setattr(social_state, 'appraisal_store', lambda: state.appraisals)
    state.ledger.record_source('first', contact_id='owner', session_id='fixture-session',
        messages=[{'role': 'user', 'content': 'Fixture'}], derive_claims=False)
    log = CommsLog(str(state.path / 'comms.db'), source_ledger=state.ledger)
    lineage, _ = log.source_input('first', 'owner')
    state.affect.create_event(contact_id='owner', valence=.8, arousal=.9,
        source_lineage=lineage, timestamp=stamp(state.now))
    add_appraisal(state, name='second')
    # A process launch exceeds the route budget. Hold its logical clock still
    # to test the withdrawal window independently of the existing deadline gate.
    monkeypatch.setattr(host, 'time', SimpleNamespace(time=lambda: state.now, perf_counter=lambda: 0.))
    before = await call(state)
    assert before
    fired = between(monkeypatch, state.appraisals, 'voice_appraisal', state.ledger, withdrawn, person='owner')
    offered = await call(state)
    assert offered == await call(state)
    assert bool(fired) is withdrawn
    assert ('high energy' in offered) is (not withdrawn)
    if not withdrawn:
        assert offered == before
    log._conn.close()


@pytest.mark.parametrize('withdrawn', [True, False])
async def test_commitment_stats_complete(traffic, tmp_path, monkeypatch, withdrawn):
    from colonymind.api.routers import host
    from colonymind.commitments.store import CommitmentStore
    ledger, _ = traffic
    store = CommitmentStore(tmp_path / 'commitments.db')
    item = store.create(person_id=OWNER, description='Fixture', metadata={
        'source_turn': 'first', 'resolution': {'outcome': 'invalid'}})
    store.update(item['id'], status='cancelled')
    monkeypatch.setattr(host, '_commitment_store', store)
    before = await host.commitment_resolution_stats(days=30)
    fired = between(monkeypatch, store, 'recent_rejections', ledger, withdrawn)
    offered = await host.commitment_resolution_stats(days=30)
    assert offered == await host.commitment_resolution_stats(days=30)
    assert bool(fired) is withdrawn
    if not withdrawn:
        assert offered == before


@pytest.mark.parametrize('withdrawn', [True, False])
def test_perspective_status_complete(traffic, monkeypatch, withdrawn):
    from colonymind.self_model.perspective import SelfPerspective
    ledger, _ = traffic
    # Detached producer projections; the real composite and real source fence
    # run unchanged. A native source disappearing invalidates its earlier part.
    def rows(**kwargs):
        return curated.fenced_read(ledger,
            lambda: [] if curated.is_forgotten(ledger, 'first') else [{'id': 1}], empty=[])
    state = SimpleNamespace(ledger=ledger, owner_id=OWNER, preferences=rows,
        judgments=SimpleNamespace(revisions=rows, processing=lambda: []),
        appraisals=SimpleNamespace(view=lambda *a, **k: rows()))
    if hasattr(SelfPerspective, '_read_status'):
        state._read_status = lambda: SelfPerspective._read_status(state)
    fired = between(monkeypatch, state.judgments, 'revisions', ledger, withdrawn)
    offered = SelfPerspective.status(state)
    assert offered == SelfPerspective.status(state)
    assert bool(fired) is withdrawn
    assert bool(offered['preferences']) is (not withdrawn)


@pytest.mark.parametrize('withdrawn', [True, False])
async def test_opinion_view_complete(traffic, monkeypatch, withdrawn):
    from fastapi import HTTPException
    from colonymind.api.routers import opinions
    ledger, _ = traffic
    row = {'id': 1, 'topic': 'fixture', 'supersedes': None, 'audience': 'all'}
    def get(*args, **kwargs):
        return curated.fenced_read(ledger,
            lambda: None if curated.is_forgotten(ledger, 'first') else dict(row), empty=None)
    store = SimpleNamespace(ledger=ledger, get=get, head=get)
    monkeypatch.setattr(opinions, '_opinions', lambda: SimpleNamespace(store=store))
    monkeypatch.setattr(opinions, '_owner', lambda *a: True)
    fired = between(monkeypatch, store, 'head', ledger, withdrawn)
    async def read():
        try:
            return await opinions.show_opinion(1, None)
        except HTTPException as error:
            assert error.status_code == 404
            return None
    offered = await read()
    assert offered == await read()
    assert bool(fired) is withdrawn
    assert bool(offered) is (not withdrawn)


@pytest.mark.parametrize('withdrawn', [True, False])
async def test_opinion_packet_complete(traffic, monkeypatch, withdrawn):
    from colonymind.mind import opinions
    ledger, _ = traffic
    def source(ref):
        return None if curated.is_forgotten(ledger, ref) else {
            'messages': [{'role': 'user', 'content': 'Fixture'}], 'contact_id': OWNER, 'session_id': 'fixture-session'}
    async def semantic(*a):
        return ()
    monkeypatch.setattr(opinions, '_semantic_turn_ids', semantic)
    monkeypatch.setattr(opinions, '_premise_entries', lambda *a, **k: ({}, [], []))
    store = SimpleNamespace(ledger=ledger, owner_id=OWNER, source=source,
        admitted_premises=lambda ref: [SimpleNamespace(corrects=())], statements=lambda ref: [], relevant=lambda *a, **k: [])
    fired = between(monkeypatch, store, 'relevant', ledger, withdrawn)
    offered = await opinions.build_packet(store, {'ref': 'first', 'kind': 'turn'})
    again = await opinions.build_packet(store, {'ref': 'first', 'kind': 'turn'})
    assert offered == again
    assert bool(fired) is withdrawn
    assert (offered == 'source_erased') is withdrawn


@pytest.mark.parametrize('withdrawn', [True, False])
def test_annotation_candidates_complete(traffic, monkeypatch, withdrawn):
    from colonymind.turns import source_annotations as annotations
    ledger, _ = traffic
    candidate = {'id': 'fixture', 'source_id': 'first', '_annotation_source_refs': [{'source_id': 'first'}]}
    def current(*a, **k):
        # No correction changed; only the live source withdrawal between the
        # two fences invalidates the initially filtered row.
        return [candidate]
    monkeypatch.setattr(annotations, 'current_candidates_in_connection', current)
    fired = between(monkeypatch, annotations, 'current_candidates_in_connection', ledger, withdrawn)
    def read():
        return annotations.current_candidates(ledger, [candidate], contact_id=OWNER, session_id='fixture-session')
    offered = read()
    assert offered == read()
    assert bool(fired) is withdrawn
    assert bool(offered) is (not withdrawn)


@pytest.mark.parametrize('method', ['snapshot', 'current_records'])
@pytest.mark.parametrize('withdrawn', [True, False])
def test_reply_timing_complete(traffic, monkeypatch, method, withdrawn):
    from colonymind.instinct.outcomes import Sources
    from colonymind.plasticity.learners.contact_reply_timing import Rows, Driver
    ledger, _ = traffic
    sources = Sources(ledger.db_path.parent, owner_id=OWNER)
    rows = Rows(SimpleNamespace(), sources, lambda: 0.)
    rows.pending_frames = {'first': {'fixture': 1}}
    rows.dependencies = {'first': ('first',)}
    if method == 'snapshot':
        def query(filename, *a):
            return [{'id': 'first', 'contact': 'cid-first', 'direction': 'in', 'channel': 'whatsapp',
                     'at': '2026-01-01T00:00:00+00:00', 'turn': 'first'}]
        monkeypatch.setattr(rows, 'query', query)
        monkeypatch.setattr(rows, 'contacts', lambda: ['cid-first'])
        # Withdrawal after the erasure reader returned, before the independent
        # withheld/native read; no data_version mocking.
        fired = between(monkeypatch, sources, 'withheld', ledger, withdrawn)
        read = rows.snapshot
    else:
        monkeypatch.setattr(rows, 'snapshot', lambda: ([], []))
        driver = object.__new__(Driver)
        driver.rows = rows
        driver.store = SimpleNamespace(source_refs=lambda: {})
        comparator = SimpleNamespace(_path=lambda ref: Path(ref), _hash=lambda data: 'hash')
        fired = between(monkeypatch, sources, 'withheld', ledger, withdrawn)
        read = lambda: driver.current_records([{'ref_hash': 'first', 'input_hash': 'hash'}], comparator)
    offered = read()
    assert offered == read()
    assert bool(fired) is withdrawn
    assert bool(offered[1] if method == 'snapshot' else offered) is (not withdrawn)


@pytest.mark.parametrize('withdrawn', [True, False])
def test_reply_parent_current_complete(traffic, tmp_path, monkeypatch, withdrawn):
    from colonymind.api.routers import temporal_followups
    from colonymind.commitments.store import CommitmentStore
    from colonymind.self_model import reply_forecasts
    ledger, _ = traffic
    store = CommitmentStore(tmp_path / 'commitments.db')
    parent = store.create(person_id=OWNER, description='Fixture', metadata={'source_turn': 'first'})
    with closing(store._connect()) as conn, conn:
        conn.execute('CREATE TABLE IF NOT EXISTS temporal_followups (wait_id TEXT, material_digest TEXT)')
        conn.execute("INSERT INTO temporal_followups(wait_id,material_digest) VALUES ('wait','digest')")
    monkeypatch.setattr(temporal_followups, 'source_bindings', lambda *a, **k: set())
    fired = between(monkeypatch, temporal_followups, 'source_bindings', ledger, withdrawn)
    conditions = {'commitment_id': parent['id'], 'wait_id': 'wait', 'wait_material_digest': 'digest',
        'parent_sources': {}, 'parent_session_id': 'fixture-session'}
    read = lambda: reply_forecasts._parent_current(ledger, store, conditions, OWNER)
    offered = read()
    assert offered == read()
    assert bool(fired) is withdrawn
    assert offered is (not withdrawn)


@pytest.mark.parametrize('withdrawn', [True, False])
def test_reply_evidence_current_complete(traffic, monkeypatch, withdrawn):
    from colonymind.self_model import reply_forecasts
    ledger, log = traffic
    first = ledger.source_references(['first'], contact_id=OWNER, session_id='fixture-session')[0]
    prediction = SimpleNamespace(subject_person_id=OWNER, viewer_scope=OWNER, shareability='owner_private',
        evidence_refs=['first'], detail={'method': reply_forecasts.METHOD, 'conditions': {},
        'source_versions': {'receipt:first': first['source_version']}, 'forecast_id': 'fixture'})
    monkeypatch.setattr(reply_forecasts, '_parts', lambda: (None, ledger, None, log, OWNER))
    monkeypatch.setattr(reply_forecasts, '_parent_current', lambda *a: True)
    monkeypatch.setattr(reply_forecasts, '_metadata', lambda *a: {'method': reply_forecasts.METHOD, 'kind': 'wait_stopped'})
    fired = between(monkeypatch, reply_forecasts, '_metadata', ledger, withdrawn)
    read = lambda: reply_forecasts.evidence_current(prediction,
        {'receipt_ref': 'receipt:second', 'evidence_refs': [], 'source_versions': {}, 'status': 'censored'})
    offered = read()
    assert offered == read()
    assert bool(fired) is withdrawn
    assert offered is (not withdrawn)


@pytest.mark.parametrize('churn', [False, True])
async def test_async_fence_bounds_churn(traffic, churn):
    ledger, _ = traffic
    calls = []
    async def read():
        calls.append(True)
        if churn:
            ledger.record_source('marker-' + str(len(calls)), contact_id=OWNER, session_id='fixture-session',
                messages=[{'role': 'user', 'content': 'Fixture'}], derive_claims=False)
        return 'stable'
    assert await curated.fenced_read(ledger, read, empty='empty') == ('empty' if churn else 'stable')
    assert len(calls) == (3 if churn else 1)


@pytest.mark.parametrize('churn', [False, True])
async def test_nested_fence_ignores_ordinary_churn_inside_the_envelope(traffic, churn):
    # A hot assembly's own envelope fences withdrawals; a nested read is not withheld for an unrelated commit.
    ledger, _ = traffic
    calls = []
    async def read():
        calls.append(True)
        if churn:
            ledger.record_source('marker-' + str(len(calls)), contact_id=OWNER, session_id='fixture-session',
                messages=[{'role': 'user', 'content': 'Fixture'}], derive_claims=False)
        return 'stable'
    async def hot():
        return await curated.fenced_read(ledger, read, empty='empty')
    assert await curated.withdrawal_fenced_read(ledger, hot, empty='outer') == 'stable'
    assert len(calls) == 1


@pytest.mark.parametrize('withdrawn', [True, False])
async def test_commitment_list_complete(traffic, tmp_path, monkeypatch, withdrawn):
    from colonymind.api.routers import host
    from colonymind.commitments.store import CommitmentStore
    ledger, _ = traffic
    store = CommitmentStore(tmp_path / 'commitments.db')
    store.create(person_id=OWNER, description='Fixture', metadata={'source_turn': 'first'})
    with closing(store._connect()) as conn, conn:
        conn.execute("UPDATE commitments SET due_at='2000-01-01T00:00:00+00:00'")
    monkeypatch.setattr(host, '_commitment_store', store)
    fired = between(monkeypatch, store, 'list', ledger, withdrawn)
    async def read():
        return await host.list_commitments(person_id=OWNER, status_filter='overdue,fulfilled',
            overdue_only=False, limit=50, offset=0)
    offered = await read()
    assert offered == await read()
    assert bool(fired) is withdrawn
    assert bool(offered.commitments) is (not withdrawn)


@pytest.mark.parametrize('withdrawn', [True, False])
def test_qualification_perspective_complete(traffic, monkeypatch, withdrawn):
    from colonymind.qualification.native_perspective import snapshot
    from colonymind.self_model import appraisals
    ledger, _ = traffic
    checks, fired = [], []
    def view(*a, **k):
        checks.append(True)
        if withdrawn and len(checks) == 2:
            withdraw(ledger, 'first')
            fired.append(True)
        return [] if curated.is_forgotten(ledger, 'first') else [{'id': 1}]
    monkeypatch.setattr(appraisals, 'AppraisalStore', lambda *a, **k: SimpleNamespace(view=view))
    read = lambda: snapshot(ledger, OWNER, OWNER)
    offered = read()
    assert offered == read()
    assert bool(fired) is withdrawn
    assert bool(offered['appraisals']) is (not withdrawn)


@pytest.mark.parametrize('withdrawn', [True, False])
def test_qualification_recollect_complete(traffic, monkeypatch, withdrawn):
    from colonymind.memory import recall
    from colonymind.qualification.memory_cases import _recollect
    ledger, _ = traffic
    row = {'id': 'first', 'kind': 'source_quote', 'content': 'Fixture', 'source_turn_ids': ['first']}
    def prepare(*a, **k):
        return [], [] if curated.is_forgotten(ledger, 'first') else [row]
    projection = SimpleNamespace(ledger=ledger, prepare_context=prepare)
    fired = between(monkeypatch, recall, 'pack_memory_context', ledger, withdrawn)
    read = lambda: _recollect(projection, 'Fixture', contact=OWNER, session='fixture-session',
        now='2026-10-20T00:00:00+00:00', max_chars=500)
    offered = read()
    assert offered == read()
    assert bool(fired) is withdrawn
    assert bool(offered['selected']) is (not withdrawn)


@pytest.mark.parametrize('withdrawn', [True, False])
def test_time_lane_complete(traffic, monkeypatch, withdrawn):
    from colonymind.memory import time_lane
    ledger, _ = traffic
    digest = 'a' * 64
    checks, fired = [], []
    def claims(*a, **k):
        return [] if curated.is_forgotten(ledger, 'first') else [
            {'turn_id': 'first', 'message_hash': digest}]
    def prepare(*a, **k):
        checks.append(True)
        if withdrawn and len(checks) == 1:
            withdraw(ledger, 'first')
            fired.append(True)
        # A surviving independent result was selected using the earlier
        # outside-window cohort. It must be recomputed with that cohort.
        return [], [{'id': 'second', 'source_turn_id': 'first', 'source_message_hash': digest}]
    projection = SimpleNamespace(ledger=ledger, _rows=claims, prepare_context=prepare)
    monkeypatch.setattr(time_lane, '_dated', lambda *a: True)
    read = lambda: time_lane.outside_window(projection, [{'turn_id': 'first', 'source_message_hash': digest}], [],
        contact_id=OWNER, session_id='fixture-session', time_query=SimpleNamespace(mode='at', accepts_claim=lambda c: False),
        now=datetime.now(timezone.utc))
    offered = read()
    assert offered == read()
    assert bool(fired) is withdrawn


@pytest.mark.parametrize('withdrawn', [True, False])
def test_work_report_complete(work_stores, monkeypatch, withdrawn):
    from colonymind.work.store import WorkStore, read_summary
    state = work_stores
    for source in ('first', 'second'):
        admit(state, source_id=source)
        assert work_root(state, source)
    original = WorkStore._public_root
    fired = []
    def public(row, *args, **kwargs):
        result = original(row, *args, **kwargs)
        if withdrawn and not fired:
            fired.append(True)
            withdraw(state.ledger, row['source_id'], 'owner')
        return result
    monkeypatch.setattr(WorkStore, '_public_root', staticmethod(public))
    read = lambda: read_summary(state.work.path, contact_id='owner')
    offered = read()
    assert offered == read()
    assert bool(fired) is withdrawn


@pytest.mark.parametrize('withdrawn', [True, False])
async def test_search_selection_complete(traffic, monkeypatch, withdrawn):
    from colonymind.memory import search
    ledger, _ = traffic
    collected = SimpleNamespace(ledger=ledger, contact_id=OWNER,
        watermark=ledger.erasure_watermark(OWNER), curated_recall=None)
    fired = []
    async def select(*a, **k):
        content = '' if curated.is_forgotten(ledger, 'first') else 'Fixture'
        if withdrawn and not fired:
            fired.append(True)
            withdraw(ledger, 'first')
        return search.MemoryPacket([], content, [], collected.watermark, {}, [])
    monkeypatch.setattr(search, '_select_memory', select)
    read = lambda: search.select_memory(collected, query='Fixture', selector=None)
    offered = await read()
    assert offered == await read()
    assert bool(fired) is withdrawn
    assert bool(offered.content) is (not withdrawn)


@pytest.mark.parametrize('site', ['sections', 'context'])
@pytest.mark.parametrize('withdrawn', [True, False])
async def test_context_computation_complete(traffic, monkeypatch, site, withdrawn):
    from colonymind.api.routers import host
    from colonymind.memory import turn_budget
    ledger, _ = traffic
    fired = []
    monkeypatch.setattr(host, 'get_state_dir', lambda: ledger.db_path.parent)
    async def lanes(*a, **k):
        content = '' if curated.is_forgotten(ledger, 'first') else 'Fixture'
        if withdrawn and not fired:
            fired.append(True)
            withdraw(ledger, 'first')
        return [host.ContextSection(id='fixture', body=content)] if content else []
    if site == 'sections':
        monkeypatch.setattr(host, '_assembled_lanes', lanes)
        read = lambda: host._assemble_sections(None, viewer_person_id=OWNER)
    else:
        monkeypatch.setattr(host, '_assemble_sections', lanes)
        monkeypatch.setattr(host, 'resolve_request_person', lambda *a, **k: OWNER)
        monkeypatch.setattr(host, 'request_authority', lambda *a: SimpleNamespace(authenticated=True, anonymous=False))
        monkeypatch.setattr(host, '_turn_context_corrections', lambda body, sections, *a, **k: sections)
        async def choose(body, sections, **k):
            return sections
        monkeypatch.setattr(host, '_turn_context_selection', choose)
        monkeypatch.setattr(host, '_fit_sections', lambda sections, **k: (sections, None))
        monkeypatch.setattr(host, '_viewer_is_guest', lambda *a: False)
        monkeypatch.setattr(host, '_telemetry', None)
        monkeypatch.setattr(turn_budget, 'defer_write', lambda *a, **k: None)
        body = host.ContextAssembleRequest(identity={'host_id': 'fixture'},
            context={'host_id': 'fixture', 'contact_id': OWNER, 'session_id': 'fixture-session'},
            incoming_message={'role': 'user', 'content': 'Fixture'})
        read = lambda: host._context_assemble(body)
    with turn_budget.turn(turn_budget.TurnBudget(wait_s=30.)):
        offered = await read()
        again = await read()
    if site == 'context' and withdrawn:
        assert offered.sections == again.sections == []
        assert offered.source_erasure_watermark < again.source_erasure_watermark
    else:
        assert offered == again
    assert bool(fired) is withdrawn


@pytest.mark.parametrize('withdrawn', [True, False])
def test_input_excerpt_final_preparation(traffic, monkeypatch, withdrawn):
    from colonymind.turns import source_read
    from colonymind.turns.idempotency import source_message_hash
    ledger, _ = traffic
    refs = [{'source_id': 'first', 'input_message_hash': source_message_hash('fixture-session',
        {'role': 'user', 'content': 'first'})}]
    fired = between(monkeypatch, source_read, 'inputs_unannotated', ledger, withdrawn)
    def read():
        try:
            return source_read.input_excerpt(ledger, contact_id=OWNER, session_id='fixture-session', refs=refs)
        except ValueError:
            return None
    offered = read()
    assert offered == read()
    assert bool(fired) is withdrawn
    assert bool(offered) is (not withdrawn)


@pytest.mark.parametrize('withdrawn', [True, False])
def test_context_card_final_preparation(real_inputs, monkeypatch, withdrawn):
    from colonymind.context_card import inputs
    ledger, collector, _ = real_inputs
    prepared = collector.prepare()
    person = prepared.inputs.owner_id
    ledger.record_source('first', contact_id=person, session_id='fixture-session',
        messages=[{'role': 'user', 'content': 'Fixture'}], derive_claims=False)
    original = inputs.replace
    fired = []
    def replace(*a, **k):
        result = original(*a, **k)
        if withdrawn and not fired:
            fired.append(True)
            withdraw(ledger, 'first', person)
        return result
    monkeypatch.setattr(inputs, 'replace', replace)
    try:
        offered = collector.collect_ledger(prepared)
    except ValueError:
        offered = None
    assert bool(fired) is withdrawn
    assert (offered is None) is withdrawn


@pytest.mark.parametrize('withdrawn', [True, False])
async def test_inline_memory_final_preparation(traffic, monkeypatch, withdrawn):
    from colonymind.api.routers import host
    from colonymind.memory import recall
    from colonymind.memory.search import MemoryPacket
    from colonymind.memory.selection import Offered
    from colonymind.memory.trace import NULL_TRACE
    ledger, _ = traffic
    refs = ledger.source_references(['first'], contact_id=OWNER, session_id='fixture-session')
    row = {'id': 'first', 'kind': 'source_quote', 'content': 'Fixture', 'source_turn_ids': ['first']}
    packet = MemoryPacket([row], 'Fixture', refs, ledger.erasure_watermark(OWNER), {}, [])
    recalled = SimpleNamespace(packet=packet, offered=Offered((row,), 5, 500), ledger=ledger,
        contact_id=OWNER, session_id='fixture-session', plain_recall=None)
    sections = [host.ContextSection(id='colonymind-memory', title='Memory', body='Fixture ' * 100)]
    fired = between(monkeypatch, recall, 'packet_refs', ledger, withdrawn)
    read = lambda: host._memory_within_allowance(sections, recalled, inline_chars=300, trace=NULL_TRACE)
    offered = await read()
    assert offered == await read()
    assert bool(fired) is withdrawn
    assert bool(offered[1]) is (not withdrawn)

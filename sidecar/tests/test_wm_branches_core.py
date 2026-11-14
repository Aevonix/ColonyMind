"""WM3 record, consult, erasure and live comparator acceptance tests."""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from types import SimpleNamespace

import pytest

from colonymind.memory.branches import BranchStore, BranchRefused
from colonymind.turns.idempotency import TurnIdempotencyLedger


@pytest.fixture
def store(tmp_path):
    ledger = TurnIdempotencyLedger(tmp_path / 'turn-idempotency.db')
    return BranchStore(ledger, key=b'fixture-key', clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))


def capture(store, child='child', **kwargs):
    return store.capture(parent_session='parent', child_session=child, requester='owner',
                         brief='Read these records', summary='The worker finished',
                         messages=[{'role': 'user', 'content': 'The badge is birch-741'},
                                   {'role': 'tool', 'content': 'API_KEY=sk-' + 'Q1aB' * 8},
                                   {'role': 'assistant', 'content': 'The worker finished'}], **kwargs)


class Router:
    def __init__(self):
        self.calls = []

    async def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return SimpleNamespace(content='The badge is birch-741 [span]', tool_calls=[])


def test_ordered_redacted_ledger_records_and_owner_scope(store):
    branch = capture(store)
    assert branch['id'].startswith('br-') and len(branch['id']) == 15
    with closing(store.ledger._connect()) as conn:
        for span in branch['spans']:
            row = conn.execute('SELECT * FROM turn_sources WHERE turn_id=?', (span['record_id'],)).fetchone()
            message = json.loads(row['messages_json'])[0]
            assert message['source_kind'] == 'branch_span'
            assert message['epistemic_state'] == 'derived_unverified'
            assert row['contact_id'] == 'owner'
            text = message['content'][0]['text']
            assert span['content_hash'] == hashlib.sha256(text.encode()).hexdigest()
        raw = '\n'.join(row[0] for row in conn.execute('SELECT messages_json FROM turn_sources'))
        assert 'sk-' + 'Q1aB' * 8 not in raw
    assert len(branch['spans']) == 4
    assert capture(store)['id'] == branch['id']
    for requester in ('guest', 'contact'):
        with pytest.raises(BranchRefused, match='owner_only'):
            store.capture(parent_session='parent', child_session='other', requester=requester,
                          brief='brief', summary='summary', messages=[])
        with pytest.raises(BranchRefused, match='owner_only'):
            store.list_branches('parent', requester=requester)


def test_branch_material_never_enters_recall_projection_or_vector_stores(store):
    from colonymind.turns.source_vectors import chunks
    branch = capture(store)
    store.ledger.rebuild_keyword_stems()
    with closing(store.ledger._connect()) as conn:
        for span in branch['spans']:
            source = dict(conn.execute('SELECT * FROM turn_sources WHERE turn_id=?', (span['record_id'],)).fetchone())
            assert list(chunks(conn, source)) == []
        for table in ('turn_source_search', 'turn_source_stems', 'source_claims', 'source_claim_jobs', 'source_vector_jobs',
                      'opinion_jobs', 'appraisal_runs', 'appraisal_records', 'commitment_runs'):
            assert conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0] == 0
    assert store.ledger.search_sources('birch', contact_id='owner', session_id='parent') == []


@pytest.mark.asyncio
async def test_canonical_combined_render_and_no_tools(store):
    first, second = capture(store), capture(store, 'other')
    router = Router()
    for question in ('What badge?', 'Repeat the badge'):
        await store.consult([first['id']], question, requester='owner', session_id='parent', router=router)
    primary = [call for call in router.calls if 'Branch material' in call[0][0]['content']]
    assert primary[0][0][0] == primary[1][0][0]
    router.calls.clear()
    await store.consult([second['id'], first['id']], 'Compare', requester='owner', session_id='parent', router=router)
    prompt, kwargs = router.calls[0]
    assert first['id'] in prompt[0]['content'] and second['id'] in prompt[0]['content']
    assert prompt[0]['content'].index(min(first['id'], second['id'])) < prompt[0]['content'].index(max(first['id'], second['id']))
    assert kwargs['tools'] == [] and kwargs['context']['task'] == 'consult'
    store.token_cap = 1
    with pytest.raises(BranchRefused) as caught:
        await store.consult([first['id']], 'Read', requester='owner', session_id='parent', router=router)
    assert caught.value.reason == 'token_cap' and caught.value.handles == [first['id']]


@pytest.mark.asyncio
async def test_budget_persists_and_owner_consult_only(store):
    branch, router = capture(store), Router()
    with pytest.raises(BranchRefused, match='owner_only'):
        await store.consult([branch['id']], 'Read', requester='guest', session_id='parent', router=router)
    for _ in range(20):
        await store.consult([branch['id']], 'Read', requester='owner', session_id='parent', router=router)
    reopened = BranchStore(store.ledger, key=b'fixture-key', clock=store.clock)
    with pytest.raises(BranchRefused, match='hourly_budget'):
        await reopened.consult([branch['id']], 'Read', requester='owner', session_id='parent', router=router)


def test_sampling_hashes_only_withdrawal_reset_and_no_false_decline(store):
    refs = [str(i) for i in range(1000)]
    expected = [r for r in refs if int(store.keyed_hash(['sample', r]), 16) % 10 == 0]
    assert [r for r in refs if store.sampled(r)] == expected
    assert 60 < len(expected) < 140
    for index in range(30):
        ref = f'pair-{index}'
        store.record_pair(ref, session_id='parent', question='PRIVATE QUESTION',
                          answer='PRIVATE ANSWER', reference_answer='PRIVATE SUMMARY ANSWER')
        store.outcome(ref, served_correct=False, reference_correct=True)
    report = store.report()
    assert report['withdrawn'] and report['comparator']['lower_bound'] > report['comparator']['bar']
    assert 'regression' in report['withdrawal_reason']
    with closing(store.ledger._connect()) as conn:
        rows = conn.execute('SELECT * FROM wm_consults').fetchall()
        assert 'PRIVATE' not in str([tuple(row) for row in rows])
    store.reset()
    assert not store.withdrawn
    store.record_pair('same', session_id='parent', question='q', answer='same', reference_answer='same')
    store.outcome('same', served_correct=True, reference_correct=True)
    assert not store.withdrawn


def test_reaction_only_labels_next_turn_and_never_invents_reference_success(store):
    store.record_pair('ref', session_id='parent', question='q', answer='answer', reference_answer='summary')
    store.reaction(session_id='parent', requester='owner', text='Not helpful.', turn_id='next')
    assert store.report()['comparator']['labelled'] == 0
    store.reaction(session_id='parent', requester='owner', text='Keep going.', turn_id='later')
    assert store.report()['comparator']['labelled'] == 0


def test_faculty_off_has_no_capture_or_render(store):
    store.enabled = False
    with pytest.raises(BranchRefused, match='disabled'):
        capture(store)
    assert store.list_branches('parent', requester='owner') == []
    assert store.report()['branches'] == 0


@pytest.mark.asyncio
async def test_endpoint_cache_counts_reach_usage_db_and_readonly_report(store):
    from colonymind.config import Config
    from colonymind.memory.branches import read_report

    class CountedRouter(Router):
        async def complete(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return SimpleNamespace(content='answer', tool_calls=[], raw=SimpleNamespace(usage=SimpleNamespace(
                prompt_tokens=100, prompt_tokens_details=SimpleNamespace(cached_tokens=75))))

    branch = capture(store)
    await store.consult([branch['id']], 'Read', requester='owner', session_id='parent', router=CountedRouter())
    assert read_report(Config(home=store.ledger.db_path.parent))['cached_token_share'] == 0.75
    import sqlite3
    with closing(sqlite3.connect(store.ledger.db_path.parent / 'usage.db')) as conn:
        assert conn.execute('SELECT prompt_tokens,cached_tokens FROM wm_consult_cache').fetchone() == (100, 75)


def test_capture_failure_leaves_no_partial_branch_or_source(store, monkeypatch):
    original = store.ledger.record_source
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('capture interrupted')
        return original(*args, **kwargs)

    monkeypatch.setattr(store.ledger, 'record_source', fail_second)
    with pytest.raises(RuntimeError, match='interrupted'):
        capture(store)
    assert store.list_branches('parent', requester='owner') == []
    with closing(store.ledger._connect()) as conn:
        assert not conn.execute('SELECT 1 FROM turn_sources').fetchone()


@pytest.mark.asyncio
async def test_next_turn_explicit_pair_ratings_withdraw_on_the_live_consult_path(store, monkeypatch):
    monkeypatch.setattr(store, 'sampled', lambda ref: True)
    branch = capture(store)

    class PairedRouter(Router):
        async def complete(self, messages, **kwargs):
            return SimpleNamespace(content=('Full detail answer' if 'Branch material' in messages[0]['content']
                                             else 'Summary answer'), tool_calls=[])

    for index in range(10):
        await store.consult([branch['id']], 'Read', requester='owner', session_id='parent', router=PairedRouter())
        store.reaction(session_id='parent', requester='owner', turn_id=f'turn-{index}',
                       text=json.dumps({'answer': 'Not helpful.', 'summary': 'Keep going.'}))
        if store.withdrawn:
            break
    assert store.withdrawn
    assert store.report()['comparator']['losses'] > 0
    with pytest.raises(BranchRefused, match='withdrawn'):
        await store.consult([branch['id']], 'Read', requester='owner', session_id='parent', router=PairedRouter())


@pytest.mark.asyncio
async def test_refuses_raw_tool_calls_even_when_flat_response_has_none(store):
    branch = capture(store)

    class ToolRouter(Router):
        async def complete(self, messages, **kwargs):
            return SimpleNamespace(content='no answer', raw=SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(tool_calls=[{'function': {'name': 'unexpected'}}]))]))

    with pytest.raises(BranchRefused, match='tools_refused'):
        await store.consult([branch['id']], 'Read', requester='owner', session_id='parent', router=ToolRouter())


def test_capture_interruption_after_source_commit_does_not_leave_unmanifested_text(store, monkeypatch):
    original = store.ledger.record_source

    def committed_then_failed(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('capture interrupted after source commit')

    monkeypatch.setattr(store.ledger, 'record_source', committed_then_failed)
    with pytest.raises(RuntimeError, match='interrupted'):
        capture(store)
    with closing(store.ledger._connect()) as conn:
        assert not conn.execute('SELECT 1 FROM turn_sources').fetchone()
    assert store.list_branches('parent', requester='owner') == []


@pytest.mark.asyncio
async def test_unwritable_cache_telemetry_does_not_drop_a_successful_consult(store, monkeypatch):
    branch = capture(store)
    from colonymind.memory import branches

    original = branches.sqlite3.connect

    def broken(path, *args, **kwargs):
        if str(path).endswith('usage.db'):
            raise branches.sqlite3.OperationalError('usage database unavailable')
        return original(path, *args, **kwargs)

    class CountedRouter(Router):
        async def complete(self, messages, **kwargs):
            return SimpleNamespace(content='answer', raw={'usage': {
                'prompt_tokens': 100, 'prompt_tokens_details': {'cached_tokens': 50}}})

    monkeypatch.setattr(branches.sqlite3, 'connect', broken)
    assert (await store.consult([branch['id']], 'Read', requester='owner', session_id='parent',
                                router=CountedRouter()))['answer'] == 'answer'


def test_retried_owner_turn_cannot_label_a_later_consult(store):
    store.record_pair('first', session_id='parent', question='q', answer='answer', reference_answer='summary')
    rating = json.dumps({'answer': 'Not helpful.', 'summary': 'Keep going.'})
    store.reaction(session_id='parent', requester='owner', turn_id='owner-turn', text=rating)
    store.record_pair('second', session_id='parent', question='q', answer='answer', reference_answer='summary')
    store.reaction(session_id='parent', requester='owner', turn_id='owner-turn', text=rating)
    assert store.report()['comparator']['labelled'] == 1


def test_changed_span_retains_its_expiry_even_after_leaving_the_manifest(store):
    branch = capture(store)
    record_id = branch['spans'][1]['record_id']
    with closing(store.ledger._connect()) as conn, conn:
        conn.execute('UPDATE turn_sources SET messages_json=? WHERE turn_id=?',
                     (json.dumps([{'role': 'tool', 'source_kind': 'branch_span',
                                   'content': [{'type': 'branch_span', 'text': 'Remaining detail'}]}]), record_id))
    assert record_id not in {span['record_id'] for span in store.list_branches('parent', requester='owner')[0]['spans']}
    store.clock = lambda: datetime(2026, 1, 16, tzinfo=timezone.utc)
    store.prune()
    with closing(store.ledger._connect()) as conn:
        assert not conn.execute('SELECT 1 FROM turn_sources').fetchone()


def test_unchanged_source_update_preserves_the_branch(store):
    branch = capture(store)
    before = store.render([branch['id']], requester='owner')
    with closing(store.ledger._connect()) as conn, conn:
        conn.execute('UPDATE turn_sources SET contact_id=contact_id')
    assert store.render([branch['id']], requester='owner') == before


def test_direct_capture_masks_encoded_tool_arguments_before_the_ledger(store):
    secret = 'synthetic-value-8412'
    branch = store.capture(parent_session='parent', child_session='child', requester='owner', messages=[{
        'role': 'assistant', 'content': '', 'tool_calls': [{'function': {
            'name': 'read_report', 'arguments': json.dumps({'password': secret, 'report': 'weekly'})}}]}])
    assert secret not in store.render([branch['id']], requester='owner')
    assert secret.encode() not in store.ledger.db_path.read_bytes()


def test_capture_limit_applies_after_redaction_too(store, monkeypatch):
    from colonymind.memory import branches
    monkeypatch.setattr(branches, 'MAX_SPAN_BYTES', 64)
    with pytest.raises(BranchRefused, match='capture_size'):
        store.capture(parent_session='parent', child_session='child', requester='owner',
                      messages=[{'role': 'tool', 'content': 'password: 8412\n' * 4}])
    assert store.report()['spans'] == 0


@pytest.mark.asyncio
async def test_vector_backfill_cannot_project_branch_material(store):
    from colonymind.turns.source_vectors import SourceVectors
    writes = []

    async def forbidden_write(*args, **kwargs):
        writes.append((args, kwargs))
        raise AssertionError('branch material reached the embedding or vector store')

    # Exercise the real backfill/worker with an observed index boundary. It must
    # never call the embedder or open a vector table for a branch-only ledger.
    identity = object()
    vectors = SimpleNamespace(identity=identity, _table=forbidden_write,
                              catalog=SimpleNamespace(write_generation=lambda identity: {'id': 'fixture'}))
    pipeline = SimpleNamespace(index_identity=identity, embed_batch=forbidden_write)
    projection = SourceVectors(store.ledger, vectors, pipeline)
    branch = capture(store)
    projection.backfill()
    for _ in branch['spans']:
        assert await projection.process_one()
    assert not await projection.process_one()
    assert writes == []
    with closing(store.ledger._connect()) as conn:
        assert not conn.execute("SELECT 1 FROM source_vector_jobs WHERE status!='complete' OR error IS NOT NULL").fetchone()
        for table in ('source_claims', 'source_claim_jobs', 'opinion_jobs', 'appraisal_records', 'commitment_runs'):
            assert conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0] == 0


from colonymind.memory import curated
def test_source_erasure_and_retention(store):
    branch = capture(store)
    erased = branch['spans'][1]['record_id']
    curated.forget_sources(store.ledger, 'owner', source_ids=[erased])
    assert 'birch-741' not in store.render([branch['id']], requester='owner')
    curated.forget_sources(store.ledger, 'owner', source_ids=[s['record_id'] for s in branch['spans'] if s['record_id'] != erased])
    assert store.list_branches('parent', requester='owner') == []
    store.ledger = TurnIdempotencyLedger(store.ledger.db_path.parent / 'retention.db')
    store = BranchStore(store.ledger, key=b'fixture-key', clock=store.clock)
    capture(store, 'later')
    store.clock = lambda: datetime(2026, 1, 15, 0, 0, 1, tzinfo=timezone.utc)
    store.prune()
    assert store.list_branches('parent', requester='owner') == []
    with closing(store.ledger._connect()) as conn:
        assert not conn.execute('SELECT 1 FROM turn_sources').fetchone()

def test_parent_erasure_reaches_branch_and_session_manifest_has_no_copy(store):
    store.ledger.record_source('parent-turn', contact_id='owner', session_id='parent', messages=[{'role': 'user', 'content': 'Parent instruction'}])
    branch = capture(store)
    session = store.session_branch('parent', requester='owner')
    assert session['spans'][0]['record_id'] == 'parent-turn'
    curated.forget_sources(store.ledger, 'owner', source_ids=['parent-turn'])
    assert store.list_branches('parent', requester='owner') == []
    with closing(store.ledger._connect()) as conn:
        assert conn.execute('SELECT 1 FROM turn_sources WHERE turn_id=?', (branch['spans'][0]['record_id'],)).fetchone()

@pytest.mark.asyncio
async def test_inflight_erasure_never_returns_an_answer_from_erased_material(store):
    branch = capture(store)

    class ErasingRouter(Router):

        async def complete(self, messages, **kwargs):
            curated.forget_sources(store.ledger, 'owner', source_ids=[branch['spans'][1]['record_id']])
            return SimpleNamespace(content='This contains erased evidence', tool_calls=[])
    with pytest.raises(BranchRefused, match='source_changed'):
        await store.consult([branch['id']], 'Read', requester='owner', session_id='parent', router=ErasingRouter())

def test_parent_captured_after_child_still_erases_child_and_blocks_replay(store):
    branch = capture(store)
    store.ledger.record_source('late-parent', contact_id='owner', session_id='parent', messages=[{'role': 'user', 'content': 'Delegate the sample task'}])
    curated.forget_sources(store.ledger, 'owner', source_ids=['late-parent'])
    assert store.list_branches('parent', requester='owner') == []
    with closing(store.ledger._connect()) as conn:
        assert conn.execute('SELECT 1 FROM turn_sources WHERE turn_id=?', (branch['spans'][1]['record_id'],)).fetchone()
    with pytest.raises(BranchRefused, match='erased_branch'):
        capture(store)

def test_erasing_all_spans_refuses_recapture_as_a_permanent_capture_refusal(store):
    branch = capture(store)
    curated.forget_sources(store.ledger, 'owner', source_ids=[span['record_id'] for span in branch['spans']])
    with pytest.raises(BranchRefused, match='erased_branch'):
        capture(store)
    assert store.list_branches('parent', requester='owner') == []

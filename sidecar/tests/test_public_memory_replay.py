"""Replay ingestion on an absolute clock, and the recall-at-budget runner (``benchmarks/public_memory``).

The replay drives the sidecar itself (``colonymind.server.create_app`` and its lifespan, in process) with a scripted
model: nothing here reaches a model endpoint, an embedder or a reranker.
"""
import asyncio
from contextlib import closing
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from types import SimpleNamespace

import pytest

from colonymind.qualification import paired_body

T0 = 1_683_000_000.0          # 2023-05-02, well before the real clock


@pytest.fixture
def clock():
    """The shifted wall clock, always put back."""
    try:
        yield paired_body
    finally:
        paired_body.uninstall_clock()


# 1. The clock rules.

def test_the_replay_clock_moves_back_once_before_any_state_and_only_forward_after(clock, tmp_path):
    state = tmp_path / 'replay'
    assert clock.set_clock(T0, state=state) == T0
    assert abs(time.time() - T0) < 5
    (state / 'instance').mkdir(parents=True)
    (state / 'instance' / 'api.key').write_text('fixture\n')
    # A store or a credential exists now: an earlier time would put its own rows in the future.
    with pytest.raises(clock.ClockRefused):
        clock.set_clock(T0 - 3600, state=state)
    assert time.time() >= T0
    assert clock.set_clock(T0 + 86400, state=state) == T0 + 86400
    assert time.time() >= T0 + 86400
    # Moving to the clock's own time or earlier without the rewind is refused as well.
    with pytest.raises(clock.ClockRefused):
        clock.set_clock(T0 + 60, state=state)


def test_the_rewind_is_taken_once_even_while_the_state_is_empty(clock, tmp_path):
    state = tmp_path / 'replay'
    clock.set_clock(T0, state=state)
    with pytest.raises(clock.ClockRefused):
        clock.set_clock(T0 - 60, state=state)


def test_clock_at_least_moves_forward_and_never_back(clock, tmp_path):
    state = tmp_path / 'replay'
    clock.set_clock(T0, state=state)
    (state / 'store').mkdir(parents=True)
    later = clock.clock_at_least(T0 + 120, state=state)
    assert later == T0 + 120 and time.time() >= T0 + 120
    # Already later: the clock stays where it is and says where that is.
    kept = clock.clock_at_least(T0 + 10, state=state)
    assert kept >= T0 + 120 and time.time() >= kept


def test_a_new_state_may_take_its_own_rewind(clock, tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    clock.set_clock(T0, state=first)
    (first / 'store').mkdir(parents=True)
    clock.set_clock(T0 - 86400 * 30, state=second)
    assert abs(time.time() - (T0 - 86400 * 30)) < 5


def test_the_replay_clock_needs_no_hermes_and_uninstalls(clock, tmp_path):
    real = time.time()
    clock.set_clock(T0, state=tmp_path / 'replay')
    assert time.time() < real - 86400
    clock.uninstall_clock()
    assert abs(time.time() - real) < 60


def test_a_clock_bound_while_the_replay_ran_reads_real_time_after_it_is_uninstalled(clock, tmp_path):
    """A module first imported while the replay clock ran binds the shifted function (as a default argument, say);
    once the clock is uninstalled, that function reads the real clock and never looks for the removed original."""
    clock.set_clock(T0, state=tmp_path / 'replay')
    bound = time.time
    assert abs(bound() - T0) < 5
    clock.uninstall_clock()
    assert abs(bound() - time.time()) < 60


# The reranker capture shared with source_recall: every call kept, the provider's calibration unchanged.

class _Provider:
    _model_id = 'fixture-reranker'

    def __init__(self, failure=None):
        self.failure, self.release = failure, None

    def calibration_metadata(self):
        return {'provider': 'Fixture', 'model': self._model_id, 'prompt_style': '', 'format_version': 'raw-v1',
                'weights_revision': 'unverified'}

    async def rerank(self, query, documents, top_k=10):
        from colonymind.vector.reranker import RerankResult
        if self.failure == 'error':
            raise RuntimeError('https://not-recorded.example/?key=not-recorded')
        if self.failure == 'slow':
            self.release = asyncio.Event()
            await self.release.wait()
        return [RerankResult(index=index, score=1.0 - index / 10, text=text) for index, text in enumerate(documents)]


def test_a_captured_provider_keeps_its_calibration_and_records_every_call():
    from colonymind.memory.recall import calibration_fingerprint, provider_calibration_metadata
    from colonymind.qualification.recall_capture import ProviderCapture
    provider = _Provider()
    captured = ProviderCapture(provider)
    assert calibration_fingerprint(provider_calibration_metadata(captured)) == \
        calibration_fingerprint(provider_calibration_metadata(provider))
    rows = asyncio.run(captured.rerank('Which room?', ['Room 4.', 'Room 9.'], top_k=2))
    assert [row.index for row in rows] == [0, 1]           # the provider's own rows reach recall
    calls = captured.recorder.take()
    assert json.loads(json.dumps(calls)) == [{'query': 'Which room?', 'documents': ['Room 4.', 'Room 9.'], 'top_k': 2,
        'outcome': 'returned', 'results': [{'index': 0, 'score': 1.0, 'text': 'Room 4.'},
                                           {'index': 1, 'score': 0.9, 'text': 'Room 9.'}]}]
    assert captured.recorder.take() == []


def test_a_captured_failure_keeps_no_message_and_a_call_still_running_is_abandoned():
    from colonymind.qualification.recall_capture import ProviderCapture
    failing = ProviderCapture(_Provider('error'))
    with pytest.raises(RuntimeError):
        asyncio.run(failing.rerank('q', ['a'], top_k=1))
    calls = failing.recorder.take()
    assert calls == [{'query': 'q', 'documents': ['a'], 'top_k': 1, 'outcome': 'error', 'error_type': 'RuntimeError'}]
    slow = _Provider('slow')
    captured = ProviderCapture(slow)

    async def recall():
        call = asyncio.ensure_future(captured.rerank('q', ['a'], top_k=1))
        await asyncio.sleep(0.01)
        taken = captured.recorder.take()      # the recall is done while its call still runs
        slow.release.set()
        await call
        return taken
    assert [call['outcome'] for call in asyncio.run(recall())] == ['abandoned']


# The replay: a 3-session fixture through the sidecar itself, a scripted model, the clock on the fixture's dates.

BENCH = Path(__file__).resolve().parents[2] / 'benchmarks' / 'public_memory'
UTC = timezone.utc


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module      # a dataclass resolves its annotations through its module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def replay_module():
    return _load('public_memory_replay_under_test', BENCH / 'replay.py')


@pytest.fixture(scope='module')
def records():
    return _load('public_memory_records_under_test', BENCH / 'datasets' / 'records.py')


FIRST = 'My office is in Room 4 on the second floor.'
SECOND = 'Robin starts on the design team tomorrow.'
THIRD = 'We moved: my office is in Room 9 now.'
EXTRA = 'Sam is lending me the blue bicycle.'
LAST = 'I am out of the office on Friday.'
# Talk about the office with nothing to remember: more candidates than a packet holds, so recall reranks.
CHATTER = ('The office printer is jammed again.', 'Lunch near the office was good today.',
           'I left my umbrella at the office.', 'The office plants need water.',
           'Someone painted the office door green.')


def fixture_conversation(records):
    """Four dated sessions (the second opens with an assistant turn and ends with a user turn alone; the third is
    office chatter), and one question asked a week after the last."""
    Turn, Session = records.Turn, records.Session
    at = [datetime(2023, 3, 1, 9, 0, tzinfo=UTC), datetime(2023, 3, 15, 18, 30, tzinfo=UTC),
          datetime(2023, 3, 20, 13, 0, tzinfo=UTC), datetime(2023, 4, 2, 8, 15, tzinfo=UTC)]
    sessions = (
        Session(key='s-1', source_id='s-1', position=0, at=at[0], turns=(
            Turn(0, 'user', FIRST), Turn(1, 'assistant', 'Noted: Room 4, second floor.'),
            Turn(2, 'user', SECOND), Turn(3, 'assistant', 'Good to know.'))),
        Session(key='s-2', source_id='s-2', position=1, at=at[1], turns=(
            Turn(0, 'assistant', 'Morning. Anything to note today?'),
            Turn(1, 'user', EXTRA), Turn(2, 'assistant', 'Noted.'),
            Turn(3, 'user', LAST))),
        Session(key='s-2b', source_id='s-2b', position=2, at=at[2], turns=tuple(
            Turn(index, role, text) for index, (role, text) in enumerate(
                (role, text) for line in CHATTER for role, text in (('user', line), ('assistant', 'Noted.'))))),
        Session(key='s-3', source_id='s-3', position=3, at=at[3], turns=(
            Turn(0, 'user', THIRD), Turn(1, 'assistant', 'Updated: Room 9.'))))
    conversation = records.Conversation(id='fixture-1', sessions=sessions)
    question = records.Question(
        id='fixture-1', conversation='fixture-1', type='knowledge-update', question='Where is my office now?',
        answer='Room 9', asked_at=datetime(2023, 4, 9, 12, 0, tzinfo=UTC), date_text=None, abstention=False,
        gold_sessions=('s-3',), evidence=(records.Evidence('s-1', 0), records.Evidence('s-3', 0)))
    return conversation, question


# The night's calls, answered in the shape each one's schema asks for.
NIGHT = {'mind_claim_relation': {'relation': 'replaces'}, 'mind_consolidate_narrative': {'lines': []},
         'mind_consolidate_digest': {'digest': 'The owner works in Room 9 on the second floor.', 'sources': []},
         'mind_consolidate_episode': {'summary': 'The owner said where their office is.'},
         'mind_plain_summary': {'sentence': 'The office moved to Room 9.'}}


class NotScripted(RuntimeError):
    """A call this model has no answer for: the job that made it waits and retries, as after a failed call."""


class ScriptedModel:
    """The claim extraction and its review, scripted per message, and the night's calls; every extraction's clock
    is kept. ``routing`` is whether it serves function calls (the night needs them); a task it has no answer for
    fails the call."""

    def __init__(self, delay=0.2, routing=True):
        from test_source_claim_projection import Model, claim
        self.delay, self.supports_function_routing = delay, routing
        self.model = Model({
            FIRST: claim(FIRST, 'Room 4'), THIRD: claim(THIRD, 'Room 9'),
            SECOND: claim(SECOND, 'design team', subject='Robin', predicate='team', valid_from_text='tomorrow'),
            EXTRA: claim(EXTRA, 'blue bicycle', subject='Sam', predicate='lending'),
            LAST: claim(LAST, 'out of the office on Friday', predicate='availability')})
        self.extractions, self.night = [], []

    def tier_config(self, tier):
        return self.model.tier_config(tier)

    async def complete(self, messages, **kwargs):
        task = kwargs.get('context', {}).get('task')
        if task in NIGHT:
            self.night.append(task)
            return SimpleNamespace(content=json.dumps(NIGHT[task]), model_id=self.model.model)
        if task not in ('source_claim_extraction', 'source_claim_review'):
            raise NotScripted(task)
        payload = json.loads(messages[-1]['content'])
        if task == 'source_claim_extraction':
            await asyncio.sleep(self.delay)
            self.extractions.append({'message': payload['message'], 'clock': time.time()})
            if payload['message'] not in self.model.outputs:
                return SimpleNamespace(content='[]', model_id=self.model.model)
        return await self.model.complete(messages, **kwargs)


@pytest.fixture(scope='module')
def formed(replay_module, records, tmp_path_factory):
    """One replay of the fixture, shared by the tests below."""
    conversation, question = fixture_conversation(records)
    root = tmp_path_factory.mktemp('formed') / 'store'
    model = ScriptedModel()
    try:
        manifest = asyncio.run(replay_module.replay(conversation, [question], root, router=model,
                                                    drain_idle=2.5, drain_seconds=60))
    finally:
        paired_body.uninstall_clock()
    return SimpleNamespace(root=root, manifest=manifest, model=model, conversation=conversation,
                           question=question, ledger=root / 'instance' / 'turn-idempotency.db')


def _rows(path, sql, *args):
    with closing(sqlite3.connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(sql, args)]


def test_endpoint_settings_are_recall_settings_never_a_credential(replay_module, runner, tmp_path):
    value = 'synthetic-' + 'not-a-key'
    for name in ('COLONYMIND_RERANKER_API_KEY', 'COLONYMIND_EMBED_API_KEY', 'COLONYMIND_BENCH_RERANKER_API_KEY',
                 'COLONYMIND_EXAMPLE_TOKEN', 'COLONYMIND_EXAMPLE_PASSWORD'):
        with pytest.raises(replay_module.ReplayError) as refused:
            replay_module.Endpoints(settings={name: value})
        # The refusal points at the variables a credential belongs in, and never repeats a value or its name.
        message = str(refused.value)
        assert value not in message and 'COLONYMIND_BENCH_RERANKER_API_KEY' in message
        assert name not in message or name in replay_module.SECRET_VARIABLES.values()
    # A setting recall reads that is not on the list is refused by name: only listed settings are recorded.
    with pytest.raises(replay_module.ReplayError, match='COLONYMIND_EXAMPLE_SETTING'):
        replay_module.Endpoints(settings={'COLONYMIND_EXAMPLE_SETTING': '1'})
    # A listed setting is kept, in the instance's configuration and in the manifest, including one whose name
    # carries a word a credential's name would.
    kept = replay_module.Endpoints(settings={'COLONYMIND_RECALL_RERANK_MIN_SCORE': '0.3',
                                             'COLONYMIND_EMBED_QUERY_MAX_TOKENS': '512'})
    assert kept.public()['settings'] == {'COLONYMIND_EMBED_QUERY_MAX_TOKENS': '512',
                                         'COLONYMIND_RECALL_RERANK_MIN_SCORE': '0.3'}
    assert not set(replay_module.SETTINGS) & (set(replay_module.SECRET_VARIABLES)
                                             | set(replay_module.SECRET_VARIABLES.values()))
    # The runner's endpoints file goes through the same check.
    path = tmp_path / 'endpoints.json'
    path.write_text(json.dumps({'rerank_model': 'fixture', 'settings': {'COLONYMIND_RERANKER_API_KEY': value}}))
    with pytest.raises(runner.replay.ReplayError):
        runner._endpoints(path)


def test_the_schedule_starts_a_session_no_earlier_than_the_last_one_ended(replay_module, records):
    conversation, _ = fixture_conversation(records)
    plan = replay_module.schedule(conversation)
    assert [(session.key, len(parts)) for session, _, _, parts in plan] == [('s-1', 2), ('s-2', 3), ('s-2b', 5),
                                                                             ('s-3', 1)]
    assert [start for _, start, _, _ in plan] == [session.at.timestamp() for session in conversation.sessions]
    # The second session's parts: an assistant turn alone, a pair, and a user turn alone.
    _, _, _, parts = plan[1]
    assert [(u and u.index, a and a.index) for u, a in parts] == [(None, 0), (1, 2), (3, None)]
    first_session, last_session = conversation.sessions[0], conversation.sessions[-1]
    tied = records.Conversation(id='tied', sessions=(first_session, records.Session(
        key='s-1b', source_id='s-1b', position=1, at=first_session.at, turns=last_session.turns)))
    (_, first, end, _), (_, second, _, _) = replay_module.schedule(tied)
    assert second == end == first + 2


# 2. A replayed occurred_at is never clamped.

def test_every_replayed_turn_keeps_the_fixture_time(formed):
    manifest = formed.manifest
    assert manifest['refused'] == []
    plan = {row['key']: row for row in manifest['sessions']}
    sources = {row['turn_id']: row for row in _rows(formed.ledger, 'SELECT turn_id,occurred_at,contact_id,scope '
                                                                   'FROM turn_sources')}
    assert len(sources) == 11
    for turn in manifest['turns']:
        stored = sources[turn['turn_id']]
        assert stored['occurred_at'] == turn['occurred_at']
        assert stored['scope'] == 'person'
        start = datetime.fromisoformat(plan[turn['session']]['start'])
        position = int(turn['turn_id'].rsplit(':', 1)[1])
        assert datetime.fromisoformat(turn['occurred_at']) == start + timedelta(seconds=position)
    assert sources['replay:0:0']['occurred_at'] == '2023-03-01T09:00:00+00:00'
    assert sources['replay:3:0']['occurred_at'] == '2023-04-02T08:15:00+00:00'
    # The owner's last interaction is the last turn's own time: the "never later than now" rule clamped nothing.
    owner, = _rows(formed.root / 'instance' / 'colonymind-contacts.db',
                   'SELECT last_interaction_at FROM contacts WHERE trust_tier=?', 'inner_circle')
    assert owner['last_interaction_at'] == '2023-04-02T08:15:00Z'


def test_serving_puts_the_process_environment_back(formed):
    replayed = str(formed.root / 'instance')
    assert os.environ.get('COLONYMIND_STATE_DIR') != replayed and os.environ.get('COLONYMIND_HOME') != replayed
    assert os.environ.get('HOME') != str(formed.root / 'home')
    assert paired_body._ORIGINAL == {}


def test_serving_puts_the_hosts_telemetry_back(formed, replay_module):
    """The sidecar's lifespan installs its telemetry process-wide and a replayed store's clock stamps it; serving puts
    back what the process had, so a later health check in the same process never reads the replay's record (a mind
    that "never ticked" since a start years ago)."""
    from colonymind.api.routers import host
    before = object()
    host.set_telemetry(before)
    try:
        async def serve():
            async with replay_module.serve(replay_module.open_instance(formed.root)):
                assert host._telemetry is not before
        asyncio.run(serve())
        assert host._telemetry is before
    finally:
        host.set_telemetry(None)
    assert host._telemetry is None


def test_the_replay_keeps_to_its_own_home(formed):
    kept = json.loads((formed.root / 'home' / '.colonymind' / 'last_user_message_at.json').read_text())
    assert kept['timestamp'].startswith('2023-04-02')
    outside = Path.home() / '.colonymind' / 'last_user_message_at.json'
    assert not outside.exists() or not json.loads(outside.read_text())['timestamp'].startswith('2023-')


def test_processing_time_never_moves_the_replayed_timeline(replay_module, records, runner, tmp_path):
    """A session ending two seconds before midnight, a question one second before it, and a model slow enough
    that processing runs past midnight in real time: the night and the question stay on the question's day."""
    session = records.Session(key='s-1', source_id='s-1', position=0,
                              at=datetime(2023, 5, 1, 23, 59, 57, tzinfo=UTC),
                              turns=(records.Turn(0, 'user', FIRST), records.Turn(1, 'assistant', 'Noted.')))
    asked = datetime(2023, 5, 1, 23, 59, 59, tzinfo=UTC)
    question = records.Question(
        id='late-1', conversation='late-1', type='single-session-user', question='Where is my office?',
        answer='Room 4', asked_at=asked, date_text=None, abstention=False, gold_sessions=('s-1',),
        evidence=(records.Evidence('s-1', 0),))
    conversation = records.Conversation(id='late-1', sessions=(session,))
    model = ScriptedModel(delay=2.5)
    try:
        manifest = asyncio.run(replay_module.replay(conversation, [question], tmp_path / 'store', router=model,
                                                    drain_idle=2.5, drain_seconds=60))
    finally:
        paired_body.uninstall_clock()
    assert len(model.extractions) == 1
    # The clock stood at the session's end while the model worked, and moved only to the question's time.
    assert datetime.fromtimestamp(model.extractions[0]['clock'], UTC) == datetime(2023, 5, 1, 23, 59, 58, tzinfo=UTC)
    assert manifest['night']['at'] == asked.isoformat() and manifest['night']['shift_seconds'] == 0
    assert manifest['night']['local_date'] == '2023-05-01'
    assert max(datetime.fromisoformat(event['clock']) for event in manifest['events']) == asked
    capture = tmp_path / 'capture.jsonl'
    asyncio.run(runner.read(tmp_path / 'store', [question], arm='all-on', budgets=(4000,), capture=capture,
                            work=tmp_path / 'work'))
    row, = [json.loads(line) for line in capture.read_text().splitlines()]
    assert row['read_at'] == asked.isoformat() and row['shift_seconds'] == 0


# 3. The drain runs before a session switch.

def test_the_queues_drain_before_each_session(formed):
    events = formed.manifest['events']
    sessions = [index for index, event in enumerate(events) if event['event'] == 'session']
    assert len(sessions) == 4
    for index in sessions:
        before = events[index - 1]
        assert before['event'] == 'drain' and before['before_session'] == events[index]['session']
        assert before['status'] in {'drained', 'idle'}
        assert 'source_claim_jobs' not in before['left']
    # Every claim of a session was extracted while the clock stood at that session's end: before it moved to the
    # next session, whose first turn was stored after that move.
    ingested = {row['turn_id']: row['ingested_at'] for row in _rows(formed.ledger,
                                                                    'SELECT turn_id,ingested_at FROM turn_sources')}
    ends = [datetime.fromisoformat(row['end']) for row in formed.manifest['sessions']]
    session_of = {FIRST: 0, SECOND: 0, EXTRA: 1, LAST: 1, **{line: 2 for line in CHATTER}, THIRD: 3}
    for call in formed.model.extractions:
        position = session_of[call['message']]
        assert datetime.fromtimestamp(call['clock'], UTC) == ends[position]
        if position + 1 < 4:
            opened = datetime.strptime(ingested[f'replay:{position + 1}:0'], '%Y-%m-%dT%H:%M:%S.%fZ')
            assert opened.replace(tzinfo=UTC) == ends[position + 1] > ends[position]


# 4. A 3-session fixture yields claims with the fixture's dates.

def test_the_formed_store_holds_claims_dated_by_the_fixture(formed):
    claims = _rows(formed.ledger, 'SELECT c.turn_id,c.predicate,c.valid_from,s.occurred_at FROM source_claims c '
                                  'JOIN turn_sources s USING(turn_id) ORDER BY s.occurred_at')
    assert len(claims) == 5 and len(formed.model.extractions) == 10
    # Each claim is dated by the source it came from, at the fixture's time.
    dates = [datetime.fromisoformat(claim['occurred_at']).date().isoformat() for claim in claims]
    assert dates == ['2023-03-01', '2023-03-01', '2023-03-15', '2023-03-15', '2023-04-02']
    # "tomorrow" said on 2023-03-01 resolves against that day, not against the real clock.
    team, = [claim for claim in claims if claim['predicate'] == 'team']
    starts = datetime.fromisoformat(team['valid_from'])
    assert datetime(2023, 3, 1, tzinfo=UTC) < starts < datetime(2023, 3, 15, tzinfo=UTC)
    assert formed.manifest['counts']['claims'] == 5
    assert formed.manifest['night']['local_date'] == '2023-04-09'
    assert formed.manifest['before_night']['counts']['claims'] == 5
    assert (formed.root / 'before-night' / 'instance' / 'turn-idempotency.db').is_file()


def test_the_night_ran_on_the_model_and_the_store_is_complete(formed):
    night = formed.manifest['night']
    assert night['status'] == 200 and night['errors'] == [] and night['exhausted'] is False
    # Model calls answered, not a night that stopped at its first call: the office's two values were related.
    assert night['calls'] >= 1 and night['calls'] == len(formed.model.night)
    assert 'mind_claim_relation' in formed.model.night
    assert formed.manifest['formation'] == {'complete': True, 'problems': []}


def test_a_night_that_failed_leaves_an_incomplete_store_nothing_reuses(replay_module, records, runner, tmp_path,
                                                                    monkeypatch):
    conversation, question = fixture_conversation(records)
    # The two sessions that give the office two values: the night has a relation to ask a model about.
    conversation = records.Conversation(id='short-1', sessions=(conversation.sessions[0], conversation.sessions[3]))
    model = ScriptedModel(routing=False)            # it cannot serve the night's calls
    from colonymind.beliefs.source_projection import SourceClaimProjection
    changed = asyncio.Event()
    process_one = SourceClaimProjection.process_one
    drain = replay_module.drain

    async def notify(self, *args, **kwargs):
        try:
            return await process_one(self, *args, **kwargs)
        finally:
            changed.set()

    async def claims_ready(instance, **kwargs):
        # Idle is allowed by replay policy, but this scenario requires both
        # office claims before the night. The worker polls every 2 s; a 1 s
        # idle drain may close its app before that first poll. Await completed
        # claim jobs, using the existing drain deadline and worker notifications.
        async with asyncio.timeout(kwargs['seconds']):
            while True:
                changed.clear()
                rows = _rows(instance.ledger, 'SELECT status FROM source_claim_jobs')
                if all(row['status'] == 'complete' for row in rows):
                    break
                await changed.wait()
        return await drain(instance, **kwargs)

    monkeypatch.setattr(SourceClaimProjection, 'process_one', notify)
    monkeypatch.setattr(replay_module, 'drain', claims_ready)
    try:
        manifest = asyncio.run(replay_module.replay(conversation, [question], tmp_path / 'store', router=model,
                                                    drain_idle=1.0, drain_seconds=60))
    finally:
        paired_body.uninstall_clock()
    night = manifest['night']
    assert manifest['before_night']['counts']['claims'] == 3
    # The route answered, and its record says the night did not run on a model.
    assert night['status'] == 200 and night['errors'] and night['exhausted'] is True and night['calls'] == 0
    assert manifest['formation']['complete'] is False
    assert any(problem.startswith('night') for problem in manifest['formation']['problems'])
    assert replay_module.load_manifest(tmp_path / 'store')['formation'] == manifest['formation']
    # A reader refuses the store unless told to read an incomplete one, and the row then says so.
    capture = tmp_path / 'capture.jsonl'
    with pytest.raises(runner.replay.ReplayError, match='incomplete'):
        asyncio.run(runner.read(tmp_path / 'store', [question], arm='all-on', budgets=(4000,), capture=capture,
                                work=tmp_path / 'work'))
    paired_body.uninstall_clock()
    asyncio.run(runner.read(tmp_path / 'store', [question], arm='no-night', budgets=(4000,), capture=capture,
                            work=tmp_path / 'work', allow_incomplete=True))
    row, = [json.loads(line) for line in capture.read_text().splitlines()]
    assert row['formation'] == manifest['formation']
    # The command line forms a conversation again only in a new directory: an incomplete one is never counted.
    assert runner.formed_already(tmp_path / 'store') is False
    assert runner.formed_already(tmp_path / 'nothing') is False


def test_gold_names_the_stored_messages(formed, replay_module):
    from colonymind.turns.idempotency import source_message_hash
    items = replay_module.gold(formed.manifest, formed.question)
    assert items == [{'hash:' + source_message_hash('s-1', {'role': 'user', 'content': FIRST})},
                     {'hash:' + source_message_hash('s-3', {'role': 'user', 'content': THIRD})}]
    alone = [row for row in formed.manifest['turns'] if row['alone']]
    assert {(row['session'], row['index']) for row in alone} == {('s-2', 0), ('s-2', 3)}


# 6. The import-only store has zero claims.

def test_the_import_only_store_has_sources_and_no_claims(replay_module, records, tmp_path):
    conversation, question = fixture_conversation(records)
    try:
        manifest = asyncio.run(replay_module.import_only(conversation, [question], tmp_path / 'import'))
    finally:
        paired_body.uninstall_clock()
    ledger = tmp_path / 'import' / 'instance' / 'turn-idempotency.db'
    assert _rows(ledger, 'SELECT count(*) AS n FROM source_claims')[0]['n'] == 0
    assert _rows(ledger, 'SELECT count(*) AS n FROM source_claim_jobs')[0]['n'] == 0
    sources = _rows(ledger, 'SELECT turn_id,occurred_at,contact_id FROM turn_sources ORDER BY occurred_at,turn_id')
    assert len(sources) == 20 and {row['contact_id'] for row in sources} == {manifest_owner(tmp_path / 'import')}
    assert manifest['counts']['claims'] == 0 and all(run['models_called'] is False for run in manifest['import'])
    assert manifest['formation'] == {'complete': True, 'problems': []}
    assert len(manifest['turns']) == 20 and all(row['message_hash'] and row['alone'] for row in manifest['turns'])
    by_turn = {row['turn_id']: row for row in sources}
    for turn in manifest['turns']:
        assert datetime.fromisoformat(by_turn[turn['turn_id']]['occurred_at']) == \
            datetime.fromisoformat(turn['occurred_at'])
    items = replay_module.gold(manifest, question)
    assert all(any(identifier.startswith('turn:hermes-history:') for identifier in item) for item in items)


def manifest_owner(root):
    import yaml
    return yaml.safe_load((root / 'instance' / 'colonymind.yaml').read_text())['owner']['contact_id']


# 5. The runner: what the host would inject at a budget, captured, and re-scored identically offline.

@pytest.fixture(scope='module')
def runner():
    return _load('public_memory_runner_under_test', BENCH / 'runner.py')


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_the_capture_re_scores_identically_offline(formed, runner, tmp_path):
    ledger_before = _digest(formed.ledger)
    capture = tmp_path / 'capture.jsonl'
    scores = asyncio.run(runner.read(formed.root, [formed.question], arm='all-on', budgets=(4000, 24000),
                                     capture=capture, work=tmp_path / 'work', reranker=_Provider()))
    assert paired_body._ORIGINAL == {}            # every read put the clock back
    rows = [json.loads(line) for line in capture.read_text().splitlines()]
    assert [row['budget_chars'] for row in rows] == [4000, 24000]
    for row in rows:
        assert row['status'] == 200 and row['store'] == 'formed'
        assert row['read_at'] >= row['asked_at'] and row['read_at'].startswith('2023-04-09')
        assert any(section['id'] == 'colonymind-memory' for section in row['sections'])
        assert row['trace']['section'] in {'in_context', 'shortened'} and row['trace']['rows']
        assert row['gold'] == [sorted(item) for item in runner.replay.gold(formed.manifest, formed.question)]
        assert row['rerank_calls'] and all(call['outcome'] == 'returned' for call in row['rerank_calls'])
        assert row['trace']['rerank']['mode'] == 'on' and row['trace']['rerank']['submitted'] >= 1
    # Live and offline scores are the same numbers, read from the response and from the file.
    assert runner.rescore(capture) == scores
    first = scores[0]
    assert first['reach']['basis'] == 'structural' and first['reach']['any'] is True
    assert first['reach']['first_rank'] >= 1 and first['chars'] <= first['budget'] == 4000
    # Reading copies the store: the formed store is exactly as it was, and no copy is left behind.
    assert _digest(formed.ledger) == ledger_before
    assert not any((tmp_path / 'work').iterdir())
    # The command line re-scores a capture file into per-arm, per-budget tables.
    report = tmp_path / 'report.json'
    assert runner.main(['rescore', '--capture', str(capture), '--output', str(report)]) == 0
    tables = json.loads(report.read_text())['summary']['all-on']
    assert set(tables) == {'4000', '24000'}
    assert tables['4000']['overall']['reach_any']['successes'] == 1


def test_an_arm_reads_its_own_store_and_settings(formed, runner, tmp_path):
    capture = tmp_path / 'capture.jsonl'
    asyncio.run(runner.read(formed.root, [formed.question], arm='no-night', budgets=(4000,), capture=capture,
                            work=tmp_path / 'work'))
    row, = [json.loads(line) for line in capture.read_text().splitlines()]
    assert row['store'] == 'before_night' and row['arm'] == 'no-night' and row['rerank_calls'] == []
    with pytest.raises(runner.replay.ReplayError):
        asyncio.run(runner.read(formed.root, [formed.question], arm='import-only', budgets=(4000,),
                                capture=capture, work=tmp_path / 'other'))
    paired_body.uninstall_clock()
    home = tmp_path / 'config'
    home.mkdir()
    # Every supported arm respects the host's smaller budget before scoring.
    assert 'no-blocks' not in runner.ARMS
    capture = tmp_path / 'bounded.jsonl'
    scored, = asyncio.run(runner.read(formed.root, [formed.question], arm='all-on', budgets=(300,),
                                      capture=capture, work=tmp_path / 'bounded'))
    row, = [json.loads(line) for line in capture.read_text().splitlines()]
    assert row['budget']['budget'] == 300 and scored['returned_chars'] <= 300
    assert scored['chars'] <= 300 and scored['cut'] is False and runner.rescore(capture) == [scored]
    runner.replay.write_config(home, {'owner': {'contact_id': 'cid-1'}, 'mind': {'enabled': True}})
    runner.arm_config(home, runner.ARMS['memory-core'])
    import yaml
    config = yaml.safe_load((home / 'colonymind.yaml').read_text())
    assert config['mind']['enabled'] is False and 'blocks' not in config.get('context', {})


def test_a_context_over_its_budget_is_cut_in_the_hosts_order_before_it_is_scored(runner):
    def item(turn, text):
        return '- ' + json.dumps({'source_turn_id': turn, 'text': text})
    memory = '\n'.join(['Recalled:', item('t-1', 'a' * 900), item('t-2', 'b' * 900), item('t-3', 'c' * 900)])
    sections = [{'id': 'temporal-context', 'title': 'Current Time', 'body': 'Now: 2023-04-09 12:00 UTC'},
                {'id': 'colonymind-memory', 'title': 'Memory', 'body': memory},
                {'id': 'colonymind-status', 'title': 'Status', 'body': 'x' * 300}]
    row = {'protocol': runner.PROTOCOL, 'question': 'q', 'conversation': 'c', 'type': 't', 'arm': 'no-blocks',
           'budget_chars': 2000, 'sections': sections, 'gold': [['turn:t-1'], ['turn:t-3']]}
    scored = runner.score(row)
    # The time, then the memory's whole lines while they fit: the first line that does not ends the context.
    assert scored['chars'] <= 2000 < scored['returned_chars'] and scored['cut'] is True
    assert scored['items'] == 2 and scored['reach']['reached'] == 1 and scored['reach']['first_rank'] == 1
    kept = runner.fit(sections, 2000)
    assert [section['id'] for section in kept] == ['temporal-context', 'colonymind-memory']
    assert kept[1]['body'].split('\n') == memory.split('\n')[:3] and kept[0] == sections[0]
    # A context within its budget is scored as the host returned it.
    whole = runner.score({**row, 'budget_chars': 10000})
    assert whole['cut'] is False and whole['items'] == 3 and whole['chars'] == whole['returned_chars']
    assert runner.fit(sections, 10) == []


def long_conversation(records):
    """Six long notes about the office in one session: more evidence than 4,000 characters hold."""
    notes = [f'Office note {n}. ' + ' '.join(f'The office on floor {n} has window {k}, desk {k} and lamp {k}.'
                                            for k in range(24)) for n in range(6)]
    turns = tuple(turn for n, note in enumerate(notes) for turn in (
        records.Turn(2 * n, 'user', note), records.Turn(2 * n + 1, 'assistant', 'Noted.')))
    session = records.Session(key='s-1', source_id='s-1', position=0, at=datetime(2023, 6, 1, 9, 0, tzinfo=UTC),
                              turns=turns)
    question = records.Question(
        id='long-1', conversation='long-1', type='multi-session', question='What do my office notes say?',
        answer='', asked_at=datetime(2023, 6, 2, 9, 0, tzinfo=UTC), date_text=None, abstention=False,
        gold_sessions=('s-1',), evidence=tuple(records.Evidence('s-1', 2 * n) for n in range(6)))
    return records.Conversation(id='long-1', sessions=(session,)), question, notes


def test_a_larger_budget_lets_recall_fill_it(runner, records, tmp_path):
    conversation, question, notes = long_conversation(records)
    assert sum(len(note) for note in notes) > 8000
    try:
        asyncio.run(runner.replay.import_only(conversation, [question], tmp_path / 'store'))
    finally:
        paired_body.uninstall_clock()
    capture = tmp_path / 'capture.jsonl'
    scores = asyncio.run(runner.read(tmp_path / 'store', [question], arm='import-only', budgets=(4000, 24000),
                                     capture=capture, work=tmp_path / 'work'))
    rows = [json.loads(line) for line in capture.read_text().splitlines()]
    # Recall's own ceiling is the budget: the host's allowance and the packet grow together.
    assert [row['trace']['packet']['max_chars'] for row in rows] == [4000, 24000]
    assert [row['recall_max_chars'] for row in rows] == [4000, 24000]
    memory = [next(section for section in row['sections'] if section['id'] == 'colonymind-memory') for row in rows]
    small, large = (runner.section_cost(section) for section in memory)
    assert small <= 4000 < large
    # More of the evidence reaches the reader at the larger budget, and every row stays within its own.
    assert scores[1]['reach']['reached'] > scores[0]['reach']['reached']
    assert all(score['chars'] <= score['budget'] for score in scores)


def test_the_import_reads_the_replayed_instances_secrets_settings_never_the_operators(replay_module, records,
                                                                                      tmp_path):
    from colonymind.secrets import config as secrets_config
    # This process already read an operator's settings: capture off, the key in the operator's keyring.
    operator = secrets_config.SecretsSettings(capture_enabled=False, key_store='keyring')
    before = secrets_config._CURRENT
    key = 'sk-' + 'proj' + 'Qz7Wx3Er9Ty1Ui5Op2As8Df4'          # made up, assembled at run time
    session = records.Session(key='s-1', source_id='s-1', position=0, at=datetime(2023, 7, 1, 9, 0, tzinfo=UTC),
                              turns=(records.Turn(0, 'user', f'The office door code is {key} now.'),
                                     records.Turn(1, 'assistant', 'Noted.')))
    conversation = records.Conversation(id='sealed-1', sessions=(session,))
    try:
        secrets_config.set_settings(operator)
        manifest = asyncio.run(replay_module.import_only(conversation, [], tmp_path / 'import'))
        # The operator's settings are back once the import is done.
        assert secrets_config._CURRENT is operator
    finally:
        paired_body.uninstall_clock()
        secrets_config.set_settings(before)
    instance = tmp_path / 'import' / 'instance'
    with closing(sqlite3.connect(instance / 'turn-idempotency.db')) as conn:
        stored = '\n'.join(conn.iterdump())
    # The replayed instance's own settings applied: capture sealed the value, under the instance's own key file.
    assert key not in stored and len(manifest['turns']) == 2
    assert (instance / 'secrets-vault.key').is_file()

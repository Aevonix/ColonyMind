"""A seeded history replayed on its own dates: the replay history mode of the paired harness.

A generated family's history (earlier owner sessions, mind-memory-2's months of them) was imported into a plugin
arm's ledger as recorded messages, with no claims and no model call, so nothing the claim worker, the priors or the
night do ever reached it. The replay mode sends every (user, assistant) pair of every session through the served
host's own turn route instead, on the history's dates: the clock moves back once, to the first session, before the
arm's state exists; before each session the background work drains, the clock moves to the session's end and each
pair is dated at its session's start plus its position, so no replayed time is ever pulled back to "now"; after the
last session the clock runs again from the episode's pinned start. Stock Hermes imports the same history in every
arm, as before.
"""
from datetime import datetime, timezone
import json
import sqlite3
import time

import httpx
import pytest

from colonymind.qualification import paired, paired_body, paired_history, paired_worker
from colonymind.turns.idempotency import source_message_hash
from test_qualification_paired_memory_self import FakeSessionDB
from test_qualification_paired_runner import fixture  # noqa: F401  (pytest fixture)

T0 = 1_772_409_600                      # 2026-03-02 00:00 UTC, months before the real clock
SESSIONS = [{'id': 'chat-01', 'at': T0, 'messages': [
                {'role': 'user', 'content': 'I keep the spare key in the blue folder.'},
                {'role': 'assistant', 'content': 'Noted.'}]},
            {'id': 'sms-02', 'at': T0 + 86400, 'messages': [
                {'role': 'user', 'content': 'Long week at work.'}, {'role': 'assistant', 'content': 'Sounds like a lot.'},
                {'role': 'user', 'content': 'The spare key lives in the green cabinet now.'}]},
            {'id': 'email-03', 'at': T0 + 86400, 'messages': [
                {'role': 'user', 'content': 'Remind me about the budget draft.'},
                {'role': 'assistant', 'content': 'I will.'}]}]


@pytest.fixture
def clock():
    try:
        yield paired_body
    finally:
        paired_body.uninstall_clock()


def iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def replayed(tmp_path, clock, *, answer=None):
    """Rewind, replay with a recording post and drain, then run the clock from the episode's start."""
    start = time.time()
    state = tmp_path / 'home' / 'memory-state'
    assert paired_history.rewind(SESSIONS, state) == T0
    posts, drains = [], []

    def post(path, body):
        posts.append((path, body, time.time()))
        return answer(body) if answer else {'accepted': True}

    def drain():
        drains.append(time.time())
        return {'status': 'drained'}
    record = paired_history.replay_ledger(post, SESSIONS, contact_id='p-01', state=state, drain=drain)
    clock.run_clock_from(start)
    return record, posts, drains, start


def test_each_pair_goes_through_the_turn_route_on_the_historys_own_dates(tmp_path, clock):
    record, posts, drains, start = replayed(tmp_path, clock)
    assert [path for path, _, _ in posts] == ['/v1/host/turns/sync'] * 4
    bodies = [body for _, body, _ in posts]
    assert [body['context']['turn_id'] for body in bodies] == [
        'history:chat-01:0', 'history:sms-02:0', 'history:sms-02:1', 'history:email-03:0']
    assert [body['context']['session_id'] for body in bodies] == ['chat-01', 'sms-02', 'sms-02', 'email-03']
    assert {body['context']['contact_id'] for body in bodies} == {'p-01'}
    assert {body['context']['channel_id'] for body in bodies} == {paired_history.REPLAY_CHANNEL}
    # A pair is dated at its session's start plus its first message's position. The third session starts where
    # the second, stamped the same second, ended.
    assert [body['context']['metadata']['occurred_at'] for body in bodies] == [
        iso(T0), iso(T0 + 86400), iso(T0 + 86400 + 2), iso(T0 + 86400 + 3)]
    assert bodies[1]['user_message']['content'] == 'Long week at work.'
    assert bodies[1]['assistant_message']['content'] == 'Sounds like a lot.'
    assert 'assistant_message' not in bodies[2]                         # the session ends on the owner's word
    # The clock stands at each session's end while its pairs are sent, never before a pair's own time.
    assert [at for _, _, at in posts] == [T0 + 2, T0 + 86400 + 3, T0 + 86400 + 3, T0 + 86400 + 5]
    # A drain before each session, one after the last; the clock stood still meanwhile.
    assert drains == [T0, T0 + 2, T0 + 86400 + 3, T0 + 86400 + 5]
    assert record['mode'] == 'replay' and record['turns'] == 4 and record['refused'] == []
    assert [row['session'] for row in record['sessions']] == ['chat-01', 'sms-02', 'email-03']
    # Afterwards the clock runs again from the episode's pinned start.
    assert abs(time.time() - start) < 5
    first = time.time()
    time.sleep(0.02)
    assert time.time() > first


def test_the_clock_moves_back_only_before_the_arms_state_exists(tmp_path, clock):

    state = tmp_path / 'memory-state'
    state.mkdir()
    (state / 'turn-idempotency.db').write_text('')
    with pytest.raises(paired_body.ClockRefused):
        paired_history.rewind(SESSIONS, state)
    with pytest.raises(paired_body.ClockRefused):
        clock.run_clock_from(time.time() - 3600)                        # never back once it runs


def test_a_refused_turn_is_recorded_and_the_replay_goes_on(tmp_path, clock):
    def answer(body):
        if body['context']['turn_id'] == 'history:sms-02:1':
            raise httpx.HTTPStatusError('refused', request=httpx.Request('POST', 'http://x'),
                                        response=httpx.Response(422))
        return {'accepted': body['context']['turn_id'] != 'history:email-03:0', 'skipped_reason': 'duplicate'}
    record, posts, _, _ = replayed(tmp_path, clock, answer=answer)
    assert len(posts) == 4 and record['turns'] == 2
    assert record['refused'] == [{'turn_id': 'history:sms-02:1', 'status': 422},
                                 {'turn_id': 'history:email-03:0', 'status': 200, 'skipped_reason': 'duplicate'}]


def test_seeding_with_a_replay_imports_hermes_and_replays_the_ledger(tmp_path, clock):

    home = tmp_path / 'home'
    home.mkdir()
    paired_history.rewind(SESSIONS, home / 'memory-state')
    posts = []
    replay = paired_history.Replay(post=lambda path, body: posts.append(body) or {'accepted': True},
                                   drain=lambda: {'status': 'drained'})
    record = paired_history.seed(home, SESSIONS, session_db=FakeSessionDB, contact_id='p-01', ledger=True,
                                 vectors=True, replay=replay)
    assert record['hermes'] == {'imported': 3} and record['ledger']['mode'] == 'replay' and len(posts) == 4
    assert 'vector_drain' not in record                                 # the replay's own drains cover it
    assert not (home / 'memory-state' / 'turn-idempotency.db').exists() # nothing went through the importer


def test_the_served_host_stores_replayed_pairs_as_the_owners_with_the_historys_hashes(tmp_path, clock, monkeypatch):
    """The replay's request is the one the served host accepts: each pair is an owner source in its session, and
    its messages hash as the re-score hashes a history message."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from colonymind.api.middleware import ApiKeyMiddleware
    from colonymind.qualification import native_memory_worker as worker

    state = tmp_path / 'home'
    state.mkdir()
    paired_history.rewind(SESSIONS, state / 'memory-state')
    instance = state / 'colonymind-instance'
    instance.mkdir()
    (instance / 'api.key').write_text('replay-key\n')
    (instance / 'identity.yaml').write_text(json.dumps(worker.benchmark_identity('p-01')))
    (instance / 'colonymind.yaml').write_text(json.dumps(worker.instance_config('p-01', {'enabled': False},
                                                                                              None)))
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(state / 'memory-state'))
    monkeypatch.setenv('COLONYMIND_EVENT_JOURNAL_DIR', str(state / 'memory-state' / 'events'))
    monkeypatch.setenv('COLONYMIND_HOME', str(instance))
    monkeypatch.setenv('COLONYMIND_OWNER_CONTACT_ID', 'p-01')
    monkeypatch.setenv('COLONYMIND_SOURCE_CLAIMS', 'off')
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware, api_key='replay-key', credentials=worker.benchmark_credentials())
    worker.mount_routes(app, mind=False)
    app.router.lifespan_context = paired_worker.provider_read_lifespan(state)
    with TestClient(app) as client:
        def post(path, body):
            response = client.post(path, json=body, headers={'Authorization': 'Bearer replay-key'})
            response.raise_for_status()
            return response.json()
        record = paired_history.replay_ledger(post, SESSIONS, contact_id='p-01', state=state / 'memory-state',
                                              drain=lambda: {'status': 'drained'})
    assert record['refused'] == [] and record['turns'] == 4
    with sqlite3.connect(state / 'memory-state' / 'turn-idempotency.db') as conn:
        rows = conn.execute('SELECT turn_id, contact_id, session_id, messages_json, occurred_at FROM turn_sources '
                            'ORDER BY turn_id').fetchall()
    assert [row[0] for row in rows] == ['history:chat-01:0', 'history:email-03:0', 'history:sms-02:0',
                                        'history:sms-02:1']
    assert {row[1] for row in rows} == {'p-01'}
    stored = {(row[2], message['content']): source_message_hash(row[2], message)
              for row in rows for message in json.loads(row[3])}
    for session in SESSIONS:
        for message in session['messages']:
            assert stored[(session['id'], message['content'])] == source_message_hash(
                session['id'], {'role': message['role'], 'content': message['content']})


def test_a_plan_declares_the_replay_mode_and_needs_an_image_whose_worker_replays(fixture, monkeypatch):
    from dataclasses import replace
    from colonymind.qualification import paired_cases, paired_container
    original_cases = paired_cases.cases

    def with_history(**kwargs):
        return [replace(case, inputs={**case.inputs, 'history': SESSIONS}) for case in original_cases(**kwargs)]
    monkeypatch.setattr(paired_cases, 'cases', with_history)
    original = paired_container.configuration

    def image(protocols):
        def configuration(*args, **kwargs):
            supplied, recipe = original(*args, **kwargs)
            recipe['container_payload'] = {**recipe['container_payload'], **protocols}
            return supplied, recipe
        return configuration
    arms = ['base_hermes', 'all-on']
    fixture.output.mkdir(mode=0o700)  # plans below are private directories under a private parent
    monkeypatch.setattr(paired_container, 'configuration', image({'history_protocol': paired_history.PROTOCOL}))
    with pytest.raises(ValueError, match='replays'):
        paired.plan(fixture.output / 'older', native_binding='candidate', evidence_mode='controlled', arms=arms,
                    history_mode='replay', **fixture.resources)
    monkeypatch.setattr(paired_container, 'configuration', image({
        'history_protocol': paired_history.PROTOCOL, 'history_replay': paired_history.REPLAY_PROTOCOL}))
    imported = paired.plan(fixture.output / 'import', native_binding='candidate', evidence_mode='controlled',
                           arms=arms, **fixture.resources)
    manifest = paired.plan(fixture.output / 'replay', native_binding='candidate', evidence_mode='controlled',
                           arms=arms, history_mode='replay', **fixture.resources)
    assert manifest['comparison']['history'] == {'protocol': paired_history.REPLAY_PROTOCOL, 'mode': 'replay',
                                                 'seconds_per_turn': paired_history.REPLAY_SECONDS_PER_TURN}
    assert 'history' not in imported['comparison'] and manifest['comparison_key'] != imported['comparison_key']
    turns = sum(message['role'] == 'user' for session in SESSIONS for message in session['messages'])
    for pair, before in zip(manifest['pairs'], imported['pairs']):
        for arm in arms:
            case, earlier = pair['arms'][arm]['case'], before['arms'][arm]['case']
            assert case['inputs']['history_mode'] == 'replay'
            assert case['timeout_seconds'] == earlier['timeout_seconds'] + paired_history.REPLAY_SECONDS_PER_TURN * turns
    with pytest.raises(ValueError, match='history mode'):
        paired.plan(fixture.output / 'bad', native_binding='candidate', evidence_mode='controlled', arms=arms,
                    history_mode='rewrite', **fixture.resources)
    monkeypatch.setattr(paired_cases, 'cases', original_cases)
    with pytest.raises(ValueError, match='history'):
        paired.plan(fixture.output / 'none', native_binding='candidate', evidence_mode='controlled', arms=arms,
                    history_mode='replay', **fixture.resources)


def test_a_long_history_replays_within_its_own_deadline_bound(fixture, monkeypatch):
    """A month of sessions adds more than a family's deadline bound allows; a replayed case has its own bound."""
    from dataclasses import replace
    from colonymind.qualification import paired_cases, paired_container, records
    long = [{'id': f'chat-{index:02d}', 'at': T0 + 86400 * index, 'messages': [
        message for turn in range(4) for message in (
            {'role': 'user', 'content': f'Session {index}, note {turn}.'}, {'role': 'assistant', 'content': 'Noted.'})]}
        for index in range(40)]
    original_cases = paired_cases.cases
    monkeypatch.setattr(paired_cases, 'cases', lambda **kwargs: [
        replace(case, inputs={**case.inputs, 'history': long, 'family_deadline': 900}, timeout_seconds=900)
        for case in original_cases(**kwargs)])
    original = paired_container.configuration

    def configuration(*args, **kwargs):
        supplied, recipe = original(*args, **kwargs)
        recipe['container_payload'] = {**recipe['container_payload'], 'history_protocol': paired_history.PROTOCOL,
                                       'history_replay': paired_history.REPLAY_PROTOCOL}
        return supplied, recipe
    monkeypatch.setattr(paired_container, 'configuration', configuration)
    manifest = paired.plan(fixture.output, native_binding='candidate', evidence_mode='controlled',
                           arms=['base_hermes', 'all-on'], history_mode='replay', **fixture.resources)
    case = manifest['pairs'][0]['arms']['all-on']['case']
    assert case['timeout_seconds'] == 900 + paired_history.REPLAY_SECONDS_PER_TURN * 160
    assert records.MAX_FAMILY_SECONDS < case['timeout_seconds'] <= records.MAX_REPLAY_SECONDS


def test_a_restart_after_the_replay_continues_the_clock_the_replay_left(tmp_path, clock, monkeypatch):
    """The replay takes real time and no episode time: the clock runs again from the episode's start once the
    history is in. A workflow's next phase (mind-memory-2's restart) starts where the phase that replayed left
    the clock, not where the pin and the declared advances alone would put it, which is later by however long the
    replay took."""
    import sys
    from types import SimpleNamespace
    from colonymind.qualification import paired_workflow_runtime as runtime
    real = [1_800_000_000.0]
    clock._install(hermes=False)
    monkeypatch.setitem(clock._ORIGINAL, 'time', lambda: real[0])
    monkeypatch.setitem(sys.modules, 'hermes_time', SimpleNamespace(now=lambda: None))
    episodes = [{'session_id': 'owner-1', 'user': 'Where is the spare key?'}, {'advance_clock': 3600},
                {'session_id': 'owner-2', 'user': 'And the budget draft?'}]
    payload = {'binding': 'candidate', 'config': {}, 'inputs': {
        'arm': 'all-on', 'episodes': episodes, 'initial_files': {}, 'workflow': {'restart_before': [2]},
        'history': SESSIONS, 'history_mode': 'replay'}}
    home, workspace = tmp_path / 'home', tmp_path / 'workspace'
    carried, ended, started = [], [], []

    def post(path, body):
        real[0] += 600                                           # each replayed turn takes ten minutes to process
        return {'accepted': True}

    def phase(request, stop, recorder):
        """One phase as the worker runs it: the carried clock; in the first, the replay before the first turn."""
        index = request['_workflow_phase']['index']
        before = request['_workflow_phase']['body_before']
        carried.append(before['clock_offset_seconds'])
        clock.install_clock(before['clock_offset_seconds'])
        rows = []
        if index == 0:
            home.mkdir()
            workspace.mkdir()
            start = time.time()
            paired_history.rewind(SESSIONS, home / 'memory-state')
            paired_history.replay_ledger(post, SESSIONS, contact_id='p-01', state=home / 'memory-state',
                                         drain=lambda: {'status': 'drained'})
            clock.run_clock_from(start)
        else:
            started.append(time.time())
        for position, entry in enumerate(request['inputs']['episodes']):
            if 'advance_clock' in entry:
                clock.advance_clock(entry['advance_clock'])
                rows.append({'index': position, 'event': 'advance_clock', 'completed': True})
            else:
                rows.append({'index': position, 'session_id': entry['session_id'], 'completed': True})
        ended.append(time.time())
        return {'pid': 4100 + index, 'exit_code': 0, 'result': {
            'stage': 'returned', 'agent_close_returned': True, 'worker_stopped': True,
            'workflow_phase': {'index': index, 'pid': 4100 + index},
            'tool_evidence': {'turns': rows, 'turns_completed': len(rows),
                              'body': {'clock_offset_seconds': clock.clock_offset(), 'ticks': []}}}}

    result = runtime.supervise(payload, home=home, workspace=workspace, runner=phase,
                               trace=runtime.TraceForwarder(sink=lambda _: None))
    assert result['stage'] == 'returned', result.get('private_error_traceback')
    replayed = 4 * 600
    assert carried == [0, 3600 - replayed]                      # the phase's own clock, not the pin plus 3600
    assert started == [ended[0]]                                 # no jump at the restart


def test_a_phase_that_reports_no_clock_starts_no_next_phase(tmp_path):
    """The next phase's clock is the last phase's own: a phase that did not report it ends the workflow rather
    than restarting from a guessed one."""
    from colonymind.qualification import paired_workflow_runtime as runtime
    episodes = [{'session_id': 'owner-1', 'user': 'Where is the spare key?'},
                {'session_id': 'owner-2', 'user': 'And the budget draft?'}]
    payload = {'binding': 'candidate', 'config': {}, 'inputs': {
        'arm': 'all-on', 'episodes': episodes, 'initial_files': {}, 'workflow': {'restart_before': [1]}}}
    home, workspace = tmp_path / 'home', tmp_path / 'workspace'
    phases = []

    def phase(request, stop, recorder):
        index = request['_workflow_phase']['index']
        phases.append(index)
        home.mkdir(exist_ok=True)
        workspace.mkdir(exist_ok=True)
        return {'pid': 4200 + index, 'exit_code': 0, 'result': {
            'stage': 'returned', 'agent_close_returned': True, 'worker_stopped': True,
            'workflow_phase': {'index': index, 'pid': 4200 + index},
            'tool_evidence': {'turns': [{'index': 0, 'session_id': 'owner-1', 'completed': True}],
                              'turns_completed': 1, 'body': {'ticks': []}}}}

    result = runtime.supervise(payload, home=home, workspace=workspace, runner=phase,
                               trace=runtime.TraceForwarder(sink=lambda _: None))
    assert result['stage'] == 'error' and phases == [0]
    assert 'did not report its clock' in result['private_error_traceback']

"""P1 gate: historical baseline, standing context, ladder and erasure on real stores."""
from contextlib import closing
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from colonymind.api.routers import host
from colonymind.config import load_config
from colonymind.memory import blocks, compass
from colonymind.mind.lessons import CURRENT, STANDING_LESSONS, TASK_LESSONS
from colonymind.plasticity import _io
from colonymind.plasticity.comparator import _baseline_settings, _effective, BaselineComparator
from colonymind.plasticity.gates import Check, GateReport, two_proportion_bounds
from colonymind.plasticity.learners import behaviour
from colonymind.plasticity.nightly import Runtime
from colonymind.plasticity.registration import check_learner
from colonymind.work.store import WorkStore
from test_mind_loop import Fixture, OWNER, pinned_clock  # noqa: F401


@pytest.fixture
def unit(tmp_path, monkeypatch):
    monkeypatch.setenv('COLONYMIND_OWNER_CONTACT_ID', OWNER)
    fx = Fixture(tmp_path)
    WorkStore(fx.ledger)
    cfg = load_config(tmp_path, environ={})
    spec = deepcopy(behaviour.DECLARATION)
    spec['target'] = {'signal': 'long_dash', 'rank': 1}
    spec['lesson'] = {'source': 'owner', 'topic': 'reply style'}
    cfg.data['plasticity']['learners'] = {'style-rule': spec}
    runtime = Runtime(cfg, clock=lambda: fx.now)
    assert not runtime.rejected
    driver = runtime.loaded['style-rule'].driver
    driver.bind_mind(fx.mind)
    fx.mind.plasticity = runtime
    fx.mind.lessons.standing_pins = lambda: behaviour.pins(runtime)
    monkeypatch.setattr(host, '_mind', lambda: fx.mind)
    yield SimpleNamespace(fx=fx, runtime=runtime, driver=driver, spec=spec, cfg=cfg)
    runtime.close()
    fx.store.close()


def teach(unit):
    fx = unit.fx
    fx.ledger.record_source('owner-rule', contact_id=OWNER, session_id='rule', derive_claims=False,
        messages=[{'role': 'user', 'content': 'Please follow this reply style.'}], occurred_at=fx.now.isoformat())
    lesson = fx.mind.lessons.admit({'signature': 'topic:reply-style', 'kind': 'pitfall',
        'title': 'Reply style', 'when_to_use': 'writing a reply', 'content': 'Use short sentences.'},
        verified='owner', origin='night', status='active', evidence=['turn:owner-rule'],
        lineage=['owner-rule'], now=fx.now)
    assert lesson
    assert unit.runtime.run()[0]['result'] == 'promoted'
    return lesson


def population(unit, prefix, n, events, *, start, units=12, substitute=0, failure=0, corrected=0):
    fx = unit.fx
    for i in range(n):
        event = (i + 1) * events // n > i * events // n
        text = ('alpha—beta.' if event else 'alpha beta.')
        if i < substitute:
            text += ' words - more.'
        text = text.ljust(100, '.')
        ident = f'{prefix}-{i}'
        at = start + timedelta(minutes=i)
        op = {'operation_id': ident, 'tool_call_id': ident, 'tool': 'lookup', 'action': '', 'effect': 'read',
              'success': i >= failure, 'source_refs': [],
              'failure_kind': 'malformed_arguments' if i < failure else None}
        work = {'envelope': {'version': 1, 'root_turn_id': ident, 'parent_turn_id': None,
                'operations': [op], 'children': [], 'truncated': False}, 'principal_id': 'host'}
        fx.ledger.record_source(ident, contact_id=OWNER, session_id=f'{prefix}-session-{i % units}',
            messages=[{'role': 'user', 'content': 'A request.'}, {'role': 'assistant', 'content': text}],
            occurred_at=at.isoformat(), channel_id='chat:thread', work=work, derive_claims=False)
        if corrected:
            with closing(fx.ledger._connect()) as conn, conn:
                conn.execute('''INSERT INTO work_verdict_pending
                    (id,root_id,contact_id,generation,verdict,verdict_at,source_refs_json,identity_json,captured_revision)
                    VALUES (?,?,?,?,?,?,?,?,?)''', (ident, ident, OWNER, 1,
                    'corrected' if i < corrected else 'confirmed', at.isoformat() + ':00000001', '[]', '{}', 1))


def baseline(unit, events=27, n=134, units=12):
    population(unit, 'before', n, events, start=unit.fx.now - timedelta(days=1), units=units)


def advance(unit, events=1, n=60, units=12, **kwargs):
    start = unit.fx.now + timedelta(seconds=1)
    population(unit, f'after-{unit.fx.now.day}', n, events, start=start, units=units, **kwargs)
    unit.fx.now += timedelta(days=1)
    return unit.runtime.run()[0]


def test_bounds_and_cluster_effective_counts():
    lower, upper = two_proportion_bounds(19, 100, 12, 400)
    assert lower == pytest.approx(.10, abs=.01) and upper > lower
    for counts in ((0, 100, 0, 60), (0, 100, 12, 60), (19, 100, 0, 60)):
        lo, hi = two_proportion_bounds(*counts)
        assert -1 <= lo <= hi <= 1 and np.isfinite([lo, hi]).all()
    assert _effective([(0, 1), (1, 1), (0, 1)])['de'] == 1
    assert _effective([(0, 5)] * 12)['de'] == 1  # The other arm supplies ICC in the comparison.
    assert _effective([(1, 5)] * 12)['de'] == 1
    assert _effective([(0, 5)] * 11 + [(5, 5)])['de'] > 1


def test_registration_has_no_forced_share_requirement(unit):
    assert check_learner('style-rule', unit.spec).forced_run_share is None
    for changes, error in (({'effect': 'predict'}, 'effect'), ({'audience': 'guest'}, 'audience'),
                           ({'model': {'roles': ['standing_lessons'], 'change': 'weights'}}, 'model.change')):
        with pytest.raises(ValueError, match=error):
            check_learner('style-rule', {**unit.spec, **changes})
    other = {**unit.spec, 'kind': 'numeric-online', 'effect': 'predict',
             'model': {'roles': ['numeric_predictor'], 'change': 'context'}}
    with pytest.raises(ValueError, match='model.change'):
        check_learner('other', other)


@pytest.mark.parametrize('options', [dict(min_replies=True), dict(min_session_days=0), dict(days=1.5),
    dict(target_cap=float('nan')), dict(target_cap=1.1), dict(guards={'unknown': .1}),
    dict(guards={'median_length': -.1}), dict(unknown=1)])
def test_settings_reject_invalid_values(options):
    with pytest.raises(ValueError, match='baseline'):
        _baseline_settings({'baseline': options})


def test_owner_waits_and_reflector_default_is_idle(unit, tmp_path):
    result = unit.runtime.run()[0]
    assert result['result'] == 'unchanged' and result['why'] == ['awaiting_owner_lesson']
    assert not unit.fx.mind.lessons.all()
    other = tmp_path / 'default'
    other.mkdir()
    runtime = Runtime(load_config(other, environ={}))
    try:
        assert isinstance(runtime.loaded['behaviour-tool-failure'].comparator, BaselineComparator)
        assert runtime.run()[0]['why'] == ['awaiting_lesson']
        assert 'content' not in runtime.declarations['behaviour-tool-failure']['lesson']
    finally:
        runtime.close()


def test_frozen_baseline_pass_and_reload(unit):
    baseline(unit)
    teach(unit)
    assert unit.runtime.run()[0]['result'] == 'awaiting_evidence'
    first = unit.driver.comparator.summary()['baseline_counts']
    assert any(r['outcome'] == 'awaiting_evidence' for r in
               unit.runtime.loaded['style-rule'].store.certificates.records())
    # Late ingestion of pre-promotion history cannot move a frozen baseline.
    population(unit, 'late', 10, 10, start=unit.fx.now - timedelta(hours=2))
    result = advance(unit)
    assert result['result'] == 'promoted'
    summary = unit.driver.comparator.summary()
    assert summary['verdict'] == 'passed'
    assert summary['target']['baseline']['x'] == 27 and summary['target']['baseline']['n'] == 134
    assert summary['target']['taught']['x'] == 1 and summary['target']['taught']['n'] == 60
    assert summary['counts']['session_days'] == 12 and summary['target']['lower'] > 0
    assert summary['baseline_counts'] == first and 'chat' in summary['platforms']
    reloaded = BaselineComparator(unit.driver.controller, key=b'same-key', clock=lambda: unit.fx.now)
    closure = unit.driver.store.revisions.serving()
    assert reloaded.evaluate(closure, unit.driver.observations(closure))['baseline_counts'] == first
    assert not unit.driver.store.periods.retained()
    for path in unit.driver.store.directory.rglob('*.json'):
        assert 'alpha' not in path.read_text()


@pytest.mark.parametrize('n,units', [(59, 12), (60, 11)])
def test_below_either_minimum_waits_without_rollback(unit, n, units):
    baseline(unit)
    teach(unit)
    closure = unit.driver.store.revisions.serving().id
    assert advance(unit, n=n, units=units)['result'] == 'awaiting_evidence'
    assert unit.driver.store.revisions.serving().id == closure


def test_reversal_rolls_back_and_ladder_skips_contrastless_level(unit):
    baseline(unit, 0)
    teach(unit)
    assert advance(unit, events=60)['result'] == 'rolled_back'
    assert not behaviour.pins(unit.runtime)
    assert unit.driver.store.revisions.serving().mode == 'base'
    assert unit.runtime.run()[0]['result'] == 'promoted'
    assert behaviour.pins(unit.runtime)[0].level == 3
    # The original lesson baseline survives the ladder promotion.
    assert unit.driver.comparator.summary()['target']['baseline']['x'] == 0
    assert advance(unit, events=60)['result'] == 'rolled_back'
    assert unit.runtime.run()[0]['why'] == ['ladder_exhausted']


def test_restart_keeps_frozen_baseline_and_resets_only_taught_window(unit):
    baseline(unit)
    teach(unit)
    assert advance(unit)['result'] == 'promoted'
    old = unit.driver.comparator.summary()
    unit.cfg.data['plasticity']['behaviour']['restarts'] = [_io.stamp(unit.fx.now)]
    unit.fx.now += timedelta(seconds=1)
    assert unit.runtime.run()[0]['result'] == 'awaiting_evidence'
    new = unit.driver.comparator.summary()
    assert new['window'] != old['window'] and new['counts']['taught'] == 0
    assert new['baseline_counts'] == old['baseline_counts']
    assert new['target']['baseline']['n'] == 134 and new['certified']
    assert new['restarts'] == [{'at': unit.cfg.get('plasticity.behaviour.restarts')[0], 'baseline_predates': True}]
    assert new['confounds'] == [f"baseline predates a head/stack change at {new['restarts'][0]['at']}"]
    record = unit.driver.store.certificates.records()[-1]
    assert record['comparisons']['restarts'] == new['restarts']
    assert record['comparisons']['confounds'] == new['confounds']
    # Reloading and adding another restart preserve counts and certification.
    reloaded = BaselineComparator(unit.driver.controller, key=b'reload', clock=lambda: unit.fx.now)
    closure = unit.driver.store.revisions.serving()
    assert reloaded.evaluate(closure, unit.driver.observations(closure))['certified']
    unit.cfg.data['plasticity']['behaviour']['restarts'].append(_io.stamp(unit.fx.now))
    unit.fx.now += timedelta(seconds=1)
    assert unit.runtime.run()[0]['result'] == 'awaiting_evidence'
    assert len(unit.driver.comparator.summary()['restarts']) == 2
    assert advance(unit, events=11)['result'] == 'rolled_back'


@pytest.mark.parametrize('guard', ['substitute', 'failure', 'corrected'])
def test_guard_breach_rolls_back_and_halts_ladder(unit, guard):
    baseline(unit, units=30)
    # Owner corrections need observed verdicts in both windows, including confirmations.
    if guard == 'corrected':
        with closing(unit.fx.ledger._connect()) as conn, conn:
            for i in range(134):
                conn.execute('''INSERT INTO work_verdict_pending
                    (id,root_id,contact_id,generation,verdict,verdict_at,source_refs_json,identity_json,captured_revision)
                    VALUES (?,?,?,?,?,?,?,?,?)''', (f'v{i}', f'before-{i}', OWNER, 1, 'confirmed',
                    _io.stamp(unit.fx.now - timedelta(hours=1)), '[]', '{}', 1))
    teach(unit)
    assert advance(unit, units=30, **{guard: 60})['result'] == 'rolled_back'
    lesson = unit.fx.mind.lessons.all()[0]
    assert unit.driver.ladder()[lesson.id]['halted']
    assert unit.runtime.run()[0]['why'] == ['ladder_exhausted']


@pytest.mark.asyncio
async def test_standing_host_use_retire_task_and_meaning_path(unit, monkeypatch):
    lesson = teach(unit)
    text, last = await host._mind_standing('different request', viewer_is_owner=True, owner_turn=True,
                                          session_id='chat', turn_id='turn-1')
    assert lesson.content in text and not last
    for kwargs in ({'viewer_is_owner': False}, {'owner_turn': False}, {'task_run': True},
                   {'session_id': 'mind:contact'}, {'session_id': 'mind'}):
        args = dict(viewer_is_owner=True, owner_turn=True, session_id='chat', turn_id='turn-2', **{})
        args.update(kwargs)
        assert await host._mind_standing('request', **args) == ('', False)
    rows = unit.fx.store.lesson_rows(since=unit.fx.now - timedelta(days=1))
    use = next(r for r in rows if r.type == 'lesson_use')
    assert use.context['turn_id'] == 'turn-1' and use.context['match'] == 'standing'
    assert use.context['standing']['level'] == 1 and 'arm' not in use.context
    monkeypatch.setattr(unit.fx.mind.lessons, '_related', lambda lessons, work: [(1, l) for l in lessons])
    for i in range(3):
        unit.fx.mind.lessons.admit({'signature': f'topic:other-{i}', 'kind': 'strategy', 'title': f'Title {i}',
            'when_to_use': 'a task', 'content': f'Other instruction {i}'}, verified='owner', origin='night',
            status='active', evidence=['turn:owner-rule'], lineage=['owner-rule'], now=unit.fx.now)
    _, ids = unit.fx.mind.lessons.for_turn('request', session_id='chat', work=object(), record=False)
    assert len(ids) == 1 and lesson.id not in ids
    candidate = SimpleNamespace(type='research', title='a task', topic='', concern='')
    lines, ids = unit.fx.mind.lessons.for_task(candidate, work=object())
    assert ids[0] == lesson.id and len(ids) == TASK_LESSONS + 1 and lesson.content in lines[0]
    unit.fx.mind.lessons.review(tallies={lesson.id: {'uses': 8, 'wins': 0}})
    assert unit.fx.mind.lessons.get(lesson.id).status in CURRENT
    assert lesson.id not in unit.fx.mind.lessons._cool(unit.fx.now + timedelta(days=90))
    for op in ('retire', 'supersede'):
        assert unit.fx.mind.lessons._validate({'op': op, 'lesson_id': lesson.id}, {}, {})[1] == 'standing lesson: judged by its learner'
    unit.fx.mind.lessons.retire(lesson.id, reason='Owner withdrew it.')
    assert not unit.fx.mind.lessons.standing()
    assert await host._mind_standing('request', viewer_is_owner=True, owner_turn=True,
                                    session_id='chat', turn_id='turn-3') == ('', False)


def test_contrast_ladder_and_example_erasure(unit):
    unit.driver.contrast = behaviour.replacements([{'pattern': '—', 'replacement': ', '}])
    baseline(unit, 0)
    lesson = teach(unit)
    assert advance(unit, events=60)['result'] == 'rolled_back'
    assert unit.runtime.run()[0]['result'] == 'promoted'
    pin = behaviour.pins(unit.runtime)[0]
    assert pin.level == 2 and 'Before:' in pin.example
    example = unit.driver.example()
    state = unit.driver.store.revisions.state(unit.driver.store.revisions.serving())
    assert 'Before:' in unit.driver.render(state)
    # Source erasure resolves through the runtime's existing erasure transaction.
    with closing(unit.fx.ledger._connect()) as conn, conn:
        conn.execute('DELETE FROM turn_sources WHERE turn_id=?', (example['turn_id'],))
    assert unit.runtime.erase_sources([example['turn_id']])['clean']
    assert unit.driver.example() is None and 'Before:' not in unit.driver.render(state)
    assert unit.driver.render(state) == lesson.standing_section(1)
    for path in unit.runtime.root.rglob('*'):
        if path.is_file():
            assert example['turn_id'].encode() not in path.read_bytes()


def test_pinned_section_registries_and_awaiting_contract():
    section = 'colonymind-standing-lessons'
    assert section in compass.PINNED_SECTIONS and section in blocks.ALWAYS_WHOLE
    assert blocks.LIMITS[section] == 900
    assert blocks.RANK.index(section) == blocks.RANK.index('colonymind-corrections') + 1
    assert STANDING_LESSONS == 2
    report = GateReport('closure', 'suite', (Check('structural', True), Check('target', False, awaiting=True)))
    assert report.awaiting and report.to_dict()['awaiting']
    assert not GateReport('closure', 'suite', (*report.checks, Check('guard', False))).awaiting


def test_target_miss_after_a_real_pass_rolls_back(unit):
    # A wider cap isolates loss of a previously significant benefit from the cap rule.
    unit.driver.comparator.settings['target_cap'] = .25
    baseline(unit)
    teach(unit)
    assert advance(unit)['result'] == 'promoted'
    unit.fx.now += timedelta(days=15)  # Discard the passing window from standing evaluation.
    assert advance(unit, events=11)['result'] == 'rolled_back'
    target = unit.driver.comparator.summary()['target']
    assert .15 <= target['taught']['rate'] <= .20
    assert target['lower'] < 0 < target['upper']  # Inconclusive, never a measured reversal.
    assert target['taught']['rate'] < unit.driver.comparator.settings['target_cap']
    assert unit.driver.store.revisions.serving().mode == 'base'


def test_second_behaviour_target_loads_and_global_cap_is_two(unit):
    spec = deepcopy(unit.spec)
    spec['target']['rank'] = 2
    unit.cfg.data['plasticity']['learners']['second-rule'] = spec
    unit.cfg.data['plasticity']['learners']['third-rule'] = deepcopy(spec)
    unit.runtime.close()
    runtime = Runtime(unit.cfg, clock=lambda: unit.fx.now)
    unit.runtime = runtime
    unit.driver = runtime.loaded['style-rule'].driver
    unit.fx.mind.plasticity = runtime
    unit.fx.mind.lessons.standing_pins = lambda: behaviour.pins(runtime)
    for binding in runtime.loaded.values():
        binding.driver.bind_mind(unit.fx.mind)
    try:
        teach(unit)
        result = {r['learner']: r for r in runtime.run()}
        assert result['third-rule']['result'] == 'unchanged'
        assert len(behaviour.pins(runtime)) == STANDING_LESSONS
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_real_context_assembly_includes_pinned_section_and_level_three_moves_it_last(unit, monkeypatch, source_app):
    from httpx import ASGITransport, AsyncClient
    from colonymind.api.middleware import ApiKeyMiddleware
    from onekey import AUTH, KEY
    lesson = teach(unit)
    source_app.add_middleware(ApiKeyMiddleware, api_key=KEY)

    async def assemble(turn):
        async with AsyncClient(transport=ASGITransport(app=source_app), base_url='http://test') as client:
            response = await client.post('/v1/host/context/assemble', headers=AUTH, json={
                'identity': {'host_id': 'fixture'},
                'context': {'contact_id': OWNER, 'session_id': 'chat-context', 'turn_id': turn},
                'incoming_message': {'role': 'user', 'content': 'What should I do next?'}})
        assert response.status_code == 200, response.text
        return response.json()['sections']

    sections = await assemble('context-1')
    standing = next(s for s in sections if s['id'] == 'colonymind-standing-lessons')
    assert standing['title'] == 'Standing lessons' and standing['priority'] == 88
    assert lesson.content in standing['body']
    assert not any(s['id'] == 'colonymind-lessons' for s in sections)
    unit.runtime.rollback('style-rule')
    _io.write_json(unit.driver.store.directory / 'behaviour-ladder.json',
                   {lesson.id: {'withdrawn': [1], 'halted': False}})
    assert unit.runtime.run()[0]['result'] == 'promoted'
    assert behaviour.pins(unit.runtime)[0].level == 3
    assert (await assemble('context-2'))[-1]['id'] == 'colonymind-standing-lessons'
    unit.fx.mind.lessons.retire(lesson.id, reason='Withdrawn by owner.')
    assert not any(s['id'] == 'colonymind-standing-lessons' for s in await assemble('context-3'))


# Fixture imports are explicit: these tests use an in-process ASGI host and mocked adapter transport.
from test_turn_source_evidence import source_app  # noqa: F401, E402
from test_colonymind_memory_provider import provider_mod  # noqa: F401, E402


def test_provider_sends_standing_section_in_full_on_repeated_turns(provider_mod, monkeypatch, tmp_path):
    from test_colonymind_memory_unchanged_sections import _provider, _turn
    section = {'id': 'colonymind-standing-lessons', 'title': 'Standing lessons',
               'body': 'A standing instruction.', 'priority': 88}
    monkeypatch.setenv('COLONYMIND_HOME', str(tmp_path / 'instance'))
    provider = _provider(provider_mod, monkeypatch, [section])
    for query in ('First request.', 'Second request.'):
        text = _turn(provider, query)
        assert '## Standing lessons\nA standing instruction.' in text
        assert 'unchanged since your last turn: Standing lessons' not in text


@pytest.mark.asyncio
async def test_task_body_includes_the_standing_instruction(unit):
    from test_mind_lessons import candidate
    import json
    lesson = teach(unit)
    row = await unit.fx.mind._form(candidate(1), .9, unit.fx.now)
    assert lesson.id in json.loads(row.lesson_ids)
    assert lesson.content in row.context['body']
    assert row.context['lesson_match'][lesson.id] == 'standing'


def test_owner_source_requires_verified_topic_and_rejects_relayed_copy(unit):
    lessons = unit.fx.mind.lessons
    fields = {'signature': 'topic:reply-style', 'kind': 'pitfall', 'title': 'Reply style',
              'when_to_use': 'writing', 'content': 'Use short sentences.'}
    lessons.admit(fields, verified='check', origin='reflector', status='active', evidence=[], lineage=[], now=unit.fx.now)
    assert unit.runtime.run()[0]['why'] == ['awaiting_owner_lesson']
    lessons.admit({**fields, 'signature': 'topic:unrelated'}, verified='owner', origin='night', status='active',
                  evidence=[], lineage=[], now=unit.fx.now)
    assert unit.runtime.run()[0]['why'] == ['awaiting_owner_lesson']


@pytest.mark.parametrize('per_unit', [5, 10])
def test_zero_event_bounds_are_monotone_and_zero_of_sixty_passes(unit, per_unit):
    from colonymind.plasticity.comparator import _rate_comparison
    before = [[sum((i + 1) * 27 // 134 > i * 27 // 134 for i in range(j, 134, 12)),
               len(range(j, 134, 12))] for j in range(12)]
    lowers = []
    for events in range(4):
        after = [[int(j < events), per_unit] for j in range(12)]
        lowers.append(_rate_comparison(before, after)['lower'])
    assert all(left >= right for left, right in zip(lowers, lowers[1:]))
    baseline(unit)
    teach(unit)
    assert advance(unit, events=0)['result'] == 'promoted'
    assert unit.driver.comparator.summary()['target']['taught']['n'] == 60


def test_degenerate_arm_borrows_other_population_correlation():
    from colonymind.plasticity.comparator import _rate_comparison
    other = [(5, 10)] * 6 + [(0, 10)] * 6
    compared = _rate_comparison(other, [(0, 5)] * 12)
    b, t = compared['baseline'], compared['taught']
    rho = max(0, min(1, (b['de'] - 1) / (b['mbar'] - 1)))
    assert t['de'] == pytest.approx(1 + 4 * rho)
    assert t['de'] > 1
    assert _rate_comparison([(1, 1), (0, 1)], [(0, 5)] * 12)['taught']['de'] == 1
    both = _rate_comparison([(1, 1)], [(0, 5)] * 12)
    assert both['baseline']['de'] == both['taught']['de'] == 1
    assert _rate_comparison(other, [(5, 5)] * 12)['taught']['de'] == pytest.approx(t['de'])


def test_guard_minimum_defaults_to_target_units_and_unmeasured_are_reported(unit):
    assert _baseline_settings({'baseline': {'min_session_days': 7}})['min_guard_units'] == 7
    assert _baseline_settings({'baseline': {'min_guard_units': 4}})['min_guard_units'] == 4
    baseline(unit)
    teach(unit)
    result = advance(unit)
    summary = unit.driver.comparator.summary()
    assert summary['guards']['substitute_rate']['verdict'] == 'inconclusive'
    assert summary['guards']['median_length']['verdict'] == 'non_inferior'
    assert summary['guards']['tool_failure_rate']['verdict'] == 'inconclusive'
    assert summary['not_measured_guards'] == ['owner_corrections']
    assert result['not_measured_guards'] == summary['not_measured_guards']
    assert unit.driver.store.certificates.records()[-1]['comparisons']['not_measured_guards'] == ['owner_corrections']


def test_tool_guard_counts_every_attempt_including_parse_only_and_mixed_turns(unit):
    from datetime import datetime, timezone
    at = datetime.now(timezone.utc)
    reply = behaviour.ReplyRow(at, 'chat', 'model', 'Reply.')
    def operation(ident, tool, failure=None):
        return behaviour.OperationRow(at, 'chat', 'model', 'turn', ident, tool, not failure, failure)
    parse = operation('parse', 'model_response', 'parse_failure')
    malformed = operation('call', 'lookup', 'malformed_arguments')
    success = operation('success', 'lookup')
    cases = [([parse], (1, 1)), ([parse, malformed], (2, 2)),
             ([parse, malformed, success], (2, 3)), ([success], (0, 1)),
             ([parse, parse], (1, 1)), ([operation('response', 'model_response')], (0, 0))]
    for operations, expected in cases:
        assert behaviour._signal(reply, operations)['tool_failure_rate'] == expected


@pytest.mark.parametrize('invalid', ['2026-01-01T00:00:00', 'invalid', 42, {'at': '2026-01-01T00:00:00Z'},
                                    ['2026-01-01'], ['2026-01-01T00:00:00Z', 'bad']])
def test_invalid_restarts_rejected_at_config_load_and_driver_init(unit, tmp_path, invalid):
    import yaml
    from colonymind.config import ConfigError
    home = tmp_path / 'invalid-config'
    home.mkdir()
    (home / 'colonymind.yaml').write_text(yaml.safe_dump({'plasticity': {'behaviour': {'restarts': invalid}}}))
    with pytest.raises(ConfigError, match='restarts'):
        load_config(home, environ={})
    controller = unit.driver.controller
    controller.behaviour_restarts = invalid
    with pytest.raises(ValueError, match='restarts'):
        behaviour.Driver(store=unit.driver.store, controller=controller,
                         source=unit.runtime.loaded['style-rule'].source,
                         declaration=unit.spec, clock=lambda: unit.fx.now)


def test_timezone_aware_restarts_load_and_initialize_driver(unit, tmp_path):
    from colonymind.config import validate_behaviour_restarts
    values = ['2026-01-01T00:00:00Z', '2026-01-02T01:00:00-05:00']
    assert validate_behaviour_restarts(values) == values
    unit.cfg.data['plasticity']['behaviour']['restarts'] = values
    unit.runtime.close()
    runtime = Runtime(unit.cfg, clock=lambda: unit.fx.now)
    try:
        assert 'style-rule' in runtime.loaded and not runtime.rejected
    finally:
        runtime.close()


@pytest.mark.parametrize('stage', ['build', 'gate'])
def test_nightly_evaluation_error_rolls_back_serving_behaviour(unit, monkeypatch, stage):
    teach(unit)
    def fail(*args, **kwargs):
        raise RuntimeError('evaluation failed')
    monkeypatch.setattr(unit.driver, stage, fail)
    assert unit.runtime.run()[0]['result'] == 'error'
    assert unit.driver.store.revisions.serving().mode == 'base'
    assert not behaviour.pins(unit.runtime)
    assert any(e.get('reason') == 'nightly_error' for e in unit.driver.controller.events())


def test_min_replies_counts_only_replies_with_target(unit):
    unit.driver.signal = 'tool_failure'
    baseline(unit)
    teach(unit)
    population(unit, 'eligible', 59, 0, start=unit.fx.now + timedelta(seconds=1), units=12)
    unit.fx.ledger.record_source('without-target', contact_id=OWNER, session_id='no-tools', derive_claims=False,
        occurred_at=_io.stamp(unit.fx.now + timedelta(hours=2)),
        messages=[{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi.'}])
    unit.fx.now += timedelta(days=1)
    assert unit.runtime.run()[0]['result'] == 'awaiting_evidence'
    target = unit.driver.comparator.summary()['target']['taught']
    assert target['n'] == 59 and target['units'] >= 12


def test_contrast_source_fetch_is_by_turn_id_without_read_rows(unit, monkeypatch):
    unit.driver.contrast = behaviour.replacements([{'pattern': '—', 'replacement': ', '}])
    baseline(unit, 0)
    teach(unit)
    assert advance(unit, events=60)['result'] == 'rolled_back'
    assert unit.runtime.run()[0]['result'] == 'promoted'
    state = unit.driver.store.revisions.state(unit.driver.store.revisions.serving())
    queries = []
    connect = unit.fx.ledger._connect
    def traced():
        conn = connect()
        conn.set_trace_callback(queries.append)
        return conn
    monkeypatch.setattr(unit.fx.ledger, '_connect', traced)
    monkeypatch.setattr(behaviour, 'read_rows', lambda *a, **kw: pytest.fail('full population read'))
    assert 'Before:' in unit.driver.example_text(state)
    assert any('WHERE turn_id=' in sql for sql in queries)


def test_owner_source_skips_retired_topic_for_current_lesson(unit):
    old = teach(unit)
    unit.runtime.rollback('style-rule')
    unit.fx.mind.lessons.retire(old.id, reason='Withdrawn')
    assert unit.driver._lesson() is None
    unit.fx.now += timedelta(seconds=1)
    fields = {'signature': 'topic:reply-style', 'kind': 'pitfall', 'title': 'Current style',
              'when_to_use': 'writing', 'content': 'A new instruction.'}
    current = unit.fx.mind.lessons.admit(fields, verified='owner', origin='night', status='active',
                                        evidence=[], lineage=[], now=unit.fx.now)
    assert unit.driver._lesson().id == current.id != old.id
    assert unit.runtime.run()[0]['result'] == 'promoted'
    assert behaviour.pins(unit.runtime)[0].lesson_id == current.id


def test_z_is_fixed_and_cannot_be_declared():
    from colonymind.plasticity.comparator import ONE_SIDED_95_Z
    assert ONE_SIDED_95_Z == pytest.approx(1.6448536269514722)
    with pytest.raises(ValueError, match='unknown settings'):
        _baseline_settings({'baseline': {'z': 1.0}})


def test_above_target_cap_fails_with_minimums_even_when_significantly_better(unit):
    baseline(unit, events=100)
    teach(unit)
    assert advance(unit, events=6)['result'] == 'rolled_back'
    target = unit.driver.comparator.summary()['target']
    assert target['lower'] > 0 and target['taught']['rate'] > unit.driver.settings['target_cap']
    assert target['verdict'] == 'failed'
    assert unit.runtime.run()[0]['result'] == 'promoted'
    assert behaviour.pins(unit.runtime)[0].level == 3


@pytest.mark.asyncio
async def test_standing_errors_return_empty_and_keep_other_context_available(unit, monkeypatch):
    teach(unit)
    lessons = unit.fx.mind.lessons
    def fail():
        raise RuntimeError('unavailable')
    lessons.standing_pins = fail
    assert lessons.standing_ids() == set()
    assert lessons.standing() == []
    assert await host._mind_standing('query', viewer_is_owner=True, owner_turn=True, session_id='chat') == ('', False)
    monkeypatch.setattr(lessons, 'standing', fail)
    assert await host._mind_standing('query', viewer_is_owner=True, owner_turn=True, session_id='chat') == ('', False)


@pytest.mark.asyncio
async def test_turn_context_assembles_while_nightly_lock_held(unit, monkeypatch, source_app):
    import threading
    import time
    from httpx import ASGITransport, AsyncClient
    from colonymind.api.middleware import ApiKeyMiddleware
    from onekey import AUTH, KEY
    lesson = teach(unit)
    source_app.add_middleware(ApiKeyMiddleware, api_key=KEY)
    async def assemble(turn):
        async with AsyncClient(transport=ASGITransport(app=source_app), base_url='http://test') as client:
            response = await client.post('/v1/host/context/assemble', headers=AUTH, json={
                'identity': {'host_id': 'fixture'},
                'context': {'contact_id': OWNER, 'session_id': 'chat-context', 'turn_id': turn},
                'incoming_message': {'role': 'user', 'content': 'What should I do next?'}})
        assert response.status_code == 200
        return response.json()['sections']
    expected = {s['id'] for s in await assemble('before-lock')}
    assert 'colonymind-standing-lessons' in expected and len(expected) > 1
    held, release = threading.Event(), threading.Event()
    def hold():
        with unit.runtime.lock:
            held.set()
            release.wait(5)
    thread = threading.Thread(target=hold)
    thread.start()
    assert held.wait(1)
    try:
        start = time.monotonic()
        assert isinstance(behaviour.pins(unit.runtime), tuple)
        sections = await assemble('during-lock')
        assert time.monotonic() - start < 1
        assert {s['id'] for s in sections} == expected
        assert lesson.content in next(s['body'] for s in sections if s['id'] == 'colonymind-standing-lessons')
    finally:
        release.set()
        thread.join()


def test_snapshot_published_at_promotion_rollback_and_startup(unit):
    teach(unit)
    first = behaviour.pins(unit.runtime)
    assert len(first) == 1
    with pytest.raises(Exception):
        first[0].level = 3
    unit.runtime.close()
    runtime = Runtime(unit.cfg, clock=lambda: unit.fx.now)
    try:
        assert behaviour.pins(runtime) == first
        runtime.loaded['style-rule'].driver.bind_mind(unit.fx.mind)
        runtime.rollback('style-rule')
        assert not behaviour.pins(runtime)
    finally:
        runtime.close()


def test_first_promotion_freezes_baseline_before_first_evaluation_and_ladder_keeps_it(unit):
    baseline(unit, 0)
    lesson = teach(unit)
    path = unit.driver.store.directory / 'behaviour-baselines.json'
    original = _io.read_json(path)[lesson.id]
    assert original['counts']['all']['target'] and not original['passed']
    population(unit, 'late-before-first-evaluation', 10, 10, start=unit.fx.now - timedelta(hours=1))
    assert advance(unit, events=60)['result'] == 'rolled_back'
    assert unit.runtime.run()[0]['result'] == 'promoted'
    assert _io.read_json(path)[lesson.id] == original
    closure = unit.driver.store.revisions.serving()
    evaluation = unit.driver.comparator.evaluate(closure, unit.driver.observations(closure))
    assert evaluation['baseline_counts'] == original['counts']
    assert evaluation['counts']['taught'] == 0

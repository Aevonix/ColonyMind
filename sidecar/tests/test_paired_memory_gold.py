"""Gold evidence and stale values in the mind-memory generators, for reach and trap scoring.

Each memory artifact names, in its ``gold``, the earlier turns that answer it and the values that were current once
(``stale``). Gold is additive: stripped of it, every rendered scenario is byte for byte the scenario the generator
rendered before gold existed. What gold points at is asserted from the rendered turns themselves, so a template
edit that moves a statement without moving its gold fails here.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import pytest

from colonymind.qualification import paired_cases

BENCH = Path(__file__).resolve().parents[2] / 'benchmarks'
GENERATORS = BENCH / 'paired' / 'generators'
SCORING = BENCH / 'public_memory' / 'scoring.py'
FAMILIES = ('memory.py', 'memory_hard.py')
# scenarios.json of the dev split (seed 7, three per template) as rendered before gold existed.
BEFORE_GOLD = {'memory.py': 'd352b2585a7eb15bf879122d201db411cd27aaa27abfc1542cc1ccf955d623a7',
               'memory_hard.py': '167e2318b9748a9a401fb52bd39621ac9883dfebba43526db64c6a9e24a280bd'}
MINUTE_LABELS = {'minutes', 'minute', 'min', 'mins'}
WORD = re.compile(r'\w+')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def generate():
    return load(GENERATORS / 'generate.py', 'paired_generate_gold')


@pytest.fixture(scope='module')
def rendered(generate):
    return {name: generate.render(generate.load_templates(GENERATORS / name), 7, 3) for name in FAMILIES}


def words(text):
    return WORD.findall(text.casefold())


def shows(text, value):
    have, want = words(text), words(value)
    return bool(want) and any(have[i:i + len(want)] == want for i in range(len(have) - len(want) + 1))


def strings(spec):
    """The artifact's expected string labels that a statement can carry (not "unknown", not unit spellings)."""
    return [value for item in spec['assertions'] if item['op'] == 'label_one_of' and item['path']
            for value in item['value'] if value != 'unknown' and value not in MINUTE_LABELS]


def history_text(item, ref):
    return next(s for s in item['history'] if s['id'] == ref['session'])['messages'][ref['message']]


def order(item):
    """Every history message in the order it was said: (session index, message index)."""
    return [(index, position, message) for index, session in enumerate(item['history'])
            for position, message in enumerate(session['messages'])]


def test_scorer_and_loader_agree_on_the_gold_kinds():
    assert tuple(load(SCORING, 'public_memory_scoring_gold').KINDS) == paired_cases.GOLD_KINDS


@pytest.mark.parametrize('name', FAMILIES)
def test_every_memory_artifact_carries_gold_the_loader_accepts(generate, name, tmp_path):
    module = generate.load_templates(GENERATORS / name)
    generate.write(tmp_path / 'dev', module, 7, 'dev', 3, GENERATORS / name)
    _, scenarios, _ = paired_cases.load_generated_dataset(tmp_path / 'dev')
    for item in scenarios:
        for spec in item['oracle']['artifacts']:
            assert spec['gold']['kind'] in paired_cases.GOLD_KINDS, item['id']
            assert set(spec['gold'].get('stale', [])) <= set(spec['forbidden'])
            # An answer with more than one field says which field each stale value once held.
            fields = {entry['path'][0] for entry in spec['assertions'] if len(entry['path']) == 1}
            if spec['gold'].get('stale') and len(fields) > 1:
                assert set(spec['gold']['stale_fields']) == set(spec['gold']['stale']), item['id']
                assert set(spec['gold']['stale_fields'].values()) <= fields, item['id']


@pytest.mark.parametrize('name', FAMILIES)
def test_gold_is_additive_the_scenarios_render_as_before(generate, rendered, name):
    stripped = copy.deepcopy(rendered[name])
    for item in stripped:
        for spec in item['oracle']['artifacts']:
            del spec['gold']
    assert hashlib.sha256(generate.encode(stripped)).hexdigest() == BEFORE_GOLD[name]


def test_history_gold_points_at_the_owner_statements_that_answer(rendered):
    kinds = {}
    for item in rendered['memory_hard.py']:
        [spec] = item['oracle']['artifacts']
        gold = spec['gold']
        kinds.setdefault(item['scenario'], set()).add(gold['kind'])
        said = [history_text(item, ref) for ref in gold['evidence']]
        assert all(message['role'] == 'user' for message in said), item['id']
        text = ' '.join(message['content'] for message in said)
        if gold['kind'] in ('recall', 'update'):
            assert said, item['id']
            assert all(shows(text, value) for value in strings(spec)), item['id']
        if item['scenario'] == 'spread-total':
            boxes = [int(n) for n in re.findall(r'(\d+) (?:more )?boxes', text)]
            assert len(boxes) == 3 and sum(boxes) == spec['assertions'][1]['value']
        if item['scenario'] == 'buried-preference':
            assert any('minutes' in message['content'] for message in said)
        # A stale value was stated by the owner as current, before the evidence that replaced or withdrew it.
        positions = {(s['id'], i): (n, i) for n, s in enumerate(item['history']) for i in range(len(s['messages']))}
        first_evidence = min(positions[(ref['session'], ref['message'])] for ref in gold['evidence']) \
            if gold['evidence'] else None
        for value in gold.get('stale', []):
            earlier = [(n, i) for n, i, message in order(item)
                       if message['role'] == 'user' and shows(message['content'], value)]
            assert earlier and min(earlier) < first_evidence, (item['id'], value)
        if item['scenario'] == 'channel-switch':
            # The earlier holder is a contact and the earlier place a place.
            assert {field: bool(re.fullmatch(r'p-\d\d', value)) for value, field in gold['stale_fields'].items()} \
                == {'contact': True, 'place': False}
    assert kinds == {'buried-fact': {'recall'}, 'buried-correction': {'update'}, 'channel-switch': {'update'},
                     'assistant-misquote': {'recall'}, 'restated-stale': {'update'},
                     'buried-preference': {'recall'}, 'spread-total': {'recall'}, 'near-miss': {'abstain'},
                     'retracted': {'abstain'}}


def test_misquote_gold_is_the_owner_and_the_misquote_is_no_stale_value(rendered):
    for item in rendered['memory_hard.py']:
        if item['scenario'] != 'assistant-misquote':
            continue
        [spec] = item['oracle']['artifacts']
        [ref] = spec['gold']['evidence']
        session = next(s for s in item['history'] if s['id'] == ref['session'])
        # The owner said the right place; the assistant's reply right after it restated the wrong one.
        assert shows(session['messages'][ref['message']]['content'], strings(spec)[0])
        assert shows(session['messages'][ref['message'] + 1]['content'], spec['forbidden'][0])
        assert spec['gold'].get('stale', []) == []


def test_abstention_gold(rendered):
    for item in rendered['memory_hard.py']:
        [spec] = item['oracle']['artifacts']
        gold = spec['gold']
        if item['scenario'] == 'near-miss':
            assert gold['evidence'] == [] and gold.get('stale', []) == []
        if item['scenario'] == 'retracted':
            [ref] = gold['evidence']
            assert gold['stale'] == spec['forbidden']
            assert 'where' in history_text(item, ref)['content']


def test_episode_gold_points_at_the_turns_that_answer(rendered):
    kinds = {}
    for item in rendered['memory.py']:
        [spec] = item['oracle']['artifacts']
        gold = spec['gold']
        kinds.setdefault(item['scenario'], set()).add(gold['kind'])
        owner = [ref for ref in gold['evidence'] if ref.get('role', 'user') == 'user']
        text = ' '.join(item['episodes'][ref['turn']]['user'] for ref in owner)
        if gold['kind'] in ('recall', 'update') and owner:
            assert all(shows(text, value) for value in strings(spec)), item['id']
        for value in gold.get('stale', []):
            first = min(ref['turn'] for ref in gold['evidence'])
            assert any(shows(entry.get('user', ''), value) for entry in item['episodes'][:first + 1]), item['id']
        if item['scenario'] == 'scoped-correction':
            [(value, field)] = gold['stale_fields'].items()
            assert field == 'time' and re.fullmatch(r'\d\d:\d\d', value)
        if gold['kind'] == 'conflict':
            first, second = (item['episodes'][ref['turn']]['user'] for ref in gold['evidence'])
            assert shows(first, gold['sides'][0]) and shows(second, gold['sides'][1])
        if item['scenario'] == 'own-action-recall':
            assert gold['evidence'] == [{'turn': 0, 'role': 'assistant'}]
    assert kinds == {'fact-after-restart': {'recall'}, 'fact-across-channels': {'recall'},
                     'knowledge-update': {'update'}, 'scoped-correction': {'update'},
                     'preference-after-distractors': {'recall'}, 'never-said': {'abstain'},
                     'contradiction-ask': {'conflict'}, 'own-action-recall': {'recall'}}


def _rewrite(directory, change):
    scenarios = json.loads((directory / 'scenarios.json').read_text())
    change(scenarios)
    raw = (json.dumps(scenarios, indent=1, sort_keys=True) + '\n').encode()
    manifest = json.loads((directory / 'manifest.json').read_text())
    manifest['files']['scenarios.json'] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    (directory / 'scenarios.json').write_bytes(raw)
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')


def _gold(scenarios, scenario):
    return next(item for item in scenarios if item['scenario'] == scenario)['oracle']['artifacts'][0]['gold']


@pytest.mark.parametrize('name, scenario, change', [
    ('memory_hard.py', 'buried-fact', lambda gold: gold['evidence'][0].update(session='nowhere-99')),
    ('memory_hard.py', 'buried-fact', lambda gold: gold['evidence'][0].update(message=999)),
    ('memory_hard.py', 'buried-fact', lambda gold: gold.update(kind='guess')),
    ('memory_hard.py', 'buried-fact', lambda gold: gold.update(evidence=[])),
    ('memory_hard.py', 'buried-fact', lambda gold: gold['evidence'].append(dict(gold['evidence'][0]))),
    ('memory_hard.py', 'buried-correction', lambda gold: gold.update(stale=['not forbidden'])),
    ('memory_hard.py', 'buried-correction', lambda gold: gold.update(stale=[])),
    ('memory_hard.py', 'buried-fact', lambda gold: gold.update(extra=1)),
    ('memory_hard.py', 'channel-switch', lambda gold: gold.pop('stale_fields')),
    ('memory_hard.py', 'channel-switch', lambda gold: gold['stale_fields'].update(
        {value: 'colour' for value in gold['stale']})),
    ('memory_hard.py', 'buried-correction', lambda gold: gold.update(stale_fields={'not stale': 'place'})),
    ('memory.py', 'contradiction-ask', lambda gold: gold.update(sides=['09:15'])),
    ('memory.py', 'contradiction-ask', lambda gold: gold.pop('sides')),
    ('memory.py', 'knowledge-update', lambda gold: gold['evidence'][0].update(turn=2)),  # a night entry
    ('memory.py', 'knowledge-update', lambda gold: gold['evidence'][0].update(role='system')),
])
def test_the_loader_refuses_gold_that_points_nowhere(generate, name, scenario, change, tmp_path):
    module = generate.load_templates(GENERATORS / name)
    generate.write(tmp_path / 'dev', module, 7, 'dev', 1, GENERATORS / name)
    paired_cases.load_generated_dataset(tmp_path / 'dev')
    _rewrite(tmp_path / 'dev', lambda scenarios: change(_gold(scenarios, scenario)))
    with pytest.raises(ValueError, match='gold'):
        paired_cases.load_generated_dataset(tmp_path / 'dev')

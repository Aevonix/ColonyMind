"""The harder memory family (mind-memory-2): facts buried in a long seeded history, among near-duplicates.

Rendered and graded offline; nothing here calls a model. What each template makes hard is asserted from the
rendered history itself, so a template edit that quietly makes a type easy (the fact restated late, the
stale value gone, the assistant's restatement dropped) fails here, not in a pilot.
"""
import importlib.util
import json
from pathlib import Path
import re

import pytest

from colonymind.qualification import paired_cases, paired_history
from colonymind.qualification.paired_cases import cases as real_cases  # bound before the runner fixture stubs it
from colonymind.qualification.paired_workflow_runtime import validate_workflow

GENERATORS = Path(__file__).resolve().parents[2] / 'benchmarks' / 'paired' / 'generators'
MODULE = GENERATORS / 'memory_hard.py'
CONTACT = re.compile(r'p-\d\d')
NIGHT = [{'advance_clock': 86400}, {'tick': 1}]
# Dev split, per-template 3, seed 7: the manifest hashes the template and engine sources, so any edit to
# memory_hard.py or generate.py is a new dataset. Update deliberately, with the generators README and the plan.
PINNED_DEV_SPLIT = '013e0a010481b02b59f2117b53f715895c3395fd5641d8f883813c2d77f53ef0'
PINNED_DEV_SCENARIOS = 'd57b9adde9031380477b4f069e68fc3139955a95b00cda7a99eea0cb20439ba7'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def generate():
    return load(GENERATORS / 'generate.py', 'paired_generate_memory_hard')


@pytest.fixture(scope='module')
def module(generate):
    return generate.load_templates(MODULE)


def render(generate, module, seed=11, per_template=4):
    return generate.render(module, seed, per_template)


def owner_lines(item):
    return [(s['id'], m['content']) for s in item['history'] for m in s['messages'] if m['role'] == 'user']


def assistant_lines(item):
    return [(s['id'], m['content']) for s in item['history'] for m in s['messages'] if m['role'] == 'assistant']


def index_of(item, session_id):
    return [s['id'] for s in item['history']].index(session_id)


def expectations(item):
    artifact = item['oracle']['artifacts'][0]
    expect = {rule['path'][0]: rule['value'] for rule in artifact['assertions'] if rule['path']}
    return expect, artifact['forbidden']


def said(lines, *words):
    """The (session, text) lines that carry every word."""
    return [(session, text) for session, text in lines if all(word in text for word in words)]


def test_family_shape(generate, module):
    scenarios = render(generate, module, per_template=3)
    assert module.FAMILY == 'mind-memory-2' and len(module.TEMPLATES) == 9
    groups = {name: group for name, (group, _) in module.TEMPLATES.items()}
    assert sorted(name for name, group in groups.items() if group == 'abstain') == ['near-miss', 'retracted']
    assert len(scenarios) == 27 and sum(item['family'] == 'abstain' for item in scenarios) == 6
    for item in scenarios:
        text = json.dumps(item)
        assert not re.search(r'p-\d(?!\d)', text), 'every contact id is fixed width'
        assert not re.search(r'[^\x00-\x7f]', text), 'plain ASCII: no dashes or quotes a template did not mean'
        contacts = json.loads(item['initial_files']['contacts.json'])
        assert set(CONTACT.findall(json.dumps([item['history'], item['episodes']]))) <= set(contacts)
        # The live episode: a neutral statement, the night, a restart and the probe in a new session.
        episodes = item['episodes']
        assert episodes[1:3] == NIGHT and [e.get('session_id') for e in episodes] == ['owner-1', None, None, 'owner-2']
        assert episodes[0]['user'] in module.SETUP and '?' not in episodes[0]['user']
        assert validate_workflow(item['workflow'], episodes)['restart_before'] == [3]
        probe = episodes[-1]['user']
        assert 'answer.json' in probe and 'as exactly {' in probe and '}}' not in probe and '{thing}' not in probe
        assert not re.search(r'\b(today|yesterday)\b', text, re.IGNORECASE)
        # The long history: 28 to 40 sessions in time order, named by channel, weeks before the probe.
        history = paired_history.validate_history(item['history'])
        assert module.SESSIONS[0] <= len(history) <= module.SESSIONS[1]
        assert [s['at'] for s in history] == sorted(s['at'] for s in history)
        assert all(re.fullmatch(r'(chat|email|sms)-\d\d', s['id']) for s in history)
        assert all(len(s['messages']) >= 4 and len(s['messages']) % 2 == 0 for s in history)
        assert all([m['role'] for m in s['messages']] == ['user', 'assistant'] * (len(s['messages']) // 2)
                   for s in history)
        # No forbidden value can match a right answer by accident.
        expect, forbidden = expectations(item)
        labels = [label for value in expect.values() if isinstance(value, list) for label in value]
        assert all(bad.casefold() not in label.casefold() and label.casefold() not in bad.casefold()
                   for bad in forbidden for label in labels)
        assert all(bad.casefold() not in f'{{"{key}": "unknown"}}' for bad in forbidden for key in expect)


def test_a_probe_never_tells_a_recall_type_from_an_abstention(generate, module):
    """The place probes are word for word the same for a buried fact, a near miss and a retraction, and every
    probe that asks for a name offers "unknown"."""
    scenarios = render(generate, module)
    for item in scenarios:
        probe = item['episodes'][-1]['user']
        expect, _ = expectations(item)
        if any(isinstance(value, list) for value in expect.values()) and 'duration' not in expect:
            assert '"unknown"' in probe, item['id']
        if item['scenario'] in ('buried-fact', 'near-miss', 'retracted'):
            thing = re.search(r'keep the (.+?)[?,]', probe).group(1)
            assert probe in {template.replace('{thing}', thing) for template in module.PLACE_PROBES}, item['id']


def test_every_template_buries_its_fact_among_near_duplicates(generate, module):
    scenarios = render(generate, module, seed=5)
    seen = set()
    for item in scenarios:
        name, expect, forbidden = item['scenario'], *expectations(item)
        owner, assistant, count = owner_lines(item), assistant_lines(item), len(item['history'])
        seen.add(name)
        if name == 'buried-fact':
            place = expect['place'][0]
            thing = re.search(r'keep the (.+?)[?,]', item['episodes'][-1]['user']).group(1)
            [(session, _)] = said(owner, thing, place)
            assert index_of(item, session) < module.EARLY and count - index_of(item, session) > 20
            assert len(said(owner, thing)) >= 3, 'the item comes up again without its place'
            assert not said(owner + assistant, thing, 'borrowed') and not said(owner, 'Lent the ' + thing)
        elif name in ('buried-correction', 'restated-stale'):
            [new], [old] = expect['place'], forbidden
            item_, contact = re.search(r'monthly (.+?) review with (p-\d\d)', item['episodes'][-1]['user']).groups()
            [(change, _)] = said(owner, item_, contact, new)
            (first, _), *rest = said(owner, item_, contact, old)
            assert all(session == change for session, _ in rest), 'the old venue again only in the change'
            assert index_of(item, first) < module.EARLY and index_of(item, change) < count - module.TAIL
            others = said(owner, item_ + ' review') + said(owner, contact)
            assert len({text for _, text in others}) >= 3, 'the review and the contact come up elsewhere'
            if name == 'buried-correction':
                assert index_of(item, change) - index_of(item, first) >= module.MIN_GAP
                assert not said(assistant, item_, contact, old), 'no restatement: the correction is only far away'
            else:
                restated = said(assistant, item_, contact, old)
                assert len(restated) == 2 and all(index_of(item, s) < index_of(item, change) for s, _ in restated)
                assert not said(assistant, item_, contact, new), 'the new venue is said once, by the owner'
        elif name == 'channel-switch':
            before, old_place = forbidden
            holder, new_place = expect['contact'][0], expect['place'][0]
            thing = next(g for g in re.search(r'has my (.+?) at the moment|for my (.+?),',
                                              item['episodes'][-1]['user']).groups() if g)
            [(lent, _)] = said(owner, thing, before, old_place)
            [(moved, text)] = said(owner, thing, holder, new_place)
            assert lent.startswith('email-') and moved.startswith('sms-') and text == text.lower()
            assert said(owner, thing, before, 'back') and index_of(item, lent) < index_of(item, moved)
            assert not set(re.findall(r'\w+', text)) & {'who', 'where', 'moment', 'laptop', 'device'}
        elif name == 'assistant-misquote':
            [right], [wrong] = expect['place'], forbidden
            assert (right, wrong) in module.MISQUOTES
            thing = re.search(r'(?:said|say) the (.+?) (?:should go|goes)', item['episodes'][-1]['user']).group(1)
            [(stated, _)] = said(owner, thing, right)
            assert not said(owner, wrong), 'the owner never says the restated place'
            assert not said(assistant, thing, right), 'the assistant never says the right one for the item'
            restatements = said(assistant, thing, wrong)
            assert len(restatements) == 3 and restatements[0][0] == stated
            assert all(index_of(item, s) > index_of(item, stated) for s, _ in restatements[1:])
        elif name == 'buried-preference':
            [(session, _)] = said(owner, 'minutes')
            assert index_of(item, session) < module.EARLY and expect['unit'] == module.MINUTE_LABELS
            durations = said(owner, 'hours') + said(owner, 'booked')
            assert len({text for _, text in durations}) >= 4, 'other sessions state other durations'
            assert expect['duration'] in [60 * int(h) for _, text in durations for h in re.findall(r'(\d) hours', text)]
        elif name == 'spread-total':
            item_, contact = re.search(r'boxes of the (.+?) files (?:has )?(p-\d\d)?', item['episodes'][-1]['user']).groups()
            contact = contact or re.search(r'(p-\d\d)', item['episodes'][-1]['user']).group(1)
            counts = [int(re.search(r'(\d+)(?: more)? boxes', text).group(1))
                      for _, text in said(owner, contact, item_ + ' files')]
            assert len(counts) == 3 and expect['boxes'] == sum(counts)
            assert len({s for s, _ in said(owner, contact, item_ + ' files')}) == 3, 'three sessions'
            assert len(said(owner, item_ + ' files')) > 3, 'another contact drops off the same files'
            assert len(said(owner, contact, 'boxes')) > 3, 'the contact drops off other files too'
        elif name == 'near-miss':
            assert expect == {'place': ['unknown']} and len(forbidden) == 4
            thing = re.search(r'keep the (.+?)[?,]', item['episodes'][-1]['user']).group(1)
            *places, venue = forbidden
            assert set(places) <= set(module.PLACES) and venue in module.VENUES and said(owner, thing, venue)
            mentions = said(owner, thing)
            assert len(mentions) >= 3 and not said(owner + assistant, thing, ' kept in ')
            assert not said(owner + assistant, thing, ' put the ')
            sessions = {s for s, _ in mentions}
            assert all(any(s in sessions for s, _ in said(owner, place)) for place in places)
        elif name == 'retracted':
            [old] = forbidden
            thing = re.search(r'keep the (.+?)[?,]', item['episodes'][-1]['user']).group(1)
            [(stated, _)] = said(owner, thing, old)
            withdrawn = [s for s, text in said(owner, thing) if 'no longer' in text or 'Scratch' in text]
            assert len(withdrawn) == 1 and index_of(item, withdrawn[0]) > index_of(item, stated)
            assert expect == {'place': ['unknown']}
        else:
            raise AssertionError(name)
    assert seen == set(module.TEMPLATES)


def test_the_filler_never_contradicts_itself(generate, module):
    """Every thing has at most one place and is never both kept somewhere and lent out; every review one venue."""
    for item in render(generate, module, seed=3):
        places, loans, venues = {}, set(), {}
        for _, text in owner_lines(item):
            kept = re.match(r'(?:The (.+) is kept in the (.+) now\.|I put the (.+) in the (.+); that is where it lives\.)$', text)
            if kept:
                thing, place = (kept.group(1), kept.group(2)) if kept.group(1) else (kept.group(3), kept.group(4))
                assert places.setdefault(thing, place) == place or item['scenario'] == 'retracted', (item['id'], text)
            lent = re.match(r'(?:(p-\d\d) borrowed my (.+) for a while\.|Lent the (.+) to (p-\d\d)\.)$', text)
            if lent:
                loans.add(lent.group(2) or lent.group(3))
            review = re.match(r'(?:The monthly (.+) review with (p-\d\d) is held in the (.+)\.|(p-\d\d) and I hold the '
                              r'monthly (.+) review in the (.+)\.)$', text)
            if review:
                key = (review.group(1), review.group(2)) if review.group(1) else (review.group(5), review.group(4))
                venue = review.group(3) or review.group(6)
                assert venues.setdefault(key, venue) == venue, (item['id'], text)
        assert not loans & set(places), item['id']


def test_same_seed_same_bytes_and_the_loader_builds_restart_cases_with_history(generate, module, tmp_path):
    first = generate.write(tmp_path / 'a', module, 7, 'dev', 1, MODULE)
    assert generate.write(tmp_path / 'b', module, 7, 'dev', 1, MODULE) == first
    assert generate.write(tmp_path / 'c', module, 8, 'dev', 1, MODULE) != first
    manifest, scenarios, verified = paired_cases.load_generated_dataset(tmp_path / 'a')
    assert verified == first and manifest['dataset_id'] == 'mind-memory-2'
    assert manifest['families'] == {'recall': 7, 'abstain': 2}
    cases = {case.id: case for case in paired_cases.cases('base_hermes', dataset_dir=tmp_path / 'a')}
    case = cases['assistant-misquote.01']
    contract = {'restart_before': [3], 'snapshot_after': [], 'read_failures': []}
    assert case.inputs['workflow'] == contract and case.oracle['workflow_contract'] == contract
    assert case.inputs['history'] == next(s for s in scenarios if s['id'] == case.id)['history']
    # The history import, the embedding drain and the night need more than the default 600 s.
    assert case.timeout_seconds == 900 and case.inputs['family_deadline'] == 900
    assert case.record()['timeout_seconds'] == 900
    [right] = case.oracle['artifacts'][0]['assertions'][1]['value']
    checks = paired_cases.assess({'effects': {'turns_completed': 2, 'declared_turns': 4, 'artifacts': {
        'answer.json': json.dumps({'place': right})}}}, case.oracle)
    assert checks['artifact:answer.json'] is True


@pytest.mark.parametrize('answer, passes', [
    ({'place': 'RIGHT'}, True), ({'place': ' right '}, True), ({'place': 'WRONG'}, False),
    ({'place': 'unknown'}, False), ({'place': 'right', 'note': 'wrong'}, False)])
def test_the_misquote_grader_takes_the_owners_words_only(generate, module, answer, passes):
    item = next(s for s in render(generate, module) if s['scenario'] == 'assistant-misquote')
    [right], [wrong] = item['oracle']['artifacts'][0]['assertions'][1]['value'], item['oracle']['artifacts'][0]['forbidden']
    text = json.dumps(answer).replace('RIGHT', right.upper()).replace('right', right).replace('WRONG', wrong)
    checks = paired_cases.assess({'effects': {'turns_completed': 2, 'declared_turns': 4,
                                              'artifacts': {'answer.json': text}}}, item['oracle'])
    assert checks['artifact:answer.json'] is passes


def test_dev_split_is_pinned(generate, module, tmp_path):
    content = generate.write(tmp_path / 'dev', module, 7, 'dev', 3, MODULE)
    assert content == PINNED_DEV_SPLIT, f'dev split seed 7 changed; a template edit is a new dataset: {content}'
    manifest = json.loads((tmp_path / 'dev' / 'manifest.json').read_text())
    assert manifest['files']['scenarios.json']['sha256'] == PINNED_DEV_SCENARIOS
    assert manifest['families'] == {'recall': 21, 'abstain': 6} and manifest['scenario_count'] == 27


from test_qualification_paired_runner import fixture  # noqa: E402,F401  (pytest fixture)


def test_a_plan_runs_the_family_with_its_arms(generate, module, fixture, tmp_path, monkeypatch):
    """The dev split plans with the family's arms against an image that seeds history, restarts processes and
    applies the release arms; every case carries the history and the family's deadline."""
    from colonymind.qualification import paired, paired_container
    from colonymind.qualification.paired_workflow_runtime import PROTOCOL as workflow_protocol
    monkeypatch.setattr(paired_cases, 'cases', real_cases)
    original = paired_container.configuration

    def configuration(*args, **kwargs):
        supplied, recipe = original(*args, **kwargs)
        recipe['container_payload'] = {**recipe['container_payload'], 'history_protocol': paired_history.PROTOCOL,
                                       'workflow_protocol': workflow_protocol}
        return supplied, recipe
    monkeypatch.setattr(paired_container, 'configuration', configuration)
    generate.write(tmp_path / 'dev', module, 7, 'dev', 1, MODULE)
    arms = ['base_hermes', 'full', 'full-plus-learning']
    manifest = paired.plan(fixture.output, dataset_dir=tmp_path / 'dev', native_binding='candidate',
                           evidence_mode='controlled', arms=arms, reference_arm='base_hermes', **fixture.resources)
    assert manifest['dataset']['version'] == 'mind-memory-2' and list(manifest['comparison']['profiles']) == arms
    assert len(manifest['pairs']) == len(module.TEMPLATES)
    for pair in manifest['pairs']:
        for arm in arms:
            case = pair['arms'][arm]['case']
            assert case['inputs']['history'] and case['inputs']['family_deadline'] == 900
            assert case['timeout_seconds'] == 900 and case['inputs']['workflow']['restart_before'] == [3]

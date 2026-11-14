"""mind-recurring-1: one correction, then unseen related probes after delays and after later learning, with no model.

The family (benchmarks/paired/generators/recurring.py), what the loader validates of its probe metadata
(paired_cases) and the campaign report's recurrence rows (paired_report).
"""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

import pytest

from colonymind.mind.lessons import terms
from colonymind.qualification import paired, paired_cases
from colonymind.qualification.records import MAX_CAMPAIGN_OUTPUT_BYTES, MAX_CAMPAIGN_SECONDS

from test_qualification_paired_campaign import attempt, write_dataset
from test_qualification_body_events import run_worker, stubbed_hermes  # noqa: F401  (pytest fixture)
from test_qualification_paired_runner import fixture  # noqa: F401  (pytest fixture)

REPOSITORY = Path(__file__).resolve().parents[2]
GENERATORS = REPOSITORY / 'benchmarks' / 'paired' / 'generators'
GROUPS = {'procedure': {'dispatch-code', 'crate-count'}, 'retrieval': {'dock-lookup', 'member-discount'},
          'tool-misuse': {'case-file', 'settings-edit'}}
# The dev split, per-template 1 (6 campaigns, 66 probes), seed 7. The manifest hashes the template and engine
# sources, so any edit to recurring.py or generate.py is a new dataset: update this deliberately, together with
# benchmarks/paired/generators/README.md.
PINNED_DEV_SPLIT = {7: '8ec3a143be5507e4ad67545fc94dc5bc11474264a93f54e7e27401f07bc2606a'}
CONTACT = re.compile(r'p-\d\d')
ARMS = ('full-lessons', 'full')
# The runner fixture replaces paired_cases.cases and the evaluators with controlled ones; these plans need the real ones.
CASES, EVALUATORS = paired_cases.cases, dict(paired_cases.EVALUATORS)


def engine():
    spec = importlib.util.spec_from_file_location('paired_generate_recurring', GENERATORS / 'generate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def generate():
    return engine()


@pytest.fixture(scope='module')
def family(generate):
    module = generate.load_templates(GENERATORS / 'recurring.py')
    return module, generate.render(module, 7, 1)


def design_named(module, name):
    return next(design for design in module.DESIGNS if design.name == name)


def replay(generate, module, item):
    """The campaign's plan from the scenario's own seed: the rows the rendering was made from."""
    return module.plan(generate.Draw(item['seed']), design_named(module, item['scenario']))


def sessions(item):
    """Owner turns by session, in order."""
    found = {}
    for entry in item['episodes']:
        if 'user' in entry:
            found.setdefault(entry['session_id'], []).append(entry['user'])
    return found


def probes(item, *kinds):
    return [spec for spec in item['oracle']['artifacts'] if spec['probe']['kind'] in kinds]


def expected_file(spec):
    """The file the oracle wants, built from its assertions."""
    value = {}
    for rule in spec['assertions']:
        if rule['op'] == 'keys_equal':
            continue
        [key] = rule['path']
        value[key] = rule['value'][0] if rule['op'] == 'label_one_of' else rule['value']
    return value


def grade(item, files):
    declared = len(item['episodes'])
    return paired_cases.assess({'effects': {'turns_completed': declared, 'declared_turns': declared,
                                            'artifacts': files}}, item['oracle'])


def solve(name, text, files):
    """What the procedure the correction stated gives for a request, from only what the agent sees: the request's
    words and the seeded files. Written apart from the generator, so an oracle only the draws could know fails here."""
    contacts = json.loads(files['contacts.json'])
    contact = CONTACT.search(text).group()
    if name == 'dispatch-code':
        order = int(re.search(r'\b(\d{4})\b', text).group(1))
        if 'pickup' in text:
            return {'value': 'none'}
        region = re.search(r'[Rr]egion ([A-Z]\d)\b', text).group(1)
        return {'value': f'{region}-{order % 1000:03d}' + ('-X' if 'express' in text.casefold() else '')}
    if name == 'crate-count':
        if 'paperwork' in text:
            return {'value': 0}
        items = int(re.search(r'(\d+) (?:fragile )?items', text).group(1))
        return {'value': math.ceil(items / (6 if 'fragile' in text else 12))}
    if name == 'dock-lookup':
        entry = json.loads(files['docks.json']).get(re.search(r'V-\d{3}', text).group())
        return {'value': 'unlisted' if entry is None else entry.get('temporary', entry['dock'])}
    if name == 'member-discount':
        amount = int(re.search(r'(?<![-\d])(\d{2,3})(?!\d)', text.replace(contact, '')).group(1))
        if 'gift card' in text:
            return {'discount': 'none'}
        rate = {'gold': 15, 'silver': 10}[contacts[contact]['tier']]
        return {'discount': amount * rate // 100 + (20 if amount >= 500 else 0)}
    if name == 'case-file':
        return {'queue': 'none' if 'thanks' in text else 'returns' if 'damaged' in text else 'carriers'}
    if name == 'settings-edit':
        config = json.loads(files[re.search(r'svc-\d\d\.json', text).group()])
        key = next(key for key in ('retries', 'timeout', 'backoff', 'workers') if key in text)
        return config if config['locked'] else {**config, key: int(re.search(key + r'\D*?(\d+)', text).group(1))}
    raise AssertionError(name)


def solved_path(name, text, spec):
    """Where the result goes: named in the turn, or, for a case, the file the procedure names."""
    if name == 'case-file':
        number = re.search(r'(?<!order )(?<!\d)(\d{4})(?!\d)', text).group(1)
        return f'case-{number}.json'
    assert spec['path'] in text
    return spec['path']


def day_of_sessions(item):
    day, found = 1, {}
    for entry in item['episodes']:
        if 'tick' in entry:
            day += 1
        elif 'user' in entry:
            found.setdefault(entry['session_id'], set()).add(day)
    return found


# --- The family ------------------------------------------------------------------------------------------


def test_the_family_is_found_deterministic_and_loads_as_campaigns(generate, family, tmp_path):
    module, scenarios = family
    assert generate.FAMILIES['recurring'] == GENERATORS / 'recurring.py' and module.FAMILY == 'mind-recurring-1'
    assert generate.render(module, 7, 1) == scenarios and generate.render(module, 11, 1) != scenarios
    for group, names in GROUPS.items():
        assert {item['scenario'] for item in scenarios if item['family'] == group} == names
    for seed, expected in PINNED_DEV_SPLIT.items():
        content = generate.write(tmp_path / str(seed), module, seed, 'dev', 1, GENERATORS / 'recurring.py')
        assert content == expected, f'dev split seed {seed} changed; a template edit is a new dataset'
    manifest, loaded, verified = paired_cases.load_generated_dataset(tmp_path / '7')
    assert verified == PINNED_DEV_SPLIT[7] and manifest['dataset_id'] == 'mind-recurring-1'
    assert manifest['families'] == {'procedure': 2, 'retrieval': 2, 'tool-misuse': 2}
    assert all(paired_cases.validate_probes(item, recurring=True) for item in loaded)
    assert sum(len(probes(item, 'warranted', 'control')) for item in loaded) == 66
    for item in loaded:
        assert len(probes(item, 'warranted')) == 9 and len(probes(item, 'training')) == 2
        assert sorted(spec['probe']['control'] for spec in probes(item, 'control')) == ['contact_rule', 'scope']


def test_the_loader_builds_campaign_cases_with_the_deadline_and_the_read_only_skill_tools(generate, tmp_path):
    module = generate.load_templates(GENERATORS / 'recurring.py')
    generate.write(tmp_path / '7', module, 7, 'dev', 1, GENERATORS / 'recurring.py')
    for arm in ARMS:
        cases = paired_cases.cases(arm, dataset_dir=tmp_path / '7', profile={'name': arm, **paired.PROFILES[arm]})
        for case in cases:
            campaign = case.inputs['campaign']
            assert set(campaign) == {'protocol', 'days', 'snapshot_after'} and campaign['snapshot_after']
            assert campaign['protocol'] == paired_cases.CAMPAIGN_PROTOCOL and campaign['days'] == module.DAYS
            assert case.timeout_seconds == 600 + paired_cases.CAMPAIGN_DAY_SECONDS * module.DAYS == 11400
            assert case.timeout_seconds < MAX_CAMPAIGN_SECONDS and case.max_output_bytes == MAX_CAMPAIGN_OUTPUT_BYTES
            assert case.inputs['skill_tools'] == 'read' and case.inputs['tool_loading'] == 'eager'
            assert case.inputs['message_timestamps'] == 'gateway' and case.inputs['environment_note'] == 'messaging'
    assert paired_cases.GENERATED_SKILL_TOOLS['mind-recurring-1'] == 'read'


def test_every_artifact_is_graded_against_an_oracle_the_request_and_the_files_reproduce(generate, family):
    module, scenarios = family
    for item in scenarios:
        laid = replay(generate, module, item)
        by_session, files = sessions(item), item['initial_files']
        owners = {row['path']: row['design'].name for row in laid['rows']}
        right = {}
        for spec in item['oracle']['artifacts']:
            session = spec['probe'].get('session') or next(row['session'] for row in laid['rows']
                                                            if row['path'] == spec['path'])
            text = by_session[session][0]
            name = owners[spec['path']]
            assert solved_path(name, text, spec) == spec['path'], (item['id'], text)
            answer = solve(name, text, files)
            assert answer == expected_file(spec), (item['id'], text, answer)
            if spec['probe']['kind'] == 'training':
                # The correction states the result the file must hold.
                correction = by_session[session][1]
                assert correction.startswith('Verdict on ') and all(
                    json.dumps(value) in correction or str(value) in correction for value in answer.values())
            right[spec['path']] = json.dumps(answer)
        checks = grade(item, right)
        assert all(checks['artifact:' + spec['path']] for spec in item['oracle']['artifacts'])
        # The scope control's right result is the procedure's own abstention.
        [scope] = [spec for spec in probes(item, 'control') if spec['probe']['control'] == 'scope']
        answer = expected_file(scope)
        if item['scenario'] == 'settings-edit':
            assert answer == json.loads(files[scope['path']]) and answer['locked'] is True
        else:
            [value] = answer.values()
            assert value in ('none', 'unlisted', 0)
        # A missing file, a wrong value, or the forbidden token fails exactly that probe.
        for spec in probes(item, 'warranted', 'control'):
            path, expected = spec['path'], expected_file(spec)
            assert grade(item, {**right, path: None})['artifact:' + path] is False
            wrong = {key: ('other' if isinstance(value, str) else value + 1 if not isinstance(value, bool) else not value)
                     for key, value in expected.items()}
            assert grade(item, {**right, path: json.dumps(wrong)})['artifact:' + path] is False
            for forbidden in spec.get('forbidden', []):
                assert forbidden.casefold() not in json.dumps(expected).casefold()
                assert spec['probe'].get('control') == 'contact_rule'
                leaked = json.dumps({**expected, 'note': forbidden.lower()})
                assert grade(item, {**right, path: leaked})['artifact:' + path] is False


def test_probes_are_unseen_instances_and_the_contact_rule_would_change_the_answer(generate, family):
    module, scenarios = family
    for item in scenarios:
        laid = replay(generate, module, item)
        seen = []
        for row in sorted(laid['rows'], key=lambda row: (row['day'], row['session'])):
            token = re.compile(r'(?<![\w-])' + re.escape(row['instance']['token'].strip()) + r'(?!\w)')
            assert not any(token.search(text) for text in seen), (item['id'], token)
            seen.append(row['text'])
        [rule] = [row for row in laid['rows'] if row['case'] == 'contact_rule']
        [inbound] = [entry['inbound'] for entry in item['episodes'] if 'inbound' in entry]
        # Following the contact's rule instead of the owner's gives another answer: a forbidden token, a gold rate
        # on a silver record, twenty to a crate, or a changed timeout.
        name = item['scenario']
        answer = rule['instance']['answer']
        if name in ('dispatch-code', 'dock-lookup', 'case-file'):
            assert rule['instance']['forbidden'] and rule['instance']['forbidden'][0] not in json.dumps(answer)
        elif name == 'member-discount':
            assert laid['contacts'][rule['contact']]['tier'] == 'silver' and 'gold' in inbound['text']
        elif name == 'crate-count':
            items = rule['instance']['slots']['items']
            assert math.ceil(items / 20) != answer
        else:
            assert 'timeout' in inbound['text'] and answer['timeout'] != 99


def test_reworded_probes_share_at_most_one_term_with_the_title_and_when_to_use(generate, family):
    module, scenarios = family
    assert module.PARAPHRASE_SHARED_MAX == 1
    for design in module.DESIGNS:
        anchor = terms(f'{design.title} {design.when}')
        assert len(anchor) >= 3, design.name
        assert len(design.paraphrases) == len(set(design.paraphrases)) == 2 * len(module.DELAYS)
    for item in scenarios:
        design = design_named(module, item['scenario'])
        anchor = terms(f'{design.title} {design.when}')
        by_session = sessions(item)
        failure, correction = by_session['day-01']
        assert f'The procedure for {design.title}, used when {design.when}: {design.rule}' in correction
        assert 2 * len(anchor & terms(failure)) > len(anchor)
        reworded = []
        for spec in probes(item, 'warranted', 'control'):
            [text] = by_session[spec['probe']['session']]
            shared = anchor & terms(text)
            if spec['probe']['relation'] == 'same_wording':
                assert 2 * len(shared) > len(anchor), (text, shared)
            else:
                assert len(shared) <= module.PARAPHRASE_SHARED_MAX, (text, shared)
                assert spec['probe']['kind'] == 'warranted'
                reworded.append(text)
        assert len(reworded) == 2 * len(module.DELAYS)
        # Worded differently from each other: six distinct rewordings, none the failure's.
        laid = replay(generate, module, item)
        wordings = [row['wording'] for row in laid['rows'] if row.get('relation') in ('paraphrase', 'new_instance')]
        assert None not in wordings and len(set(wordings)) == len(wordings) == 6


def test_every_rewording_of_every_design_holds_the_bound_in_both_branches(generate):
    module = generate.load_templates(GENERATORS / 'recurring.py')
    for index, design_class in enumerate(module.DESIGNS):
        design = design_class(generate.Draw(index))
        anchor = terms(f'{design.title} {design.when}')
        for case in ('base', 'other', 'scope', 'contact_rule'):
            contact = f'p-{10 + index:02d}'
            instance = design.instance(case, contact, design.contact_record(contact, case))
            if case in ('base', 'other'):
                for wording in range(len(design.paraphrases)):
                    text = design.turn(instance, wording, design.path(instance, 'task-02'))
                    assert len(anchor & terms(text)) <= module.PARAPHRASE_SHARED_MAX, (design.name, text)
                    assert '?' not in text
            if case != 'other':
                text = design.turn(instance, None, design.path(instance, 'task-02'))
                assert 2 * len(anchor & terms(text)) > len(anchor), (design.name, case, text)


def test_the_delays_and_the_interfering_correction_sit_on_their_declared_days(generate, family):
    module, scenarios = family
    assert module.DELAYS == (3, 7, 14) and module.INTERFERENCE_DAYS == (6, 11)
    for item in scenarios:
        assert paired_cases.campaign_days(item['episodes']) == module.DAYS == 15
        days = day_of_sessions(item)
        [target] = [spec['probe'] for spec in probes(item, 'training') if spec['probe']['correction'] == 'target']
        [other] = [spec['probe'] for spec in probes(item, 'training') if spec['probe']['correction'] == 'interfering']
        assert target['day'] == module.CORRECTION_DAY == 1 and other['day'] in module.INTERFERENCE_DAYS
        by_delay = {}
        for spec in probes(item, 'warranted', 'control'):
            probe = spec['probe']
            assert probe['delay'] in module.DELAYS and probe['day'] == target['day'] + probe['delay']
            assert days[probe['session']] == {probe['day']}
            assert probe['after_interference'] is (other['day'] < probe['day'])
            if probe['kind'] == 'warranted':
                by_delay.setdefault(probe['delay'], []).append(probe['relation'])
            else:
                assert probe['relation'] == 'same_wording'
        assert {delay: sorted(relations) for delay, relations in by_delay.items()} == {
            delay: sorted(module.RELATIONS) for delay in module.DELAYS}
        # Each correction names the owner session that teaches it: the request, then the correction, on its day.
        by_session = sessions(item)
        for spec in probes(item, 'training'):
            probe = spec['probe']
            assert days[probe['session']] == {probe['day']}
            request, correction = by_session[probe['session']]
            assert not request.startswith('Verdict') and correction.startswith('Verdict on ')
            assert spec['path'] in correction
        # Every day ends with a one-day clock advance and one tick; the contact's rule arrives on day 2.
        kinds = [next(key for key in entry if key != 'session_id') for entry in item['episodes']]
        assert kinds.count('advance_clock') == kinds.count('tick') == 15 and kinds.count('inbound') == 1
        for index, kind in enumerate(kinds):
            if kind == 'advance_clock':
                assert item['episodes'][index] == {'advance_clock': 86400} and item['episodes'][index + 1] == {'tick': 1}
        assert kinds[:kinds.index('inbound')].count('tick') == module.INBOUND_DAY - 1
        # One session per probe: no probe sees another in its conversation.
        units = [spec['probe']['session'] for spec in probes(item, 'warranted', 'control')]
        assert len(set(units)) == len(units) and all(len(sessions(item)[session]) == 1 for session in units)
    # Across renders the interference falls before the 7-day probes in some campaigns and after them in others.
    placed = {spec['probe']['day'] for seed in range(1, 6) for item in generate.render(
        generate.load_templates(GENERATORS / 'recurring.py'), seed, 1)
        for spec in probes(item, 'training') if spec['probe']['correction'] == 'interfering'}
    assert placed == set(module.INTERFERENCE_DAYS)


def test_contacts_are_fixed_width_with_records_and_turns_are_statements(generate, family):
    module, scenarios = family
    for item in scenarios:
        text = json.dumps(item)
        assert not re.search(r'p-\d(?!\d)', text), 'every contact id is fixed width'
        contacts = json.loads(item['initial_files']['contacts.json'])
        assert set(CONTACT.findall(json.dumps(item['episodes']))) <= set(contacts)
        assert all(record['address'] == 'capture:' + contact and record['channel'] in module.CHANNELS
                   for contact, record in contacts.items())
        [inbound] = [entry for entry in item['episodes'] if 'inbound' in entry]
        message = inbound['inbound']
        assert inbound['session_id'] == module.CONTACT_SESSION
        assert message['channel'] == contacts[message['contact']]['channel']
        # The contact who asked for another rule is the one the contact-rule probe is about, and nowhere else.
        [rule] = [spec for spec in probes(item, 'control') if spec['probe']['control'] == 'contact_rule']
        by_session = sessions(item)
        assert message['contact'] in by_session[rule['probe']['session']][0]
        assert not any(message['contact'] in turn for session, turns in by_session.items()
                       if session != rule['probe']['session'] for turn in turns)
        for turn in [turn for turns in by_session.values() for turn in turns] + [message['text']]:
            assert '?' not in turn and turn.strip()
            assert not re.search(r'\b(cron|timer|alarm|schedule|terminal|today|yesterday)\b', turn, re.IGNORECASE)
        # The failure's turn and the probes never state the procedure; only the corrections do.
        for session, turns in by_session.items():
            assert 'procedure' not in turns[0].casefold() and not turns[0].startswith('Verdict')
            assert sum('The procedure for' in turn for turn in turns) == (1 if len(turns) == 2 else 0)


# --- The loader's validation of the probe metadata ---------------------------------------------------------


def test_the_loader_refuses_recurrence_metadata_that_does_not_match_the_campaign(family, tmp_path):
    module, scenarios = family

    def refused(name, change):
        broken = copy.deepcopy(scenarios)
        change(broken[0])
        with pytest.raises(ValueError, match='probe'):
            paired_cases.load_generated_dataset(write_dataset(tmp_path / name, broken, dataset_id=module.FAMILY))

    def first(item, *kinds):
        return next(spec['probe'] for spec in probes(item, *kinds))

    def unit_after(item):
        return next(spec['probe'] for spec in probes(item, 'warranted') if spec['probe']['after_interference'])

    paired_cases.load_generated_dataset(write_dataset(tmp_path / 'good', scenarios, dataset_id=module.FAMILY))
    refused('delay-type', lambda item: first(item, 'warranted').update(delay='3'))
    refused('delay-negative', lambda item: first(item, 'warranted').update(delay=-1))
    refused('delay-day', lambda item: first(item, 'warranted').update(delay=first(item, 'warranted')['delay'] + 1))
    refused('relation', lambda item: first(item, 'warranted').update(relation='synonym'))
    refused('interference-type', lambda item: first(item, 'warranted').update(after_interference=1))
    refused('interference-wrong', lambda item: unit_after(item).update(after_interference=False))
    refused('session-day', lambda item: first(item, 'warranted').update(session='day-01'))
    refused('session-missing', lambda item: first(item, 'warranted').update(session='day-99-1'))
    refused('partial', lambda item: first(item, 'control').pop('relation'))
    refused('correction-on-probe', lambda item: first(item, 'warranted').update(correction='target'))
    refused('correction-name', lambda item: first(item, 'training').update(correction='later'))
    refused('two-targets', lambda item: [spec['probe'].update(correction='target') for spec in probes(item, 'training')])
    refused('recurrence-on-training', lambda item: first(item, 'training').update(delay=0))
    refused('interference-first', lambda item: next(
        spec['probe'] for spec in probes(item, 'training') if spec['probe']['correction'] == 'interfering').update(day=1))


def test_the_loader_checks_each_correction_against_the_turns_that_teach_it(family, tmp_path):
    """Labels that agree with each other but not with the episodes are refused: a correction's day is the day its
    owner session actually holds its request and its correction, whatever the metadata says."""
    module, scenarios = family

    def refused(name, change):
        broken = copy.deepcopy(scenarios)
        change(broken[0])
        with pytest.raises(ValueError, match='probe'):
            paired_cases.load_generated_dataset(write_dataset(tmp_path / name, broken, dataset_id=module.FAMILY))

    def correction(item, which):
        return next(spec['probe'] for spec in probes(item, 'training') if spec['probe']['correction'] == which)

    def teaching(item, which):
        """The session that holds the correction's request and verdict, found from the episodes alone."""
        day = correction(item, which)['day']
        return next(session for session, turns in sessions(item).items()
                    if len(turns) == 2 and day_of_sessions(item)[session] == {day})

    def move(item, session, day):
        """Move a session's owner turns to the start of ``day``; the metadata stays as it was."""
        moved = [entry for entry in item['episodes'] if entry.get('session_id') == session]
        kept = [entry for entry in item['episodes'] if entry.get('session_id') != session]
        ticks = [index for index, entry in enumerate(kept) if 'tick' in entry]
        at = 0 if day == 1 else ticks[day - 2] + 1
        item['episodes'] = kept[:at] + moved + kept[at:]

    def relabel(item):
        """The review's case: the interference said to be on day 2 and every probe after it, turns unchanged."""
        correction(item, 'interfering').update(day=2)
        for spec in probes(item, 'warranted', 'control'):
            spec['probe']['after_interference'] = True

    def drop_verdict(item):
        session = teaching(item, 'interfering')
        verdict = [entry for entry in item['episodes'] if entry.get('session_id') == session][1]
        item['episodes'].remove(verdict)
        item['oracle']['declared_turns'] -= 1

    item = scenarios[0]
    assert day_of_sessions(item)[teaching(item, 'interfering')] == {correction(item, 'interfering')['day']} != {2}
    refused('interference-relabelled', relabel)
    refused('interference-turns-moved', lambda item: move(item, teaching(item, 'interfering'), 2))
    refused('target-turns-moved', lambda item: move(item, teaching(item, 'target'), 2))
    refused('verdict-missing', drop_verdict)
    refused('correction-session-missing', lambda item: correction(item, 'interfering').pop('session', None))
    refused('correction-session-inbound', lambda item: correction(item, 'target').update(session=module.CONTACT_SESSION))
    refused('correction-session-a-probe', lambda item: correction(item, 'interfering').update(
        session=next(spec['probe']['session'] for spec in probes(item, 'warranted'))))


def test_a_recurring_family_declares_its_metadata_on_every_campaign_and_no_other_family_does(generate, family,
                                                                                              tmp_path):
    """Whether a campaign is recurring follows from its family, not from what its probes happen to declare: a
    mind-recurring-1 campaign stripped of all its recurrence metadata would take no snapshots, leave the
    recurrence rows and be graded on its last day's files, so it is refused, in one campaign or in all of them.
    Another family's campaign declaring the metadata is refused as well, and mind-improve-1 loads as it did."""
    module, scenarios = family
    stripped_keys = paired_cases.RECURRENCE_KEYS | paired_cases.CORRECTION_KEYS

    def strip(item):
        for spec in item['oracle']['artifacts']:
            for key in stripped_keys:
                spec['probe'].pop(key, None)

    def unprobed(item):
        for spec in item['oracle']['artifacts']:
            del spec['probe']

    def refused(name, change, dataset_id=module.FAMILY, which=slice(0, 1)):
        broken = copy.deepcopy(scenarios)
        for item in broken[which]:
            change(item)
        with pytest.raises(ValueError, match='recurring'):
            paired_cases.load_generated_dataset(write_dataset(tmp_path / name, broken, dataset_id=dataset_id))

    assert module.FAMILY in paired_cases.RECURRING_FAMILIES
    refused('one-stripped', strip)
    refused('all-stripped', strip, which=slice(None))
    refused('none-a-campaign', unprobed, which=slice(None))
    refused('another-family', lambda item: None, dataset_id='mind-other-1')
    improve = generate.load_templates(GENERATORS / 'improve.py')
    assert improve.FAMILY not in paired_cases.RECURRING_FAMILIES
    manifest, loaded, _ = paired_cases.load_generated_dataset(
        write_dataset(tmp_path / 'improve', generate.render(improve, 7, 1)))
    assert manifest['dataset_id'] == improve.FAMILY and all(paired_cases.validate_probes(item, recurring=False) for item in loaded)


def test_a_recurring_delay_is_that_many_days_of_clock(family, tmp_path):
    """A probe's delay counts days of the campaign's clock, not ticks alone: every day ends with one day's clock
    advance and then one tick, and nothing else advances the clock or ticks. The review's case, every advance cut to
    one second, would label a probe 14 seconds after the correction as 14-day retention."""
    module, scenarios = family
    day = {'advance_clock': 86400}

    def refused(name, change):
        broken = copy.deepcopy(scenarios)
        episodes = broken[0]['episodes']
        change(episodes)
        broken[0]['oracle']['declared_turns'] = len(episodes)
        with pytest.raises(ValueError, match='days are days of its clock'):
            paired_cases.load_generated_dataset(write_dataset(tmp_path / name, broken, dataset_id=module.FAMILY))

    def advances(episodes):
        return [entry for entry in episodes if 'advance_clock' in entry]

    def first_turn(episodes, kind):
        """The index of the first owner turn of the first probe of ``kind``'s session."""
        session = next(spec['probe']['session'] for spec in probes(scenarios[0], kind))
        return next(index for index, entry in enumerate(episodes) if entry.get('session_id') == session)

    def drop(episodes, entry):
        episodes.remove(entry)

    assert paired_cases.RECURRING_DAY_END == (day, {'tick': 1})
    assert all(entry == day for entry in advances(scenarios[0]['episodes']))
    refused('all-one-second', lambda episodes: [entry.update(advance_clock=1) for entry in advances(episodes)])
    refused('one-shortened', lambda episodes: advances(episodes)[4].update(advance_clock=3600))
    refused('one-lengthened', lambda episodes: advances(episodes)[4].update(advance_clock=2 * 86400))
    refused('one-missing', lambda episodes: drop(episodes, advances(episodes)[4]))
    refused('last-day-unended', lambda episodes: episodes.__delitem__(slice(-2, None)))
    refused('extra-advance', lambda episodes: episodes.insert(first_turn(episodes, 'warranted'), {'advance_clock': 3600}))
    refused('extra-tick', lambda episodes: episodes.insert(first_turn(episodes, 'control'), {'tick': 1}))
    refused('double-tick', lambda episodes: next(entry for entry in episodes if 'tick' in entry).update(tick=2))
    refused('tick-before-advance', lambda episodes: episodes.__setitem__(
        slice(-2, None), [{'tick': 1}, {'advance_clock': 86400}]))


# --- Each result graded as its own session left it ------------------------------------------------------------


def last_turns(item):
    """``{session: index of its last episode entry}``, from the episodes alone."""
    return {entry['session_id']: index for index, entry in enumerate(item['episodes']) if 'user' in entry}


def settled(record, files):
    """An attempt's effects in which every declared snapshot already held the final files."""
    declared = len(record['inputs']['episodes'])
    return {'turns_completed': declared, 'declared_turns': declared, 'artifacts': dict(files),
            'campaign_snapshots': {str(index): {} for index in record['inputs']['campaign']['snapshot_after']}}


def right_files(item, record):
    return {**item['initial_files'],
            **{spec['path']: json.dumps(expected_file(spec)) for spec in record['oracle']['artifacts']}}


def test_a_recurring_case_snapshots_the_workspace_after_each_result_session(family, records):
    """Every artifact of a recurring campaign is graded after the last turn of the session that asks for it (a
    probe's one turn, a correction's verdict): the case asks the worker for a snapshot there, and the oracle says
    which snapshot grades which file."""
    module, scenarios = family
    for item, record in zip(scenarios, records['full']):
        ends, graded = last_turns(item), record['oracle']['graded_after']
        assert graded == {spec['path']: ends[spec['probe']['session']] for spec in item['oracle']['artifacts']}
        assert record['inputs']['campaign']['snapshot_after'] == sorted(set(graded.values()))
        for spec in item['oracle']['artifacts']:
            entry = item['episodes'][graded[spec['path']]]
            assert entry['session_id'] == spec['probe']['session']
            if spec['probe']['kind'] == 'training':
                assert entry['user'].startswith('Verdict on ')


def test_the_improve_family_still_grades_its_final_workspace(generate, tmp_path):
    """Only a campaign whose results name their sessions is graded on snapshots; mind-improve-1's cases are as
    they were."""
    module = generate.load_templates(GENERATORS / 'improve.py')
    directory = write_dataset(tmp_path / 'improve', generate.render(module, 7, 1))
    for case in paired_cases.cases('full', dataset_dir=directory, profile={'name': 'full', **paired.PROFILES['full']}):
        assert set(case.inputs['campaign']) == {'protocol', 'days'} and 'graded_after' not in case.oracle


def test_each_result_is_graded_on_the_workspace_its_own_session_left(family, records):
    """The review's case: right files that appear only at the end earn nothing at their delay. A file repaired on a
    later day fails its probe, a file spoiled on a later day keeps its pass, and a snapshot the worker never took
    grades as a missing file."""
    module, scenarios = family
    wrong = json.dumps({'value': 'wrong'})
    for item, record in zip(scenarios, records['full']):
        oracle, right = record['oracle'], right_files(item, record)
        paths = [spec['path'] for spec in oracle['artifacts']]
        declared = len(item['episodes'])
        final_only = paired_cases.assess({'effects': {'turns_completed': declared, 'declared_turns': declared,
                                                      'artifacts': right}}, oracle)
        assert not any(final_only['artifact:' + path] for path in paths)
        assert all(paired_cases.assess({'effects': settled(record, right)}, oracle)['artifact:' + path]
                   for path in paths)
        early = next(spec for spec in probes(item, 'warranted') if spec['probe']['delay'] == 3)['path']
        late = next(spec for spec in probes(item, 'warranted') if spec['probe']['delay'] == 14)['path']
        at = str(oracle['graded_after'][early])
        for then in (None, wrong):
            repaired = settled(record, right)
            repaired['campaign_snapshots'][at][early] = then
            checks = paired_cases.assess({'effects': repaired}, oracle)
            assert [path for path in paths if not checks['artifact:' + path]] == [early]
        spoiled = settled(record, {**right, early: wrong})
        for index in spoiled['campaign_snapshots']:
            spoiled['campaign_snapshots'][index][early] = right[early]
        checks = paired_cases.assess({'effects': spoiled}, oracle)
        assert all(checks['artifact:' + path] for path in paths)
        missing = settled(record, right)
        del missing['campaign_snapshots'][str(oracle['graded_after'][late])]
        checks = paired_cases.assess({'effects': missing}, oracle)
        assert [path for path in paths if not checks['artifact:' + path]] == [
            path for path in paths if oracle['graded_after'][path] == oracle['graded_after'][late]]
        broken = {**settled(record, right), 'campaign_snapshots': ['not', 'a', 'map']}
        assert not any(paired_cases.assess({'effects': broken}, oracle)['artifact:' + path] for path in paths)


def test_forbidden_hits_count_the_file_as_it_was_graded(family, records):
    """The report's forbidden hits read the same file the checks graded: a tempting value written after the probe
    is not a hit at its delay, and one written at the probe and removed later still is."""
    from colonymind.qualification import paired_report
    module, scenarios = family
    tempted = 0

    def rows(index, arm, record):
        nonlocal tempted
        item = scenarios[index]
        right = right_files(item, record)
        effects = settled(record, right)
        for spec in record['oracle']['artifacts']:
            if not spec.get('forbidden'):
                continue
            tempting = json.dumps({'value': spec['forbidden'][0]})
            if arm == 'full':
                effects['campaign_snapshots'][str(record['oracle']['graded_after'][spec['path']])][spec['path']] = tempting
                tempted += 1
            else:
                effects['artifacts'][spec['path']] = tempting
                for snapshot in effects['campaign_snapshots'].values():
                    snapshot[spec['path']] = right[spec['path']]
        row = attempt(record)
        row['effects'].update(effects)
        return row
    manifest, pairs = cohort(records, rows)
    report = paired_report._campaign(manifest, pairs, list(ARMS), 'full-lessons')
    assert tempted > 0 and report['forbidden_hits'] == {'full-lessons': 0, 'full': tempted}


def test_the_worker_keeps_each_declared_snapshot_as_its_differences_from_the_final_workspace(
        stubbed_hermes, monkeypatch, capsys):  # noqa: F811
    """The real worker loop: the workspace after each declared entry, stored as what differs from the final
    workspace (null for a file not there yet), so the result stays small when later days leave earlier files be."""
    import sys
    from types import SimpleNamespace
    workspace = stubbed_hermes.root / 'workspace'
    edits = {'first': {'a.json': '1'}, 'second': {'b.json': 'x'}, 'third': {'a.json': '2', 'c.json': 'c'},
             'fourth': {'b.json': None}}

    class Agent:
        def __init__(self, **kwargs):
            self.session_id = kwargs['session_id']

        def run_conversation(self, user, *, system_message, conversation_history):
            for path, text in edits[self.session_id].items():
                (workspace / path).unlink() if text is None else (workspace / path).write_text(text)
            return {'completed': True, 'messages': [], 'final_response': 'Done.'}

        def close(self):
            pass
    monkeypatch.setitem(sys.modules, 'run_agent', SimpleNamespace(AIAgent=Agent))
    episodes = [{'session_id': 'first', 'user': 'Write a.'}, {'session_id': 'second', 'user': 'Write b.'},
                {'tick': 1}, {'session_id': 'third', 'user': 'Rewrite a.'}, {'session_id': 'fourth', 'user': 'Drop b.'}]
    request = {'binding': 'candidate', 'config': {'model': {'default': 'test'}},
               'inputs': {'arm': 'base_hermes', 'initial_files': {'seed.json': '{}'}, 'max_iterations': 4,
                          'max_output_tokens': 64, 'settle_seconds': 0, 'worker_wait_seconds': 1,
                          'episodes': episodes, 'campaign': {'protocol': paired_cases.CAMPAIGN_PROTOCOL, 'days': 1,
                                                             'snapshot_after': [0, 1, 3]}}}
    code, result = run_worker(monkeypatch, capsys, request)
    assert code == 0, result.get('private_error_traceback')
    effects = result['tool_evidence']
    assert effects['artifacts'] == {'seed.json': '{}', 'a.json': '2', 'c.json': 'c'}
    assert effects['campaign_snapshots'] == {'0': {'a.json': '1', 'c.json': None},
                                             '1': {'a.json': '1', 'b.json': 'x', 'c.json': None},
                                             '3': {'b.json': 'x'}}


@pytest.mark.parametrize('declared', [[4], [1, 1], [2, 1], [-1], ['0'], 'all'])
def test_the_worker_refuses_malformed_snapshot_turns_before_any_turn(stubbed_hermes, monkeypatch, capsys,  # noqa: F811
                                                                    declared):
    request = {'binding': 'candidate', 'config': {'model': {'default': 'test'}},
               'inputs': {'arm': 'base_hermes', 'initial_files': {}, 'max_iterations': 4, 'max_output_tokens': 64,
                          'settle_seconds': 0, 'episodes': [{'session_id': 'first', 'user': 'Write a.'}, {'tick': 1},
                                                            {'session_id': 'second', 'user': 'Write b.'}, {'tick': 1}],
                          'campaign': {'protocol': paired_cases.CAMPAIGN_PROTOCOL, 'days': 2,
                                       'snapshot_after': declared}}}
    with pytest.raises(ValueError):
        run_worker(monkeypatch, capsys, request)
    assert stubbed_hermes.calls == []


def test_without_declared_snapshots_the_worker_records_none(stubbed_hermes, monkeypatch, capsys):  # noqa: F811
    request = {'binding': 'candidate', 'config': {'model': {'default': 'test'}},
               'inputs': {'arm': 'base_hermes', 'initial_files': {}, 'max_iterations': 4, 'max_output_tokens': 64,
                          'settle_seconds': 0, 'episodes': [{'session_id': 'first', 'user': 'Write a.'}, {'tick': 1}],
                          'campaign': {'protocol': paired_cases.CAMPAIGN_PROTOCOL, 'days': 1}}}
    code, result = run_worker(monkeypatch, capsys, request)
    assert code == 0 and 'campaign_snapshots' not in result['tool_evidence']


def with_snapshot_worker(monkeypatch, supported=True):
    """The runner fixture's image, declaring (or not) the worker's campaign snapshots."""
    from colonymind.qualification import paired_container, paired_worker
    configured = paired_container.configuration

    def configuration(*args, **kwargs):
        supplied, recipe = configured(*args, **kwargs)
        payload = dict(recipe['container_payload'])
        if supported:
            payload['campaign_snapshots'] = paired_worker.CAMPAIGN_SNAPSHOTS_PROTOCOL
        return supplied, {**recipe, 'container_payload': payload}
    monkeypatch.setattr(paired_container, 'configuration', configuration)


def test_a_plan_refuses_an_image_whose_worker_takes_no_campaign_snapshots(fixture, monkeypatch, family,  # noqa: F811
                                                                         tmp_path):
    module, scenarios = family
    monkeypatch.setattr(paired_cases, 'cases', CASES)
    monkeypatch.setattr(paired_cases, 'EVALUATORS', EVALUATORS)
    directory = write_dataset(tmp_path / 'data', scenarios, dataset_id=module.FAMILY)
    fixture.output.mkdir(mode=0o700)
    with_snapshot_worker(monkeypatch, supported=False)
    with pytest.raises(ValueError, match='snapshot'):
        paired.plan(fixture.output / 'refused', native_binding='candidate', evidence_mode='controlled',
                    dataset_dir=directory, arms=list(ARMS), reference_arm='full-lessons',
                    case_ids=[scenarios[0]['id']], **fixture.resources)


# --- The campaign report's recurrence rows ------------------------------------------------------------------


@pytest.fixture(scope='module')
def records(family, tmp_path_factory):
    """Each arm's case records for the six dev campaigns, as a frozen plan holds them."""
    module, scenarios = family
    directory = write_dataset(tmp_path_factory.mktemp('recurring') / 'campaigns', scenarios, dataset_id=module.FAMILY)
    return {arm: [case.record() for case in paired_cases.cases(arm, dataset_dir=directory,
                                                                  profile={'name': arm, **paired.PROFILES[arm]})]
            for arm in ARMS}


def cohort(records, rows, arms=ARMS):
    """A frozen manifest and summarize()-shaped pairs; ``rows(index, arm, record)`` gives each attempt."""
    from colonymind.qualification import paired_report
    manifest = {'pairs': [], 'sha256': '1' * 64,
                'comparison': {'campaign': {'protocol': paired_cases.CAMPAIGN_PROTOCOL, **paired.CAMPAIGN},
                               'rule': {**paired.RULE, 'unit': 'probe', 'cluster': 'campaign'}}}
    pairs = []
    for index in range(len(records[arms[0]])):
        results = {arm: rows(index, arm, records[arm][index]) for arm in arms}
        manifest['pairs'].append({'arms': {arm: {'case': records[arm][index]} for arm in arms}})
        pairs.append({'scenario_id': records[arms[0]][index]['id'], 'results': results,
                      'completion': {arm: paired_report._completion(results[arm]) for arm in arms}})
    return manifest, pairs


def passing(record, rule):
    """The artifact paths whose probe metadata ``rule`` accepts."""
    return {spec['path'] for spec in record['oracle']['artifacts'] if rule(spec['probe'])}


def treatment(probe):
    """Known pattern: same wording always, a paraphrase only 3 days on, a new instance only before the interference."""
    if probe['kind'] != 'warranted':
        return False
    return (probe['relation'] == 'same_wording' or (probe['relation'] == 'paraphrase' and probe['delay'] == 3)
            or (probe['relation'] == 'new_instance' and not probe['after_interference']))


def comparator(probe):
    return probe['kind'] == 'warranted' and probe['relation'] == 'same_wording' and probe['delay'] == 3


def test_the_report_gives_pass_rates_by_delay_relation_and_interference_clustered_by_campaign(records):
    from colonymind.qualification import paired_report
    manifest, pairs = cohort(records, lambda index, arm, record: attempt(
        record, passing(record, treatment if arm == 'full' else comparator)))
    report = paired_report._campaign(manifest, pairs, list(ARMS), 'full-lessons')
    recurrence = report['recurrence']
    assert recurrence['axes'] == ['delay', 'relation', 'interference', 'delay_interference']
    assert 'bootstrap' in recurrence['basis']
    full, reference = recurrence['arms']['full'], recurrence['arms']['full-lessons']
    assert list(full['delay']) == ['3', '7', '14'] and list(full['relation']) == list(paired_cases.PROBE_RELATIONS)
    # Delay 3: every warranted probe passes in full (no campaign has its interference before day 4); one in three
    # passes in the comparator. The same share in every campaign: the campaign interval is a single point.
    three = full['delay']['3']
    assert (three['probes'], three['campaigns'], three['pass_rate']) == (18, 6, 1.0)
    assert (three['ci']['lower'], three['ci']['upper'], three['ci']['clusters']) == (1.0, 1.0, 6)
    assert reference['delay']['3']['pass_rate'] == pytest.approx(1 / 3)
    assert full['relation']['same_wording']['pass_rate'] == 1.0
    assert full['relation']['paraphrase']['pass_rate'] == pytest.approx(1 / 3)
    # Interference: the known counts, recomputed from the campaigns' own metadata.
    units = [spec['probe'] for record in records['full'] for spec in record['oracle']['artifacts']
             if spec['probe']['kind'] == 'warranted']
    for side, after in (('before', False), ('after', True)):
        chosen = [probe for probe in units if probe['after_interference'] is after]
        row = full['interference'][side]
        assert row['probes'] == len(chosen) and row['pass_rate'] == pytest.approx(
            sum(map(treatment, chosen)) / len(chosen))
    assert full['relation']['new_instance']['pass_rate'] == pytest.approx(
        sum(not probe['after_interference'] for probe in units if probe['relation'] == 'new_instance') / 18)
    # The contrast rows: treatment minus comparator per group, clustered by campaign.
    [contrast] = recurrence['contrasts']
    assert (contrast['treatment'], contrast['comparator'], contrast['unavailable_campaigns']) == ('full', 'full-lessons', 0)
    row = contrast['delay']['3']
    assert row['delta_pp'] == pytest.approx(200 / 3) and row['wins'] == 12 and row['losses'] == 0
    assert row['ci_pp']['lower'] == pytest.approx(200 / 3) and row['ci_pp']['clusters'] == 6
    assert contrast['relation']['same_wording']['delta_pp'] == pytest.approx(100 * 2 / 3)
    # The markdown carries the rows.
    lines = '\n'.join(paired_report._campaign_lines({'campaign': report}))
    assert '## Recurrence' in lines and '| full | delay | 3 | 18 (6) | 1.000 | 1.000 to 1.000 |' in lines


def test_the_interference_rows_are_also_split_within_each_delay(records):
    """Every 3-day probe comes before the interfering correction and every 14-day probe after it, so the plain
    before and after rows also differ in delay. The crossed rows compare like with like: only the 7-day probes fall
    on both sides (day 6 or day 11), and their before and after rows separate later learning from time."""
    from colonymind.qualification import paired_report
    manifest, pairs = cohort(records, lambda index, arm, record: attempt(
        record, passing(record, treatment if arm == 'full' else comparator)))
    recurrence = paired_report._campaign(manifest, pairs, list(ARMS), 'full-lessons')['recurrence']
    crossed = recurrence['arms']['full']['delay_interference']
    # The dev split: three campaigns interfere on day 6, three on day 11.
    assert list(crossed) == ['3:before', '7:before', '7:after', '14:after']
    assert {group: (row['probes'], row['campaigns']) for group, row in crossed.items()} == {
        '3:before': (18, 6), '7:before': (9, 3), '7:after': (9, 3), '14:after': (18, 6)}
    # Known pattern: at 7 days the same wording passes on both sides, the paraphrase on neither, the new instance
    # only before the interference.
    assert crossed['7:before']['pass_rate'] == pytest.approx(2 / 3)
    assert crossed['7:after']['pass_rate'] == pytest.approx(1 / 3)
    assert crossed['3:before']['pass_rate'] == 1.0 and crossed['14:after']['pass_rate'] == pytest.approx(1 / 3)
    # The plain rows mix delays: their gap is wider than the like-for-like one at 7 days.
    plain = recurrence['arms']['full']['interference']
    assert plain['before']['pass_rate'] - plain['after']['pass_rate'] > (
        crossed['7:before']['pass_rate'] - crossed['7:after']['pass_rate'])
    [contrast] = recurrence['contrasts']
    assert contrast['delay_interference']['7:before']['delta_pp'] == pytest.approx(200 / 3)
    assert contrast['delay_interference']['7:after']['delta_pp'] == pytest.approx(100 / 3)
    assert (contrast['delay_interference']['7:after']['wins'], contrast['delay_interference']['7:after']['losses']) == (3, 0)
    lines = '\n'.join(paired_report._campaign_lines({'campaign': paired_report._campaign(
        manifest, pairs, list(ARMS), 'full-lessons')}))
    assert '| full | delay_interference | 7:after | 9 (3) | 0.333 |' in lines


def test_the_interval_resamples_whole_campaigns(records):
    """Campaigns that pass everything beside campaigns that pass nothing: the probes of one campaign move together,
    so the interval is wide, where a bootstrap over single probes of 54 would be narrow."""
    from colonymind.qualification import paired_report
    manifest, pairs = cohort(records, lambda index, arm, record: attempt(
        record, passing(record, lambda probe: probe['kind'] == 'warranted' and index % 2 == 0)))
    row = paired_report._campaign(manifest, pairs, list(ARMS), 'full-lessons')['recurrence']['arms']['full']['delay']['7']
    assert row['pass_rate'] == 0.5 and row['campaigns'] == 6
    assert row['ci']['lower'] <= 0.2 and row['ci']['upper'] >= 0.8


def test_an_unavailable_campaign_leaves_the_rows_of_the_other_campaigns(records):
    from colonymind.qualification import paired_report

    def rows(index, arm, record):
        if (index, arm) == (0, 'full'):
            return attempt(record, outcome='error')
        return attempt(record, passing(record, treatment))
    manifest, pairs = cohort(records, rows)
    recurrence = paired_report._campaign(manifest, pairs, list(ARMS), 'full-lessons')['recurrence']
    assert recurrence['arms']['full']['delay']['3']['campaigns'] == 5
    assert recurrence['arms']['full-lessons']['delay']['3']['campaigns'] == 6
    assert recurrence['contrasts'][0]['unavailable_campaigns'] == 1
    assert recurrence['contrasts'][0]['delay']['3']['campaigns'] == 5


def test_a_recurring_campaign_declares_no_old_family_probe_and_shows_no_old_family_row(records):
    from colonymind.qualification import paired_report
    manifest, pairs = cohort(records, lambda index, arm, record: attempt(record, passing(record, treatment)))
    report = paired_report._campaign(manifest, pairs, list(ARMS), 'full-lessons')
    assert report['old_family'] == []
    lines = '\n'.join(paired_report._campaign_lines({'campaign': report}))
    assert '## Campaign probes' in lines and '| none: the family declares no old-family probe |' in lines


def test_lesson_diagnostics_attribute_uses_to_the_probe_session_and_split_by_match_kind(records):
    """A probe's lesson is the one used in its own session: a lesson in the same-wording probe's session does not
    count for the paraphrase probe of the same day. With the match kind recorded on each use (lesson selection by
    meaning), the use rate splits by it."""
    from colonymind.qualification import paired_report

    def rows(index, arm, record):
        warranted = [spec for spec in record['oracle']['artifacts'] if spec['probe']['kind'] == 'warranted']
        same = next(spec for spec in warranted if spec['probe']['relation'] == 'same_wording')
        para = next(spec for spec in warranted if spec['probe']['relation'] == 'paraphrase'
                    and spec['probe']['day'] == same['probe']['day'])
        uses = [{'lesson_id': 'L-1', 'session_id': same['probe']['session'], 'result': 'win',
                 paired_report.LESSON_MATCH_FIELD: 'own_class'},
                {'lesson_id': 'L-1', 'session_id': para['probe']['session'], 'result': None,
                 paired_report.LESSON_MATCH_FIELD: 'meaning'}]
        if arm == 'full-lessons':
            uses = uses[:1]
        lessons = {'lessons': [{'id': 'L-1', 'verified': 'owner', 'status': 'active'}], 'uses': uses}
        return attempt(record, {same['path']}, lessons=lessons)
    manifest, pairs = cohort(records, rows)
    lessons = paired_report._campaign(manifest, pairs, list(ARMS), 'full-lessons')['lessons']
    full, reference = lessons['full'], lessons['full-lessons']
    assert full['probes'] == {'eligible': 66, 'with_lesson': 12, 'with_lesson_passed': 6}
    assert reference['probes'] == {'eligible': 66, 'with_lesson': 6, 'with_lesson_passed': 6}
    assert full['by_match'] == {
        'meaning': {'uses': 6, 'scored_uses': 0, 'wins': 0, 'probes_with_lesson': 6, 'probes_with_lesson_passed': 0},
        'own_class': {'uses': 6, 'scored_uses': 6, 'wins': 6, 'probes_with_lesson': 6, 'probes_with_lesson_passed': 6}}
    assert reference['by_match'] == {
        'own_class': {'uses': 6, 'scored_uses': 6, 'wins': 6, 'probes_with_lesson': 6, 'probes_with_lesson_passed': 6}}
    assert 'by match' in '\n'.join(paired_report._campaign_lines(
        {'campaign': paired_report._campaign(manifest, pairs, list(ARMS), 'full-lessons')}))


def test_the_dev_split_bytes_are_pinned_in_the_readme(generate, tmp_path):
    module = generate.load_templates(GENERATORS / 'recurring.py')
    content = generate.write(tmp_path / '7', module, 7, 'dev', 1, GENERATORS / 'recurring.py')
    scenarios_sha = hashlib.sha256((tmp_path / '7' / 'scenarios.json').read_bytes()).hexdigest()
    readme = (GENERATORS / 'README.md').read_text()
    assert f'| recurring | 1 | 7 | `{content}` | `{scenarios_sha}` |' in readme


# --- The harness end to end, with a scripted agent in place of a model ------------------------------------------


def scripted_agent(module, *, lexical_only):
    """An agent that knows the procedure: it answers every probe session from what it sees (the request and the
    seeded files), or, with ``lexical_only``, only the requests worded like the failure, as a lesson found by its
    words alone would, and then writes the answers it skipped after the last day, as a later repair. It writes
    nothing for the training days. The workspace is snapshotted after the declared turns, as the worker does."""
    from colonymind.qualification.paired_worker import campaign_snapshots

    async def consume(inputs, context):
        context.observe({'role': inputs['role'], 'selected_binding': context.router.binding,
                         'prior_attempts': [], 'outcome': 'returned'})
        design = design_named(module, inputs['scenario'])
        anchor = terms(f'{design.title} {design.when}')
        files = dict(inputs['initial_files'])
        said = {}
        for entry in inputs['episodes']:
            if 'user' in entry:
                said.setdefault(entry['session_id'], []).append(entry['user'])
        declared, taken, skipped = inputs['campaign']['snapshot_after'], {}, []
        for index, entry in enumerate(inputs['episodes']):
            if 'user' in entry and len(said[entry['session_id']]) == 1:
                text = entry['user']
                named = re.search(r'Write the result to (\S+\.json)|(svc-\d\d\.json)', text)
                path = next(group for group in named.groups() if group) if named else solved_path(design.name, text, {})
                answer = json.dumps(solve(design.name, text, files))
                if lexical_only and 2 * len(anchor & terms(text)) <= len(anchor):
                    skipped.append((path, answer))
                else:
                    files[path] = answer
            if index in declared:
                taken[str(index)] = dict(files)
        files.update(skipped)
        count = len(inputs['episodes'])
        return {'output': 'done', 'effects': {
            'declared_turns': count, 'turns_completed': count, 'turns': [{'completed': True}] * count,
            'artifacts': files, 'campaign_snapshots': campaign_snapshots(taken, files),
            'resource_usage': {'coverage': 'complete', 'total_model_calls': 20, 'input_tokens': 2000,
                               'output_tokens': 200, 'background_model_calls': 0}}}
    return consume


def test_one_campaign_per_arm_runs_through_plan_run_and_report(fixture, monkeypatch, family, tmp_path):  # noqa: F811
    """What the wave's dev smoke shows, without a model: the plan freezes the campaign (the probe unit, the campaign
    cluster, the read-only skill tools, the campaign deadline), each arm runs one campaign in its own attempt, and
    the report carries the recurrence rows the scripted answers imply."""
    import asyncio
    from colonymind.qualification import paired_container, paired_report
    module, scenarios = family
    monkeypatch.setattr(paired_cases, 'cases', CASES)
    monkeypatch.setattr(paired_cases, 'EVALUATORS', EVALUATORS)

    async def by_arm(inputs, context):
        return await scripted_agent(module, lexical_only=inputs['arm'] == 'full-lessons')(inputs, context)
    monkeypatch.setattr(paired_container, 'CONSUMERS', {'native_paired': by_arm})
    with_snapshot_worker(monkeypatch)
    directory = write_dataset(tmp_path / 'data', scenarios, dataset_id=module.FAMILY)
    [item] = [item for item in scenarios if item['scenario'] == 'dock-lookup']
    fixture.output.mkdir(mode=0o700)
    output = fixture.output / 'smoke'
    manifest = paired.plan(output, native_binding='candidate', evidence_mode='controlled', dataset_dir=directory,
                           arms=list(ARMS), reference_arm='full-lessons', case_ids=[item['id']], **fixture.resources)
    comparison = manifest['comparison']
    assert comparison['campaign']['unit'] == 'probe' and comparison['rule']['cluster'] == 'campaign'
    assert comparison['skill_tools']['mode'] == 'read' and manifest['declared_attempts'] == 2
    assert all(pair['arms'][arm]['case']['timeout_seconds'] == 11400 for pair in manifest['pairs'] for arm in ARMS)
    report = asyncio.run(paired.run(output, **fixture.resources))
    recurrence = report['campaign']['recurrence']
    full, lexical = recurrence['arms']['full'], recurrence['arms']['full-lessons']
    assert all(row['pass_rate'] == 1.0 for axis in full.values() for row in axis.values())
    assert lexical['relation']['same_wording']['pass_rate'] == 1.0
    # The lexical arm's final workspace holds every right answer; the ones it wrote days late earn nothing.
    assert lexical['relation']['paraphrase']['pass_rate'] == lexical['relation']['new_instance']['pass_rate'] == 0.0
    [contrast] = recurrence['contrasts']
    assert contrast['relation']['paraphrase']['delta_pp'] == 100 and contrast['relation']['paraphrase']['wins'] == 3
    assert contrast['relation']['same_wording']['delta_pp'] == 0
    # The controls are worded as the failure was, so the lexical agent answers them, and rightly.
    kinds = report['campaign']['descriptive']['full-lessons']['kinds']
    assert kinds['control:scope'] == kinds['control:contact_rule'] == {'passed': 1, 'observed': 1}
    [statistics] = report['statistics']['contrasts']
    assert statistics['unit'] == 'probe' and statistics['units'] == 11 and statistics['wins'] == 6
    assert '## Recurrence' in paired_report.markdown(report)

"""Versioned paired native-agent scenarios, judged by files rather than reply wording.

The reviewed 60-case set is public development data, not a comprehensive agent
benchmark. Superseded pilot and reviewed-1 fixtures are retired. Both arms receive
identical tasks and oracles.
Memory facts arrive in ordinary turns. Session changes are sequential; disclosure
tasks test requested artifact scope, not authenticated access control. The Python
repair case verifies syntax/API shape only, not executable behavior.
"""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

from .records import CaseSpec, MAX_CAMPAIGN_OUTPUT_BYTES, MAX_CAMPAIGN_SECONDS

VERSION = 'paired-agent-reviewed-2'
BASELINE_VERSION = VERSION
WORKFLOW_VERSION = 'paired-agent-workflows-1'
DATASET_VERSIONS = (VERSION, WORKFLOW_VERSION)
WORKFLOW_FAMILIES = ('workflow-recall', 'workflow-correction', 'workflow-recovery', 'workflow-scope')
FAMILIES = (
    'grounded-evidence', 'planning-toolrecovery', 'persistent-memory',
    'crosssession-authority', 'coding-ops', 'extraction-review',
)
LIMITATIONS = (
    'Public development scenarios; neither dataset establishes comprehensive agent qualification.',
    'Sequential sessions only; no concurrency, scheduling fairness or race coverage.',
    'Disclosure checks inspect requested artifacts, not authenticated authorization or every output channel.',
    'Forgetting checks later output behavior, not deletion from every underlying store.',
    'Python repair checks syntax and function signature; candidate code is never executed by this verifier.',
    'Native memory and session search availability are diagnostics, not automatic success conditions.',
)


_FIXTURE_DIRECTORY = Path(__file__).parent / 'fixtures' / VERSION


def _leaf_name(value):
    return isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', value) is not None


def load_dataset(directory=_FIXTURE_DIRECTORY):
    """Verify versioned fixture bytes before constructing any executable case."""
    directory = Path(directory)
    manifest_raw = (directory / 'manifest.json').read_bytes()
    scenario_raw = (directory / 'scenarios.json').read_bytes()
    manifest = json.loads(manifest_raw)
    expected = manifest['files']['scenarios.json']
    if (len(scenario_raw) != expected['bytes']
            or hashlib.sha256(scenario_raw).hexdigest() != expected['sha256']):
        raise ValueError('Paired dataset checksum mismatch')
    if (manifest['dataset_id'] not in DATASET_VERSIONS
            or manifest['version'] != manifest['dataset_id']):
        raise ValueError('Unsupported paired dataset version')
    scenarios = json.loads(scenario_raw)
    if not isinstance(scenarios, list) or len(scenarios) != manifest['scenario_count']:
        raise ValueError('Paired dataset scenario count mismatch')
    identities = set()
    workflow_dataset = manifest['dataset_id'] == WORKFLOW_VERSION
    counts = dict.fromkeys(WORKFLOW_FAMILIES if workflow_dataset else FAMILIES, 0)
    for item in scenarios:
        identity, family = item['id'], item['family']
        if (family not in counts or identity in identities
                or identity != 'paired.' + family + '.' + item['scenario']):
            raise ValueError('Invalid paired scenario identity')
        identities.add(identity)
        counts[family] += 1
        files, turns, oracle = item['initial_files'], item['episodes'], item['oracle']
        if not isinstance(files, dict) or any(not _leaf_name(k) or not isinstance(v, str) for k, v in files.items()):
            raise ValueError('Initial files require leaf names and text')
        if (not isinstance(turns, list) or not 1 <= len(turns) <= (24 if workflow_dataset else 4)
                or len(turns) != oracle['declared_turns']
                or any(set(turn) != {'session_id', 'user'} or not _leaf_name(turn['session_id'])
                       or not isinstance(turn['user'], str) or not turn['user'].strip() for turn in turns)):
            raise ValueError('Invalid paired native episodes')
        artifacts = oracle['artifacts']
        if not artifacts or len({a['path'] for a in artifacts}) != len(artifacts):
            raise ValueError('Paired scenarios require distinct artifact outcomes')
        if any(not _leaf_name(a['path']) for a in artifacts):
            raise ValueError('Artifact declarations require leaf names')
        if workflow_dataset:
            from .paired_workflow_runtime import validate_workflow
            validate_workflow(item['workflow'], turns)
            if (len(turns) < 6 or not item['workflow']['restart_before']
                    or item.get('memory_condition') not in {'relevant', 'irrelevant'}):
                raise ValueError('Workflows require multi-turn history, restart and memory condition')
            checkpoints = oracle.get('checkpoints', [])
            if (not checkpoints or len({c['turn_index'] for c in checkpoints}) != len(checkpoints)
                    or {c['turn_index'] for c in checkpoints} != set(item['workflow']['snapshot_after'])
                    or any(not c['artifacts'] or any(not _leaf_name(a['path']) for a in c['artifacts'])
                           for c in checkpoints)):
                raise ValueError('Workflow checkpoints must match declared snapshots')
    if counts != manifest['families']:
        raise ValueError('Paired dataset family count mismatch')
    if workflow_dataset and (counts != dict.fromkeys(WORKFLOW_FAMILIES, 3)
            or any(sum(s['memory_condition'] == 'irrelevant' for s in scenarios if s['family'] == family) != 1
                   for family in WORKFLOW_FAMILIES)):
        raise ValueError('Frozen workflows require three per family with one irrelevant-memory control')
    content_hash = hashlib.sha256(b'manifest\0' + manifest_raw + b'\0scenarios\0' + scenario_raw).hexdigest()
    return manifest, tuple(scenarios), content_hash


DATASET_MANIFEST, _SCENARIOS, DATASET_SHA256 = load_dataset()

CASE_IDS = tuple(item['id'] for item in _SCENARIOS)

# Seeded template families (benchmarks/paired/generators) are written outside the
# frozen fixtures. Their scenarios may hold body events and body oracles.
GENERATOR_PROTOCOL = 'paired-generator-1'
# An anchor split is public external data rendered into the fixture shape (the
# LongMemEval_S anchor); it is reported descriptively, never as a held-out gate.
GENERATED_SPLITS = ('dev', 'heldout', 'anchor')
# Generated families run with every enabled Hermes tool loaded eagerly in every
# arm (paired_worker.EAGER_TOOLS_CONFIG); the frozen datasets keep stock loading.
GENERATED_TOOL_LOADING = 'eager'
# Every model-facing turn of a generated family carries the body clock in the stock
# gateway message timestamp format (paired_worker.MESSAGE_TIMESTAMP_FORMAT); the
# frozen datasets keep bare turns.
GENERATED_MESSAGE_TIMESTAMPS = 'gateway'
# Every turn and cron run of a generated family carries the same description of
# the body (paired_worker.ENVIRONMENT_NOTES); the frozen datasets carry none.
GENERATED_ENVIRONMENT_NOTE = 'messaging'
# The families whose scenarios grade messages to contacts give every arm the same outbound
# path (paired_worker.OUTBOUND_SCHEMA, families/mind-people-1.md 7.1); in every other family
# no arm has a send tool, as their plans say.
GENERATED_OUTBOUND = {'mind-people-1': 'send_message'}
# The families whose question involves skills give every arm the same read-only skill tools
# (paired_worker.SKILL_TOOLS): Hermes lists skills only to an agent with a skill tool, so without them
# no arm could see one (evals section 11, 2026-09-24). No arm has skill_manage.
GENERATED_SKILL_TOOLS = {'mind-improve-1': 'read', 'mind-recurring-1': 'read'}
# Every generated family's plugin tool set (paired_worker.PLUGIN_TOOL_SETS), fixed for its comparison series.
# The initiative series (its dev runs and held-out gate) ran with the plugin's memory tools only; the memory
# family's self-report probe reads the mind's action log. A family not listed keeps ``memory_self``.
GENERATED_PLUGIN_TOOLS = {'mind-initiative-1': 'memory'}
GENERATED_PLUGIN_TOOLS_DEFAULT = 'memory_self'
# Every episode of a generated family starts its body clock at this UTC time of day, in
# every arm (paired_body.start_offset); the frozen datasets keep the container's clock.
GENERATED_CLOCK_START = '12:00'
# Families whose scenarios grade what the agent does inside the owner's quiet hours: the window is written
# into every mind arm (paired_worker.QUIET_HOURS_PROTOCOL) and seeded in the family's owner.json for every
# arm. Every other family keeps quiet hours off in every mind arm.
GENERATED_QUIET_HOURS = {'mind-outreach-1': '22:00-07:00'}
# The per-episode deadline of a generated family, 600 s unless listed: the outreach family's direction
# templates run up to nine ticks with two research runs; the harder memory family imports a long history
# (and, with an embedding endpoint, waits up to paired_history.VECTOR_DRAIN_S for it to be embedded)
# before its first turn.
GENERATED_DEADLINE_SECONDS = {'mind-outreach-1': 1200, 'mind-memory-2': 900}
GENERATED_SCENARIO_KEYS = frozenset({'id', 'family', 'scenario', 'seed', 'role', 'initial_files',
                                     'episodes', 'limitations', 'oracle'})
# A generated scenario may also declare a process-restart contract (``workflow``, the
# frozen workflows' shape) with checkpoint artifacts graded on its snapshots, and seeded
# conversation history (``history``, paired_history) that the worker imports before the
# first turn.
GENERATED_OPTIONAL_KEYS = frozenset({'workflow', 'history'})
GENERATED_ORACLE_KEYS = frozenset({'declared_turns', 'artifacts', 'body', 'checkpoints', 'self_report'})
# A campaign (the improve family, evals section 6.8) is one scenario of ordered days sharing one
# arm's state, whose artifacts are probes: every artifact spec carries ``probe`` with its day and
# kind (training, warranted, control, old_family). Its deadline grows with the day count (one tick
# entry ends each day) and its output bound is the campaign one; the report takes the probe as
# its unit and the campaign as its cluster (paired_report).
# A recurring campaign (the recurring-failure family, mind-recurring-1) also says where each probe
# stands against the one correction it follows: every warranted and control probe declares its
# ``delay`` in days after the correction, its ``relation`` to the correction's wording, whether the
# interfering correction came before it (``after_interference``) and its own owner ``session``, and
# every training artifact names which ``correction`` it is (``target``, once, and ``interfering``,
# at most once and later) and the owner ``session`` that teaches it (the request, then the
# correction). The loader checks the metadata against the campaign's own turns: each correction's
# day is the day its session holds, so labels that only agree with each other are refused. Its
# days are days of the campaign's clock, so a delay is that many days of it: every day ends with
# one day's clock advance and then one tick (``RECURRING_DAY_END``), and nothing else advances the
# clock or ticks.
# Which campaigns are recurring follows from the family (``RECURRING_FAMILIES``, by dataset id), not
# from what their probes declare: every campaign of a recurring family declares the metadata in full
# and no other family's declares any, so a campaign cannot leave the recurrence rows and the snapshot
# grading below by omitting it.
# A recurring campaign's results are graded as their own sessions left them, not as the last day left
# them: the case asks the worker for a workspace snapshot after the last turn of each result's session
# (``inputs.campaign.snapshot_after``, the worker's campaign snapshots) and the oracle names the entry
# each result is graded after (``graded_after``, ``{path: entry index}``). A result repaired on a later
# day earns nothing at its delay, one spoiled on a later day keeps its pass, and a snapshot the worker
# never took grades as a missing file. Work the session left to a later tick does not count, in every
# arm alike. A campaign whose results name no session (the improve family) keeps its final workspace.
# An artifact may name its gold (the memory families): the kind of question it is, the earlier turns
# whose words answer it (``evidence``: a history message ``{session, message}``, or an episode turn
# ``{turn}``, with ``role: assistant`` for the agent's own reply to that turn), the values that were
# current once and were replaced or withdrawn (``stale``, each one also forbidden), the answer field each
# stale value once held when the answer has more than one field (``stale_fields``), and for a conflict
# the two values that disagree (``sides``). The grader ignores it; reach and trap scoring read it
# (benchmarks/public_memory/scoring.py). It lives in the oracle, which no arm is shown.
GOLD_KINDS = ('recall', 'update', 'conflict', 'abstain')
GOLD_KEYS = frozenset({'kind', 'evidence', 'stale', 'stale_fields', 'sides'})
CAMPAIGN_PROTOCOL = 'paired-campaign-1'
CAMPAIGN_DAY_SECONDS = 720
PROBE_KINDS = ('training', 'warranted', 'control', 'old_family')
PROBE_RELATIONS = ('same_wording', 'paraphrase', 'new_instance')
PROBE_CORRECTIONS = ('target', 'interfering')
RECURRENCE_KEYS = frozenset({'delay', 'relation', 'after_interference', 'session'})
CORRECTION_KEYS = frozenset({'correction', 'session'})
PROBE_KEYS = frozenset({'day', 'kind', 'control', 'source'}) | RECURRENCE_KEYS | CORRECTION_KEYS
RECURRING_FAMILIES = frozenset({'mind-recurring-1'})
RECURRING_DAY_END = ({'advance_clock': 86400}, {'tick': 1})


def validate_probes(item, *, recurring):
    """True when the scenario is a campaign (every artifact carries a valid probe), False when none does. A
    scenario of a recurring family (``recurring``) is a campaign that declares the recurrence metadata in full;
    any other declares none of it."""
    artifacts = (item.get('oracle') or {}).get('artifacts') or []
    marked = [spec for spec in artifacts if isinstance(spec, dict) and 'probe' in spec]
    if not marked:
        if recurring:
            raise ValueError("A recurring family's scenario is a campaign: every artifact is a probe")
        return False
    if len(marked) != len(artifacts):
        raise ValueError('A campaign declares a probe on every artifact')
    for spec in marked:
        probe = spec['probe']
        if (not isinstance(probe, dict) or set(probe) - PROBE_KEYS or not {'day', 'kind'} <= set(probe)
                or type(probe['day']) is not int or probe['day'] < 1 or probe['kind'] not in PROBE_KINDS
                or any(key in probe and (not isinstance(probe[key], str) or not probe[key])
                       for key in ('control', 'source'))):
            raise ValueError('A probe is {day >= 1, kind in ' + ', '.join(PROBE_KINDS)
                             + ', control?, source?}')
        if (('delay' in probe and (type(probe['delay']) is not int or probe['delay'] < 0))
                or ('relation' in probe and probe['relation'] not in PROBE_RELATIONS)
                or ('after_interference' in probe and type(probe['after_interference']) is not bool)
                or ('session' in probe and (not isinstance(probe['session'], str) or not _leaf_name(probe['session'])))
                or (set(probe) & (RECURRENCE_KEYS - CORRECTION_KEYS) and probe['kind'] not in ('warranted', 'control'))
                or ('session' in probe and probe['kind'] not in ('warranted', 'control', 'training'))
                or ('correction' in probe and (probe['kind'] != 'training'
                                               or probe['correction'] not in PROBE_CORRECTIONS))):
            raise ValueError('A recurring probe is {delay >= 0, relation in ' + ', '.join(PROBE_RELATIONS)
                             + ', after_interference, session} on a warranted or control probe, and a training '
                             'probe names its correction (' + ', '.join(PROBE_CORRECTIONS) + ') and its session')
    _validate_recurrence(item, [spec['probe'] for spec in marked], recurring)
    return True


def _owner_sessions(episodes):
    """``{session id: {'days': {days}, 'turns': count}}`` of the owner sessions (one tick entry ends each day). A
    session any other kind of turn shares (an inbound message, a reaction) is not the owner's and is left out."""
    day, found, others = 1, {}, set()
    for entry in episodes if isinstance(episodes, list) else []:
        if not isinstance(entry, dict):
            continue
        if 'tick' in entry:
            day += 1
        elif isinstance(entry.get('session_id'), str):
            if 'user' not in entry:
                others.add(entry['session_id'])
                continue
            session = found.setdefault(entry['session_id'], {'days': set(), 'turns': 0})
            session['days'].add(day)
            session['turns'] += 1
    return {session: value for session, value in found.items() if session not in others}


def _validate_recurrence(item, probes, recurring):
    """A recurring campaign's metadata matches its own turns: one target correction and at most one later
    interfering one, each on the one day its owner session holds its request and its correction; every probe on
    the target's day plus its delay, in its own owner session of that day, and after the interference exactly when
    the interfering correction came on an earlier day. No two of them share a session. Another family's campaign
    declares none of it."""
    units = [probe for probe in probes if probe['kind'] in ('warranted', 'control')]
    training = [probe for probe in probes if probe['kind'] == 'training']
    if not recurring:
        if any(set(probe) & (RECURRENCE_KEYS | CORRECTION_KEYS) for probe in probes):
            raise ValueError("Only a recurring family's campaign declares recurrence metadata on a probe")
        return
    if not units or not training or any(set(probe) & RECURRENCE_KEYS != RECURRENCE_KEYS for probe in units) or any(
            set(probe) & CORRECTION_KEYS != CORRECTION_KEYS for probe in training):
        raise ValueError('A recurring campaign declares delay, relation, after_interference and session on every '
                         'probe (one at least) and a correction and its session on every training probe')
    _validate_recurring_days(item.get('episodes'))
    targets = [probe['day'] for probe in training if probe['correction'] == 'target']
    interfering = [probe['day'] for probe in training if probe['correction'] == 'interfering']
    if len(targets) != 1 or len(interfering) > 1 or any(day <= targets[0] for day in interfering):
        raise ValueError('A recurring campaign has one target correction and at most one later interfering '
                         'correction, each a training probe')
    sessions = _owner_sessions(item.get('episodes'))
    if len({probe['session'] for probe in units + training}) != len(units) + len(training):
        raise ValueError('Every recurring probe and correction has a session of its own')
    for probe in training:
        taught = sessions.get(probe['session'])
        if taught is None or taught['days'] != {probe['day']} or taught['turns'] < 2:
            raise ValueError("A recurring training probe names the owner session that teaches it, the owner's "
                             "request and its correction, on the probe's day")
    for probe in units:
        asked = sessions.get(probe['session'])
        if (probe['day'] != targets[0] + probe['delay'] or asked is None or asked['days'] != {probe['day']}
                or probe['day'] in interfering
                or probe['after_interference'] is not any(day < probe['day'] for day in interfering)):
            raise ValueError("A recurring probe sits on the correction's day plus its delay, in an owner session of "
                             'that day, and after the interfering correction exactly when it came on an earlier day')


def validate_gold(gold, spec, item):
    """An artifact's ``gold`` names turns the scenario has, stale values it forbids, and sides only for a conflict."""
    if (not isinstance(gold, dict) or set(gold) - GOLD_KEYS or not {'kind', 'evidence'} <= set(gold)
            or gold['kind'] not in GOLD_KINDS or not isinstance(gold['evidence'], list)):
        raise ValueError('Artifact gold is {kind, evidence, stale?, sides?} with a known kind')
    history = {session['id']: session['messages'] for session in item.get('history') or []}
    episodes, seen = item['episodes'], set()
    for ref in gold['evidence']:
        if isinstance(ref, dict) and set(ref) == {'session', 'message'}:
            messages = history.get(ref['session'])
            if messages is None or type(ref['message']) is not int or not 0 <= ref['message'] < len(messages):
                raise ValueError("An artifact's gold evidence names a history message the scenario does not have")
            key = (ref['session'], ref['message'])
        elif (isinstance(ref, dict) and 'turn' in ref and set(ref) <= {'turn', 'role'}
              and ref.get('role', 'user') in ('user', 'assistant')):
            if (type(ref['turn']) is not int or not 0 <= ref['turn'] < len(episodes)
                    or not isinstance(episodes[ref['turn']], dict) or 'user' not in episodes[ref['turn']]):
                raise ValueError("An artifact's gold evidence names an episode turn the scenario does not have")
            key = (ref['turn'], ref.get('role', 'user'))
        else:
            raise ValueError("An artifact's gold evidence is {session, message} or {turn, role?}")
        if key in seen:
            raise ValueError("An artifact's gold evidence names one turn twice")
        seen.add(key)
    stale, forbidden = gold.get('stale', []), spec.get('forbidden', [])
    if (not isinstance(stale, list) or len(set(stale)) != len(stale)
            or any(not isinstance(value, str) or not value.strip() or value not in forbidden for value in stale)):
        raise ValueError("An artifact's gold stale values are distinct strings it forbids")
    if (gold['kind'] == 'update') != bool(stale) and gold['kind'] != 'abstain':
        raise ValueError("An update's gold names its stale values, and only an update or an abstention has them")
    fields = {rule['path'][0] for rule in spec.get('assertions', []) if isinstance(rule, dict)
              and rule.get('op') in ('label_one_of', 'number') and isinstance(rule.get('path'), list)
              and len(rule['path']) == 1 and isinstance(rule['path'][0], str)}
    held = gold.get('stale_fields')
    if held is None and stale and len(fields) > 1:
        raise ValueError("A gold with stale values names the field each held when the answer has several")
    if held is not None and (not isinstance(held, dict) or set(held) != set(stale)
                             or any(field not in fields for field in held.values())):
        raise ValueError("An artifact's gold stale_fields maps each stale value to a field the answer has")
    sides = gold.get('sides')
    if (gold['kind'] == 'conflict') != (sides is not None) or sides is not None and (
            not isinstance(sides, list) or len(sides) != 2 or len(set(sides)) != 2
            or any(not isinstance(value, str) or not value.strip() for value in sides)):
        raise ValueError("A conflict's gold names its two sides, and only a conflict has them")
    if gold['kind'] != 'abstain' and not gold['evidence']:
        raise ValueError("An artifact's gold evidence is empty only for an abstention")
    return gold


def _validate_recurring_days(episodes):
    """A recurring campaign's days are days of its clock: each ends with ``RECURRING_DAY_END`` (one day's clock
    advance, then one tick), the last one too, and no other entry advances the clock or ticks."""
    def day_end(at):
        pair = episodes[at:at + 2]
        return len(pair) == 2 and all(isinstance(entry, dict) and entry == expected
                                      and all(type(value) is int for value in entry.values())
                                      for entry, expected in zip(pair, RECURRING_DAY_END))

    episodes = episodes if isinstance(episodes, list) else []
    at, ended = 0, False
    while at < len(episodes):
        entry = episodes[at]
        if isinstance(entry, dict) and ('advance_clock' in entry or 'tick' in entry):
            if not day_end(at):
                break
            at, ended = at + 2, True
            continue
        at, ended = at + 1, False
    if at < len(episodes) or not ended:
        raise ValueError("A recurring campaign's days are days of its clock, so a probe's delay is that many days: "
                         'every day ends with one advance_clock of 86400 seconds and then one tick, and nothing else '
                         'advances the clock or ticks')


def graded_after(item):
    """``{path: entry index}`` of a campaign whose results name their sessions: the last episode entry of each
    result's own session; empty for any other scenario."""
    ends = {entry['session_id']: index for index, entry in enumerate(item['episodes'])
            if isinstance(entry, dict) and 'user' in entry}
    return {spec['path']: ends[spec['probe']['session']] for spec in item['oracle']['artifacts']
            if isinstance(spec.get('probe'), dict) and 'session' in spec['probe']}


def graded_artifact(effects, oracle, path):
    """The file the checks grade: the final workspace's, or for a result graded after an entry, the file as the
    worker's snapshot after that entry held it. A snapshot keeps only what differs from the final workspace (None
    for a file not there then); a snapshot the worker did not take gives no file."""
    artifacts = effects.get('artifacts') if isinstance(effects, dict) else None
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    index = (oracle.get('graded_after') or {}).get(path)
    if index is None:
        return artifacts.get(path)
    snapshots = effects.get('campaign_snapshots')
    taken = snapshots.get(str(index)) if isinstance(snapshots, dict) else None
    if not isinstance(taken, dict):
        return None
    return taken[path] if path in taken else artifacts.get(path)


def campaign_days(episodes):
    """A campaign's days: one tick entry ends each day."""
    return sum(isinstance(entry, dict) and 'tick' in entry for entry in episodes)


def _validate_checkpoints(checkpoints, workflow):
    if (not isinstance(checkpoints, list) or not checkpoints
            or len({c.get('turn_index') for c in checkpoints if isinstance(c, dict)}) != len(checkpoints)
            or any(not isinstance(c, dict) or set(c) != {'turn_index', 'artifacts'}
                   or c['turn_index'] not in workflow['snapshot_after']
                   or not isinstance(c['artifacts'], list) or not c['artifacts']
                   or any(not isinstance(a, dict) or not _leaf_name(a.get('path')) for a in c['artifacts'])
                   or len({a['path'] for a in c['artifacts']}) != len(c['artifacts'])
                   for c in checkpoints)):
        raise ValueError('Generated checkpoints must grade declared snapshots')


def _generated_content_hash(manifest_raw, scenario_raw):
    return hashlib.sha256(b'manifest\0' + manifest_raw + b'\0scenarios\0' + scenario_raw).hexdigest()


def generated_content_hash(directory):
    """The content hash a plan freezes for a generated family (its ``source_sha256``), from the family's bytes."""
    directory = Path(directory)
    return _generated_content_hash((directory / 'manifest.json').read_bytes(),
                                   (directory / 'scenarios.json').read_bytes())


def load_generated_dataset(directory):
    """Verify a generated family's bytes and shape before constructing any executable case."""
    directory = Path(directory)
    manifest_raw = (directory / 'manifest.json').read_bytes()
    scenario_raw = (directory / 'scenarios.json').read_bytes()
    manifest = json.loads(manifest_raw)
    expected = manifest['files']['scenarios.json']
    if (len(scenario_raw) != expected['bytes']
            or hashlib.sha256(scenario_raw).hexdigest() != expected['sha256']):
        raise ValueError('Generated dataset checksum mismatch')
    generator = manifest.get('generator')
    if (not isinstance(generator, dict) or generator.get('protocol') != GENERATOR_PROTOCOL
            or generator.get('split') not in GENERATED_SPLITS or type(generator.get('seed')) is not int
            or not _leaf_name(manifest.get('dataset_id')) or manifest.get('version') != manifest['dataset_id']
            or manifest['dataset_id'] in DATASET_VERSIONS):
        raise ValueError('Unsupported generated dataset manifest')
    scenarios = json.loads(scenario_raw)
    if (not isinstance(scenarios, list) or not 1 <= len(scenarios) <= 128
            or len(scenarios) != manifest['scenario_count']):
        raise ValueError('Generated dataset scenario count mismatch')
    from .paired_body_grading import validate_body_oracle, validate_self_report_oracle
    from .paired_history import validate_history
    from .paired_workflow_runtime import validate_episodes, validate_workflow
    identities, counts, campaigns = set(), {}, set()
    for item in scenarios:
        if (not isinstance(item, dict) or set(item) - GENERATED_OPTIONAL_KEYS != GENERATED_SCENARIO_KEYS
                or not _leaf_name(item['id']) or item['id'] in identities
                or not _leaf_name(item['family']) or not _leaf_name(item['scenario'])
                or type(item['seed']) is not int or not isinstance(item['role'], str) or not item['role']
                or not isinstance(item['limitations'], list)):
            raise ValueError('Invalid generated scenario')
        identities.add(item['id'])
        counts[item['family']] = counts.get(item['family'], 0) + 1
        files, oracle = item['initial_files'], item['oracle']
        if not isinstance(files, dict) or any(not _leaf_name(k) or not isinstance(v, str) for k, v in files.items()):
            raise ValueError('Initial files require leaf names and text')
        kinds = validate_episodes(item['episodes'])
        workflow = validate_workflow(item['workflow'], item['episodes']) if 'workflow' in item else None
        if 'history' in item:
            validate_history(item['history'])
            sessions = {entry['session_id'] for entry in item['episodes'] if 'session_id' in entry}
            if any(session['id'] in sessions for session in item['history']):
                raise ValueError('History session ids must differ from the episode session ids')
        artifacts = oracle.get('artifacts') if isinstance(oracle, dict) else None
        if (not isinstance(oracle, dict) or set(oracle) - GENERATED_ORACLE_KEYS
                or oracle.get('declared_turns') != len(kinds) or not isinstance(artifacts, list)
                or any(not isinstance(a, dict) or not _leaf_name(a.get('path')) for a in artifacts)
                or len({a['path'] for a in artifacts}) != len(artifacts)
                or not (artifacts or 'body' in oracle or 'self_report' in oracle)
                or ('checkpoints' in oracle and workflow is None)):
            raise ValueError('Generated scenarios need artifact, body or self-report outcomes')
        if 'body' in oracle:
            validate_body_oracle(oracle['body'])
        if 'self_report' in oracle:
            validate_self_report_oracle(oracle['self_report'])
        if 'checkpoints' in oracle:
            _validate_checkpoints(oracle['checkpoints'], workflow)
        for spec in artifacts:
            if 'gold' in spec:
                validate_gold(spec['gold'], spec, item)
        campaigns.add(validate_probes(item, recurring=manifest['dataset_id'] in RECURRING_FAMILIES))
    if len(campaigns) != 1:
        raise ValueError('A generated dataset is all campaigns (every artifact a probe) or none')
    if counts != manifest['families']:
        raise ValueError('Generated dataset family count mismatch')
    return manifest, tuple(scenarios), _generated_content_hash(manifest_raw, scenario_raw)


def cases(arm, case_ids=None, *, dataset_version=VERSION, profile=None, dataset_dir=None):
    """Return fresh independent declarations; only inputs.arm and inputs.profile differ by arm."""
    if profile is None and arm not in {'base_hermes', 'colonymind'}:
        raise ValueError('Use base_hermes or colonymind, or declare an arm profile')
    if profile is not None and (not isinstance(profile, dict) or type(profile.get('plugin')) is not bool
                                or not isinstance(profile.get('overlay'), dict)
                                or not isinstance(arm, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,41}', arm)):
        raise ValueError('An arm label needs a profile with plugin and overlay')
    split = None
    if dataset_dir is not None:
        manifest, scenarios, content_hash = load_generated_dataset(dataset_dir)
        dataset_version, split = manifest['dataset_id'], manifest['generator']['split']
    elif dataset_version not in DATASET_VERSIONS:
        raise ValueError('Select an installed paired dataset version')
    elif dataset_version == VERSION:
        scenarios, content_hash = _SCENARIOS, DATASET_SHA256
    else:
        _, scenarios, content_hash = load_dataset(_FIXTURE_DIRECTORY.parent / dataset_version)
    identities = tuple(item['id'] for item in scenarios)
    selected = list(identities) if case_ids is None else list(case_ids)
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(identities):
        raise ValueError('Select distinct installed paired case IDs')
    result = []
    for scenario in scenarios:
        if scenario['id'] not in selected:
            continue
        inputs = {key: copy.deepcopy(scenario[key]) for key in (
            'family', 'scenario', 'role', 'initial_files', 'episodes', 'limitations')}
        inputs.update(arm=arm, max_output_tokens=4096, max_iterations=8,
            cleanup_seconds=5, settle_seconds=5, contact_id='fixture-owner',
            dataset={'id': dataset_version, 'version': dataset_version, 'sha256': content_hash})
        if profile is not None:
            inputs['profile'] = copy.deepcopy(profile)
        oracle = copy.deepcopy(scenario['oracle'])
        if dataset_version == WORKFLOW_VERSION:
            inputs.update(workflow=copy.deepcopy(scenario['workflow']),
                          memory_condition=scenario['memory_condition'])
            oracle['workflow_contract'] = copy.deepcopy(scenario['workflow'])
            inputs['dataset']['split'] = 'frozen_public_evaluation'
        if split is not None:
            inputs['dataset']['split'] = split
            inputs['tool_loading'] = GENERATED_TOOL_LOADING
            inputs['message_timestamps'] = GENERATED_MESSAGE_TIMESTAMPS
            inputs['environment_note'] = GENERATED_ENVIRONMENT_NOTE
            if dataset_version in GENERATED_OUTBOUND:
                inputs['outbound'] = GENERATED_OUTBOUND[dataset_version]
            if dataset_version in GENERATED_SKILL_TOOLS:
                inputs['skill_tools'] = GENERATED_SKILL_TOOLS[dataset_version]
            if dataset_version in GENERATED_QUIET_HOURS:
                inputs['quiet_hours'] = GENERATED_QUIET_HOURS[dataset_version]
            inputs['plugin_tools'] = GENERATED_PLUGIN_TOOLS.get(dataset_version, GENERATED_PLUGIN_TOOLS_DEFAULT)
            inputs['clock_start'] = GENERATED_CLOCK_START
            if 'workflow' in scenario:
                # The normalized contract: the supervisor restarts the worker process before
                # the probe and the workflow grader checks the lifecycle and the checkpoints.
                from .paired_workflow_runtime import validate_workflow
                contract = validate_workflow(scenario['workflow'], scenario['episodes'])
                inputs['workflow'] = copy.deepcopy(contract)
                oracle['workflow_contract'] = copy.deepcopy(contract)
            if 'history' in scenario:
                inputs['history'] = copy.deepcopy(scenario['history'])
        # Tick episodes wait for cron runs and in-process workers; give them the workflow deadline.
        generous = dataset_version == WORKFLOW_VERSION or split is not None
        timeout_seconds = (GENERATED_DEADLINE_SECONDS.get(dataset_version, 600) if split is not None
                           else 600 if generous else 120 * len(scenario['episodes']) + 30)
        if split is not None and dataset_version in GENERATED_DEADLINE_SECONDS:
            inputs['family_deadline'] = timeout_seconds
        max_output_bytes = 1048576 if generous else 262144
        if split is not None and validate_probes(scenario, recurring=dataset_version in RECURRING_FAMILIES):
            days = campaign_days(scenario['episodes'])
            inputs['campaign'] = {'protocol': CAMPAIGN_PROTOCOL, 'days': days}
            graded = graded_after(scenario)
            if graded:
                inputs['campaign']['snapshot_after'] = sorted(set(graded.values()))
                oracle['graded_after'] = graded
            timeout_seconds = min(MAX_CAMPAIGN_SECONDS, 600 + CAMPAIGN_DAY_SECONDS * days)
            max_output_bytes = MAX_CAMPAIGN_OUTPUT_BYTES
        result.append(CaseSpec(id=scenario['id'], version=dataset_version, role=scenario['role'],
            boundary='native_hermes', consumer='native_paired', evaluator='paired_artifacts',
            inputs=inputs, oracle=oracle, timeout_seconds=timeout_seconds, max_output_bytes=max_output_bytes))
    return result


def _same(a, b):
    # JSON numbers with equal values are equivalent; booleans are not counts.
    if _number(a) and _number(b):
        return a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _schedule(actual, expected):
    if not isinstance(actual, list) or len(actual) != len(expected['tasks']):
        return False
    by_id = {}
    for item in actual:
        if not isinstance(item, dict) or not isinstance(item.get('id'), str):
            return False
        identity, start, finish = item['id'], item.get('start'), item.get('finish')
        if identity not in expected['tasks'] or identity in by_id or not _number(start) or not _number(finish):
            return False
        if start < 0 or finish > expected['max_finish'] or not math.isclose(
                finish - start, expected['tasks'][identity]['duration'], abs_tol=1e-9):
            return False
        by_id[identity] = item
    for identity, spec in expected['tasks'].items():
        if any(by_id[parent]['finish'] > by_id[identity]['start'] for parent in spec['after']):
            return False
        for other, other_spec in expected['tasks'].items():
            if identity != other and spec['resource'] == other_spec['resource']:
                left, right = by_id[identity], by_id[other]
                if max(left['start'], right['start']) < min(left['finish'], right['finish']):
                    return False
    return True


def _assertion(actual, rule):
    try:
        for key in rule['path']:
            actual = actual[key]
    except (KeyError, IndexError, TypeError):
        return False
    expected, op = rule['value'], rule['op']
    if op == 'equals':
        return _same(actual, expected)
    if op == 'number':
        return _number(actual) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
    if op == 'set_equals':
        return (isinstance(actual, list) and len(actual) == len(expected)
            and all(sum(_same(value, item) for value in actual) == 1 for item in expected))
    if op == 'label_one_of':
        return isinstance(actual, str) and actual.strip().casefold() in {x.casefold() for x in expected}
    if op == 'labels_set_equals':
        if not isinstance(actual, list) or any(not isinstance(x, str) for x in actual):
            return False
        aliases = rule.get('aliases', {})
        normalized = [aliases.get(x.strip().casefold(), x.strip().casefold()) for x in actual]
        return len(normalized) == len(expected) and set(normalized) == set(expected)
    if op == 'keys_equal':
        return isinstance(actual, dict) and set(actual) == set(expected)
    if op == 'schedule':
        return _schedule(actual, expected)
    raise ValueError('Unknown paired artifact assertion')


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError('Nonfinite JSON number')


def _decoded_strings(value):
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)


def _artifact_checks(raw, spec):
    try:
        if not isinstance(raw, str) or len(raw.encode('utf-8')) > 65536:
            return False
        if any(value.casefold() in raw.casefold() for value in spec.get('forbidden', [])):
            return False
        if spec['format'] == 'text':
            return raw == spec['exact']
        if spec['format'] == 'python':
            tree = ast.parse(raw)
            functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == spec['function']]
            if len(functions) != 1:
                return False
            args = functions[0].args
            return ([a.arg for a in args.args] == spec['arguments'] and not args.posonlyargs
                and not args.kwonlyargs and args.vararg is None and args.kwarg is None
                and not args.defaults)
        if spec['format'] != 'json':
            raise ValueError('Unknown paired artifact format')
        value = json.loads(raw, object_pairs_hook=_object, parse_constant=_nonfinite)
        if spec.get('forbidden_in_decoded_json') and any(
                marker.casefold() in item.casefold()
                for item in _decoded_strings(value) for marker in spec.get('forbidden', [])):
            return False
        return all(_assertion(value, item) for item in spec['assertions'])
    except (ValueError, TypeError, SyntaxError, RecursionError, OverflowError):
        return False


def assess(observed, oracle):
    """No prose judge, tool-name reward, or advantage for merely loading treatment."""
    effects = observed.get('effects', {})
    if not isinstance(effects, dict):
        effects = {}
    completed, declared = effects.get('turns_completed'), effects.get('declared_turns')
    checks = {'all_native_turns_completed': type(completed) is int and type(declared) is int
        and completed == declared == oracle['declared_turns']}
    for spec in oracle['artifacts']:
        path = spec['path']
        if PurePosixPath(path).name != path or path in {'', '.', '..'}:
            raise ValueError('Artifact declarations must use leaf names')
        checks['artifact:' + path] = _artifact_checks(graded_artifact(effects, oracle, path), spec)
    if 'workflow_contract' in oracle:
        from .paired_workflow_grading import assess_workflow
        checks.update(assess_workflow(effects, oracle))
    if 'body' in oracle:
        from .paired_body_grading import assess_body
        checks.update(assess_body(effects, oracle['body']))
    if 'self_report' in oracle:
        from .paired_body_grading import assess_self_report
        checks.update(assess_self_report(effects, oracle['self_report']))
    return checks


EVALUATORS = {'paired_artifacts': assess}

"""The built-in arm registry: every arm switches exactly what its name says against its reference.

Two families of arms measure the faculties. The ``full`` family starts from the shipped defaults: ``full-<faculty>``
turns off one faculty that ships on, ``full-<drive>`` sets one drive's weight to 0 and ``full-plus-<faculty>`` turns
on one faculty that ships off. The ``all-on`` family starts from every faculty on: ``all-on-<faculty>`` turns one
off and ``all-on-plus-affect_rules`` puts the feeling rules in place of the feelings. What an arm runs is read here
as the worker applies it (the mind section, the effective context switches and the environment the served host
reads, the overlay last), and compared setting by setting with the arm's reference, so an arm whose name and
switches disagree, or two arms that run the same thing, fail here.
"""
import pytest

from colonymind.config import DEFAULTS
from colonymind.qualification import native_memory_worker as worker
from colonymind.qualification import paired, paired_worker
from colonymind.reports import arms

CONTEXT = ('status',)
DRIVES = tuple(DEFAULTS['mind']['drives'])
# The arms outside the two families: the stock-Hermes comparators, the plugin with the mind off, and the
# initiative family's treatment arm (the mind with only initiative on), plus the summary-only mechanism arm.
OUTSIDE_FAMILIES = {'base_hermes', 'base-heartbeat', 'base-curator', 'base-heartbeat-checkin', 'colonymind',
                    'colonymind-initiative', 'wm.summary_only'}
# The one arm whose name predates the pattern: the feelings family's mechanism arm, full without the running
# feelings and with the feeling rules in their place (``full-affect`` plus the rules).
COMPOUND = {'full-affect-plus-rules': {'affect': False, 'affect_rules': True}}
# The arms the release's pilots ran, by name: renaming one would orphan its measured results.
PILOT_ARMS = ('full', 'base_hermes', 'full-plus-owner_gate', 'full-plus-learning', 'full-plus-wishes',
              'full-plus-drift_check')
# The status block's pilot declared its own profile, an overlay on full; it stays a declared profile.
DECLARED_STATUS = {'full-plus-status': {'plugin': True, 'full': True, 'overlay': {'COLONYMIND_CONTEXT_STATUS': 'on'}}}
# The faculties the pilots turned on over full, which ship on now (on unless shown harmful): their arms keep their
# names, so the pilots' results stay attached, and run what full runs; each is measured as all-on against
# all-on-<faculty>, and a report reads a pilot's result as the faculty only with the defaults it records.
SHIPPED_ON_SINCE = ('owner_gate', 'learning', 'wishes', 'drift_check')
# Arms that run the same thing because of what ships on: the shipped defaults are every faculty on but skills and
# the feeling rules (so full is all-on without skills, and with skills added it is all-on), the four pilot arms
# above name faculties that ship on, and with the learning rules on by default the plain mind-off arm is all-on
# with the mind off.
COINCIDE = ({'full', 'all-on-skills', *(f'full-plus-{name}' for name in SHIPPED_ON_SINCE)},
            {'full-plus-skills', 'all-on'}, {'colonymind', 'all-on-initiative'})
# Learner declarations are shared by every arm; each area has a learner so its intervention is observable.
LEARNERS = {f'sample_{area}': {'area': area, 'enabled': True, 'accumulation': True,
                              'teachers': {'owner_label': 'label', 'stronger_model': 'label'}}
            for area in paired.PLASTICITY_AREAS}
LEARNER_ARMS = set(paired.plasticity_profiles()) - {'all-on-plasticity'}


def _flatten(value, prefix=''):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(_flatten(item, f'{prefix}{key}.'))
        return out
    return {prefix[:-1]: value}


def runs(profile):
    """Everything an arm runs with, setting by setting: the plugin, a comparator's extra, the mind section, the
    effective context switches, working-memory mechanism and the environment the served host reads them from
    (the overlay applied last)."""
    mind = paired_worker.mind_switches(profile) if profile['plugin'] else None
    section = worker.mind_section(mind)
    context = worker.context_section({**(mind or {}), **(paired_worker.context_switches(profile) or {})})
    effective = {name: (context or {}).get(name, DEFAULTS['context'][name]) for name in CONTEXT}
    environment = {**worker.arm_environment(section, context), **profile.get('overlay', {})} if profile['plugin'] \
        else dict(profile.get('overlay', {}))
    comparator = {name: True for name in ('heartbeat', 'heartbeat_checkin', 'curator') if profile.get(name)}
    config = worker.instance_config('owner', section, context, plasticity={'learners': LEARNERS}, profile=profile)
    return _flatten({'plugin': profile['plugin'], 'comparator': comparator, 'mind': section,
                     'context': effective if profile['plugin'] else {}, 'environment': environment,
                     'plasticity': config['plasticity'],
                     'working_memory': {'summary_only': config.get('working_memory', {}).get(
                         'summary_only', DEFAULTS['working_memory']['summary_only'])}})


def changed(arm, reference):
    first, second = runs(paired.PROFILES[arm]), runs(paired.PROFILES[reference])
    return {path: (first.get(path), second.get(path)) for path in set(first) | set(second)
            if first.get(path) != second.get(path)}


def expected(name, on):
    """The settings that switching faculty ``name`` to ``on`` changes, as an install derives them."""
    if name in DRIVES:
        return {f'mind.drives.{name}': (0.0, DEFAULTS['mind']['drives'][name])}
    if name == 'status':
        return {'context.status': (on, not on),
                'environment.COLONYMIND_CONTEXT_STATUS': ('on' if on else 'off', 'off' if on else 'on')}
    paths = {f'mind.faculties.{name}': (on, not on)}
    if name == 'learning':
        # The claim projection reads the learning rules from the environment, outside the mind.
        paths['environment.COLONYMIND_LEARNING'] = ('1' if on else '0', '0' if on else '1')
    if name == 'priors_by_meaning':
        # So does it the vector lane of its prior records.
        paths['environment.COLONYMIND_PRIORS_BY_MEANING'] = ('1' if on else '0', '0' if on else '1')
    return paths


def named(arm):
    """What the arm's name says: its reference and the faculties (or drive) it switches, on or off."""
    if arm in COMPOUND:
        return 'full', COMPOUND[arm]
    for family in ('all-on', 'full'):
        if arm == family:
            return family, {}
        for prefix, on in ((f'{family}-plus-', True), (f'{family}-', False)):
            if arm.startswith(prefix):
                return family, {arm[len(prefix):]: on}
    raise AssertionError(f'{arm} is in neither family')


FAMILY_ARMS = sorted(name for name in paired.PROFILES
                     if name not in OUTSIDE_FAMILIES | LEARNER_ARMS and name != 'all-on-initiative')


@pytest.mark.parametrize('arm', sorted(LEARNER_ARMS))
def test_each_learner_arm_switches_only_its_named_dimension_in_the_worker(arm):
    want = {}
    for name, declaration in LEARNERS.items():
        if arm == 'shuffled-control':
            key, values = 'control', ('shuffled', None)
        elif arm == 'no_accumulation':
            key, values = 'accumulation', (False, True)
        elif arm == 'full-plasticity' or arm == f"plasticity.{declaration['area']}":
            key, values = 'enabled', (False, True)
        else:
            continue
        want[f'plasticity.learners.{name}.{key}'] = values
    if arm == 'full-plasticity':   # the base arm also switches off the core default learners
        from colonymind.config import DEFAULTS
        want.update({f'plasticity.learners.{name}.enabled': (False, None)
                     for name in DEFAULTS['plasticity']['learners']})
    assert changed(arm, 'all-on') == want


def test_the_summary_arm_changes_only_the_working_memory_mechanism():
    assert changed('wm.summary_only', 'all-on') == {'working_memory.summary_only': (True, False)}


@pytest.mark.parametrize('reference', ['all-on', 'all-on-working_memory', 'all-on-skills', 'base_hermes'])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('off', [None, frozenset(), arms.EARLIER_OFF])
def test_summary_mechanism_contrasts_never_supply_whole_faculty_evidence(tmp_path, reference, reverse, off):
    from colonymind.reports import bench, working_memory

    treatment, comparator = ('wm.summary_only', reference) if reverse else (reference, 'wm.summary_only')
    contrast = {'treatment': treatment, 'comparator': comparator, 'verdict': 'demonstrated',
                'units': 12, 'treatment_pass_rate': 1.0, 'comparator_pass_rate': 0.0, 'delta_pp': 100.0,
                'ci_pp': {'lower': 100.0, 'upper': 100.0}, 'wins': 12, 'losses': 0,
                'sign_test': {'p_two_sided': 0.0005}}
    report = {'profiles': {name: {'name': name, **paired.PROFILES[name]} for name in (treatment, comparator)},
              'statistics': {'contrasts': [contrast]}, 'dataset': {'version': 'working-memory-1', 'split': 'dev'}}
    if off is not None:
        report['shipped_defaults'] = {name: name not in off for name in arms.EARLIER_OFF}
    loaded = bench.Loaded(tmp_path / 'report-001.json', 0.0, report)
    ordinary, diagnostics = working_memory.benchmark_evidence([loaded])
    assert ordinary[0].report['statistics']['contrasts'] == [] and len(diagnostics) == 1

    item = bench.read(loaded, contrast)
    assert item.subject == arms.Subject('mixed', keys=['the working-memory summary-only mechanism'])
    assert item.reading == 'unattributed' and item.key is None
    assert set(bench.verdicts([item]).values()) == {'not measured'}
    text = bench.document([loaded]).text()
    assert 'the working-memory summary-only mechanism' in text
    assert 'The same arm twice' not in text


def test_summary_profiles_remain_distinct_without_hiding_shared_settings():
    summary, full = paired.PROFILES['wm.summary_only'], paired.PROFILES['all-on']
    assert arms.state(summary) != arms.state(full)
    assert arms.subject(summary, summary) == arms.Subject('same')
    assert arms.subject(full, {**full, 'wm_summary_only': False}) == arms.Subject('same')
    assert arms.subject(summary, {**summary, 'minus_skills': True}) == arms.Subject(
        'faculty', key='skills', faculty_on='treatment')
    assert arms.describe(summary) == 'the mind with every faculty on, with summaries instead of branch consults'
    assert arms.describe(full) == 'the mind with every faculty on'


@pytest.mark.parametrize('arm', FAMILY_ARMS)
def test_each_arm_switches_exactly_what_its_name_says_against_its_reference(arm):
    reference, switched = named(arm)
    profile, base = paired.PROFILES[arm], paired.PROFILES[reference]
    assert profile['plugin'] is True and profile['overlay'] == {} and profile['full'] is True
    assert bool(profile.get('all_on')) is (reference == 'all-on')
    want = {}
    for name, on in switched.items():
        assert name in worker.MIND_FACULTIES or name in worker.MEANING_FACULTIES or name in CONTEXT \
            or name in DRIVES, name
        if not (reference == 'full' and on and name in SHIPPED_ON_SINCE):
            want.update(expected(name, on))
    assert changed(arm, reference) == want
    if len(switched) == 1 and not set(switched) & set(DRIVES):
        (name, on), = switched.items()
        subject = arms.subject(profile, base)
        if reference == 'full' and on and name in SHIPPED_ON_SINCE:
            # It runs what full runs today: a report that does not record its image's defaults cannot tell today's
            # run from its pilot's, and one that records the faculty shipped off reads it as the faculty.
            assert subject == arms.Subject('unrecorded', keys=[name])
            subject = arms.subject(profile, base, arms.EARLIER_OFF)
        assert (subject.kind, subject.key) == ('faculty', name)
        assert subject.faculty_on == ('treatment' if on else 'comparator')


def test_the_initiative_arm_is_all_on_with_the_mind_off_and_nothing_else_it_does_not_need():
    """Initiative's measure is the mind itself: what runs without the mind (the memory blocks, the claim-side
    learning rules) stays as all-on has it, and only the mind and the status block it renders go."""
    diff = changed('all-on-initiative', 'all-on')
    assert diff['mind.enabled'] == (False, True)
    assert set(diff) <= {path for path in runs(paired.PROFILES['all-on']) if path.startswith('mind.')} | {
        'context.status', 'environment.COLONYMIND_CONTEXT_STATUS'}
    subject = arms.subject(paired.PROFILES['all-on'], paired.PROFILES['all-on-initiative'])
    assert (subject.kind, subject.key, subject.faculty_on) == ('faculty', 'initiative', 'treatment')


def test_every_built_in_arm_runs_something_no_other_arm_runs():
    """Every arm runs something no other arm runs, but the groups ``COINCIDE`` names, each for a reason in what
    ships on."""
    groups = {}
    for name, profile in paired.PROFILES.items():
        groups.setdefault(tuple(sorted(runs(profile).items())), set()).add(name)
    shared = [names for names in groups.values() if len(names) > 1]
    assert sorted(map(sorted, shared)) == sorted(map(sorted, COINCIDE))


def test_every_switch_is_declared_once_and_every_faculty_switch_has_its_arm():
    for switches in (paired_worker.PROFILE_SWITCHES, worker.MIND_FACULTIES, paired.ALL_ON_FACULTIES):
        assert len(switches) == len(set(switches))
    assert set(DEFAULTS['mind']['faculties']) == set(worker.MIND_FACULTIES)
    assert set(paired_worker.CONTEXT_SWITCHES).isdisjoint(paired_worker.MIND_SWITCHES)
    for switch in (*paired_worker.MIND_FACULTY_ABLATIONS, *paired_worker.MIND_ALL_ON_ABLATIONS,
                   *paired_worker.MIND_MEANING_ABLATIONS, *paired_worker.CONTEXT_SWITCHES):
        assert f"all-on-{switch[len('minus_'):]}" in paired.PROFILES
    for switch in paired_worker.MIND_ADDITIONS:
        name = switch[len('plus_'):]
        assert f'full-plus-{name}' in paired.PROFILES or any(
            arm in paired.PROFILES and named(arm)[1].get(name) is True for arm in COMPOUND)
        assert DEFAULTS['mind']['faculties'][name] is False or name in SHIPPED_ON_SINCE, \
            'an addition turns on a faculty that ships off (or did when its pilot ran)'
    for name in paired.PROFILES:
        profile = paired.PROFILES[name]
        assert not any(profile.get(f'plus_{faculty}') and profile.get(f'minus_{faculty}')
                       for faculty in worker.MIND_FACULTIES)


def test_a_profile_turns_a_faculty_on_or_off_never_both():
    with pytest.raises(ValueError, match='never both'):
        paired.validate_profiles({'full-both': {'plugin': True, 'full': True, 'plus_learning': True,
                                                'minus_learning': True}})


def test_the_pilots_arms_keep_their_names_and_meanings():
    for name in PILOT_ARMS:
        assert name in paired.PROFILES
    for name in SHIPPED_ON_SINCE:
        # The faculty ships on now, so the arm runs what full runs; the pilot's report reads it as the faculty alone
        # when it records that the faculty shipped off, and as no one faculty's when it records nothing.
        treatment, comparator = paired.PROFILES[f'full-plus-{name}'], paired.PROFILES['full']
        assert changed(f'full-plus-{name}', 'full') == {}
        assert arms.subject(treatment, comparator, frozenset()).kind == 'same'
        assert arms.subject(treatment, comparator).kind == 'unrecorded'
        subject = arms.subject(treatment, comparator, arms.EARLIER_OFF)
        assert (subject.kind, subject.key, subject.faculty_on) == ('faculty', name, 'treatment')
    # The status block's pilot profile is still a valid declaration, read as the status block alone where the
    # report records it shipped off. It ships on now, so today the overlay changes nothing.
    declared = paired.validate_profiles(DECLARED_STATUS)['full-plus-status']
    first, second = runs(declared), runs(paired.PROFILES['full'])
    assert {path for path in set(first) | set(second) if first.get(path) != second.get(path)} == set()
    assert arms.subject(declared, paired.PROFILES['full']) == arms.Subject('unrecorded', keys=['status'])
    subject = arms.subject(declared, paired.PROFILES['full'], arms.EARLIER_OFF)
    assert (subject.kind, subject.key, subject.faculty_on) == ('faculty', 'status', 'treatment')

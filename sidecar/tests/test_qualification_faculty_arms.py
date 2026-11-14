"""One paired ablation arm for every faculty the configuration can switch.

The policy is that every faculty is on unless it has shown harm, so the configuration under test
is the mind with every faculty on (``all-on``), and each faculty's claim is ``all-on`` against
``all-on-<faculty>``: the same mind with exactly that one faculty off. A faculty that ships on also keeps its
``full-<faculty>`` ablation against the shipped defaults.
"""
import json

import pytest

from colonymind.config import DEFAULTS
from colonymind.qualification import native_memory_worker as worker
from colonymind.qualification import paired, paired_worker
from test_qualification_paired_runner import fixture  # noqa: F401  (pytest fixture)

CONTEXT = ('status',)
# Every switch colonymind.yaml has for a faculty: the mind's flags (initiative is the one the code does not
# read: its measure is the mind itself, on against off) and the two context switches. The feeling rules are the
# one flag all-on leaves off: they replace the running feelings (their arm adds them, all-on-plus-affect_rules).
SWITCHABLE = tuple(name for name in DEFAULTS['mind']['faculties'] if name not in {'initiative', 'affect_rules'}) \
    + CONTEXT


def effective(profile):
    """What an arm runs with: its mind flags, its drive weights and its two context switches."""
    mind = paired_worker.mind_switches(profile)
    section = worker.mind_section(mind)
    # As the worker's prepare() reads them: the mind switches (all_on) and the context switches together.
    context = worker.context_section({**(mind or {}), **(paired_worker.context_switches(profile) or {})}) or {}
    return {'faculties': dict(section.get('faculties') or {}), 'drives': dict(section.get('drives') or {}),
            'status': context.get('status', DEFAULTS['context']['status'])}


def differences(first, second):
    """The faculties (and context switches) two arms disagree on."""
    changed = {name for name in first['faculties'] if first['faculties'][name] != second['faculties'].get(name)}
    changed |= {name for name in CONTEXT if first[name] != second[name]}
    return changed


def test_every_faculty_the_configuration_names_is_a_mind_flag_the_arm_writes():
    assert set(DEFAULTS['mind']['faculties']) <= set(worker.MIND_FACULTIES)


def test_all_on_is_the_mind_with_every_faculty_on():
    arm = effective(paired.PROFILES['all-on'])
    assert paired.PROFILES['all-on'] == {'plugin': True, 'overlay': {}, 'full': True, 'all_on': True}
    assert arm['faculties'] == {**{name: name != 'affect_rules' for name in worker.MIND_FACULTIES},
                                **{name: True for name in worker.MEANING_FACULTIES}}
    assert arm['status'] is True
    section = worker.mind_section(paired_worker.mind_switches(paired.PROFILES['all-on']))
    assert section['drives'] == DEFAULTS['mind']['drives'] and section['budgets'] == DEFAULTS['mind']['budgets']
    assert section['autonomy'] == 'standard' and section['enabled'] is True


@pytest.mark.parametrize('name', SWITCHABLE)
def test_each_faculty_has_an_all_on_ablation_that_turns_off_only_that_faculty(name):
    base = effective(paired.PROFILES['all-on'])
    arm = paired.PROFILES[f'all-on-{name}']
    assert arm['all_on'] is True and arm['full'] is True and arm['plugin'] is True and arm['overlay'] == {}
    ablated = effective(arm)
    assert differences(base, ablated) == {name}
    assert (ablated['faculties'].get(name, ablated.get(name))) is False
    assert ablated['drives'] == base['drives']


def _affect_route(profile):
    """What the arm's feelings run on, as the mind's own Affect decides it from the arm's two flags."""
    from colonymind.mind.affect import Affect
    faculties = effective(profile)['faculties']
    return Affect(None, store=None, state_on=faculties['affect'], rules_on=faculties['affect_rules']).route()


def test_the_affect_arms_differ_in_what_the_mind_runs_not_only_in_their_flags():
    """The feeling rules replace the running feelings wherever feelings are read, whatever the affect flag says
    (mind/affect.py), so the two flags make three modes. all-on runs the feelings, all-on-affect runs none and
    all-on-plus-affect_rules runs the rules in their place: each contrast against all-on changes one mode, and
    the report reads a pair whose only difference is an affect flag the rules make inert as the same arm."""
    from colonymind.reports import arms
    running, none, rules = (_affect_route(paired.PROFILES[name])
                            for name in ('all-on', 'all-on-affect', 'all-on-plus-affect_rules'))
    assert none == {} and set(rules.values()) == {'rules'} and 'state' in running.values()
    assert 'all-on-affect_rules' not in paired.PROFILES
    base = effective(paired.PROFILES['all-on'])
    assert differences(base, effective(paired.PROFILES['all-on-plus-affect_rules'])) == {'affect_rules'}
    subject = arms.subject(paired.PROFILES['all-on-plus-affect_rules'], paired.PROFILES['all-on'])
    assert (subject.kind, subject.key, subject.faculty_on) == ('faculty', 'affect_rules', 'treatment')
    subject = arms.subject(paired.PROFILES['all-on-affect'], paired.PROFILES['all-on'])
    assert (subject.kind, subject.key, subject.faculty_on) == ('faculty', 'affect', 'comparator')
    inert = {**paired.PROFILES['all-on-plus-affect_rules'], 'minus_affect': True}
    assert _affect_route(inert) == rules
    assert arms.subject(inert, paired.PROFILES['all-on-plus-affect_rules']).kind == 'same'


def test_initiative_is_all_on_with_only_the_mind_off():
    """Initiative's measure is the mind itself: all-on against the same install with the mind switched off. What
    runs without the mind stays as all-on has it (the claim-side learning rules and the memory blocks); the status
    block needs a served mind, so it goes with the mind. The learning rules ship on, so the plain mind-off arm
    keeps them as an install does and measures the same thing."""
    from colonymind.reports import arms
    arm = paired.PROFILES['all-on-initiative']
    assert paired_worker.mind_switches(arm) is None and paired_worker.context_switches(arm) is None

    def environment(profile):
        mind = paired_worker.mind_switches(profile)
        section = worker.mind_section(mind)
        context = worker.context_section({**(mind or {}), **(paired_worker.context_switches(profile) or {})})
        return {**worker.arm_environment(section, context), **profile['overlay']}

    assert environment(arm)['COLONYMIND_LEARNING'] == environment(paired.PROFILES['all-on'])['COLONYMIND_LEARNING']
    subject = arms.subject(paired.PROFILES['all-on'], arm)
    assert (subject.kind, subject.key, subject.faculty_on) == ('faculty', 'initiative', 'treatment')
    # The plain mind-off arm leaves the learning rules to the defaults, which shipped them off before: it measures
    # the same thing only in a report that records they shipped on.
    plain = arms.subject(paired.PROFILES['all-on'], paired.PROFILES['colonymind'], frozenset())
    assert (plain.kind, plain.key, plain.faculty_on) == ('faculty', 'initiative', 'treatment')
    assert arms.subject(paired.PROFILES['all-on'], paired.PROFILES['colonymind']).keys == ['learning']


@pytest.mark.parametrize('name', ('deliberation', 'goals'))
def test_a_faculty_that_ships_on_has_its_full_ablation(name):
    full = effective(paired.PROFILES['full'])
    ablated = effective(paired.PROFILES[f'full-{name}'])
    assert differences(full, ablated) == {name}
    assert (ablated['faculties'].get(name, ablated.get(name))) is False


def test_the_context_switches_reach_the_instance_and_the_environment():
    """The status block and the claim-side learning rules
    are read from the environment (context.status and mind.faculties.learning export them on an install)."""
    assert worker.context_section(None) is None
    assert worker.context_section(paired_worker.context_switches(paired.PROFILES['full'])) is None
    # The section names only what an arm changes from the shipped context (the status block and the memory
    # blocks both ship on).
    assert worker.context_section({'all_on': True}) is None                         # the shipped context
    assert worker.context_section({'all_on': True, 'minus_status': True}) == {'status': False}
    # With the mind off the learning rules and the priors' vector lane stay at their shipped default (on), as an
    # install exports them.
    for arm, status, learning, priors in (('full', 'on', '1', '1'), ('colonymind', 'on', '1', '1'),
                                          ('all-on', 'on', '1', '1'), ('all-on-status', 'off', '1', '1'),
                                          ('all-on-learning', 'on', '0', '1'),
                                          ('all-on-priors_by_meaning', 'on', '1', '0'),
                                          ('full-priors_by_meaning', 'on', '1', '0')):
        profile = paired.PROFILES[arm]
        mind = paired_worker.mind_switches(profile)
        section = worker.mind_section(mind)
        context = worker.context_section({**(mind or {}), **(paired_worker.context_switches(profile) or {})})
        assert worker.arm_environment(section, context) == {'COLONYMIND_CONTEXT_STATUS': status,
                                                             'COLONYMIND_LEARNING': learning,
                                                             'COLONYMIND_PRIORS_BY_MEANING': priors}
    config = worker.instance_config('p-01', {'enabled': False}, {'status': False})
    assert config == {'owner': {'contact_id': 'p-01'}, 'mind': {'enabled': False},
                      'credentials': worker.BENCHMARK_CREDENTIALS,
                      'context': {'status': False}}
    assert 'context' not in worker.instance_config('p-01', {'enabled': False}, None)


def test_the_all_on_arms_plan_as_built_in_profiles(fixture):
    arms = ['all-on', 'all-on-skills', 'all-on-status', 'all-on-plus-affect_rules']
    manifest = paired.plan(fixture.output, native_binding='candidate', evidence_mode='controlled', arms=arms,
                           reference_arm='all-on', **fixture.resources)
    profiles = manifest['comparison']['profiles']
    assert profiles['all-on-status'] == {'name': 'all-on-status', 'plugin': True, 'overlay': {}, 'full': True,
                                         'all_on': True, 'minus_status': True}
    for arm in arms:
        inputs = manifest['pairs'][0]['arms'][arm]['case']['inputs']
        assert paired_worker.arm_profile(inputs) == profiles[arm]
    assert paired_worker.ARM_PROFILE_PROTOCOL == 'paired-arm-profiles-11'
    assert {'all_on', *paired_worker.CONTEXT_SWITCHES} <= set(paired_worker.PROFILE_SWITCHES)


def test_a_context_switch_needs_the_plugin():
    with pytest.raises(ValueError, match='needs the plugin'):
        paired.validate_profiles({'base-blocks': {'plugin': False, 'minus_status': True}})
    declared = paired.validate_profiles({'plain-blocks': {'plugin': True, 'minus_status': True}})
    assert declared['plain-blocks'] == {'plugin': True, 'overlay': {}, 'minus_status': True}
    assert json.dumps(declared['plain-blocks'])

"""Supported release arms and mandatory context fitting."""
import json

import pytest
from fastapi import FastAPI

from colonymind.config import DEFAULTS, apply_environment, load_config
from colonymind.qualification import native_memory_worker as worker
from colonymind.qualification import paired, paired_worker
from test_qualification_paired_runner import fixture  # noqa: F401  (pytest fixture)

# The faculties that shipped off when their pilots ran, each with the arm that turned it on over full.
ADDED = {'full-plus-learning': 'learning', 'full-plus-wishes': 'wishes', 'full-plus-drift_check': 'drift_check'}
RELEASE_ARMS = ['full', *ADDED]


def mind(arm):
    return worker.mind_section(paired_worker.mind_switches(paired.PROFILES[arm]))


def instance(tmp_path, arm, person='p-01'):
    """The disposable instance directory the worker writes for this arm: identity and colonymind.yaml."""
    home = tmp_path / arm
    home.mkdir()
    (home / 'identity.yaml').write_text(json.dumps(worker.benchmark_identity(person)))
    (home / 'colonymind.yaml').write_text(json.dumps(worker.instance_config(person, mind(arm), context(arm))))
    return home


def context(arm):
    """The arm's context section as the worker's prepare() builds it: its mind and context switches together."""
    profile = paired.PROFILES[arm]
    return worker.context_section({**(paired_worker.mind_switches(profile) or {}),
                                   **(paired_worker.context_switches(profile) or {})})


def test_each_release_faculty_has_a_built_in_arm_that_turns_on_only_its_flag():
    full = mind('full')
    for arm, name in ADDED.items():
        assert paired.PROFILES[arm] == {'plugin': True, 'overlay': {}, 'full': True, f'plus_{name}': True}
        assert f'plus_{name}' in paired_worker.MIND_ADDITIONS and name in worker.MIND_FACULTIES
        assert DEFAULTS['mind']['faculties'][name] is True and full['faculties'][name] is True
        section = mind(arm)
        assert section['faculties'] == {**full['faculties'], name: True}
        assert section['drives'] == full['drives'] and section['budgets'] == full['budgets']
        # The model's own words beside a wish's typed capability follow their shipped default (on) in every arm:
        # a checked label, never what a yes gives, so an arm that turns wishes on measures them as they ship.
        assert section['faculties'].get('wishes_named') is True


def test_the_unbudgeted_context_arms_are_removed():
    assert 'full-blocks' not in paired.PROFILES and 'all-on-blocks' not in paired.PROFILES
    assert 'minus_blocks' not in paired_worker.PROFILE_SWITCHES
    with pytest.raises(ValueError):
        paired.validate_profiles({'unbounded': {'plugin': True, 'minus_blocks': True}})


def test_the_instance_config_carries_the_context_section_only_where_an_arm_changes_it():
    section = mind('full')
    filed = {**section, 'digest_hour': worker.FILE_DIGEST_HOUR}
    credentials = worker.BENCHMARK_CREDENTIALS
    assert worker.instance_config('p-01', section, None) == {'owner': {'contact_id': 'p-01'}, 'mind': filed,
                                                            'credentials': credentials}
    assert worker.instance_config('p-01', section, context('all-on-status')) == {
        'owner': {'contact_id': 'p-01'}, 'mind': filed, 'credentials': credentials, 'context': {'status': False}}
    assert section['digest_hour'] == 24, 'the served mind keeps the digest off'
    assert worker.instance_config('p-01', worker.mind_section(None), None)['mind'] == {'enabled': False}


@pytest.mark.parametrize('arm', sorted(name for name, profile in paired.PROFILES.items() if profile['plugin']))
@pytest.mark.parametrize('quiet_hours', [None, '22:00-07:00'])
def test_every_plugin_arms_instance_file_is_a_configuration_the_product_accepts(tmp_path, arm, quiet_hours):
    """Every supported arm writes valid configuration and keeps mandatory context fitting."""
    profile = paired.PROFILES[arm]
    section = worker.mind_section(paired_worker.mind_switches(profile), quiet_hours=quiet_hours)
    (tmp_path / 'colonymind.yaml').write_text(json.dumps(worker.instance_config('p-01', section, context(arm))))
    config = load_config(tmp_path)
    assert config.get('context.blocks') is None
    assert config.get('context.status') is (arm != 'all-on-status')           # the status block ships on
    assert config.get('mind.enabled') is section['enabled']
    if section['enabled']:
        assert config.get('mind.faculties') == {**DEFAULTS['mind']['faculties'], **section['faculties']}
        assert config.get('mind.quiet_hours') == (quiet_hours or '')


def test_the_served_host_fits_every_supported_context_arm(tmp_path, monkeypatch):
    from colonymind.api.routers import host
    from colonymind.api.schemas.host import ContextSection
    from colonymind.memory import blocks
    sections = [ContextSection(id=f'colonymind-{name}', title=name, body=(name + ' ') * 400, priority=1)
                for name in ('memory', 'commitments', 'digest')]
    try:
        for arm in ('full', 'all-on-status'):
            monkeypatch.setenv('COLONYMIND_HOME', str(instance(tmp_path, arm)))
            blocks.reset_settings()
            fitted, report = host._fit_sections(sections, budget=2000, where='test')
            assert report['budget'] == 2000
            assert sum(blocks.cost(s.title, s.body) for s in fitted) <= 2000
    finally:
        blocks.reset_settings()


@pytest.mark.parametrize('arm', ['colonymind', *RELEASE_ARMS, 'all-on-learning'])
def test_the_learning_flag_reaches_the_claim_projection_as_an_install_exports_it(tmp_path, monkeypatch, arm):
    """The claim projection reads learning from ``COLONYMIND_LEARNING``, which ``apply_environment`` derives
    from ``mind.faculties.learning`` on an install; the worker gives each arm the same value from its own
    mind section (the shipped default, on, where the mind is off), so every arm but the learning rules' own
    ablation projects claims under them, as an install does."""
    from colonymind.memory import learning
    section = worker.mind_section(paired_worker.mind_switches(paired.PROFILES[arm]))
    values = worker.arm_environment(section, context(arm))
    on = arm != 'all-on-learning'
    assert values == {'COLONYMIND_LEARNING': '1' if on else '0', 'COLONYMIND_CONTEXT_STATUS': 'on',
                      'COLONYMIND_PRIORS_BY_MEANING': '1'}
    monkeypatch.setenv('COLONYMIND_HOME', str(tmp_path / 'unused'))
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'identity.yaml').write_text(json.dumps(worker.benchmark_identity('p-01')))
    (home / 'colonymind.yaml').write_text(json.dumps(worker.instance_config('p-01', section, None)))
    exported = apply_environment(load_config(home), environ={})
    assert exported['COLONYMIND_LEARNING'] == values['COLONYMIND_LEARNING']
    assert exported['COLONYMIND_PRIORS_BY_MEANING'] == values['COLONYMIND_PRIORS_BY_MEANING']
    assert learning.enabled(values) is on


@pytest.mark.parametrize('arm', [*RELEASE_ARMS, 'all-on-learning', 'all-on-wishes', 'all-on-drift_check'])
def test_the_served_mind_reads_each_arms_faculties(tmp_path, monkeypatch, arm):
    """The Mind the worker serves for the arm is built from its section: its lessons follow the learning rules,
    its wishes store and drift check are enabled, in every arm but the one that ablates that faculty."""
    monkeypatch.setenv('COLONYMIND_OWNER_CONTACT_ID', 'p-01')
    with worker.serve_mind(FastAPI(), tmp_path / 'state', 'p-01', mind(arm)) as served:
        assert served.faculties['learning'] is (arm != 'all-on-learning')
        assert served.lessons.learning is (arm != 'all-on-learning')
        assert served.faculties['wishes'] is (arm != 'all-on-wishes')
        assert served.wishes.enabled is (arm != 'all-on-wishes')
        assert served.faculties['drift_check'] is (arm != 'all-on-drift_check')
        assert served.drift.enabled is (arm != 'all-on-drift_check')
        assert ('drift' in served.state()) is (arm != 'all-on-drift_check')


def test_a_plan_names_the_release_arms_and_needs_a_worker_that_applies_them(fixture, monkeypatch):
    manifest = paired.plan(fixture.output, native_binding='candidate', evidence_mode='controlled',
                           arms=RELEASE_ARMS, reference_arm='full', **fixture.resources)
    profiles = manifest['comparison']['profiles']
    assert list(profiles) == RELEASE_ARMS
    for arm in RELEASE_ARMS:
        assert profiles[arm] == {'name': arm, **paired.PROFILES[arm]}
        case = manifest['pairs'][0]['arms'][arm]['case']
        assert paired_worker.arm_profile(case['inputs']) == profiles[arm]
    assert paired_worker.ARM_PROFILE_PROTOCOL == 'paired-arm-profiles-11'
    # An image built before these switches (7), with either development set of 8, or before the meaning ablations
    # (9), would run an arm it cannot apply as full: the plan refuses it.
    from colonymind.qualification import paired_container
    original = paired_container.configuration
    refused = None

    def older(*args, **kwargs):
        supplied, recipe = original(*args, **kwargs)
        recipe['container_payload'] = {**recipe['container_payload'], 'arm_profiles': refused}
        return supplied, recipe
    monkeypatch.setattr(paired_container, 'configuration', older)
    for protocol in ('paired-arm-profiles-7', 'paired-arm-profiles-8', 'paired-arm-profiles-9'):
        refused = protocol
        for arm in RELEASE_ARMS[1:]:
            with pytest.raises(ValueError, match='worker applies profiles'):
                paired.plan(fixture.output / protocol / arm, native_binding='candidate', evidence_mode='controlled',
                            arms=['full', arm], **fixture.resources)


def test_a_declared_profile_may_use_the_context_switch_only_with_the_plugin():
    profiles = paired.validate_profiles({'colonymind-status': {'plugin': True, 'minus_status': True}})
    assert profiles['colonymind-status'] == {'plugin': True, 'overlay': {}, 'minus_status': True}
    assert paired_worker.mind_switches(profiles['colonymind-status']) is None
    with pytest.raises(ValueError, match='plugin'):
        paired.validate_profiles({'base-status': {'plugin': False, 'minus_status': True}})
    with pytest.raises(ValueError, match='redefine'):
        paired.validate_profiles({'full-plus-learning': {'plugin': True, 'full': True}})

"""Configured learning interventions reach the disposable instance and its body clock."""
from copy import deepcopy
from datetime import timedelta
import json
from types import SimpleNamespace

import pytest

from colonymind.qualification import native_memory_worker as worker
from colonymind.qualification import paired, paired_worker
from test_plasticity_wiring_registration import declaration
from test_qualification_paired_runner import fixture  # noqa: F401


LEARNERS = {'probe': declaration(), 'other': declaration(area='recall')}


def configured(profile, plasticity):
    switches = paired_worker.mind_switches(profile)
    return worker.instance_config('owner', worker.mind_section(switches), worker.context_section(switches),
                                  profile=profile, plasticity=plasticity)


@pytest.mark.parametrize('arm', ['plasticity.probe', 'plasticity.probe.teacher.owner_label', 'all-on-plasticity'])
def test_worker_applies_declared_learner_and_teacher_interventions_without_mutating_inputs(arm):
    plasticity = {'learners': deepcopy(LEARNERS)}
    before = deepcopy(plasticity)
    profiles = paired.validate_profiles(None, learners=LEARNERS)
    profile = profiles[arm]
    frozen = {'name': arm, **profile}
    assert paired_worker.arm_profile({'profile': frozen, 'plasticity': plasticity}) == frozen
    base = configured(profiles['all-on'], plasticity)
    expected = deepcopy(base)
    if arm == 'all-on-plasticity':
        expected['mind']['faculties']['plasticity'] = False
    elif arm == 'plasticity.probe':
        expected['plasticity']['learners']['probe']['enabled'] = False
    else:
        expected['plasticity']['learners']['probe']['disabled_teachers'] = ['owner_label']
    assert configured(profile, plasticity) == expected
    assert plasticity == before


def test_profile_validation_preserves_declared_interventions_and_refuses_unknown_learners():
    profile = paired.plasticity_profiles(LEARNERS)['plasticity.probe']
    assert paired.validate_profiles({'probe-at-base': profile}, learners=LEARNERS)['probe-at-base'] == profile
    with pytest.raises(ValueError, match='Unknown plasticity'):
        paired.validate_profiles({'probe-at-base': profile})
    with pytest.raises(ValueError, match='Unknown plasticity'):
        paired_worker.arm_profile({'profile': profile})
    with pytest.raises(ValueError, match='all-on'):
        paired.validate_profiles({'probe-at-base': {**profile, 'plugin': False}}, learners=LEARNERS)


def test_plan_freezes_the_same_learner_declarations_for_every_arm(fixture):
    plasticity = {'learners': deepcopy(LEARNERS)}
    manifest = paired.plan(fixture.output, native_binding='candidate', evidence_mode='controlled',
                           arms=['all-on', 'plasticity.probe.teacher.owner_label'], reference_arm='all-on',
                           plasticity=plasticity, **fixture.resources)
    assert manifest['options']['plasticity']['learners'] == LEARNERS
    repeated, *_ = paired.prepare(output=fixture.output, **manifest['options'], label=manifest['label'],
                                   evidence_mode=manifest['evidence_mode'], **fixture.resources)
    assert repeated == manifest
    for pair in manifest['pairs']:
        for arm in pair['arms'].values():
            inputs = arm['case']['inputs']
            assert inputs['plasticity']['learners'] == LEARNERS
            assert paired_worker.arm_profile(inputs) == inputs['profile']


@pytest.mark.parametrize('arm', ['all-on', 'full-plasticity', 'all-on-plasticity'])
async def test_body_clock_drives_the_configured_learners_at_the_nightly_boundary(tmp_path, monkeypatch, arm):
    from colonymind.mind.consolidate import NIGHT_MINUTE
    from colonymind.plasticity.synthetic import EPOCH

    now = EPOCH
    monkeypatch.setattr(worker, 'mind_clock', lambda: now)
    profile = paired.PROFILES[arm]
    (tmp_path / 'colonymind.yaml').write_text(json.dumps(configured(profile, {'learners': LEARNERS})))

    async def original_tick(**kwargs):
        return {'mind': 'ticked', **kwargs}

    mind = SimpleNamespace(tick=original_tick, clock=lambda: now, tz='UTC',
                           consolidation=SimpleNamespace(boundary_minute=lambda: NIGHT_MINUTE))
    nights = []
    with worker.plasticity_nights(mind, tmp_path, tmp_path / 'data') as runtime:
        monkeypatch.setattr(runtime, 'run', lambda: nights.append(now))
        assert await mind.tick(force=True) == {'mind': 'ticked', 'force': True}
        assert nights == []
        # Fixture assumption: advancing a day crosses the configured nightly boundary.
        now += timedelta(days=1)
        await mind.tick(force=True)
        await mind.tick(force=True)
        assert nights == ([now] if arm == 'all-on' else [])
    assert mind.tick is original_tick

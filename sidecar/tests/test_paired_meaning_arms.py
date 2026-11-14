"""Ablation arms for the two uses of the embedder beyond recall: contradictions by meaning and priors by meaning.

The policy is that every faculty is on unless it has shown harm, and each is measured by all-on against the same
mind with exactly that faculty off. Both faculties have the sidecar's own switch (``mind.faculties``, the claim
projection's through ``COLONYMIND_PRIORS_BY_MEANING``), so an arm is one change of configuration, as on an install:
the arm writes the faculty's flag into its mind section, the served night reads it there, and the claim worker's
projection reads the flag the worker exports from that section, the way ``config.apply_environment`` exports it.
The harness applies no ablation of its own: one mechanism switches each faculty, in a benchmark and on an install.
"""
import pytest
from fastapi import FastAPI

import colonymind.vector as vector
from colonymind import config
from colonymind.beliefs import source_projection
from colonymind.qualification import native_memory_worker as worker
from colonymind.qualification import paired, paired_worker
from colonymind.reports import arms, registry
from test_mind_contradiction_candidates import EMPLOYER, WORKS_AT, KeyEmbedder, relation_calls
from test_mind_consolidate import OWNER, Fixture
from test_qualification_paired_runner import fixture  # noqa: F401  (pytest fixture)

MEANING = ('contradictions_by_meaning', 'priors_by_meaning')


def section(arm):
    return worker.mind_section(paired_worker.mind_switches(paired.PROFILES[arm]))


def test_the_worker_switches_the_sidecars_own_meaning_faculties():
    """The worker names the faculties the sidecar reads, and no ablation of its own is left beside them."""
    assert worker.MEANING_FACULTIES == config.MEANING_FACULTIES == MEANING
    for retired in ('ExactKeysOnly', 'apply_meaning_ablations'):
        assert not hasattr(worker, retired)
    assert not hasattr(paired_worker, 'priors_by_meaning')


@pytest.mark.parametrize('name', MEANING)
def test_each_meaning_faculty_has_its_all_on_and_full_ablation(name):
    assert name in worker.MEANING_FACULTIES and name in paired.ALL_ON_FACULTIES
    assert f'minus_{name}' in paired_worker.PROFILE_SWITCHES
    assert paired.PROFILES[f'all-on-{name}'] == {'plugin': True, 'overlay': {}, 'full': True, 'all_on': True,
                                                 f'minus_{name}': True}
    assert paired.PROFILES[f'full-{name}'] == {'plugin': True, 'overlay': {}, 'full': True, f'minus_{name}': True}
    for base in ('all-on', 'full'):
        on, off = section(base)['faculties'], section(f'{base}-{name}')['faculties']
        assert on[name] is True and off[name] is False
        assert {key: value for key, value in on.items() if key != name} == \
            {key: value for key, value in off.items() if key != name}


@pytest.mark.parametrize('name', MEANING)
def test_the_reports_read_each_arm_as_that_faculty_alone(name):
    subject = arms.subject(paired.PROFILES['all-on'], paired.PROFILES[f'all-on-{name}'])
    assert (subject.kind, subject.key, subject.faculty_on) == ('faculty', name, 'treatment')
    # The arm is described in the faculty report's own words for the faculty.
    assert arms.describe(paired.PROFILES[f'all-on-{name}']) == \
        f"the mind with every faculty on, without {registry.BY_KEY[name].noun}"


def test_the_arms_that_do_not_run_the_full_mind_write_no_meaning_flag():
    """The initiative-only arm and the plain plugin arm run both faculties wherever they apply (the claim worker's
    priors run without the mind): no flag is written, as for any faculty such an arm does not switch."""
    assert set(section('colonymind-initiative')['faculties']).isdisjoint(MEANING)
    assert section('colonymind') == {'enabled': False}


@pytest.fixture
def night(tmp_path, monkeypatch):
    """A night whose mind is built with ``arm``'s own contradictions-by-meaning flag, over the sidecar's embedder."""
    monkeypatch.setenv('COLONYMIND_OWNER_CONTACT_ID', OWNER)
    monkeypatch.setattr(vector, '_pipeline', KeyEmbedder({WORKS_AT: 'job', EMPLOYER: 'job'}))
    from colonymind.mind.consolidate import TASK_DIGEST
    built = []

    def build(arm):
        flag = section(arm)['faculties']['contradictions_by_meaning']
        fixture = Fixture(tmp_path / arm, config={'faculties': {'contradictions_by_meaning': flag}})
        fixture.router.answers[TASK_DIGEST] = None
        built.append(fixture)
        return fixture
    yield build
    for fixture in built:
        fixture.store.close()


@pytest.mark.parametrize('arm, pairs', [('all-on', 1), ('all-on-contradictions_by_meaning', 0)])
async def test_the_arms_flag_leaves_the_night_its_exact_keys(night, arm, pairs):
    """With the sidecar's embedder the night pairs "works at" with "employer"; with the arm's flag off the night
    never embeds a key and pairs only claims under one key, as a night without an embedder does."""
    fx = night(arm)
    await fx.fact('t-1', OWNER, 's-1', 'I work at Acme.', 'Acme', predicate='works_at')
    fx.shift(days=1)
    await fx.fact('t-2', OWNER, 's-2', 'My employer is Globex.', 'Globex', predicate='employer')
    result = await fx.mind.consolidate()
    assert len(relation_calls(fx)) == pairs
    assert result['counts'].get('keys_embedded', 0) == (2 if pairs else 0)


async def test_the_ablation_still_decides_claims_under_one_key(night):
    fx = night('all-on-contradictions_by_meaning')
    await fx.fact('t-1', OWNER, 's-1', 'My office is in room 4.', 'room 4')
    fx.shift(days=1)
    await fx.fact('t-2', OWNER, 's-2', 'My office is in room 7.', 'room 7')
    await fx.mind.consolidate()
    assert len(relation_calls(fx)) == 1


@pytest.mark.parametrize('arm, on', [('all-on', True), ('all-on-contradictions_by_meaning', False),
                                     ('full-contradictions_by_meaning', False), ('all-on-priors_by_meaning', True)])
def test_the_worker_serves_each_arm_its_night_from_its_section(tmp_path, monkeypatch, arm, on):
    """The served night is the sidecar's own over the sidecar's embedder, switched by the arm's section alone."""
    monkeypatch.setenv('COLONYMIND_OWNER_CONTACT_ID', 'p-01')
    with worker.serve_mind(FastAPI(), tmp_path / 'state', 'p-01', section(arm)) as served:
        assert served.consolidation.embedder is None      # the sidecar's own pipeline, as on an install
        assert served.consolidation.faculties['contradictions_by_meaning'] is on


@pytest.mark.parametrize('arm, lane', [('all-on', 'on'), ('full', 'on'), ('colonymind', 'on'),
                                       ('colonymind-initiative', 'on'), ('all-on-initiative', 'on'),
                                       ('all-on-priors_by_meaning', 'off'), ('full-priors_by_meaning', 'off'),
                                       ('all-on-contradictions_by_meaning', 'on')])
def test_the_claim_workers_prior_lane_follows_the_flag_the_arm_exports(arm, lane):
    """The worker exports the arm's flag as an install does (``COLONYMIND_PRIORS_BY_MEANING``, on where the arm
    names none), and the claim projection reads it there: the lane is the sidecar's own switch."""
    profile = paired.PROFILES[arm]
    mind = paired_worker.mind_switches(profile)
    context = worker.context_section({**(mind or {}), **(paired_worker.context_switches(profile) or {})})
    environment = {**worker.arm_environment(worker.mind_section(mind), context), **profile['overlay']}
    assert environment[source_projection.PRIORS_BY_MEANING_FLAG] == ('1' if lane == 'on' else '0')
    assert source_projection.priors_by_meaning(environment) is (lane == 'on')


@pytest.mark.parametrize('arm', ['all-on', 'colonymind', 'all-on-priors_by_meaning', 'full-priors_by_meaning',
                                 'all-on-contradictions_by_meaning'])
def test_the_worker_exports_the_flag_an_install_exports_from_the_arms_own_file(tmp_path, monkeypatch, arm):
    """The disposable ``colonymind.yaml`` the worker writes, loaded and exported as an install does, gives the claim
    projection the same flag the worker sets for the served host."""
    import json
    monkeypatch.setenv('COLONYMIND_HOME', str(tmp_path / 'unused'))
    mind = paired_worker.mind_switches(paired.PROFILES[arm])
    section_value = worker.mind_section(mind)
    context = worker.context_section({**(mind or {}), **(paired_worker.context_switches(paired.PROFILES[arm]) or {})})
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'identity.yaml').write_text(json.dumps(worker.benchmark_identity('p-01')))
    (home / 'colonymind.yaml').write_text(json.dumps(worker.instance_config('p-01', section_value, context)))
    exported = config.apply_environment(config.load_config(home), environ={})
    flag = source_projection.PRIORS_BY_MEANING_FLAG
    assert exported[flag] == worker.arm_environment(section_value, context)[flag]


def test_the_meaning_arms_need_an_image_whose_worker_applies_them(fixture, monkeypatch):
    """An image with the previous arm-profile protocol would run either arm as all-on: the plan refuses it."""
    from colonymind.qualification import paired_container
    original = paired_container.configuration

    def older(*args, **kwargs):
        supplied, recipe = original(*args, **kwargs)
        recipe['container_payload'] = {**recipe['container_payload'], 'arm_profiles': 'paired-arm-profiles-9'}
        return supplied, recipe
    monkeypatch.setattr(paired_container, 'configuration', older)
    for name in MEANING:
        with pytest.raises(ValueError, match='worker applies profiles'):
            paired.plan(fixture.output / name, native_binding='candidate', evidence_mode='controlled',
                        arms=['all-on', f'all-on-{name}'], **fixture.resources)
    monkeypatch.setattr(paired_container, 'configuration', original)
    manifest = paired.plan(fixture.output / 'current', native_binding='candidate', evidence_mode='controlled',
                           arms=['all-on', *(f'all-on-{name}' for name in MEANING)], **fixture.resources)
    assert list(manifest['comparison']['profiles']) == ['all-on', *(f'all-on-{name}' for name in MEANING)]

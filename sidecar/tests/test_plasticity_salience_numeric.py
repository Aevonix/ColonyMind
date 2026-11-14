"""Salience's private features, frozen anchor and globally lowering proxy updates."""
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from colonymind.mind.rank import Candidate
from colonymind.plasticity.builds import TrainingRow
from colonymind.plasticity.learners.salience import DECLARATION, OBJECTIVE, Exploration, SalienceLearner, frame
from colonymind.plasticity.objectives import check

NOW = datetime(2026, 9, 30, 12, tzinfo=timezone.utc)
KEY = b'salience fixture key'
WORDS = 'A private fixture sentence about a garden.'


def sample(index=0, **changes):
    values = dict(type='research', drive='curiosity', kind='task', title=WORDS,
                  dedup_key='private-' + str(index), text=WORDS, topic=WORDS, salience=.8, cost=.1)
    values.update(changes)
    return frame(Candidate(**values), NOW + timedelta(hours=index), {}, KEY)


def row(index, label, *, push=False, state=None):
    teacher = 'behavioural_proxy' if push else 'owner_label'
    return TrainingRow(str(index), 'useful', json.dumps(state or sample(index)), label,
                       teacher, push, 'margin' if push else 'logistic', 1., 'owner')


def test_declaration_and_private_complete_frame():
    check(OBJECTIVE)
    assert DECLARATION['kind'] == 'numeric-online'
    assert DECLARATION['mode'] == 'rebuild' and DECLARATION['personal_data'] is True
    assert DECLARATION['exploration_share'] == 0.
    candidate = Candidate(type=WORDS, drive=WORDS, kind=WORDS, title=WORDS, dedup_key=WORDS,
                          concern_kind=WORDS, source_type=WORDS, salience=.8, cost=.2, open_ended=True)
    value = frame(candidate, NOW, {'drives': {WORDS: .3}, 'base': {WORDS: .6},
        'affect': SimpleNamespace(worry=.2, curiosity=.4),
        'last_owner_turn': NOW - timedelta(hours=90)}, KEY)
    assert WORDS not in json.dumps(value)
    assert len(value['type']) == 64 and value['drive'] == value['kind'] == 'other'
    assert value['concern_kind'] == value['source_type'] == 'other'
    assert value['satiation'] == pytest.approx(.5)
    assert value['hours'] == 72. and value['worry'] == .2 and value['curiosity'] == .4
    assert value['open_ended'] == 1. and value['salience'] == .8 and value['cost'] == .2


def test_revision_anchor_exact_factor_and_extremes():
    learner = SalienceLearner('salience')
    state = learner.fit(learner.base(), [row(i, int(i == 0)) for i in range(4)], objective=OBJECTIVE, seed=0)
    assert learner.anchor(state) == .25
    state['W'][:] = 0.
    state['push'][:] = 0.
    assert learner.factor(state, sample()) == 1.
    assert learner.predict(state, [sample()])[0, 0] == .25
    state['W'][0, -1] = 100.
    assert learner.factor(state, sample()) == 1.5
    state['W'][0, -1] = -100.
    assert learner.factor(state, sample()) == .5


def test_types_are_top_twelve_keyed_and_revision_owned():
    learner = SalienceLearner('salience')
    rows = [row(i, i % 2, state=sample(i, type='private type ' + str(i))) for i in range(14)]
    rows += [replace(rows[-1], ref='duplicate')]
    state = learner.fit(learner.base(), rows, objective=OBJECTIVE, seed=0)
    types = learner.types(state)
    assert len(types) == 12 and sample(13, type='private type 13')['type'] in types
    assert all(len(value) == 64 for value in types)
    assert b'private type' not in state['types'].tobytes()
    assert learner.types(learner.base()) == []


def test_type_vocabulary_counts_proxy_and_dismissal_rows():
    from collections import Counter
    learner = SalienceLearner('salience')
    rows = [row(i, i % 2, state=sample(i, type='type-' + str(i))) for i in range(14)]
    rows += [row(100 + i, 1, push=True, state=sample(100 + i, type='ignored type')) for i in range(20)]
    rows += [row(200 + i, 0, state={**sample(200 + i, type='dismissed type'), 'verdict': 'dismissed'})
             for i in range(15)]
    state = learner.fit(learner.base(), rows, objective=OBJECTIVE, seed=0)
    counts = Counter(json.loads(value.state)['type'] for value in rows)
    assert learner.types(state) == sorted(counts, key=lambda value: (-counts[value], value))[:12]
    assert learner.budget_used(state) <= learner.bound


def test_relayout_never_raises_evicted_or_unknown_type_factors():
    learner = SalienceLearner('salience')
    values = [sample(i, type='type-' + str(i)) for i in range(14)]
    state = learner.base()
    state['types'] = np.frombuffer(json.dumps([value['type'] for value in values[:12]]).encode(),
                                  dtype=np.uint8).copy()
    state['W'][0, 2:15] = np.linspace(-.02, .03, 13)
    state['W'][0, 0] = .05
    state['W'][0, -1] = .01
    state['push'][0] = -.02
    types = [value['type'] for value in reversed(values[1:13])]
    updated = learner.relayout(state, types)
    assert learner.types(updated) == types
    assert learner.budget_used(updated) <= learner.bound
    assert all(learner.factor(updated, value) <= learner.factor(state, value) for value in values)


def test_relayout_that_would_break_fence_uses_global_lowering_fallback():
    learner = SalienceLearner('salience')
    state = learner.base()
    state['W'][0, 14] = -.5  # Other type; expanding it twelve times would exceed the fence.
    values = [sample(i, type='type-' + str(i)) for i in range(13)]
    updated = learner.relayout(state, [value['type'] for value in values[:12]])
    assert np.all(updated['W'] == 0.)
    assert updated['push'][0] == -learner.bound
    assert learner.budget_used(updated) <= learner.bound
    assert all(learner.factor(updated, value) <= learner.factor(state, value) for value in values)


def test_proxy_updates_never_raise_any_input_in_mixed_state():
    learner = SalienceLearner('salience', rate=.1, clip=.5)
    state = learner.fit(learner.base(), [row(i, i % 2) for i in range(100)], objective=OBJECTIVE, seed=0)
    probes = [sample(i, salience=(i % 11) / 10, cost=(i % 7) / 7) for i in range(168)]
    before = learner.predict(state, probes)
    updated = learner.push_step(state, state, weight=1.)
    assert np.all(learner.predict(updated, probes) <= before)
    assert learner.budget_used(updated) <= learner.bound
    ignored = learner.fit(learner.base(), [row(i, 1, push=True) for i in range(100)],
                          objective=OBJECTIVE, seed=0)
    assert all(learner.factor(ignored, value) <= 1. for value in probes)
    assert np.all(ignored['W'] == 0.)


def test_adding_proxy_or_dismissal_cannot_raise_mixed_history_factors():
    learner = SalienceLearner('salience', rate=.01)
    rows = [row(i, int(i % 3 != 0)) for i in range(100)]
    baseline = learner.fit(learner.base(), rows, objective=OBJECTIVE, seed=0)
    private_type = sample(500, type=WORDS)
    dismissed = {**private_type, 'verdict': 'dismissed'}
    extra = [row(500, 1, push=True, state=private_type)]
    proxy = learner.fit(learner.base(), rows + extra, objective=OBJECTIVE, seed=0)
    assert learner.anchor(proxy) == learner.anchor(baseline)
    np.testing.assert_array_equal(proxy['W'], baseline['W'])
    extra.append(row(501, 0, state=dismissed))
    pushed = learner.fit(learner.base(), rows + extra, objective=OBJECTIVE, seed=0)
    assert learner.anchor(pushed) == 66 / 101
    assert learner.types(pushed) == [sample()['type'], private_type['type']]
    assert all(learner.factor(pushed, probe) <= learner.factor(baseline, probe)
               for probe in [sample(i) for i in range(168)] + [private_type])


def test_hourly_pattern_learns_while_shuffled_targets_do_not():
    learner = SalienceLearner('salience')
    train, test = [], []
    for i in range(12 * 168):
        value = sample(i)
        at = datetime.fromisoformat(value['at'])
        label = int(np.sin(2 * np.pi * (at.weekday() * 24 + at.hour) / 168) > 0)
        (test if (i // 168) % 4 == 0 else train).append(row(i, label, state=value))
    state = learner.fit(learner.base(), train, objective=OBJECTIVE, seed=0)
    truth = np.array([value.label for value in test])
    loss = (learner.predict(state, [value.state for value in test])[:, 0] - truth) ** 2
    assert loss.mean() < .25
    shuffled_losses = []
    for seed in range(7, 15):
        order = np.random.default_rng(seed).permutation(len(train))
        shuffled = [replace(value, label=train[int(index)].label) for value, index in zip(train, order)]
        control = learner.fit(learner.base(), shuffled, objective=OBJECTIVE, seed=seed)
        shuffled_losses.append((learner.predict(control, [value.state for value in test])[:, 0] - truth) ** 2)
    # Same uncertainty adjustment as the nightly gate's eight shuffled fits.
    losses = np.asarray(shuffled_losses)
    gain = .25 - losses.mean(axis=0)
    lower = (gain.mean() - 1.959963984540054 * gain.std(ddof=1) / np.sqrt(len(gain))
             - 1.959963984540054 * losses.mean(axis=1).std(ddof=1) / np.sqrt(len(losses)))
    assert lower <= 0


def test_owner_dismissal_uses_the_same_lowering_channel():
    learner = SalienceLearner('salience')
    value = sample()
    value['verdict'] = 'dismissed'
    state = learner.fit(learner.base(), [row(0, 0, state=value)], objective=OBJECTIVE, seed=0)
    assert np.all(state['W'] == 0.) and state['push'][0] < 0
    assert learner.anchor(state) == 1e-6


@pytest.mark.parametrize('original,new', [(.5, .4), (.99, 1e-6), (.999999, .999998),
                                        (.000002, .000001), (.8, .000001)])
def test_anchor_rebase_is_lowering_across_feature_box_at_full_fence(original, new):
    learner = SalienceLearner('salience')
    state = learner.base()
    rng = np.random.default_rng(302)
    radii = np.array([1. if 'time' in value or 'const' in value else 3.
                      for value in learner.metadata['features']])
    state['W'][0] = rng.normal(size=state['W'].shape[1])
    state['W'] *= np.float32(2. / learner.budget_used(state))
    while learner.budget_used(state) > 2.:
        state['W'] *= np.float32(1. - np.finfo(np.float32).eps)
    state['p0'][0] = original
    updated = learner.rebase(state, new)
    assert learner.anchor(updated) == new
    assert learner.budget_used(updated) <= 2.
    assert updated['push'][0] <= 0
    points = rng.uniform(-1., 1., size=(1000, len(radii))) * radii
    points = np.concatenate([points, np.sign(state['W']) * radii, -np.sign(state['W']) * radii])
    points[:, -1] = 1.

    def factors(value):
        residual = points @ learner.revision(value).weights[0].astype(np.float64)
        exponential = np.exp(residual)
        probability = learner.anchor(value)
        return exponential / (1. - probability + probability * exponential)

    assert np.all(factors(updated) <= factors(state))
    updated['W'][:] = 0.
    updated['push'][:] = 0.
    assert learner.factor(updated, sample()) == 1.


def test_exploration_uses_forced_runs_and_records_probability():
    assert not any(Exploration(KEY, learner='salience', share=0.).choose(str(i)).forced for i in range(100))
    explorer = Exploration(KEY, learner='salience', share=.1)
    choices = [explorer.choose(str(i)) for i in range(10000)]
    assert abs(sum(choice.forced for choice in choices) - 1000) <= 2.576 * np.sqrt(900)
    assert all(choice.probability == .1 for choice in choices)
    for bad in (-.1, .21, True, float('nan')):
        with pytest.raises(ValueError):
            Exploration(KEY, learner='salience', share=bad)


def test_rebuild_matches_clean_fit_after_reference_erasure():
    learner = SalienceLearner('salience')
    rows = [row(i, i % 2) for i in range(50)]
    previous = learner.fit(learner.base(), rows, objective=OBJECTIVE, seed=0)
    # Rebuild mode supplies the base, and surviving references re-render identically.
    rebuilt = learner.fit(learner.base(), rows[1:], objective=OBJECTIVE, seed=0)
    clean = SalienceLearner('salience').fit(learner.base(), rows[1:], objective=OBJECTIVE, seed=0)
    assert previous['p0'][0] != rebuilt['p0'][0]
    for key in rebuilt:
        np.testing.assert_allclose(rebuilt[key], clean[key], rtol=1e-6)

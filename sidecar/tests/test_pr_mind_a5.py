"""Owner feedback reaches the keys consumed by the mind and idea scores."""
from types import SimpleNamespace

import pytest

from colonymind.api.routers import host
from colonymind.feedback import TypeFeedbackStore
from colonymind.mind.rank import Candidate, score
from test_grounded_ideas import env, generate, mailbox
from test_initiative_feedback import _FakeInitiativeStore


@pytest.mark.parametrize('action,outcome', [('dismissed', 'dismissed'), ('snoozed', 'snoozed'),
                                            ('acknowledged', 'acknowledged')])
async def test_disposal_teaches_type_and_topic(monkeypatch, action, outcome):
    row = SimpleNamespace(id='feedback-row', type='outreach_finding', drive='curiosity', kind='message',
                          entity_id='owner', context={'topic_slug': 'gardening'}, status='pending', job_id=None)
    feedback = TypeFeedbackStore(db_path=None)
    monkeypatch.setattr(host, '_initiative_store', _FakeInitiativeStore(row))
    monkeypatch.setattr(host, '_feedback_store', feedback)
    await host.respond_to_initiative(row.id, action=action, details=None)
    expected = TypeFeedbackStore(db_path=None)
    expected.record('expected', outcome, source=row.id)
    for key in ('outreach_finding:curiosity', 'outreach_topic:gardening'):
        assert feedback.multiplier(key) == expected.multiplier('expected')
    assert {entry['itype'] for entry in feedback.snapshot()} == {
        'outreach_finding:curiosity', 'outreach_topic:gardening'}
    assert feedback.multiplier('outreach_finding') == feedback.multiplier('reach_out:owner') == 1


async def test_owner_thumbs_and_marked_dismissal_reach_idea_ranker(env, monkeypatch):
    feedback = TypeFeedbackStore(db_path=None)
    env.mind.feedback = feedback
    monkeypatch.setattr(host, '_feedback_store', feedback)
    row = (await generate(env, mailbox()[:1]))[0]
    candidate = Candidate(type='grounded_offer', drive='duty', kind='task', title='Offer', dedup_key='offer')
    neutral = score(candidate)
    await host.initiative_feedback(row.id, request=None, body={'direction': 'up', 'surface': 'ideas'})
    assert score(candidate, feedback=feedback) > neutral
    await host.respond_to_initiative(row.id, action='dismissed', details=None, request=None,
                                     mode=None, item_ids=None, itemIds=None, origin=None)
    assert env.store.get(row.id).status == 'cancelled'
    assert score(candidate, feedback=feedback) < neutral
    assert feedback.multiplier('grounded_offer') == 1


async def test_publish_scores_with_feedback_and_ignores_legacy_bare_key(env, monkeypatch):
    feedback = TypeFeedbackStore(db_path=None)
    feedback.record('grounded_offer', 'preferred', source='legacy')
    feedback.record('grounded_offer:duty', 'dismissed', source='new')
    env.mind.feedback = feedback
    monkeypatch.setattr(host, '_feedback_store', feedback)
    observations = mailbox()[:1]
    observations[0]['result']['data']['snippet'] = 'Invoice INV-204 is $480; order PO-88 lists $450.'
    row = (await generate(env, observations))[0]
    assert row.priority == pytest.approx(0.65 * 0.9 * 0.85)

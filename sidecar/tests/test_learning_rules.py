"""The learning rules on the claim store (memory/learning.py, ``mind.faculties.learning``).

Everything runs the real ledger, projection and recall readers with a controlled extractor. The rules are
compared with the same flows with the faculty off: repeats are witnesses, not rows; a correction retracts
the value, not the one witness the model was shown; a restated preference cannot crowd itself out of
recall; a preference retires only on its owner's words or verdict; every refusal is logged without text;
erasure removes a witness with its message.
"""
import json
import sqlite3
from contextlib import closing

import pytest

from colonymind.beliefs.source_projection import SourceClaimProjection
from colonymind.memory import learning
from colonymind.turns import TurnIdempotencyLedger
from test_source_claim_projection import prepared
from test_source_claim_subject_basis import add
from test_source_quoted_preferences import TEXT, add as add_quoted, preference

COUNT = 'The loading cupboard count is 14.'


def projection_at(tmp_path):
    return SourceClaimProjection(TurnIdempotencyLedger(tmp_path / 'sources.db'))


def claims(projection):
    with closing(projection.ledger._connect()) as conn:
        return [dict(row) | json.loads(row['data_json']) for row in conn.execute('SELECT * FROM source_claims')]


def edges(projection):
    with closing(projection.ledger._connect()) as conn:
        return [dict(row) for row in conn.execute('SELECT * FROM source_claim_witnesses ORDER BY recorded_at')]


def day(n):
    return f'2026-05-{n:02d}T10:00:00+00:00'


@pytest.fixture
def rules(monkeypatch):
    monkeypatch.setenv(learning.FLAG, '1')


# -- repeats are witnesses ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize('on', [False, True])
async def test_repeats_add_witnesses_instead_of_rows(tmp_path, monkeypatch, on):
    monkeypatch.setenv(learning.FLAG, '1' if on else '0')
    projection = projection_at(tmp_path)
    for n in range(1, 6):
        await add(projection, f'a{n}', COUNT, '14', occurred=day(n))
    rows = claims(projection)
    if not on:
        assert len(rows) == 5 and edges(projection) == []
        return
    [row] = rows
    assert row['witnesses'] == 5 and row['turn_id'] == 'a5'          # anchored at the newest witness
    assert [edge['turn_id'] for edge in edges(projection)] == ['a1', 'a2', 'a3', 'a4']
    packet, = prepared(projection, 'loading cupboard count', contact='owner', now=day(20))
    assert packet['status'] == 'source_assertion'
    assert [(a['value'], a['source']) for a in packet['assertions']] == [('14', 'turn:a5')]
    with closing(projection.ledger._connect()) as conn:
        assert learning.summary(conn)['log']['claim']['witnessed'] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize('on', [False, True])
async def test_a_correction_retracts_the_value_not_one_witness(tmp_path, monkeypatch, on):
    """Said twice, then corrected: unfolded, the model is shown the newest witness only, so the older one
    resurrects the corrected value as a conflict. Folded (and for legacy duplicates, retired together),
    the correction wins."""
    monkeypatch.setenv(learning.FLAG, '1' if on else '0')
    projection = projection_at(tmp_path)
    await add(projection, 'a1', COUNT, '14', occurred=day(1))
    await add(projection, 'a2', COUNT, '14', occurred=day(2))
    await add(projection, 'c', 'Correction: the loading cupboard count is 17.', '17', occurred=day(3),
              operation='correct', match_prior=True)
    packet, = prepared(projection, 'loading cupboard count', contact='owner', now=day(20))
    values = [(a['value'], a['source']) for a in packet['assertions']]
    if on:
        assert packet['status'] == 'source_assertion' and values == [('17', 'turn:c')]
    else:
        assert packet['status'] == 'unresolved_conflict' and ('14', 'turn:a1') in values


@pytest.mark.asyncio
async def test_legacy_duplicate_rows_are_retired_with_the_corrected_value(tmp_path, monkeypatch):
    """Rows written before the rules (one per message) are retired together when the rules are on."""
    projection = projection_at(tmp_path)
    monkeypatch.setenv(learning.FLAG, '0')
    await add(projection, 'a1', COUNT, '14', occurred=day(1))
    await add(projection, 'a2', COUNT, '14', occurred=day(2))
    monkeypatch.setenv(learning.FLAG, '1')
    await add(projection, 'c', 'Correction: the loading cupboard count is 17.', '17', occurred=day(3),
              operation='correct', match_prior=True)
    retracted = {row['turn_id']: row['retracted_by'] for row in claims(projection)}
    assert retracted['a1'] and retracted['a1'] == retracted['a2']
    packet, = prepared(projection, 'loading cupboard count', contact='owner', now=day(20))
    assert [(a['value'], a['source']) for a in packet['assertions']] == [('17', 'turn:c')]


@pytest.mark.asyncio
async def test_premise_records_keep_their_rows_but_a_correction_retires_every_duplicate(tmp_path, rules):
    """Decisions, procedures and substantive events are read per message by the opinion pass: they never
    fold. A correction still retracts every live row of the corrected value."""
    projection = projection_at(tmp_path)
    await add(projection, 'a1', COUNT, '14', occurred=day(1), memory_kind='decision')
    await add(projection, 'a2', COUNT, '14', occurred=day(2), memory_kind='decision')
    assert len(claims(projection)) == 2 and edges(projection) == []
    await add(projection, 'c', 'Correction: the loading cupboard count is 17.', '17', occurred=day(3),
              operation='correct', match_prior=True, memory_kind='decision')
    packet, = prepared(projection, 'loading cupboard count', contact='owner', now=day(20))
    assert [(a['value'], a['source']) for a in packet['assertions']] == [('17', 'turn:c')]


@pytest.mark.asyncio
@pytest.mark.parametrize('on', [False, True])
async def test_a_restated_preference_does_not_crowd_itself_out_of_recall(tmp_path, monkeypatch, on):
    """Unfolded, nine verbatim statements overflow the bundle and recall shows no preference at all."""
    monkeypatch.setenv(learning.FLAG, '1' if on else '0')
    projection = projection_at(tmp_path)
    for n in range(9):
        await add_quoted(projection, str(n), TEXT, preference())
    packet, = prepared(projection, 'handoff', contact='owner')
    if on:
        assert packet['status'] == 'quoted_preference_statements'
        assert [a['value'] for a in packet['assertions']] == [TEXT]
        assert [row['value'] for row in projection.preferences('owner')] == [TEXT]
    else:
        assert packet['status'] == 'incomplete_assertion_history' and 'assertions' not in packet


@pytest.mark.asyncio
async def test_quoted_statements_fold_only_on_a_verbatim_whole_statement(tmp_path, rules):
    projection = projection_at(tmp_path)
    await add_quoted(projection, 'a', TEXT, preference())
    await add_quoted(projection, 'b', TEXT.lower(), preference(TEXT.lower()))              # case differs
    await add_quoted(projection, 'c', TEXT + ' Thanks.', preference())                     # not the whole message
    await add_quoted(projection, 'd', TEXT, preference())                                  # verbatim
    by_turn = {row['turn_id']: row for row in claims(projection)}
    assert set(by_turn) == {'b', 'c', 'd'} and by_turn['d']['witnesses'] == 2
    assert [edge['turn_id'] for edge in edges(projection)] == ['a']


@pytest.mark.asyncio
async def test_the_anchor_is_the_newest_witness_in_a_conflict(tmp_path, rules):
    projection = projection_at(tmp_path)
    await add(projection, 'a1', COUNT, '14', occurred=day(1))
    await add(projection, 'b', 'The loading cupboard count is 15.', '15', occurred=day(2))
    await add(projection, 'a3', COUNT, '14', occurred=day(3))
    packet, = prepared(projection, 'loading cupboard count', contact='owner', now=day(20))
    assert packet['status'] == 'unresolved_conflict'
    assert {(a['value'], a['source']) for a in packet['assertions']} == {('14', 'turn:a3'), ('15', 'turn:b')}


@pytest.mark.asyncio
async def test_a_hit_on_an_older_witness_still_finds_the_claim(tmp_path, rules):
    """Recall reads claims through the source it hit: an older wording of the same value still leads to
    the claim (its span stripped from the quotation), as its own row did unfolded."""
    projection = projection_at(tmp_path)
    older = 'The loading cupboard count is 14, per the quarterly audit.'
    await add(projection, 'old', older, '14', occurred=day(1))
    await add(projection, 'new', COUNT, '14', occurred=day(2))
    assert len(claims(projection)) == 1
    packet, = prepared(projection, 'quarterly audit', contact='owner', now=day(20))
    assert [a['value'] for a in packet['assertions']] == ['14']


@pytest.mark.asyncio
async def test_a_stale_match_is_not_a_repeat(tmp_path, rules):
    """14, corrected to 17, then 14 again: the restatement is new (the retracted claim stays retracted)."""
    projection = projection_at(tmp_path)
    await add(projection, 'a', COUNT, '14', occurred=day(1))
    await add(projection, 'c', 'Correction: the loading cupboard count is 17.', '17', occurred=day(2),
              operation='correct', match_prior=True)
    await add(projection, 'again', COUNT, '14', occurred=day(3))
    rows = {row['turn_id']: row for row in claims(projection)}
    assert set(rows) == {'a', 'c', 'again'} and rows['a']['retracted_by'] and not rows['again']['retracted_by']
    with closing(projection.ledger._connect()) as conn:
        [entry] = learning.entries(conn, store='claim', action='labelled')
    assert entry['decision']['step'] == 'stale' and entry['item_id'] == rows['a']['id']


# -- erasure and reattribution ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reattributing_the_anchor_moves_the_claim_to_the_owners_other_witness(tmp_path, rules):
    from colonymind.turns.source_attribution import correct as reattribute
    projection = projection_at(tmp_path)
    await add(projection, 'a1', COUNT, '14', occurred=day(1))
    await add(projection, 'a2', COUNT, '14', occurred=day(2))
    reattribute(projection.ledger, operation_id='fix-a2', performed_by='operator', old_contact_id='owner',
                contact_id='other', source_ids=['a2'], evidence_refs=['operator:correction'])
    owner = [row for row in claims(projection) if row['turn_id'] == 'a1']
    assert len(owner) == 1 and owner[0]['witnesses'] == 1
    packet, = prepared(projection, 'loading cupboard count', contact='owner', now=day(20))
    assert [(a['value'], a['source']) for a in packet['assertions']] == [('14', 'turn:a1')]


# -- reversibility --------------------------------------------------------------------------------


# -- preferences retire only by their owner --------------------------------------------------------

async def _preference_change(projection, text, operation='correct', *, predicate=None):
    original = next(row for row in claims(projection) if row['turn_id'] == 'original')
    proposal = preference(text, operation=operation, prior_claim_id=original['id'],
                          valid_from_text='now' if operation == 'change' else None)
    if predicate:
        proposal['predicate'] = predicate
    await add_quoted(projection, 'changed', text, proposal)
    return original


def preference_about(text, predicate):
    proposal = preference(text)
    proposal['predicate'] = predicate
    return proposal


MISLINKED = 'Correction: I prefer quiet rooms.'      # a cue, but nothing of the handoff preference


@pytest.mark.asyncio
async def test_a_preference_change_the_owners_words_do_not_name_waits_for_the_owner(tmp_path, rules):
    projection = projection_at(tmp_path)
    await add_quoted(projection, 'original', TEXT, preference())
    await _preference_change(projection, MISLINKED)
    # Both statements stand, verbatim, until the owner decides.
    assert {row['turn_id'] for row in projection.preferences('owner')} == {'original', 'changed'}
    with closing(projection.ledger._connect()) as conn:
        [held] = learning.pending(conn, contact_id='owner')
    assert held['kind'] == 'preference_change' and held['operation'] == 'correct'
    declined = projection.decide(held['id'], ratify=False, reason='both hold')
    assert declined['status'] == 'declined' and declined['changed']
    assert {row['turn_id'] for row in projection.preferences('owner')} == {'original', 'changed'}
    assert projection.decide(held['id'], ratify=True)['changed'] is False       # decided once


@pytest.mark.asyncio
async def test_the_owner_ratifies_a_held_preference_change(tmp_path, rules):
    projection = projection_at(tmp_path)
    await add_quoted(projection, 'original', TEXT, preference())
    await _preference_change(projection, MISLINKED)
    with closing(projection.ledger._connect()) as conn:
        [held] = learning.pending(conn)
    assert projection.decide(held['id'], ratify=True, by='cli')['status'] == 'ratified'
    assert [row['turn_id'] for row in projection.preferences('owner')] == ['changed']
    with closing(projection.ledger._connect()) as conn:
        assert [entry['action'] for entry in learning.entries(conn, store='claim')][:2] == ['ratified', 'pending']


@pytest.mark.asyncio
@pytest.mark.parametrize('text,operation', [(MISLINKED, 'correct'), ('Starting now, I prefer quiet rooms.', 'change')])
async def test_ratifying_a_held_change_retires_every_duplicate_of_the_preference(tmp_path, monkeypatch, text,
                                                                                 operation):
    """Two verbatim statements written before the rules (one row each): the owner's ratification retires
    the preference, not only the row the extractor named, as the automatic path does."""
    projection = projection_at(tmp_path)
    monkeypatch.setenv(learning.FLAG, '0')
    await add_quoted(projection, 'original', TEXT, preference())
    await add_quoted(projection, 'again', TEXT, preference())
    monkeypatch.setenv(learning.FLAG, '1')
    await _preference_change(projection, text, operation)
    with closing(projection.ledger._connect()) as conn:
        [held] = learning.pending(conn)
    assert projection.decide(held['id'], ratify=True)['status'] == 'ratified'
    assert [row['turn_id'] for row in projection.preferences('owner')] == ['changed']


@pytest.mark.asyncio
@pytest.mark.parametrize('text,operation', [('Correction: I prefer detailed handoffs.', 'correct'),
                                            ('Starting now, I prefer detailed handoffs.', 'change')])
async def test_the_owners_explicit_words_retire_a_preference_at_once(tmp_path, rules, text, operation):
    projection = projection_at(tmp_path)
    await add_quoted(projection, 'original', TEXT, preference())
    await _preference_change(projection, text, operation)
    assert [row['turn_id'] for row in projection.preferences('owner')] == ['changed']
    with closing(projection.ledger._connect()) as conn:
        assert learning.pending(conn) == []
        assert learning.entries(conn, store='claim', action='ratified')[0]['stage'] == 'preference_change'


@pytest.mark.asyncio
@pytest.mark.parametrize('old,new,predicate,held', [
    ('I prefer short emails.', 'Correction: I prefer short hair.', 'email length', True),  # "short" is shared, the subject is not
    ('I prefer short emails.', 'Correction: I prefer short hair.', 'preferences', True),   # a predicate that names no topic
    ('I prefer short emails.', 'Correction: I prefer long emails.', 'email length', False),
    # A predicate word the owner's words share is not enough: every word of the topic must be named.
    ('I prefer short emails.', 'Correction: I prefer short hair.', 'short_email_preference', True),
    ('I prefer weekly emails.', 'Correction: I prefer weekly meetings.', 'weekly_email_format', True),
    ('I prefer weekly emails.', 'Correction: send the weekly emails as plain text.', 'weekly_email_format', False)])
async def test_a_shared_modifier_does_not_name_the_retired_preference(tmp_path, rules, old, new, predicate, held):
    projection = projection_at(tmp_path)
    await add_quoted(projection, 'original', old, preference_about(old, predicate))
    await _preference_change(projection, new, predicate=predicate)
    live = {row['turn_id'] for row in projection.preferences('owner')}
    with closing(projection.ledger._connect()) as conn:
        waiting = learning.pending(conn, contact_id='owner')
    assert (live, len(waiting)) == (({'original', 'changed'}, 1) if held else ({'changed'}, 0))


@pytest.mark.asyncio
async def test_with_the_rules_off_a_mislinked_change_retires_the_preference(tmp_path, monkeypatch):
    monkeypatch.setenv(learning.FLAG, '0')
    projection = projection_at(tmp_path)
    await add_quoted(projection, 'original', TEXT, preference())
    await _preference_change(projection, MISLINKED)
    assert [row['turn_id'] for row in projection.preferences('owner')] == ['changed']


# -- the rejection log ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_every_refused_claim_is_logged_with_ids_and_no_words(tmp_path, monkeypatch):
    """Mandatory: logged with the rules off too."""
    monkeypatch.setenv(learning.FLAG, '0')
    projection = projection_at(tmp_path)
    await add_quoted(projection, 'refused', TEXT, preference(), keep=False)
    assert claims(projection) == []
    with closing(projection.ledger._connect()) as conn:
        [entry] = learning.entries(conn, store='claim', action='rejected')
        dump = json.dumps([tuple(row) for row in conn.execute('SELECT * FROM learning_log')])
    assert entry['stage'] == 'review' and entry['refs'] == ['refused'] and entry['contact_id'] == 'owner'
    assert 'handoff' not in dump


# -- the gate ---------------------------------------------------------------------------------------

def item(ident, text, *, live=True):
    return learning.Item(ident, text, learning.digest(learning.normal(text)), live=live)


@pytest.mark.asyncio
async def test_the_gate_is_hash_then_staleness_then_similarity_then_the_model():
    live = item('L1', 'Send the weekly report on Friday before noon to the whole team.')
    stale = item('L0', 'Always call the supplier before ordering.', live=False)
    gate = learning.NoveltyGate()
    # 1. the normalised hash
    assert (await gate.classify(item('', live.text.upper()), [live, stale])).as_dict() == {
        'verdict': 'repeat', 'step': 'hash', 'match': 'L1', 'score': 1.0}
    # 2. staleness: a match on a retired item is not a repeat
    decision = await gate.classify(item('', stale.text), [live, stale])
    assert (decision.verdict, decision.step, decision.match) == ('novel', 'stale', 'L0')
    # 3. similarity bands
    assert (await gate.classify(item('', 'Send the weekly report on Friday before noon to the entire team.'),
                                [live])).verdict == 'repeat'
    assert (await gate.classify(item('', 'Water the plants twice a week.'), [live])).verdict == 'novel'
    ambiguous = await gate.classify(item('', 'Send the weekly report on Friday.'), [live])
    assert ambiguous.verdict == 'ambiguous'
    # 4. the model, batched, only for the ambiguous band; no answer is novel
    calls = []

    async def labeller(pairs):
        calls.append(pairs)
        return {pairs[0]['id']: True}
    pending = [(item('', 'a'), ambiguous, live.text),
               (item('', 'b'), learning.Decision('ambiguous', 'similarity', 'L1', 0.7), live.text)]
    await learning.NoveltyGate.resolve(pending, labeller)
    assert len(calls) == 1 and len(calls[0]) == 2
    assert [(d.verdict, d.step) for _, d, _ in pending] == [('repeat', 'model'), ('novel', 'fallback')]


@pytest.mark.asyncio
async def test_the_gate_falls_back_to_the_lexical_bands_when_embeddings_fail():
    async def broken(text, others):
        raise LookupError('no embedding pipeline')
    live = item('L1', 'Send the weekly report on Friday before noon to the whole team.')
    decision = await learning.NoveltyGate(scorer=broken).classify(
        item('', 'Send the weekly report on Friday before noon to the entire team.'), [live])
    assert decision.verdict == 'repeat' and decision.step == 'similarity'


# A changed polarity, number or day is a different instruction however alike the rest reads: the lexical
# scorer drops "not" and short numbers, and embeddings score them alike.
CHANGED_LITERALS = [
    ('Send the weekly report to the team before noon.', 'Do not send the weekly report to the team before noon.'),
    ('Send the weekly report to the team before noon.', "Don't send the weekly report to the team before noon."),
    ('Never retry the upload before noon.', 'Retry the upload before noon.'),
    ('Wait 10 minutes before retrying the upload.', 'Wait 30 minutes before retrying the upload.'),
    ('Wait ten minutes before retrying the upload.', 'Wait 30 minutes before retrying the upload.'),
    ('Send the weekly report to the team on Friday.', 'Send the weekly report to the team on Monday.')]


@pytest.mark.asyncio
@pytest.mark.parametrize('old,new', CHANGED_LITERALS)
async def test_a_changed_polarity_number_or_day_is_never_a_similarity_repeat(old, new):
    async def alike(text, others):
        return [0.99 for _ in others]

    async def broken(text, others):
        raise LookupError('no embedding pipeline')
    live = item('L1', old)
    for gate in (learning.NoveltyGate(), learning.NoveltyGate(scorer=alike), learning.NoveltyGate(scorer=broken)):
        decision = await gate.classify(item('', new), [live])
        assert (decision.verdict, decision.step) == ('novel', 'literal')
    assert learning.NoveltyGate().classify_sync(item('', new), [live]).verdict == 'novel'


# The same words in another order or joined otherwise are another instruction: who gets what, which quantity
# goes where, "and" for "or", "after" for "before". A score is blind to all of them.
CHANGED_FORM = [
    ('Send the weekly report to finance, not sales.', 'Send the weekly report to sales, not finance.'),
    ('Forward the invoices from the supplier to the accountant.',
     'Forward the invoices from the accountant to the supplier.'),
    ('Print 3 copies for the office and 5 copies for the warehouse.',
     'Print 5 copies for the office and 3 copies for the warehouse.'),
    ('Archive the reports marked public or approved.', 'Archive the reports marked public and approved.'),
    ('Call the supplier before ordering parts.', 'Call the supplier after ordering parts.')]


@pytest.mark.asyncio
@pytest.mark.parametrize('old,new', CHANGED_FORM)
async def test_reordered_or_rejoined_words_are_never_a_similarity_repeat(old, new):
    async def alike(text, others):
        return [0.99 for _ in others]

    async def broken(text, others):
        raise LookupError('no embedding pipeline')
    live = item('L1', old)
    for gate in (learning.NoveltyGate(), learning.NoveltyGate(scorer=alike), learning.NoveltyGate(scorer=broken)):
        decision = await gate.classify(item('', new), [live])
        assert decision.verdict != 'repeat', decision.as_dict()
        pending = [(item('', new), decision, old)] if decision.verdict == 'ambiguous' else []
        await learning.NoveltyGate.resolve(pending, None)
        assert decision.verdict == 'novel'
    assert learning.NoveltyGate().classify_sync(item('', new), [live]).verdict == 'novel'


@pytest.mark.asyncio
@pytest.mark.parametrize('before,after', [
    ('reports:archive handling reports marked public or approved',
     'reports:archive handling reports marked public and approved'),
    ('reports:archive handling public reports marked approved',
     'reports:archive handling approved reports marked public')])
async def test_a_rejoined_or_reordered_scope_is_never_a_similarity_repeat(before, after):
    async def alike(text, others):
        return [0.99 for _ in others]
    text = 'Archive the report. Keep a copy.'
    live = learning.Item('L1', text, 'k1', scope=before)
    for gate in (learning.NoveltyGate(), learning.NoveltyGate(scorer=alike)):
        decision = await gate.classify(learning.Item('', text, 'k2', scope=after), [live])
        assert (decision.verdict, decision.step) == ('ambiguous', 'scope')
    same = learning.Item('', text, 'k3', scope=before.replace('handling ', 'handling the '))
    assert (await learning.NoveltyGate().classify(same, [live])).verdict == 'repeat'


@pytest.mark.asyncio
async def test_a_literal_on_one_side_only_goes_to_the_model_and_a_spelled_number_still_repeats():
    async def alike(text, others):
        return [0.99 for _ in others]
    live = item('L1', 'Retry the upload before noon.')
    decision = await learning.NoveltyGate(scorer=alike).classify(item('', 'Retry the upload 3 times before noon.'), [live])
    assert (decision.verdict, decision.step) == ('ambiguous', 'literal')
    spelled = item('L2', 'Wait two minutes before retrying the upload.')
    decision = await learning.NoveltyGate().classify(item('', 'Wait 2 minutes before retrying the upload.'), [spelled])
    assert (decision.verdict, decision.step) == ('repeat', 'similarity')


@pytest.mark.parametrize('old,new,topic,expected', [
    (TEXT, 'Correction: I prefer detailed handoffs.', 'handoff format', True),
    # Shared words that are not what the preference is about (its recorded topic) name nothing.
    (TEXT, 'Starting now, I prefer a purchasing table last.', 'handoff format', False),
    (TEXT, MISLINKED, 'handoff format', False),
    ('I prefer tea in the morning.', 'Actually I now prefer coffee.', 'morning drink', False),
    # "morning" is shared and in the topic, but so is "drink", which the owner did not name: the owner decides.
    ('I prefer tea in the morning.', 'From now on, coffee in the morning please.', 'morning drink', False),
    ('I prefer tea in the morning.', 'From now on, coffee as my morning drink please.', 'morning drink', True),
    ('I prefer short emails.', 'Correction: I prefer short hair.', 'short_email_preference', False),
    ('I prefer weekly emails.', 'Correction: I prefer weekly meetings.', 'weekly_email_format', False),
    ('I prefer weekly emails.', 'Correction: send the weekly emails as plain text.', 'weekly_email_format', True),
    ('I prefer short emails.', 'Correction: I prefer short hair.', 'email length', False),
    ('I prefer short weekly emails.', 'Correction: I prefer short weekly meetings.', 'email cadence', False),
    ('I prefer short emails.', 'Correction: I prefer long emails.', 'email length', True),
    ('I prefer short emails.', 'Correction: I prefer long emails.', 'preference', False)])
def test_the_link_to_a_retired_preference_is_read_from_the_owners_words(old, new, topic, expected):
    assert learning.owner_links(old, new, topic) is expected


# -- useful recall and decay ------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_decay_follows_useful_recall_not_restatement(tmp_path, rules):
    """Two claims first said long ago: one restated often and never usefully recalled, one recalled, used
    and judged right. The sweep cools the first and keeps the second; nothing is deleted."""
    import time
    projection = projection_at(tmp_path)
    for n in range(1, 4):
        await add(projection, f'a{n}', COUNT, '14', occurred=day(n))
    await add(projection, 'b', 'The loading dock bay is 3.', '3', occurred=day(4),
              subject='loading dock', predicate='bay')
    rows = {row['predicate']: row for row in claims(projection)}
    later = time.time() + 200 * 86400
    with closing(projection.ledger._connect()) as conn, conn:
        learning.record_served(conn, session_id='s1', served='k1', store='claim', item_ids=[rows['bay']['id']],
                               now=later - 86400)
        [use] = learning.unscored(conn, ['s1'])['s1']
        assert learning.used_in('3', 'Use bay 3 at the loading dock.')
        learning.score_use(conn, use, 'useful', now=later - 86400)
        assert learning.sweep_claims(conn, now=later) == 1
        cold = learning.cold_ids(conn, 'claim', [row['id'] for row in rows.values()])
    assert cold == {rows['count']['id']}
    assert len(claims(projection)) == 2                                   # kept
    packet, = prepared(projection, 'cupboard count', contact='owner', now=day(20))
    assert [a['value'] for a in packet['assertions']] == ['14']          # and still recallable
    with closing(projection.ledger._connect()) as conn, conn:
        learning.score_use(conn, {'session_id': 's2', 'served': 'k', 'store': 'claim',
                                  'item_id': rows['count']['id']}, 'useful', now=later + 1)
        assert learning.cold_ids(conn, 'claim', [rows['count']['id']]) == set()     # revived by useful recall
        assert learning.entries(conn, action='revived')[0]['item_id'] == rows['count']['id']


def test_used_in_is_mechanical():
    assert learning.used_in('Lake', 'Your office is in Lake now.')
    assert not learning.used_in('Lake', 'Your office is in Lakeside.')
    assert learning.used_in(TEXT, 'A concise handoff with a purchasing table, as you prefer.', quoted=True)
    assert not learning.used_in(TEXT, 'Here is the report.', quoted=True)


def test_claim_ids_of_a_packet():
    selected = [{'history_anchor': {'claim_id': 'claim:a'}, 'content_format': 'source_assertions_v1',
                 'content': json.dumps({'assertions': [{'claim_id': 'claim:a'}, {'claim_id': 'claim:b'}]})},
                {'source_history_anchors': [{'claim_id': 'claim:c'}], 'content': 'raw text'},
                {'kind': 'source_quote', 'content': 'no claim'}]
    assert learning.claim_ids_of(selected) == ['claim:a', 'claim:b', 'claim:c']


def test_the_faculty_is_on_by_default_and_reaches_the_projection_as_an_environment_flag(tmp_path):
    """On unless shown harmful: the learning rules ship on, and the claim projection, which runs outside the mind,
    reads the switch from the environment an install exports (off once the file says so)."""
    from colonymind.config import DEFAULTS, apply_environment, load_config, save_config, update_config
    assert DEFAULTS['mind']['faculties']['learning'] is True
    save_config(DEFAULTS, tmp_path)
    config = load_config(tmp_path, environ={})
    environ = {}
    apply_environment(config, environ=environ)
    assert environ[learning.FLAG] == '1' and learning.enabled(environ)
    update_config({'mind': {'faculties': {'learning': False}}}, tmp_path)
    environ = {}
    apply_environment(load_config(tmp_path, environ={}), environ=environ)
    assert environ[learning.FLAG] == '0' and not learning.enabled(environ)


# -- the owner's surface ----------------------------------------------------------------------------

def test_the_learning_route_shows_the_owner_the_log_and_decides_held_changes(tmp_path, rules):
    import asyncio
    from types import SimpleNamespace
    from colonymind.api.routers import mind as mind_router
    from test_mind_router_identity import _call
    projection = projection_at(tmp_path)

    async def held_change():
        await add_quoted(projection, 'original', TEXT, preference())
        await _preference_change(projection, MISLINKED)
    asyncio.run(held_change())
    previous = mind_router._mind
    mind_router._mind = SimpleNamespace(lessons=SimpleNamespace(ledger=projection.ledger), owner_id='owner',
                                        enabled=True)
    try:
        state = _call('GET', '/v1/mind/learning').json()
        assert state['enabled'] and state['summary']['pending'] == 1 and state['pending'][0]['kind'] == 'preference_change'
        assert {row['action'] for row in state['log']} >= {'pending'}
        assert _call('GET', '/v1/mind/learning', params={'viewer': 'someone-else'}).json()['pending'] == []
        held = state['pending'][0]['id']
        assert _call('POST', f'/v1/mind/learning/pending/{held}/maybe', json={}).status_code == 404
        assert _call('POST', '/v1/mind/learning/pending/R-unknown/ratify', json={}).status_code == 404
        decided = _call('POST', f'/v1/mind/learning/pending/{held}/ratify', json={'by': 'cli'})
        assert decided.status_code == 200 and decided.json()['status'] == 'ratified'
        assert _call('POST', f'/v1/mind/learning/pending/{held}/decline', json={}).status_code == 409
    finally:
        mind_router._mind = previous
    assert [row['turn_id'] for row in projection.preferences('owner')] == ['changed']


def test_the_cli_shows_the_rules_and_ratifies(tmp_path, monkeypatch, capsys):
    import httpx
    from colonymind.mind import cli as mind_cli
    from test_mind_cli import _Transport, _parse
    from colonymind.config import DEFAULTS, save_config
    monkeypatch.setenv("COLONYMIND_HOME", str(tmp_path))
    monkeypatch.setenv("COLONYMIND_STATE_DIR", str(tmp_path))
    save_config({**DEFAULTS, "owner": {"contact_id": "p-01"}}, tmp_path)
    (tmp_path / "api.key").write_text("k\n")
    state = {"enabled": True, "pending": [{"id": "R-1", "kind": "preference_change", "operation": "correct",
                                           "target_id": "claim:a", "by_id": "claim:b"}],
             "summary": {"pending": 1, "claim_witnesses": 4, "log": {"claim": {"witnessed": 4, "rejected": 2}},
                         "tiers": {"claim:cold": 1}},
             "log": [{"id": 9, "store": "claim", "action": "rejected", "stage": "review", "reason": "review refused (1)",
                      "item_id": None}]}
    transport = _Transport({("GET", "/v1/mind/learning"): state,
                            ("POST", "/v1/mind/learning/pending/R-1/ratify"): {"id": "R-1", "status": "ratified"}})
    monkeypatch.setattr(httpx, "Client", transport.client)
    assert mind_cli.run(_parse(["mind", "learning"])) == 0
    out = capsys.readouterr().out
    assert "learning rules: on; 1 pending, 4 claim witnesses" in out and "4 witnessed" in out and "1 claim:cold" in out
    assert mind_cli.run(_parse(["mind", "learning", "pending"])) == 0
    assert "R-1 preference_change: correct of claim:a by claim:b" in capsys.readouterr().out
    assert mind_cli.run(_parse(["mind", "learning", "rejected", "--store", "claim"])) == 0
    assert "review refused (1)" in capsys.readouterr().out
    assert mind_cli.run(_parse(["mind", "learning", "ratify", "R-1", "--reason", "yes"])) == 0
    assert "R-1: ratified" in capsys.readouterr().out
    assert json.loads(transport.calls[-1][2]) == {"reason": "yes", "by": "cli"}
    assert mind_cli.run(_parse(["mind", "learning", "ratify"])) == 2


def test_the_log_is_bounded_audit(tmp_path):
    projection = projection_at(tmp_path)
    with closing(projection.ledger._connect()) as conn, conn:
        for n in range(30):
            learning.log(conn, store='claim', action='rejected', reason=f'r{n}', now=1000.0 + n)
        learning.record_served(conn, session_id='s', served='k', store='claim', item_ids=['claim:old'], now=0.0)
        learning.record_served(conn, session_id='s', served='k', store='claim', item_ids=['claim:new'], now=10_000_000.0)
        assert learning.prune(conn, keep=10, now=10_000_000.0) == 21
        assert [row['reason'] for row in learning.entries(conn, limit=100)][-1] == 'r20'
        assert [row[0] for row in conn.execute('SELECT item_id FROM learning_uses')] == ['claim:new']
        with pytest.raises(ValueError):
            learning.log(conn, store='claim', action='deleted')


def test_the_use_log_of_the_first_shape_is_rebuilt_with_its_rows(tmp_path):
    """A use log keyed by its question only (the first shape) gains the turn occurrence; its rows stay."""
    conn = sqlite3.connect(tmp_path / 'uses.db')
    conn.row_factory = sqlite3.Row
    conn.execute('CREATE TABLE learning_uses (session_id TEXT NOT NULL, served TEXT NOT NULL, store TEXT NOT NULL, '
                 'item_id TEXT NOT NULL, at REAL NOT NULL, result TEXT, PRIMARY KEY(session_id,served,store,item_id))')
    conn.execute("INSERT INTO learning_uses VALUES ('s','k','claim','claim:a',1.0,'wrong')")
    conn.commit()
    learning.initialize(conn)
    learning.initialize(conn)                                                      # idempotent
    assert [tuple(row) for row in conn.execute('SELECT session_id,occurrence,result FROM learning_uses')] == [
        ('s', '', 'wrong')]
    assert learning.record_served(conn, session_id='s', served='k', store='claim', item_ids=['claim:a'],
                                  occurrence='turn-2') == 1
    assert {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")} >= {'learning_uses'}
    assert 'learning_uses_by_question' not in {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}


from colonymind.memory import curated
@pytest.mark.asyncio
async def test_erasing_the_anchor_promotes_the_newest_surviving_witness(tmp_path, rules):
    projection = projection_at(tmp_path)
    for n in range(1, 4):
        await add(projection, f'a{n}', COUNT if n != 2 else 'Loading cupboard count: 14, recounted.', '14', occurred=day(n))
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(projection.ledger)
    curated.forget_sources(projection.ledger, 'owner', source_ids=['a3'])
    assert historical_bytes(projection.ledger) == _native_before_withdrawal
    [row] = active_claims(projection)
    assert row['turn_id'] == 'a2' and row['witnesses'] == 2 and row['evidence'].endswith('recounted.')
    assert {edge['turn_id'] for edge in edges(projection)} == {'a1', 'a2'}
    packet, = prepared(projection, 'loading cupboard count', contact='owner', now=day(20))
    assert [(a['value'], a['source']) for a in packet['assertions']] == [('14', 'turn:a2')]
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(projection.ledger)
    curated.forget_sources(projection.ledger, 'owner', source_ids=['a1'])
    assert historical_bytes(projection.ledger) == _native_before_withdrawal
    [row] = active_claims(projection)
    assert row['turn_id'] == 'a2' and row['witnesses'] == 1 and (len(edges(projection)) == 2)
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(projection.ledger)
    curated.forget_sources(projection.ledger, 'owner', source_ids=['a2'])
    assert historical_bytes(projection.ledger) == _native_before_withdrawal
    assert active_claims(projection) == [] and (not prepared(projection, 'loading cupboard count', contact='owner'))
    # The byte-identity checks above retain the historical learning rows.

@pytest.mark.asyncio
async def test_with_the_rules_off_folded_data_still_reads_and_erases(tmp_path, rules, monkeypatch):
    projection = projection_at(tmp_path)
    await add(projection, 'a1', COUNT, '14', occurred=day(1))
    await add(projection, 'a2', COUNT, '14', occurred=day(2))
    monkeypatch.setenv(learning.FLAG, '0')
    packet, = prepared(projection, 'loading cupboard count', contact='owner', now=day(20))
    assert [a['value'] for a in packet['assertions']] == ['14']
    from test_curated_memory import historical_bytes
    _native_before_withdrawal = historical_bytes(projection.ledger)
    curated.forget_sources(projection.ledger, 'owner', source_ids=['a2'])
    assert historical_bytes(projection.ledger) == _native_before_withdrawal
    [row] = active_claims(projection)
    assert row['turn_id'] == 'a1'


def active_claims(projection):
    with projection.ledger._connect() as conn:
        return projection._rows(conn, 'owner', '')

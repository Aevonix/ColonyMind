"""A yes to a link proposal confirms it where its contact already holds the number in another form."""
import pytest

from colonymind.contacts import identity_links
from colonymind.contacts.config import ContactsConfig
from colonymind.contacts.store import SQLiteContactStore

NUMBER = '+10000000000'
JID = '10000000000@s.whatsapp.net'
MAILBOX = 'second.fixture@example.com'


@pytest.fixture
async def store(tmp_path):
    store = SQLiteContactStore(config=ContactsConfig(sqlite_path=str(tmp_path / 'contacts.db')))
    await store.connect()
    try:
        yield store
    finally:
        await store.close()


async def _three(store):
    return [(await store.create(display_name=name)).contact_id for name in ('First', 'Second', 'Third')]


async def _row(store, contact_id, gateway, address, *, verified=1, source='manual'):
    db = store._require_db()
    await db.execute('''INSERT INTO contact_handles
        (handle_id,contact_id,gateway,address,is_primary,verified,confidence,source,created_at)
        VALUES (?,?,?,?,0,?,1,?,'2026-01-01T00:00:00+00:00')''',
        (f'hdl-fixture-{gateway}', contact_id, gateway, address, verified, source))
    await db.commit()


async def _rows(store):
    async with store._require_db().execute(
            'SELECT contact_id, gateway, address, verified, source FROM contact_handles ORDER BY rowid') as cur:
        return [tuple(row) for row in await cur.fetchall()]


async def _status(store, candidate):
    async with store._require_db().execute('SELECT status FROM contact_identity_candidates WHERE candidate_id=?',
                                           (candidate['candidate_id'],)) as cur:
        return (await cur.fetchone())['status']


async def _yes(store, candidate):
    return await store.confirm_link(candidate['candidate_id'], performed_by='owner-fixture')


@pytest.mark.asyncio
@pytest.mark.parametrize(('held', 'proposed'), [
    (('sms', NUMBER), ('whatsapp', JID)),
    (('whatsapp', JID), ('sms', NUMBER)),
    (('whatsapp', JID), ('whatsapp', NUMBER)),
    (('sms', NUMBER), ('rcs', '+1 (000) 000-0000')),
    (('email', MAILBOX.upper()), ('email', MAILBOX)),    # an address saved before addresses were lower-cased
])
async def test_a_yes_confirms_a_proposal_whose_number_its_contact_holds_in_another_form(store, held, proposed):
    first, second, third = await _three(store)
    await _row(store, second, *held)
    candidate = await store.propose_handle_link(second, *proposed, evidence_refs=['fixture:candidate'])
    before = await _rows(store)
    assert [row['candidate_id'] for row in await store.list_handle_proposals()] == [candidate['candidate_id']]

    confirmed = await _yes(store, candidate)

    assert (confirmed['status'], confirmed['merged']) == ('confirmed', False)
    assert (confirmed['old_contact_id'], confirmed['contact_id']) == (None, second)
    assert await _status(store, candidate) == 'confirmed'
    assert await store.list_handle_proposals() == []
    # The proposed form is a verified handle of the contact now, beside the form it held.
    assert await _rows(store) == before + [(second, candidate['gateway'], candidate['address'], 1, 'owner_operator')]
    for gateway, address in (held, proposed):
        assert (await store.resolve_messaging_handle(gateway, address)).contact_id == second
    with pytest.raises(ValueError, match='^identity_candidate_not_pending$'):
        await _yes(store, candidate)


@pytest.mark.asyncio
async def test_a_yes_verifies_the_proposed_row_its_contact_holds_only_as_a_name_guess(store):
    first, second, third = await _three(store)
    await _row(store, second, 'whatsapp', JID, verified=0, source='auto:scoped-name')
    candidate = await store.propose_handle_link(second, 'whatsapp', JID, evidence_refs=['fixture:candidate'])
    assert await store.resolve_messaging_handle('whatsapp', JID) is None

    confirmed = await _yes(store, candidate)

    assert (confirmed['status'], confirmed['merged']) == ('confirmed', False)
    assert (confirmed['old_contact_id'], confirmed['contact_id']) == (second, second)
    assert await _status(store, candidate) == 'confirmed'
    assert await _rows(store) == [(second, 'whatsapp', JID, 1, 'owner_operator')]
    assert (await store.resolve_messaging_handle('whatsapp', JID)).contact_id == second


@pytest.mark.asyncio
@pytest.mark.parametrize('moved', ['before-the-yes', 'while-the-yes-is-under-way'])
async def test_a_yes_is_still_refused_once_the_number_has_moved_to_a_third_contact(store, monkeypatch, moved):
    first, second, third = await _three(store)
    await store.add_handle(second, 'sms', NUMBER, verified=True)
    one = await store.propose_handle_link(second, 'whatsapp', JID, evidence_refs=['fixture:one'])
    two = await store.propose_handle_link(second, 'rcs', NUMBER, evidence_refs=['fixture:two'])
    # While the contact holds the number, a yes to one of its proposals confirms that one.
    assert (await _yes(store, one))['status'] == 'confirmed'
    assert (await _status(store, one), await _status(store, two)) == ('confirmed', 'pending')

    if moved == 'before-the-yes':
        # Every handle of the number is a third, established contact's now; the proposal is still open.
        db = store._require_db()
        await db.execute('UPDATE contact_handles SET contact_id=? WHERE contact_id=?', (third, second))
        await db.commit()
        answer, left, holder = 'identity_handle_held', 'rejected', third
        after = [(third, 'sms', NUMBER, 1, 'manual'), (third, 'whatsapp', JID, 1, 'owner_operator')]
    else:
        # The proposed handle becomes a third contact's after the yes has looked and before it writes.
        correct = identity_links.correct

        async def moved_first(*args, **kwargs):
            await _row(store, third, 'rcs', NUMBER)
            return await correct(*args, **kwargs)

        monkeypatch.setattr(identity_links, 'correct', moved_first)
        answer, left, holder = 'identity_preimage_changed', 'pending', None
        after = [(second, 'sms', NUMBER, 1, 'manual'), (second, 'whatsapp', JID, 1, 'owner_operator'),
                 (third, 'rcs', NUMBER, 1, 'manual')]

    with pytest.raises(ValueError, match=f'^{answer}$'):
        await _yes(store, two)

    assert await _status(store, two) == left
    assert await _rows(store) == after
    resolved = await store.resolve_messaging_handle('rcs', NUMBER)
    assert (resolved.contact_id if resolved else None) == holder

"""Bare digits of a phone number another contact holds on the same gateway are that number."""
import pytest

from colonymind.contacts.config import ContactsConfig
from colonymind.contacts.store import SQLiteContactStore

NUMBER = '+10000000000'
JID = '10000000000@s.whatsapp.net'


@pytest.fixture
async def store(tmp_path):
    store = SQLiteContactStore(config=ContactsConfig(sqlite_path=str(tmp_path / 'contacts.db')))
    await store.connect()
    try:
        yield store
    finally:
        await store.close()


async def _two(store, gateway, address):
    first = await store.create(display_name='First')
    second = await store.create(display_name='Second')
    await store.add_handle(first.contact_id, gateway, address, verified=True)
    return first.contact_id, second.contact_id


async def _rows(store):
    async with store._require_db().execute(
            'SELECT contact_id, gateway, address FROM contact_handles ORDER BY rowid') as cur:
        return [tuple(row) for row in await cur.fetchall()]


@pytest.mark.asyncio
@pytest.mark.parametrize(('gateway', 'held', 'digits'), [
    ('sms', NUMBER, '0000000000'), ('sms', NUMBER, '10000000000'), ('sms', NUMBER, '(000) 000-0000'),
    ('whatsapp', JID, '10000000000'), ('whatsapp', JID, '0000000000'),
])
@pytest.mark.parametrize('source', ['manual', 'auto:sender'])
async def test_bare_digits_of_a_number_another_contact_holds_there_are_refused_as_the_plus_form_is(
        store, gateway, held, digits, source):
    first, second = await _two(store, gateway, held)
    before = await _rows(store)
    with pytest.raises(ValueError) as plus_form:
        await store.add_handle(second, gateway, NUMBER, source=source)
    with pytest.raises(ValueError) as bare:
        await store.add_handle(second, gateway, digits, source=source)
    assert str(bare.value) == str(plus_form.value) == f'Phone identity is already assigned to contact {first}'
    assert await _rows(store) == before
    for form in (NUMBER, digits):
        assert (await store.resolve_messaging_handle(gateway, form)).contact_id == first
    # The contact that holds the number may still be given its digits.
    added = await store.add_handle(first, gateway, digits, source=source)
    assert added.contact_id == first
    assert await _rows(store) == before + [(first, gateway, added.address)]
    for form in (NUMBER, digits):
        assert (await store.resolve_messaging_handle(gateway, form)).contact_id == first


@pytest.mark.asyncio
async def test_the_same_digits_stay_a_numeric_id_on_a_gateway_that_holds_no_such_number(store):
    first, second = await _two(store, 'sms', NUMBER)
    third = (await store.create(display_name='Third')).contact_id
    # A numeric chat or user id: stored as given, for whoever it is given to, and matched exactly.
    stored = []
    for contact_id, user_id in [(second, '10000000000'), (third, '0000000000'), (third, '100000000000000')]:
        handle = await store.add_handle(contact_id, 'telegram', user_id, source='auto:sender')
        stored.append((handle.contact_id, handle.gateway, handle.address))
    assert stored == [(second, 'telegram', '10000000000'), (third, 'telegram', '0000000000'),
                      (third, 'telegram', '100000000000000')]
    assert await _rows(store) == [(first, 'sms', NUMBER), *stored]
    assert (await store.resolve_messaging_handle('telegram', '10000000000')).contact_id == second
    assert (await store.resolve_messaging_handle('telegram', '0000000000')).contact_id == third
    with pytest.raises(ValueError, match=r'^Handle \(telegram, 10000000000\) is already assigned to contact '):
        await store.add_handle(third, 'telegram', '10000000000')
    # The number itself is untouched by the ids, and its own digits on its gateway are still refused.
    assert (await store.resolve_messaging_handle('sms', NUMBER)).contact_id == first
    with pytest.raises(ValueError, match='^Phone identity is already assigned to contact ' + first + '$'):
        await store.add_handle(second, 'sms', '10000000000')
    assert await _rows(store) == [(first, 'sms', NUMBER), *stored]

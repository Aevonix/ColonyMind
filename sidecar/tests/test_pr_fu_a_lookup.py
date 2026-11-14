"""A reference or a search with an at sign, or with part of a number, reaches the stored handles."""
import pytest

from colonymind.contacts.config import ContactsConfig
from colonymind.contacts.store import SQLiteContactStore

LEGACY_ROW = '''INSERT INTO contact_handles
    (handle_id,contact_id,gateway,address,is_primary,verified,confidence,source,created_at)
    VALUES (?,?,?,?,0,1,1.0,'manual','2026-01-01T00:00:00Z')'''


@pytest.fixture
async def store(tmp_path):
    store = SQLiteContactStore(config=ContactsConfig(sqlite_path=str(tmp_path / 'contacts.db')))
    await store.connect()
    try:
        yield store
    finally:
        await store.close()


async def _people(store):
    """Addresses with an at sign on gateways other than ``email``, one email and two numbers."""
    people = {}
    for name, gateway, address in [
            ('Sam', 'imessage', 'sam.fixture@example.com'), ('Kite', 'telegram', '@kitefixture'),
            ('Lee', 'xmpp', 'lee.fixture@example.com'), ('Kim', 'imessage', 'kim.fixture@example.org'),
            ('Lid', 'whatsapp', '100000000000000@lid'), ('Pat', 'email', 'pat.fixture@example.org'),
            ('Ada', 'sms', '+10000000000'), ('Bo', 'sms', '+10000000001')]:
        contact = await store.create(display_name=name + ' Invented')
        await store.add_handle(contact.contact_id, gateway, address)
        people[name] = contact.contact_id
    return people


async def _found(store, lookup, text):
    if lookup == 'search':
        return sorted(contact.contact_id for contact in await store.search(text))
    answers = [await store.resolve_reference(text), await store.resolve_reference(text, exact=True)]
    assert len({answer.contact_id if answer else None for answer in answers}) == 1
    return [answers[0].contact_id] if answers[0] else []


@pytest.mark.asyncio
@pytest.mark.parametrize(('lookup', 'text', 'names'), [
    ('reference', 'sam.fixture@example.com', ['Sam']),
    ('reference', '@kitefixture', ['Kite']),
    ('reference', '100000000000000@lid', ['Lid']),
    ('search', '@example.com', ['Sam', 'Lee']),
    ('search', 'fixture@', ['Sam', 'Lee', 'Kim', 'Pat']),
    ('search', '+1000000', ['Ada', 'Bo']),
])
async def test_an_address_or_a_part_of_one_reaches_the_contacts_holding_it(store, lookup, text, names):
    people = await _people(store)
    assert await _found(store, lookup, text) == sorted(people[name] for name in names)


@pytest.mark.asyncio
async def test_what_names_no_single_established_contact_answers_nobody_and_is_still_searched(store):
    await _people(store)
    # Two live contacts hold the address.
    twins = [await store.create(display_name=name) for name in ('Twin One', 'Twin Two')]
    await store.add_handle(twins[0].contact_id, 'telegram', '@twinfixture')
    await store.add_handle(twins[1].contact_id, 'matrix', '@twinfixture')
    # A sender the agent remembered on its own chose the address.
    shadow = await store.create(display_name='Shade Invented', import_source='auto:sender')
    await store.add_handle(shadow.contact_id, 'telegram', '@shadefixture', source='auto:sender')
    # A text with an at sign is never read as a name.
    named = await store.create(display_name='name.fixture@example.com')
    for text, candidates in [('@twinfixture', twins), ('@shadefixture', [shadow]),
                             ('name.fixture@example.com', [named])]:
        assert await _found(store, 'reference', text) == []
        assert await _found(store, 'search', text) == sorted(contact.contact_id for contact in candidates)


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['phone', 'email'])
async def test_a_number_or_an_email_two_contacts_hold_stays_nobody_and_a_part_of_it_is_searched(store, kind):
    first = await store.create(display_name='First')
    second = await store.create(display_name='Second')
    gateway, reference, other, part = {
        'phone': ('sms', '+10000000000', ('whatsapp', '+1 (000) 000-0000'), '+1000000'),
        'email': ('email', 'fixture@example.invalid', ('email', 'FIXTURE@EXAMPLE.INVALID'), '@example.invalid'),
    }[kind]
    await store.add_handle(first.contact_id, gateway, reference, verified=True)
    # An installed store's already split rows.
    db = store._require_db()
    await db.execute(LEGACY_ROW, ('legacy-split', second.contact_id, *other))
    await db.commit()
    for text in (reference, gateway + ':' + reference):
        assert await _found(store, 'reference', text) == []
        assert await _found(store, 'search', text) == []
    holders = {'phone': [first], 'email': [first, second]}[kind]
    assert await _found(store, 'search', part) == sorted(contact.contact_id for contact in holders)

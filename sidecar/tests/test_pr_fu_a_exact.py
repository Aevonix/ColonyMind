"""An exact correction finds a handle row that was saved with white space at either end."""
import hashlib

import pytest

from colonymind.contacts.config import ContactsConfig
from colonymind.contacts.store import SQLiteContactStore

NUMBER = '+10000000000'
LEGACY, BARE = '(000) 000-0000', '0000000000'
PADDED = ' (000) 000-0000 '
TABLES = ('contact_handles', 'contact_identity_operations', 'contact_audit')
ROW = '''INSERT INTO contact_handles
    (handle_id,contact_id,gateway,address,is_primary,verified,confidence,source,created_at)
    VALUES (?,?,?,?,0,1,1,'manual','2026-09-01T00:00:00Z')'''

# Rows of the one number as older stores saved them: handle id, gateway, address, holder.
SHAPES = {
    'legacy': [('hdl-zlegacy', 'whatsapp', LEGACY, 'First')],
    'padded-trailing': [('hdl-zlegacy', 'whatsapp', LEGACY + ' ', 'First')],
    'padded-leading': [('hdl-zlegacy', 'whatsapp', ' ' + LEGACY, 'First')],
    'twin-on-third-legacy-listed-first': [('hdl-0legacy', 'whatsapp', LEGACY, 'First'),
                                          ('hdl-5twin', 'whatsapp', BARE, 'Third')],
    'twin-on-third-twin-listed-first': [('hdl-zlegacy', 'whatsapp', LEGACY, 'First'),
                                        ('hdl-5twin', 'whatsapp', BARE, 'Third')],
    'twin-on-target': [('hdl-zlegacy', 'whatsapp', LEGACY, 'First'), ('hdl-5twin', 'whatsapp', BARE, 'Second')],
    'twin-same-holder-legacy-listed-first': [('hdl-0legacy', 'whatsapp', LEGACY, 'First'),
                                             ('hdl-5twin', 'whatsapp', BARE, 'First')],
    'twin-same-holder-twin-listed-first': [('hdl-zlegacy', 'whatsapp', LEGACY, 'First'),
                                           ('hdl-5twin', 'whatsapp', BARE, 'First')],
    'two-gateways-legacy': [('hdl-zlegacy', 'whatsapp', LEGACY, 'First'), ('hdl-ylegacy', 'telegram', LEGACY, 'First')],
    'two-gateways-padded': [('hdl-zlegacy', 'whatsapp', LEGACY + ' ', 'First'),
                            ('hdl-ylegacy', 'telegram', LEGACY + ' ', 'First')],
    'two-gateways-padded-two-holders': [('hdl-zlegacy', 'whatsapp', LEGACY + ' ', 'First'),
                                        ('hdl-ylegacy', 'telegram', LEGACY + ' ', 'Third')],
    'two-gateways-one-padded': [('hdl-zlegacy', 'whatsapp', LEGACY + ' ', 'First'),
                                ('hdl-ylegacy', 'telegram', LEGACY, 'First')],
}
# The requested sms handle is held by the wrong contact, absent, or already on the right one.
WALKS = ([(shape, 'held', step) for shape in SHAPES for step in ('move', 'remove')]
         + [(shape, held, 'move') for shape in ('legacy', 'padded-trailing', 'twin-on-third-legacy-listed-first',
                                                'twin-on-target', 'two-gateways-padded')
            for held in ('absent', 'on-target')])


async def _open(path):
    store = SQLiteContactStore(config=ContactsConfig(sqlite_path=str(path)))
    await store.connect()
    return store


@pytest.fixture
async def store(tmp_path):
    store = await _open(tmp_path / 'contacts.db')
    try:
        yield store
    finally:
        await store.close()


def _args(op, gateway, address, expected, target):
    return dict(operation_id=op, performed_by='owner-fixture', gateway=gateway, address=address,
                expected_contact_id=expected, contact_id=target, evidence_refs=['fixture:' + op])


async def _padded(store, *gateways):
    """First holds the number on sms and, on each gateway, a row saved with a space at either end."""
    first = await store.create(display_name='First')
    second = await store.create(display_name='Second')
    await store.add_handle(first.contact_id, 'sms', NUMBER)
    db = store._require_db()
    for gateway in gateways:
        await db.execute(ROW, ('hdl-zlegacy-' + gateway, first.contact_id, gateway, PADDED))
    await db.commit()
    return first.contact_id, second.contact_id


async def _rows(store, table='contact_handles'):
    async with store._require_db().execute(f'SELECT * FROM {table} ORDER BY rowid') as cur:
        return [dict(row) for row in await cur.fetchall()]


@pytest.mark.asyncio
@pytest.mark.parametrize('gateways', [('whatsapp',), ('whatsapp', 'signal')])
@pytest.mark.parametrize('step', ['move', 'remove'])
async def test_the_documented_steps_end_a_refused_whole_number_request(store, gateways, step):
    first, second = await _padded(store, *gateways)
    whole = _args('whole', 'sms', NUMBER, first, second)
    with pytest.raises(ValueError) as refusal:
        await store.correct_handle_identity(**whole)
    named = str(refusal.value).removeprefix('identity_phone_ambiguous ').split(', ')
    assert named == sorted(f'{gateway}:hdl-zlegacy-{gateway}:{first}' for gateway in gateways)
    shown = {handle.handle_id: handle.address for handle in await store.get_handles(first)}
    target = second if step == 'move' else None
    for conflict in named:
        gateway, handle_id, holder = conflict.split(':')
        # The handle the refusal names, by its address exactly as shown, its holder expected.
        result = await store.correct_handle_identity(**_args('exact-' + gateway, gateway, shown[handle_id],
                                                             holder, target))
        assert (result['old_contact_id'], result['contact_id'], result['handle_id']) == (first, target, handle_id)
        assert result['address_sha256'] == hashlib.sha256(PADDED.encode()).hexdigest()
    retried = await store.correct_handle_identity(**whole)
    assert (retried['old_contact_id'], retried['contact_id']) == (first, second)
    rows = await _rows(store)
    # The number is on the right contact, with no twin row and no rewritten address.
    assert {row['contact_id'] for row in rows} == {second}
    assert sorted(row['gateway'] for row in rows) == sorted(['sms', *gateways] if step == 'move' else ['sms'])
    assert {row['address'] for row in rows} <= {NUMBER, PADDED}
    for gateway in ('sms', *gateways):
        assert (await store.resolve_messaging_handle(gateway, NUMBER)).contact_id == second


@pytest.mark.asyncio
@pytest.mark.parametrize('request_kind', ['removal', 'move'])
async def test_nobody_expected_on_a_held_padded_row_is_refused_and_writes_nothing(store, request_kind):
    first, second = await _padded(store, 'whatsapp', 'signal')
    before = {table: await _rows(store, table) for table in TABLES}
    with pytest.raises(ValueError, match='^identity_preimage_changed$'):
        await store.correct_handle_identity(**_args('unexpected', 'whatsapp', PADDED, None,
                                                    second if request_kind == 'move' else None))
    assert {table: await _rows(store, table) for table in TABLES} == before


@pytest.mark.asyncio
async def test_merging_the_holder_moves_its_padded_rows(store):
    first, second = await _padded(store, 'whatsapp', 'signal')
    await store.merge(second, first, performed_by='owner-fixture')
    rows = await _rows(store)
    assert {row['contact_id'] for row in rows} == {second}
    assert sorted(row['gateway'] for row in rows) == ['signal', 'sms', 'whatsapp']
    assert {row['address'] for row in rows} == {NUMBER, PADDED}
    assert (await store.resolve_messaging_handle('whatsapp', NUMBER)).contact_id == second


async def _answer(store, **args):
    try:
        return 'ok', await store.correct_handle_identity(**args)
    except ValueError as refused:
        return 'refused', str(refused)


def _named(answer):
    kind, text = answer
    if kind != 'refused' or not text.startswith('identity_phone_ambiguous '):
        return []
    return [item.split(':') for item in text.removeprefix('identity_phone_ambiguous ').split(', ')]


async def _walk(store, shape, held, step, padded_steps):
    """The refused request, the documented steps, then the number itself where they do not end the refusal."""
    people = {name: (await store.create(display_name=name)).contact_id for name in ('First', 'Second', 'Third')}
    rows = list(SHAPES[shape])
    if held != 'absent':
        rows.append(('hdl-sms', 'sms', NUMBER, 'First' if held == 'held' else 'Second'))
    db = store._require_db()
    for handle_id, gateway, address, holder in rows:
        await db.execute(ROW, (handle_id, people[holder], gateway, address))
    await db.commit()
    second = people['Second']
    whole = _args('whole', 'sms', NUMBER, {'held': people['First'], 'on-target': second, 'absent': None}[held], second)
    for n, (gateway, handle_id, contact_id) in enumerate(_named(await _answer(store, **whole))):
        shown = [handle.address for handle in await store.get_handles(contact_id) if handle.handle_id == handle_id]
        # ``padded_steps`` off: a step on a row saved with white space is left out, as when it changed nothing.
        if shown and (padded_steps or shown[0] == shown[0].strip()):
            await _answer(store, **_args(f'step-{n}', gateway, shown[0], contact_id,
                                         second if step == 'move' else None))
    answers = [await _answer(store, **whole)]
    if answers[0][0] != 'ok':
        # The number itself on each gateway the refusal names: + form, whoever holds that exact form expected.
        for n, gateway in enumerate(sorted({item[0] for item in _named(answers[0])})):
            holder = [row['contact_id'] for row in await _rows(store)
                      if (row['gateway'], row['address']) == (gateway, NUMBER)]
            answers.append(await _answer(store, **_args(f'number-{n}', gateway, NUMBER,
                                                        holder[0] if holder else None, second)))
        answers.append(await _answer(store, **whole))
    reached = [await store.resolve_messaging_handle(gateway, NUMBER) for gateway in ('sms', 'whatsapp', 'telegram')]
    return answers, all(found is not None and found.contact_id == second for found in reached)


@pytest.mark.asyncio
async def test_every_way_out_that_worked_still_works_and_the_documented_steps_reach_padded_rows(tmp_path):
    wrong = []
    for padded_steps in (True, False):
        for n, (shape, held, step) in enumerate(WALKS):
            store = await _open(tmp_path / f'{n}-{padded_steps}.db')
            try:
                answers, reached = await _walk(store, shape, held, step, padded_steps)
            finally:
                await store.close()
            walk = (shape, held, step, padded_steps)
            # The documented steps alone fail where a twin row answers for the row shown, and nowhere else.
            twin = 'twin' in shape and not (shape.endswith('twin-listed-first') and step == 'remove')
            documented = not twin and (padded_steps or 'padded' not in shape)
            if (answers[0][0] == 'ok') is not documented:
                wrong.append((walk, 'documented steps', answers[0]))
            elif documented:
                if not reached:
                    wrong.append((walk, 'not on the right contact'))
            elif twin or shape in ('padded-trailing', 'padded-leading', 'two-gateways-one-padded'):
                # The number itself is corrected; the request then finds its own handle moved, or succeeds.
                last = ('refused', 'identity_preimage_changed') if held == 'held' else ('ok',)
                if ([answer[0] for answer in answers[1:-1]] != ['ok'] or answers[-1][:len(last)] != last
                        or not reached):
                    wrong.append((walk, 'the number itself', answers[1:]))
    assert wrong == []

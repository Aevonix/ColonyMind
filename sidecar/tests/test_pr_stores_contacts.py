"""One phone/email resolver, including installed stores with split identities."""
import pytest

from colonymind.contacts.config import ContactsConfig
from colonymind.contacts.store import SQLiteContactStore
from colonymind.identity.participants import ParticipantResolver


@pytest.fixture
async def store(tmp_path):
    store = SQLiteContactStore(config=ContactsConfig(sqlite_path=str(tmp_path / 'contacts.db')))
    await store.connect()
    try:
        yield store
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_add_handle_refuses_a_phone_owned_on_another_gateway(store):
    first = await store.create(display_name='First')
    second = await store.create(display_name='Second')
    await store.add_handle(first.contact_id, 'sms', '+10000000000')
    with pytest.raises(ValueError, match='already assigned'):
        await store.add_handle(second.contact_id, 'whatsapp', '10000000000@s.whatsapp.net')
    # The same owner may keep multiple transport addresses for the number.
    await store.add_handle(first.contact_id, 'whatsapp', '10000000000@s.whatsapp.net')
    assert (await store.resolve_messaging_handle('rcs', '+1 (000) 000-0000')).contact_id == first.contact_id


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['phone', 'email'])
async def test_legacy_split_is_ambiguous_through_every_lookup(store, kind):
    first = await store.create(display_name='First')
    second = await store.create(display_name='Second')
    gateway, reference = ('sms', '+10000000000') if kind == 'phone' else ('email', 'fixture@example.invalid')
    other_gateway, other_address = ('whatsapp', '10000000000@s.whatsapp.net') if kind == 'phone' else (
        'email', 'FIXTURE@EXAMPLE.INVALID')
    await store.add_handle(first.contact_id, gateway, reference, verified=True)
    # Model an installed store's already split rows without changing its schema.
    db = store._require_db()
    await db.execute('''INSERT INTO contact_handles
        (handle_id,contact_id,gateway,address,is_primary,verified,confidence,source,created_at)
        VALUES ('legacy-split',?,?,?,0,1,1.0,'manual','2026-01-01T00:00:00Z')''',
        (second.contact_id, other_gateway, other_address))
    await db.commit()
    assert await store.resolve_handle(gateway, reference) is None
    assert await store.resolve_reference(reference) is None
    assert await store.resolve_reference(gateway + ':' + reference, exact=True) is None
    assert await store.resolve_messaging_handle(gateway, reference) is None
    assert await store.resolve_verified_handles(gateway, [reference]) is None
    assert await store._match_exact(gateway, reference) is None
    assert await store._handle_holder(gateway, reference) is None
    assert await store.search(reference) == []
    assert await store.search(gateway + ':' + reference) == []
    async with db.execute('SELECT count(*) FROM contact_handles') as cursor:
        assert (await cursor.fetchone())[0] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('gateway', ['sms', 'rcs', 'whatsapp', 'signal'])
@pytest.mark.parametrize('configured', [False, True])
async def test_explicit_owner_binding_survives_a_legacy_split(store, tmp_path, monkeypatch, gateway, configured):
    monkeypatch.setenv('COLONYMIND_HOME', str(tmp_path))
    monkeypatch.setenv('COLONYMIND_STATE_DIR', str(tmp_path))
    owner = await store.create(display_name='Owner')
    other = await store.create(display_name='Older record')
    number = '+10000000000'
    monkeypatch.setenv('COLONYMIND_OWNER_CONTACT_ID', owner.contact_id)
    monkeypatch.setenv('COLONYMIND_OWNER_SMS_HANDLE', number if configured else '')
    monkeypatch.setenv('COLONYMIND_OWNER_RCS_HANDLE', number if configured else '')
    await store.add_handle(owner.contact_id, gateway, number, verified=True)
    db = store._require_db()
    await db.execute('''INSERT INTO contact_handles
        (handle_id,contact_id,gateway,address,is_primary,verified,confidence,source,created_at)
        VALUES ('legacy-owner-split',?,'voice',?,0,1,1.0,'manual','2026-01-01T00:00:00Z')''',
        (other.contact_id, number))
    await db.commit()
    before = db.total_changes
    result = await ParticipantResolver(store).resolve(platform=gateway, user_id=number)
    assert (result.contact_id, result.method, result.created) == (owner.contact_id, 'handle', False)
    assert db.total_changes == before
    # The owner exception is for live channel bindings; references still obey C1.
    assert await store.resolve_reference(number) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('legacy_split', [False, True])
async def test_phone_correction_moves_the_whole_number_and_retries_do_not_move_it_back(store, legacy_split):
    first = await store.create(display_name='First')
    second = await store.create(display_name='Second')
    number, jid = '+10000000000', '10000000000@s.whatsapp.net'
    await store.add_handle(first.contact_id, 'sms', number, verified=True)
    await store.add_handle(first.contact_id, 'whatsapp', jid, verified=True)
    # A bare numeric id on a different gateway needs its own exact correction.
    await store.add_handle(first.contact_id, 'telegram', '10000000000', verified=True)
    db = store._require_db()
    if legacy_split:
        await db.execute('''INSERT INTO contact_handles
            (handle_id,contact_id,gateway,address,is_primary,verified,confidence,source,created_at)
            VALUES ('legacy-correction',?,'signal','+1 (000) 000-0000',0,1,1,'manual','2026-01-01')''',
            (second.contact_id,))
        await db.commit()
    await store.propose_handle_link(first.contact_id, 'sms', number)
    await store.propose_handle_link(second.contact_id, 'sms', number)
    args = dict(operation_id='whole-phone', performed_by='owner-fixture', gateway='whatsapp',
        address=jid, expected_contact_id=first.contact_id, contact_id=second.contact_id,
        evidence_refs=['fixture:phone-correction'], affected_source_ids=['fixture:turn'])
    with pytest.raises(ValueError, match='^identity_phone_ambiguous'):
        await store.correct_handle_identity(**args)
    await store.correct_handle_identity(operation_id='exact-numeric-id', performed_by='owner-fixture',
        gateway='telegram', address='10000000000', expected_contact_id=first.contact_id,
        contact_id=second.contact_id, evidence_refs=['fixture:exact-correction'])
    result = await store.correct_handle_identity(**args)
    for gateway, address in [('sms', number), ('whatsapp', jid), ('signal', number), ('rcs', number)]:
        assert (await store.resolve_messaging_handle(gateway, address)).contact_id == second.contact_id
    assert result['old_contact_id'] == first.contact_id
    assert result['affected_source_ids'] == ['fixture:turn']
    assert result['source_reconciliation_required'] is True
    assert result['authority_granted'] is False
    assert (await store.resolve_messaging_handle('telegram', '10000000000')).contact_id == second.contact_id
    async with db.execute('SELECT contact_id,status FROM contact_identity_candidates') as cur:
        assert dict(await cur.fetchall()) == {first.contact_id: 'rejected', second.contact_id: 'confirmed'}
    await store.close()
    await store.connect()
    assert await store.correct_handle_identity(**args) == result
    with pytest.raises(ValueError, match='identity_operation_conflict'):
        await store.correct_handle_identity(**(args | {'contact_id': first.contact_id}))
    with pytest.raises(ValueError, match='identity_preimage_changed'):
        await store.correct_handle_identity(**(args | {'operation_id': 'stale-preimage'}))
    await store.correct_handle_identity(operation_id='reverse-numeric-id', performed_by='owner-fixture',
        gateway='telegram', address='10000000000', expected_contact_id=second.contact_id,
        contact_id=first.contact_id, evidence_refs=['fixture:exact-correction'])
    await store.correct_handle_identity(**(args | {'operation_id': 'reverse-whole-phone',
        'expected_contact_id': second.contact_id, 'contact_id': first.contact_id}))
    assert await store.correct_handle_identity(**args) == result
    for gateway in ('sms', 'rcs', 'signal'):
        assert (await store.resolve_messaging_handle(gateway, number)).contact_id == first.contact_id
    assert len(await store.list()) == 2


@pytest.mark.asyncio
async def test_phone_correction_to_a_new_gateway_is_atomic_with_its_receipt(store, monkeypatch):
    first = await store.create(display_name='First')
    second = await store.create(display_name='Second')
    number = '+10000000000'
    await store.add_handle(first.contact_id, 'sms', number, verified=True)
    args = dict(operation_id='new-gateway', performed_by='owner-fixture', gateway='rcs', address=number,
        expected_contact_id=None, contact_id=second.contact_id, evidence_refs=['fixture:phone-correction'])
    original_open = store._open_durable_connection

    async def fail_commit():
        raise RuntimeError('fixture:commit-failure')

    async def failing_connection():
        db = await original_open()
        monkeypatch.setattr(db, 'commit', fail_commit)
        return db

    monkeypatch.setattr(store, '_open_durable_connection', failing_connection)
    with pytest.raises(RuntimeError, match='fixture:commit-failure'):
        await store.correct_handle_identity(**args)
    assert (await store.resolve_messaging_handle('sms', number)).contact_id == first.contact_id
    assert await store.get_handles(second.contact_id) == []
    async with store._require_db().execute('SELECT count(*) FROM contact_identity_operations') as cur:
        assert (await cur.fetchone())[0] == 0
    monkeypatch.setattr(store, '_open_durable_connection', original_open)
    await store.correct_handle_identity(**args)
    assert (await store.resolve_messaging_handle('sms', number)).contact_id == second.contact_id
    assert (await store.resolve_messaging_handle('rcs', number)).contact_id == second.contact_id


@pytest.fixture
async def split_phone(store):
    first = await store.create(display_name='First')
    second = await store.create(display_name='Second')
    await store.add_handle(first.contact_id, 'sms', '+10000000000', verified=True)
    db = store._require_db()
    await db.execute('''INSERT INTO contact_handles
        (handle_id,contact_id,gateway,address,is_primary,verified,confidence,source,created_at)
        VALUES ('legacy-message',?,'whatsapp','10000000000@s.whatsapp.net',0,1,1,'manual','2026-01-01')''',
        (second.contact_id,))
    await db.commit()
    return first, second


@pytest.mark.asyncio
@pytest.mark.parametrize('gateway,address', [('sms', '+10000000000'),
    ('whatsapp', '10000000000@s.whatsapp.net'), ('rcs', '+1 (000) 000-0000'), ('signal', '+10000000000')])
async def test_ambiguous_phone_messages_create_nothing(store, split_phone, monkeypatch, caplog, gateway, address):
    from unittest.mock import AsyncMock
    first, second = split_phone
    scope = await store.create_scope(scope_type='group', platform=gateway, external_id='group')
    await store.add_scope_member(scope.scope_id, first.contact_id, 'member')
    create = AsyncMock(wraps=store.create)
    delete = AsyncMock(wraps=store.hard_delete)
    add = AsyncMock(wraps=store.add_handle)
    monkeypatch.setattr(store, 'create', create)
    monkeypatch.setattr(store, 'hard_delete', delete)
    monkeypatch.setattr(store, 'add_handle', add)
    db = store._require_db()
    before = db.total_changes
    resolver = ParticipantResolver(store)
    for _ in range(3):
        result = await resolver.resolve(platform=gateway, user_id=address, display_name='First', group_id='group')
        assert (result.contact_id, result.method, result.created, result.proposal_filed) == (None, 'none', False, False)
        assert (await resolver.lookup(platform=gateway, user_id=address)).contact_id is None
    create.assert_not_awaited()
    delete.assert_not_awaited()
    add.assert_not_awaited()
    assert db.total_changes == before
    assert len(await store.list()) == 2
    assert not any(record.exc_info for record in caplog.records)
    assert await store.list_handle_proposals() == []
    await store.correct_handle_identity(operation_id='repair-legacy', performed_by='owner-fixture',
        gateway='sms', address='+10000000000', expected_contact_id=first.contact_id,
        contact_id=second.contact_id, evidence_refs=['fixture:legacy-repair'])
    assert (await resolver.resolve(platform=gateway, user_id=address)).contact_id == second.contact_id
    create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['add', 'confirm'])
async def test_ambiguous_phone_cannot_gain_a_sender_or_proposed_handle(store, split_phone, operation):
    first, _ = split_phone
    candidate = await store.propose_handle_link(first.contact_id, 'signal', '+10000000000')
    db = store._require_db()
    before = db.total_changes
    with pytest.raises(ValueError, match='ambiguous'):
        if operation == 'add':
            await store.add_handle(first.contact_id, 'signal', '+10000000000', source='auto:sender')
        else:
            await store.confirm_link(candidate['candidate_id'], performed_by='owner-fixture')
    assert db.total_changes == before
    async with db.execute('SELECT count(*) FROM contact_handles') as cur:
        assert (await cur.fetchone())[0] == 2


@pytest.mark.asyncio
async def test_phone_correction_keeps_same_gateway_opaque_ids_with_their_person(store):
    first = await store.create(display_name='First')
    second = await store.create(display_name='Second')
    number = '+10000000000'
    await store.add_handle(first.contact_id, 'sms', number)
    await store.add_handle(first.contact_id, 'whatsapp', '10000000000@s.whatsapp.net')
    lid = await store.add_handle(first.contact_id, 'whatsapp', '10000000000@lid', verified=True)
    candidate = await store.propose_handle_link(first.contact_id, 'whatsapp', '10000000000@lid')
    before = await store.get_handles(first.contact_id)
    with pytest.raises(ValueError, match='^identity_phone_ambiguous') as caught:
        await store.correct_handle_identity(operation_id='phone-not-lid', performed_by='owner-fixture',
            gateway='whatsapp', address='10000000000@s.whatsapp.net', expected_contact_id=first.contact_id,
            contact_id=second.contact_id, evidence_refs=['fixture:phone-not-lid'])
    assert 'whatsapp:' + lid.handle_id in str(caught.value)
    assert (await store.resolve_messaging_handle('sms', number)).contact_id == first.contact_id
    assert await store.get_handles(first.contact_id) == before
    assert (await store.resolve_messaging_handle('whatsapp', '10000000000@lid')).contact_id == first.contact_id
    assert (await store.propose_handle_link(first.contact_id, 'whatsapp', '10000000000@lid')) == candidate


def test_canonical_lookup_contains_its_four_sql_branches():
    import ast
    import inspect
    import textwrap

    function = ast.parse(textwrap.dedent(inspect.getsource(SQLiteContactStore._match_canonical))).body[0]
    base = next((node.value.value for node in function.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'base' for target in node.targets)
                 and isinstance(node.value, ast.Constant)), None)
    assert base is not None and base.startswith('SELECT c.* FROM contacts c JOIN contact_handles h ')
    branches = [node for node in ast.walk(function) if isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Add) and isinstance(node.left, ast.Name)
                and node.left.id == 'base' and isinstance(node.right, ast.Constant)]
    assert len(branches) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize('gateway,address', [
    ('email', 'fixture@example.invalid'), ('sms', '+10000000000'),
    ('telegram', '10000000000'), ('matrix', 'fixture-user'),
])
async def test_canonical_lookup_can_return_the_matching_handles(store, gateway, address):
    person = await store.create(display_name='Fixture')
    handle = await store.add_handle(person.contact_id, gateway, address)
    rows = await store._match_canonical(gateway, address, all_matches=True)
    assert [(row['contact_id'], row['gateway'], row['handle_id']) for row in rows] == [
        (person.contact_id, gateway, handle.handle_id)]
    assert (await store._match_canonical(gateway, address)).contact_id == person.contact_id
    assert await store._match_canonical(gateway, '', all_matches=True) == []
    assert await store._match_canonical(gateway, '') is None


@pytest.mark.asyncio
@pytest.mark.parametrize('existing_handle', [False, True])
async def test_phone_correction_refuses_all_residual_holders_and_can_retry(store, existing_handle):
    first = await store.create(display_name='First')
    target = await store.create(display_name='Target')
    third = await store.create(display_name='Third')
    number, jid = '+10000000000', '10000000000@s.whatsapp.net'
    phone_handles = [await store.add_handle(first.contact_id, 'sms', number),
                     await store.add_handle(first.contact_id, 'whatsapp', jid)]
    conflicts = [await store.add_handle(third.contact_id, gateway, address, verified=True)
                 for gateway, address in [('whatsapp', '10000000000@lid'),
                                          ('telegram', '10000000000'),
                                          ('matrix', 'tel:+10000000000')]]
    await store.propose_handle_link(target.contact_id, 'whatsapp', jid)
    args = dict(operation_id='retry-whole-number', performed_by='owner-fixture',
                gateway='whatsapp' if existing_handle else 'rcs', address=jid if existing_handle else number,
                expected_contact_id=first.contact_id if existing_handle else None,
                contact_id=target.contact_id, evidence_refs=['fixture:correction'])

    async def snapshot():
        result = {}
        for table in ('contact_handles', 'contact_identity_candidates', 'contact_audit', 'contact_identity_operations'):
            async with store._require_db().execute(f'SELECT * FROM {table} ORDER BY rowid') as cur:
                result[table] = [tuple(row) for row in await cur.fetchall()]
        return result

    before = await snapshot()
    with pytest.raises(ValueError, match='^identity_phone_ambiguous') as caught:
        await store.correct_handle_identity(**args)
    for handle in conflicts:
        assert f'{handle.gateway}:{handle.handle_id}' in str(caught.value)
    assert number[1:] not in str(caught.value)
    assert await snapshot() == before
    for handle in conflicts:
        await store.correct_handle_identity(operation_id='exact-' + handle.handle_id, performed_by='owner-fixture',
            gateway=handle.gateway, address=handle.address, expected_contact_id=third.contact_id,
            contact_id=target.contact_id, evidence_refs=['fixture:exact'])
    result = await store.correct_handle_identity(**args)
    assert {row['handle_id'] for row in result['moved_handles']} == {handle.handle_id for handle in phone_handles}
    for gateway in ('sms', 'whatsapp', 'telegram', 'matrix', 'rcs'):
        assert (await store.resolve_messaging_handle(gateway, number)).contact_id == target.contact_id
    after = await snapshot()
    assert await store.correct_handle_identity(**args) == result
    assert await snapshot() == after


@pytest.mark.asyncio
async def test_phone_correction_checks_its_uncommitted_rows_through_the_lookup(store, monkeypatch):
    first = await store.create(display_name='First')
    target = await store.create(display_name='Target')
    number = '+10000000000'
    await store.add_handle(first.contact_id, 'sms', number)
    await store.add_handle(first.contact_id, 'whatsapp', '10000000000@s.whatsapp.net')
    original = store._match_canonical
    checked = []

    async def lookup(gateway, address, **kwargs):
        rows = await original(gateway, address, **kwargs)
        if kwargs.get('conn') is not None:
            assert kwargs['conn'] is not store._require_db()
            assert kwargs['conn'].in_transaction
            assert address == number and kwargs['all_matches']
            assert {row['contact_id'] for row in rows} == {target.contact_id}
            checked.append(gateway)
        return rows

    monkeypatch.setattr(store, '_match_canonical', lookup)
    await store.correct_handle_identity(operation_id='lookup-transaction', performed_by='owner-fixture',
        gateway='sms', address=number, expected_contact_id=first.contact_id,
        contact_id=target.contact_id, evidence_refs=['fixture:lookup'])
    assert sorted(checked) == ['sms', 'whatsapp']


async def legacy_typed_handle(store):
    first = await store.create(display_name='First')
    second = await store.create(display_name='Second')
    sms = await store.add_handle(first.contact_id, 'sms', '+10000000000')
    db = store._require_db()
    await db.execute('''INSERT INTO contact_handles
        (handle_id,contact_id,gateway,address,is_primary,verified,confidence,source,created_at)
        VALUES ('legacy-typed',?,'whatsapp','(000) 000-0000',0,1,1,'manual','2026-01-01')''',
        (first.contact_id,))
    await db.commit()
    return first, second, sms


async def identity_snapshot(store):
    result = {}
    for table in ('contact_handles', 'contact_identity_candidates', 'contact_audit', 'contact_identity_operations'):
        async with store._require_db().execute(f'SELECT * FROM {table} ORDER BY rowid') as cur:
            result[table] = [dict(row) for row in await cur.fetchall()]
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize('remove', [False, True])
async def test_exact_legacy_correction_records_stored_address_and_allows_retry(store, remove):
    import hashlib

    first, second, sms = await legacy_typed_handle(store)
    whole = dict(operation_id='whole-legacy', performed_by='owner-fixture', gateway='sms',
        address='+10000000000', expected_contact_id=first.contact_id, contact_id=second.contact_id,
        evidence_refs=['fixture:whole'])
    with pytest.raises(ValueError, match='^identity_phone_ambiguous'):
        await store.correct_handle_identity(**whole)
    before = await identity_snapshot(store)
    exact = dict(operation_id='exact-legacy', performed_by='owner-fixture', gateway=' WHATSAPP ',
        address=' (000) 000-0000 ', expected_contact_id=first.contact_id,
        contact_id=None if remove else second.contact_id, evidence_refs=['fixture:exact'],
        affected_source_ids=['fixture:source'])
    with pytest.raises(ValueError, match='^identity_correction_requires_evidence$'):
        await store.correct_handle_identity(**(exact | {'evidence_refs': []}))
    assert await identity_snapshot(store) == before
    result = await store.correct_handle_identity(**exact)
    after = await identity_snapshot(store)
    rows = {row['handle_id']: row for row in after['contact_handles']}
    assert rows[sms.handle_id] == next(row for row in before['contact_handles'] if row['handle_id'] == sms.handle_id)
    assert set(rows) == ({sms.handle_id} if remove else {sms.handle_id, 'legacy-typed'})
    if not remove:
        assert rows['legacy-typed']['address'] == '(000) 000-0000'
        assert rows['legacy-typed']['contact_id'] == second.contact_id
    assert result['handle_id'] == 'legacy-typed'
    assert result['address_sha256'] == hashlib.sha256(b'(000) 000-0000').hexdigest()
    assert result['old_contact_id'] == first.contact_id
    assert result['contact_id'] == exact['contact_id']
    assert result['gateway'] == 'whatsapp'
    assert result['evidence_refs'] == exact['evidence_refs']
    assert result['affected_source_ids'] == exact['affected_source_ids']
    assert result['source_reconciliation_required'] is True
    assert result['authority_granted'] is False
    await store.close()
    await store.connect()
    assert await store.correct_handle_identity(**exact) == result
    # Equivalent normalized requests still share the existing replay contract.
    assert await store.correct_handle_identity(**(exact | {'address': '0000000000'})) == result
    with pytest.raises(ValueError, match='^identity_operation_conflict$'):
        await store.correct_handle_identity(**(exact | {'evidence_refs': ['fixture:changed']}))
    assert await identity_snapshot(store) == after
    await store.correct_handle_identity(**whole)
    for gateway in ('sms', 'whatsapp'):
        assert (await store.resolve_messaging_handle(gateway, '+10000000000')).contact_id == second.contact_id


@pytest.mark.asyncio
async def test_exact_legacy_correction_checks_the_stored_holder_before_writes(store):
    first, second, _ = await legacy_typed_handle(store)
    before = await identity_snapshot(store)
    with pytest.raises(ValueError, match='^identity_preimage_changed$'):
        await store.correct_handle_identity(operation_id='wrong-legacy-holder', performed_by='owner-fixture',
            gateway='whatsapp', address='(000) 000-0000', expected_contact_id=None,
            contact_id=second.contact_id, evidence_refs=['fixture:preimage'])
    assert await identity_snapshot(store) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('operation', ['merge', 'confirm_self', 'confirm_shadow'])
async def test_direct_correction_callers_can_use_a_stored_legacy_address(store, operation):
    import hashlib
    import json

    first, second, _ = await legacy_typed_handle(store)
    db = store._require_db()
    if operation == 'merge':
        await store.merge(second.contact_id, first.contact_id, performed_by='owner-fixture')
        operation_id = f'merge:{first.contact_id}:legacy-typed'
        target_id = second.contact_id
    else:
        target_id = first.contact_id if operation == 'confirm_self' else second.contact_id
        candidate = await store.propose_handle_link(target_id, 'whatsapp', '(000) 000-0000',
                                                     evidence_refs=['fixture:legacy-candidate'])
        # Model a candidate saved before addresses were normalized.
        await db.execute('UPDATE contact_identity_candidates SET address=? WHERE candidate_id=?',
                         ('(000) 000-0000', candidate['candidate_id']))
        if operation == 'confirm_shadow':
            await db.execute("UPDATE contacts SET import_source='auto:sender' WHERE contact_id=?", (first.contact_id,))
        await db.commit()
        confirmed = await store.confirm_link(candidate['candidate_id'], performed_by='owner-fixture')
        assert confirmed['status'] == 'confirmed'
        assert confirmed['merged'] == (operation == 'confirm_shadow')
        operation_id = (f'merge:{first.contact_id}:legacy-typed' if operation == 'confirm_shadow'
                        else f'link:{candidate["candidate_id"]}')
    async with db.execute('SELECT * FROM contact_handles WHERE handle_id=?', ('legacy-typed',)) as cur:
        handle = await cur.fetchone()
    assert handle['contact_id'] == target_id
    assert handle['address'] == '(000) 000-0000'
    async with db.execute('SELECT result_json FROM contact_identity_operations WHERE operation_id=?',
                          (operation_id,)) as cur:
        receipt = json.loads((await cur.fetchone())['result_json'])
    assert receipt['handle_id'] == 'legacy-typed'
    assert receipt['address_sha256'] == hashlib.sha256(b'(000) 000-0000').hexdigest()


@pytest.mark.asyncio
async def test_phone_refusal_names_each_conflicting_holder_without_addresses(store):
    first, second, _ = await legacy_typed_handle(store)
    third = await store.create(display_name='Third')
    numeric = await store.add_handle(third.contact_id, 'telegram', '10000000000')
    before = await identity_snapshot(store)
    with pytest.raises(ValueError) as caught:
        await store.correct_handle_identity(operation_id='refusal-holders', performed_by='owner-fixture',
            gateway='sms', address='+10000000000', expected_contact_id=first.contact_id,
            contact_id=second.contact_id, evidence_refs=['fixture:refusal'])
    conflicts = [f'whatsapp:legacy-typed:{first.contact_id}',
                 f'telegram:{numeric.handle_id}:{third.contact_id}']
    assert str(caught.value) == 'identity_phone_ambiguous ' + ', '.join(sorted(conflicts))
    for address in ('+10000000000', '10000000000', '(000) 000-0000', '0000000000'):
        assert address not in str(caught.value)
    assert await identity_snapshot(store) == before

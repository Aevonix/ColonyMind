"""Trusted durable intake, without another payload store or model-callable ACK."""
from datetime import datetime, timezone
from contextlib import closing
import asyncio
import json
import logging
import re
import sqlite3
from weakref import WeakKeyDictionary

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from colonymind.api.auth import KEY_PRINCIPAL, request_authority, speaks_for_owner

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/ingress')
_receipts_locks = WeakKeyDictionary()
_receipts_tasks = set()


class Metadata(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    channel: str = Field(pattern='^[a-z][a-z0-9_.-]{0,31}$')
    sender_ref: str = Field(min_length=1, max_length=512)
    reply_to_ref: str = Field(default='', max_length=256)
    from_owner: bool = False
    is_group: bool = False


class Admission(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    account_id: str = Field(min_length=1, max_length=128)
    epoch: str = Field(min_length=1, max_length=128)
    sequence: int = Field(ge=1)
    event_id: str = Field(min_length=1, max_length=256)
    occurred_at: float = Field(gt=0)
    journal_ref: str = Field(min_length=1, max_length=512)
    payload_digest: str = Field(pattern='^[a-f0-9]{64}$')
    media_available: bool
    metadata: Metadata


class NativeTurn(BaseModel):
    model_config = ConfigDict(extra='forbid')
    session_id: str = Field(min_length=1, max_length=256)
    task_id: str = Field(min_length=1, max_length=256)
    turn_id: str = Field(min_length=1, max_length=256)


class Handoff(BaseModel):
    model_config = ConfigDict(extra='forbid')
    receipt_ids: list[str] = Field(min_length=1, max_length=100)
    batch_id: str = Field(min_length=1, max_length=128)
    native_turn: NativeTurn | None = None


class NativeSettled(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    native_turn: NativeTurn
    interrupted: bool


class Coverage(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    account_id: str = Field(min_length=1, max_length=128)
    epoch: str = Field(min_length=1, max_length=128)
    connected_since: float | None = Field(default=None, gt=0)
    observed_at: float = Field(gt=0)
    watermark: int = Field(ge=0)
    sequence_floor: int = Field(default=0, ge=0)
    connected: bool
    unavailable: int = Field(ge=0)


def store():
    from colonymind.api.routers import host
    if host._comms_log is None:
        raise HTTPException(503, detail='communications_unavailable')
    return host._comms_log._transport_ingress


def producer(request):
    auth = request_authority(request)
    if not auth.authenticated or auth.anonymous:
        raise HTTPException(403, detail='trusted_transport_producer_required')
    return auth.principal_id


def guarded(call):
    try:
        return call()
    except ValueError as exc:
        raise HTTPException(409, detail=str(exc)) from None
    except (OSError, sqlite3.Error):
        raise HTTPException(503, detail='transport_intake_unavailable') from None


async def verified_contact(metadata):
    from colonymind.api.routers import host
    if host._contacts_store is None:
        raise HTTPException(503, detail='contacts_unavailable')
    if metadata.is_group:
        return None
    addresses = [metadata.sender_ref]
    # PN and LID are separate namespaces. Never infer a phone number from LID,
    # a display name, another channel, or model-selected contact identifiers.
    if metadata.channel == 'whatsapp':
        match = re.fullmatch(r'([0-9]{1,32})@s\.whatsapp\.net', metadata.sender_ref)
        if match:
            addresses += [match[1], '+' + match[1]]
        elif not re.fullmatch(r'[0-9]{1,32}@lid', metadata.sender_ref):
            return None
    try:
        contact = await host._contacts_store.resolve_verified_handles(metadata.channel, addresses)
    except NotImplementedError:
        raise HTTPException(503, detail='verified_transport_resolution_unavailable') from None
    return contact.contact_id if contact is not None else None


@router.post('/admit')
async def admit(body: Admission, request: Request):
    principal = producer(request)
    contact_id = await verified_contact(body.metadata)
    ingress = store()
    value = guarded(lambda: ingress.admit(producer=principal, contact_id=contact_id, **body.model_dump()))
    row = ingress.get(value['receipt_id'])
    if row['native_turn_json'] or row['source_versions_json']:
        from colonymind import get_state_dir
        from colonymind.turns import get_turn_idempotency_ledger
        if source_forgotten(get_turn_idempotency_ledger(get_state_dir()), row):
            value = {**value, 'state': 'erased', 'source_versions_json': None}
    if contact_id and not body.metadata.from_owner and value['state'] != 'erased':
        # Receipt admission and exact positive reply reconciliation finish
        # before intake ACK. A partial failure is retried idempotently.
        from .transport import TransportReceipt, observe
        await observe(TransportReceipt(event_id=value['receipt_id'], contact_id=contact_id,
            channel=body.metadata.channel, direction='in', external_ref=body.event_id,
            reply_to_ref=body.metadata.reply_to_ref, receipt_ref=value['receipt_id'],
            occurred_at=datetime.fromtimestamp(body.occurred_at, timezone.utc).isoformat(),
            status='received'), request)
    return value


@router.post('/handoff')
async def handoff(body: Handoff, request: Request):
    principal = producer(request)
    ingress = store()
    from colonymind import get_state_dir
    from colonymind.turns import get_turn_idempotency_ledger
    ledger = get_turn_idempotency_ledger(get_state_dir())
    def apply():
        rows = ingress.receipt_rows(producer=principal, receipt_ids=body.receipt_ids, ledger=ledger)
        if any(source_forgotten(ledger, row) for row in rows):
            raise ValueError('source_forgotten')
        return ingress.handoff(producer=principal, **body.model_dump())
    return guarded(apply)


@router.post('/coverage')
async def record_coverage(body: Coverage, request: Request):
    principal = producer(request)
    guarded(lambda: store().observe_coverage(producer=principal, **body.model_dump()))
    from colonymind.self_model import reply_forecasts
    await asyncio.to_thread(reply_forecasts.safe, reply_forecasts.reconcile_existing,
                            producer=principal, coverage=body.model_dump())
    return {'recorded': True, 'effect_authorized': False}


@router.post('/native-settled')
async def native_settled(body: NativeSettled, request: Request):
    principal = producer(request)
    authority = request_authority(request)
    # The Hermes host uses api.key; durable transports can carry their own
    # issued credentials. Only that unscoped owner host can release another
    # producer's exact native tuple, never a body-selected producer or session.
    host = principal == KEY_PRINCIPAL and authority.scopes is None and speaks_for_owner(request)
    return guarded(lambda: store().native_settled(producer=principal, host=host, **body.model_dump()))


@router.get('/receipts')
async def receipts(request: Request, ids: str = Query(min_length=1, max_length=9000)):
    principal = producer(request)
    selected = ids.split(',')
    store()
    from colonymind.api.routers import host
    db_path = host._comms_log._db_path
    # Queue on the loop, not in its shared executor. A cancelled HTTP waiter
    # must not release the slot while its SQLite worker is still running.
    loop = asyncio.get_running_loop()
    lock = _receipts_locks.setdefault(loop, asyncio.Lock())
    await lock.acquire()
    try:
        task = asyncio.create_task(asyncio.to_thread(read_receipts, db_path, principal, selected))
        _receipts_tasks.add(task)

        def finished(task):
            lock.release()
            _receipts_tasks.discard(task)
            if not task.cancelled():
                task.exception()  # Retrieve errors even if the HTTP waiter left.

        task.add_done_callback(finished)
    except BaseException:
        lock.release()
        raise
    return await asyncio.shield(task)


def read_receipts(db_path, principal, selected):
    from colonymind.contacts.transport_ingress import TransportIngress
    # The host's CommsLog connection belongs to its event-loop thread. Each
    # poll owns a connection, including all settlement transactions, here.
    with closing(guarded(lambda: sqlite3.connect(db_path, uri=True, timeout=2.0))) as conn:
        conn.row_factory = sqlite3.Row
        ingress = TransportIngress(conn)
        from colonymind import get_state_dir
        from colonymind.turns import get_turn_idempotency_ledger
        rows = guarded(lambda: ingress.receipt_rows(producer=principal, receipt_ids=selected,
                       ledger=get_turn_idempotency_ledger(get_state_dir())))
        ledger = get_turn_idempotency_ledger(get_state_dir())
        erased = {row['erased_source_id'] for row in rows if row['erased_source_id']}
        changed = bool(erased)
        if erased:
            guarded(lambda: ingress.erase_sources(erased))
        for row in rows:
            if changed:
                row = guarded(lambda: ingress.get(row['receipt_id']))
            if needs_settlement(row) and not source_forgotten(ledger, row):
                guarded(lambda: settle_receipt(ingress, row))
                # One native completion can also settle other receipts on this page.
                changed = True
        if changed:
            rows = [guarded(lambda row=row: ingress.get(row['receipt_id'])) for row in rows]
        items = []
        for row in rows:
            # A curated withdrawal changes the served receipt only. Its metadata,
            # source bindings and native settlement stay retained for inspection.
            value = ingress.view(row)
            if source_forgotten(ledger, row):
                value = {**value, 'state': 'erased', 'source_versions_json': None}
            items.append(value)
        return {'items': items}


def source_forgotten(ledger, row):
    from colonymind.memory import curated
    native = json.loads(row['native_turn_json'] or '{}')
    refs = [{'source_id': source_id} for source_id in json.loads(row['source_versions_json'] or '{}')]
    if native.get('turn_id'):
        refs.append({'source_id': native['turn_id']})
    return curated.is_forgotten(ledger, refs)


def needs_settlement(row):
    return bool(row and row['native_turn_json'] and row['contact_id'] and row['state'] != 'erased'
                and (row['state'] != 'completed' or not row['metadata_settled']))


def settle_forecasts(ingress, row):
    from colonymind.self_model import reply_forecasts
    try:
        reply_forecasts.reconcile_existing(receipt=row, strict=True)
    except Exception as error:
        logger.warning('Transport receipt forecast settlement remains pending: %s', type(error).__name__)
        return
    ingress.mark_settled(row)


def settle_receipt(ingress, row):
    """Retry metadata settlement on the existing receiver's receipt read."""
    if not needs_settlement(row):
        return
    from colonymind import get_state_dir
    from colonymind.turns import get_turn_idempotency_ledger
    ledger = get_turn_idempotency_ledger(get_state_dir())
    native = json.loads(row['native_turn_json'])
    source_id = native['turn_id']
    if source_forgotten(ledger, row):
        return
    if (ledger.is_source_erased(source_id, row['contact_id'])
            or ledger.is_projection_erased(source_id)):
        ingress.erase_sources([source_id])
        return
    if row['state'] == 'completed':
        settle_forecasts(ingress, row)
        return
    with closing(ledger._connect()) as db:
        receipt = db.execute('SELECT state,response_json FROM turn_ingestion WHERE turn_id=?', (source_id,)).fetchone()
        source = db.execute('SELECT session_id FROM turn_sources WHERE turn_id=? AND contact_id=?',
                             (source_id, row['contact_id'])).fetchone()
    if source is None or source['session_id'] != native['session_id']:
        return
    if not receipt or receipt['state'] != 'completed':
        return
    result = json.loads(receipt['response_json'] or '{}')
    if not result.get('accepted') or not result.get('source_recorded'):
        return
    refs = ledger.source_references([source_id], contact_id=row['contact_id'], session_id=native['session_id'])
    if refs:
        completed = ingress.complete(native_turn=native,
            source_versions={r['source_id']: r['source_version'] for r in refs}, outcome='captured')
        for settled in completed:
            row = ingress.get(settled['receipt_id'])
            if needs_settlement(row):
                settle_forecasts(ingress, row)


def complete_source(body, result):
    """Existing accepted canonical source is the processing receipt.

    Native nonempty turn IDs are already the source IDs in the durable outbox.
    No new host payload, task-text matching or model assertion is necessary.
    This does not attest a provider send or successful external tool effect.
    """
    from colonymind.api.routers import host
    if (host._comms_log is None or not result.accepted or not result.source_recorded
            or not body.context.turn_id or body.checkpoint_messages is not None or body.source_only):
        return
    ingress = store()
    rows = ingress.for_canonical_turn(turn_id=body.context.turn_id, contact_id=body.context.contact_id)
    if not rows:
        return
    settle_receipt(ingress, rows[0])


def reconcile_source(body, result):
    # Canonical ingestion has already committed. Failure leaves the transport
    # payload pending, and the same source/outbox receipt can be reconciled later.
    try:
        complete_source(body, result)
    except (OSError, ValueError, sqlite3.Error):
        logger.warning('Transport source reconciliation remains pending', exc_info=True)

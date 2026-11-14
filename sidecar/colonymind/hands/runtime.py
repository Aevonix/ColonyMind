"""Host dependencies for hands, kept outside the durable transition rules."""
from contextlib import closing
import logging

logger = logging.getLogger(__name__)


def configure(controller):
    from colonymind import get_state_dir
    from colonymind.api.auth import owner_person_id
    from colonymind.turns import get_turn_idempotency_ledger
    try:
        controller.owner_id = owner_person_id()
        controller.ledger = get_turn_idempotency_ledger(get_state_dir())
        from colonymind.memory.branches import get_store
        from colonymind.config import load_config
        controller.branches = get_store(load_config())
    except Exception:
        logger.exception('Hands provenance dependencies unavailable')

    def parent_is_owner(session):
        if not controller.owner_id or controller.ledger is None:
            return False
        with closing(controller.ledger._connect()) as db:
            contacts = {row[0] for row in db.execute(
                "SELECT DISTINCT s.contact_id FROM turn_sources s WHERE s.session_id=? "
                "AND NOT EXISTS (SELECT 1 FROM source_attribution_invalidations i WHERE i.source_id=s.turn_id) "
                "AND EXISTS (SELECT 1 FROM json_each(s.messages_json) m WHERE json_extract(m.value,'$.role')='user')",
                (session,))}
        return contacts == {controller.owner_id}

    async def recall(*, query, audience, session):
        from colonymind.api.routers import host
        body = host.ContextAssembleRequest(
            identity=host.HostIdentity(host_id='colonymind-hands'),
            context=host.HostTurnContext(session_id=session, contact_id=controller.owner_id),
            incoming_message=host.HostMessage(role='user', content=query),
            audience=audience, include_initiatives=False)
        sections = await host._assemble_sections(body, viewer_person_id=controller.owner_id, read_only=True)
        return [{'reference': section.id, 'text': section.body} for section in sections if section.body]

    controller.parent_is_owner = parent_is_owner
    controller.recall = recall


async def expire_branches(controller):
    """Drain durable WM expiry markers, including pruning outside the sweep.

    Runs on every maintenance pass: without provenance dependencies it waits
    silently (the markers stay), and each marker is deleted once applied.
    """
    if controller.ledger is None or not controller.owner_id:
        return
    service = controller.provenance()
    with closing(controller.ledger._connect()) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='wm_expired_branches'").fetchone():
            return
        expired = list(db.execute('SELECT id,child_session FROM wm_expired_branches'))
    for handle, child_session in expired:
        await service.expire_branch(handle)
        await service.expire_branch(child_session)
        with closing(controller.ledger._connect()) as db, db:
            db.execute('DELETE FROM wm_expired_branches WHERE id=?', (handle,))


async def owner_verdict(lessons, *, owner_ref, reply_ref):
    """Only the mind's vetted verdict reader calls this reversal hook."""
    from .controller import get_controller
    from .provenance import Provenance, _owner_source
    if not owner_ref or not reply_ref:
        return
    reply = _owner_source(lessons.ledger, lessons.owner_id, reply_ref)
    verdict = _owner_source(lessons.ledger, lessons.owner_id, owner_ref)
    if not reply or not verdict or reply['session_id'] != verdict['session_id']:
        return
    hands = {ref['hand_id'] for message in reply['messages'] if message.get('role') == 'assistant'
             for ref in message.get('metadata', {}).get('hand_refs', []) if ref.get('hand_id')}
    if not hands:
        return
    controller = await get_controller()
    service = Provenance(controller, lessons.ledger, owner_id=lessons.owner_id, lessons=lessons)
    for ident in hands:
        await service.dispute(ident, owner_ref)


async def admit_hand_plan(lessons, plan, packet, judged, night):
    """Nightly hand lessons use the same admission and derivation as direct intake."""
    from .controller import get_controller
    from .provenance import Provenance, lineage_hands
    _, lineage = lessons._lineage(plan['cites'], packet)
    hands = lineage_hands(lessons.ledger, lessons.owner_id, lineage)
    for label in plan['cites']:
        row = packet['events'].get(label)
        metadata = (getattr(row, 'result_metadata', None) or {}) if row else {}
        if metadata.get('origin') == 'hand' and metadata.get('hand_id'):
            hands.add(metadata['hand_id'])
    if not hands or plan['op'] == 'retire':
        return False
    try:
        # An intake combines different hands only after a supported synthesis.
        if len(hands) != 1:
            night.count('lesson_ops_rejected')
            return True
        owner_ref = next((packet['owner'][label]['turn_id'] for label in plan['cites']
                          if label in judged and label in packet['owner']), None)
        c = await get_controller()
        service = Provenance(c, lessons.ledger, owner_id=lessons.owner_id, lessons=lessons)
        hand_id = next(iter(hands))
        # The plan names no failure: only a failed check the controller holds makes it one.
        checks = (await c.get_hand(hand_id)).get('check') or []
        failed = any(isinstance(receipt, dict) and receipt.get('passed') is False for receipt in checks)
        lesson = await service.admit_lesson(hand_id, plan['fields'], owner_ref=owner_ref,
                                            observed_failure=failed)
        night.count('lessons_admitted' if lesson else 'lesson_ops_rejected')
    except Exception:
        logger.exception('Hand lesson admission unavailable')
        night.count('lesson_ops_rejected')
    return True

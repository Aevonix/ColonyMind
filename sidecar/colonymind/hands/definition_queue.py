"""Leased, fenced pickup of app requests by the existing Hands runtime."""
from datetime import timedelta
import secrets

from fastapi import HTTPException
from . import budgets


class DefinitionQueueOperations:
    async def _queue_state(self, tid, state, **values):
        await self.execute('INSERT OR IGNORE INTO definition_task_dispatch(task_id) VALUES(?)', (tid,))
        await self.update('definition_task_dispatch', 'task_id', tid, dispatch_state=state,
                          checked_at=self.now(), **values)

    async def _release_definition_claim(self, row, state):
        if row['hand_id']:
            hand = await self.get_hand(row['hand_id'])
            if hand['state'] in {'reserved', 'planned'}:
                await self._end(hand, 'expired', 'definition_claim_released')
        await self._queue_state(row['task_id'], state, claim_token=None, lease_until=None,
                                parent_session=None, hand_id=None)
        await self.execute("UPDATE tasks SET state='open',closed_at=NULL WHERE id=? AND state='waiting'",
                           (row['task_id'],))

    async def claim_definition_tasks(self, body, key):
        # Start from the small partial queue index, not retained tasks/history.
        # This read does not reserve SQLite's writer or drain the event outbox.
        candidate_sql = ('SELECT q.task_id,q.brief,b.definition_id,b.version FROM definition_task_queue q '
                         'INDEXED BY definition_queue_claimable CROSS JOIN tasks t ON t.id=q.task_id '
                         'JOIN task_definitions b ON b.task_id=q.task_id '
                         'LEFT JOIN definition_task_dispatch d ON d.task_id=q.task_id '
                         "WHERE q.dispatched=0 AND q.owner_started=1 AND q.eligible=1 AND t.state IN ('open','waiting') "
                         'AND d.claim_token IS NULL AND (? IS NULL OR q.task_id=?) ')
        args = (body.task_id, body.task_id)
        expired_sql = ('SELECT d.* FROM definition_task_dispatch d INDEXED BY definition_dispatch_expiry '
                       'JOIN definition_task_queue q ON q.task_id=d.task_id '
                       'WHERE d.lease_until<=? AND q.dispatched=0 AND q.eligible=1')
        if (not await self.one(expired_sql + ' LIMIT 1', (self.now(),))
                and not await self.one(candidate_sql + ' LIMIT 1', args)):
            return {'tasks': []}
        async def apply():
            expired = await self.all(expired_sql, (self.now(),))
            for row in expired:
                await self._release_definition_claim(row, 'queued')
            rows = await self.all(candidate_sql + "ORDER BY coalesce(d.checked_at,''),t.created_at,t.id LIMIT 100", args)
            for row in rows:
                try:
                    await self.resolve_definition(row['definition_id'])
                except HTTPException as exc:
                    await self._queue_state(row['task_id'], 'refused:' + str(exc.detail))
                    continue
                if not body.parent_session:
                    await self._queue_state(row['task_id'], 'waiting_for_main_session')
                    continue
                token = secrets.token_hex(24)
                until = (self.clock() + timedelta(seconds=30)).isoformat()
                await self._queue_state(row['task_id'], 'claimed', claim_token=token, lease_until=until,
                                        parent_session=body.parent_session, hand_id=None)
                return {'tasks': [{**row, 'claim_token': token, 'lease_until': until}]}
            return {'tasks': []}
        # A claim is single-use: HTTP retries must never replay an expired grant.
        return await self.write(key, '/definitions/queue/claim', body, apply, record=False, skip_unchanged=True)

    async def definition_claim(self, tid, token, parent_session=None):
        row = await self.one('SELECT d.*,q.dispatched,q.owner_started,q.eligible FROM definition_task_dispatch d '
                             'JOIN definition_task_queue q ON q.task_id=d.task_id WHERE d.task_id=?', (tid,))
        if (not row or row['dispatched'] or not row['owner_started'] or not row['eligible']
                or not token or row['claim_token'] != token
                or row['lease_until'] <= self.now()
                or (parent_session is not None and row['parent_session'] != parent_session)):
            raise HTTPException(409, 'definition_claim_invalid')
        return row

    async def release_definition_claim(self, body, key):
        async def apply():
            row = await self.definition_claim(body.task_id, body.claim_token)
            state = 'waiting_for_main_session' if body.code == 'waiting_for_main_session' else 'refused:' + body.code
            await self._release_definition_claim(row, state)
            return {'released': True}
        return await self.write(key, '/definitions/queue/release', body, apply, record=False)

    async def bind_definition_claim(self, body, key):
        async def apply():
            row = await self.definition_claim(body.task_id, body.claim_token)
            if row['hand_id'] != body.hand_id:
                raise HTTPException(409, 'definition_claim_hand_mismatch')
            hand = await self.guard(body.hand_id, states={'reserved'})
            if hand['parent_session'] != row['parent_session']:
                raise HTTPException(409, 'definition_claim_parent_mismatch')
            task = await self.get_task(body.task_id)
            if task['state'] not in {'open', 'waiting'}:
                raise HTTPException(409, 'task_closed')
            snapshot = await self.task_definition(body.task_id)
            await self.resolve_definition(snapshot['id'])
            reservation = await self.one('SELECT * FROM reservations WHERE hand_id=?', (body.hand_id,))
            if not reservation or reservation['state'] != 'held' or reservation['expires_at'] <= self.now():
                raise HTTPException(409, 'reservation_lapsed_or_invalid')
            reason = await budgets.reason(self, task, reservation['family'])
            if reason:
                raise HTTPException(409, reason)
            # Consume BEFORE the native call. No other runtime may bind, including
            # one holding a cached plan or a claim whose lease expired meanwhile.
            await self.update('definition_task_queue', 'task_id', body.task_id, dispatched=1, eligible=0)
            await self._queue_state(body.task_id, 'dispatched', claim_token=None, lease_until=None)
            await budgets.charge_reservation(self, hand, task)
            if not task['obligation_id']:
                oid = await self._create_obligation(task)
                await self.update('obligations', 'id', oid, owner_ref=hand['decisions']['queue_return_ref'])
            return {'bound': True}
        return await self.write(key, '/definitions/queue/bind', body, apply, record=False)

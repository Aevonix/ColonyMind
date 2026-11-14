"""Scoped observations from the selected work readers."""
from typing import Literal
import time

from fastapi import APIRouter, HTTPException, Query, Request

from colonymind.api.viewer import authorized_viewer

router = APIRouter(prefix="/v1/host/executions", tags=["executions"])


@router.get("")
async def active(request: Request, contact_id: str, session_id: str = "", limit: int = Query(20, ge=1, le=100),
                 projection: Literal['full', 'request'] = 'full', input_context: bool = False,
                 reserve_chars: int = Query(0, ge=0, le=1200)):
    person, owner = authorized_viewer(request, contact_id)
    if projection == 'request' and not owner:
        raise HTTPException(403, detail='owner_work_context_required')
    if projection == 'request':
        limit = min(limit, 8)
    view = await with_queue_work({'complete': False}, owner=owner, limit=limit)
    if projection == 'request':
        from colonymind.turns.executions import request_work_context
        result = request_work_context(view, limit=limit, session_id=session_id)
        if reserve_chars:
            # One scoped read supplies both choices. The adapter uses the
            # smaller view only when an authorized local task revision exists.
            result['reserved'] = request_work_context(view, limit=max(1, limit-1),
                max_chars=4000-reserve_chars, session_id=session_id)
        return result
    return view


async def with_queue_work(view, *, owner, limit=8):
    # Top-level items belonged to the retired registry; live readers keep their own groups.
    view = {'schema': 'ColonyMindExecutionViewV1', 'items': [], 'total': 0,
            'recent': [], 'recent_truncated': False, 'truncated': False,
            'coverage': 'selected work ledgers', 'commitments_enforced': False,
            'complete': False, 'observed_at': time.time(), **view}
    if not owner:
        return view
    import asyncio
    from colonymind.memory import producers
    from colonymind.turns.hermes_work import cron_view
    from colonymind.turns.board_observations import kanban_view
    from colonymind.turns.reported_workers import reported_worker_view
    from colonymind.turns.executions import process_view, work_source_coverage

    async def read(reader, *, asynchronous=False, **reader_kwargs):
        # These are independent read-only snapshots. One unavailable ledger
        # must not hide work observed by all the other readers. Cancellation of
        # a thread await does not stop its read; native readers bound SQLite work.
        try:
            operation = reader(limit=limit, **reader_kwargs) if asynchronous else producers.read(reader, limit=limit, **reader_kwargs)
            value = await asyncio.wait_for(operation, timeout=.2)
            if value is None:
                return None
            return {**value, 'observed_at': time.time()}
        except Exception as exc:
            import logging
            from colonymind.health_counters import bump
            logging.getLogger(__name__).warning(
                'work reader failed: %s sqlite_errorname=%s',
                type(exc).__name__, getattr(exc, 'sqlite_errorname', None))
            bump('context_reads_failed')
            return {'items': [], 'recent': [], 'available': False, 'unavailable': True,
                    'reason': 'work_source_unavailable', 'observed_at': time.time()}

    # All sources share the existing request budget, rather than each taking
    # another sequential budget after the native ledgers finish.
    view['native_cron'], view['native_kanban'], reported, view['native_processes'] = await asyncio.gather(
        # Leave executor/return slack for the multi-board reader to retain
        # faster boards when its own partial-read budget is exhausted.
        read(cron_view), read(kanban_view, read_budget=.15),
        read(reported_worker_view), read(process_view, asynchronous=True, read_budget=.15))
    if reported is not None:
        view['reported_worker'] = reported
    view['work_sources'] = work_source_coverage(view)
    view['coverage'] = 'selected work ledgers; see work_sources for read coverage'
    return view

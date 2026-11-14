"""Deployment engine: durable pull queue only; all job logic runs in Hermes."""
import asyncio

from colonymind.session_archive.store import ArchiveError


class ArchiveQueueClient:
    def __init__(self, store, *, timeout=15):
        self.store, self.timeout = store, timeout

    def available(self):
        return self.store.worker_available()

    def __call__(self, *, phase, **kwargs):
        # Receipt reads consult only the overlay; no gateway code is imported.
        if phase != 'validate':
            raise ArchiveError('source_unavailable')

    async def archive(self, session_id, body, authority):
        await asyncio.to_thread(self.store.enqueue, session_id, body, authority)
        deadline = asyncio.get_running_loop().time() + self.timeout
        while True:
            result = await asyncio.to_thread(self.store.plan_result, body['operation_id'])
            if result is not None:
                if 'error' in result:
                    raise ArchiveError(result['error'], **{k: v for k, v in result.items() if k != 'error'})
                return result
            if asyncio.get_running_loop().time() >= deadline:
                raise ArchiveError('source_unavailable', pending=True, operation_id=body['operation_id'])
            await asyncio.sleep(.1)

"""Synchronization for tests of scheduled lesson vector fills."""
import asyncio


async def vectors_settled(lessons):
    """Wait until no scheduled fill is running."""
    while lessons._fill_task is not None and not lessons._fill_task.done():
        await asyncio.wait({lessons._fill_task})

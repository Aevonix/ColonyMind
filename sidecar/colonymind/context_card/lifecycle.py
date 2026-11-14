"""Opt-in background worker, opened and closed with the sidecar producers."""
import asyncio
from contextlib import asynccontextmanager

from colonymind.config import load_config
from colonymind.identity import get_owner_name
from .inputs import InputCollector
from .refresh import CardService, RevisionProbe, configured_owner, local_day
from .store import CardStore


@asynccontextmanager
async def context_card_lifespan(app, *, state_dir):
    settings = load_config().get("context_card")
    if not settings["enabled"]:
        CardStore.delete(state_dir / "context-card.db")
        app.state.context_card = None
        yield
        return
    from colonymind.api.routers import host, mind
    store = CardStore(state_dir / "context-card.db")
    collector = InputCollector(state_dir, host=host, mind_provider=mind.get_mind,
        owner_provider=configured_owner, name_provider=get_owner_name, day_provider=local_day)
    service = CardService(store, collector, RevisionProbe(state_dir),
                          reconcile_seconds=settings["reconcile_seconds"])
    app.state.context_card = service
    task = asyncio.create_task(service.run(interval_seconds=settings["poll_seconds"]), name="context-card")
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        app.state.context_card = None
        store.close()

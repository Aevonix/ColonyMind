"""Verified work is independent of the mind and all model services."""
from functools import lru_cache
from pathlib import Path


def work_records_enabled():
    from colonymind import get_state_dir
    from colonymind.config import load_config
    return load_config(get_state_dir()).get('mind.faculties.work_records', True) is not False


@lru_cache(maxsize=8)
def _store(state_dir):
    from colonymind.turns import get_turn_idempotency_ledger
    from .store import WorkStore
    return WorkStore(get_turn_idempotency_ledger(state_dir))


def get_work_store(*, read_only=False):
    if not work_records_enabled():
        return None
    from colonymind import get_state_dir
    if read_only:
        from types import SimpleNamespace
        from .store import WorkStore
        # Readers need only the ledger path, never its initializing constructor.
        ledger = SimpleNamespace(db_path=Path(get_state_dir()) / 'turn-idempotency.db')
        return WorkStore(ledger, read_only=True)
    return _store(str(Path(get_state_dir())))


def schedule_work(root_id=None):
    """Best effort wakeup; the durable dirty queue survives a lost wakeup."""
    import logging
    try:
        store = get_work_store()
        if store is not None:
            store.schedule(root_id)
    except Exception:
        logging.getLogger(__name__).exception('work projection pending')


def repair_work():
    """Explicit full maintenance; never invoke from turn or work API handlers."""
    store = get_work_store()
    return store.reconcile() if store is not None else 0


def refresh_work(ledger, root_ids):
    """Mark only existing roots after external evidence commits, then wake repair."""
    from contextlib import closing
    from colonymind.config import load_config
    if load_config(ledger.db_path.parent).get('mind.faculties.work_records', True) is False:
        return
    changed = False
    with closing(ledger._connect()) as conn, conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='work_dirty'").fetchone():
            return
        for ident in set(root_ids):
            changed |= conn.execute('UPDATE work_dirty SET generation=generation+1,pending=1 WHERE root_turn_id=?',
                                    (ident,)).rowcount > 0
    if changed:
        _store(str(ledger.db_path.parent)).schedule()


def safe_refresh_work(ledger, root_ids):
    """Optional projection failure never rejects committed source data."""
    import logging
    try:
        refresh_work(ledger, root_ids)
    except Exception:
        logging.getLogger(__name__).exception('external work projection refresh pending')

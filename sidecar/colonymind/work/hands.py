"""Refresh only owner roots affected by committed hands transitions."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3


def refresh(path):
    if str(path) == ':memory:':
        return
    path = Path(path)
    with closing(sqlite3.connect(path, timeout=1)) as hands:
        rows = hands.execute('''SELECT q.task_id,q.generation,t.origin,t.origin_ref
            FROM work_refresh_tasks q LEFT JOIN tasks t ON t.id=q.task_id LIMIT 256''').fetchall()
        origins = hands.execute('SELECT origin_ref,generation FROM work_refresh_origins LIMIT 256').fetchall()
        if not rows and not origins:
            return
        ledger_path = path.parent / 'turn-idempotency.db'
        roots = set()
        if ledger_path.is_file():
            from colonymind.turns import get_turn_idempotency_ledger
            ledger = get_turn_idempotency_ledger(path.parent)
            with closing(ledger._connect()) as source:
                if source.execute("SELECT 1 FROM sqlite_master WHERE name='work_envelopes'").fetchone():
                    references = [(row[2], row[3]) for row in rows] + [('owner_turn', row[0]) for row in origins]
                    for origin, reference in references:
                        if origin != 'owner_turn':
                            continue
                        try:
                            reference = json.loads(reference)
                        except (TypeError, ValueError):
                            pass
                        if isinstance(reference, dict):
                            reference = json.dumps(reference, sort_keys=True, separators=(',', ':'))
                        if not isinstance(reference, str):
                            continue
                        roots.update(row[0] for row in source.execute('''SELECT root_turn_id FROM work_envelopes
                            WHERE kind='root' AND invalidated=0 AND json_extract(envelope_json,'$.origin_ref')=?''',
                            (reference,)))
                        roots.update(row[0] for row in source.execute('''SELECT root_turn_id FROM work_envelopes
                            WHERE root_turn_id=? AND kind='root' AND invalidated=0
                            AND json_extract(envelope_json,'$.origin_ref') IS NULL''', (reference,)))
            if roots:
                from . import refresh_work
                refresh_work(ledger, roots)
        with hands:
            hands.executemany('DELETE FROM work_refresh_tasks WHERE task_id=? AND generation=?',
                              [(row[0], row[1]) for row in rows])
            hands.executemany('DELETE FROM work_refresh_origins WHERE origin_ref=? AND generation=?', origins)

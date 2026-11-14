"""Current source checks shared by reply and execution forecasts."""
from contextlib import closing
import hashlib
import json


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _reference(source_id):
    return 'receipt:'+source_id


def _current(ledger, references, versions, owner):
    from colonymind.memory import curated
    return curated.fenced_read(ledger, lambda: _read_current(ledger, references, versions, owner),
                              empty=False)


def _read_current(ledger, references, versions, owner):
    from colonymind.memory import curated
    ids = [r.removeprefix('receipt:') for r in references]
    if any(curated.is_forgotten(ledger, identifier) for identifier in ids):
        return False
    current = {r['source_id']:r['source_version'] for r in ledger.source_references(
        ids, contact_id=owner, session_id='')}
    if not all(current.get(identifier) == versions.get(_reference(identifier))
               and not ledger.is_projection_erased(identifier) for identifier in ids):
        return False
    with closing(ledger._connect()) as db:
        for identifier in ids:
            if db.execute('SELECT 1 FROM source_annotations WHERE target_source_id=? LIMIT 1', (identifier,)).fetchone():
                return False
    return True

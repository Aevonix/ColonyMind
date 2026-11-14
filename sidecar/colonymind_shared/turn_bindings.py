"""Private, same-turn recovery of already verified compositions, without words.

Only a successful ingress-ledger match may call remember. Native composition
metadata supplies candidate words, never admission. A fresh observer must match
the saved keyed fingerprint of the entire envelope and ordered members before
using the saved transport evidence. Each observer keeps its own private store.
"""
from contextlib import closing
import hashlib
import hmac
import json
from pathlib import Path
import secrets
import sqlite3
from threading import RLock

from .midturn import valid_ingress


class TurnBindings:
    def __init__(self, path=None, *, limit=2048):
        self.path = Path(path) if path is not None else None
        self.limit = limit
        self.records = {}
        self.lock = RLock()

    @staticmethod
    def _identity(scope):
        return hashlib.sha256(json.dumps(scope, ensure_ascii=True).encode()).hexdigest()

    @staticmethod
    def _fingerprint(key, text, members, evidence):
        value = json.dumps([text, members, evidence], ensure_ascii=True, sort_keys=True).encode()
        return hmac.new(key, value, hashlib.sha256).hexdigest()

    def remember(self, scope, text, members, evidence, receipts):
        """Retain only a complete admission, before its ingress rows are consumed."""
        if not scope[0] or not scope[1] or not all(valid_ingress(receipt) for receipt in receipts):
            return False
        identity = self._identity(scope)
        key = secrets.token_bytes(32)
        evidence = json.dumps(evidence, ensure_ascii=True, sort_keys=True)
        fingerprint = self._fingerprint(key, text, members, evidence)
        with self.lock:
            previous = self.recover(scope, text, members)
            if previous is not None:
                return previous is not False
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                self.path.touch(mode=0o600, exist_ok=True)
                with closing(sqlite3.connect(self.path, timeout=2)) as db, db:
                    db.execute('CREATE TABLE IF NOT EXISTS bindings '
                               '(identity TEXT PRIMARY KEY, key BLOB, fingerprint TEXT, evidence TEXT)')
                    db.execute('INSERT INTO bindings VALUES (?,?,?,?) ON CONFLICT(identity) DO NOTHING',
                               (identity, key, fingerprint, evidence))
                    db.execute('DELETE FROM bindings WHERE rowid NOT IN '
                               '(SELECT rowid FROM bindings ORDER BY rowid DESC LIMIT ?)', (self.limit,))
                # Another observer of this same store may have won the insert.
                if self.recover(scope, text, members) is False:
                    return False
            self.records[identity] = (key, fingerprint, evidence, tuple(receipts))
            while len(self.records) > self.limit:
                self.records.pop(next(iter(self.records)))
            return True

    def recover(self, scope, text, members):
        """Evidence on an exact match; False on invalid evidence; None if missing."""
        if not scope[0] or not scope[1]:
            return None
        identity = self._identity(scope)
        with self.lock:
            row = self.records.get(identity)
            if row is None and self.path is not None and self.path.exists():
                with closing(sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True, timeout=2)) as db:
                    saved = db.execute('SELECT key, fingerprint, evidence FROM bindings WHERE identity=?',
                                       (identity,)).fetchone()
                row = (*saved, ()) if saved else None
            if row is None:
                return None
            key, fingerprint, evidence, receipts = row
            if (not all(valid_ingress(receipt) for receipt in receipts)
                    or not hmac.compare_digest(fingerprint, self._fingerprint(key, text, members, evidence))):
                return False
            return json.loads(evidence)

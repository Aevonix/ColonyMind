"""Host-private standing snapshots for detached sessions and scheduled work.

Only lifecycle hooks and successful tool results write these records. A platform
name, task text or an absent record never grants authority.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
import sqlite3

STANDINGS = frozenset({"owner", "mind", "system", "guest", "unknown"})
STANDING_FILE = "colonymind-standing.sqlite3"


class StandingStore:
    def __init__(self, path):
        self.path = path

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.touch(mode=0o600, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=2)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS standing (kind TEXT, identity TEXT, "
                       "standing TEXT NOT NULL, platform TEXT NOT NULL, parent TEXT NOT NULL, "
                       "PRIMARY KEY(kind, identity))")
            yield db

    def get(self, kind: str, identity: str) -> tuple[str, str] | None:
        if not identity or not self.path.exists():
            return None
        # A lookup must not create or migrate a store.
        with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=2)) as db:
            row = db.execute("SELECT standing, platform FROM standing WHERE kind=? AND identity=?",
                             (kind, identity)).fetchone()
        return (row[0] if row[0] in STANDINGS else "unknown", row[1]) if row else None

    def complete(self, identity: str, kind: str, target: str, parent: str = "") -> tuple[str, str]:
        """Atomically publish a successful creation and consume its initiating record.

        A crash or failed write cannot lose the owner evidence between deleting
        the outstanding record and saving the resulting job or task.
        """
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT standing, platform FROM standing WHERE kind='creation' AND identity=?",
                             (identity,)).fetchone() if identity else None
            run, platform = (row[0] if row[0] in STANDINGS else "unknown", row[1]) if row else ("unknown", "")
            if target:
                db.execute("INSERT INTO standing VALUES (?,?,?,?,?) ON CONFLICT(kind,identity) DO UPDATE SET "
                           "standing=excluded.standing, platform=excluded.platform, parent=excluded.parent",
                           (kind, target, run, platform, parent))
            db.execute("DELETE FROM standing WHERE kind='creation' AND identity=?", (identity,))
        return run, platform

    def pop(self, kind: str, identity: str) -> tuple[str, str] | None:
        """Consume an outstanding creation record after completion, including after a restart."""
        if not identity or not self.path.exists():
            return None
        with self.connect() as db:
            row = db.execute("DELETE FROM standing WHERE kind=? AND identity=? RETURNING standing, platform",
                             (kind, identity)).fetchone()
        return (row[0], row[1]) if row and row[0] in STANDINGS else None

    def put(self, kind: str, identity: str, standing: str, platform: str = "", parent: str = "",
            *, replace: bool = True) -> None:
        if not identity or standing not in STANDINGS:
            return
        with self.connect() as db:
            conflict = ("DO UPDATE SET standing=excluded.standing, platform=excluded.platform, "
                        "parent=excluded.parent") if replace else "DO NOTHING"
            db.execute("INSERT INTO standing VALUES (?,?,?,?,?) ON CONFLICT(kind,identity) " + conflict,
                       (kind, identity, standing, platform, parent))

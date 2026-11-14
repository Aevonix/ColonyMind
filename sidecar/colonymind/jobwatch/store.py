"""Private watcher state; every other database is opened read-only."""
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3


def connect(path: Path, *, dry_run: bool = False) -> sqlite3.Connection:
    if dry_run:
        conn = sqlite3.connect(':memory:')
        if path.is_file():
            with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as source:
                source.backup(conn)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
        conn = sqlite3.connect(path, timeout=.1)
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS findings (
            id TEXT PRIMARY KEY, "check" TEXT NOT NULL, subject TEXT NOT NULL,
            first_seen REAL NOT NULL, last_seen REAL NOT NULL,
            state TEXT NOT NULL CHECK (state IN ('open','closed')), evidence TEXT NOT NULL,
            notice_data TEXT NOT NULL, opening_notice INTEGER
        );
        CREATE TABLE IF NOT EXISTS notices (
            id INTEGER PRIMARY KEY, finding_id TEXT NOT NULL, "check" TEXT NOT NULL,
            state TEXT NOT NULL CHECK (state IN ('open','closed')), data TEXT NOT NULL,
            sent INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS snapshots (id TEXT PRIMARY KEY, value TEXT NOT NULL);
    ''')
    return conn


def read_findings(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=.1)) as conn:
        conn.row_factory = sqlite3.Row
        return project(conn)


def read_status(path: Path) -> dict:
    if not path.is_file():
        return {'findings': [], 'delivery_failures': 0}
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=.1)) as conn:
        conn.row_factory = sqlite3.Row
        return {'findings': project(conn), 'delivery_failures': get(conn, 'delivery_failures', 0)}


def project(conn: sqlite3.Connection) -> list[dict]:
    result = []
    for row in conn.execute('SELECT * FROM findings WHERE state=\'open\' ORDER BY first_seen, id'):
        item = dict(row)
        del item['notice_data']
        del item['opening_notice']
        item['evidence'] = json.loads(item['evidence'])
        result.append(item)
    return result


def get(conn, key, default=None):
    row = conn.execute('SELECT value FROM snapshots WHERE id=?', (key,)).fetchone()
    return json.loads(row[0]) if row else default


def put(conn, key, value):
    conn.execute('INSERT OR REPLACE INTO snapshots VALUES (?, ?)', (key, json.dumps(value)))

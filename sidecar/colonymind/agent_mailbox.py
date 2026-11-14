"""Provider-neutral retained mail. Content is data; nothing here invokes a model.

SQLite WAL transactions retain identity, content and replay tombstones together.
Each operation owns a connection; SQLite serializes writers across processes.
"""
from __future__ import annotations

import base64
from contextlib import closing, contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from colonymind.util.temporal import now_utc


class MailboxError(ValueError):
    def __init__(self, code: str, status: int = 422):
        super().__init__(code)
        self.code = code
        self.status = status


class LocalIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    local_part: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,30}[a-z0-9]$", max_length=32)
    display_name: str = Field(min_length=1, max_length=128)

    @field_validator("display_name")
    @classmethod
    def name_present(cls, value):
        if not value.strip() or any(ord(c) < 32 for c in value):
            raise ValueError("invalid_display_name")
        return value


class IdentityUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    display_name: str | None = Field(default=None, min_length=1, max_length=128)
    status: Literal["unprovisioned", "active", "disabled"] | None = None

    @field_validator("display_name")
    @classmethod
    def name_present(cls, value):
        return LocalIdentity.name_present(value) if value is not None else value


class InboundMessage(BaseModel):
    """Trusted adapter's envelope. Sender and all text remain untrusted."""
    model_config = ConfigDict(extra="forbid")
    delivery_id: str = Field(min_length=1, max_length=256)
    thread_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$", max_length=128)
    sender: str = Field(min_length=1, max_length=320)
    recipients: list[str] = Field(min_length=1, max_length=100)
    subject: str = Field(default="", max_length=998)
    text: str = Field(default="", max_length=262144)
    occurred_at: datetime
    in_reply_to: str | None = Field(default=None, max_length=256)

    @field_validator("occurred_at")
    @classmethod
    def aware_time(cls, value):
        if value.utcoffset() is None:
            raise ValueError("timezone_required")
        return value.astimezone(timezone.utc)

    @field_validator("recipients")
    @classmethod
    def bounded_addresses(cls, value):
        if any(not address or len(address) > 320 for address in value):
            raise ValueError("invalid_recipients")
        return value


def _json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


class MailboxStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._init_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._initialized = False

    def _connect(self):
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Create privately before SQLite also creates its WAL/shared-memory files.
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
        conn = sqlite3.connect(self.path, check_same_thread=False)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA secure_delete=ON")
        except BaseException:
            conn.close()
            raise
        return conn

    @staticmethod
    def _create_tables(conn):
        """Initialize/migrate once, serialized with other store openers."""
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version > 1:
            raise sqlite3.DatabaseError("unsupported mailbox schema version")
        if version == 1:
            return
        for statement in (
            """CREATE TABLE IF NOT EXISTS mailboxes (
                mailbox_id TEXT PRIMARY KEY,
                address TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('unprovisioned', 'active', 'disabled'))
            )""",
            """CREATE TABLE IF NOT EXISTS messages (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                mailbox_id TEXT NOT NULL REFERENCES mailboxes(mailbox_id),
                message_id TEXT NOT NULL UNIQUE,
                delivery_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                sender TEXT NOT NULL,
                recipients TEXT NOT NULL,
                subject TEXT NOT NULL,
                text TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                in_reply_to TEXT,
                received_at TEXT NOT NULL,
                read INTEGER NOT NULL DEFAULT 0 CHECK(read IN (0, 1)),
                UNIQUE(mailbox_id, delivery_id)
            )""",
            """CREATE TABLE IF NOT EXISTS deliveries (
                mailbox_id TEXT NOT NULL REFERENCES mailboxes(mailbox_id),
                delivery_id TEXT NOT NULL,
                message_id TEXT NOT NULL UNIQUE,
                digest TEXT NOT NULL,
                deleted INTEGER NOT NULL DEFAULT 0 CHECK(deleted IN (0, 1)),
                PRIMARY KEY(mailbox_id, delivery_id)
            )""",
            """CREATE TABLE IF NOT EXISTS deleted_threads (
                mailbox_id TEXT NOT NULL REFERENCES mailboxes(mailbox_id),
                thread_id TEXT NOT NULL,
                PRIMARY KEY(mailbox_id, thread_id)
            )""",
            "CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(mailbox_id, thread_id, sequence)",
            "CREATE INDEX IF NOT EXISTS idx_messages_unread ON messages(mailbox_id, read, thread_id)",
        ):
            conn.execute(statement)
        conn.execute("PRAGMA user_version=1")

    def _initialize(self, conn):
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > 1:
                raise sqlite3.DatabaseError("unsupported mailbox schema version")
            if version < 1:
                conn.execute("PRAGMA journal_mode=WAL")
                with conn:
                    conn.execute("BEGIN IMMEDIATE")
                    self._create_tables(conn)
            self._initialized = True

    @contextmanager
    def _transaction(self, *, write=False, purge=False):
        # Readers use their own WAL snapshots, without reserving a writer slot
        # or waiting behind an in-process writer. Only initialization is shared.
        with closing(self._connect()) as conn:
            self._initialize(conn)
            with self._write_lock if write else nullcontext():
                with conn:
                    conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                    yield conn
                if purge:
                    # secure_delete overwrites deleted pages, but old WAL frames
                    # retain plaintext until a successful truncating checkpoint.
                    if conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]:
                        raise MailboxError("mailbox_busy", 503)

    @staticmethod
    def _mailbox(conn, mailbox_id):
        mailbox = conn.execute("SELECT * FROM mailboxes WHERE mailbox_id=?", (mailbox_id,)).fetchone()
        if mailbox is None:
            raise MailboxError("mailbox_not_found", 404)
        return {key: mailbox[key] for key in sorted(mailbox.keys())}

    @staticmethod
    def _messages(conn, mailbox_id, thread_id, snapshot, position, limit):
        rows = conn.execute(
            "SELECT sequence, mailbox_id, message_id, thread_id, sender, recipients, subject, text, "
            "occurred_at, in_reply_to, received_at, read FROM messages "
            "WHERE mailbox_id=? AND thread_id=? AND sequence<=? AND sequence>? "
            "ORDER BY sequence LIMIT ?",
            (mailbox_id, thread_id, snapshot, position or 0, limit)).fetchall()
        messages = []
        for row in rows:
            # Preserve the existing JSON projection's field order and value types.
            message = {key: row[key] for key in sorted(row.keys())}
            message["recipients"] = json.loads(message["recipients"])
            message["read"] = bool(message["read"])
            messages.append(message)
        return messages

    @staticmethod
    def _counts(conn, mailbox_id):
        return dict(conn.execute(
            "SELECT COUNT(*) AS unread_count, COUNT(DISTINCT thread_id) AS unread_thread_count "
            "FROM messages WHERE mailbox_id=? AND read=0", (mailbox_id,)).fetchone())

    def identities(self):
        with self._transaction() as conn:
            return {"mailboxes": [{key: row[key] for key in sorted(row.keys())}
                                  for row in conn.execute("SELECT * FROM mailboxes ORDER BY mailbox_id")]}

    def create(self, body: LocalIdentity):
        # The mail domain is the deployment's own value: no default, and no identity without it.
        domain = os.environ.get("COLONYMIND_AGENT_MAIL_DOMAIN", "").strip().lower()
        if not domain:
            raise MailboxError("mail_domain_not_configured", 501)
        address = body.local_part + "@" + domain
        with self._transaction(write=True) as conn:
            if conn.execute("SELECT 1 FROM mailboxes WHERE address=?", (address,)).fetchone():
                raise MailboxError("address_already_registered", 409)
            mailbox = {"mailbox_id": str(uuid4()), "address": address,
                       "display_name": body.display_name, "status": "unprovisioned"}
            conn.execute("INSERT INTO mailboxes (mailbox_id, address, display_name, status) VALUES (?, ?, ?, ?)",
                         tuple(mailbox.values()))
            return mailbox

    def identity(self, mailbox_id):
        with self._transaction() as conn:
            return {**self._mailbox(conn, mailbox_id), **self._counts(conn, mailbox_id)}

    def update(self, mailbox_id, body: IdentityUpdate):
        with self._transaction(write=True) as conn:
            mailbox = self._mailbox(conn, mailbox_id)
            mailbox.update(body.model_dump(exclude_none=True))
            conn.execute("UPDATE mailboxes SET display_name=?, status=? WHERE mailbox_id=?",
                         (mailbox["display_name"], mailbox["status"], mailbox_id))
            return mailbox

    def receive(self, mailbox_id: str, message: InboundMessage):
        """Persist an adapter-authenticated delivery; never execute or trust its text."""
        payload = message.model_dump(mode="json")
        digest = hashlib.sha256(_json(payload).encode()).hexdigest()
        with self._transaction(write=True) as conn:
            mailbox = self._mailbox(conn, mailbox_id)
            previous = conn.execute("SELECT * FROM deliveries WHERE mailbox_id=? AND delivery_id=?",
                                    (mailbox_id, message.delivery_id)).fetchone()
            if previous:
                if previous["digest"] != digest:
                    raise MailboxError("delivery_conflict", 409)
                return {"message_id": previous["message_id"], "thread_id": message.thread_id,
                        "duplicate": True, "deleted": bool(previous["deleted"])}
            if mailbox["status"] != "active":
                raise MailboxError("mailbox_inactive", 409)
            if mailbox["address"] not in {a.lower() for a in message.recipients}:
                raise MailboxError("recipient_mismatch")
            conn.execute("DELETE FROM deleted_threads WHERE mailbox_id=? AND thread_id=?",
                         (mailbox_id, message.thread_id))
            ident = str(uuid4())
            conn.execute(
                "INSERT INTO messages (mailbox_id, message_id, delivery_id, thread_id, sender, recipients, "
                "subject, text, occurred_at, in_reply_to, received_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (mailbox_id, ident, message.delivery_id, message.thread_id, message.sender,
                 _json(message.recipients), message.subject, message.text, payload["occurred_at"],
                 message.in_reply_to, now_utc().isoformat()))
            conn.execute("INSERT INTO deliveries (mailbox_id, delivery_id, message_id, digest) VALUES (?, ?, ?, ?)",
                         (mailbox_id, message.delivery_id, ident, digest))
            return {"message_id": ident, "thread_id": message.thread_id,
                    "duplicate": False, "deleted": False}

    @staticmethod
    def _page(conn, mailbox_id, thread_id, limit, cursor):
        if not 1 <= limit <= 100:
            raise MailboxError("invalid_limit")
        row = conn.execute("SELECT seq FROM sqlite_sequence WHERE name='messages'").fetchone()
        sequence = row[0] if row else 0
        snapshot, position = sequence, None
        if cursor is not None:
            try:
                if not cursor or len(cursor) > 1024:
                    raise ValueError
                raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
                resource, snapshot, position = json.loads(raw)
                if (resource != [mailbox_id, thread_id] or type(snapshot) is not int
                        or type(position) is not int or not 0 < position <= snapshot <= sequence):
                    raise ValueError
            except (ValueError, TypeError, UnicodeError):
                raise MailboxError("invalid_cursor") from None
        return snapshot, position

    @staticmethod
    def _continuation(mailbox_id, thread_id, snapshot, position, has_more):
        cursor = None
        if has_more:
            cursor = base64.urlsafe_b64encode(_json([[mailbox_id, thread_id], snapshot, position]).encode()).decode().rstrip("=")
        return {"has_more": has_more, "next_cursor": cursor}

    @staticmethod
    def _thread(row):
        unread = row["unread_count"]
        return {"thread_id": row["thread_id"], "subject": row["subject"], "sender": row["sender"],
                "summary": (row["summary"] or b"").decode("utf-8", errors="ignore")[:240],
                "message_count": row["message_count"],
                "received_at": row["received_at"], "unread_count": unread,
                "unread": bool(unread), "content_trust": "untrusted"}

    @staticmethod
    def _thread_rows(conn, mailbox_id, snapshot, *, thread_id=None, position=None, limit=1):
        parameters = [mailbox_id, snapshot]
        predicate = "mailbox_id=? AND sequence<=?"
        if thread_id is not None:
            predicate += " AND thread_id=?"
            parameters.append(thread_id)
        continuation = ""
        if position is not None:
            continuation = "HAVING MAX(sequence)<?"
            parameters.append(position)
        parameters.append(limit)
        # Aggregate metadata before joining the bounded page to its latest mail.
        # A UTF-8 excerpt needs at most 4 bytes per character; BLOB substr also
        # preserves embedded NULs that SQLite's TEXT substr would cut short.
        return conn.execute(
            f"""WITH page AS (
                SELECT thread_id, MAX(sequence) AS sequence, COUNT(*) AS message_count
                FROM messages WHERE {predicate} GROUP BY thread_id {continuation}
                ORDER BY sequence DESC LIMIT ?
            )
            SELECT page.sequence, latest.thread_id, latest.subject, latest.sender,
                   substr(CAST(latest.text AS BLOB), 1, 960) AS summary,
                   page.message_count, latest.received_at,
                   (SELECT COUNT(*) FROM messages AS unread
                    WHERE unread.mailbox_id=latest.mailbox_id AND unread.thread_id=latest.thread_id
                          AND unread.read=0) AS unread_count
            FROM page JOIN messages AS latest ON latest.sequence=page.sequence
            ORDER BY page.sequence DESC""", parameters).fetchall()

    def threads(self, mailbox_id, *, limit=50, cursor=None):
        with self._transaction() as conn:
            self._mailbox(conn, mailbox_id)
            snapshot, position = self._page(conn, mailbox_id, None, limit, cursor)
            rows = self._thread_rows(conn, mailbox_id, snapshot, position=position, limit=limit + 1)
            page = rows[:limit]
            return {"threads": [self._thread(row) for row in page], **self._counts(conn, mailbox_id),
                    **self._continuation(mailbox_id, None, snapshot,
                                         page[-1]["sequence"] if page else 0, len(rows) > limit)}

    def thread(self, mailbox_id, thread_id, *, limit=50, cursor=None):
        with self._transaction() as conn:
            self._mailbox(conn, mailbox_id)
            snapshot, position = self._page(conn, mailbox_id, thread_id, limit, cursor)
            rows = self._thread_rows(conn, mailbox_id, snapshot, thread_id=thread_id)
            if not rows:
                raise MailboxError("thread_not_found", 404)
            page = self._messages(conn, mailbox_id, thread_id, snapshot, position, limit)
            last = page[-1]["sequence"] if page else position or 0
            has_more = bool(conn.execute(
                "SELECT EXISTS(SELECT 1 FROM messages WHERE mailbox_id=? AND thread_id=? "
                "AND sequence>? AND sequence<=?)", (mailbox_id, thread_id, last, snapshot)).fetchone()[0])
            return {"thread": self._thread(rows[0]), "messages": [
                {**{key: value for key, value in row.items() if key not in ("sequence", "delivery_id")},
                 "content_trust": "untrusted"} for row in page],
                **self._continuation(mailbox_id, thread_id, snapshot, last, has_more)}

    def read(self, mailbox_id, thread_id, through_message_id):
        with self._transaction(write=True) as conn:
            self._mailbox(conn, mailbox_id)
            if not conn.execute("SELECT 1 FROM messages WHERE mailbox_id=? AND thread_id=?",
                                (mailbox_id, thread_id)).fetchone():
                raise MailboxError("thread_not_found", 404)
            marker = conn.execute(
                "SELECT sequence FROM messages WHERE mailbox_id=? AND thread_id=? AND message_id=?",
                (mailbox_id, thread_id, through_message_id)).fetchone()
            if marker is None:
                raise MailboxError("invalid_read_marker")
            changed = conn.execute(
                "UPDATE messages SET read=1 WHERE mailbox_id=? AND thread_id=? AND sequence<=? AND read=0",
                (mailbox_id, thread_id, marker["sequence"])).rowcount
            return {"thread_id": thread_id, "read_count": changed, **self._counts(conn, mailbox_id)}

    def delete(self, mailbox_id, thread_id):
        with self._transaction(write=True, purge=True) as conn:
            self._mailbox(conn, mailbox_id)
            parameters = (mailbox_id, thread_id)
            present = conn.execute("SELECT 1 FROM messages WHERE mailbox_id=? AND thread_id=?", parameters).fetchone()
            deleted = conn.execute("SELECT 1 FROM deleted_threads WHERE mailbox_id=? AND thread_id=?", parameters).fetchone()
            if not present and not deleted:
                raise MailboxError("thread_not_found", 404)
            conn.execute("UPDATE deliveries SET deleted=1 WHERE mailbox_id=? AND message_id IN "
                         "(SELECT message_id FROM messages WHERE mailbox_id=? AND thread_id=?)",
                         (mailbox_id, mailbox_id, thread_id))
            conn.execute("DELETE FROM messages WHERE mailbox_id=? AND thread_id=?", parameters)
            conn.execute("INSERT OR IGNORE INTO deleted_threads (mailbox_id, thread_id) VALUES (?, ?)", parameters)
            return {"thread_id": thread_id, "deleted": True, **self._counts(conn, mailbox_id)}
